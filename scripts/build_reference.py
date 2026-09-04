#!/usr/bin/env python3
"""Rebuild ViRDM's frozen joint video-text reference from clean LMDB rows."""

from __future__ import annotations

import argparse
import bisect
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_prompt_lmdb import discover_sources, inspect_existing, write_lmdb
from utils.dataset import CleanLatentLMDBDataset, PromptLMDBDataset
from utils.wan_wrapper import WanVAEWrapper
from virdm_integration.joint_text import (
    VIRDM_JOINT_BANDWIDTH_SCALE,
    VIRDM_JOINT_TEXT_DIM,
    VIRDM_JOINT_TEXT_MODEL,
    VIRDM_JOINT_TEXT_PRETRAINED,
    couple_video_text,
    prompt_rows_sha256,
    virdm_joint_contract,
)
from virdm_integration.reference import file_sha256
from virdm_integration.reference_builder import (
    build_nystrom_reference,
    median_bandwidth,
)


REFERENCE_REVISION = "virdm_joint_reference_v1"
SOURCE_DATASET_REPO = "zhuhz22/Causal-Forcing-data"
EXPECTED_VISUAL_DIM = 1024
EXPECTED_ROWS = 6505


def atomic_torch_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_numpy_save(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class CleanLatentSources:
    """Canonical concatenation of the official merged LMDB or its 15 shards."""

    def __init__(self, sources: list[Path], *, readahead: bool = False):
        self.sources = list(sources)
        self.datasets = [
            CleanLatentLMDBDataset(str(source), readahead=readahead)
            for source in self.sources
        ]
        self.offsets = [0]
        for dataset in self.datasets:
            self.offsets.append(self.offsets[-1] + len(dataset))

    def __len__(self) -> int:
        return self.offsets[-1]

    def __getitem__(self, index: int):
        if not 0 <= int(index) < len(self):
            raise IndexError(index)
        source_index = bisect.bisect_right(self.offsets, int(index)) - 1
        local_index = int(index) - self.offsets[source_index]
        row = self.datasets[source_index][local_index]
        row["idx"] = int(index)
        return row

    def close(self) -> None:
        for dataset in self.datasets:
            dataset.env.close()


def load_prompts(sources: list[Path], limit: int) -> list[str]:
    prompts: list[str] = []
    for source in sources:
        dataset = PromptLMDBDataset(str(source), readahead=False)
        remaining = int(limit) - len(prompts)
        if remaining <= 0:
            break
        prompts.extend(
            str(dataset[index]["prompts"])
            for index in range(min(len(dataset), remaining))
        )
        dataset.env.close()
    if len(prompts) != int(limit):
        raise RuntimeError(f"read {len(prompts)} prompts, expected {limit}")
    return prompts


def distributed_context() -> tuple[int, int, int, torch.device]:
    if not {"RANK", "WORLD_SIZE", "LOCAL_RANK"}.issubset(os.environ):
        if not torch.cuda.is_available():
            raise RuntimeError("reference extraction requires a CUDA GPU")
        torch.cuda.set_device(0)
        return 0, 1, 0, torch.device("cuda", 0)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return dist.get_rank(), dist.get_world_size(), local_rank, torch.device(
        "cuda", local_rank
    )


@torch.no_grad()
def extract_visual_shard(
    args,
    *,
    sources: list[Path],
    rank: int,
    world_size: int,
    device: torch.device,
) -> Path:
    shard_path = (
        args.output_dir
        / "shards"
        / f"visual_rank{rank:02d}_of{world_size:02d}.pt"
    )
    if args.skip_existing and shard_path.is_file():
        print(f"[rank {rank}] reuse {shard_path}", flush=True)
        return shard_path

    dataset = CleanLatentSources(sources, readahead=False)
    row_count = min(len(dataset), args.max_videos or len(dataset))
    indices = list(range(rank, row_count, world_size))
    vae = WanVAEWrapper().to(device=device, dtype=torch.bfloat16).eval()
    vae.requires_grad_(False)
    from virdm_integration.vjepa21_video import NativeVJEPA21VideoEncoder

    encoder = NativeVJEPA21VideoEncoder(
        "vjepa2_1_vit_large_384",
        str(args.vjepa_checkpoint),
        checkpoint_key="ema_encoder",
        input_height=480,
        input_width=832,
        input_frames=81,
        padded_frames=82,
        temporal_pad="replicate_last",
        feature_dim=EXPECTED_VISUAL_DIM,
        pool="global_mean_all_tokens",
        activation_checkpointing=False,
        device=device,
    )
    contract = encoder.contract()
    features = torch.empty(
        len(indices), EXPECTED_VISUAL_DIM, dtype=torch.float32, device="cpu"
    )
    started = time.perf_counter()
    for output_index, video_index in enumerate(indices):
        latent = dataset[video_index]["clean_latent"].unsqueeze(0).to(
            device=device, dtype=torch.bfloat16
        )
        decoded = vae.decode_to_pixel(latent, use_cache=False)
        if tuple(decoded.shape) != (1, 81, 3, 480, 832):
            raise RuntimeError(
                f"decoded row {video_index} has shape {tuple(decoded.shape)}"
            )
        pixels = (decoded.float() * 0.5 + 0.5).clamp(0.0, 1.0)
        feature = encoder(pixels)
        if tuple(feature.shape) != (1, EXPECTED_VISUAL_DIM):
            raise RuntimeError(
                f"V-JEPA row {video_index} has shape {tuple(feature.shape)}"
            )
        features[output_index].copy_(feature[0].float().cpu())
        del latent, decoded, pixels, feature
        completed = output_index + 1
        if completed % args.log_every == 0 or completed == len(indices):
            elapsed = max(time.perf_counter() - started, 1.0e-6)
            print(
                f"[rank {rank}] visual {completed}/{len(indices)} "
                f"({completed / elapsed:.3f} videos/s)",
                flush=True,
            )
    atomic_torch_save(
        {
            "video_indices": torch.tensor(indices, dtype=torch.long),
            "features": features,
            "encoder_contract": contract,
            "rank": rank,
            "world_size": world_size,
        },
        shard_path,
    )
    dataset.close()
    del encoder, vae, dataset, features
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[rank {rank}] wrote {shard_path}", flush=True)
    return shard_path


def consolidate_visual_shards(
    args, *, world_size: int, row_count: int
) -> tuple[torch.Tensor, dict]:
    ordered = torch.empty(row_count, EXPECTED_VISUAL_DIM, dtype=torch.float32)
    seen = torch.zeros(row_count, dtype=torch.bool)
    contract = None
    for rank in range(world_size):
        shard_path = (
            args.output_dir
            / "shards"
            / f"visual_rank{rank:02d}_of{world_size:02d}.pt"
        )
        if not shard_path.is_file():
            raise FileNotFoundError(f"missing visual shard: {shard_path}")
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        if int(shard.get("world_size", -1)) != world_size:
            raise ValueError(f"world-size mismatch in {shard_path}")
        indices = shard["video_indices"].long()
        features = shard["features"].float()
        if tuple(features.shape) != (indices.numel(), EXPECTED_VISUAL_DIM):
            raise ValueError(f"invalid visual shard shape in {shard_path}")
        shard_contract = dict(shard["encoder_contract"])
        if contract is None:
            contract = shard_contract
        elif contract != shard_contract:
            raise ValueError("V-JEPA contracts differ across extraction ranks")
        if bool(seen[indices].any()):
            raise ValueError(f"duplicate visual indices in {shard_path}")
        ordered[indices] = features
        seen[indices] = True
    if not bool(seen.all()):
        missing = (~seen).nonzero().flatten()[:20].tolist()
        raise RuntimeError(f"missing visual rows: {missing}")
    return ordered.contiguous(), dict(contract or {})


def text_tokenization_stats(tokenizer, prompts: list[str]) -> dict:
    cleaned = [tokenizer.clean_fn(prompt) for prompt in prompts]
    untruncated = tokenizer.tokenizer(
        cleaned,
        add_special_tokens=True,
        padding=False,
        truncation=False,
    )["input_ids"]
    lengths = sorted(len(row) for row in untruncated)
    context_length = int(tokenizer.context_length)
    truncated = sum(length > context_length for length in lengths)
    return {
        "context_length": context_length,
        "truncation": True,
        "truncated_prompt_count": truncated,
        "truncated_prompt_fraction": truncated / len(lengths),
        "token_length_min": lengths[0],
        "token_length_p50": lengths[len(lengths) // 2],
        "token_length_p95": lengths[int(0.95 * len(lengths))],
        "token_length_p99": lengths[int(0.99 * len(lengths))],
        "token_length_max": lengths[-1],
    }


@torch.no_grad()
def encode_text_table(
    args, prompts: list[str], device: torch.device
) -> tuple[np.ndarray, dict]:
    import open_clip

    from open_clip.tokenizer import HFTokenizer

    tokenizer = HFTokenizer(
        str(args.siglip_checkpoint.parent),
        context_length=64,
        clean="canonicalize",
    )
    tokenization = text_tokenization_stats(tokenizer, prompts)
    model, _, _ = open_clip.create_model_and_transforms(
        VIRDM_JOINT_TEXT_MODEL,
        pretrained=str(args.siglip_checkpoint),
    )
    model = model.to(device).eval().requires_grad_(False)
    encoded = []
    for start in range(0, len(prompts), args.text_batch):
        batch = prompts[start : start + args.text_batch]
        tokens = tokenizer(batch).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rows = F.normalize(model.encode_text(tokens).float(), dim=-1)
        encoded.append(rows.cpu())
        print(
            f"text {min(start + len(batch), len(prompts))}/{len(prompts)}",
            flush=True,
        )
    table = torch.cat(encoded, dim=0)
    if tuple(table.shape) != (len(prompts), VIRDM_JOINT_TEXT_DIM):
        raise RuntimeError(f"SigLIP2 produced {tuple(table.shape)}")
    norm_error = float((torch.linalg.vector_norm(table, dim=1) - 1.0).abs().max())
    if not bool(torch.isfinite(table).all()) or norm_error > 5.0e-4:
        raise RuntimeError(f"invalid SigLIP2 table; max norm error={norm_error}")
    array = table.numpy().astype(np.float32, copy=False)
    del model, table, encoded
    gc.collect()
    torch.cuda.empty_cache()
    return array, {"tokenization": tokenization, "max_l2_norm_error": norm_error}


def source_metadata(sources: list[Path]) -> dict:
    files = [source / "data.mdb" for source in sources]
    if len(files) == 1:
        return {
            "source_lmdb_bytes": files[0].stat().st_size,
            "source_lmdb_sha256": file_sha256(files[0]),
        }
    return {
        "source_lmdb_shards": [
            {
                "name": source.name,
                "bytes": data.stat().st_size,
                "sha256": file_sha256(data),
            }
            for source, data in zip(sources, files)
        ]
    }


def build_joint_reference(
    args,
    *,
    sources: list[Path],
    world_size: int,
    device: torch.device,
) -> None:
    dataset_rows = 0
    for path in sources:
        prompt_dataset = PromptLMDBDataset(str(path), readahead=False)
        dataset_rows += len(prompt_dataset)
        prompt_dataset.env.close()
    row_count = min(dataset_rows, args.max_videos or dataset_rows)
    prompts = load_prompts(sources, row_count)
    prompt_sha = prompt_rows_sha256(prompts)
    if args.max_videos == 0:
        if row_count != EXPECTED_ROWS:
            raise RuntimeError(f"canonical reference requires {EXPECTED_ROWS} rows")
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        expected_sha = manifest["virdm_training_assets"]["prompt_rows_sha256"]
        if prompt_sha != expected_sha:
            raise RuntimeError(
                f"ordered prompt SHA {prompt_sha} != canonical {expected_sha}"
            )
    prompt_data_path = args.output_dir / "prompt_data"
    prompt_contract = inspect_existing(prompt_data_path)
    if prompt_contract is None:
        if prompt_data_path.exists():
            raise RuntimeError(
                f"non-matching prompt LMDB already exists: {prompt_data_path}"
            )
        write_lmdb(prompts, prompt_data_path)
    elif prompt_contract != (row_count, prompt_sha):
        raise RuntimeError(
            f"prompt LMDB contract {prompt_contract} != {(row_count, prompt_sha)}"
        )

    visual, visual_contract = consolidate_visual_shards(
        args, world_size=world_size, row_count=row_count
    )
    visual_metadata = {
        "format": "virdm_visual_feature_pool_v1",
        **source_metadata(sources),
        "source_dataset_repo_id": SOURCE_DATASET_REPO,
        "num_videos": row_count,
        "num_rows": row_count,
        "rows_per_video": 1,
        "encoder_type": "vjepa21_video",
        "encoder_id": "vjepa2_1_vit_large_384",
        "encoder_checkpoint_filename": args.vjepa_checkpoint.name,
        "encoder_checkpoint_sha256": file_sha256(args.vjepa_checkpoint),
        "seed": args.seed,
        **visual_contract,
    }
    visual_path = args.output_dir / "visual_features_fp32.pt"
    atomic_torch_save(
        {
            "features": visual,
            "n": row_count,
            "d_in": EXPECTED_VISUAL_DIM,
            "metadata": visual_metadata,
        },
        visual_path,
    )
    torch.manual_seed(args.seed)
    visual_sigma = median_bandwidth(
        visual, max_subsample=10000, scale=VIRDM_JOINT_BANDWIDTH_SCALE
    )

    text, text_details = encode_text_table(args, prompts, device)
    text_path = args.output_dir / "siglip2_text_fp32.npy"
    atomic_numpy_save(text, text_path)
    text_sha = file_sha256(text_path)
    text_receipt = {
        "format": "virdm_siglip2_text_table_v1",
        "num_prompts": row_count,
        "prompt_rows_sha256": prompt_sha,
        "text_encoder_id": VIRDM_JOINT_TEXT_MODEL,
        "text_encoder_pretrained": VIRDM_JOINT_TEXT_PRETRAINED,
        "text_encoder_checkpoint_filename": args.siglip_checkpoint.name,
        "text_encoder_checkpoint_sha256": file_sha256(args.siglip_checkpoint),
        "text_tokenizer_files": {
            name: file_sha256(args.siglip_checkpoint.parent / name)
            for name in (
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
            )
        },
        "text_feature_dim": VIRDM_JOINT_TEXT_DIM,
        "text_feature_normalize": True,
        "text_table_sha256": text_sha,
        **text_details,
    }
    atomic_json_save(text_receipt, args.output_dir / "text_table_receipt.json")

    text_tensor = torch.from_numpy(np.ascontiguousarray(text)).float()
    torch.manual_seed(args.seed)
    text_sigma = median_bandwidth(text_tensor, max_subsample=2000, scale=1.0)
    beta = float(visual_sigma) / float(text_sigma)
    joint = couple_video_text(visual, text_tensor, beta).contiguous()
    joint_contract = virdm_joint_contract(
        visual_contract,
        text_table_sha256=text_sha,
        prompt_rows_sha256_value=prompt_sha,
        bandwidth_scale=VIRDM_JOINT_BANDWIDTH_SCALE,
    )
    metadata = {
        **visual_metadata,
        **joint_contract,
        "format": REFERENCE_REVISION,
        "virdm_reference_revision": REFERENCE_REVISION,
        "num_rows": row_count,
        "num_videos": row_count,
        "sigma_scale": VIRDM_JOINT_BANDWIDTH_SCALE,
        "joint_visual_median_sigma": float(visual_sigma),
        "joint_sigma_img": float(visual_sigma),
        "joint_s_txt": float(text_sigma),
        "joint_beta": float(beta),
        "nystrom_landmarks": args.landmarks,
        "fit_n": min(args.fit_n, row_count),
        "kmeans_iters": args.kmeans_iters,
        "krr_n": min(args.krr_n, row_count),
        "jitter": 1.0e-6,
        "median_max_subsample": 2000,
    }
    joint_path = args.output_dir / "joint_features_fp32.pt"
    atomic_torch_save(
        {
            "features": joint,
            "n": row_count,
            "d_in": joint.shape[1],
            "metadata": metadata,
        },
        joint_path,
    )
    metadata["feature_pool_sha256"] = file_sha256(joint_path)
    print(
        "joint reference: "
        f"video_sigma={visual_sigma:.9g} text_sigma={text_sigma:.9g} "
        f"beta={beta:.9g} rows={row_count} dim={joint.shape[1]}",
        flush=True,
    )
    bundle = build_nystrom_reference(
        joint,
        visual_sigma,
        n_landmarks=args.landmarks,
        fit_n=min(args.fit_n, row_count),
        kmeans_iterations=args.kmeans_iters,
        krr_n=min(args.krr_n, row_count),
        seed=args.seed,
        jitter=1.0e-6,
        device=device,
    )
    bundle.update(
        {
            "beta": float(beta),
            "s_txt": float(text_sigma),
            "sigma_scale": VIRDM_JOINT_BANDWIDTH_SCALE,
            "d_img": EXPECTED_VISUAL_DIM,
            "d_txt": VIRDM_JOINT_TEXT_DIM,
            "metadata": metadata,
        }
    )
    reference_path = args.output_dir / f"reference_M{args.landmarks}.pt"
    atomic_torch_save(bundle, reference_path)
    reference_sha = file_sha256(reference_path)
    receipt = {
        **metadata,
        "prompt_rows_sha256": prompt_sha,
        "visual_feature_pool_sha256": file_sha256(visual_path),
        "text_table_sha256": text_sha,
        "joint_feature_pool_sha256": file_sha256(joint_path),
        "reference_bundle_sha256": reference_sha,
        "sigma": float(bundle["sigma"]),
        "k_rr": float(bundle["k_rr"]),
        "beta": float(beta),
        "s_txt": float(text_sigma),
        "Z_shape": list(bundle["Z"].shape),
        "alpha_shape": list(bundle["alpha"].shape),
    }
    atomic_json_save(receipt, args.output_dir / "receipt.json")

    config = OmegaConf.load(args.base_config)
    config.virdm_reference_bundle_path = str(reference_path)
    config.virdm_reference_bundle_sha256 = reference_sha
    config.virdm_joint_text_table_path = str(text_path)
    config.virdm_joint_text_table_sha256 = text_sha
    config.virdm_joint_prompt_rows_sha256 = prompt_sha
    config.virdm_expected_reference_rows = row_count
    config.virdm_nystrom_landmarks = args.landmarks
    config.data_path = str(prompt_data_path)
    OmegaConf.save(config, args.output_dir / "config_virdm_rebuilt_reference.yaml")
    print(f"wrote {reference_path} sha256={reference_sha}", flush=True)


def parse_args():
    artifact_root = Path(
        os.environ.get("VIRDM_ARTIFACT_ROOT", REPO_ROOT / "artifacts")
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Merged clean_data LMDB or root of the 15 official shards.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=artifact_root / "rebuilt_reference"
    )
    parser.add_argument(
        "--wan-model-root",
        type=Path,
        default=artifact_root / "wan" / "Wan2.1-T2V-1.3B",
    )
    parser.add_argument(
        "--vjepa-checkpoint",
        type=Path,
        default=artifact_root / "vjepa" / "vjepa2_1_vitl_dist_vitG_384.pt",
    )
    parser.add_argument(
        "--siglip-checkpoint",
        type=Path,
        default=artifact_root / "siglip2" / "open_clip_model.safetensors",
    )
    parser.add_argument(
        "--base-config", type=Path, default=REPO_ROOT / "config_virdm_bs64_1x8.yaml"
    )
    parser.add_argument(
        "--manifest", type=Path, default=REPO_ROOT / "artifacts" / "manifest.json"
    )
    parser.add_argument("--phase", choices=("all", "extract", "assemble"), default="all")
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--landmarks", type=int, default=4096)
    parser.add_argument("--fit-n", type=int, default=6505)
    parser.add_argument("--kmeans-iters", type=int, default=20)
    parser.add_argument("--krr-n", type=int, default=6505)
    parser.add_argument("--text-batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    if args.max_videos < 0:
        parser.error("--max-videos cannot be negative")
    if args.landmarks <= 0:
        parser.error("--landmarks must be positive")
    for name in ("source", "output_dir", "wan_model_root", "vjepa_checkpoint", "base_config", "manifest"):
        setattr(args, name, Path(getattr(args, name)).expanduser().resolve())
    # Keep the snapshot directory when the checkpoint itself is a Hub symlink;
    # its sibling tokenizer files are part of the pinned text contract.
    args.siglip_checkpoint = args.siglip_checkpoint.expanduser().absolute()
    if not args.vjepa_checkpoint.is_file():
        parser.error(f"missing V-JEPA checkpoint: {args.vjepa_checkpoint}")
    if args.phase != "extract":
        siglip_files = [
            args.siglip_checkpoint,
            args.siglip_checkpoint.parent / "tokenizer.json",
            args.siglip_checkpoint.parent / "tokenizer_config.json",
            args.siglip_checkpoint.parent / "special_tokens_map.json",
        ]
        missing = [str(path) for path in siglip_files if not path.is_file()]
        if missing:
            parser.error(f"missing SigLIP2 files: {missing}")
    if args.phase != "assemble" and not (args.wan_model_root / "Wan2.1_VAE.pth").is_file():
        parser.error(f"missing Wan VAE under: {args.wan_model_root}")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = discover_sources(args.source)
    os.environ["VIRDM_WAN_MODEL_ROOT"] = str(args.wan_model_root)
    rank, world_size, local_rank, device = distributed_context()
    if args.phase != "assemble":
        extract_visual_shard(
            args,
            sources=sources,
            rank=rank,
            world_size=world_size,
            device=device,
        )
    if dist.is_initialized():
        dist.barrier(device_ids=[local_rank])
        dist.destroy_process_group()
    if args.phase == "extract" or rank != 0:
        return
    build_joint_reference(
        args,
        sources=sources,
        world_size=world_size,
        device=device,
    )


if __name__ == "__main__":
    main()
