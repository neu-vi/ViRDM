import argparse
import gc
import os
import re
import sys

import torch
from einops import rearrange
from omegaconf import OmegaConf
from torchvision.io import write_video

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pipeline import CausalInferencePipeline
from utils.dataset import TextDataset
from utils.misc import set_seed
from utils.prompt_embedding_cache import PromptEmbeddingLMDBCache
from wan.modules.attention import require_flash_attention_2


def sanitize_filename(text: str, max_length: int = 96) -> str:
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", "_", text.strip())
    text = text[:max_length].strip("_")
    return text or "sample"


def raw_filename(text: str) -> str:
    text = text.strip()
    if not text:
        raise ValueError("Prompt is empty; cannot build raw filename")
    if "/" in text or "\x00" in text:
        raise ValueError(f"Prompt contains unsupported filename characters: {text!r}")
    return text


def load_generator_state(checkpoint_path: str, use_ema: bool):
    try:
        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint must be a dictionary")
    if use_ema:
        if "generator_ema" not in state_dict:
            raise KeyError("--use_ema requested but checkpoint has no generator_ema")
        key = "generator_ema"
    else:
        if "generator" not in state_dict:
            raise KeyError("checkpoint has no generator state")
        key = "generator"
    generator_state = state_dict[key]
    fixed = {}
    for name, value in generator_state.items():
        if name.startswith("model._fsdp_wrapped_module."):
            name = name.replace("model._fsdp_wrapped_module.", "model.", 1)
        fixed[name] = value
    return fixed


class CachedPromptTextEncoder(torch.nn.Module):
    def __init__(self, cache_path: str):
        super().__init__()
        self.cache = PromptEmbeddingLMDBCache(cache_path)
        self._device = torch.device("cpu")
        self._dtype = torch.bfloat16

    def to(self, *args, **kwargs):
        device = kwargs.get("device")
        dtype = kwargs.get("dtype")
        if args:
            first = args[0]
            if isinstance(first, (str, torch.device)):
                device = first
            elif isinstance(first, torch.dtype):
                dtype = first
        if device is not None:
            self._device = torch.device(device)
        if dtype is not None:
            self._dtype = dtype
        return self

    def forward(self, text_prompts):
        return {
            "prompt_embeds": self.cache.get_batch(
                text_prompts,
                device=self._device,
                dtype=self._dtype,
            )
        }


def write_video_with_fallback(output_path: str, frames: torch.Tensor, fps: int):
    frames = frames.clamp(0, 255).to(torch.uint8)
    try:
        write_video(output_path, frames, fps=fps)
    except ImportError as exc:
        if "PyAV" not in str(exc):
            raise
        import imageio.v2 as imageio

        imageio.mimsave(output_path, frames.numpy(), fps=fps, macro_block_size=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--extended_prompt_path", type=str, default="")
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--num_output_frames", type=int, default=21)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--num_samples_per_prompt", type=int, default=1)
    parser.add_argument("--prompt_embedding_cache_path", type=str, default="")
    parser.add_argument("--offload_generator_before_decode", action="store_true")
    parser.add_argument(
        "--naming",
        type=str,
        default="prompt_index",
        choices=["prompt", "index", "prompt_index", "raw_prompt", "raw_prompt_index"],
    )
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda")
    torch.set_grad_enabled(False)
    set_seed(args.seed)

    config = OmegaConf.load(args.config_path)
    if str(config.virdm_attention_backend) != "flash_attn_2":
        raise ValueError("ViRDM inference requires flash_attn_2")
    if not bool(config.virdm_require_flash_attention):
        raise ValueError("ViRDM inference must require FlashAttention")
    attention_backend = require_flash_attention_2(
        str(config.virdm_flash_attn_version)
    )
    if rank == 0:
        print(f"ATTENTION_BACKEND {attention_backend}", flush=True)

    cached_text_encoder = (
        CachedPromptTextEncoder(args.prompt_embedding_cache_path)
        if args.prompt_embedding_cache_path
        else None
    )

    pipeline = CausalInferencePipeline(
        config, device=device, text_encoder=cached_text_encoder
    )

    generator_state = load_generator_state(args.checkpoint_path, use_ema=args.use_ema)
    pipeline.generator.load_state_dict(generator_state, strict=True, assign=True)
    del generator_state
    gc.collect()

    if pipeline.text_encoder is not None:
        pipeline.text_encoder.to(device=device, dtype=torch.bfloat16)
    pipeline.generator.to(device=device, dtype=torch.bfloat16)
    pipeline.vae.to(device=device, dtype=torch.bfloat16)

    dataset = TextDataset(
        prompt_path=args.prompt_path,
        extended_prompt_path=args.extended_prompt_path or None,
    )
    if args.limit > 0:
        all_indices = range(min(args.limit, len(dataset)))
    else:
        all_indices = range(len(dataset))
    indices = list(all_indices)[rank::world_size]
    if args.num_samples_per_prompt <= 0:
        raise ValueError("--num_samples_per_prompt must be positive")

    os.makedirs(args.output_folder, exist_ok=True)

    for idx in indices:
        batch = dataset[idx]
        prompt = batch["prompts"]
        conditioned_prompt = batch.get("extended_prompts", prompt)
        safe_prompt = sanitize_filename(prompt)
        raw_prompt_name = None
        if args.naming in {"raw_prompt", "raw_prompt_index"}:
            raw_prompt_name = raw_filename(prompt)

        for sample_idx in range(args.num_samples_per_prompt):
            global_idx = idx * args.num_samples_per_prompt + sample_idx
            if args.naming == "index":
                filename = f"{global_idx:04d}.mp4"
            elif args.naming == "prompt":
                suffix = f"-{sample_idx}" if args.num_samples_per_prompt > 1 else ""
                filename = f"{safe_prompt}{suffix}.mp4"
            elif args.naming == "raw_prompt":
                suffix = f"-{sample_idx}" if args.num_samples_per_prompt > 1 else ""
                filename = f"{raw_prompt_name}{suffix}.mp4"
            elif args.naming == "raw_prompt_index":
                filename = f"{raw_prompt_name}-{sample_idx}.mp4"
            else:
                filename = f"{safe_prompt}-{global_idx:04d}.mp4"

            output_path = os.path.join(args.output_folder, filename)
            if os.path.exists(output_path):
                print(f"Skipping existing {output_path}", flush=True)
                continue

            if args.offload_generator_before_decode:
                pipeline.text_encoder.to(device)
                pipeline.generator.to(device)
                torch.cuda.empty_cache()

            generator = torch.Generator(device=device)
            generator.manual_seed(
                args.seed + idx * args.num_samples_per_prompt + sample_idx
            )
            sampled_noise = torch.randn(
                [1, args.num_output_frames, 16, 60, 104],
                device=device,
                dtype=torch.bfloat16,
                generator=generator,
            )

            video, latents = pipeline.inference(
                noise=sampled_noise,
                text_prompts=[conditioned_prompt],
                return_latents=True,
                return_video=not args.offload_generator_before_decode,
            )
            if args.offload_generator_before_decode:
                pipeline.text_encoder.to("cpu")
                pipeline.generator.to("cpu")
                torch.cuda.empty_cache()
                video = pipeline.vae.decode_to_pixel(latents, use_cache=False)
                video = (video * 0.5 + 0.5).clamp(0, 1)

            video = 255.0 * rearrange(video, "b t c h w -> b t h w c").cpu()
            write_video_with_fallback(output_path, video[0], fps=16)
            print(f"Wrote {output_path}", flush=True)
            pipeline.vae.model.clear_cache()

    done_name = "export.done" if world_size == 1 else f"export.rank{rank:03d}.done"
    done_path = os.path.join(args.output_folder, done_name)
    with open(done_path, "w", encoding="utf-8") as fp:
        fp.write("ok\n")
    print(f"Wrote {done_path}", flush=True)


if __name__ == "__main__":
    main()
