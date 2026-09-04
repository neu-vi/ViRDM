"""Focused trainer for text-conditioned ViRDM updates."""

from __future__ import annotations
from dataclasses import dataclass
import gc
import math
import os
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from model import ViRDM
from third_party.dense_flow import DenseFlowModel
from virdm_integration import (
    VIRDM_JOINT_POOL,
    couple_video_text,
    load_reference,
    load_frozen_text_table,
    mmd_nystrom_with_terms,
    prompt_rows_sha256,
    virdm_joint_contract,
    self_normalize_virdm_loss,
)
from virdm_integration.reference import file_sha256
from utils.dataset import PromptLMDBDataset, cycle
from utils.distributed import (
    differentiable_all_gather,
    fsdp_state_dict,
    fsdp_wrap,
    get_fsdp_wrap_kwargs,
    launch_distributed_job,
)
from utils.misc import resolve_base_seed, set_seed
from utils.prompt_embedding_cache import PromptEmbeddingLMDBCache
from wan.modules.attention import require_flash_attention_2


def _normalize_state_dict_keys(state_dict):
    fixed = {}
    for key, value in state_dict.items():
        if key.startswith("model._fsdp_wrapped_module."):
            key = key.replace("model._fsdp_wrapped_module.", "model.", 1)
        fixed[key] = value
    return fixed


@dataclass
class _RolloutChunk:
    noise: torch.Tensor
    context_noise: torch.Tensor
    first_frame_exit_index: int
    first_frame_trajectory_noise: torch.Tensor
    block_exit_indices: torch.Tensor
    block_trajectory_noise: torch.Tensor
    conditional: dict
    latent: torch.Tensor
    dataset_index: int
    prompt: str


class Trainer:
    """Focused ViRDM trainer."""

    model_cls = ViRDM
    dataset_cls = PromptLMDBDataset

    def __init__(self, config):
        self._validate_static_config(config)
        self.dynamic_reg_enabled = bool(config.virdm_dynamic_reg_enabled)
        self.dynamic_reg_weight = float(config.virdm_dynamic_reg_weight)
        self.dynamic_reg_model = None
        self.dataset_cls = PromptLMDBDataset
        launch_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        launch_local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        if launch_world_size != int(config.virdm_expected_world_size):
            raise RuntimeError(
                f"ViRDM world size {launch_world_size} != {config.virdm_expected_world_size}"
            )
        if launch_local_world_size != int(config.virdm_expected_local_world_size):
            raise RuntimeError(
                f"ViRDM local world size {launch_local_world_size} != {config.virdm_expected_local_world_size}"
            )
        self._initialize_training_runtime(config)
        self.generator_updates = 0
        self.grad_accum = int(config.virdm_grad_accum_steps)
        self.virdm_generation_mode = str(config.virdm_generation_mode)
        self.bidirectional_full_video = (
            self.virdm_generation_mode == "full_video_bidirectional"
        )
        self.bidirectional_random_x0_4step = bool(
            config.virdm_bidirectional_random_x0_4step
        )
        self.all_chunks_random_x0_4step = bool(config.virdm_all_chunks_random_x0_4step)
        self.heterogeneous_chunkwise_schedule = bool(
            self.all_chunks_random_x0_4step
            and list(config.virdm_first_frame_denoising_step_list)
        )
        self.rollout_step_count = len(list(config.denoising_step_list))
        self.log_full_all_chunks_rollout = bool(
            config.virdm_log_rollout_force_all_chunks_full
        )
        expected_rows = self.world_size * self.grad_accum * int(config.batch_size)
        if expected_rows != int(config.virdm_expected_global_rows):
            raise ValueError(
                f"ViRDM global rows {expected_rows} != {config.virdm_expected_global_rows}"
            )
        from virdm_integration.vjepa21_video import NativeVJEPA21VideoEncoder

        self.virdm_encoder = NativeVJEPA21VideoEncoder(
            config.virdm_encoder_id,
            config.virdm_encoder_checkpoint_path,
            checkpoint_key=str(config.virdm_encoder_checkpoint_key),
            input_height=int(config.virdm_input_height),
            input_width=int(config.virdm_input_width),
            input_frames=int(config.virdm_video_frames),
            padded_frames=int(config.virdm_video_padded_frames),
            temporal_pad=str(config.virdm_video_temporal_pad),
            feature_dim=int(config.virdm_feature_dim),
            pool=str(config.virdm_pool),
            activation_checkpointing=bool(
                config.virdm_encoder_activation_checkpointing
            ),
            device=self.device,
        )
        checkpoint_sha = file_sha256(config.virdm_encoder_checkpoint_path)
        contract = virdm_joint_contract(
            self.virdm_encoder.contract(),
            text_table_sha256=str(config.virdm_joint_text_table_sha256),
            prompt_rows_sha256_value=str(config.virdm_joint_prompt_rows_sha256),
            bandwidth_scale=float(config.virdm_joint_bandwidth_scale),
            visual_feature_dim=int(config.virdm_feature_dim),
            text_feature_dim=int(config.virdm_joint_text_feature_dim),
            text_model=str(config.virdm_joint_text_encoder_id),
            text_pretrained=str(config.virdm_joint_text_encoder_pretrained),
        )
        self.virdm_reference = load_reference(
            config.virdm_reference_bundle_path,
            device=self.device,
            expected_encoder_id=config.virdm_encoder_id,
            expected_checkpoint_sha256=checkpoint_sha,
            expected_height=int(config.virdm_input_height),
            expected_width=int(config.virdm_input_width),
            expected_rows=int(config.virdm_expected_reference_rows),
            expected_reference_revision=str(config.virdm_reference_revision),
            expected_encoder_contract=contract,
            expected_landmarks=int(config.virdm_nystrom_landmarks),
            expected_feature_dim=int(config.virdm_joint_feature_dim),
            expected_input_mode=str(config.virdm_input_mode),
            expected_pool=VIRDM_JOINT_POOL,
            expected_rows_per_video=1,
        )
        if self.virdm_reference.sha256 != str(
            config.virdm_reference_bundle_sha256
        ):
            raise RuntimeError("ViRDM reference checksum mismatch")
        if self.virdm_reference.beta is None:
            raise ValueError("ViRDM reference is missing its text scale")
        metadata_beta = float(self.virdm_reference.metadata.get("joint_beta", math.nan))
        if not math.isclose(
            float(self.virdm_reference.beta), metadata_beta, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("ViRDM text scale does not match reference metadata")
        self.joint_text_beta = float(self.virdm_reference.beta)
        self.joint_text_table = load_frozen_text_table(
            config.virdm_joint_text_table_path,
            expected_sha256=str(config.virdm_joint_text_table_sha256),
            expected_rows=int(config.virdm_expected_reference_rows),
            expected_dim=int(config.virdm_joint_text_feature_dim),
            device=self.device,
        )
        prompt_contract = [None, None]
        if self.is_main_process:
            prompts = [
                str(self.dataset[index]["prompts"])
                for index in range(len(self.dataset))
            ]
            prompt_contract = [len(prompts), prompt_rows_sha256(prompts)]
        dist.broadcast_object_list(prompt_contract, src=0)
        expected_prompt_contract = [
            int(config.virdm_expected_reference_rows),
            str(config.virdm_joint_prompt_rows_sha256),
        ]
        if prompt_contract != expected_prompt_contract:
            raise RuntimeError(
                f"prompt table {prompt_contract} != {expected_prompt_contract}"
            )
        if file_sha256(config.virdm_taew_checkpoint_path) != str(
            config.virdm_taew_checkpoint_sha256
        ):
            raise RuntimeError("TAEW2.1 checkpoint checksum mismatch")
        from virdm_integration.taew21 import TAEW21FullDecoder

        self.taew_decoder = TAEW21FullDecoder(
            checkpoint_path=str(config.virdm_taew_checkpoint_path),
            device=self.device,
            dtype=self.dtype,
            parallel=False,
        )
        self._load_dynamic_reg_model()
        if self.is_main_process:
            print(
                f"Loaded ViRDM: reference_sha256={self.virdm_reference.sha256[:16]} text_scale={self.joint_text_beta} dynamic_reg={self.dynamic_reg_enabled}",
                flush=True,
            )

    def _initialize_training_runtime(self, config) -> None:
        """Build the generator-only distributed runtime used by ViRDM."""
        self.config = config
        self.step = 0
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.attention_backend = self._validate_attention_backend(config)
        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        if self.is_main_process:
            print(f"ATTENTION_BACKEND {self.attention_backend}", flush=True)
        self.disable_wandb = config.disable_wandb
        self.max_steps = int(config.max_steps)
        base_seed = (
            [resolve_base_seed(int(config.seed))] if self.is_main_process else [None]
        )
        dist.broadcast_object_list(base_seed, src=0)
        config.seed = int(base_seed[0])
        set_seed(int(config.seed) + global_rank)
        if self.is_main_process and (not self.disable_wandb):
            wandb_mode = str(getattr(config, "wandb_mode", "online"))
            wandb_host = str(config.wandb_host).strip() or None
            wandb_key = config.wandb_key
            if isinstance(wandb_key, str):
                wandb_key = wandb_key.strip() or None
            if wandb_mode == "online":
                wandb.login(host=wandb_host, key=wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=os.environ.get("VIRDM_RUN_NAME", config.config_name),
                mode=wandb_mode,
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir,
            )
        self.output_path = config.logdir
        self.model = self.model_cls(config, device=self.device)
        self.prompt_embedding_cache = None
        if getattr(config, "prompt_embedding_cache_path", ""):
            self.prompt_embedding_cache = PromptEmbeddingLMDBCache(
                config.prompt_embedding_cache_path
            )
        self.model.generator = fsdp_wrap(
            self.model.generator,
            **get_fsdp_wrap_kwargs(
                config, "generator", default_transformer_modules=["causal_wan_block"]
            ),
        )
        if self.model.text_encoder is not None:
            self.model.text_encoder = fsdp_wrap(
                self.model.text_encoder,
                **get_fsdp_wrap_kwargs(
                    config,
                    "text_encoder",
                    default_cpu_offload=getattr(
                        config, "text_encoder_cpu_offload", False
                    ),
                ),
            )
            if getattr(
                config, "text_encoder_fsdp_wrap_strategy", "none"
            ) == "none" and (not getattr(config, "text_encoder_cpu_offload", False)):
                self.model.text_encoder = self.model.text_encoder.to(
                    device=self.device, dtype=self.dtype
                )
        self.generator_optimizer = torch.optim.AdamW(
            [
                parameter
                for parameter in self.model.generator.parameters()
                if parameter.requires_grad
            ],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )
        self.dataset = self.dataset_cls(
            config.data_path,
            max_pair=int(getattr(config, "max_pair", 100000000.0)),
            readahead=bool(getattr(config, "lmdb_readahead", False)),
        )
        sampler = torch.utils.data.distributed.DistributedSampler(
            self.dataset, shuffle=True, drop_last=True, seed=int(config.seed)
        )
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=getattr(config, "dataloader_num_workers", 8),
        )
        self.dataloader = cycle(dataloader)
        if self.is_main_process:
            print(f"DATASET SIZE {len(self.dataset)}", flush=True)
        if getattr(config, "generator_ckpt", ""):
            self._load_generator_checkpoint(config.generator_ckpt)
        self.max_grad_norm_generator = float(config.max_grad_norm_generator)

    def _load_generator_checkpoint(self, checkpoint_path: str) -> None:
        print(f"Loading pretrained generator from {checkpoint_path}", flush=True)
        expected_sha = str(getattr(self.config, "generator_ckpt_sha256", ""))
        if expected_sha and file_sha256(checkpoint_path) != expected_sha:
            raise RuntimeError("generator initialization checksum mismatch")
        state_dict = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True, mmap=True
        )
        if "generator" in state_dict:
            state_dict = _normalize_state_dict_keys(state_dict["generator"])
        elif "generator_ema" in state_dict:
            state_dict = _normalize_state_dict_keys(state_dict["generator_ema"])
        elif "model" in state_dict:
            state_dict = state_dict["model"]
        self.model.generator.load_state_dict(state_dict, strict=True)

    def _get_conditioning(self, text_prompts):
        if self.prompt_embedding_cache is not None:
            return {
                "prompt_embeds": self.prompt_embedding_cache.get_batch(
                    text_prompts, device=self.device, dtype=self.dtype
                )
            }
        with torch.no_grad():
            return self.model.text_encoder(text_prompts=text_prompts)

    def _load_dynamic_reg_model(self) -> None:
        if not self.dynamic_reg_enabled:
            return
        checkpoint_path = os.path.realpath(
            str(self.config.virdm_dynamic_reg_checkpoint_path)
        )
        expected_sha = str(self.config.virdm_dynamic_reg_checkpoint_sha256)
        actual_sha = file_sha256(checkpoint_path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"dynamic-reg checkpoint SHA {actual_sha} != configured {expected_sha}"
            )
        args = OmegaConf.create(
            {
                "small": False,
                "mixed_precision": False,
                "alternate_corr": False,
                "dropout": 0,
            }
        )
        model = DenseFlowModel(args)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint = {
            key.removeprefix("module."): value for (key, value) in checkpoint.items()
        }
        model.load_state_dict(checkpoint, strict=True)
        model = model.to(device=self.device, dtype=torch.float32).eval()
        model.requires_grad_(False)
        self.dynamic_reg_model = model

    @staticmethod
    def _dynamic_reg_top5_flow_mean(flow: torch.Tensor) -> torch.Tensor:
        if flow.ndim != 4 or flow.shape[1] != 2:
            raise ValueError(f"flow must be [B,2,H,W], got {tuple(flow.shape)}")
        magnitude = torch.linalg.vector_norm(flow.float(), dim=1).flatten(1)
        top_count = int(magnitude.shape[1] * 0.05)
        if top_count <= 0:
            raise ValueError("flow map is too small for a top-5% mean")
        return torch.topk(
            magnitude, top_count, dim=1, largest=True, sorted=False
        ).values.mean(dim=1)

    def _dynamic_reg_pair_scores(self, frames_255: torch.Tensor) -> torch.Tensor:
        if self.dynamic_reg_model is None:
            raise RuntimeError("dynamic-reg flow model was not initialized")
        if frames_255.ndim != 4 or frames_255.shape[1] != 3:
            raise ValueError(
                f"dynamic-reg frames must be [T,3,H,W], got {tuple(frames_255.shape)}"
            )
        pair_batch = int(self.config.virdm_dynamic_reg_pair_batch)
        rows = []
        for start in range(0, frames_255.shape[0] - 1, pair_batch):
            stop = min(start + pair_batch, frames_255.shape[0] - 1)
            (_, flow) = self.dynamic_reg_model(
                frames_255[start:stop],
                frames_255[start + 1 : stop + 1],
                iters=int(self.config.virdm_dynamic_reg_iters),
                test_mode=True,
            )
            rows.append(self._dynamic_reg_top5_flow_mean(flow))
        return torch.cat(rows, dim=0)

    def _dynamic_reg_s10_hinge_latent_gradients(
        self, chunks: list[_RolloutChunk]
    ) -> tuple[list[torch.Tensor], dict[str, float]]:
        if not self.dynamic_reg_enabled:
            raise RuntimeError("dynamic-reg gradients requested while disabled")
        threshold = float(self.config.virdm_dynamic_reg_flow_threshold)
        required_hits = int(self.config.virdm_dynamic_reg_required_hits)
        frame_stride = int(
            round(
                float(self.config.virdm_dynamic_reg_video_fps)
                / float(self.config.virdm_dynamic_reg_eval_fps)
            )
        )
        sampled_frames = (81 + frame_stride - 1) // frame_stride
        expected_pairs = sampled_frames - 1
        gradients = []
        score_rows = []
        replay_score_errors = []
        for accum_index, chunk in enumerate(chunks, start=1):
            self._progress("dynamic_reg_score", accum_index=accum_index)
            with torch.no_grad():
                decoded = self._decode_virdm_value(chunk.latent.detach())
                frames_255 = (
                    (decoded[0, ::frame_stride].float() * 0.5 + 0.5)
                    .clamp(0.0, 1.0)
                    .mul(255.0)
                )
                if frames_255.shape[0] != sampled_frames:
                    raise RuntimeError(
                        f"dynamic-reg expected {sampled_frames} sampled frames, got {frames_255.shape[0]}"
                    )
                scores = self._dynamic_reg_pair_scores(frames_255)
                if scores.shape != (expected_pairs,):
                    raise RuntimeError(
                        f"dynamic-reg expected {expected_pairs} pair scores, got {tuple(scores.shape)}"
                    )
                (top_scores, top_indices) = torch.topk(
                    scores, required_hits, largest=True, sorted=True
                )
                selected_score = top_scores[-1]
                selected_index = int(top_indices[-1])
                is_active = bool(selected_score < threshold)
                score_rows.append(scores.detach())
                del frames_255, top_scores, top_indices
            if not is_active:
                gradients.append(torch.zeros_like(chunk.latent, dtype=torch.float32))
                replay_score_errors.append(0.0)
                del decoded
                continue
            decoded_leaf = decoded.detach().float().requires_grad_(True)
            del decoded
            selected_frames = (
                (
                    decoded_leaf[
                        0,
                        selected_index * frame_stride : selected_index * frame_stride
                        + frame_stride
                        + 1 : frame_stride,
                    ]
                    * 0.5
                    + 0.5
                )
                .clamp(0.0, 1.0)
                .mul(255.0)
            )
            replay_score = self._dynamic_reg_pair_scores(selected_frames)[0]
            raw_loss = F.relu(threshold - replay_score)
            pixel_gradient = torch.autograd.grad(
                raw_loss, decoded_leaf, only_inputs=True
            )[0]
            replay_score_errors.append(
                float((replay_score.detach() - selected_score).abs())
            )
            gradient = self._virdm_pixel_vjp_to_latent(
                chunk.latent, pixel_gradient.to(chunk.latent)
            )
            if not torch.isfinite(gradient).all():
                raise RuntimeError("dynamic-reg latent gradient contains NaN or Inf")
            gradients.append(gradient.detach().float())
            del decoded_leaf, selected_frames, replay_score, raw_loss, pixel_gradient
            torch.cuda.empty_cache()
        local_scores = torch.stack(score_rows, dim=0)
        gathered = [torch.empty_like(local_scores) for _ in range(self.world_size)]
        dist.all_gather(gathered, local_scores)
        scores = torch.cat(gathered, dim=0)
        top_scores = torch.topk(
            scores, required_hits, dim=1, largest=True, sorted=True
        ).values
        s10 = top_scores[:, -1]
        raw_losses = F.relu(threshold - s10)
        hit_counts = (scores > threshold).sum(dim=1)
        local_replay_error = torch.tensor(
            [max(replay_score_errors, default=0.0)],
            device=self.device,
            dtype=torch.float32,
        )
        dist.all_reduce(local_replay_error, op=dist.ReduceOp.MAX)
        logs = {
            "dynamic_reg/raw_loss": float(raw_losses.mean()),
            "dynamic_reg/weighted_loss": float(raw_losses.mean())
            * self.dynamic_reg_weight,
            "dynamic_reg/weight": self.dynamic_reg_weight,
            "dynamic_reg/threshold_px": threshold,
            "dynamic_reg/s10_mean_px": float(s10.mean()),
            "dynamic_reg/s10_min_px": float(s10.min()),
            "dynamic_reg/s10_q25_px": float(torch.quantile(s10, 0.25)),
            "dynamic_reg/pair_score_mean_px": float(scores.mean()),
            "dynamic_reg/pair_score_max_px": float(scores.max()),
            "dynamic_reg/hit_count_mean": float(hit_counts.float().mean()),
            "dynamic_reg/pass_fraction": float(
                (hit_counts >= required_hits).float().mean()
            ),
            "dynamic_reg/active_fraction": float((s10 < threshold).float().mean()),
            "dynamic_reg/replay_score_max_abs": float(local_replay_error),
        }
        return (gradients, logs)

    def _dynamic_reg_latent_gradients(
        self, chunks: list[_RolloutChunk]
    ) -> tuple[list[torch.Tensor], dict[str, float]]:
        return self._dynamic_reg_s10_hinge_latent_gradients(chunks)

    def _decode_virdm_value(self, latent: torch.Tensor) -> torch.Tensor:
        if self.taew_decoder is None:
            raise RuntimeError("TAEW2.1 decoder was not initialized")
        return self.taew_decoder.decode_value(latent)

    def _couple_joint_text_feature(
        self,
        visual_features: torch.Tensor,
        chunk: _RolloutChunk,
        *,
        beta: float | None = None,
    ) -> torch.Tensor:
        """Apply the pinned ``[phi | beta*tau]`` feature coupling."""
        if self.joint_text_table is None or self.joint_text_beta is None:
            raise RuntimeError("joint text artifacts were not initialized")
        if tuple(visual_features.shape) != (1, int(self.config.virdm_feature_dim)):
            raise RuntimeError(
                f"joint feature coupling requires one visual row, got {tuple(visual_features.shape)}"
            )
        index = int(chunk.dataset_index)
        if not 0 <= index < int(self.joint_text_table.shape[0]):
            raise IndexError(f"joint text dataset index {index} is out of range")
        text_rows = self.joint_text_table[index : index + 1]
        return couple_video_text(
            visual_features,
            text_rows,
            self.joint_text_beta if beta is None else float(beta),
        )

    def _virdm_pixel_vjp_to_latent(
        self,
        latent: torch.Tensor,
        pixel_gradient: torch.Tensor,
        *,
        straight_through_clamp: bool = False,
    ) -> torch.Tensor:
        if self.taew_decoder is None:
            raise RuntimeError("TAEW2.1 decoder was not initialized")
        gradient = self.taew_decoder.full_vjp(
            latent, pixel_gradient, straight_through_clamp=straight_through_clamp
        )
        if tuple(gradient.shape) != tuple(latent.shape):
            raise RuntimeError(
                f"latent VJP {tuple(gradient.shape)} != {tuple(latent.shape)}"
            )
        return gradient

    @staticmethod
    def _validate_attention_backend(config) -> str:
        """Fail closed when reproducing the reported ViRDM training recipe."""
        expected_backend = str(config.virdm_attention_backend)
        expected_version = str(config.virdm_flash_attn_version)
        if expected_backend != "flash_attn_2":
            raise ValueError(
                "the released ViRDM training recipe requires flash_attn_2"
            )
        if not bool(config.virdm_require_flash_attention):
            raise ValueError(
                "the released ViRDM training recipe must require FlashAttention"
            )
        return require_flash_attention_2(expected_version)

    @staticmethod
    def _validate_static_config(config) -> None:
        forbidden_values = {"resume_ckpt": ""}
        for name, expected in forbidden_values.items():
            actual = getattr(config, name, expected)
            if actual != expected:
                raise ValueError(f"ViRDM requires {name}={expected!r}, got {actual!r}")
        exact = {
            "trainer": "virdm",
            "distribution_loss": "virdm",
            "batch_size": 1,
            "num_training_frames": 21,
            "same_step_across_blocks": True,
            "i2v": False,
            "independent_first_frame": False,
            "virdm_encoder_id": "vjepa2_1_vit_large_384",
            "virdm_input_mode": "native_video",
            "virdm_input_height": 480,
            "virdm_input_width": 832,
            "virdm_pool": "global_mean_all_tokens",
            "virdm_feature_dim": 1024,
            "virdm_video_frames": 81,
            "virdm_video_padded_frames": 82,
            "virdm_video_temporal_pad": "replicate_last",
            "virdm_joint_bandwidth_scale": 1.0,
            "virdm_joint_feature_dim": 2176,
            "virdm_expected_global_rows": 64,
            "virdm_nystrom_landmarks": 4096,
            "virdm_log_rollout_video": True,
            "virdm_attention_backend": "flash_attn_2",
            "virdm_flash_attn_version": "2.8.3.post1",
            "virdm_require_flash_attention": True,
        }
        for name, expected in exact.items():
            actual = getattr(config, name)
            if actual != expected:
                raise ValueError(f"ViRDM requires {name}={expected!r}, got {actual!r}")
        world_size = int(config.virdm_expected_world_size)
        local_world_size = int(config.virdm_expected_local_world_size)
        grad_accum = int(config.virdm_grad_accum_steps)
        if world_size not in {1, 8} or local_world_size != world_size:
            raise ValueError(
                "ViRDM supports a single-node 1-GPU or 8-GPU training topology"
            )
        if grad_accum * world_size * int(config.batch_size) != 64:
            raise ValueError(
                "ViRDM requires a global generated population of 64 videos"
            )
        mode = str(config.virdm_generation_mode)
        if mode == "chunkwise":
            later_step_count = len(list(config.denoising_step_list))
            if later_step_count not in {1, 2, 4}:
                raise ValueError(
                    "chunkwise ViRDM requires one, two, or four later steps"
                )
            first_chunk_steps = list(config.virdm_first_frame_denoising_step_list)
            first_chunk_enhancement = later_step_count < 4
            rollout_exact = {
                "num_frame_per_block": 3,
                "virdm_all_chunks_random_x0_4step": True,
                "virdm_bidirectional_random_x0_4step": False,
                "virdm_replay_first_frame_trajectory_noise": (
                    first_chunk_enhancement
                ),
                "virdm_log_rollout_force_all_chunks_full": True,
            }
            expected_first_chunk_steps = (
                [1000, 750, 500, 250] if first_chunk_enhancement else []
            )
            if first_chunk_steps != expected_first_chunk_steps:
                raise ValueError(
                    "chunkwise ViRDM first-chunk schedule does not match its recipe"
                )
        elif mode == "full_video_bidirectional":
            rollout_exact = {
                "num_frame_per_block": 21,
                "virdm_all_chunks_random_x0_4step": False,
                "virdm_bidirectional_random_x0_4step": True,
                "virdm_replay_first_frame_trajectory_noise": True,
                "virdm_log_rollout_force_all_chunks_full": False,
            }
            if list(config.denoising_step_list) != [1000, 750, 500, 250]:
                raise ValueError("bidirectional ViRDM requires four denoising steps")
            if list(config.virdm_first_frame_denoising_step_list):
                raise ValueError(
                    "bidirectional ViRDM has no separate first-chunk steps"
                )
        else:
            raise ValueError(f"unsupported ViRDM generation mode {mode!r}")
        for name, expected in rollout_exact.items():
            actual = getattr(config, name)
            if actual != expected:
                raise ValueError(f"ViRDM requires {name}={expected!r}, got {actual!r}")
        if bool(config.virdm_dynamic_reg_enabled):
            weight = float(config.virdm_dynamic_reg_weight)
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError("dynamic regularizer weight must be positive")

    def _next_virdm_batch(self):
        return next(self.dataloader)

    def _sync_time(self, start: float) -> float:
        torch.cuda.synchronize(self.device)
        return time.perf_counter() - start

    def _global_max_scalar(self, value: float) -> float:
        tensor = torch.tensor([value], device=self.device, dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return float(tensor.item())

    def _progress(
        self, phase: str, *, accum_index: int | None = None, detail: str = ""
    ) -> None:
        if not self.is_main_process:
            return
        fields = [
            "VIRDM_PROGRESS",
            f"step={self.step}",
            f"update={self.generator_updates + int(phase != 'generator_complete')}",
            f"phase={phase}",
        ]
        if accum_index is not None:
            fields.append(f"accum={accum_index}/{self.grad_accum}")
        if detail:
            fields.append(detail)
        fields.extend(
            [
                f"gpu_allocated_gib={torch.cuda.memory_allocated(self.device) / 2 ** 30:.3f}",
                f"gpu_reserved_gib={torch.cuda.memory_reserved(self.device) / 2 ** 30:.3f}",
            ]
        )
        print(" ".join(fields), flush=True)

    def _make_rollout_chunk(self, batch) -> _RolloutChunk:
        prompts = batch["prompts"]
        if len(prompts) != 1:
            raise ValueError(f"ViRDM requires batch size 1, got {len(prompts)}")
        conditional = self._get_conditioning(prompts)
        noise = torch.randn(1, 21, 16, 60, 104, device=self.device, dtype=self.dtype)
        with torch.no_grad():
            (
                latent,
                _,
                _,
                _,
                context_noise,
                first_frame_exit_index,
                first_frame_trajectory_noise,
                block_exit_indices,
                block_trajectory_noise,
            ) = self.model.rollout_from_noise(noise, conditional)
        index = batch.get("idx", -1)
        if torch.is_tensor(index):
            index = int(index.flatten()[0])
        return _RolloutChunk(
            noise=noise.detach(),
            context_noise=context_noise.detach(),
            first_frame_exit_index=int(first_frame_exit_index),
            first_frame_trajectory_noise=first_frame_trajectory_noise.detach(),
            block_exit_indices=block_exit_indices.detach(),
            block_trajectory_noise=block_trajectory_noise.detach(),
            conditional={key: value.detach() for (key, value) in conditional.items()},
            latent=latent.detach(),
            dataset_index=int(index),
            prompt=str(prompts[0]),
        )

    @torch.no_grad()
    def _full_all_chunks_rollout_video(
        self, chunk: _RolloutChunk
    ) -> torch.Tensor | None:
        """Re-run accumulation zero at the deterministic evaluation schedule."""
        if not self.all_chunks_random_x0_4step:
            raise RuntimeError("full chunkwise rollout log requires its isolated mode")
        later_exit = self.rollout_step_count - 1
        fixed_exits = torch.full(
            (7,), later_exit, device=chunk.noise.device, dtype=torch.long
        )
        first_exit = None
        if self.heterogeneous_chunkwise_schedule:
            fixed_exits[0] = 3
            first_exit = 3
        self._progress(
            "rollout_visualization",
            detail=f"first_chunk_exit_index={int(fixed_exits[0])} later_chunk_exit_index={later_exit}",
        )
        (latent, _, _, _, replay_context_noise, _, _, exit_indices, _) = (
            self.model.rollout_from_noise(
                chunk.noise,
                chunk.conditional,
                context_noise_replay=chunk.context_noise,
                first_frame_exit_index_replay=first_exit,
                block_exit_indices_replay=fixed_exits,
            )
        )
        if tuple(exit_indices.shape) != (7,) or bool(
            (exit_indices != fixed_exits).any()
        ):
            raise RuntimeError("visualization rollout did not use its fixed schedule")
        if float((replay_context_noise - chunk.context_noise).abs().max()) != 0.0:
            raise RuntimeError("visualization rollout changed replayed context noise")
        if not self.is_main_process:
            return None
        decoded = self._decode_virdm_value(latent.detach())
        return (
            (decoded[0].float() * 0.5 + 0.5)
            .clamp(0, 1)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .cpu()
        )

    @torch.no_grad()
    def _latent_features(
        self, chunk: _RolloutChunk, *, capture_video: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        decoded = self._decode_virdm_value(chunk.latent)
        video = None
        if capture_video:
            video = (
                (decoded[0].float() * 0.5 + 0.5)
                .clamp(0, 1)
                .mul(255.0)
                .round()
                .to(torch.uint8)
                .cpu()
            )
        images = (decoded.float() * 0.5 + 0.5).clamp(0, 1)
        visual_features = self.virdm_encoder(images).detach().float()
        features = self._couple_joint_text_feature(visual_features, chunk)
        expected_shape = (1, self.virdm_reference.feature_dim)
        if tuple(features.shape) != expected_shape:
            raise RuntimeError(
                f"ViRDM features {tuple(features.shape)} != {expected_shape}"
            )
        return (features.detach().float(), video)

    def _feature_vjp_to_latent(
        self,
        chunk: _RolloutChunk,
        feature_gradient: torch.Tensor,
        pass1_features: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        with torch.no_grad():
            decoded_value = self._decode_virdm_value(chunk.latent)
        decoded = decoded_value.detach().requires_grad_(True)
        images = (decoded.float() * 0.5 + 0.5).clamp(0, 1)
        visual_features = self.virdm_encoder(images)
        replay_features = self._couple_joint_text_feature(visual_features, chunk)
        parity_max = float((replay_features.detach() - pass1_features).abs().max())
        tolerance = float(self.config.virdm_feature_replay_atol)
        if parity_max > tolerance:
            raise RuntimeError(
                f"feature replay mismatch {parity_max:.6g} > {tolerance:.6g}"
            )
        torch.autograd.backward(replay_features, feature_gradient.to(replay_features))
        if decoded.grad is None or not torch.isfinite(decoded.grad).all():
            raise RuntimeError("feature encoder produced an invalid pixel VJP")
        pixel_gradient = decoded.grad.detach()
        del replay_features, visual_features, images, decoded, decoded_value
        torch.cuda.empty_cache()
        latent_gradient = self._virdm_pixel_vjp_to_latent(chunk.latent, pixel_gradient)
        if not torch.isfinite(latent_gradient).all():
            raise RuntimeError("pixel decoder produced an invalid latent VJP")
        return (latent_gradient.detach().float(), parity_max)

    def _mmd_feature_gradients(
        self, chunks: list[_RolloutChunk]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], dict, torch.Tensor | None]:
        capture_video = bool(
            self.is_main_process
            and (not self.disable_wandb)
            and (not self.config.no_visualize)
            and self.config.virdm_log_rollout_video
        )
        pass1 = []
        rollout_video = None
        for index, chunk in enumerate(chunks):
            self._progress("features", accum_index=index + 1)
            (features, video) = self._latent_features(
                chunk, capture_video=capture_video and index == 0
            )
            pass1.append(features)
            if video is not None:
                rollout_video = video
        current_local = torch.cat(pass1, dim=0).detach().requires_grad_(True)
        expected_local_rows = self.grad_accum * int(self.config.batch_size)
        if int(current_local.shape[0]) != expected_local_rows:
            raise RuntimeError(
                f"local rows {current_local.shape[0]} != {expected_local_rows}"
            )
        self._progress(
            "kernel",
            detail=f"local_rows={current_local.shape[0]} global_rows={current_local.shape[0] * self.world_size}",
        )
        global_features = differentiable_all_gather(current_local)
        expected_global_rows = int(self.config.virdm_expected_global_rows)
        if int(global_features.shape[0]) != expected_global_rows:
            raise RuntimeError(
                f"global rows {global_features.shape[0]} != {expected_global_rows}"
            )
        (raw, terms) = mmd_nystrom_with_terms(
            global_features,
            self.virdm_reference,
            gen_chunk=int(self.config.virdm_kernel_gen_chunk),
        )
        normalized = self_normalize_virdm_loss(
            raw, eps=float(self.config.virdm_self_normalize_eps)
        )
        (normalized / float(self.grad_accum)).backward()
        if current_local.grad is None or not torch.isfinite(current_local.grad).all():
            raise RuntimeError("ViRDM feature gradient is missing or non-finite")
        feature_gradients = list(current_local.grad.detach().split(1, dim=0))
        logs = {
            "raw_mmd": float(raw.detach()),
            "normalized_mmd": float(normalized.detach()),
            "k_gg": float(terms["k_gg"].detach()),
            "k_gr_nystrom": float(terms["k_gr_nystrom"].detach()),
            "k_rr": float(terms["k_rr"].detach()),
            "global_rows": float(global_features.shape[0]),
            "global_video_batch": float(self.world_size * len(chunks)),
            "joint_text_beta": float(self.joint_text_beta),
        }
        del global_features
        return (pass1, feature_gradients, logs, rollout_video)

    def _assert_frozen_parameter_grads(self) -> None:
        modules = {
            "encoder": self.virdm_encoder,
            "decoder": self.taew_decoder,
            "dynamic_reg": self.dynamic_reg_model,
        }
        offenders = []
        for module_name, module in modules.items():
            if module is None:
                continue
            if any(
                (
                    parameter.grad is not None
                    and bool(parameter.grad.detach().abs().max() > 0)
                    for parameter in module.parameters()
                )
            ):
                offenders.append(module_name)
        if offenders:
            raise RuntimeError(f"frozen modules accumulated gradients: {offenders}")

    def _generator_update(self) -> dict:
        phase_start = time.perf_counter()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        self.model.eval()
        self.virdm_encoder.eval()
        self.generator_optimizer.zero_grad(set_to_none=True)
        rollout_start = time.perf_counter()
        self._progress("generator_begin", detail=f"accum_total={self.grad_accum}")
        chunks = []
        for accum_index in range(self.grad_accum):
            self._progress("rollout", accum_index=accum_index + 1)
            chunk = self._make_rollout_chunk(self._next_virdm_batch())
            chunks.append(chunk)
            shared_exit = (
                int(chunk.block_exit_indices[0])
                if chunk.block_exit_indices.numel()
                else chunk.first_frame_exit_index
            )
            self._progress(
                "rollout_complete",
                accum_index=accum_index + 1,
                detail=f"first_exit={chunk.first_frame_exit_index} shared_exit={shared_exit}",
            )
        forced_rollout_video = None
        if (
            self.log_full_all_chunks_rollout
            and (not self.disable_wandb)
            and (not self.config.no_visualize)
            and self.config.virdm_log_rollout_video
        ):
            forced_rollout_video = self._full_all_chunks_rollout_video(chunks[0])
        rollout_seconds = self._sync_time(rollout_start)
        feature_start = time.perf_counter()
        (pass1_features, feature_gradients, logs, rollout_video) = (
            self._mmd_feature_gradients(chunks)
        )
        if forced_rollout_video is not None:
            rollout_video = forced_rollout_video
        logs["timing/feature_and_kernel_seconds"] = self._sync_time(feature_start)
        logs["generation/bidirectional"] = float(self.bidirectional_full_video)
        first_exits = torch.tensor(
            [chunk.first_frame_exit_index for chunk in chunks],
            device=self.device,
            dtype=torch.float32,
        )
        if self.all_chunks_random_x0_4step:
            chunk_exits = torch.stack([chunk.block_exit_indices for chunk in chunks])
            if tuple(chunk_exits.shape[1:]) != (7,):
                raise RuntimeError("chunkwise rollout must contain seven chunks")
            if self.heterogeneous_chunkwise_schedule:
                if bool((chunk_exits[:, 0].float() != first_exits).any()):
                    raise RuntimeError("first-chunk replay state is inconsistent")
                if bool((chunk_exits[:, 1:] != chunk_exits[:, 1:2]).any()):
                    raise RuntimeError("later chunks did not share one exit")
                later_exits = chunk_exits[:, 1]
                logs["first_chunk_exit_mean"] = float(first_exits.mean())
                logs["later_chunk_exit_mean"] = float(later_exits.float().mean())
                del later_exits
            else:
                if bool((first_exits != -1).any()):
                    raise RuntimeError(
                        "shared-schedule chunkwise rollout used a separate first exit"
                    )
                if bool((chunk_exits != chunk_exits[:, :1]).any()):
                    raise RuntimeError("chunks did not share one exit")
                logs["shared_chunk_exit_mean"] = float(
                    chunk_exits[:, 0].float().mean()
                )
            del chunk_exits
        elif self.bidirectional_full_video:
            if bool(
                ((first_exits < 0) | (first_exits >= self.rollout_step_count)).any()
            ):
                raise RuntimeError("invalid bidirectional exit")
            logs["bidirectional_exit_mean"] = float(first_exits.mean())
        del first_exits
        if rollout_video is not None:
            logs["rollout_video"] = wandb.Video(
                rollout_video.numpy(),
                fps=int(self.config.virdm_rollout_video_fps),
                format="mp4",
                caption=f"step={self.step}, dataset_index={chunks[0].dataset_index}, prompt={chunks[0].prompt}",
            )
            del rollout_video
        vjp_start = time.perf_counter()
        mmd_gradients = []
        feature_parity_max = 0.0
        for index, (chunk, feature_gradient, cached_features) in enumerate(
            zip(chunks, feature_gradients, pass1_features), start=1
        ):
            self._progress("feature_vjp", accum_index=index)
            gc.collect()
            torch.cuda.empty_cache()
            (gradient, parity) = self._feature_vjp_to_latent(
                chunk, feature_gradient, cached_features
            )
            mmd_gradients.append(gradient)
            feature_parity_max = max(feature_parity_max, parity)
        logs["feature_replay_max_abs"] = feature_parity_max
        logs["timing/feature_vjp_seconds"] = self._sync_time(vjp_start)
        del feature_gradients, pass1_features
        torch.cuda.empty_cache()
        if self.dynamic_reg_enabled:
            dynamic_start = time.perf_counter()
            (dynamic_gradients, dynamic_logs) = self._dynamic_reg_latent_gradients(
                chunks
            )
            if len(dynamic_gradients) != len(mmd_gradients):
                raise RuntimeError("dynamic regularizer gradient count mismatch")
            coefficient = self.dynamic_reg_weight / float(self.grad_accum)
            combined = [
                mmd.float() + coefficient * dynamic.float()
                for (mmd, dynamic) in zip(mmd_gradients, dynamic_gradients)
            ]
            logs.update(dynamic_logs)
            logs["timing/dynamic_reg_vjp_seconds"] = self._sync_time(dynamic_start)
            del dynamic_gradients
        else:
            combined = mmd_gradients
        backward_start = time.perf_counter()
        replay_max = 0.0
        for index, (chunk, total_gradient) in enumerate(zip(chunks, combined), start=1):
            self._progress("generator_replay", accum_index=index)
            (
                recomputed,
                _,
                _,
                _,
                replay_context_noise,
                replay_first_exit,
                replay_first_trajectory,
                replay_block_exits,
                replay_block_trajectory,
            ) = self.model.rollout_from_noise(
                chunk.noise,
                chunk.conditional,
                context_noise_replay=chunk.context_noise,
                first_frame_exit_index_replay=chunk.first_frame_exit_index,
                first_frame_trajectory_noise_replay=chunk.first_frame_trajectory_noise,
                block_exit_indices_replay=chunk.block_exit_indices,
                block_trajectory_noise_replay=chunk.block_trajectory_noise,
            )
            if replay_first_exit != chunk.first_frame_exit_index:
                raise RuntimeError("first exit changed during replay")
            if not torch.equal(replay_block_exits, chunk.block_exit_indices):
                raise RuntimeError("chunk exits changed during replay")
            if float((replay_context_noise - chunk.context_noise).abs().max()) != 0.0:
                raise RuntimeError("context noise changed during replay")
            if (
                chunk.first_frame_trajectory_noise.numel()
                and float(
                    (replay_first_trajectory - chunk.first_frame_trajectory_noise)
                    .abs()
                    .max()
                )
                != 0.0
            ):
                raise RuntimeError("first trajectory noise changed during replay")
            if (
                chunk.block_trajectory_noise.numel()
                and float(
                    (replay_block_trajectory - chunk.block_trajectory_noise).abs().max()
                )
                != 0.0
            ):
                raise RuntimeError("chunk trajectory noise changed during replay")
            replay_max = max(
                replay_max, float((recomputed.detach() - chunk.latent).abs().max())
            )
            torch.autograd.backward(recomputed, total_gradient.to(recomputed))
            del recomputed
        replay_tolerance = float(self.config.virdm_rollout_replay_atol)
        if replay_max > replay_tolerance:
            raise RuntimeError(
                f"rollout replay mismatch {replay_max:.6g} > {replay_tolerance:.6g}"
            )
        logs["rollout_replay_max_abs"] = replay_max
        logs["timing/generator_backward_seconds"] = self._sync_time(backward_start)
        self._assert_frozen_parameter_grads()
        generator_grad_norm = self.model.generator.clip_grad_norm_(
            self.max_grad_norm_generator
        )
        if not torch.isfinite(generator_grad_norm):
            raise RuntimeError("generator gradient norm is non-finite")
        self.generator_optimizer.step()
        self.generator_optimizer.zero_grad(set_to_none=True)
        self.generator_updates += 1
        logs["virdm_objective_value_proxy"] = logs["normalized_mmd"] + logs.get(
            "dynamic_reg/weighted_loss", 0.0
        )
        logs.update(
            {
                "generator_grad_norm": float(generator_grad_norm),
                "generator_updates": float(self.generator_updates),
                "timing/rollout_seconds": rollout_seconds,
                "timing/generator_update_seconds": self._sync_time(phase_start),
                "gpu/peak_allocated_gib": self._global_max_scalar(
                    torch.cuda.max_memory_allocated(self.device) / 2**30
                ),
                "gpu/peak_reserved_gib": self._global_max_scalar(
                    torch.cuda.max_memory_reserved(self.device) / 2**30
                ),
            }
        )
        del chunks, mmd_gradients, combined
        gc.collect()
        torch.cuda.empty_cache()
        self._progress("generator_complete")
        return logs

    def _log_step(self, generator_logs: dict) -> None:
        if not self.is_main_process:
            return
        logs = {"step": self.step}
        wandb_generator_keys = (
            "raw_mmd",
            "k_gg",
            "k_gr_nystrom",
            "k_rr",
            "generator_grad_norm",
            "dynamic_reg/raw_loss",
            "dynamic_reg/weighted_loss",
            "dynamic_reg/s10_mean_px",
            "dynamic_reg/hit_count_mean",
            "dynamic_reg/pass_fraction",
            "dynamic_reg/pair_score_mean_px",
            "gpu/peak_allocated_gib",
            "gpu/peak_reserved_gib",
        )
        logs.update(
            {
                f"generator/{key}": generator_logs[key]
                for key in wandb_generator_keys
                if key in generator_logs
            }
        )
        if "rollout_video" in generator_logs:
            logs["rollout/rollout_video"] = generator_logs["rollout_video"]
        if self.disable_wandb:
            print(logs, flush=True)
        else:
            wandb.log(logs, step=self.step)

    def save(self) -> None:
        state_dict = {
            "step": self.step,
            "generator": fsdp_state_dict(self.model.generator),
        }
        if self.is_main_process:
            output_dir = os.path.join(
                self.output_path, f"checkpoint_model_{self.step:06d}"
            )
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "model.pt")
            torch.save(state_dict, output_path)
            print(f"Model saved to {output_path}", flush=True)

    def train(self) -> None:
        while self.step < self.max_steps:
            generator_logs = self._generator_update()
            self.step += 1
            if not self.config.no_save and self.step % self.config.log_iters == 0:
                self.save()
            self._log_step(generator_logs)
            if self.step % self.config.gc_interval == 0:
                gc.collect()
                torch.cuda.empty_cache()
