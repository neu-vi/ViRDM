"""Frozen full-video V-JEPA 2.1 encoder used by ViRDM.

Wan's 81 decoded frames are padded by repeating the final frame once so the
V-JEPA temporal tubelet size of two consumes every original frame.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.utils.checkpoint
from torch import nn

VJEPA21_VITL_SOURCE_REVISION = "204698b45b3712590f06245fbfba32d3be539812"


def _clean_encoder_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state.items():
        key = str(key).replace("module.", "").replace("backbone.", "")
        cleaned[key] = value
    return cleaned


class NativeVJEPA21VideoEncoder(nn.Module):
    """V-JEPA 2.1 ViT-L/16, full 82-frame native-resolution video."""

    def __init__(
        self,
        model_id: str,
        checkpoint_path: str,
        *,
        checkpoint_key: str = "ema_encoder",
        input_height: int = 480,
        input_width: int = 832,
        input_frames: int = 81,
        padded_frames: int = 82,
        temporal_pad: str = "replicate_last",
        feature_dim: int = 1024,
        pool: str = "global_mean_all_tokens",
        activation_checkpointing: bool = True,
        device: torch.device | str = "cuda",
    ) -> None:
        super().__init__()
        if str(model_id) != "vjepa2_1_vit_large_384":
            raise ValueError(f"unsupported V-JEPA 2.1 model_id {model_id!r}")
        if pool not in {"global_mean_all_tokens", "mean_spatial_per_tubelet"}:
            raise ValueError(f"unsupported full-video V-JEPA pooling {pool!r}")
        if (int(input_frames), int(padded_frames), str(temporal_pad)) != (
            81,
            82,
            "replicate_last",
        ):
            raise ValueError(
                "V-JEPA v1 requires 81 -> 82 replicate-last temporal padding"
            )
        if (int(input_height), int(input_width), int(feature_dim)) != (480, 832, 1024):
            raise ValueError("V-JEPA v1 requires native 81x480x832 video and dim=1024")

        source_root = str(
            (Path(__file__).resolve().parents[1] / "third_party" / "vjepa2").resolve()
        )
        if source_root not in sys.path:
            sys.path.insert(0, source_root)
        checkpoint = Path(checkpoint_path).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"V-JEPA 2.1 checkpoint not found: {checkpoint}")

        from app.vjepa_2_1.models.vision_transformer import vit_large

        self.model_id = str(model_id)
        self.checkpoint_path = str(checkpoint)
        self.checkpoint_key = str(checkpoint_key)
        self.source_root = source_root
        self.source_revision = VJEPA21_VITL_SOURCE_REVISION
        self.input_height = int(input_height)
        self.input_width = int(input_width)
        self.input_frames = int(input_frames)
        self.padded_frames = int(padded_frames)
        self.temporal_pad = str(temporal_pad)
        self.feature_dim = int(feature_dim)
        self.pool = str(pool)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.patch_size = 16
        self.tubelet_size = 2
        self.temporal_tokens = self.padded_frames // self.tubelet_size
        self.height_tokens = self.input_height // self.patch_size
        self.width_tokens = self.input_width // self.patch_size
        self.expected_tokens = (
            self.temporal_tokens * self.height_tokens * self.width_tokens
        )

        self.model = vit_large(
            patch_size=self.patch_size,
            img_size=(384, 384),
            num_frames=64,
            tubelet_size=self.tubelet_size,
            use_sdpa=True,
            use_silu=False,
            wide_silu=True,
            uniform_power=False,
            use_rope=True,
            img_temporal_dim_size=1,
            interpolate_rope=True,
            use_activation_checkpointing=False,
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if self.checkpoint_key in payload:
            state = payload[self.checkpoint_key]
        elif all(torch.is_tensor(value) for value in payload.values()):
            state = payload
        else:
            raise KeyError(
                f"V-JEPA checkpoint has no {self.checkpoint_key!r} encoder state"
            )
        incompatible = self.model.load_state_dict(
            _clean_encoder_state(state), strict=True
        )
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"V-JEPA state mismatch: {incompatible}")
        del payload, state
        self.model.to(device).eval().requires_grad_(False)

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1, 1),
            persistent=False,
        )

    @staticmethod
    def pad_video(video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5 or int(video.shape[1]) != 81:
            raise ValueError(
                f"V-JEPA video must be [B,81,3,H,W], got {tuple(video.shape)}"
            )
        return torch.cat([video, video[:, -1:]], dim=1)

    @staticmethod
    def pool_tokens(tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(
                f"V-JEPA tokens must be [B,N,D], got {tuple(tokens.shape)}"
            )
        return tokens.float().mean(dim=1)

    @staticmethod
    def pool_video_tokens(
        tokens: torch.Tensor,
        *,
        temporal_tokens: int,
        height_tokens: int,
        width_tokens: int,
    ) -> dict[str, torch.Tensor]:
        if tokens.ndim != 3:
            raise ValueError(
                f"V-JEPA tokens must be [B,N,D], got {tuple(tokens.shape)}"
            )
        expected = int(temporal_tokens) * int(height_tokens) * int(width_tokens)
        if int(tokens.shape[1]) != expected:
            raise ValueError(f"V-JEPA token count {tokens.shape[1]} != {expected}")
        grid = tokens.float().reshape(
            tokens.shape[0],
            int(temporal_tokens),
            int(height_tokens),
            int(width_tokens),
            tokens.shape[-1],
        )
        tubelet = grid.mean(dim=(2, 3))
        return {
            "global_mean_all_tokens": tubelet.mean(dim=1),
            "mean_spatial_per_tubelet": tubelet,
        }

    def contract(self, *, pool: str | None = None) -> dict[str, Any]:
        pool = self.pool if pool is None else str(pool)
        if pool not in {"global_mean_all_tokens", "mean_spatial_per_tubelet"}:
            raise ValueError(f"unsupported V-JEPA pooling contract {pool!r}")
        return {
            "input_mode": "native_video",
            "input_height": self.input_height,
            "input_width": self.input_width,
            "input_frames": self.input_frames,
            "padded_frames": self.padded_frames,
            "temporal_pad": self.temporal_pad,
            "patch_size": self.patch_size,
            "tubelet_size": self.tubelet_size,
            "temporal_tokens": self.temporal_tokens,
            "height_tokens": self.height_tokens,
            "width_tokens": self.width_tokens,
            "token_count": self.expected_tokens,
            "token_output": "final_layer_norm",
            "pool": pool,
            "pooled_rows_per_video": (
                1 if pool == "global_mean_all_tokens" else self.temporal_tokens
            ),
            "feature_dim": self.feature_dim,
            "feature_normalize": False,
            "mean": [float(value) for value in self.mean.flatten()],
            "std": [float(value) for value in self.std.flatten()],
            "source_revision": self.source_revision,
            "checkpoint_key": self.checkpoint_key,
            "is_causal": False,
            "use_rope": True,
            "interpolate_rope": True,
        }

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    def _forward_tokens(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5 or tuple(video.shape[1:3]) != (81, 3):
            raise ValueError(
                f"V-JEPA video must be [B,81,3,H,W], got {tuple(video.shape)}"
            )
        if tuple(video.shape[-2:]) != (self.input_height, self.input_width):
            raise ValueError(
                f"V-JEPA native input must be {(self.input_height, self.input_width)}, "
                f"got {tuple(video.shape[-2:])}"
            )
        padded = self.pad_video(video).permute(0, 2, 1, 3, 4)
        normalized = (padded.float() - self.mean) / self.std
        self.model.use_activation_checkpointing = bool(
            self.activation_checkpointing and torch.is_grad_enabled()
        )
        with torch.autocast("cuda", enabled=video.is_cuda, dtype=torch.bfloat16):
            tokens = self.model(normalized, training=False)
        if tuple(tokens.shape[1:]) != (self.expected_tokens, self.feature_dim):
            raise RuntimeError(
                f"V-JEPA produced tokens {tuple(tokens.shape)}, expected "
                f"[B,{self.expected_tokens},{self.feature_dim}]"
            )
        return tokens

    def forward_views(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute global and 41-tubelet views from one backbone forward."""

        return self.pool_video_tokens(
            self._forward_tokens(video),
            temporal_tokens=self.temporal_tokens,
            height_tokens=self.height_tokens,
            width_tokens=self.width_tokens,
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        pooled = self.forward_views(video)[self.pool]
        if self.pool == "mean_spatial_per_tubelet":
            # Present temporal tubelets to the MMD code as ordinary rows while
            # retaining one full-video backbone forward and its joint VJP.
            pooled = pooled.flatten(0, 1)
        return pooled
