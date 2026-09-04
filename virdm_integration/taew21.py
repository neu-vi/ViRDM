"""Frozen TAEW2.1 decoder used by the ViRDM pixel-gradient path."""

from __future__ import annotations

import os

import torch
from torch import nn
from torch.nn import functional as F

from third_party.taehv.taehv import TAEHV, apply_model_with_memblocks


class TAEW21FullDecoder(nn.Module):
    """Decode native Wan2.1 latents with the tiny TAEW2.1 decoder.

    The ViRDM decoder contract remains ``[-1, 1]`` even though the decoder
    TAEW emits ``[0, 1]``.  This keeps V-JEPA preprocessing and rollout-video
    logging consistent with the Wan-VAE path.
    """

    def __init__(
        self,
        *,
        checkpoint_path: str,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        parallel: bool = False,
    ) -> None:
        super().__init__()
        checkpoint_path = os.path.realpath(checkpoint_path)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"TAEW2.1 checkpoint not found: {checkpoint_path}")
        self._apply_model_with_memblocks = apply_model_with_memblocks
        self.decoder = TAEHV(checkpoint_path).to(device=device, dtype=dtype).eval()
        self.decoder.requires_grad_(False)
        self.parallel = bool(parallel)
        self.output_dtype = dtype

    def _decode_unclamped_01(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 5 or tuple(latent.shape[1:]) != (21, 16, 60, 104):
            raise ValueError(
                "TAEW2.1 native decoder requires latent [B,21,16,60,104], got "
                f"{tuple(latent.shape)}"
            )
        if bool(self.decoder.is_h3):
            raise RuntimeError(
                "TAEW2.1 Wan wrapper does not support the H3 decode path"
            )
        skip_trim = bool(self.decoder.is_cogvideox) and latent.shape[1] % 2 == 0
        pixels_01 = self._apply_model_with_memblocks(
            self.decoder.decoder,
            latent.to(dtype=self.output_dtype),
            self.parallel,
            False,
        )
        if int(self.decoder.patch_size) > 1:
            pixels_01 = F.pixel_shuffle(pixels_01, int(self.decoder.patch_size))
        if not skip_trim:
            pixels_01 = pixels_01[:, int(self.decoder.frames_to_trim) :]
        if tuple(pixels_01.shape[1:]) != (81, 3, 480, 832):
            raise RuntimeError(
                "TAEW2.1 decoded an unexpected video shape " f"{tuple(pixels_01.shape)}"
            )
        return pixels_01

    def _decode(
        self, latent: torch.Tensor, *, straight_through_clamp: bool = False
    ) -> torch.Tensor:
        raw_01 = self._decode_unclamped_01(latent)
        clamped_01 = raw_01.clamp(0, 1)
        pixels_01 = (
            raw_01 + (clamped_01 - raw_01).detach()
            if straight_through_clamp
            else clamped_01
        )
        return pixels_01.mul(2.0).sub(1.0)

    @torch.no_grad()
    def decode_value(self, latent: torch.Tensor) -> torch.Tensor:
        return self._decode(latent).to(dtype=self.output_dtype)

    def full_vjp(
        self,
        latent: torch.Tensor,
        pixel_gradient: torch.Tensor,
        *,
        straight_through_clamp: bool = False,
    ) -> torch.Tensor:
        latent_leaf = latent.detach().requires_grad_(True)
        decoded = self._decode(
            latent_leaf,
            straight_through_clamp=straight_through_clamp,
        )
        if tuple(pixel_gradient.shape) != tuple(decoded.shape):
            raise ValueError(
                f"TAEW pixel gradient {tuple(pixel_gradient.shape)} != decoded "
                f"shape {tuple(decoded.shape)}"
            )
        (latent_gradient,) = torch.autograd.grad(
            decoded,
            latent_leaf,
            grad_outputs=pixel_gradient.to(decoded),
            retain_graph=False,
            create_graph=False,
        )
        return latent_gradient.detach().float()
