from typing import Optional

import torch
from torch import nn

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder


class BaseModel(nn.Module):
    """Generator, frozen text encoder, and shared scheduler."""

    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.args = args
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32

        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}),
        )
        self.generator.model.requires_grad_(True)
        self.text_encoder = (
            None
            if getattr(args, "prompt_embedding_cache_path", "")
            else WanTextEncoder().requires_grad_(False)
        )
        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

        configured_steps = torch.tensor(
            args.denoising_step_list, dtype=torch.long, device=device
        )
        if args.warp_denoising_step:
            timesteps = torch.cat(
                (self.scheduler.timesteps.cpu(), torch.tensor([0.0]))
            ).to(device)
            configured_steps = timesteps[1000 - configured_steps]
        self.denoising_step_list = configured_steps


class ReplayableVideoModel(BaseModel):
    """Shared helpers for replayable video generation."""

    def __init__(self, args, device):
        super().__init__(args, device)
        self.inference_pipeline = None

    def _slice_generated_output(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.shape[1] != 21:
            raise ValueError(f"ViRDM requires 21 latent frames, got {latent.shape[1]}")
        return latent.to(self.dtype)

    def _consistency_backward_simulation(
        self,
        noise: torch.Tensor,
        clean_image_or_video: Optional[torch.Tensor],
        denoise_steps: Optional[int] = None,
        **conditioning,
    ):
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()
        return self.inference_pipeline.inference_with_trajectory(
            noise=noise,
            clean_image_or_video=clean_image_or_video,
            denoise_steps=denoise_steps,
            **conditioning,
        )

    def _initialize_inference_pipeline(self):
        raise NotImplementedError
