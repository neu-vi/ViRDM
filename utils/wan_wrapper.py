import os
import types
from typing import List, Optional
import torch

from utils.scheduler import SchedulerInterface, FlowMatchScheduler
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.vae import _video_vae
from wan.modules.t5 import umt5_xxl
from wan.modules.causal_model import CausalWanModel


def _wan_model_root(model_name: str = "Wan2.1-T2V-1.3B") -> str:
    configured = os.environ.get("VIRDM_WAN_MODEL_ROOT")
    return os.path.realpath(configured or os.path.join("wan_models", model_name))


class WanTextEncoder(torch.nn.Module):
    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        assign_load: bool = False,
    ) -> None:
        super().__init__()

        old_default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            self.text_encoder = (
                umt5_xxl(
                    encoder_only=True,
                    return_tokenizer=False,
                    dtype=dtype,
                    device=torch.device("cpu"),
                    init_weights_enabled=False,
                )
                .eval()
                .requires_grad_(False)
            )
        finally:
            torch.set_default_dtype(old_default_dtype)

        self.text_encoder.load_state_dict(
            torch.load(
                os.path.join(_wan_model_root(), "models_t5_umt5-xxl-enc-bf16.pth"),
                map_location="cpu",
                weights_only=False,
            ),
            assign=assign_load,
        )

        self.tokenizer = HuggingfaceTokenizer(
            name=os.path.join(_wan_model_root(), "google", "umt5-xxl"),
            seq_len=512,
            clean="whitespace",
        )

    @property
    def device(self):
        return next(self.text_encoder.parameters()).device

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True
        )
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        return {"prompt_embeds": context}


class WanVAEWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        mean = [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ]
        std = [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        # init model
        self.model = (
            _video_vae(
                pretrained_path=os.path.join(_wan_model_root(), "Wan2.1_VAE.pth"),
                z_dim=16,
            )
            .eval()
            .requires_grad_(False)
        )

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device, dtype = pixel.device, pixel.dtype
        scale = [
            self.mean.to(device=device, dtype=dtype),
            1.0 / self.std.to(device=device, dtype=dtype),
        ]

        output = [
            self.model.encode(u.unsqueeze(0), scale).float().squeeze(0) for u in pixel
        ]
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel(
        self, latent: torch.Tensor, use_cache: bool = False
    ) -> torch.Tensor:
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [
            self.mean.to(device=device, dtype=dtype),
            1.0 / self.std.to(device=device, dtype=dtype),
        ]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for u in zs:
            output.append(
                decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0)
            )
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output


class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
        self,
        model_name="Wan2.1-T2V-1.3B",
        timestep_shift=8.0,
        local_attn_size=-1,
        sink_size=0,
        model_path=None,
    ):
        super().__init__()

        model_path = model_path or _wan_model_root(model_name)
        self.model = CausalWanModel.from_pretrained(
            model_path, local_attn_size=local_attn_size, sink_size=sink_size
        )
        self.model.eval()

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        self.seq_len = 32760  # [1, 21, 16, 60, 104]
        self.tokens_per_frame = self.seq_len // 21
        self.post_init()

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def _convert_flow_pred_to_x0(
        self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device),
            [flow_pred, xt, self.scheduler.sigmas, self.scheduler.timesteps],
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor = None,
        conditional_dict: dict = None,
        timestep: torch.Tensor = None,
        kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
        cache_start: Optional[int] = None,
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]
        runtime_seq_len = max(
            self.tokens_per_frame,
            noisy_image_or_video.shape[1] * self.tokens_per_frame,
        )

        input_timestep = timestep

        if kv_cache is not None:
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep,
                context=prompt_embeds,
                seq_len=runtime_seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start,
            ).permute(0, 2, 1, 3, 4)
        else:
            if clean_x is not None:
                # Condition on provided clean latents.
                flow_pred = self.model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4),  # => [B, C, F, H, W]
                    t=input_timestep,
                    context=prompt_embeds,
                    seq_len=runtime_seq_len,
                    clean_x=clean_x.permute(0, 2, 1, 3, 4),  # => [B, C, F, H, W]
                    aug_t=aug_t,
                ).permute(0, 2, 1, 3, 4)
            else:
                flow_pred = self.model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4),
                    t=input_timestep,
                    context=prompt_embeds,
                    seq_len=runtime_seq_len,
                ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])

        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler
        )
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler
        )
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler
        )
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()
