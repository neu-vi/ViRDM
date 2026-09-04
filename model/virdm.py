"""Causal and bidirectional rollout model for ViRDM training."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from model.base import ReplayableVideoModel
from pipeline import ViRDMTrainingPipeline


class ViRDM(ReplayableVideoModel):
    """Wan generator with replayable ViRDM rollout schedules."""

    def __init__(self, args, device):
        super().__init__(args, device)
        self.num_frame_per_block = int(getattr(args, "num_frame_per_block", 1))
        self.same_step_across_blocks = bool(
            getattr(args, "same_step_across_blocks", True)
        )
        self.num_training_frames = int(getattr(args, "num_training_frames", 21))
        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.independent_first_frame = bool(
            getattr(args, "independent_first_frame", False)
        )
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        if bool(getattr(args, "gradient_checkpointing", False)):
            self.generator.enable_gradient_checkpointing()
        self.inference_pipeline: ViRDMTrainingPipeline | None = None

        self.virdm_generation_mode = str(
            getattr(args, "virdm_generation_mode", "chunkwise")
        )
        self.virdm_bidirectional_full_video = (
            self.virdm_generation_mode == "full_video_bidirectional"
        )
        self.virdm_bidirectional_random_x0_4step = bool(
            getattr(args, "virdm_bidirectional_random_x0_4step", False)
        )
        self.virdm_all_chunks_random_x0_4step = bool(
            getattr(args, "virdm_all_chunks_random_x0_4step", False)
        )
        if self.virdm_all_chunks_random_x0_4step:
            configured_first_block_steps = list(
                getattr(args, "virdm_first_frame_denoising_step_list", [])
            )
            if configured_first_block_steps:
                first_block_steps = torch.tensor(
                    configured_first_block_steps,
                    dtype=torch.long,
                    device=self.device,
                )
                if args.warp_denoising_step:
                    timesteps = torch.cat(
                        (
                            self.scheduler.timesteps.detach().cpu(),
                            torch.tensor([0], dtype=torch.float32),
                        )
                    ).to(self.device)
                    first_block_steps = timesteps[1000 - first_block_steps]
                self.virdm_first_frame_denoising_step_list = first_block_steps
            else:
                self.virdm_first_frame_denoising_step_list = None
        elif self.virdm_bidirectional_full_video:
            self.virdm_first_frame_denoising_step_list = (
                self.denoising_step_list
                if self.virdm_bidirectional_random_x0_4step
                else None
            )
        else:
            first_frame_steps = torch.tensor(
                list(args.virdm_first_frame_denoising_step_list),
                dtype=torch.long,
                device=self.device,
            )
            if args.warp_denoising_step:
                timesteps = torch.cat(
                    (
                        self.scheduler.timesteps.detach().cpu(),
                        torch.tensor([0], dtype=torch.float32),
                    )
                ).to(self.device)
                first_frame_steps = timesteps[1000 - first_frame_steps]
            self.virdm_first_frame_denoising_step_list = first_frame_steps
        self._validate_virdm_config()

    def _validate_virdm_config(self) -> None:
        checks = {
            "i2v": False,
            "num_training_frames": 21,
        }
        for name, expected in checks.items():
            actual = getattr(self.args, name)
            if actual != expected:
                raise ValueError(
                    f"ViRDM v1 requires {name}={expected!r}, got {actual!r}"
                )
        if not bool(self.args.same_step_across_blocks):
            raise ValueError("ViRDM requires same_step_across_blocks=true")
        if self.virdm_generation_mode not in {
            "chunkwise",
            "full_video_bidirectional",
        }:
            raise ValueError(
                f"unsupported virdm_generation_mode={self.virdm_generation_mode!r}"
            )
        block_size = int(self.args.num_frame_per_block)
        supported_block_sizes = {21} if self.virdm_bidirectional_full_video else {3}
        if block_size not in supported_block_sizes:
            raise ValueError(
                f"ViRDM generation mode {self.virdm_generation_mode!r} "
                f"requires num_frame_per_block in {supported_block_sizes}, "
                f"got {block_size}"
            )
        if int(self.args.num_training_frames) % block_size != 0:
            raise ValueError(
                "num_training_frames must be divisible by num_frame_per_block"
            )
        context_noise = int(self.args.context_noise)
        if context_noise not in {0, 100, 357}:
            raise ValueError(
                "ViRDM supports replayable context_noise in {0,100,357}, "
                f"got {context_noise}"
            )
        bidirectional_random_x0_4step = bool(
            getattr(self, "virdm_bidirectional_random_x0_4step", False)
        )
        all_chunks_random_x0_4step = bool(
            getattr(self, "virdm_all_chunks_random_x0_4step", False)
        )
        if bidirectional_random_x0_4step or all_chunks_random_x0_4step:
            if not 1 <= len(self.denoising_step_list) <= 4:
                raise ValueError(
                    "random-x0 replay requires one to four denoising steps, got "
                    f"{self.denoising_step_list.tolist()}"
                )
        elif len(self.denoising_step_list) != 1:
            raise ValueError(
                "ViRDM replay received an unexpected denoising-step count, got "
                f"{self.denoising_step_list.tolist()}"
            )
        configured_first_frame_steps = list(
            self.args.virdm_first_frame_denoising_step_list
        )
        if all_chunks_random_x0_4step:
            if configured_first_frame_steps and configured_first_frame_steps != [
                1000,
                750,
                500,
                250,
            ]:
                raise ValueError(
                    "heterogeneous chunkwise random-x0 requires first-block steps "
                    "[1000,750,500,250]"
                )
            replay_first_block = bool(
                self.args.virdm_replay_first_frame_trajectory_noise
            )
            if replay_first_block != bool(configured_first_frame_steps):
                raise ValueError(
                    "heterogeneous chunkwise random-x0 must replay its first-block "
                    "trajectory when a separate first-chunk schedule is configured"
                )
        elif self.virdm_bidirectional_full_video:
            if configured_first_frame_steps:
                raise ValueError(
                    "bidirectional full-video generation requires an empty "
                    "virdm_first_frame_denoising_step_list"
                )
            replay_trajectory = bool(
                self.args.virdm_replay_first_frame_trajectory_noise
            )
            if replay_trajectory != bidirectional_random_x0_4step:
                raise ValueError(
                    "bidirectional random-x0 truncation must replay its full-video "
                    "trajectory noise"
                )
        if bool(getattr(self.args, "ts_schedule", False)) or bool(
            getattr(self.args, "ts_schedule_max", False)
        ):
            raise ValueError(
                "heterogeneous first-frame rollout requires ts_schedule=false and "
                "ts_schedule_max=false"
            )

    def _initialize_inference_pipeline(self):
        """Create the ViRDM-only replayable first-frame training pipeline."""

        self.inference_pipeline = ViRDMTrainingPipeline(
            denoising_step_list=self.denoising_step_list,
            first_frame_denoising_step_list=(
                self.virdm_first_frame_denoising_step_list
            ),
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=self.args.independent_first_frame,
            same_step_across_blocks=self.args.same_step_across_blocks,
            last_step_only=self.args.last_step_only,
            num_max_frames=self.num_training_frames,
            context_noise=self.args.context_noise,
            gradient_num_frames=self.num_training_frames,
            gradient_window_position="tail",
            bidirectional_full_video=self.virdm_bidirectional_full_video,
            bidirectional_random_x0_4step=(self.virdm_bidirectional_random_x0_4step),
            all_chunks_random_x0_4step=(self.virdm_all_chunks_random_x0_4step),
        )

    def rollout_from_noise(
        self,
        noise: torch.Tensor,
        conditional_dict: dict,
        *,
        context_noise_replay: torch.Tensor | None = None,
        first_frame_exit_index_replay: int | None = None,
        first_frame_trajectory_noise_replay: torch.Tensor | None = None,
        block_exit_indices_replay: torch.Tensor | None = None,
        block_trajectory_noise_replay: torch.Tensor | None = None,
        force_all_chunks_full: bool = False,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        int,
        int,
        torch.Tensor,
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Replayable 21-frame ViRDM rollout from caller-owned noise."""

        expected_tail = (16, 60, 104)
        if noise.ndim != 5 or tuple(noise.shape[2:]) != expected_tail:
            raise ValueError(
                f"noise must be [B,T,{expected_tail}], got {tuple(noise.shape)}"
            )
        num_frames = int(noise.shape[1])
        if num_frames != 21:
            raise ValueError(f"ViRDM requires 21 latent frames, got {num_frames}")
        if force_all_chunks_full and not self.virdm_all_chunks_random_x0_4step:
            raise ValueError("force_all_chunks_full requires all-chunks random-x0 mode")
        if not self.virdm_all_chunks_random_x0_4step:
            if block_exit_indices_replay is not None:
                if block_exit_indices_replay.numel():
                    raise ValueError(
                        "nonempty block-exit replay requires all-chunks random-x0 mode"
                    )
                block_exit_indices_replay = None
            if block_trajectory_noise_replay is not None:
                if block_trajectory_noise_replay.numel():
                    raise ValueError(
                        "nonempty block-trajectory replay requires all-chunks random-x0 mode"
                    )
                block_trajectory_noise_replay = None
        denoise_steps = 0
        if self.virdm_bidirectional_random_x0_4step or (
            self.virdm_all_chunks_random_x0_4step and not force_all_chunks_full
        ):
            denoise_steps = None
        elif force_all_chunks_full:
            denoise_steps = 3
        simulation_kwargs = dict(
            noise=noise,
            clean_image_or_video=None,
            denoise_steps=denoise_steps,
            context_noise_replay=context_noise_replay,
            return_context_noise_replay=True,
            first_frame_exit_index_replay=first_frame_exit_index_replay,
            first_frame_trajectory_noise_replay=(first_frame_trajectory_noise_replay),
            return_first_frame_replay=True,
            block_exit_indices_replay=block_exit_indices_replay,
            block_trajectory_noise_replay=block_trajectory_noise_replay,
            return_block_replay=self.virdm_all_chunks_random_x0_4step,
            **conditional_dict,
        )
        if getattr(self.args, "generator_activation_cpu_offload", False):
            with torch.autograd.graph.save_on_cpu(pin_memory=False):
                simulation_result = self._consistency_backward_simulation(
                    **simulation_kwargs
                )
        else:
            simulation_result = self._consistency_backward_simulation(
                **simulation_kwargs
            )
        if self.virdm_all_chunks_random_x0_4step:
            (
                pred,
                denoised_from,
                denoised_to,
                captured_context_noise,
                first_frame_exit_index,
                captured_first_frame_trajectory_noise,
                captured_block_exit_indices,
                captured_block_trajectory_noise,
            ) = simulation_result
        else:
            (
                pred,
                denoised_from,
                denoised_to,
                captured_context_noise,
                first_frame_exit_index,
                captured_first_frame_trajectory_noise,
            ) = simulation_result
            captured_block_exit_indices = noise.new_empty((0,), dtype=torch.long)
            captured_block_trajectory_noise = noise.new_empty(
                (0, 0, *tuple(noise.shape))
            )
        pred = self._slice_generated_output(pred)
        return (
            pred,
            None,
            -1 if denoised_from is None else denoised_from,
            -1 if denoised_to is None else denoised_to,
            captured_context_noise.detach(),
            int(first_frame_exit_index),
            captured_first_frame_trajectory_noise.detach(),
            captured_block_exit_indices.detach(),
            captured_block_trajectory_noise.detach(),
        )
