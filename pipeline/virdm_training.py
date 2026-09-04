from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import SchedulerInterface
from typing import List, Optional
import torch
import torch.distributed as dist


class ViRDMTrainingPipeline:
    def __init__(
        self,
        denoising_step_list: List[int],
        scheduler: SchedulerInterface,
        generator: WanDiffusionWrapper,
        num_frame_per_block=3,
        independent_first_frame: bool = False,
        same_step_across_blocks: bool = False,
        last_step_only: bool = False,
        num_max_frames: int = 21,
        context_noise: int = 0,
        gradient_num_frames: Optional[int] = None,
        gradient_window_position: str = "tail",
        first_frame_denoising_step_list: Optional[List[int]] = None,
        bidirectional_full_video: bool = False,
        bidirectional_random_x0_4step: bool = False,
        all_chunks_random_x0_4step: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[
                :-1
            ]  # remove the zero timestep for inference
        self.first_frame_denoising_step_list = first_frame_denoising_step_list
        if (
            self.first_frame_denoising_step_list is not None
            and self.first_frame_denoising_step_list[-1] == 0
        ):
            self.first_frame_denoising_step_list = self.first_frame_denoising_step_list[
                :-1
            ]

        # Wan specific hyperparameters
        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise
        self.i2v = False

        self.kv_cache1 = None
        self.kv_cache2 = None
        self.independent_first_frame = independent_first_frame
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only = last_step_only
        self.kv_cache_size = num_max_frames * self.frame_seq_length
        self.gradient_num_frames = gradient_num_frames
        self.gradient_window_position = gradient_window_position
        self.bidirectional_full_video = bool(bidirectional_full_video)
        self.bidirectional_random_x0_4step = bool(bidirectional_random_x0_4step)
        self.all_chunks_random_x0_4step = bool(all_chunks_random_x0_4step)
        self.heterogeneous_chunkwise_schedule = bool(
            self.all_chunks_random_x0_4step
            and self.first_frame_denoising_step_list is not None
        )

        if self.all_chunks_random_x0_4step:
            if self.bidirectional_full_video:
                raise ValueError(
                    "all-chunks random-x0 and bidirectional rollout are mutually exclusive"
                )
            if self.independent_first_frame:
                raise ValueError(
                    "all-chunks random-x0 does not support an independent first frame"
                )
            if not 1 <= len(self.denoising_step_list) <= 4:
                raise ValueError(
                    "all-chunks random-x0 requires a one-to-four-step later-block schedule"
                )
            if (
                self.heterogeneous_chunkwise_schedule
                and len(self.first_frame_denoising_step_list) != 4
            ):
                raise ValueError(
                    "heterogeneous chunkwise random-x0 requires a four-step first-block schedule"
                )
            if not self.same_step_across_blocks:
                raise ValueError(
                    "all-chunks random-x0 requires synchronized exits; in the "
                    "heterogeneous recipe the later temporal blocks share one exit"
                )

        if self.bidirectional_full_video:
            if self.independent_first_frame:
                raise ValueError(
                    "bidirectional full-video rollout does not support an "
                    "independent first frame"
                )
            if self.num_frame_per_block != num_max_frames:
                raise ValueError(
                    "bidirectional full-video rollout requires one temporal "
                    f"block covering all {num_max_frames} latent frames, got "
                    f"num_frame_per_block={self.num_frame_per_block}"
                )
            if (
                self.first_frame_denoising_step_list is not None
                and not self.bidirectional_random_x0_4step
            ):
                raise ValueError(
                    "bidirectional full-video rollout cannot use a separate "
                    "first-frame denoising schedule"
                )
            if self.bidirectional_random_x0_4step:
                if not 1 <= len(self.denoising_step_list) <= 4:
                    raise ValueError(
                        "bidirectional random-x0 rollout requires a one-to-four-step schedule"
                    )
                if (
                    self.first_frame_denoising_step_list is None
                    or len(self.first_frame_denoising_step_list)
                    != len(self.denoising_step_list)
                    or not bool(
                        torch.equal(
                            torch.as_tensor(self.first_frame_denoising_step_list),
                            torch.as_tensor(self.denoising_step_list),
                        )
                    )
                ):
                    raise ValueError(
                        "bidirectional random-x0 rollout requires one shared full-video schedule"
                    )

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            # Generate random indices
            indices = torch.randint(
                low=0, high=num_denoising_steps, size=(num_blocks,), device=device
            )
            if self.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)

        dist.broadcast(indices, src=0)  # Broadcast the random indices to all ranks
        return indices.tolist()

    def inference_with_trajectory(
        self,
        noise: torch.Tensor,
        clean_image_or_video: torch.Tensor = None,  # same shape as noise
        initial_latent: Optional[torch.Tensor] = None,
        return_sim_step: bool = False,
        denoise_steps: Optional[int] = None,
        context_noise_replay: Optional[torch.Tensor] = None,
        return_context_noise_replay: bool = False,
        first_frame_exit_index_replay: Optional[int] = None,
        first_frame_trajectory_noise_replay: Optional[torch.Tensor] = None,
        return_first_frame_replay: bool = False,
        block_exit_indices_replay: Optional[torch.Tensor] = None,
        block_trajectory_noise_replay: Optional[torch.Tensor] = None,
        return_block_replay: bool = False,
        gradient_frame_index: Optional[int] = None,
        first_frame_skip_parameter_grad: bool = False,
        dynamic_kv_cache: bool = False,
        **conditional_dict,
    ) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = noise.shape
        if self.bidirectional_full_video:
            if initial_latent is not None:
                raise ValueError(
                    "bidirectional full-video rollout does not support initial_latent"
                )
            if num_frames != self.num_frame_per_block:
                raise ValueError(
                    "bidirectional full-video rollout requires exactly one complete "
                    f"{self.num_frame_per_block}-frame latent block, got {num_frames}"
                )
        if not self.independent_first_frame or (
            self.independent_first_frame and initial_latent is not None
        ):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = (
            num_frames + num_input_frames
        )  # add the initial latent frames
        if gradient_frame_index is not None:
            if batch_size != 1:
                raise ValueError("gradient_frame_index currently requires batch size 1")
            gradient_frame_index = int(gradient_frame_index)
            if gradient_frame_index < 0 or gradient_frame_index >= num_output_frames:
                raise ValueError(
                    f"gradient_frame_index={gradient_frame_index} is outside "
                    f"[0, {num_output_frames})"
                )
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype,
        )
        if context_noise_replay is not None and tuple(
            context_noise_replay.shape
        ) != tuple(noise.shape):
            raise ValueError(
                "context_noise_replay must match initial noise shape, got "
                f"{tuple(context_noise_replay.shape)} != {tuple(noise.shape)}"
            )
        if (
            self.bidirectional_full_video
            and context_noise_replay is not None
            and bool(context_noise_replay.count_nonzero())
        ):
            raise ValueError(
                "bidirectional full-video replay expects the unused context-noise "
                "state to be exactly zero"
            )
        captured_context_noise = (
            torch.empty_like(noise) if return_context_noise_replay else None
        )
        if self.first_frame_denoising_step_list is None:
            if first_frame_exit_index_replay not in {None, -1}:
                raise ValueError(
                    "first-frame replay state requires first_frame_denoising_step_list"
                )
            if (
                first_frame_trajectory_noise_replay is not None
                and first_frame_trajectory_noise_replay.numel() != 0
            ):
                raise ValueError(
                    "first-frame trajectory replay must be empty when the "
                    "first-frame schedule is disabled"
                )
        if (
            first_frame_skip_parameter_grad
            and self.first_frame_denoising_step_list is None
        ):
            raise ValueError(
                "first_frame_skip_parameter_grad requires first-frame denoising steps"
            )
        if return_block_replay and not self.all_chunks_random_x0_4step:
            raise ValueError(
                "block replay state is only available in all-chunks random-x0 mode"
            )
        if (
            block_exit_indices_replay is not None
            or block_trajectory_noise_replay is not None
        ) and not self.all_chunks_random_x0_4step:
            raise ValueError("block replay tensors require all-chunks random-x0 mode")

        # Release the previous step's caches before allocating new buffers.
        self.kv_cache1 = None
        self.kv_cache2 = None
        self.crossattn_cache = None
        # Keep allocator blocks warm across short prefix rollouts to avoid a
        # device synchronization for every generated video.
        if not dynamic_kv_cache:
            torch.cuda.empty_cache()

        # Step 1: Initialize KV cache to all zeros
        self._initialize_kv_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device,
            num_frames=num_output_frames if dynamic_kv_cache else None,
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = (
                torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            )
            # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
            output[:, :1] = initial_latent
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
            current_start_frame += 1

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        # In out training, self.independent_first_frame is False
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(
            len(all_num_frames), num_denoising_steps, device=noise.device
        )
        if denoise_steps is not None:
            denoise_steps = int(denoise_steps)
            if denoise_steps < 0 or denoise_steps >= num_denoising_steps:
                raise ValueError(
                    f"denoise_steps={denoise_steps} is outside [0, {num_denoising_steps})"
                )
            exit_flags = [denoise_steps for _ in exit_flags]
        elif self.all_chunks_random_x0_4step:
            exit_flags = [int(exit_flags[0])] * len(exit_flags)
        if block_exit_indices_replay is not None:
            replay_exit_indices = torch.as_tensor(
                block_exit_indices_replay,
                device=noise.device,
                dtype=torch.long,
            )
            if tuple(replay_exit_indices.shape) != (len(all_num_frames),):
                raise ValueError(
                    "block_exit_indices_replay must have shape "
                    f"[{len(all_num_frames)}], got {tuple(replay_exit_indices.shape)}"
                )
            max_replay_steps = max(
                len(self.denoising_step_list),
                (
                    0
                    if self.first_frame_denoising_step_list is None
                    else len(self.first_frame_denoising_step_list)
                ),
            )
            if bool(
                (
                    (replay_exit_indices < 0)
                    | (replay_exit_indices >= max_replay_steps)
                ).any()
            ):
                raise ValueError(
                    f"block replay exit indices must lie in [0,{max_replay_steps})"
                )
            if self.all_chunks_random_x0_4step:
                if self.heterogeneous_chunkwise_schedule:
                    if int(replay_exit_indices[0]) >= len(
                        self.first_frame_denoising_step_list
                    ):
                        raise ValueError(
                            "first-block replay exit exceeds its four-step schedule"
                        )
                    if bool(
                        (replay_exit_indices[1:] >= len(self.denoising_step_list)).any()
                    ):
                        raise ValueError(
                            "later-block replay exit exceeds its capped schedule"
                        )
                    if bool((replay_exit_indices[1:] != replay_exit_indices[1]).any()):
                        raise ValueError(
                            "heterogeneous chunkwise replay requires one exit shared by later blocks"
                        )
                elif bool((replay_exit_indices != replay_exit_indices[0]).any()):
                    raise ValueError("all-chunks replay requires one shared exit index")
            exit_flags = [int(value) for value in replay_exit_indices.tolist()]

        captured_block_trajectory_noise = None
        if self.all_chunks_random_x0_4step:
            expected_block_trajectory_shape = (
                len(all_num_frames),
                max(
                    len(self.denoising_step_list),
                    (
                        0
                        if self.first_frame_denoising_step_list is None
                        else len(self.first_frame_denoising_step_list)
                    ),
                )
                - 1,
                batch_size,
                self.num_frame_per_block,
                num_channels,
                height,
                width,
            )
            if block_trajectory_noise_replay is not None:
                if tuple(block_trajectory_noise_replay.shape) != (
                    expected_block_trajectory_shape
                ):
                    raise ValueError(
                        "block trajectory-noise replay shape "
                        f"{tuple(block_trajectory_noise_replay.shape)} != "
                        f"{expected_block_trajectory_shape}"
                    )
                for block_index, exit_index in enumerate(exit_flags):
                    unused = block_trajectory_noise_replay[
                        block_index, int(exit_index) :
                    ]
                    if unused.numel() and bool(unused.count_nonzero()):
                        raise ValueError(
                            "unused block trajectory-noise replay entries must be zero"
                        )
            if return_block_replay:
                captured_block_trajectory_noise = noise.new_zeros(
                    expected_block_trajectory_shape
                )

        # -1 plus an empty trajectory is the explicit replay contract for modes
        # without a separate first-frame denoising trajectory.
        first_frame_exit_index = -1
        captured_first_frame_trajectory_noise = []
        if self.first_frame_denoising_step_list is not None:
            first_frame_num_steps = len(self.first_frame_denoising_step_list)
            if first_frame_num_steps <= 0:
                raise ValueError("first-frame denoising schedule must not be empty")
            if (
                self.bidirectional_random_x0_4step
                and first_frame_exit_index_replay is None
            ):
                first_frame_exit_index = int(exit_flags[0])
            elif first_frame_exit_index_replay is None:
                first_frame_exit_index = self.generate_and_sync_list(
                    1, first_frame_num_steps, device=noise.device
                )[0]
            else:
                first_frame_exit_index = int(first_frame_exit_index_replay)
            if not 0 <= int(first_frame_exit_index) < first_frame_num_steps:
                raise ValueError(
                    f"first-frame exit index {first_frame_exit_index} is outside "
                    f"[0,{first_frame_num_steps})"
                )
            if self.bidirectional_random_x0_4step:
                # This is the only temporal block.  Keep denoising-range
                # reporting and replay on the same synchronized exit index.
                exit_flags = [int(first_frame_exit_index)]
            elif self.heterogeneous_chunkwise_schedule:
                if block_exit_indices_replay is not None and int(exit_flags[0]) != int(
                    first_frame_exit_index
                ):
                    raise ValueError(
                        "first-block replay exit disagrees with first-frame replay state"
                    )
                exit_flags[0] = int(first_frame_exit_index)
            if first_frame_trajectory_noise_replay is not None:
                expected_shape = (
                    int(first_frame_exit_index),
                    batch_size,
                    int(all_num_frames[0]),
                    num_channels,
                    height,
                    width,
                )
                if tuple(first_frame_trajectory_noise_replay.shape) != expected_shape:
                    raise ValueError(
                        "first-frame trajectory-noise replay shape "
                        f"{tuple(first_frame_trajectory_noise_replay.shape)} != "
                        f"{expected_shape}"
                    )

        gradient_num_frames = (
            num_output_frames
            if self.gradient_num_frames is None or self.gradient_num_frames <= 0
            else min(self.gradient_num_frames, num_output_frames)
        )
        if self.gradient_window_position == "first":
            start_gradient_frame_index = 0
            end_gradient_frame_index = gradient_num_frames
        elif self.gradient_window_position == "tail":
            start_gradient_frame_index = max(0, num_output_frames - gradient_num_frames)
            end_gradient_frame_index = num_output_frames
        else:
            raise ValueError(
                f"Unsupported gradient_window_position: {self.gradient_window_position}"
            )

        # for block_index in range(num_blocks):
        for block_index, current_num_frames in enumerate(all_num_frames):

            if True:
                noisy_input = noise[
                    :,
                    current_start_frame
                    - num_input_frames : current_start_frame
                    + current_num_frames
                    - num_input_frames,
                ]

                is_special_first_frame = (
                    block_index == 0
                    and current_start_frame == 0
                    and self.first_frame_denoising_step_list is not None
                )
                block_denoising_step_list = (
                    self.first_frame_denoising_step_list
                    if is_special_first_frame
                    else self.denoising_step_list
                )
                block_exit_index = (
                    int(first_frame_exit_index)
                    if is_special_first_frame
                    else (
                        (
                            exit_flags[1]
                            if self.heterogeneous_chunkwise_schedule
                            else exit_flags[0]
                        )
                        if self.same_step_across_blocks
                        else exit_flags[block_index]
                    )
                )

                # Step 3.1: Spatial denoising loop
                # Such a loop corresponds to the truncated denoising algorithm:
                #    T -> \tau_1 -> \tau_2 ->...-> \tau —— enable grad ——> 0
                # This cached-prefix path is intended for short denoising schedules.
                # we can inherit it for a fair comaprison. Note that as long as the conditions
                # are clean GT rather than self-generated frames, we can perform TF. So this
                # method does not conflict with TF in the frame- dimension.
                for index, current_timestep in enumerate(block_denoising_step_list):
                    exit_flag = index == block_exit_index
                    timestep = (
                        torch.ones(
                            [batch_size, current_num_frames],
                            device=noise.device,
                            dtype=torch.int64,
                        )
                        * current_timestep
                    )

                    if not exit_flag:
                        with torch.no_grad():
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame
                                * self.frame_seq_length,
                            )
                            next_timestep = block_denoising_step_list[index + 1]
                            if self.all_chunks_random_x0_4step and (
                                block_trajectory_noise_replay is not None
                            ):
                                transition_noise = block_trajectory_noise_replay[
                                    block_index, index
                                ].to(denoised_pred)
                            elif (
                                is_special_first_frame
                                and first_frame_trajectory_noise_replay is not None
                            ):
                                transition_noise = first_frame_trajectory_noise_replay[
                                    index
                                ].to(denoised_pred)
                            else:
                                transition_noise = torch.randn_like(
                                    denoised_pred.flatten(0, 1)
                                ).unflatten(0, denoised_pred.shape[:2])
                            if is_special_first_frame and return_first_frame_replay:
                                captured_first_frame_trajectory_noise.append(
                                    transition_noise.detach().clone()
                                )
                            if self.all_chunks_random_x0_4step and return_block_replay:
                                captured_block_trajectory_noise[
                                    block_index, index
                                ].copy_(transition_noise.detach())
                            noisy_input = self.scheduler.add_noise(
                                denoised_pred.flatten(0, 1),
                                transition_noise.flatten(0, 1),
                                next_timestep
                                * torch.ones(
                                    [batch_size * current_num_frames],
                                    device=noise.device,
                                    dtype=torch.long,
                                ),
                            ).unflatten(0, denoised_pred.shape[:2])
                    else:
                        if gradient_frame_index is None:
                            enable_grad = (
                                current_start_frame < end_gradient_frame_index
                                and current_start_frame + current_num_frames
                                > start_gradient_frame_index
                            )
                        else:
                            enable_grad = (
                                current_start_frame
                                <= gradient_frame_index
                                < current_start_frame + current_num_frames
                            )
                        if is_special_first_frame and first_frame_skip_parameter_grad:
                            enable_grad = False
                        if not enable_grad:
                            with torch.no_grad():
                                _, denoised_pred = self.generator(
                                    noisy_image_or_video=noisy_input,
                                    conditional_dict=conditional_dict,
                                    timestep=timestep,
                                    kv_cache=self.kv_cache1,
                                    crossattn_cache=self.crossattn_cache,
                                    current_start=current_start_frame
                                    * self.frame_seq_length,
                                )
                        else:
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame
                                * self.frame_seq_length,
                            )
                        break

            # Step 3.2: record the model's output
            output[
                :, current_start_frame : current_start_frame + current_num_frames
            ] = denoised_pred

            # Step 3.3: rerun with timestep zero to update the cache
            # A full-video bidirectional rollout has no later block that could
            # consume this cache. Its replay value is explicitly zero,
            # avoiding an otherwise redundant second 21-frame generator call.
            if self.bidirectional_full_video:
                if captured_context_noise is not None:
                    captured_context_noise.zero_()
                current_start_frame += current_num_frames
                continue
            context_timestep = torch.ones_like(timestep) * self.context_noise
            # add context noise
            replay_start = current_start_frame - num_input_frames
            replay_stop = replay_start + current_num_frames
            if context_noise_replay is None:
                # Preserve the random draw's flattened shape and layout.
                # byte-for-byte on callers that do not request replay state.
                sampled_context_noise = torch.randn_like(
                    denoised_pred.flatten(0, 1)
                ).unflatten(0, denoised_pred.shape[:2])
            else:
                sampled_context_noise = context_noise_replay[
                    :, replay_start:replay_stop
                ].to(denoised_pred)
            if captured_context_noise is not None:
                captured_context_noise[:, replay_start:replay_stop].copy_(
                    sampled_context_noise.detach()
                )
            denoised_pred = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                sampled_context_noise.flatten(0, 1),
                context_timestep
                * torch.ones(
                    [batch_size * current_num_frames],
                    device=noise.device,
                    dtype=torch.long,
                ),
            ).unflatten(0, denoised_pred.shape[:2])
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        # Step 3.5: Return the denoised timestep
        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        # T -> \tau_1 -> \tau_2 ->...-> \tau —— enable grad ——> 0
        # denoised_timestep_from = \tau
        # denoised_timestep_to = next timestep smaller than \tau
        # These are just engineering tricks
        # Align timestep sampling with the generator's denoising range.
        else:
            range_step_list = (
                self.first_frame_denoising_step_list
                if self.heterogeneous_chunkwise_schedule
                else self.denoising_step_list
            )
        if not self.same_step_across_blocks:  # handled above
            pass
        elif exit_flags[0] == len(range_step_list) - 1:
            # corner case when \tau is the smallest non-zero timestep
            denoised_timestep_to = 0
            denoised_timestep_from = (
                1000
                - torch.argmin(
                    (
                        self.scheduler.timesteps.to(noise.device)
                        - range_step_list[exit_flags[0]].to(noise.device)
                    ).abs(),
                    dim=0,
                ).item()
            )
        else:
            denoised_timestep_to = (
                1000
                - torch.argmin(
                    (
                        self.scheduler.timesteps.to(noise.device)
                        - range_step_list[exit_flags[0] + 1].to(noise.device)
                    ).abs(),
                    dim=0,
                ).item()
            )
            denoised_timestep_from = (
                1000
                - torch.argmin(
                    (
                        self.scheduler.timesteps.to(noise.device)
                        - range_step_list[exit_flags[0]].to(noise.device)
                    ).abs(),
                    dim=0,
                ).item()
            )

        first_frame_trajectory_noise = None
        if return_first_frame_replay:
            if captured_first_frame_trajectory_noise:
                first_frame_trajectory_noise = torch.stack(
                    captured_first_frame_trajectory_noise, dim=0
                )
            else:
                first_frame_trajectory_noise = noise.new_empty(
                    (
                        0,
                        batch_size,
                        int(all_num_frames[0]),
                        num_channels,
                        height,
                        width,
                    )
                )
        if return_context_noise_replay and return_first_frame_replay:
            if captured_context_noise is None:
                raise RuntimeError("context-noise capture was not initialized")
            result = (
                output,
                denoised_timestep_from,
                denoised_timestep_to,
                captured_context_noise,
                int(first_frame_exit_index),
                first_frame_trajectory_noise,
            )
            if return_block_replay:
                if captured_block_trajectory_noise is None:
                    raise RuntimeError("block trajectory capture was not initialized")
                return result + (
                    torch.tensor(exit_flags, device=noise.device, dtype=torch.long),
                    captured_block_trajectory_noise,
                )
            return result
        if return_context_noise_replay:
            if captured_context_noise is None:
                raise RuntimeError("context-noise capture was not initialized")
            return (
                output,
                denoised_timestep_from,
                denoised_timestep_to,
                captured_context_noise,
            )
        if return_sim_step:  # False
            return (
                output,
                denoised_timestep_from,
                denoised_timestep_to,
                exit_flags[0] + 1,
            )

        return output, denoised_timestep_from, denoised_timestep_to

    def _initialize_kv_cache(
        self,
        batch_size,
        dtype,
        device,
        num_frames: Optional[int] = None,
    ):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        cache_size = self.kv_cache_size
        if num_frames is not None:
            num_frames = int(num_frames)
            if num_frames <= 0:
                raise ValueError(f"num_frames must be positive, got {num_frames}")
            cache_size = num_frames * self.frame_seq_length

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append(
                {
                    "k": torch.zeros(
                        [batch_size, cache_size, 12, 128], dtype=dtype, device=device
                    ),
                    "v": torch.zeros(
                        [batch_size, cache_size, 12, 128], dtype=dtype, device=device
                    ),
                    "global_end_index": torch.tensor(
                        [0], dtype=torch.long, device=device
                    ),
                    "local_end_index": torch.tensor(
                        [0], dtype=torch.long, device=device
                    ),
                }
            )

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append(
                {
                    "k": torch.zeros(
                        [batch_size, 512, 12, 128], dtype=dtype, device=device
                    ),
                    "v": torch.zeros(
                        [batch_size, 512, 12, 128], dtype=dtype, device=device
                    ),
                    "is_init": False,
                }
            )
        self.crossattn_cache = crossattn_cache
