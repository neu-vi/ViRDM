#!/usr/bin/env python3
"""Render a locked ViRDM BS64 training recipe for one or eight GPUs."""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf


ROLLOUTS = ("chunk4", "chunk4+2", "chunk4+1", "bid4")


GLOBAL_POPULATION = 64
SUPPORTED_WORLD_SIZES = (1, 8)


def render_recipe(base: Path, rollout: str, dynamic: bool, world_size: int = 8):
    if rollout not in ROLLOUTS:
        raise ValueError(f"unsupported rollout {rollout!r}")
    if int(world_size) not in SUPPORTED_WORLD_SIZES:
        raise ValueError(
            f"world_size must be one of {SUPPORTED_WORLD_SIZES}, got {world_size}"
        )
    config = OmegaConf.load(base)
    config.resume_ckpt = ""
    config.max_steps = 20
    config.log_iters = 20
    config.virdm_dynamic_reg_enabled = bool(dynamic)
    config.virdm_dynamic_reg_weight = 5.0e-4 if dynamic else 0.0
    config.virdm_grad_accum_steps = GLOBAL_POPULATION // int(world_size)
    config.virdm_expected_world_size = int(world_size)
    config.virdm_expected_local_world_size = int(world_size)
    config.virdm_expected_global_rows = GLOBAL_POPULATION

    if rollout.startswith("chunk"):
        later_steps = {"chunk4": 4, "chunk4+2": 2, "chunk4+1": 1}[rollout]
        config.virdm_generation_mode = "chunkwise"
        config.num_frame_per_block = 3
        config.denoising_step_list = [1000, 750, 500, 250][:later_steps]
        first_chunk_enhancement = rollout != "chunk4"
        config.virdm_first_frame_denoising_step_list = (
            [1000, 750, 500, 250] if first_chunk_enhancement else []
        )
        config.virdm_all_chunks_random_x0_4step = True
        config.virdm_bidirectional_random_x0_4step = False
        config.virdm_replay_first_frame_trajectory_noise = first_chunk_enhancement
        config.virdm_log_rollout_force_all_chunks_full = True
    else:
        config.virdm_generation_mode = "full_video_bidirectional"
        config.num_frame_per_block = 21
        config.denoising_step_list = [1000, 750, 500, 250]
        config.virdm_first_frame_denoising_step_list = []
        config.virdm_all_chunks_random_x0_4step = False
        config.virdm_bidirectional_random_x0_4step = True
        config.virdm_replay_first_frame_trajectory_noise = True
        config.virdm_log_rollout_force_all_chunks_full = False
    return config


def validate(config, rollout: str, dynamic: bool, world_size: int = 8) -> None:
    assert config.trainer == "virdm"
    assert config.distribution_loss == "virdm"
    assert float(config.virdm_joint_bandwidth_scale) == 1.0
    assert int(config.batch_size) == 1
    assert int(world_size) in SUPPORTED_WORLD_SIZES
    assert int(config.virdm_grad_accum_steps) == GLOBAL_POPULATION // int(world_size)
    assert int(config.virdm_expected_world_size) == int(world_size)
    assert int(config.virdm_expected_local_world_size) == int(world_size)
    assert int(config.virdm_expected_global_rows) == GLOBAL_POPULATION
    assert (
        int(config.batch_size)
        * int(config.virdm_grad_accum_steps)
        * int(config.virdm_expected_world_size)
        == GLOBAL_POPULATION
    )
    assert float(config.lr) == 2.0e-6
    assert config.virdm_attention_backend == "flash_attn_2"
    assert config.virdm_flash_attn_version == "2.8.3.post1"
    assert bool(config.virdm_require_flash_attention)
    assert int(config.max_steps) == int(config.log_iters) == 20
    assert bool(config.virdm_dynamic_reg_enabled) is dynamic
    assert float(config.virdm_dynamic_reg_weight) == (5.0e-4 if dynamic else 0.0)
    if rollout == "bid4":
        assert config.virdm_generation_mode == "full_video_bidirectional"
        assert int(config.num_frame_per_block) == 21
        assert list(config.denoising_step_list) == [1000, 750, 500, 250]
        assert bool(config.virdm_bidirectional_random_x0_4step)
    else:
        expected = {"chunk4": 4, "chunk4+2": 2, "chunk4+1": 1}[rollout]
        assert config.virdm_generation_mode == "chunkwise"
        assert int(config.num_frame_per_block) == 3
        assert len(config.denoising_step_list) == expected
        assert bool(config.virdm_all_chunks_random_x0_4step)
        first_chunk_enhancement = rollout != "chunk4"
        assert list(config.virdm_first_frame_denoising_step_list) == (
            [1000, 750, 500, 250] if first_chunk_enhancement else []
        )
        assert bool(config.virdm_replay_first_frame_trajectory_noise) is (
            first_chunk_enhancement
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--rollout", choices=ROLLOUTS, required=True)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument(
        "--world-size", type=int, choices=SUPPORTED_WORLD_SIZES, default=8
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = render_recipe(args.base, args.rollout, args.dynamic, args.world_size)
    validate(config, args.rollout, args.dynamic, args.world_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, args.output)
    print(
        f"rendered rollout={args.rollout} dynamic={args.dynamic} "
        f"world_size={args.world_size} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
