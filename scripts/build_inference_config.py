#!/usr/bin/env python3
"""Build a locked ViRDM inference config for one released sampling recipe."""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf


RECIPES = {
    "causal4": {"block": 3, "later_steps": 4, "first_steps": 4},
    "causal2": {"block": 3, "later_steps": 2, "first_steps": 4},
    "causal1": {"block": 3, "later_steps": 1, "first_steps": 4},
    "bid4": {"block": 21, "later_steps": 4, "first_steps": 0},
}
BASE_TIMESTEPS = [1000, 750, 500, 250]


def render_inference_config(base: Path, recipe: str, num_output_frames: int):
    if recipe not in RECIPES:
        raise ValueError(f"unsupported inference recipe {recipe!r}")
    if num_output_frames <= 0:
        raise ValueError("num_output_frames must be positive")
    spec = RECIPES[recipe]
    if num_output_frames % spec["block"] != 0:
        raise ValueError(
            f"{recipe} requires num_output_frames divisible by {spec['block']}"
        )
    config = OmegaConf.load(base)
    config.num_frame_per_block = spec["block"]
    config.warp_denoising_step = True
    config.denoising_step_list = BASE_TIMESTEPS[: spec["later_steps"]]
    config.first_frame_denoising_step_list = (
        BASE_TIMESTEPS[: spec["first_steps"]] if spec["first_steps"] else []
    )
    return config


def validate_inference_config(config, recipe: str, num_output_frames: int) -> None:
    spec = RECIPES[recipe]
    assert int(config.num_frame_per_block) == spec["block"]
    assert list(config.denoising_step_list) == BASE_TIMESTEPS[: spec["later_steps"]]
    expected_first = BASE_TIMESTEPS[: spec["first_steps"]]
    assert list(config.first_frame_denoising_step_list) == expected_first
    if recipe == "bid4":
        assert num_output_frames == 21
    else:
        assert num_output_frames % 3 == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--recipe", choices=tuple(RECIPES), required=True)
    parser.add_argument("--num-output-frames", type=int, default=21)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = render_inference_config(args.base, args.recipe, args.num_output_frames)
    validate_inference_config(config, args.recipe, args.num_output_frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
