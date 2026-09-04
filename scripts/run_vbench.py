#!/usr/bin/env python3
"""Run the 16 official VBench dimensions on a standard-named video folder."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


DEFAULT_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "temporal_style",
    "appearance_style",
    "overall_consistency",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-path", type=Path, required=True)
    parser.add_argument("--full-info", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dimensions", nargs="*", default=DEFAULT_DIMENSIONS)
    parser.add_argument(
        "--local-models",
        action="store_true",
        help="Use/download VBench assets under VBENCH_CACHE_DIR.",
    )
    args = parser.parse_args()

    try:
        from vbench import VBench
        from vbench.distributed import barrier, dist_init, get_rank
    except ImportError as exc:
        raise SystemExit(
            "VBench is not installed. Follow README.md > Evaluation to create "
            "the separate evaluation environment."
        ) from exc

    if not args.videos_path.is_dir():
        raise SystemExit(f"video directory not found: {args.videos_path}")
    if not args.full_info.is_file():
        raise SystemExit(f"VBench metadata not found: {args.full_info}")
    if not args.dimensions:
        raise SystemExit("at least one VBench dimension is required")

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed and not torch.distributed.is_initialized():
        dist_init()
    rank = get_rank()
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        barrier()

    evaluator = VBench(
        torch.device(args.device), str(args.full_info), str(args.output_dir)
    )
    evaluator.evaluate(
        videos_path=str(args.videos_path),
        name=args.name,
        dimension_list=args.dimensions,
        local=args.local_models,
        mode="vbench_standard",
    )

    if distributed:
        barrier()


if __name__ == "__main__":
    main()
