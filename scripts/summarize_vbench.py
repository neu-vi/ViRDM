#!/usr/bin/env python3
"""Summarize VBench raw dimensions with the official leaderboard formula."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


NORMALIZATION = {
    "subject_consistency": (0.1462, 1.0),
    "background_consistency": (0.2615, 1.0),
    "temporal_flickering": (0.6293, 1.0),
    "motion_smoothness": (0.706, 0.9975),
    "dynamic_degree": (0.0, 1.0),
    "aesthetic_quality": (0.0, 1.0),
    "imaging_quality": (0.0, 1.0),
    "object_class": (0.0, 1.0),
    "multiple_objects": (0.0, 1.0),
    "human_action": (0.0, 1.0),
    "color": (0.0, 1.0),
    "spatial_relationship": (0.0, 1.0),
    "scene": (0.0, 0.8222),
    "appearance_style": (0.0009, 0.2855),
    "temporal_style": (0.0, 0.364),
    "overall_consistency": (0.0, 0.364),
}
DIMENSION_WEIGHT = {name: 1.0 for name in NORMALIZATION}
DIMENSION_WEIGHT["dynamic_degree"] = 0.5
QUALITY = [
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
    "dynamic_degree",
]
SEMANTIC = [
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "appearance_style",
    "temporal_style",
    "overall_consistency",
]


def find_result(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = sorted(path.glob("*_eval_results.json"))
    if len(candidates) != 1:
        raise SystemExit(
            f"expected exactly one *_eval_results.json under {path}, found {len(candidates)}"
        )
    return candidates[0]


def aggregate(raw: dict[str, float]) -> dict:
    normalized = {}
    for name, value in raw.items():
        if name not in NORMALIZATION:
            continue
        minimum, maximum = NORMALIZATION[name]
        normalized[name] = (
            (float(value) - minimum) / (maximum - minimum)
        ) * DIMENSION_WEIGHT[name]

    def group_score(names: list[str]) -> float | None:
        if any(name not in normalized for name in names):
            return None
        return sum(normalized[name] for name in names) / sum(
            DIMENSION_WEIGHT[name] for name in names
        )

    quality = group_score(QUALITY)
    semantic = group_score(SEMANTIC)
    total = None if quality is None or semantic is None else (4 * quality + semantic) / 5
    return {
        "total": total,
        "quality": quality,
        "semantic": semantic,
        "dynamic_degree": raw.get("dynamic_degree"),
        "raw_dimensions": raw,
        "missing_dimensions": sorted(set(NORMALIZATION) - set(raw)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result_path = find_result(args.input)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    raw = {
        str(name): float(value[0] if isinstance(value, list) else value)
        for name, value in payload.items()
    }
    summary = aggregate(raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
