#!/usr/bin/env python3
"""Create a publication-safe ViRDM reference without changing its numerics."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch


REFERENCE_FORMAT = "virdm_joint_reference_v1"
REFERENCE_REVISION = "virdm_joint_reference_v1"
SOURCE_DATASET_REPO = "zhuhz22/Causal-Forcing-data"
ENCODER_CHECKPOINT_FILENAME = "vjepa2_1_vitl_dist_vitG_384.pt"
FORBIDDEN_METADATA_FRAGMENTS = (
    "/mnt/",
    "/home/",
    "one_forcing",
    "one-forcing",
)


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _strings(key)
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def sanitize_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    output = dict(bundle)
    metadata = dict(output.get("metadata", {}))
    metadata.pop("source_lmdb", None)
    metadata.pop("encoder_checkpoint", None)
    metadata.pop("rdm_source_revision", None)
    metadata.update(
        {
            "format": REFERENCE_FORMAT,
            "virdm_reference_revision": REFERENCE_REVISION,
            "source_dataset_repo_id": SOURCE_DATASET_REPO,
            "encoder_checkpoint_filename": ENCODER_CHECKPOINT_FILENAME,
        }
    )
    offenders = [
        value
        for value in _strings(metadata)
        if any(fragment in value.lower() for fragment in FORBIDDEN_METADATA_FRAGMENTS)
    ]
    if offenders:
        raise ValueError(f"reference metadata still contains private paths: {offenders}")
    output["metadata"] = metadata
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.input.expanduser().resolve()
    destination = args.output.expanduser().resolve()
    if source == destination:
        raise SystemExit("--input and --output must differ")
    if destination.exists():
        raise SystemExit(f"refusing to replace existing output: {destination}")
    bundle = torch.load(source, map_location="cpu", weights_only=False)
    sanitized = sanitize_bundle(bundle)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        torch.save(sanitized, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(destination)


if __name__ == "__main__":
    main()
