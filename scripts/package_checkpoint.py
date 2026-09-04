#!/usr/bin/env python3
"""Convert a training checkpoint into a portable ViRDM generator file."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import os
from pathlib import Path

import torch


def file_sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_generator_state(
    state: dict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    normalized: OrderedDict[str, torch.Tensor] = OrderedDict()
    prefix = "model._fsdp_wrapped_module."
    for original_name, value in state.items():
        if not isinstance(original_name, str) or not isinstance(value, torch.Tensor):
            raise TypeError("generator state must map string names to tensors")
        name = (
            original_name.replace(prefix, "model.", 1)
            if original_name.startswith(prefix)
            else original_name
        )
        if name in normalized:
            raise ValueError(f"duplicate parameter after prefix normalization: {name}")
        normalized[name] = value.contiguous() if not value.is_contiguous() else value
    if not normalized:
        raise ValueError("generator state is empty")
    return normalized


def load_training_generator(
    path: Path, use_ema: bool
) -> OrderedDict[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("training checkpoint must be a dictionary")
    state_key = "generator_ema" if use_ema else "generator"
    if state_key not in payload:
        raise KeyError(f"training checkpoint has no {state_key}")
    return normalize_generator_state(payload[state_key])


def package_checkpoint(
    *,
    source: Path,
    destination: Path,
    use_ema: bool = False,
) -> str:
    if destination.suffix != ".pt":
        raise ValueError("released generator checkpoint must use the .pt suffix")
    state = load_training_generator(source, use_ema)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        torch.save({"generator": state}, temporary)
        packaged = torch.load(
            temporary, map_location="cpu", weights_only=True, mmap=True
        )
        if list(packaged) != ["generator"]:
            raise RuntimeError("released checkpoint must contain only generator")
        packaged_state = packaged["generator"]
        if set(packaged_state) != set(state):
            raise RuntimeError("packaged checkpoint key verification failed")
        for name, expected in state.items():
            if not torch.equal(packaged_state[name], expected):
                raise RuntimeError(f"packaged tensor differs from source: {name}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return file_sha256(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--use-ema", action="store_true")
    args = parser.parse_args()
    digest = package_checkpoint(
        source=args.input,
        destination=args.output,
        use_ema=args.use_ema,
    )
    print(f"wrote {args.output}")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
