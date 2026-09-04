#!/usr/bin/env python3
"""Build ViRDM's compact prompt-only LMDB from official causal-video data."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import lmdb

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.lmdb_ import get_array_shape_from_lmdb
from virdm_integration.joint_text import prompt_rows_sha256


DEFAULT_MANIFEST = REPO_ROOT / "artifacts" / "manifest.json"


def open_readonly(path: Path):
    return lmdb.open(
        str(path), readonly=True, lock=False, readahead=False, meminit=False
    )


def discover_sources(source: Path) -> list[Path]:
    source = source.expanduser().resolve()
    if (source / "data.mdb").is_file():
        return [source]
    if (source / "clean_data" / "data.mdb").is_file():
        return [source / "clean_data"]
    shards = [source / f"ODE6KCausal_chunkwise_{index}" for index in range(15)]
    missing = [str(path) for path in shards if not (path / "data.mdb").is_file()]
    if missing:
        raise SystemExit(
            f"{source} is neither a merged LMDB nor a complete 15-shard "
            f"Causal-Forcing-data download; missing {len(missing)} shard(s)"
        )
    return shards


def read_prompts(sources: list[Path]) -> list[str]:
    rows: list[str] = []
    for source in sources:
        env = open_readonly(source)
        try:
            shape = get_array_shape_from_lmdb(env, "prompts")
            count = int(shape[0])
            with env.begin() as txn:
                for index in range(count):
                    key = f"prompts_{index}_data".encode()
                    value = txn.get(key)
                    if value is None:
                        raise SystemExit(f"missing {key!r} in {source}")
                    rows.append(value.decode("utf-8"))
        finally:
            env.close()
    return rows


def inspect_existing(destination: Path) -> tuple[int, str] | None:
    if not (destination / "data.mdb").is_file():
        return None
    env = open_readonly(destination)
    try:
        count = int(get_array_shape_from_lmdb(env, "prompts")[0])
        with env.begin() as txn:
            rows = []
            for index in range(count):
                value = txn.get(f"prompts_{index}_data".encode())
                if value is None:
                    return None
                rows.append(value.decode("utf-8"))
    finally:
        env.close()
    return count, prompt_rows_sha256(rows)


def write_lmdb(rows: list[str], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp.", dir=destination.parent)
    )
    try:
        payload_size = sum(len(row.encode("utf-8")) for row in rows)
        map_size = max(64 << 20, payload_size * 4)
        env = lmdb.open(str(temporary), map_size=map_size, subdir=True)
        try:
            with env.begin(write=True) as txn:
                txn.put(b"prompts_shape", str(len(rows)).encode())
                for index, row in enumerate(rows):
                    txn.put(
                        f"prompts_{index}_data".encode(), row.encode("utf-8")
                    )
            env.sync()
        finally:
            env.close()
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Merged clean_data LMDB or root of the 15 official chunkwise shards.",
    )
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "artifacts" / "prompt_data"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    contract = manifest["virdm_training_assets"]
    expected_rows = int(contract["expected_prompt_rows"])
    expected_sha = str(contract["prompt_rows_sha256"])
    destination = args.output.expanduser().resolve()

    existing = inspect_existing(destination)
    if existing == (expected_rows, expected_sha):
        print(f"ready: {destination} ({expected_rows} verified prompts)")
        return
    if destination.exists():
        raise SystemExit(
            f"refusing to replace non-matching output {destination}; move it aside first"
        )

    sources = discover_sources(args.source)
    rows = read_prompts(sources)
    actual_sha = prompt_rows_sha256(rows)
    if len(rows) != expected_rows or actual_sha != expected_sha:
        raise SystemExit(
            f"source prompt contract mismatch: rows={len(rows)} sha256={actual_sha}; "
            f"expected rows={expected_rows} sha256={expected_sha}"
        )
    write_lmdb(rows, destination)
    if inspect_existing(destination) != (expected_rows, expected_sha):
        raise SystemExit("written prompt LMDB failed its verification pass")
    print(f"wrote: {destination} ({expected_rows} verified prompts)")


if __name__ == "__main__":
    main()
