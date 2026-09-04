#!/usr/bin/env python3
"""Fail fast when a configured runtime artifact is missing or corrupted."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import lmdb
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.lmdb_ import get_array_shape_from_lmdb, retrieve_row_from_lmdb
from virdm_integration.joint_text import prompt_rows_sha256


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: str, label: str, expected_sha: str = "") -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise SystemExit(f"missing {label}: {resolved}")
    if expected_sha:
        actual = sha256(resolved)
        if actual != expected_sha:
            raise SystemExit(
                f"{label} checksum mismatch: expected {expected_sha}, got {actual}"
            )
    return resolved


def require_prompt_lmdb(
    directory: str, *, expected_rows: int, expected_rows_sha256: str
) -> Path:
    root = Path(directory).expanduser().resolve()
    data = require_file(str(root / "data.mdb"), "prompt LMDB")
    env = lmdb.open(
        str(root), readonly=True, lock=False, readahead=False, meminit=False
    )
    try:
        shape = get_array_shape_from_lmdb(env, "prompts")
        if not shape or int(shape[0]) != int(expected_rows):
            raise SystemExit(
                f"prompt LMDB rows={shape[0] if shape else None}; "
                f"expected {expected_rows}"
            )
        prompts = [
            retrieve_row_from_lmdb(env, "prompts", str, index)
            for index in range(int(expected_rows))
        ]
    finally:
        env.close()
    actual = prompt_rows_sha256(prompts)
    if actual != expected_rows_sha256:
        raise SystemExit(
            f"prompt LMDB row checksum mismatch: expected "
            f"{expected_rows_sha256}, got {actual}"
        )
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--scope", choices=("inference", "training"), default="training"
    )
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    manifest_path = REPO_ROOT / "artifacts" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    required = []
    if args.scope == "training":
        required.extend(
            [
                (config.generator_ckpt, "generator initialization", ""),
                (
                    config.virdm_reference_bundle_path,
                    "ViRDM reference",
                    config.virdm_reference_bundle_sha256,
                ),
                (
                    config.virdm_joint_text_table_path,
                    "text feature table",
                    config.virdm_joint_text_table_sha256,
                ),
                (
                    config.virdm_encoder_checkpoint_path,
                    "V-JEPA checkpoint",
                    config.virdm_encoder_checkpoint_sha256,
                ),
                (
                    config.virdm_taew_checkpoint_path,
                    "TAEW checkpoint",
                    config.virdm_taew_checkpoint_sha256,
                ),
            ]
        )
        required[0] = (
            config.generator_ckpt,
            "generator initialization",
            config.generator_ckpt_sha256,
        )
    if args.checkpoint is not None:
        checkpoint_expected_sha = ""
        resolved_checkpoint = Path(args.checkpoint).expanduser().resolve()
        if resolved_checkpoint == Path(
            str(config.generator_ckpt)
        ).expanduser().resolve():
            checkpoint_expected_sha = str(config.generator_ckpt_sha256)
        artifact_root = Path(str(config.artifact_root)).expanduser().resolve()
        for item in manifest.get("virdm_checkpoints", {}).get("files", {}).values():
            if resolved_checkpoint == (artifact_root / item["target"]).resolve():
                checkpoint_expected_sha = str(item["sha256"])
                break
        required.append(
            (args.checkpoint, "generator checkpoint", checkpoint_expected_sha)
        )
    for path, label, expected_sha in required:
        require_file(str(path), label, str(expected_sha))

    if args.scope == "training":
        require_prompt_lmdb(
            str(config.data_path),
            expected_rows=int(config.virdm_expected_reference_rows),
            expected_rows_sha256=str(config.virdm_joint_prompt_rows_sha256),
        )

    wan_root = Path(
        os.environ.get(
            "VIRDM_WAN_MODEL_ROOT",
            str(Path(config.artifact_root) / "wan" / "Wan2.1-T2V-1.3B"),
        )
    )
    wan_hashes = manifest["official"]["wan_2_1_t2v_1_3b"]["file_sha256"]
    for relative, expected_sha in wan_hashes.items():
        require_file(
            str(wan_root / relative), f"Wan artifact {relative}", expected_sha
        )

    if args.scope == "training" and args.dynamic:
        require_file(
            str(config.virdm_dynamic_reg_checkpoint_path),
            "dynamic-regularizer checkpoint",
            str(config.virdm_dynamic_reg_checkpoint_sha256),
        )
    print("artifact preflight passed")


if __name__ == "__main__":
    main()
