#!/usr/bin/env python3
"""Assemble the complete Hugging Face model and training-asset directory."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from check_artifacts import require_file, require_prompt_lmdb


DEFAULT_MANIFEST = REPO_ROOT / "artifacts" / "manifest.json"

ASSET_CARD = """---
license: other
---

# ViRDM models and training assets

This repository contains the trained causal four-step ViRDM generators and the
compact frozen artifacts required to reproduce training.

- `checkpoints/virdm_causal4_dynamic_step20.pt`: the primary
  20-update model with dynamics regularization (`5e-4`).
- `checkpoints/virdm_causal4_nodynamic_step20.pt`: the matched model
  without dynamics regularization.

- `prompt_data/data.mdb`: 6,505 captions in the original ordered
  `zhuhz22/Causal-Forcing-data` training split; no video or latent payloads.
- `references/reference_M4096.pt`: the fixed 4,096-landmark joint video-text
  reference used by ViRDM. Its publication metadata contains stable upstream
  identifiers rather than machine-local paths.
- `references/siglip2_text_fp32.npy`: frozen SigLIP2 features aligned row for
  row with the reference and prompt table.
- `asset_receipt.json`: row contract, filenames, and SHA-256 checksums.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-lmdb", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--text-table", type=Path, required=True)
    parser.add_argument("--dynamic-checkpoint", type=Path, required=True)
    parser.add_argument("--nodynamic-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    contract = manifest["virdm_training_assets"]
    require_prompt_lmdb(
        str(args.prompt_lmdb),
        expected_rows=int(contract["expected_prompt_rows"]),
        expected_rows_sha256=str(contract["prompt_rows_sha256"]),
    )
    files = contract["files"]
    checkpoint_files = manifest["virdm_checkpoints"]["files"]
    require_file(
        str(args.reference),
        "ViRDM reference",
        files["references/reference_M4096.pt"]["sha256"],
    )
    require_file(
        str(args.text_table),
        "frozen text table",
        files["references/siglip2_text_fp32.npy"]["sha256"],
    )
    require_file(
        str(args.dynamic_checkpoint),
        "dynamic ViRDM checkpoint",
        checkpoint_files[
            "checkpoints/virdm_causal4_dynamic_step20.pt"
        ]["sha256"],
    )
    require_file(
        str(args.nodynamic_checkpoint),
        "ViRDM checkpoint without dynamics regularization",
        checkpoint_files[
            "checkpoints/virdm_causal4_nodynamic_step20.pt"
        ]["sha256"],
    )

    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to replace existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp.", dir=output.parent)
    )
    try:
        destinations = {
            "checkpoints/virdm_causal4_dynamic_step20.pt": (
                args.dynamic_checkpoint
            ),
            "checkpoints/virdm_causal4_nodynamic_step20.pt": (
                args.nodynamic_checkpoint
            ),
            "prompt_data/data.mdb": args.prompt_lmdb / "data.mdb",
            "references/reference_M4096.pt": args.reference,
            "references/siglip2_text_fp32.npy": args.text_table,
        }
        for relative, source in destinations.items():
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        receipt = {
            "format": "virdm_release_assets_v1",
            "prompt_rows": contract["expected_prompt_rows"],
            "prompt_rows_sha256": contract["prompt_rows_sha256"],
            "files": {**checkpoint_files, **files},
        }
        (temporary / "asset_receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
        (temporary / "README.md").write_text(ASSET_CARD, encoding="utf-8")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(f"release asset bundle ready: {output}")


if __name__ == "__main__":
    main()
