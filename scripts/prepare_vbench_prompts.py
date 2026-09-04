#!/usr/bin/env python3
"""Extract the unique official prompt order from VBench_full_info.json."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def rows_sha256(rows: list[str]) -> str:
    payload = "".join(f"{row}\n" for row in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def unique_rows(rows: list[str]) -> tuple[list[str], list[int]]:
    output: list[str] = []
    indices: list[int] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if row not in seen:
            seen.add(row)
            output.append(row)
            indices.append(index)
    return output, indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-info", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--extended-source",
        type=Path,
        help="Optional line-aligned extended prompts (946 or 944 rows).",
    )
    parser.add_argument("--extended-output", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()

    records = json.loads(args.full_info.read_text(encoding="utf-8"))
    source_rows = [str(record["prompt_en"]).strip() for record in records]
    if any(not row for row in source_rows):
        raise SystemExit("VBench metadata contains an empty prompt")
    prompts, kept_indices = unique_rows(source_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(prompts) + "\n", encoding="utf-8")

    extended_sha = None
    if args.extended_source is not None:
        if args.extended_output is None:
            raise SystemExit("--extended-output is required with --extended-source")
        extended_source = args.extended_source.read_text(encoding="utf-8").splitlines()
        if len(extended_source) == len(source_rows):
            extended = [extended_source[index] for index in kept_indices]
        elif len(extended_source) == len(prompts):
            extended = extended_source
        else:
            raise SystemExit(
                f"extended prompt rows={len(extended_source)}; expected "
                f"{len(source_rows)} or {len(prompts)}"
            )
        if any(not row.strip() for row in extended):
            raise SystemExit("extended prompt file contains an empty row")
        args.extended_output.parent.mkdir(parents=True, exist_ok=True)
        args.extended_output.write_text("\n".join(extended) + "\n", encoding="utf-8")
        extended_sha = rows_sha256(extended)

    receipt = {
        "format": "virdm_vbench_prompt_receipt_v1",
        "full_info": str(args.full_info.resolve()),
        "metadata_rows": len(source_rows),
        "unique_prompts": len(prompts),
        "duplicate_rows": len(source_rows) - len(prompts),
        "prompt_rows_sha256": rows_sha256(prompts),
        "extended_prompt_rows_sha256": extended_sha,
    }
    receipt_path = args.receipt or args.output.with_suffix(".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {len(prompts)} unique prompts from {len(source_rows)} metadata rows "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
