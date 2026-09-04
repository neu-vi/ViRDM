#!/usr/bin/env python3
"""Download the pinned runtime artifacts used by ViRDM.

Official upstream files are pinned in ``artifacts/manifest.json``.  The small
ViRDM-specific training bundle (prompt-only LMDB, fixed reference, and frozen
text table) is downloaded from the repository and immutable revision pinned in
the manifest. Inference scope includes the two released causal four-step
generators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "artifacts" / "manifest.json"


def file_sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified(path: Path, expected_sha256: str | None = None) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    return not expected_sha256 or file_sha256(path) == expected_sha256


def atomic_download(url: str, destination: Path, expected_sha256: str) -> None:
    if verified(destination, expected_sha256):
        print(f"ready: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.download.{os.getpid()}")
    print(f"downloading: {url}")
    try:
        with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=8 << 20)
        actual = file_sha256(temporary)
        if actual != expected_sha256:
            raise RuntimeError(
                f"checksum mismatch for {destination}: expected {expected_sha256}, got {actual}"
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"ready: {destination}")


def download_hf_file(
    *,
    repo_id: str,
    filename: str,
    revision: str,
    destination: Path,
    expected_sha256: str = "",
    repo_type: str = "model",
) -> None:
    from huggingface_hub import hf_hub_download

    destination.parent.mkdir(parents=True, exist_ok=True)
    if verified(destination, expected_sha256 or None):
        print(f"ready: {destination}")
        return
    force_download = destination.exists() and not verified(
        destination, expected_sha256 or None
    )
    cached = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            repo_type=repo_type,
            force_download=force_download,
        )
    )
    temporary = destination.with_name(f".{destination.name}.download.{os.getpid()}")
    try:
        shutil.copyfile(cached, temporary)
        if expected_sha256:
            actual = file_sha256(temporary)
            if actual != expected_sha256:
                raise RuntimeError(
                    f"checksum mismatch for {destination}: expected "
                    f"{expected_sha256}, got {actual}"
                )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"ready: {destination}")


def download_hf_snapshot(
    *,
    repo_id: str,
    revision: str,
    destination: Path,
    file_sha256: dict[str, str],
) -> None:
    from huggingface_hub import snapshot_download

    required = tuple(file_sha256)
    if all(
        verified(destination / relative, file_sha256[relative])
        for relative in required
    ):
        print(f"ready: {destination}")
        return
    destination.mkdir(parents=True, exist_ok=True)
    print(f"downloading Hugging Face snapshot: {repo_id}@{revision}")
    force_download = any(
        (destination / relative).exists()
        and not verified(destination / relative, file_sha256[relative])
        for relative in required
    )
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=destination,
        allow_patterns=list(required),
        force_download=force_download,
    )
    missing = [relative for relative in required if not (destination / relative).is_file()]
    if missing:
        raise RuntimeError(f"incomplete Wan snapshot; missing: {missing}")
    mismatched = [
        relative
        for relative in required
        if not verified(destination / relative, file_sha256[relative])
    ]
    if mismatched:
        raise RuntimeError(f"Wan snapshot checksum mismatch: {mismatched}")
    print(f"ready: {destination}")


def download_archive_member(
    *, url: str, member: str, destination: Path, expected_sha256: str
) -> None:
    if verified(destination, expected_sha256):
        print(f"ready: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="virdm-download-") as directory:
        archive = Path(directory) / "archive.zip"
        print(f"downloading: {url}")
        urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as handle:
            names = set(handle.namelist())
            if member not in names:
                raise RuntimeError(f"archive does not contain {member!r}")
            temporary = destination.with_name(
                f".{destination.name}.download.{os.getpid()}"
            )
            try:
                with handle.open(member) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=8 << 20)
                actual = file_sha256(temporary)
                if actual != expected_sha256:
                    raise RuntimeError(
                        f"checksum mismatch for {destination}: expected {expected_sha256}, got {actual}"
                    )
                os.replace(temporary, destination)
            finally:
                if temporary.exists():
                    temporary.unlink()
    print(f"ready: {destination}")


def download_virdm_files(
    *,
    files: dict,
    repo_id: str,
    repo_type: str,
    revision: str,
    root: Path,
) -> None:
    for filename, item in files.items():
        download_hf_file(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            destination=root / item["target"],
            expected_sha256=item.get("sha256", ""),
            repo_type=repo_type,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "artifacts")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--scope",
        choices=("inference", "training", "reference", "all"),
        default="inference",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Also download the frozen dense-flow checkpoint.",
    )
    parser.add_argument(
        "--virdm-repo-id",
        default=os.environ.get("VIRDM_HF_REPO", ""),
        help="Override the manifest's ViRDM training-asset repository.",
    )
    parser.add_argument(
        "--virdm-revision",
        default=os.environ.get("VIRDM_HF_REVISION", ""),
        help="Override the manifest's immutable training-asset revision.",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("format") != "virdm_artifacts_v1":
        raise SystemExit(f"unsupported artifact manifest: {args.manifest}")
    root = args.root.expanduser().resolve()
    official = manifest["official"]
    training_assets = manifest["virdm_training_assets"]
    virdm_repo_id = args.virdm_repo_id or str(training_assets.get("repo_id", ""))
    virdm_repo_type = str(training_assets.get("repo_type", "model"))
    virdm_revision = args.virdm_revision or str(training_assets.get("revision", ""))

    wan = official["wan_2_1_t2v_1_3b"]
    if args.scope == "reference":
        filename = "Wan2.1_VAE.pth"
        download_hf_file(
            repo_id=wan["repo_id"],
            filename=filename,
            revision=wan["revision"],
            destination=root / wan["target"] / filename,
            expected_sha256=wan["file_sha256"][filename],
        )
    else:
        download_hf_snapshot(
            repo_id=wan["repo_id"],
            revision=wan["revision"],
            destination=root / wan["target"],
            file_sha256=wan["file_sha256"],
        )

    if args.scope in {"inference", "training", "all"}:
        # The public Causal-ODE model initializes all released ViRDM recipes.
        ode = official["causal_ode_chunkwise"]
        download_hf_file(
            repo_id=ode["repo_id"],
            filename=ode["filename"],
            revision=ode["revision"],
            destination=root / ode["target"],
            expected_sha256=ode["sha256"],
        )

    if args.scope in {"training", "reference", "all"}:
        names = ["vjepa_2_1_vitl_384"]
        if args.scope in {"training", "all"}:
            names.append("taew_2_1")
        for name in names:
            item = official[name]
            atomic_download(item["url"], root / item["target"], item["sha256"])

    if args.scope in {"reference", "all"}:
        item = official["siglip_2_so400m_256"]
        download_hf_snapshot(
            repo_id=item["repo_id"],
            revision=item["revision"],
            destination=root / item["target"],
            file_sha256=item["file_sha256"],
        )

    if args.scope in {"training", "all"}:
        if not virdm_repo_id or not virdm_revision:
            raise SystemExit(
                "training requires a pinned ViRDM prompt/reference bundle; "
                "set repo_id and revision in artifacts/manifest.json"
            )
        download_virdm_files(
            files=training_assets["files"],
            repo_id=virdm_repo_id,
            repo_type=virdm_repo_type,
            revision=virdm_revision,
            root=root,
        )

    if args.scope in {"inference", "all"}:
        checkpoints = manifest["virdm_checkpoints"]
        download_virdm_files(
            files=checkpoints["files"],
            repo_id=str(checkpoints["repo_id"]),
            repo_type=str(checkpoints.get("repo_type", "model")),
            revision=str(checkpoints["revision"]),
            root=root,
        )

    if args.dynamic:
        flow = official["dense_flow"]
        download_archive_member(
            url=flow["archive_url"],
            member=flow["archive_member"],
            destination=root / flow["target"],
            expected_sha256=flow["sha256"],
        )

    print(f"artifact download complete: {root}")


if __name__ == "__main__":
    main()
