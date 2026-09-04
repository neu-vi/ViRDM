"""Frozen ViRDM reference loading and Nyström loss assembly."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.utils.checkpoint


def file_sha256(path: str | Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ViRDMReference:
    Z: torch.Tensor
    Z2: torch.Tensor
    alpha: torch.Tensor
    sigma: float
    k_rr: float
    metadata: Mapping[str, Any]
    sha256: str
    beta: float | None = None

    @property
    def M(self) -> int:
        return int(self.Z.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.Z.shape[1])


def load_reference(
    path: str | Path,
    *,
    device: torch.device | str,
    expected_encoder_id: str,
    expected_checkpoint_sha256: str,
    expected_height: int,
    expected_width: int,
    expected_rows: int,
    expected_reference_revision: str,
    expected_encoder_contract: Mapping[str, Any],
    expected_landmarks: int = 4096,
    expected_feature_dim: int = 1024,
    expected_input_mode: str = "native",
    expected_pool: str = "cls",
    expected_rows_per_video: int = 81,
) -> ViRDMReference:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ViRDM reference bundle not found: {path}")
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    metadata = dict(bundle.get("metadata", {}))
    expected = {
        "encoder_id": str(expected_encoder_id),
        "encoder_checkpoint_sha256": str(expected_checkpoint_sha256),
        "input_mode": str(expected_input_mode),
        "input_height": int(expected_height),
        "input_width": int(expected_width),
        "pool": str(expected_pool),
        "feature_dim": int(expected_feature_dim),
        "feature_normalize": False,
        "num_rows": int(expected_rows),
        "rows_per_video": int(expected_rows_per_video),
        "virdm_reference_revision": str(expected_reference_revision),
        "nystrom_landmarks": int(expected_landmarks),
    }
    expected.update(dict(expected_encoder_contract))
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"ViRDM reference metadata {key}={metadata.get(key)!r}, expected {value!r}"
            )
    Z = bundle["Z"].float()
    alpha = bundle["alpha"].float()
    if tuple(Z.shape) != (int(expected_landmarks), int(expected_feature_dim)):
        raise ValueError(f"ViRDM Z shape {tuple(Z.shape)} is invalid")
    if tuple(alpha.shape) != (int(expected_landmarks),):
        raise ValueError(f"ViRDM alpha shape {tuple(alpha.shape)} is invalid")
    sigma = float(bundle["sigma"])
    k_rr = float(bundle["k_rr"])
    beta_value = bundle.get("beta")
    beta = float(beta_value) if beta_value is not None else None
    if not sigma > 0 or not torch.isfinite(torch.tensor([sigma, k_rr])).all():
        raise ValueError(f"invalid ViRDM sigma/k_rr: {sigma}, {k_rr}")
    if beta is not None and (
        not beta > 0 or not bool(torch.isfinite(torch.tensor(beta)))
    ):
        raise ValueError(f"invalid ViRDM joint beta: {beta}")
    Z = Z.to(device)
    return ViRDMReference(
        Z=Z,
        Z2=(Z * Z).sum(1),
        alpha=alpha.to(device),
        sigma=sigma,
        k_rr=k_rr,
        beta=beta,
        metadata=metadata,
        sha256=file_sha256(path),
    )


def self_normalize_virdm_loss(raw: torch.Tensor, eps: float = 1.0e-7) -> torch.Tensor:
    return raw / (raw.detach().abs() + float(eps))


def mmd_nystrom_with_terms(
    feat_g: torch.Tensor,
    reference: ViRDMReference,
    *,
    gen_chunk: int = 4096,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the biased generated kernel and frozen Nyström attraction."""

    def self_kernel_mean(features: torch.Tensor, gamma: float) -> torch.Tensor:
        row_count = features.shape[0]

        def block(values, start, stop):
            rows = values[start:stop]
            row_norm = (rows * rows).sum(1)
            value_norm = (values * values).sum(1)
            distance = (
                row_norm[:, None] + value_norm[None, :] - 2.0 * (rows @ values.T)
            ).clamp_min(0)
            return torch.exp(-gamma * distance).sum()

        total = features.new_zeros(())
        for start in range(0, row_count, int(gen_chunk)):
            stop = min(start + int(gen_chunk), row_count)
            total = total + torch.utils.checkpoint.checkpoint(
                block, features, start, stop, use_reentrant=False
            )
        return total / (row_count * row_count)

    gamma = 1.0 / (2.0 * reference.sigma * reference.sigma)

    g = feat_g.float()
    k_gg = self_kernel_mean(g, gamma)
    distance = (
        g.pow(2).sum(1)[:, None] + reference.Z2[None, :] - 2.0 * (g @ reference.Z.T)
    ).clamp_min(0)
    k_gr = (torch.exp(-gamma * distance) @ reference.alpha).mean()
    k_rr = g.new_tensor(reference.k_rr)
    raw = k_gg - 2.0 * k_gr + k_rr
    return raw, {"k_gg": k_gg, "k_gr_nystrom": k_gr, "k_rr": k_rr}
