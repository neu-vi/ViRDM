"""Frozen SigLIP2 table and joint video-text feature utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch

from .reference import file_sha256


VIRDM_JOINT_TEXT_MODEL = "ViT-SO400M-16-SigLIP2-256"
VIRDM_JOINT_TEXT_PRETRAINED = "webli"
VIRDM_JOINT_TEXT_DIM = 1152
VIRDM_JOINT_BANDWIDTH_SCALE = 1.0
VIRDM_JOINT_CONSTRUCTION = "concat_single_rbf"
VIRDM_JOINT_POOL = "global_mean_all_tokens_concat_siglip2_text"


def prompt_rows_sha256(prompts: Iterable[str]) -> str:
    """Hash an ordered prompt table without delimiter ambiguity."""

    digest = hashlib.sha256()
    for index, prompt in enumerate(prompts):
        encoded = str(prompt).encode("utf-8")
        digest.update(int(index).to_bytes(8, byteorder="little", signed=False))
        digest.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def load_frozen_text_table(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_rows: int,
    expected_dim: int = VIRDM_JOINT_TEXT_DIM,
    device: torch.device | str = "cpu",
    norm_atol: float = 5.0e-4,
) -> torch.Tensor:
    """Load the official frozen, L2-normalized ``tau(c)`` table strictly."""

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"joint text table not found: {path}")
    actual_sha256 = file_sha256(path)
    if actual_sha256 != str(expected_sha256):
        raise RuntimeError(
            f"joint text table SHA-256 {actual_sha256} != configured {expected_sha256}"
        )
    array = np.load(path, allow_pickle=False)
    if array.dtype != np.float32:
        raise ValueError(f"joint text table must be float32, got {array.dtype}")
    if tuple(array.shape) != (int(expected_rows), int(expected_dim)):
        raise ValueError(
            f"joint text table shape {tuple(array.shape)} != "
            f"{(int(expected_rows), int(expected_dim))}"
        )
    table = torch.from_numpy(np.ascontiguousarray(array)).float()
    if not bool(torch.isfinite(table).all()):
        raise ValueError("joint text table contains NaN or Inf")
    norm_error = float((torch.linalg.vector_norm(table, dim=1) - 1.0).abs().max())
    if norm_error > float(norm_atol):
        raise ValueError(
            f"joint text table is not L2-normalized: max norm error {norm_error:.6g}"
        )
    return table.to(device=device)


def virdm_joint_contract(
    visual_contract: Mapping[str, object],
    *,
    text_table_sha256: str,
    prompt_rows_sha256_value: str,
    bandwidth_scale: float = VIRDM_JOINT_BANDWIDTH_SCALE,
    visual_feature_dim: int = 1024,
    text_feature_dim: int = VIRDM_JOINT_TEXT_DIM,
    text_model: str = VIRDM_JOINT_TEXT_MODEL,
    text_pretrained: str = VIRDM_JOINT_TEXT_PRETRAINED,
) -> dict[str, object]:
    """Return the immutable metadata contract for a joint reference bundle."""

    contract = dict(visual_contract)
    contract.update(
        {
            "pool": VIRDM_JOINT_POOL,
            "pooled_rows_per_video": 1,
            "rows_per_video": 1,
            "feature_dim": int(visual_feature_dim) + int(text_feature_dim),
            "joint_enable": True,
            "joint_construction": VIRDM_JOINT_CONSTRUCTION,
            "joint_visual_feature_dim": int(visual_feature_dim),
            "joint_text_feature_dim": int(text_feature_dim),
            "joint_text_encoder_id": str(text_model),
            "joint_text_encoder_pretrained": str(text_pretrained),
            "joint_text_feature_normalize": True,
            "joint_text_table_sha256": str(text_table_sha256),
            "joint_prompt_rows_sha256": str(prompt_rows_sha256_value),
            "joint_pairing": "lmdb_row_index",
            "joint_bandwidth_scale": float(bandwidth_scale),
        }
    )
    return contract


def couple_video_text(
    visual_features: torch.Tensor,
    text_rows: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Build ``[visual_features | beta * text_rows]``."""

    text_block = (float(beta) * text_rows).to(visual_features.dtype)
    return torch.cat([visual_features, text_block], dim=1)
