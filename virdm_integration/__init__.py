"""Runtime components for the focused ViRDM objective."""

from .joint_text import (
    VIRDM_JOINT_BANDWIDTH_SCALE,
    VIRDM_JOINT_CONSTRUCTION,
    VIRDM_JOINT_POOL,
    VIRDM_JOINT_TEXT_DIM,
    VIRDM_JOINT_TEXT_MODEL,
    VIRDM_JOINT_TEXT_PRETRAINED,
    couple_video_text,
    load_frozen_text_table,
    prompt_rows_sha256,
    virdm_joint_contract,
)
from .reference import (
    ViRDMReference,
    load_reference,
    mmd_nystrom_with_terms,
    self_normalize_virdm_loss,
)
from .vjepa21_video import NativeVJEPA21VideoEncoder

__all__ = [
    "NativeVJEPA21VideoEncoder",
    "ViRDMReference",
    "VIRDM_JOINT_BANDWIDTH_SCALE",
    "VIRDM_JOINT_CONSTRUCTION",
    "VIRDM_JOINT_POOL",
    "VIRDM_JOINT_TEXT_DIM",
    "VIRDM_JOINT_TEXT_MODEL",
    "VIRDM_JOINT_TEXT_PRETRAINED",
    "couple_video_text",
    "load_frozen_text_table",
    "load_reference",
    "mmd_nystrom_with_terms",
    "prompt_rows_sha256",
    "virdm_joint_contract",
    "self_normalize_virdm_loss",
]
