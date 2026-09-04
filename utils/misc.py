import numpy as np
import os
import random
import secrets
import torch

BASE_SEED_ENV = "VIRDM_BASE_SEED"
BASE_SEED_DRAW_RANGE = 1 << 31
MAX_SEED = 1 << 32


def resolve_base_seed(seed: int) -> int:
    """Turn a requested seed into the concrete base seed a run should use."""
    # set_seed() feeds numpy, which only accepts seeds in [0, 2**32).
    if not 0 <= seed < MAX_SEED:
        raise ValueError(f"seed must be in [0, 2**32); got {seed}")
    if seed != 0:
        return seed
    shared = os.environ.get(BASE_SEED_ENV, "").strip()
    if shared:
        try:
            value = int(shared)
        except ValueError as exc:
            raise ValueError(
                f"{BASE_SEED_ENV} must be an integer, got {shared!r}"
            ) from exc
        if not 0 <= value < MAX_SEED:
            raise ValueError(f"{BASE_SEED_ENV} must be in [0, 2**32); got {value}")
        return value
    return 1 + secrets.randbelow(BASE_SEED_DRAW_RANGE)


def set_seed(seed: int, deterministic: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)


def merge_dict_list(dict_list):
    if len(dict_list) == 1:
        return dict_list[0]

    merged_dict = {}
    for k, v in dict_list[0].items():
        if isinstance(v, torch.Tensor):
            if v.ndim == 0:
                merged_dict[k] = torch.stack([d[k] for d in dict_list], dim=0)
            else:
                merged_dict[k] = torch.cat([d[k] for d in dict_list], dim=0)
        else:
            # for non-tensor values, we just copy the value from the first item
            merged_dict[k] = v
    return merged_dict
