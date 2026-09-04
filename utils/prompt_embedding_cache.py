import hashlib
import io
from typing import Dict, Iterable, List

import lmdb
import torch


def prompt_cache_key(prompt: str) -> bytes:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest().encode("ascii")


class PromptEmbeddingLMDBCache:
    def __init__(self, path: str):
        self.path = path
        self.env = lmdb.open(
            path,
            readonly=True,
            lock=False,
            readahead=False,
            subdir=True,
            max_readers=512,
        )
        self._cpu_cache: Dict[str, torch.Tensor] = {}

    def _load_prompt(self, prompt: str) -> torch.Tensor:
        if prompt in self._cpu_cache:
            return self._cpu_cache[prompt]

        with self.env.begin(write=False) as txn:
            payload = txn.get(prompt_cache_key(prompt))
        if payload is None:
            raise KeyError(f"Prompt embedding not found in cache: {prompt!r}")

        tensor = torch.load(io.BytesIO(payload), map_location="cpu")
        if tensor.ndim != 2:
            raise ValueError(
                f"Expected cached prompt embedding to have shape [seq, dim], got {tuple(tensor.shape)}"
            )
        self._cpu_cache[prompt] = tensor
        return tensor

    def get_batch(self, prompts: Iterable[str], device, dtype) -> torch.Tensor:
        trimmed: List[torch.Tensor] = [self._load_prompt(prompt) for prompt in prompts]
        max_len = max(tensor.shape[0] for tensor in trimmed)
        hidden_dim = trimmed[0].shape[1]
        batch = torch.zeros(
            len(trimmed),
            max_len,
            hidden_dim,
            dtype=dtype,
            device=device,
        )
        for index, tensor in enumerate(trimmed):
            length = tensor.shape[0]
            batch[index, :length] = tensor.to(device=device, dtype=dtype, non_blocking=True)
        return batch
