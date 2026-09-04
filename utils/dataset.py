import os

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.lmdb_ import get_array_shape_from_lmdb, retrieve_row_from_lmdb


class TextDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None):
        with open(prompt_path, encoding="utf-8") as handle:
            self.prompt_list = [line.rstrip() for line in handle]
        if not self.prompt_list or any(not prompt.strip() for prompt in self.prompt_list):
            raise ValueError("prompt file must contain one non-empty prompt per line")

        if extended_prompt_path is None:
            self.extended_prompt_list = None
        else:
            with open(extended_prompt_path, encoding="utf-8") as handle:
                self.extended_prompt_list = [line.rstrip() for line in handle]
            if len(self.extended_prompt_list) != len(self.prompt_list):
                raise ValueError("prompt files must contain the same number of rows")
            if any(not prompt.strip() for prompt in self.extended_prompt_list):
                raise ValueError("extended prompt file contains an empty row")

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        item = {"prompts": self.prompt_list[idx], "idx": idx}
        if self.extended_prompt_list is not None:
            item["extended_prompts"] = self.extended_prompt_list[idx]
        return item


class PromptLMDBDataset(Dataset):
    """Prompt-only view of the row-aligned training LMDB."""

    def __init__(
        self,
        data_path: str,
        max_pair: int = int(1e8),
        readahead: bool = False,
    ):
        if not os.path.isfile(os.path.join(data_path, "data.mdb")):
            raise FileNotFoundError(
                f"expected an LMDB directory containing data.mdb: {data_path}"
            )
        self.env = lmdb.open(
            data_path,
            readonly=True,
            lock=False,
            readahead=readahead,
            meminit=False,
        )
        # ViRDM only consumes captions.  Prefer a compact prompt-only LMDB,
        # while remaining compatible with the official merged latent LMDB.
        try:
            self.rows = get_array_shape_from_lmdb(self.env, "prompts")[0]
        except (AttributeError, KeyError):
            self.rows = get_array_shape_from_lmdb(self.env, "latents")[0]
        self.max_pair = int(max_pair)

    def __len__(self):
        return min(self.rows, self.max_pair)

    def __getitem__(self, idx):
        return {
            "prompts": retrieve_row_from_lmdb(self.env, "prompts", str, idx),
            "idx": int(idx),
        }


class CleanLatentLMDBDataset(Dataset):
    """Clean-latent and prompt view used only while rebuilding the reference."""

    def __init__(
        self,
        data_path: str,
        max_pair: int = int(1e8),
        readahead: bool = False,
    ):
        if not os.path.isfile(os.path.join(data_path, "data.mdb")):
            raise FileNotFoundError(
                f"expected an LMDB directory containing data.mdb: {data_path}"
            )
        self.env = lmdb.open(
            data_path,
            readonly=True,
            lock=False,
            readahead=readahead,
            meminit=False,
        )
        self.latents_shape = get_array_shape_from_lmdb(self.env, "latents")
        self.max_pair = int(max_pair)

    def __len__(self):
        return min(int(self.latents_shape[0]), self.max_pair)

    def __getitem__(self, idx):
        latents = retrieve_row_from_lmdb(
            self.env,
            "latents",
            np.float16,
            idx,
            shape=self.latents_shape[1:],
        )
        if latents.ndim == 5:
            clean_latent = latents[-1]
        elif latents.ndim == 4:
            clean_latent = latents
        else:
            raise ValueError(
                "latent row must be [steps,T,C,H,W] or [T,C,H,W], "
                f"got {latents.shape}"
            )
        if tuple(clean_latent.shape) != (21, 16, 60, 104):
            raise ValueError(
                "clean latent must be [21,16,60,104], "
                f"got {clean_latent.shape}"
            )
        return {
            "prompts": retrieve_row_from_lmdb(self.env, "prompts", str, idx),
            "clean_latent": torch.from_numpy(clean_latent.copy()).float(),
            "idx": int(idx),
        }


def cycle(dataloader):
    while True:
        yield from dataloader
