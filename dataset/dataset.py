"""
NovaLM Disk-Backed Dataset

Loads tokenized training data from:
    data/processed/tokens/

The token files are memory-mapped, so a very large dataset
does not need to fit into RAM.

Each stored example contains:

    input  = tokens[:-1]
    target = tokens[1:]

Expected shard format:
    uint16 or uint32 binary token IDs

Manifest:
    data/processed/tokens/manifest.json
"""

from __future__ import annotations

import json
from pathlib import Path
from bisect import bisect_right

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOKENS_DIR = PROJECT_ROOT / "data" / "processed" / "tokens"
MANIFEST_PATH = TOKENS_DIR / "manifest.json"


class DiskTokenDataset(Dataset):
    """
    Dataset backed by memory-mapped token shards.

    No complete dataset is loaded into RAM.
    """

    def __init__(
        self,
        split: str = "train",
        sequence_length: int = 128,
    ):
        self.split = split
        self.sequence_length = sequence_length

        if not MANIFEST_PATH.exists():
            raise FileNotFoundError(
                f"Token manifest not found:\n{MANIFEST_PATH}\n\n"
                "Run:\n"
                "    python -m scripts.prepare_data"
            )

        with MANIFEST_PATH.open("r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        if split not in self.manifest["splits"]:
            raise ValueError(
                f"Unknown split '{split}'. "
                f"Available: {list(self.manifest['splits'])}"
            )

        split_info = self.manifest["splits"][split]

        if not split_info["shards"]:
            raise ValueError(
                f"No token shards exist for split '{split}'."
            )

        self.shards = []
        self.cumulative_sequences = []

        total = 0

        dtype_name = self.manifest.get("dtype", "uint16")

        if dtype_name == "uint16":
            self.dtype = np.uint16
        elif dtype_name == "uint32":
            self.dtype = np.uint32
        else:
            raise ValueError(
                f"Unsupported token dtype: {dtype_name}"
            )

        for shard_info in split_info["shards"]:
            path = TOKENS_DIR / shard_info["file"]

            if not path.exists():
                raise FileNotFoundError(
                    f"Missing token shard:\n{path}"
                )

            sequence_count = int(
                shard_info["sequences"]
            )

            mmap = np.memmap(
                path,
                dtype=self.dtype,
                mode="r",
                shape=(
                    sequence_count,
                    sequence_length + 1,
                ),
            )

            self.shards.append(
                {
                    "path": path,
                    "mmap": mmap,
                    "sequences": sequence_count,
                }
            )

            total += sequence_count
            self.cumulative_sequences.append(total)

        self.total_sequences = total

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, index: int):
        if index < 0:
            index += self.total_sequences

        if index < 0 or index >= self.total_sequences:
            raise IndexError(index)

        shard_index = bisect_right(
            self.cumulative_sequences,
            index,
        )

        previous_total = (
            self.cumulative_sequences[shard_index - 1]
            if shard_index > 0
            else 0
        )

        local_index = index - previous_total

        tokens = self.shards[shard_index]["mmap"][
            local_index
        ]

        input_ids = torch.from_numpy(
            tokens[:-1].astype(np.int64)
        )

        target_ids = torch.from_numpy(
            tokens[1:].astype(np.int64)
        )

        return input_ids, target_ids


def get_dataloader(
    split: str = "train",
    batch_size: int = 8,
    shuffle: bool = True,
    sequence_length: int = 128,
    num_workers: int = 0,
):
    """
    Create a DataLoader over the disk-backed dataset.
    """

    dataset = DiskTokenDataset(
        split=split,
        sequence_length=sequence_length,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return loader


if __name__ == "__main__":
    print("=" * 70)
    print("NovaLM Disk-Backed Dataset Test")
    print("=" * 70)

    dataset = DiskTokenDataset(
        split="train",
        sequence_length=128,
    )

    print(f"Training sequences: {len(dataset):,}")

    input_ids, target_ids = dataset[0]

    print(f"Input shape:  {tuple(input_ids.shape)}")
    print(f"Target shape: {tuple(target_ids.shape)}")
    print(f"Input dtype:  {input_ids.dtype}")
    print(f"Target dtype: {target_ids.dtype}")

    print()
    print("First input tokens:")
    print(input_ids[:20].tolist())

    print()
    print("First target tokens:")
    print(target_ids[:20].tolist())

    print()
    print("Dataset test passed.")
