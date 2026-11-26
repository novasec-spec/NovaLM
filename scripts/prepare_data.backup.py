"""
NovaLM Incremental Data Preparation Pipeline

Pipeline:

    data/raw/*.txt
          ↓
    conservative cleaning
          ↓
    frozen tokenizer
          ↓
    disk-backed token shards
          ↓
    train + validation manifest

Normal behavior:
    - New file          -> PROCESS
    - Existing file     -> SKIP
    - Changed file      -> STOP and warn
    - Existing shards   -> PRESERVE
    - Existing model    -> untouched

Full rebuild:
    python prepare_data.py --force

Normal update:
    python prepare_data.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from array import array
from pathlib import Path
from typing import Iterable

import numpy as np
from tokenizers import Tokenizer


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data" / "raw"
CLEANED_DIR = ROOT / "data" / "cleaned"
TOKEN_DIR = ROOT / "data" / "processed" / "tokens"

TOKENIZER_PATH = ROOT / "tokenizer" / "tokenizer.json"
MANIFEST_PATH = TOKEN_DIR / "manifest.json"


# ============================================================
# CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 128

# Each shard contains at most this many sequences.
SHARD_SEQUENCES = 50_000

# 5% validation.
VALIDATION_RATIO = 0.05

# Number of text lines processed at once.
READ_CHUNK_LINES = 2_000

# Number of texts sent to tokenizer at once.
TOKEN_BATCH_SIZE = 256

TOKEN_DTYPE = np.uint16

MAX_UINT16 = np.iinfo(np.uint16).max


# ============================================================
# HASHING
# ============================================================

def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Calculate SHA256 without loading the entire file into RAM."""

    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


# ============================================================
# JSON
# ============================================================

def atomic_write_json(
    path: Path,
    data: dict,
) -> None:
    """
    Safely write JSON.

    The file is written to a temporary file first and then
    atomically replaced.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )

    try:

        with os.fdopen(
            fd,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                data,
                f,
                indent=2,
                ensure_ascii=False,
            )

            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

        os.replace(
            temp_name,
            path,
        )

    except Exception:

        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass

        raise


def load_manifest() -> dict | None:

    if not MANIFEST_PATH.exists():
        return None

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ============================================================
# TOKENIZER
# ============================================================

def get_tokenizer_identity() -> dict:
    """Load and identify the frozen tokenizer."""

    if not TOKENIZER_PATH.exists():

        raise FileNotFoundError(
            f"Tokenizer not found:\n"
            f"{TOKENIZER_PATH}\n\n"
            f"Train the tokenizer before preparing data."
        )

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_PATH)
    )

    vocab_size = tokenizer.get_vocab_size()

    if vocab_size > MAX_UINT16:

        raise ValueError(
            f"Tokenizer vocabulary is {vocab_size:,}.\n"
            f"uint16 storage supports at most "
            f"{MAX_UINT16:,} tokens."
        )

    return {
        "path": str(
            TOKENIZER_PATH.relative_to(ROOT)
        ),
        "sha256": sha256_file(
            TOKENIZER_PATH
        ),
        "vocab_size": vocab_size,
    }


# ============================================================
# RAW FILE DISCOVERY
# ============================================================

def discover_raw_files() -> list[Path]:

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    return sorted(
        path
        for path in RAW_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() == ".txt"
    )


# ============================================================
# DATA CLEANING
# ============================================================

def clean_line(line: str) -> str:
    """
    Conservative cleaning.

    We deliberately avoid aggressive cleaning.

    We DO:
        - remove null bytes
        - normalize line endings
        - collapse repeated spaces/tabs
        - remove surrounding whitespace

    We DO NOT:
        - remove punctuation
        - remove numbers
        - remove capitalization
        - remove HTML automatically
        - rewrite sentences
        - change meaning
    """

    line = line.replace(
        "\x00",
        "",
    )

    line = line.replace(
        "\r\n",
        "\n",
    ).replace(
        "\r",
        "\n",
    )

    line = re.sub(
        r"[ \t]+",
        " ",
        line,
    )

    return line.strip()


def iter_cleaned_chunks(
    source: Path,
) -> Iterable[list[str]]:
    """
    Stream a text file in manageable batches.

    The entire source file is NEVER loaded into RAM.
    """

    batch: list[str] = []

    with source.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:

        for raw_line in f:

            cleaned = clean_line(
                raw_line
            )

            if not cleaned:
                continue

            batch.append(
                cleaned
            )

            if len(batch) >= READ_CHUNK_LINES:

                yield batch

                batch = []

    if batch:
        yield batch


# ============================================================
# SHARD WRITER
# ============================================================

class ShardWriter:
    """
    Writes fixed-length token sequences to binary shards.

    Each example contains:

        sequence_length + 1

    tokens.

    For sequence length 128:

        129 tokens

    are stored so the dataset can create:

        input  = tokens[:-1]
        target = tokens[1:]
    """

    def __init__(
        self,
        split: str,
        start_number: int,
        sequence_length: int,
        sequences_per_shard: int,
    ):

        self.split = split

        self.next_number = start_number

        self.sequence_length = sequence_length

        self.tokens_per_sequence = (
            sequence_length + 1
        )

        self.sequences_per_shard = (
            sequences_per_shard
        )

        # array('H') keeps memory usage much lower
        # than a Python list of integers.
        self.buffer = array("H")

        self.current_sequences = 0

        self.total_sequences = 0

        self.shards: list[dict] = []

    def add_tokens(
        self,
        tokens: list[int],
    ) -> None:

        if not tokens:
            return

        self.buffer.extend(
            tokens
        )

        needed = (
            self.tokens_per_sequence
        )

        while len(self.buffer) >= needed:

            sequence = self.buffer[
                :needed
            ]

            del self.buffer[
                :needed
            ]

            self.add_sequence(
                sequence
            )

    def add_sequence(
        self,
        sequence,
    ) -> None:

        self.buffer_sequence = getattr(
            self,
            "buffer_sequence",
            array("H"),
        )

        self.buffer_sequence.extend(
            sequence
        )

        self.current_sequences += 1

        if (
            self.current_sequences
            >= self.sequences_per_shard
        ):

            self.flush()

    def flush(self) -> None:

        if self.current_sequences == 0:
            return

        filename = (
            f"{self.split}-"
            f"{self.next_number:05d}.bin"
        )

        final_path = (
            TOKEN_DIR / filename
        )

        temp_path = (
            TOKEN_DIR
            / f".{filename}.tmp"
        )

        values = np.frombuffer(
            self.buffer_sequence,
            dtype=np.uint16,
        ).copy()

        values.tofile(
            temp_path
        )

        os.replace(
            temp_path,
            final_path,
        )

        self.shards.append(
            {
                "file": filename,
                "sequences": self.current_sequences,
            }
        )

        self.total_sequences += (
            self.current_sequences
        )

        self.next_number += 1

        self.buffer_sequence = array(
            "H"
        )

        self.current_sequences = 0

    def finish(self) -> None:
        """
        Write the final complete shard.

        Incomplete final sequence is intentionally ignored.
        """

        self.flush()


# ============================================================
# SHARD NUMBERING
# ============================================================

def next_shard_number(
    manifest: dict,
    split: str,
) -> int:

    numbers = []

    shards = (
        manifest
        .get("splits", {})
        .get(split, {})
        .get("shards", [])
    )

    for shard in shards:

        filename = Path(
            shard["file"]
        ).stem

        match = re.search(
            r"-(\d+)$",
            filename,
        )

        if match:
            numbers.append(
                int(match.group(1))
            )

    if not numbers:
        return 0

    return max(numbers) + 1


# ============================================================
# MANIFEST
# ============================================================

def create_manifest(
    tokenizer_identity: dict,
) -> dict:

    return {
        "format_version": 3,

        "sequence_length":
            SEQUENCE_LENGTH,

        "dtype":
            "uint16",

        "vocab_size":
            tokenizer_identity[
                "vocab_size"
            ],

        "tokenizer":
            tokenizer_identity,

        "splits": {

            "train": {
                "shards": [],
                "sequences": 0,
            },

            "val": {
                "shards": [],
                "sequences": 0,
            },
        },

        "source_files": [],
    }


def validate_manifest(
    manifest: dict,
    tokenizer_identity: dict,
) -> None:

    manifest_vocab = (
        manifest.get(
            "vocab_size"
        )
    )

    current_vocab = (
        tokenizer_identity[
            "vocab_size"
        ]
    )

    if manifest_vocab != current_vocab:

        raise RuntimeError(
            "\nVOCABULARY MISMATCH\n"
            f"Manifest:  {manifest_vocab}\n"
            f"Current:   {current_vocab}\n\n"
            "The tokenizer is frozen."
        )

    saved_tokenizer = (
        manifest.get(
            "tokenizer",
            {}
        )
    )

    saved_hash = (
        saved_tokenizer.get(
            "sha256"
        )
    )

    current_hash = (
        tokenizer_identity[
            "sha256"
        ]
    )

    if (
        saved_hash
        and saved_hash != current_hash
    ):

        raise RuntimeError(
            "\nTOKENIZER MISMATCH\n"
            f"Manifest tokenizer:\n"
            f"{saved_hash}\n\n"
            f"Current tokenizer:\n"
            f"{current_hash}\n\n"
            "Do not change the tokenizer "
            "during an existing training lineage."
        )


# ============================================================
# SOURCE REGISTRY
# ============================================================

def source_registry(
    manifest: dict,
) -> dict[str, dict]:

    return {
        item["source"]: item
        for item in manifest.get(
            "source_files",
            [],
        )
    }


def relative_source(
    path: Path,
) -> str:

    return str(
        path.relative_to(ROOT)
    )


# ============================================================
# PROCESS ONE SOURCE
# ============================================================

def process_source(
    source: Path,
    tokenizer: Tokenizer,
    train_writer: ShardWriter,
    val_writer: ShardWriter,
) -> tuple[int, int, int]:

    size_mb = (
        source.stat().st_size
        / (1024 ** 2)
    )

    print()
    print(
        f"Processing: {source}"
    )

    print(
        f"Size:       {size_mb:.2f} MB"
    )

    record_number = 0

    train_tokens = 0

    val_tokens = 0

    for texts in iter_cleaned_chunks(
        source
    ):

        for start in range(
            0,
            len(texts),
            TOKEN_BATCH_SIZE,
        ):

            batch = texts[
                start:
                start + TOKEN_BATCH_SIZE
            ]

            encoded = tokenizer.encode_batch(
                batch,
                add_special_tokens=False,
            )

            for item in encoded:

                tokens = item.ids

                if not tokens:
                    continue

                record_number += 1

                # Every 20th text record goes
                # to validation ≈ 5%.
                is_validation = (
                    record_number % 20 == 0
                )

                if is_validation:

                    val_writer.add_tokens(
                        tokens
                    )

                    val_tokens += len(
                        tokens
                    )

                else:

                    train_writer.add_tokens(
                        tokens
                    )

                    train_tokens += len(
                        tokens
                    )

        if record_number % 20_000 == 0:

            print(
                f"  records: "
                f"{record_number:,} | "
                f"train tokens: "
                f"{train_tokens:,} | "
                f"val tokens: "
                f"{val_tokens:,}"
            )

    print(
        f"Finished: {source.name}"
    )

    print(
        f"  records:     {record_number:,}"
    )

    print(
        f"  train tokens:{train_tokens:,}"
    )

    print(
        f"  val tokens:  {val_tokens:,}"
    )

    return (
        record_number,
        train_tokens,
        val_tokens,
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_data_pipeline(
    force_rebuild: bool = False,
) -> dict:

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CLEANED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TOKEN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer_identity = (
        get_tokenizer_identity()
    )

    raw_files = (
        discover_raw_files()
    )

    if not raw_files:

        raise RuntimeError(
            f"No .txt files found in:\n"
            f"{RAW_DIR}"
        )

    print(
        "=" * 70
    )

    print(
        "NovaLM Incremental Data Preparation"
    )

    print(
        "=" * 70
    )

    print(
        f"Tokenizer vocabulary: "
        f"{tokenizer_identity['vocab_size']:,}"
    )

    print(
        f"Sequence length: "
        f"{SEQUENCE_LENGTH}"
    )

    print(
        f"Validation ratio: "
        f"{VALIDATION_RATIO:.1%}"
    )

    print()

    # --------------------------------------------------------
    # FULL REBUILD
    # --------------------------------------------------------

    if force_rebuild:

        print(
            "!!! FULL REBUILD REQUESTED !!!"
        )

        print(
            "Existing token shards will be rebuilt."
        )

        print(
            "The model checkpoints will NOT be deleted."
        )

        for path in TOKEN_DIR.glob(
            "*.bin"
        ):

            path.unlink()

        if MANIFEST_PATH.exists():

            MANIFEST_PATH.unlink()

        manifest = create_manifest(
            tokenizer_identity
        )

    # --------------------------------------------------------
    # NORMAL / INCREMENTAL MODE
    # --------------------------------------------------------

    else:

        manifest = (
            load_manifest()
        )

        if manifest is None:

            manifest = create_manifest(
                tokenizer_identity
            )

        else:

            validate_manifest(
                manifest,
                tokenizer_identity,
            )

            manifest[
                "format_version"
            ] = 3

            manifest[
                "sequence_length"
            ] = SEQUENCE_LENGTH

            manifest[
                "dtype"
            ] = "uint16"

            manifest[
                "vocab_size"
            ] = tokenizer_identity[
                "vocab_size"
            ]

            manifest[
                "tokenizer"
            ] = tokenizer_identity

            manifest.setdefault(
                "splits",
                {},
            )

            manifest[
                "splits"
            ].setdefault(
                "train",
                {
                    "shards": [],
                    "sequences": 0,
                },
            )

            manifest[
                "splits"
            ].setdefault(
                "val",
                {
                    "shards": [],
                    "sequences": 0,
                },
            )

            manifest.setdefault(
                "source_files",
                [],
            )

    registry = source_registry(
        manifest
    )

    new_files = []

    skipped_files = []

    print(
        "SOURCE CHECK"
    )

    print(
        "-" * 70
    )

    for source in raw_files:

        source_key = (
            relative_source(
                source
            )
        )

        fingerprint = (
            sha256_file(
                source
            )
        )

        previous = (
            registry.get(
                source_key
            )
        )

        # ----------------------------------------------------
        # NEW FILE
        # ----------------------------------------------------

        if previous is None:

            print(
                f"[NEW]  {source_key}"
            )

            new_files.append(
                source
            )

            continue

        old_fingerprint = (
            previous.get(
                "fingerprint"
            )
        )

        # ----------------------------------------------------
        # UNCHANGED FILE
        # ----------------------------------------------------

        if (
            old_fingerprint
            == fingerprint
        ):

            print(
                f"[SKIP] {source_key}"
            )

            skipped_files.append(
                source
            )

            continue

        # ----------------------------------------------------
        # CHANGED FILE
        # ----------------------------------------------------

        raise RuntimeError(
            "\n"
            + "=" * 70
            + "\n"
            "SOURCE FILE CHANGED\n"
            + "=" * 70
            + "\n"
            f"File:\n"
            f"  {source_key}\n\n"
            f"Old SHA256:\n"
            f"  {old_fingerprint}\n\n"
            f"New SHA256:\n"
            f"  {fingerprint}\n\n"
            "The old version has already been converted "
            "into token shards.\n\n"
            "NovaLM will NOT silently process the changed "
            "file again because that could duplicate data.\n\n"
            "If the change was accidental, restore the old file.\n\n"
            "If you intentionally changed the dataset, "
            "perform a deliberate rebuild with:\n\n"
            "    python prepare_data.py --force\n"
        )

    # --------------------------------------------------------
    # NOTHING NEW
    # --------------------------------------------------------

    if not new_files:

        atomic_write_json(
            MANIFEST_PATH,
            manifest,
        )

        print()

        print(
            "=" * 70
        )

        print(
            "NO NEW DATA"
        )

        print(
            "=" * 70
        )

        print(
            f"Training sequences:   "
            f"{manifest['splits']['train']['sequences']:,}"
        )

        print(
            f"Validation sequences: "
            f"{manifest['splits']['val']['sequences']:,}"
        )

        print()
        print(
            "Nothing was reprocessed."
        )

        return manifest

    # --------------------------------------------------------
    # SHARD WRITERS
    # --------------------------------------------------------

    train_writer = ShardWriter(
        split="train",
        start_number=next_shard_number(
            manifest,
            "train",
        ),
        sequence_length=SEQUENCE_LENGTH,
        sequences_per_shard=SHARD_SEQUENCES,
    )

    val_writer = ShardWriter(
        split="val",
        start_number=next_shard_number(
            manifest,
            "val",
        ),
        sequence_length=SEQUENCE_LENGTH,
        sequences_per_shard=SHARD_SEQUENCES,
    )

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_PATH)
    )

    # --------------------------------------------------------
    # PROCESS NEW FILES
    # --------------------------------------------------------

    for source in new_files:

        source_key = (
            relative_source(
                source
            )
        )

        fingerprint = (
            sha256_file(
                source
            )
        )

        (
            record_count,
            train_token_count,
            val_token_count,
        ) = process_source(
            source=source,
            tokenizer=tokenizer,
            train_writer=train_writer,
            val_writer=val_writer,
        )

        # ----------------------------------------------------
        # SAVE CLEANED COPY
        # ----------------------------------------------------

        cleaned_path = (
            CLEANED_DIR
            / "train"
            / source.name
        )

        cleaned_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with cleaned_path.open(
            "w",
            encoding="utf-8",
        ) as output:

            for texts in iter_cleaned_chunks(
                source
            ):

                for line in texts:

                    output.write(
                        line
                    )

                    output.write(
                        "\n"
                    )

        # ----------------------------------------------------
        # REGISTER SOURCE
        # ----------------------------------------------------

        manifest[
            "source_files"
        ].append(
            {
                "source": source_key,
                "cleaned": str(
                    cleaned_path.relative_to(
                        ROOT
                    )
                ),
                "fingerprint": fingerprint,
                "bytes": source.stat().st_size,
                "records": record_count,
                "train_tokens": train_token_count,
                "val_tokens": val_token_count,
            }
        )

    # --------------------------------------------------------
    # FINISH SHARDS
    # --------------------------------------------------------

    train_writer.finish()

    val_writer.finish()

    # --------------------------------------------------------
    # UPDATE MANIFEST
    # --------------------------------------------------------

    manifest[
        "splits"
    ][
        "train"
    ][
        "shards"
    ].extend(
        train_writer.shards
    )

    manifest[
        "splits"
    ][
        "val"
    ][
        "shards"
    ].extend(
        val_writer.shards
    )

    manifest[
        "splits"
    ][
        "train"
    ][
        "sequences"
    ] += train_writer.total_sequences

    manifest[
        "splits"
    ][
        "val"
    ][
        "sequences"
    ] += val_writer.total_sequences

    atomic_write_json(
        MANIFEST_PATH,
        manifest,
    )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "DATA PREPARATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"New files processed:       "
        f"{len(new_files)}"
    )

    print(
        f"Existing files skipped:    "
        f"{len(skipped_files)}"
    )

    print(
        f"New train sequences:       "
        f"{train_writer.total_sequences:,}"
    )

    print(
        f"New validation sequences:  "
        f"{val_writer.total_sequences:,}"
    )

    print()

    print(
        f"Total train sequences:     "
        f"{manifest['splits']['train']['sequences']:,}"
    )

    print(
        f"Total validation sequences:"
        f" {manifest['splits']['val']['sequences']:,}"
    )

    print()

    print(
        f"Manifest:\n"
        f"  {MANIFEST_PATH}"
    )

    print()

    print(
        "Existing token shards were preserved."
    )

    print(
        "Existing checkpoints were NOT changed."
    )

    return manifest


# ============================================================
# CLI
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Prepare NovaLM training data incrementally."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "FULL REBUILD. Reprocess all raw text "
            "and recreate token shards. "
            "Do not use for normal data updates."
        ),
    )

    args = parser.parse_args()

    run_data_pipeline(
        force_rebuild=args.force
    )


if __name__ == "__main__":
    main()
