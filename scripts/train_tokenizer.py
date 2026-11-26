
"""
NovaLM 16K BPE Tokenizer Trainer

Trains a Byte-Level BPE tokenizer from the current NovaLM corpus.

Important:
- This creates a NEW tokenizer/model lineage.
- Once the tokenizer is used to train NovaLM, do not modify it.
- The tokenizer fingerprint is stored in tokenizer_metadata.json.
- Large files are passed directly to the Tokenizers library, which
  processes training files without requiring the whole corpus in RAM.

Run from ~/NovaLM:

    python -m scripts.train_tokenizer

Optional:

    python -m scripts.train_tokenizer --vocab-size 16000
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tokenizers import Tokenizer, decoders, pre_tokenizers

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
CLEANED_DIR = ROOT / "data" / "cleaned"

TOKENIZER_DIR = ROOT / "tokenizer"
TOKENIZER_FILE = TOKENIZER_DIR / "tokenizer.json"
METADATA_FILE = TOKENIZER_DIR / "tokenizer_metadata.json"


# ============================================================
# SETTINGS
# ============================================================

DEFAULT_VOCAB_SIZE = 16_000

SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
]


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """
    Calculate SHA-256 without loading the entire file into RAM.
    """
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def collect_training_files() -> list[Path]:
    """
    Prefer cleaned text if it exists.

    Otherwise use data/raw/*.txt.

    Validation files are intentionally excluded from tokenizer
    training so the tokenizer does not learn directly from the
    validation corpus.
    """

    cleaned_files = sorted(CLEANED_DIR.rglob("*.txt"))

    if cleaned_files:
        return cleaned_files

    raw_files = sorted(RAW_DIR.glob("*.txt"))

    return raw_files


def build_tokenizer(vocab_size: int, files: list[Path]) -> Tokenizer:
    """
    Build and train a Byte-Level BPE tokenizer.
    """

    tokenizer = Tokenizer(
        BPE(
            unk_token="<unk>",
            fuse_unk=True,
        )
    )

    tokenizer.pre_tokenizer = ByteLevel(
        add_prefix_space=False
    )

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    tokenizer.train(
        [str(path) for path in files],
        trainer,
    )

    return tokenizer


def create_metadata(
    tokenizer: Tokenizer,
    files: list[Path],
    requested_vocab_size: int,
) -> dict:

    actual_vocab_size = tokenizer.get_vocab_size()

    file_metadata = []

    for path in files:
        stat = path.stat()

        file_metadata.append(
            {
                "path": str(path.relative_to(ROOT)),
                "size_bytes": stat.st_size,
                "sha256": sha256_file(path),
            }
        )

    return {
        "tokenizer_version": 1,
        "tokenizer_type": "BPE",
        "pre_tokenizer": "ByteLevel",

        "vocab_size_requested": requested_vocab_size,
        "vocab_size_actual": actual_vocab_size,

        "special_tokens": SPECIAL_TOKENS,

        "tokenizer_file": str(
            TOKENIZER_FILE.relative_to(ROOT)
        ),

        "training_files": file_metadata,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Train the NovaLM 16K BPE tokenizer."
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
        help=f"Target vocabulary size (default: {DEFAULT_VOCAB_SIZE})",
    )

    args = parser.parse_args()

    if args.vocab_size < 256:
        raise ValueError(
            "Vocabulary size must be at least 256."
        )

    print("=" * 70)
    print("NovaLM BPE TOKENIZER TRAINER")
    print("=" * 70)

    print()
    print(f"Requested vocabulary size: {args.vocab_size:,}")

    files = collect_training_files()

    if not files:
        print()
        print("[ERROR] No training text files found.")
        print()
        print("Put .txt files in:")
        print(f"  {RAW_DIR}")
        print()
        print("Then run this command again.")
        return

    print()
    print(f"Training files: {len(files)}")

    total_bytes = 0

    for path in files:
        size = path.stat().st_size
        total_bytes += size

        print(
            f"  - {path.relative_to(ROOT)} "
            f"({size / 1024 / 1024:.2f} MB)"
        )

    print()
    print(
        f"Total training data: "
        f"{total_bytes / 1024 / 1024:.2f} MB"
    )

    print()
    print("WARNING:")
    print("This will create a NEW tokenizer.")
    print("Old NovaLM checkpoints are NOT compatible with it.")
    print()

    TOKENIZER_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("TRAINING TOKENIZER")
    print("=" * 70)

    tokenizer = build_tokenizer(
        vocab_size=args.vocab_size,
        files=files,
    )

    actual_vocab_size = tokenizer.get_vocab_size()

    print()
    print("=" * 70)
    print("TOKENIZER CREATED")
    print("=" * 70)

    print()
    print(f"Requested vocabulary: {args.vocab_size:,}")
    print(f"Actual vocabulary:    {actual_vocab_size:,}")

    # --------------------------------------------------------
    # SAVE TOKENIZER
    # --------------------------------------------------------

    tokenizer.save(
        str(TOKENIZER_FILE),
        pretty=True,
    )

    # --------------------------------------------------------
    # CREATE METADATA
    # --------------------------------------------------------

    metadata = create_metadata(
        tokenizer=tokenizer,
        files=files,
        requested_vocab_size=args.vocab_size,
    )

    # Fingerprint the final tokenizer itself.
    tokenizer_hash = sha256_file(TOKENIZER_FILE)

    metadata["tokenizer_sha256"] = tokenizer_hash

    METADATA_FILE.write_text(
        json.dumps(
            metadata,
            indent=4,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_texts = [
        "Hello world.",
        "Programming is useful.",
        "Artificial intelligence learns patterns from data.",
        "The student is learning mathematics.",
    ]

    print()
    print("=" * 70)
    print("TOKENIZER TEST")
    print("=" * 70)

    for text in test_texts:

        encoding = tokenizer.encode(text)

        decoded = tokenizer.decode(
            encoding.ids
        )

        print()
        print(f"Original: {text}")
        print(f"Tokens:   {encoding.tokens}")
        print(f"IDs:      {encoding.ids}")
        print(f"Decoded:  {decoded}")

    print()
    print("=" * 70)
    print("TOKENIZER READY")
    print("=" * 70)

    print()
    print(f"Tokenizer:")
    print(f"  {TOKENIZER_FILE}")

    print()
    print("Metadata:")
    print(f"  {METADATA_FILE}")

    print()
    print(f"Vocabulary size: {actual_vocab_size:,}")
    print(f"SHA-256:         {tokenizer_hash}")

    print()
    print("IMPORTANT:")
    print("Freeze this tokenizer before training NovaLM.")
    print("Do not retrain it after large-scale tokenization begins.")

    print("=" * 70)


if __name__ == "__main__":
    main()

