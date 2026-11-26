from pathlib import Path
import json

from tokenizers import Tokenizer


# ============================================================
# NovaLM Token Dataset Builder V2
# ============================================================

TOKENIZER_FILE = Path(
    "tokenizer/tokenizer.json"
)

SPLITS_DIR = Path(
    "data/processed/splits"
)

OUTPUT_DIR = Path(
    "data/processed/tokenized"
)

SEQUENCE_LENGTH = 32


def load_tokenizer():
    """Load the trained tokenizer."""

    if not TOKENIZER_FILE.exists():
        raise FileNotFoundError(
            "Tokenizer not found.\n"
            "Run:\n"
            "python scripts/train_tokenizer.py"
        )

    return Tokenizer.from_file(
        str(TOKENIZER_FILE)
    )


def get_special_token_id(tokenizer, token):
    """Get the ID of a special token."""

    token_id = tokenizer.token_to_id(token)

    if token_id is None:
        raise ValueError(
            f"Special token {token!r} "
            "does not exist in tokenizer."
        )

    return token_id


def load_documents(split):
    """Load documents belonging to one split."""

    split_dir = SPLITS_DIR / split

    files = sorted(
        split_dir.glob("*.txt")
    )

    documents = []

    for file in files:

        text = file.read_text(
            encoding="utf-8",
            errors="ignore"
        ).strip()

        if text:
            documents.append({
                "file": file.name,
                "text": text,
            })

    return documents


def encode_document(
    tokenizer,
    text,
    bos_id,
    eos_id
):
    """Encode one document and add boundaries."""

    encoding = tokenizer.encode(text)

    token_ids = encoding.ids

    return (
        [bos_id]
        + token_ids
        + [eos_id]
    )


def create_sequences(token_ids):
    """
    Create shifted input/target pairs.

    Example with sequence length 4:

        tokens:
        [10, 20, 30, 40, 50]

        input:
        [10, 20, 30, 40]

        target:
        [20, 30, 40, 50]
    """

    sequences = []

    if len(token_ids) < SEQUENCE_LENGTH + 1:
        return sequences

    for start in range(
        0,
        len(token_ids) - SEQUENCE_LENGTH,
        SEQUENCE_LENGTH
    ):

        chunk = token_ids[
            start:start + SEQUENCE_LENGTH + 1
        ]

        if len(chunk) != SEQUENCE_LENGTH + 1:
            continue

        sequences.append({
            "input": chunk[:-1],
            "target": chunk[1:],
        })

    return sequences


def process_split(
    tokenizer,
    split,
    bos_id,
    eos_id
):
    """Process documents independently."""

    documents = load_documents(split)

    all_sequences = []
    document_metadata = []

    total_tokens = 0

    for document in documents:

        token_ids = encode_document(
            tokenizer,
            document["text"],
            bos_id,
            eos_id
        )

        total_tokens += len(token_ids)

        sequences = create_sequences(
            token_ids
        )

        all_sequences.extend(
            sequences
        )

        document_metadata.append({
            "file": document["file"],
            "tokens": len(token_ids),
            "sequences": len(sequences),
        })

    output = {
        "split": split,
        "sequence_length": SEQUENCE_LENGTH,
        "documents": document_metadata,
        "total_tokens": total_tokens,
        "total_sequences": len(
            all_sequences
        ),
        "sequences": all_sequences,
    }

    output_file = (
        OUTPUT_DIR /
        f"{split}.json"
    )

    output_file.write_text(
        json.dumps(
            output,
            indent=2
        ),
        encoding="utf-8"
    )

    return {
        "documents": len(documents),
        "tokens": total_tokens,
        "sequences": len(
            all_sequences
        ),
    }


def main():

    print("=" * 60)
    print("NovaLM Token Dataset Builder V2")
    print("=" * 60)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    tokenizer = load_tokenizer()

    bos_id = get_special_token_id(
        tokenizer,
        "<bos>"
    )

    eos_id = get_special_token_id(
        tokenizer,
        "<eos>"
    )

    print(
        f"\nVocabulary size: "
        f"{tokenizer.get_vocab_size():,}"
    )

    print(
        f"BOS token ID: {bos_id}"
    )

    print(
        f"EOS token ID: {eos_id}"
    )

    print(
        f"Sequence length: "
        f"{SEQUENCE_LENGTH}"
    )

    print("\n" + "=" * 60)
    print("PROCESSING DATA")
    print("=" * 60)

    summary = {}

    for split in [
        "train",
        "validation",
        "test",
    ]:

        result = process_split(
            tokenizer,
            split,
            bos_id,
            eos_id
        )

        summary[split] = result

        print(f"\n{split.upper()}")

        print(
            f"Documents: "
            f"{result['documents']:,}"
        )

        print(
            f"Tokens: "
            f"{result['tokens']:,}"
        )

        print(
            f"Sequences: "
            f"{result['sequences']:,}"
        )

    summary_file = (
        OUTPUT_DIR / "summary.json"
    )

    summary_file.write_text(
        json.dumps(
            summary,
            indent=4
        ),
        encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("TOKEN DATASET CREATED")
    print("=" * 60)

    print(
        f"\nOutput directory:"
        f"\n{OUTPUT_DIR}"
    )

    print(
        f"\nSummary:"
        f"\n{summary_file}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
