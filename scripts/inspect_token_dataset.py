from pathlib import Path
import json

from tokenizers import Tokenizer


# ============================================================
# NovaLM Token Dataset Inspector V2
# ============================================================

TOKENIZER_FILE = Path(
    "tokenizer/tokenizer.json"
)

DATASET_DIR = Path(
    "data/processed/tokenized"
)


def load_dataset(split):

    dataset_file = (
        DATASET_DIR /
        f"{split}.json"
    )

    if not dataset_file.exists():
        raise FileNotFoundError(
            f"Dataset file not found: "
            f"{dataset_file}"
        )

    return json.loads(
        dataset_file.read_text(
            encoding="utf-8"
        )
    )


def inspect_split(tokenizer, split):

    data = load_dataset(split)

    print("\n" + "=" * 60)
    print(f"{split.upper()} DATASET")
    print("=" * 60)

    print(
        f"\nDocuments: "
        f"{len(data['documents']):,}"
    )

    print(
        f"Total tokens: "
        f"{data['total_tokens']:,}"
    )

    print(
        f"Total sequences: "
        f"{data['total_sequences']:,}"
    )

    print(
        f"Sequence length: "
        f"{data['sequence_length']}"
    )

    if not data["sequences"]:

        print(
            "\nNo sequences available."
        )

        return

    print("\nDocuments:")

    for document in data["documents"]:

        print(
            f"  {document['file']:<30} "
            f"tokens={document['tokens']:<5} "
            f"sequences={document['sequences']}"
        )

    example = data["sequences"][0]

    input_ids = example["input"]
    target_ids = example["target"]

    input_text = tokenizer.decode(
        input_ids
    )

    target_text = tokenizer.decode(
        target_ids
    )

    print("\n" + "-" * 60)
    print("FIRST SEQUENCE")
    print("-" * 60)

    print(
        "\nInput IDs:"
    )

    print(input_ids)

    print(
        "\nTarget IDs:"
    )

    print(target_ids)

    print(
        "\nDecoded input:"
    )

    print(
        repr(input_text)
    )

    print(
        "\nDecoded target:"
    )

    print(
        repr(target_text)
    )

    print("\n" + "-" * 60)
    print("SHIFT CHECK")
    print("-" * 60)

    correct_shift = (
        input_ids[1:]
        ==
        target_ids[:-1]
    )

    print(
        f"\nTarget is shifted by one token: "
        f"{correct_shift}"
    )

    print(
        f"Input length:  {len(input_ids)}"
    )

    print(
        f"Target length: {len(target_ids)}"
    )


def main():

    print("=" * 60)
    print("NovaLM Token Dataset Inspector V2")
    print("=" * 60)

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_FILE)
    )

    print(
        f"\nVocabulary size: "
        f"{tokenizer.get_vocab_size():,}"
    )

    for split in [
        "train",
        "validation",
        "test",
    ]:

        inspect_split(
            tokenizer,
            split
        )

    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()