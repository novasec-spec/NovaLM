from pathlib import Path

from tokenizers import Tokenizer


# ============================================================
# NovaLM Tokenizer Test V1
# ============================================================

TOKENIZER_FILE = Path(
    "tokenizer/tokenizer.json"
)


def main():

    print("=" * 60)
    print("NovaLM Tokenizer Test V1")
    print("=" * 60)

    if not TOKENIZER_FILE.exists():

        print(
            "\nTokenizer does not exist."
        )

        print(
            "Run:"
        )

        print(
            "python scripts/train_tokenizer.py"
        )

        return

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_FILE)
    )

    print(
        f"\nVocabulary size: "
        f"{tokenizer.get_vocab_size():,}"
    )

    test_texts = [
        "Hello world.",
        "Programming is useful.",
        "Artificial intelligence learns patterns from data.",
        "She is learning mathematics.",
        "He writes programs.",
    ]

    print("\n" + "=" * 60)
    print("ENCODING TEST")
    print("=" * 60)

    for text in test_texts:

        encoding = tokenizer.encode(text)

        print("\nOriginal:")
        print(text)

        print("\nTokens:")
        print(encoding.tokens)

        print("\nIDs:")
        print(encoding.ids)

        decoded = tokenizer.decode(
            encoding.ids
        )

        print("\nDecoded:")
        print(decoded)

        print("-" * 60)


if __name__ == "__main__":
    main()