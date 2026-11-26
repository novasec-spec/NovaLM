from pathlib import Path
import hashlib
import json
import re


# ============================================================
# NovaLM - Corpus Builder V1
# ============================================================

CLEANED_DATA_DIR = Path("data/cleaned")
PROCESSED_DATA_DIR = Path("data/processed")

CORPUS_FILE = PROCESSED_DATA_DIR / "corpus.txt"
METADATA_FILE = PROCESSED_DATA_DIR / "corpus_metadata.json"


def normalize_for_hash(text):
    """
    Create a normalized representation used only for
    duplicate detection.

    We do NOT modify the actual training text here.
    """

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def document_hash(text):
    """
    Generate a hash from normalized text.
    """

    normalized = normalize_for_hash(text)

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def count_words(text):
    """
    Count words using a simple regex.
    """

    return re.findall(
        r"\b[\w'-]+\b",
        text,
    )


def main():

    print("=" * 60)
    print("NovaLM Corpus Builder V1")
    print("=" * 60)

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = sorted(
        CLEANED_DATA_DIR.rglob("*.txt")
    )

    if not files:
        print("\nNo cleaned files found.")
        return

    documents = []
    hashes = set()

    total_characters = 0
    total_words = 0
    duplicate_count = 0

    for file in files:

        text = file.read_text(
            encoding="utf-8",
            errors="ignore",
        ).strip()

        if not text:
            continue

        current_hash = document_hash(text)

        # ----------------------------------------------------
        # Duplicate detection
        # ----------------------------------------------------

        if current_hash in hashes:

            duplicate_count += 1

            print(
                f"Duplicate skipped: {file}"
            )

            continue

        hashes.add(current_hash)

        words = count_words(text)

        documents.append({
            "file": str(file),
            "characters": len(text),
            "words": len(words),
            "hash": current_hash,
        })

        total_characters += len(text)
        total_words += len(words)

        print(
            f"Added: {file} "
            f"({len(words):,} words)"
        )

    # --------------------------------------------------------
    # Build corpus
    # --------------------------------------------------------

    corpus_parts = []

    for file_info in documents:

        file = Path(file_info["file"])

        text = file.read_text(
            encoding="utf-8",
            errors="ignore",
        ).strip()

        corpus_parts.append(text)

    corpus = "\n\n".join(
        corpus_parts
    )

    CORPUS_FILE.write_text(
        corpus,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "documents": len(documents),
        "duplicates_removed": duplicate_count,
        "characters": total_characters,
        "words": total_words,
        "sources": documents,
    }

    METADATA_FILE.write_text(
        json.dumps(
            metadata,
            indent=4,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CORPUS CREATED")
    print("=" * 60)

    print(
        f"Documents:          {len(documents):,}"
    )

    print(
        f"Duplicates removed: {duplicate_count:,}"
    )

    print(
        f"Characters:         {total_characters:,}"
    )

    print(
        f"Words:              {total_words:,}"
    )

    print(
        f"\nCorpus:"
        f"\n{CORPUS_FILE}"
    )

    print(
        f"\nMetadata:"
        f"\n{METADATA_FILE}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()