from pathlib import Path
from collections import Counter
import hashlib
import re


# ============================================================
# NovaLM - Data Quality Checker V1
# ============================================================

DATA_DIR = Path("data/cleaned")


# Minimum requirements
MIN_CHARACTERS = 100
MIN_WORDS = 20

# Maximum ratio of non-letter/non-number symbols
MAX_SYMBOL_RATIO = 0.30


def calculate_hash(text):
    """
    Create a unique fingerprint for a document.
    """

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def count_words(text):
    """
    Count words using a simple regex.
    """

    return re.findall(r"\b[\w'-]+\b", text)


def symbol_ratio(text):
    """
    Calculate how much of the document consists
    of unusual symbols.
    """

    if not text:
        return 1.0

    normal_characters = sum(
        char.isalnum() or char.isspace() or char in ".,!?;:'\"()-"
        for char in text
    )

    unusual_characters = len(text) - normal_characters

    return unusual_characters / len(text)


def check_document(text):
    """
    Check whether a document meets our basic
    quality requirements.
    """

    reasons = []

    words = count_words(text)

    if len(text) < MIN_CHARACTERS:
        reasons.append(
            f"too_short_characters:{len(text)}"
        )

    if len(words) < MIN_WORDS:
        reasons.append(
            f"too_few_words:{len(words)}"
        )

    ratio = symbol_ratio(text)

    if ratio > MAX_SYMBOL_RATIO:
        reasons.append(
            f"too_many_symbols:{ratio:.2f}"
        )

    return reasons


def main():

    print("=" * 60)
    print("NovaLM Data Quality Checker V1")
    print("=" * 60)

    files = list(DATA_DIR.rglob("*.txt"))

    if not files:
        print("\nNo cleaned text files found.")
        print("Run the cleaning stage first.")
        return

    hashes = {}

    accepted = 0
    rejected = 0
    duplicates = 0

    for file in files:

        text = file.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        document_hash = calculate_hash(text)

        print(f"\nFile: {file}")

        # ----------------------------------------------------
        # Duplicate detection
        # ----------------------------------------------------

        if document_hash in hashes:

            print("Status: REJECTED")
            print(
                f"Reason: duplicate of {hashes[document_hash]}"
            )

            duplicates += 1
            rejected += 1

            continue

        hashes[document_hash] = file

        # ----------------------------------------------------
        # Quality checks
        # ----------------------------------------------------

        reasons = check_document(text)

        if reasons:

            print("Status: REJECTED")

            for reason in reasons:
                print(f"Reason: {reason}")

            rejected += 1

        else:

            print("Status: ACCEPTED")

            accepted += 1

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("QUALITY REPORT")
    print("=" * 60)

    print(f"Documents checked: {len(files)}")
    print(f"Accepted:          {accepted}")
    print(f"Rejected:          {rejected}")
    print(f"Duplicates:        {duplicates}")

    if files:
        acceptance_rate = accepted / len(files) * 100
        print(f"Acceptance rate:   {acceptance_rate:.2f}%")

    print("=" * 60)


if __name__ == "__main__":
    main()