from pathlib import Path
import re


# ============================================================
# NovaLM - Data Cleaner V1
# ============================================================

RAW_DATA_DIR = Path("data/raw")
CLEANED_DATA_DIR = Path("data/cleaned")


def normalize_text(text):
    """
    Normalize raw text without changing its meaning.
    """

    # Normalize Windows/Mac line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Replace tabs with spaces
    text = text.replace("\t", " ")

    # Remove trailing spaces from each line
    lines = []

    for line in text.split("\n"):
        line = line.strip()

        if line:
            lines.append(line)

    text = "\n".join(lines)

    # Collapse repeated spaces
    text = re.sub(r"[ ]+", " ", text)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_file(input_path, output_path):
    """
    Clean one text file and save the result.
    """

    text = input_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    cleaned = normalize_text(text)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        cleaned,
        encoding="utf-8",
    )

    return len(text), len(cleaned)


def main():

    print("=" * 60)
    print("NovaLM Data Cleaner V1")
    print("=" * 60)

    files = list(RAW_DATA_DIR.rglob("*.txt"))

    if not files:
        print("\nNo .txt files found in data/raw/")
        return

    total_before = 0
    total_after = 0

    for input_path in files:

        relative_path = input_path.relative_to(RAW_DATA_DIR)

        output_path = CLEANED_DATA_DIR / relative_path

        before, after = clean_file(
            input_path,
            output_path,
        )

        total_before += before
        total_after += after

        print(f"\nFile: {relative_path}")
        print(f"Before: {before:,} characters")
        print(f"After:  {after:,} characters")

    print("\n" + "=" * 60)
    print("CLEANING COMPLETE")
    print("=" * 60)

    print(f"Total before: {total_before:,}")
    print(f"Total after:  {total_after:,}")

    removed = total_before - total_after

    print(f"Characters changed/removed: {removed:,}")

    print("\nCleaned files are located in:")
    print("data/cleaned/")


if __name__ == "__main__":
    main()