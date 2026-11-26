from pathlib import Path
from collections import Counter
import re


# ============================================================
# NovaLM - Memory-Efficient Data Inspector V2
# ============================================================

RAW_DATA_DIR = Path("data/raw")


def inspect_file(file_path: Path, max_vocab_tokens: int = 100000):
    """Inspect a file line-by-line without high RAM usage."""
    char_count = 0
    word_count = 0
    vocab = Counter()
    sampled_words = 0

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            char_count += len(line)
            words = re.findall(r"\b[\w'-]+\b", line.lower())
            word_count += len(words)
            if sampled_words < max_vocab_tokens:
                vocab.update(words)
                sampled_words += len(words)

    return {
        "path": file_path,
        "characters": char_count,
        "words": word_count,
        "vocab": vocab,
    }


def main():
    print("=" * 60)
    print("NovaLM Data Inspector V2")
    print("=" * 60)

    if not RAW_DATA_DIR.exists():
        print(f"\nNo {RAW_DATA_DIR} directory found.")
        return

    files = sorted(RAW_DATA_DIR.rglob("*.txt"))

    if not files:
        print("\nNo .txt files found.")
        print(f"Put some text files inside: {RAW_DATA_DIR}")
        return

    total_characters = 0
    total_words = 0
    overall_vocab = Counter()

    print(f"\nFiles found: {len(files)}\n")

    for file_path in files:
        stats = inspect_file(file_path)
        total_characters += stats["characters"]
        total_words += stats["words"]
        overall_vocab.update(stats["vocab"])

        print(f"File:       {file_path.name}")
        print(f"Size:       {file_path.stat().st_size / (1024*1024):.2f} MB")
        print(f"Characters: {stats['characters']:,}")
        print(f"Words:      {stats['words']:,}")
        print("-" * 60)

    print("\n" + "=" * 60)
    print("TOTAL DATASET")
    print("=" * 60)

    print(f"Files:       {len(files):,}")
    print(f"Characters:  {total_characters:,}")
    print(f"Words:       {total_words:,}")
    print(f"Unique sample vocabulary: {len(overall_vocab):,}")

    print("\nTop common words:")
    for word, count in overall_vocab.most_common(20):
        print(f"  {word:<20} {count:,}")


if __name__ == "__main__":
    main()