from pathlib import Path
import json
import random

# ============================================================
# NovaLM Dataset Splitter V3
# ============================================================

CLEANED_DATA_DIR = Path("data/cleaned")
OUTPUT_DIR = Path("data/processed/splits")

TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10

SEED = 42


def load_documents():
    """Load text documents, splitting on delimiters if present."""
    files = sorted(CLEANED_DATA_DIR.rglob("*.txt"))

    documents = []

    for file in files:
        text = file.read_text(
            encoding="utf-8",
            errors="ignore"
        ).strip()

        if not text:
            continue

        if "<|endoftext|>" in text:
            stories = [s.strip() for s in text.split("<|endoftext|>") if s.strip()]
            for idx, story in enumerate(stories):
                documents.append({
                    "source": f"{file}#story_{idx+1}",
                    "text": story,
                })
        else:
            documents.append({
                "source": str(file),
                "text": text,
            })

    # If still fewer than 3 documents, chunk paragraphs to allow splitting
    if len(documents) < 3 and documents:
        chunked = []
        for doc in documents:
            paras = [p.strip() for p in doc["text"].split("\n\n") if len(p.strip()) > 50]
            if len(paras) >= 3:
                for i, p in enumerate(paras):
                    chunked.append({"source": f"{doc['source']}#p{i+1}", "text": p})
            else:
                chunked.append(doc)
        if len(chunked) >= 3:
            documents = chunked

    return documents


def calculate_split_counts(total):
    """
    Calculate train/validation/test document counts.

    For tiny datasets, guarantee at least one validation
    and one test document when possible.
    """

    if total < 3:
        raise ValueError(
            "At least 3 documents are required "
            "for train/validation/test splits."
        )

    validation_count = max(
        1,
        round(total * VALIDATION_RATIO)
    )

    test_count = max(
        1,
        round(total * TEST_RATIO)
    )

    train_count = total - validation_count - test_count

    if train_count < 1:
        raise ValueError(
            "Not enough documents to create all three splits."
        )

    return train_count, validation_count, test_count


def split_documents(documents):
    """Randomly divide complete documents into three splits."""

    random.seed(SEED)

    documents = documents.copy()
    random.shuffle(documents)

    total = len(documents)

    train_count, validation_count, test_count = (
        calculate_split_counts(total)
    )

    train_documents = documents[:train_count]

    validation_end = train_count + validation_count

    validation_documents = documents[
        train_count:validation_end
    ]

    test_documents = documents[
        validation_end:
    ]

    return {
        "train": train_documents,
        "validation": validation_documents,
        "test": test_documents,
    }


def save_split(name, documents):
    """Save documents belonging to a dataset split."""

    split_dir = OUTPUT_DIR / name
    split_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest = []

    for document in documents:

        source_path = Path(document["source"])

        output_file = split_dir / source_path.name

        output_file.write_text(
            document["text"],
            encoding="utf-8"
        )

        manifest.append({
            "source": document["source"],
            "output": str(output_file),
            "characters": len(document["text"]),
            "words": len(document["text"].split()),
        })

    return manifest


def main():

    print("=" * 60)
    print("NovaLM Dataset Splitter V3")
    print("=" * 60)

    if not CLEANED_DATA_DIR.exists():
        print("\nCleaned data directory does not exist.")
        print("Run:")
        print("python scripts/clean_data.py")
        return

    documents = load_documents()

    if not documents:
        print("\nNo cleaned documents found.")
        return

    print(f"\nDocuments found: {len(documents)}")

    try:
        splits = split_documents(documents)

    except ValueError as error:
        print(f"\nERROR: {error}")
        return

    # Remove previous split files.
    if OUTPUT_DIR.exists():

        for path in OUTPUT_DIR.rglob("*"):

            if path.is_file():
                path.unlink()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest = {
        "version": "3.0",
        "seed": SEED,
        "ratios": {
            "train": TRAIN_RATIO,
            "validation": VALIDATION_RATIO,
            "test": TEST_RATIO,
        },
        "splits": {},
    }

    print("\n" + "=" * 60)
    print("CREATING SPLITS")
    print("=" * 60)

    for split_name, documents_in_split in splits.items():

        split_manifest = save_split(
            split_name,
            documents_in_split
        )

        manifest["splits"][split_name] = split_manifest

        print(
            f"{split_name.capitalize():<12}: "
            f"{len(documents_in_split)} documents"
        )

        for document in documents_in_split:

            print(
                f"  - {Path(document['source']).name}"
            )

    manifest_file = OUTPUT_DIR / "manifest.json"

    manifest_file.write_text(
        json.dumps(
            manifest,
            indent=4
        ),
        encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("DATASET SPLIT COMPLETE")
    print("=" * 60)

    print(f"Total documents:  {len(documents)}")
    print(
        f"Train:            {len(splits['train'])}"
    )
    print(
        f"Validation:       {len(splits['validation'])}"
    )
    print(
        f"Test:             {len(splits['test'])}"
    )

    print(f"\nRandom seed:      {SEED}")
    print(f"\nOutput:           {OUTPUT_DIR}")
    print(f"Manifest:         {manifest_file}")

    print("=" * 60)


if __name__ == "__main__":
    main()