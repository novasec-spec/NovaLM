"""
Create a validation split from existing disk-backed token shards.

No raw data is re-read.
No tokenization is performed.
Only the existing .bin token shards are reorganized.
"""

from pathlib import Path
import json
import shutil
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

TOKENS_DIR = ROOT / "data" / "processed" / "tokens"
MANIFEST_PATH = TOKENS_DIR / "manifest.json"

VALIDATION_FRACTION = 0.05
OUTPUT_SHARD_SIZE = 50_000


def main():
    print("=" * 70)
    print("NovaLM Validation Split Creator")
    print("=" * 70)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_info = manifest["splits"]["train"]

    if manifest["splits"].get("val", {}).get("sequences", 0) > 0:
        raise RuntimeError(
            "Validation data already exists. Refusing to split again."
        )

    sequence_length = manifest["sequence_length"]
    sequence_width = sequence_length + 1
    dtype = np.dtype(manifest["dtype"])

    total_sequences = train_info["sequences"]

    validation_sequences = int(total_sequences * VALIDATION_FRACTION)
    training_sequences = total_sequences - validation_sequences

    print(f"Total sequences:       {total_sequences:,}")
    print(f"Validation fraction:   {VALIDATION_FRACTION:.1%}")
    print(f"Training sequences:    {training_sequences:,}")
    print(f"Validation sequences:  {validation_sequences:,}")
    print(f"Sequence width:        {sequence_width}")
    print(f"Dtype:                 {dtype}")
    print()

    # Validation comes from the END of the dataset.
    validation_start = training_sequences

    # We will only rewrite the affected shards.
    # Everything before the validation boundary remains untouched.
    val_buffer = []
    val_buffer_sequences = 0
    val_shards = []
    val_shard_index = 0

    def flush_validation():
        nonlocal val_buffer
        nonlocal val_buffer_sequences
        nonlocal val_shard_index

        if val_buffer_sequences == 0:
            return

        array = np.concatenate(val_buffer)

        expected_values = val_buffer_sequences * sequence_width

        if array.size != expected_values:
            raise RuntimeError(
                f"Validation shard size mismatch: "
                f"{array.size:,} values != "
                f"{expected_values:,} expected"
            )

        filename = f"val-{val_shard_index:05d}.bin"
        path = TOKENS_DIR / filename

        array.astype(dtype, copy=False).tofile(path)

        val_shards.append({
            "file": filename,
            "sequences": val_buffer_sequences,
        })

        print(
            f"  Created {filename}: "
            f"{val_buffer_sequences:,} sequences"
        )

        val_shard_index += 1
        val_buffer = []
        val_buffer_sequences = 0

    global_position = 0

    affected_train_files = []

    for shard in train_info["shards"]:
        filename = shard["file"]
        shard_sequences = shard["sequences"]

        source_path = TOKENS_DIR / filename

        if not source_path.exists():
            raise FileNotFoundError(
                f"Missing token shard: {source_path}"
            )

        shard_start = global_position
        shard_end = global_position + shard_sequences

        # No validation data in this shard.
        if shard_end <= validation_start:
            global_position = shard_end
            continue

        print(
            f"Processing {filename} "
            f"({shard_sequences:,} sequences)..."
        )

        raw = np.memmap(
            source_path,
            dtype=dtype,
            mode="r",
        )

        expected_values = shard_sequences * sequence_width

        if raw.size != expected_values:
            raise RuntimeError(
                f"{filename}: "
                f"{raw.size:,} values found, "
                f"{expected_values:,} expected"
            )

        data = raw.reshape(shard_sequences, sequence_width)

        # Number of sequences that remain training data.
        train_count = max(
            0,
            min(
                shard_sequences,
                validation_start - shard_start,
            ),
        )

        # Everything after train_count becomes validation.
        val_count = shard_sequences - train_count

        # Boundary shard: keep its training portion.
        if train_count > 0 and train_count < shard_sequences:
            affected_train_files.append(filename)

            temp_path = source_path.with_suffix(".tmp")

            train_part = np.asarray(data[:train_count])

            train_part.astype(dtype, copy=False).tofile(temp_path)

            temp_path.replace(source_path)

            print(
                f"  Kept {train_count:,} training sequences "
                f"in {filename}"
            )

        # Entire shard becomes validation.
        elif train_count == 0:
            affected_train_files.append(filename)

        # Add validation portion to output buffer.
        if val_count > 0:
            val_part = np.asarray(data[train_count:])

            offset = 0

            while offset < val_count:
                remaining_capacity = (
                    OUTPUT_SHARD_SIZE - val_buffer_sequences
                )

                take = min(
                    remaining_capacity,
                    val_count - offset,
                )

                chunk = val_part[offset:offset + take]

                val_buffer.append(
                    chunk.reshape(-1)
                )

                val_buffer_sequences += take
                offset += take

                if val_buffer_sequences == OUTPUT_SHARD_SIZE:
                    flush_validation()

        del data
        del raw

        global_position = shard_end

    flush_validation()

    if len(val_shards) == 0:
        raise RuntimeError("No validation shards were created.")

    actual_validation = sum(
        shard["sequences"]
        for shard in val_shards
    )

    if actual_validation != validation_sequences:
        raise RuntimeError(
            f"Validation count mismatch: "
            f"{actual_validation:,} != "
            f"{validation_sequences:,}"
        )

    # Delete the original shards that are now completely validation data.
    # The boundary shard has already been rewritten.
    for filename in affected_train_files:
        path = TOKENS_DIR / filename

        # Boundary shard still exists and contains training data.
        # Fully-validation shards need removal.
        shard_number = int(
            filename.replace("train-", "").replace(".bin", "")
        )

        shard = next(
            s for s in train_info["shards"]
            if s["file"] == filename
        )

        shard_start = sum(
            s["sequences"]
            for s in train_info["shards"][:shard_number]
        )

        shard_end = shard_start + shard["sequences"]

        if shard_start >= validation_start and path.exists():
            path.unlink()
            print(f"  Removed old validation source: {filename}")

    # Rebuild training manifest.
    new_train_shards = []

    for shard in train_info["shards"]:
        filename = shard["file"]

        # Determine whether this shard still contains training data.
        shard_start = sum(
            s["sequences"]
            for s in train_info["shards"]
            if s["file"] < filename
        )

        shard_end = shard_start + shard["sequences"]

        if shard_start >= training_sequences:
            continue

        if shard_end <= training_sequences:
            new_train_shards.append(shard)
        else:
            # Boundary shard.
            remaining = training_sequences - shard_start

            new_train_shards.append({
                "file": filename,
                "sequences": remaining,
            })

    manifest["splits"]["train"] = {
        "shards": new_train_shards,
        "sequences": training_sequences,
    }

    manifest["splits"]["val"] = {
        "shards": val_shards,
        "sequences": actual_validation,
    }

    manifest["format_version"] = 3

    temp_manifest = MANIFEST_PATH.with_suffix(".json.tmp")

    with open(temp_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    temp_manifest.replace(MANIFEST_PATH)

    print()
    print("=" * 70)
    print("VALIDATION SPLIT COMPLETE")
    print("=" * 70)
    print(f"Training sequences:    {training_sequences:,}")
    print(f"Validation sequences:  {actual_validation:,}")
    print(f"Training shards:       {len(new_train_shards)}")
    print(f"Validation shards:     {len(val_shards)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
