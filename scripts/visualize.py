"""NovaLM training metrics and dataset visualization.

Reads the current continuous-training history and disk-backed token manifest.
Works with the NovaLM manifest v2 format:
  splits -> train/val -> shards/sequences
  source_files -> processed raw sources
  tokenizer -> sha256/vocab_size
"""

from pathlib import Path
import argparse
import csv
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs import default_config


def _safe_float(value):
    if value in (None, "", "None"):
        return None
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def load_history(history_file: Path):
    rows = []
    if not history_file.exists():
        return rows
    with open(history_file, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                step = int(row.get("step", 0))
            except (TypeError, ValueError):
                continue
            rows.append({
                "step": step,
                "epoch": int(row.get("epoch", 1) or 1),
                "train_loss": _safe_float(row.get("train_loss")),
                "train_perplexity": _safe_float(row.get("train_perplexity")),
                "val_loss": _safe_float(row.get("val_loss")),
                "val_perplexity": _safe_float(row.get("val_perplexity")),
                "learning_rate": _safe_float(row.get("learning_rate")),
            })
    return rows


def load_manifest(manifest_file: Path):
    if not manifest_file.exists():
        return {}
    with open(manifest_file, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_training_metrics(history_file=None, manifest_file=None, output_png=None):
    history_file = Path(history_file) if history_file else default_config.history_file
    manifest_file = Path(manifest_file) if manifest_file else default_config.data.manifest_file
    output_png = Path(output_png) if output_png else default_config.plot_file

    history = load_history(history_file)
    if not history:
        print(f"No training history found at {history_file}.")
        return False

    manifest = load_manifest(manifest_file)
    steps = [r["step"] for r in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("NovaLM Continuous Training Metrics", fontsize=16, fontweight="bold", y=0.98)

    # 1. Loss
    ax = axes[0, 0]
    train_loss = [r["train_loss"] for r in history]
    val_pairs = [(r["step"], r["val_loss"]) for r in history if r["val_loss"] is not None]
    ax.plot(steps, train_loss, label="Train Loss", linewidth=2)
    if val_pairs:
        ax.plot([x for x, _ in val_pairs], [y for _, y in val_pairs], label="Validation Loss", linewidth=2, marker="o", markersize=3)
    ax.set_title("Loss")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Cross Entropy")
    ax.legend()
    ax.grid(True, alpha=0.25)

    # 2. Perplexity
    ax = axes[0, 1]
    train_ppl = [(r["step"], r["train_perplexity"]) for r in history if r["train_perplexity"] is not None]
    val_ppl = [(r["step"], r["val_perplexity"]) for r in history if r["val_perplexity"] is not None]
    if train_ppl:
        ax.plot([x for x, _ in train_ppl], [y for _, y in train_ppl], label="Train PPL", linewidth=2)
    if val_ppl:
        ax.plot([x for x, _ in val_ppl], [y for _, y in val_ppl], label="Validation PPL", linewidth=2, marker="o", markersize=3)
    ax.set_title("Perplexity (Lower is Better)")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("PPL")
    ax.legend()
    ax.grid(True, alpha=0.25)

    # 3. Learning rate
    ax = axes[1, 0]
    lr_pairs = [(r["step"], r["learning_rate"]) for r in history if r["learning_rate"] is not None]
    if lr_pairs:
        ax.plot([x for x, _ in lr_pairs], [y for _, y in lr_pairs], linewidth=2)
    ax.set_title("Learning Rate")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("LR")
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    ax.grid(True, alpha=0.25)

    # 4. Current dataset composition
    ax = axes[1, 1]
    splits = manifest.get("splits", {})
    train_sequences = int(splits.get("train", {}).get("sequences", 0))
    val_sequences = int(splits.get("val", {}).get("sequences", 0))
    labels = ["Train", "Validation"]
    values = [train_sequences, val_sequences]
    if any(values):
        ax.bar(labels, values)
        ax.set_title("Current Token Dataset")
        ax.set_ylabel("Sequences")
        for i, value in enumerate(values):
            ax.text(i, value, f"{value:,}", ha="center", va="bottom", fontsize=9)
    else:
        ax.text(0.5, 0.5, "No token manifest data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    tokenizer = manifest.get("tokenizer", {})
    vocab = tokenizer.get("vocab_size", manifest.get("vocab_size", "?"))
    seq_len = manifest.get("sequence_length", "?")
    shards = len(splits.get("train", {}).get("shards", []))
    source_count = len(manifest.get("source_files", []))
    fig.text(0.01, 0.01, f"Vocab: {vocab} · Context: {seq_len} · Train shards: {shards} · Source files: {source_count}", fontsize=9)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_png, dpi=180)
    plt.close(fig)

    print(f"✓ Visual progress plots saved to: {output_png}")
    return True


def main():
    parser = argparse.ArgumentParser(description="NovaLM Training Progress Visualizer")
    parser.add_argument("--history", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    plot_training_metrics(args.history, args.manifest, args.output)


if __name__ == "__main__":
    main()
