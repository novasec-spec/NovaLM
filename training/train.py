"""
NovaLM Continuous Training Engine

Features:
- Truly continuous, resumable training preserving model, optimizer, scheduler,
  global step, and RNG states.
- Atomic checkpoint saving (prevents corruption on sudden termination or power loss).
- Graceful KeyboardInterrupt (Ctrl+C) handling to preserve progress.
- Gradient clipping and cosine learning-rate scheduling with linear warmup.
- Perplexity tracking and comprehensive CSV history logging.
- Memory-mapped zero-RAM dataloading from data/processed/tokens/.
"""

from pathlib import Path
from datetime import datetime
import argparse
import csv
import hashlib
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from tokenizers import Tokenizer

from configs import default_config
from dataset import get_dataloader
from model.novalm import NovaLM
from training.loss import NextTokenLoss


# ============================================================
# DEVICE SELECTION
# ============================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================
# REPRODUCIBILITY & RNG STATE
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_rng_states():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_states(states: dict):
    if not states:
        return
    if "python" in states and states["python"]:
        random.setstate(states["python"])
    if "numpy" in states and states["numpy"]:
        np.random.set_state(states["numpy"])
    if "torch" in states and states["torch"] is not None:
        torch.set_rng_state(states["torch"])
    if "cuda" in states and states["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["cuda"])


# ============================================================
# ATOMIC CHECKPOINT MANAGEMENT
# ============================================================

def save_atomic_checkpoint(checkpoint: dict, target_path: Path):
    """
    Safely save checkpoint by writing to a temporary file first,
    then atomically swapping to prevent corrupted files on interruption.
    """
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_name(f"{target_path.name}.tmp")

    torch.save(checkpoint, temp_path)
    os.replace(temp_path, target_path)


# ============================================================
# TOKENIZER IDENTITY
# ============================================================

def sha256_file(path: Path) -> str:
    """Return the SHA256 fingerprint of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

def get_lr(step: int, config) -> float:
    """Cosine learning rate schedule with linear warmup."""
    warmup_steps = config.training.warmup_steps
    max_steps = config.training.max_steps
    base_lr = config.training.learning_rate
    min_lr = config.training.min_learning_rate

    # Linear warmup
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)

    # Beyond max steps, hold min_lr
    if step > max_steps:
        return min_lr

    # Cosine decay down to min_lr
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (base_lr - min_lr)


# ============================================================
# HISTORY LOGGING
# ============================================================

def log_training_metrics(
    history_file: Path,
    step: int,
    epoch: int,
    train_loss: float,
    train_ppl: float,
    val_loss: float,
    val_ppl: float,
    learning_rate: float,
):
    history_file.parent.mkdir(parents=True, exist_ok=True)
    file_exists = history_file.exists()

    with open(history_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "step",
                "epoch",
                "train_loss",
                "train_perplexity",
                "val_loss",
                "val_perplexity",
                "learning_rate",
                "timestamp",
            ])
        writer.writerow([
            step,
            epoch,
            f"{train_loss:.6f}",
            f"{train_ppl:.4f}" if train_ppl is not None else "",
            f"{val_loss:.6f}" if val_loss is not None else "",
            f"{val_ppl:.4f}" if val_ppl is not None else "",
            f"{learning_rate:.8f}",
            datetime.now().isoformat(),
        ])


# ============================================================
# EVALUATION LOOP
# ============================================================

@torch.no_grad()
def evaluate(model, val_loader, loss_fn, device, max_batches: int = 25):
    """Evaluate model on validation split, computing loss and perplexity."""
    model.eval()
    total_loss = 0.0
    batch_count = 0

    for idx, (inputs, targets) in enumerate(val_loader):
        if max_batches and idx >= max_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        total_loss += loss.item()
        batch_count += 1

    model.train()
    if batch_count == 0:
        return float("inf"), float("inf")

    avg_loss = total_loss / batch_count
    perplexity = math.exp(min(avg_loss, 20.0))
    return avg_loss, perplexity


# ============================================================
# MAIN TRAINING CONTROLLER
# ============================================================

def train(
    max_steps: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    reset_training: bool = False,
):
    config = default_config
    if max_steps is not None:
        config.training.max_steps = max_steps
    if batch_size is not None:
        config.training.batch_size = batch_size
    if learning_rate is not None:
        config.training.learning_rate = learning_rate

    print("=" * 70)
    print("NovaLM Continuous Training Engine")
    print("=" * 70)

    device = get_device()
    print(f"Device: {device}")
    set_seed(config.training.seed)

    # Load dataloaders
    try:
        train_loader = get_dataloader(
            split="train",
            batch_size=config.training.batch_size,
            shuffle=True,
            sequence_length=config.model.max_sequence_length,
        )
        val_loader = get_dataloader(
            split="val",
            batch_size=config.training.batch_size,
            shuffle=False,
            sequence_length=config.model.max_sequence_length,
        )
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        return

    # Load tokenizer to verify vocabulary size
    if config.tokenizer_file.exists():
        tokenizer = Tokenizer.from_file(str(config.tokenizer_file))
        vocab_size = tokenizer.get_vocab_size()
    else:
        vocab_size = config.model.vocab_size

    config.model.vocab_size = vocab_size

    if not config.tokenizer_file.exists():
        raise FileNotFoundError(
            f"Tokenizer file not found: {config.tokenizer_file}"
        )

    tokenizer_sha256 = sha256_file(config.tokenizer_file)

    tokenizer_identity = {
        "path": str(config.tokenizer_file),
        "sha256": tokenizer_sha256,
        "vocab_size": vocab_size,
    }

    print(f"Vocabulary size:         {vocab_size:,}")
    print(f"Tokenizer SHA256:        {tokenizer_sha256}")
    print(f"Context sequence length: {config.model.max_sequence_length}")
    print(f"Training sequences:      {len(train_loader.dataset):,}")
    print(f"Validation sequences:    {len(val_loader.dataset):,}")

    # Initialize model
    model = NovaLM(
        vocab_size=vocab_size,
        max_sequence_length=config.model.max_sequence_length,
        embedding_dim=config.model.embedding_dim,
        num_heads=config.model.num_heads,
        feed_forward_dim=config.model.feed_forward_dim,
        num_layers=config.model.num_layers,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters:        {param_count:,}")

    loss_fn = NextTokenLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=(0.9, 0.95),
    )

    # State variables
    global_step = 0
    current_epoch = 1
    best_val_loss = float("inf")
    best_val_ppl = float("inf")
    resumed = False

    # Check for existing checkpoint
    latest_ckpt = config.latest_checkpoint
    if latest_ckpt.exists() and not reset_training:
        print("\n" + "=" * 70)
        print("RESUMING FROM PREVIOUS CHECKPOINT")
        print("=" * 70)
        try:
            checkpoint = torch.load(latest_ckpt, map_location=device, weights_only=False)

            saved_tokenizer = checkpoint.get("tokenizer_identity")

            if saved_tokenizer is None:
                raise RuntimeError(
                    "Checkpoint has no tokenizer identity. "
                    "This is an older/incompatible checkpoint. "
                    "Archive it and start a fresh training lineage with --reset."
                )

            if saved_tokenizer.get("sha256") != tokenizer_sha256:
                raise RuntimeError(
                    "TOKENIZER MISMATCH!\n"
                    f"Checkpoint tokenizer: {saved_tokenizer.get('sha256')}\n"
                    f"Current tokenizer:    {tokenizer_sha256}\n"
                    "Refusing to load incompatible checkpoint."
                )

            if saved_tokenizer.get("vocab_size") != vocab_size:
                raise RuntimeError(
                    "VOCABULARY MISMATCH!\n"
                    f"Checkpoint vocab: {saved_tokenizer.get('vocab_size')}\n"
                    f"Current vocab:    {vocab_size}\n"
                    "Refusing to load incompatible checkpoint."
                )

            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            global_step = checkpoint.get("global_step", 0)
            current_epoch = checkpoint.get("epoch", 1)
            best_val_loss = checkpoint.get("best_validation_loss", float("inf"))
            best_val_ppl = checkpoint.get("best_validation_ppl", math.exp(min(best_val_loss, 20.0)))

            if "rng_states" in checkpoint:
                restore_rng_states(checkpoint["rng_states"])

            resumed = True
            print(f"Checkpoint restored:     {latest_ckpt}")
            print(f"Resumed global step:     {global_step:,}")
            print(f"Resumed epoch:           {current_epoch}")
            print(f"Best validation loss:    {best_val_loss:.4f} (Perplexity: {best_val_ppl:.2f})")
            print("✓ Optimizer state successfully restored (continuous training active)")
        except Exception as e:
            print(f"[!] Warning: Failed to load checkpoint {latest_ckpt}: {e}")
            print("Starting fresh training from step 0.")
    else:
        if reset_training:
            print("\n[!] Fresh training explicitly requested (ignoring previous checkpoints).")
        else:
            print("\nNo previous checkpoint found. Initializing fresh model at step 0.")

    print("=" * 70)
    print("TRAINING STARTED (Press Ctrl+C to halt and safely save checkpoint)")
    print("=" * 70)

    model.train()
    running_loss = 0.0
    steps_in_epoch = 0

    target_max_steps = global_step + config.training.max_steps if resumed else config.training.max_steps

    def build_checkpoint_dict(val_loss_val=None, val_ppl_val=None):
        return {
            "epoch": current_epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": running_loss / max(1, steps_in_epoch),
            "validation_loss": val_loss_val if val_loss_val is not None else best_val_loss,
            "best_validation_loss": best_val_loss,
            "best_validation_ppl": best_val_ppl,
            "model_config": config.model.to_dict(),
            "training_config": config.training.to_dict(),
            "tokenizer_identity": tokenizer_identity,
            "rng_states": get_rng_states(),
            "timestamp": datetime.now().isoformat(),
        }

    try:
        while global_step < target_max_steps:
            for inputs, targets in train_loader:
                if global_step >= target_max_steps:
                    break

                inputs, targets = inputs.to(device), targets.to(device)

                # Set scheduled learning rate
                lr = get_lr(global_step, config)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                # Forward pass
                optimizer.zero_grad()
                logits = model(inputs)
                loss = loss_fn(logits, targets)

                # Backward pass
                loss.backward()

                # Gradient clipping
                if config.training.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config.training.gradient_clip_norm,
                    )

                optimizer.step()

                loss_val = loss.item()
                running_loss += loss_val
                steps_in_epoch += 1
                global_step += 1

                # Progress output every 10 steps
                if global_step % 10 == 0:
                    cur_ppl = math.exp(min(loss_val, 20.0))
                    print(
                        f"Step {global_step:06d} | "
                        f"Epoch {current_epoch:03d} | "
                        f"Loss: {loss_val:.4f} | "
                        f"PPL: {cur_ppl:.2f} | "
                        f"LR: {lr:.2e}"
                    )

                # Periodic Evaluation
                if global_step % config.training.eval_interval_steps == 0:
                    val_loss, val_ppl = evaluate(
                        model=model,
                        val_loader=val_loader,
                        loss_fn=loss_fn,
                        device=device,
                        max_batches=config.training.eval_batches,
                    )

                    train_loss_avg = running_loss / max(1, steps_in_epoch)
                    train_ppl_avg = math.exp(min(train_loss_avg, 20.0))

                    print("\n" + "-" * 70)
                    print(
                        f"EVALUATION @ Step {global_step:,} | "
                        f"Train Loss: {train_loss_avg:.4f} (PPL: {train_ppl_avg:.2f}) | "
                        f"Val Loss: {val_loss:.4f} (PPL: {val_ppl:.2f})"
                    )

                    # Log to CSV
                    log_training_metrics(
                        history_file=config.history_file,
                        step=global_step,
                        epoch=current_epoch,
                        train_loss=train_loss_avg,
                        train_ppl=train_ppl_avg,
                        val_loss=val_loss,
                        val_ppl=val_ppl,
                        learning_rate=lr,
                    )

                    # Check for new best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_val_ppl = val_ppl
                        save_atomic_checkpoint(
                            build_checkpoint_dict(val_loss, val_ppl),
                            config.best_checkpoint,
                        )
                        print(f"★ New Best Model Saved: {config.best_checkpoint.name} (Val Loss: {val_loss:.4f})")
                    print("-" * 70 + "\n")

                # Periodic Checkpoint Saving
                if global_step % config.training.checkpoint_interval_steps == 0:
                    save_atomic_checkpoint(
                        build_checkpoint_dict(),
                        config.latest_checkpoint,
                    )
                    # Periodic step snapshot
                    step_ckpt = config.checkpoint_dir / f"step_{global_step:06d}.pt"
                    save_atomic_checkpoint(build_checkpoint_dict(), step_ckpt)
                    print(f"✓ Latest checkpoint saved atomically: {config.latest_checkpoint.name}")

            current_epoch += 1
            steps_in_epoch = 0

    except KeyboardInterrupt:
        print("\n" + "!" * 70)
        print("[Ctrl+C] Training interruption detected. Safely saving checkpoint...")
        print("!" * 70)
        save_atomic_checkpoint(build_checkpoint_dict(), config.latest_checkpoint)
        print(f"✓ Checkpoint safely preserved at {config.latest_checkpoint}")
        print("You can resume anytime by running: python training/train.py")
        return

    # Final completion save
    save_atomic_checkpoint(build_checkpoint_dict(), config.latest_checkpoint)
    print("\n" + "=" * 70)
    print("TRAINING SESSION COMPLETE")
    print("=" * 70)
    print(f"Total steps trained:   {global_step:,}")
    print(f"Latest checkpoint:     {config.latest_checkpoint}")
    print(f"Best checkpoint:       {config.best_checkpoint}")
    print(f"Best validation loss:  {best_val_loss:.4f} (PPL: {best_val_ppl:.2f})")
    print(f"Metrics history saved: {config.history_file}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="NovaLM Continuous Training")
    parser.add_argument("--steps", type=int, default=None, help="Max steps to train")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--reset", action="store_true", help="Reset and start fresh from step 0")
    args = parser.parse_args()

    train(
        max_steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        reset_training=args.reset,
    )


if __name__ == "__main__":
    main()