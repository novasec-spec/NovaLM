"""
NovaLM Inference & Text Generation Module

Provides autoregressive token generation with temperature scaling,
top-k filtering, repetition penalties, and checkpoint restoration.
"""

from pathlib import Path
import argparse
import hashlib
import sys

import torch
from tokenizers import Tokenizer

from configs import default_config
from model.novalm import NovaLM


def sha256_file(path: Path) -> str:
    """Return the SHA256 fingerprint of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ============================================================
# DEVICE SELECTION
# ============================================================

def get_device() -> torch.device:
    """Select the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================
# LOAD MODEL & TOKENIZER
# ============================================================

def load_inference_artifacts(
    checkpoint_path: Path = None,
    tokenizer_path: Path = None,
    device: torch.device = None,
):
    """
    Load model weights and tokenizer for inference.
    Prefers best.pt, falls back to latest.pt if available.
    """
    if device is None:
        device = get_device()

    if tokenizer_path is None:
        tokenizer_path = default_config.tokenizer_file

    if checkpoint_path is None:
        if default_config.best_checkpoint.exists():
            checkpoint_path = default_config.best_checkpoint
        elif default_config.latest_checkpoint.exists():
            checkpoint_path = default_config.latest_checkpoint
        else:
            raise FileNotFoundError(
                f"No checkpoint found at:\n"
                f"  - {default_config.best_checkpoint}\n"
                f"  - {default_config.latest_checkpoint}\n"
                f"Train NovaLM first with: python training/train.py"
            )

    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer file not found: {tokenizer_path}\n"
            f"Run data preparation or train tokenizer first."
        )

    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Never allow inference with a tokenizer that does not match the checkpoint.
    saved_identity = checkpoint.get("tokenizer_identity")
    current_sha256 = sha256_file(tokenizer_path)
    current_vocab_size = tokenizer.get_vocab_size()

    if saved_identity is not None:
        saved_sha256 = saved_identity.get("sha256")
        saved_vocab = saved_identity.get("vocab_size")

        if saved_sha256 != current_sha256:
            raise RuntimeError(
                "TOKENIZER MISMATCH!\n"
                f"Checkpoint tokenizer: {saved_sha256}\n"
                f"Current tokenizer:    {current_sha256}\n"
                "Refusing to run inference with incompatible tokenizer."
            )
        if saved_vocab != current_vocab_size:
            raise RuntimeError(
                "VOCABULARY MISMATCH!\n"
                f"Checkpoint vocab: {saved_vocab}\n"
                f"Current vocab:    {current_vocab_size}\n"
                "Refusing to run inference with incompatible vocabulary."
            )

    model_config = checkpoint.get("model_config", default_config.model.to_dict())

    model = NovaLM(
        vocab_size=model_config["vocab_size"],
        max_sequence_length=model_config["max_sequence_length"],
        embedding_dim=model_config.get("embedding_dim", 128),
        num_heads=model_config.get("num_heads", 4),
        feed_forward_dim=model_config.get("feed_forward_dim", 512),
        num_layers=model_config.get("num_layers", 2),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, tokenizer, checkpoint, checkpoint_path


# ============================================================
# SAMPLING LOGIC
# ============================================================

def sample_next_token(
    logits: torch.Tensor,
    generated_ids: list,
    temperature: float = 0.8,
    top_k: int = 40,
    repetition_penalty: float = 1.1,
) -> int:
    """
    Sample next token with temperature, top-k filtering, and repetition penalty.
    """
    # Repetition penalty
    if repetition_penalty != 1.0 and generated_ids:
        counts = torch.bincount(torch.tensor(generated_ids, device=logits.device), minlength=logits.shape[-1])
        logits = torch.where(counts > 0, logits / repetition_penalty, logits)

    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1).item()

    logits = logits / temperature

    # Top-K filtering
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        min_val = values[-1]
        logits = torch.where(logits < min_val, torch.tensor(float("-inf"), device=logits.device), logits)

    probabilities = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probabilities, num_samples=1)
    return next_token.item()


# ============================================================
# GENERATION ENGINE
# ============================================================

@torch.no_grad()
def generate(
    model: NovaLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 60,
    temperature: float = 0.8,
    top_k: int = 40,
    repetition_penalty: float = 1.1,
    device: torch.device = None,
) -> str:
    """
    Generate new text conditioned on an initial prompt.
    """
    if device is None:
        device = next(model.parameters()).device

    encoding = tokenizer.encode(prompt)
    token_ids = encoding.ids

    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    # Prepend BOS if not present and available
    if bos_id is not None and (not token_ids or token_ids[0] != bos_id):
        token_ids = [bos_id] + token_ids

    prompt_length = len(token_ids)
    generated_sequence = list(token_ids)

    for _ in range(max_new_tokens):
        context_ids = generated_sequence[-model.max_sequence_length :]
        input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)

        logits = model(input_ids)
        next_token_logits = logits[0, -1, :]

        next_token = sample_next_token(
            next_token_logits,
            generated_ids=generated_sequence[prompt_length:],
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

        generated_sequence.append(next_token)

        if eos_id is not None and next_token == eos_id:
            break

    newly_generated_ids = generated_sequence[prompt_length:]
    return tokenizer.decode(newly_generated_ids)


# ============================================================
# CLI ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="NovaLM Text Generation")
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt")
    parser.add_argument("--tokens", type=int, default=60, help="Max new tokens to generate")
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top-K cutoff")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify tokenizer/checkpoint compatibility without generating text",
    )
    args = parser.parse_args()

    if args.check:
        print("=" * 60)
        print("NovaLM Inference Compatibility Check")
        print("=" * 60)

        tokenizer_path = default_config.tokenizer_file
        if not tokenizer_path.exists():
            print(f"Tokenizer not found: {tokenizer_path}")
            return

        if default_config.latest_checkpoint.exists():
            checkpoint_path = default_config.latest_checkpoint
        elif default_config.best_checkpoint.exists():
            checkpoint_path = default_config.best_checkpoint
        else:
            print("No checkpoint found.")
            return

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        current_sha256 = sha256_file(tokenizer_path)
        current_vocab = tokenizer.get_vocab_size()
        identity = checkpoint.get("tokenizer_identity")

        print(f"Tokenizer vocab:      {current_vocab:,}")
        print(f"Current tokenizer:    {current_sha256}")
        print(f"Checkpoint step:      {checkpoint.get('global_step', 'N/A'):,}")

        if identity is None:
            print("✗ Checkpoint has no tokenizer identity")
            print("  This checkpoint cannot be verified safely.")
            raise SystemExit(1)

        if identity.get("sha256") != current_sha256:
            print("✗ TOKENIZER MISMATCH")
            print(f"  Checkpoint: {identity.get('sha256')}")
            print(f"  Current:    {current_sha256}")
            raise SystemExit(1)

        if identity.get("vocab_size") != current_vocab:
            print("✗ VOCABULARY MISMATCH")
            print(f"  Checkpoint: {identity.get('vocab_size')}")
            print(f"  Current:    {current_vocab}")
            raise SystemExit(1)

        print("✓ Tokenizer identity matches checkpoint")
        print("✓ Safe to run inference")
        print("=" * 60)
        return

    print("=" * 60)
    print("NovaLM Text Generation")
    print("=" * 60)

    device = get_device()
    print(f"Device: {device}")

    try:
        model, tokenizer, checkpoint, ckpt_path = load_inference_artifacts(
            checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
            device=device,
        )
    except FileNotFoundError as e:
        print(f"\n{e}")
        return

    print(f"Loaded checkpoint: {ckpt_path.name}")
    print(f"Global step:       {checkpoint.get('global_step', 'N/A')}")
    print(f"Validation loss:   {checkpoint.get('validation_loss', 'N/A')}")
    print(f"Vocab size:        {tokenizer.get_vocab_size():,}")
    print("=" * 60)

    if args.prompt:
        prompts = [args.prompt]
        interactive = False
    else:
        interactive = True

    while True:
        if interactive:
            print("\nEnter prompt (or 'exit' to quit):")
            prompt = input("> ").strip()
            if not prompt or prompt.lower() in ("exit", "quit"):
                break
        else:
            prompt = prompts.pop(0)

        print(f"\nPrompt: {prompt}")
        output = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.tokens,
            temperature=args.temp,
            top_k=args.top_k,
            device=device,
        )
        print(f"Generated text:\n{output}\n" + "-" * 60)

        if not interactive:
            break


if __name__ == "__main__":
    main()
