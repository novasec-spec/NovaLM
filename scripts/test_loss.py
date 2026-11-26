
import torch

from model.novalm import NovaLM
from training.loss import NextTokenLoss


VOCAB_SIZE = 964
SEQUENCE_LENGTH = 32

EMBEDDING_DIM = 128
NUM_HEADS = 4
FEED_FORWARD_DIM = 512
NUM_LAYERS = 2

BATCH_SIZE = 4


def main():
    print("=" * 60)
    print("NovaLM Loss Test")
    print("=" * 60)

    model = NovaLM(
        vocab_size=VOCAB_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        feed_forward_dim=FEED_FORWARD_DIM,
        num_layers=NUM_LAYERS,
    )

    loss_function = NextTokenLoss()

    input_ids = torch.randint(
        0,
        VOCAB_SIZE,
        (
            BATCH_SIZE,
            SEQUENCE_LENGTH,
        ),
        dtype=torch.long,
    )

    targets = torch.randint(
        0,
        VOCAB_SIZE,
        (
            BATCH_SIZE,
            SEQUENCE_LENGTH,
        ),
        dtype=torch.long,
    )

    logits = model(input_ids)

    loss = loss_function(
        logits,
        targets,
    )

    print("\nLogits shape:")
    print(logits.shape)

    print("\nTargets shape:")
    print(targets.shape)

    print("\nLoss:")
    print(loss.item())

    print("\nLoss dtype:")
    print(loss.dtype)

    print("\n" + "=" * 60)
    print("LOSS CHECK")
    print("=" * 60)

    assert logits.shape == (
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        VOCAB_SIZE,
    )

    assert targets.shape == (
        BATCH_SIZE,
        SEQUENCE_LENGTH,
    )

    assert loss.ndim == 0

    assert torch.isfinite(loss)

    print("\n✓ Logits have correct shape")
    print("✓ Targets have correct shape")
    print("✓ Loss is a scalar")
    print("✓ Loss is finite")

    print("\n" + "=" * 60)
    print("LOSS TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
