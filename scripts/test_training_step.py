import torch

from dataset import get_dataloader
from model.novalm import NovaLM
from training.loss import NextTokenLoss


VOCAB_SIZE = 964
SEQUENCE_LENGTH = 32

EMBEDDING_DIM = 128
NUM_HEADS = 4
FEED_FORWARD_DIM = 512
NUM_LAYERS = 2

BATCH_SIZE = 4
LEARNING_RATE = 0.001


def main():
    print("=" * 60)
    print("NovaLM Single Training Step Test")
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    train_loader = get_dataloader(
        split="train",
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    inputs, targets = next(
        iter(train_loader)
    )

    print("\nInput shape:")
    print(inputs.shape)

    print("\nTarget shape:")
    print(targets.shape)

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    logits = model(inputs)

    loss = loss_function(
        logits,
        targets,
    )

    print("\nInitial loss:")
    print(loss.item())

    # --------------------------------------------------------
    # Backward pass
    # --------------------------------------------------------

    optimizer.zero_grad()

    loss.backward()

    # Check that gradients exist.
    gradient_count = 0

    for parameter in model.parameters():
        if parameter.grad is not None:
            gradient_count += 1

    print("\nParameters with gradients:")
    print(gradient_count)

    assert gradient_count > 0

    # --------------------------------------------------------
    # Parameter update
    # --------------------------------------------------------

    optimizer.step()

    print("\nOptimizer step completed.")

    print("\n" + "=" * 60)
    print("TRAINING STEP CHECK")
    print("=" * 60)

    assert torch.isfinite(loss)

    print("\n✓ Forward pass worked")
    print("✓ Loss calculation worked")
    print("✓ Backpropagation worked")
    print("✓ Gradients were produced")
    print("✓ AdamW updated the model")

    print("\n" + "=" * 60)
    print("SINGLE TRAINING STEP PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
