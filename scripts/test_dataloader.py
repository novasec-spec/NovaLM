import torch

from dataset import get_dataloader


BATCH_SIZE = 4
SEQUENCE_LENGTH = 32


def main():
    print("=" * 60)
    print("NovaLM DataLoader Test")
    print("=" * 60)

    train_loader = get_dataloader(
        split="train",
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    print(
        f"\nNumber of training sequences: "
        f"{len(train_loader.dataset)}"
    )

    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of batches: {len(train_loader)}")

    inputs, targets = next(iter(train_loader))

    print("\n" + "=" * 60)
    print("FIRST BATCH")
    print("=" * 60)

    print(f"\nInput shape:  {tuple(inputs.shape)}")
    print(f"Target shape: {tuple(targets.shape)}")

    print(f"\nInput dtype:  {inputs.dtype}")
    print(f"Target dtype: {targets.dtype}")

    print(f"\nInput device:  {inputs.device}")
    print(f"Target device: {targets.device}")

    print("\nInput batch:")
    print(inputs)

    print("\nTarget batch:")
    print(targets)

    print("\n" + "=" * 60)
    print("SHAPE CHECK")
    print("=" * 60)

    assert inputs.ndim == 2
    assert targets.ndim == 2

    assert inputs.shape == targets.shape

    assert inputs.shape[0] <= BATCH_SIZE

    assert inputs.shape[1] == SEQUENCE_LENGTH

    assert inputs.dtype == torch.long
    assert targets.dtype == torch.long

    print("\n✓ Inputs are 2-dimensional")
    print("✓ Targets are 2-dimensional")
    print("✓ Input and target shapes match")
    print("✓ Sequence length is 32")
    print("✓ Token IDs use torch.long")

    print("\n" + "=" * 60)
    print("DATALOADER TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
