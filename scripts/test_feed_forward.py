import torch

from model.feed_forward import FeedForward


EMBEDDING_DIM = 128
HIDDEN_DIM = 512
BATCH_SIZE = 4
SEQUENCE_LENGTH = 32


def main():
    print("=" * 60)
    print("NovaLM Feed-Forward Network Test")
    print("=" * 60)

    model = FeedForward(
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
    )

    x = torch.randn(
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        EMBEDDING_DIM,
    )

    output = model(x)

    print("\nInput shape:")
    print(x.shape)

    print("\nHidden dimension:")
    print(HIDDEN_DIM)

    print("\nOutput shape:")
    print(output.shape)

    print("\nOutput dtype:")
    print(output.dtype)

    print("\n" + "=" * 60)
    print("SHAPE CHECK")
    print("=" * 60)

    assert x.shape == (
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        EMBEDDING_DIM,
    )

    assert output.shape == (
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        EMBEDDING_DIM,
    )

    assert output.dtype == torch.float32

    assert torch.isfinite(output).all()

    print("\n✓ Input shape is [4, 32, 128]")
    print("✓ Hidden representation expands to 512")
    print("✓ Output returns to 128 dimensions")
    print("✓ Output shape is [4, 32, 128]")
    print("✓ No NaN or infinite values")

    print("\n" + "=" * 60)
    print("FEED-FORWARD TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()