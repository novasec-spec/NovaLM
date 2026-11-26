import torch

from model.embeddings import TokenEmbedding


VOCAB_SIZE = 964
EMBEDDING_DIM = 128
BATCH_SIZE = 4
SEQUENCE_LENGTH = 32


def main():
    print("=" * 60)
    print("NovaLM Token Embedding Test")
    print("=" * 60)

    model = TokenEmbedding(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
    )

    input_ids = torch.randint(
        low=0,
        high=VOCAB_SIZE,
        size=(BATCH_SIZE, SEQUENCE_LENGTH),
        dtype=torch.long,
    )

    print("\nInput token IDs:")
    print(input_ids)

    print("\nInput shape:")
    print(input_ids.shape)

    embeddings = model(input_ids)

    print("\nEmbedding output shape:")
    print(embeddings.shape)

    print("\nEmbedding dtype:")
    print(embeddings.dtype)

    print("\nFirst token embedding:")
    print(embeddings[0, 0])

    print("\n" + "=" * 60)
    print("SHAPE CHECK")
    print("=" * 60)

    assert input_ids.shape == (
        BATCH_SIZE,
        SEQUENCE_LENGTH,
    )

    assert embeddings.shape == (
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        EMBEDDING_DIM,
    )

    assert embeddings.dtype == torch.float32

    print("\n✓ Input shape is [4, 32]")
    print("✓ Output shape is [4, 32, 128]")
    print("✓ Embeddings are floating-point tensors")

    print("\n" + "=" * 60)
    print("EMBEDDING TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()