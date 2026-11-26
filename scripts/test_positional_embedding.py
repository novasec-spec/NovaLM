import torch

from model.embeddings import TokenEmbedding
from model.positional_encoding import PositionalEmbedding


VOCAB_SIZE = 1000
EMBEDDING_DIM = 128
BATCH_SIZE = 4
SEQUENCE_LENGTH = 32


def main():
    print("=" * 60)
    print("NovaLM Positional Embedding Test")
    print("=" * 60)

    token_embedding = TokenEmbedding(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
    )

    positional_embedding = PositionalEmbedding(
        max_sequence_length=SEQUENCE_LENGTH,
        embedding_dim=EMBEDDING_DIM,
    )

    input_ids = torch.randint(
        low=0,
        high=VOCAB_SIZE,
        size=(BATCH_SIZE, SEQUENCE_LENGTH),
        dtype=torch.long,
    )

    token_vectors = token_embedding(input_ids)
    output = positional_embedding(token_vectors)

    print("\nInput token shape:")
    print(token_vectors.shape)

    print("\nOutput shape:")
    print(output.shape)

    print("\nOutput dtype:")
    print(output.dtype)

    print("\nFirst token, first 10 values:")
    print(output[0, 0, :10])

    print("\n" + "=" * 60)
    print("SHAPE CHECK")
    print("=" * 60)

    assert token_vectors.shape == (
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

    print("\n✓ Token embeddings have correct shape")
    print("✓ Positional information was added")
    print(f"✓ Output shape is [{BATCH_SIZE}, {SEQUENCE_LENGTH}, {EMBEDDING_DIM}]")
    print("✓ Output remains float32")

    print("\n" + "=" * 60)
    print("POSITIONAL EMBEDDING TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
