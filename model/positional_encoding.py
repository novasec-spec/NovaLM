import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """
    Adds learnable positional information to token embeddings.
    """

    def __init__(self, max_sequence_length, embedding_dim):
        super().__init__()

        self.position_embedding = nn.Embedding(
            num_embeddings=max_sequence_length,
            embedding_dim=embedding_dim,
        )

    def forward(self, x):
        """
        x shape:
            [batch_size, sequence_length, embedding_dim]

        returns:
            [batch_size, sequence_length, embedding_dim]
        """

        batch_size, sequence_length, _ = x.shape

        positions = torch.arange(
            sequence_length,
            device=x.device,
        )

        position_vectors = self.position_embedding(
            positions
        )

        position_vectors = position_vectors.unsqueeze(0)

        return x + position_vectors