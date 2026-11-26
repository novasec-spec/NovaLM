import math

import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    """
    Single-head causal self-attention.

    Input:
        [batch_size, sequence_length, embedding_dim]

    Output:
        [batch_size, sequence_length, embedding_dim]
    """

    def __init__(self, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.query = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        self.key = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        self.value = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        self.output = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

    def forward(self, x):
        batch_size, sequence_length, _ = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attention_scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.embedding_dim)

        causal_mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device,
            )
        )

        attention_scores = attention_scores.masked_fill(
            causal_mask == 0,
            float("-inf"),
        )

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1,
        )

        attended = attention_weights @ v

        output = self.output(attended)

        return output