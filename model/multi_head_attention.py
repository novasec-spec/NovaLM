import math

import torch
import torch.nn as nn


class MultiHeadCausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Input:
        [batch_size, sequence_length, embedding_dim]

    Output:
        [batch_size, sequence_length, embedding_dim]
    """

    def __init__(self, embedding_dim, num_heads):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads."
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

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

        # [B, T, C] → [B, H, T, D]
        q = q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        # Attention scores:
        # [B, H, T, D] × [B, H, D, T]
        # → [B, H, T, T]
        attention_scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        # Causal mask.
        causal_mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device,
                dtype=torch.bool,
            )
        )

        attention_scores = attention_scores.masked_fill(
            ~causal_mask,
            float("-inf"),
        )

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1,
        )

        attended = attention_weights @ v

        # [B, H, T, D]
        # → [B, T, H, D]
        attended = attended.transpose(1, 2)

        # Combine heads.
        attended = attended.contiguous().view(
            batch_size,
            sequence_length,
            self.embedding_dim,
        )

        output = self.output(attended)

        return output