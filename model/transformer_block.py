import torch.nn as nn

from model.multi_head_attention import (
    MultiHeadCausalSelfAttention,
)
from model.feed_forward import FeedForward


class TransformerBlock(nn.Module):
    """
    A complete pre-layer-normalized Transformer block.

    Structure:

        x
        │
        ├── LayerNorm
        │
        ├── Multi-Head Causal Self-Attention
        │
        └── Residual Add
                │
                ▼
        LayerNorm
                │
                ▼
        Feed-Forward Network
                │
                ▼
        Residual Add
    """

    def __init__(
        self,
        embedding_dim,
        num_heads,
        feed_forward_dim,
    ):
        super().__init__()

        self.attention_norm = nn.LayerNorm(
            embedding_dim
        )

        self.attention = MultiHeadCausalSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
        )

        self.feed_forward_norm = nn.LayerNorm(
            embedding_dim
        )

        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim,
            hidden_dim=feed_forward_dim,
        )

    def forward(self, x):

        # Attention sub-layer + residual connection
        attention_input = self.attention_norm(x)

        x = x + self.attention(
            attention_input
        )

        # Feed-forward sub-layer + residual connection
        feed_forward_input = self.feed_forward_norm(x)

        x = x + self.feed_forward(
            feed_forward_input
        )

        return x