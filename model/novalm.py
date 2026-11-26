import torch
import torch.nn as nn

from model.embeddings import TokenEmbedding
from model.positional_encoding import PositionalEmbedding
from model.transformer_block import TransformerBlock


class NovaLM(nn.Module):
    """
    NovaLM V1 language model.

    Architecture:

        Token IDs
            ↓
        Token Embedding
            ↓
        Positional Embedding
            ↓
        Transformer Blocks
            ↓
        Final LayerNorm
            ↓
        Vocabulary Projection
            ↓
        Logits
    """

    def __init__(
        self,
        vocab_size,
        max_sequence_length,
        embedding_dim=256,
        num_heads=8,
        feed_forward_dim=1024,
        num_layers=6,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.feed_forward_dim = feed_forward_dim
        self.num_layers = num_layers

        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
        )

        self.positional_embedding = PositionalEmbedding(
            max_sequence_length=max_sequence_length,
            embedding_dim=embedding_dim,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    feed_forward_dim=feed_forward_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(
            embedding_dim
        )

        self.output_projection = nn.Linear(
            embedding_dim,
            vocab_size,
        )

    def forward(self, input_ids):
        """
        Args:
            input_ids:
                [batch_size, sequence_length]

        Returns:
            logits:
                [batch_size, sequence_length, vocab_size]
        """

        batch_size, sequence_length = input_ids.shape

        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"maximum context length "
                f"{self.max_sequence_length}."
            )

        x = self.token_embedding(input_ids)

        x = self.positional_embedding(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.final_norm(x)

        logits = self.output_projection(x)

        return logits
