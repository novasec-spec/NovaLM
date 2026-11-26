"""
Unit tests for NovaLM Model Components
"""

import math
import torch
import unittest

from model.novalm import NovaLM
from model.embeddings import TokenEmbedding
from model.positional_encoding import PositionalEmbedding
from model.multi_head_attention import MultiHeadCausalSelfAttention
from model.feed_forward import FeedForward
from model.transformer_block import TransformerBlock
from training.loss import NextTokenLoss


class TestNovaLMComponents(unittest.TestCase):

    def setUp(self):
        self.vocab_size = 100
        self.seq_len = 16
        self.embed_dim = 32
        self.num_heads = 2
        self.hidden_dim = 64
        self.num_layers = 2
        self.batch_size = 2

    def test_token_embedding(self):
        emb = TokenEmbedding(self.vocab_size, self.embed_dim)
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        out = emb(input_ids)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.embed_dim))

    def test_positional_embedding(self):
        pos = PositionalEmbedding(self.seq_len, self.embed_dim)
        x = torch.zeros((self.batch_size, self.seq_len, self.embed_dim))
        out = pos(x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.embed_dim))

    def test_multi_head_attention_causality(self):
        attn = MultiHeadCausalSelfAttention(self.embed_dim, self.num_heads)
        x = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        out = attn(x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.embed_dim))

    def test_feed_forward(self):
        ff = FeedForward(self.embed_dim, self.hidden_dim)
        x = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        out = ff(x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.embed_dim))

    def test_transformer_block(self):
        block = TransformerBlock(self.embed_dim, self.num_heads, self.hidden_dim)
        x = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        out = block(x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.embed_dim))

    def test_full_model_forward(self):
        model = NovaLM(
            vocab_size=self.vocab_size,
            max_sequence_length=self.seq_len,
            embedding_dim=self.embed_dim,
            num_heads=self.num_heads,
            feed_forward_dim=self.hidden_dim,
            num_layers=self.num_layers,
        )
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        logits = model(input_ids)
        self.assertEqual(logits.shape, (self.batch_size, self.seq_len, self.vocab_size))

    def test_next_token_loss(self):
        loss_fn = NextTokenLoss()
        logits = torch.randn(self.batch_size, self.seq_len, self.vocab_size)
        targets = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        loss = loss_fn(logits, targets)
        self.assertTrue(loss.item() > 0)
        self.assertFalse(torch.isnan(loss))


if __name__ == "__main__":
    unittest.main()
