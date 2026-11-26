"""
Unit tests for NovaLM Atomic Checkpointing and State Resumption
"""

from pathlib import Path
import tempfile
import torch
import unittest

from model.novalm import NovaLM
from training.train import save_atomic_checkpoint


class TestAtomicCheckpoint(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.ckpt_path = Path(self.temp_dir.name) / "latest.pt"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_atomic_checkpoint(self):
        model = NovaLM(vocab_size=100, max_sequence_length=16, embedding_dim=32, num_heads=2, feed_forward_dim=64, num_layers=1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        payload = {
            "epoch": 5,
            "global_step": 1200,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_validation_loss": 1.85,
        }

        save_atomic_checkpoint(payload, self.ckpt_path)
        self.assertTrue(self.ckpt_path.exists())

        loaded = torch.load(self.ckpt_path, weights_only=False)
        self.assertEqual(loaded["epoch"], 5)
        self.assertEqual(loaded["global_step"], 1200)
        self.assertEqual(loaded["best_validation_loss"], 1.85)
        self.assertIn("model_state_dict", loaded)
        self.assertIn("optimizer_state_dict", loaded)


if __name__ == "__main__":
    unittest.main()
