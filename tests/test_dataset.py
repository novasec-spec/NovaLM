"""
Unit tests for NovaLM Memory-Mapped Dataset and Input-Target Shift
"""

from pathlib import Path
import numpy as np
import tempfile
import unittest

from dataset.dataset import MemMapDataset


class TestMemMapDataset(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.bin_file = Path(self.temp_dir.name) / "test.bin"
        self.seq_len = 8

        # Create a synthetic sequence: 0, 1, 2, 3, ..., 99
        tokens = np.arange(100, dtype=np.uint16)
        with open(self.bin_file, "wb") as f:
            f.write(tokens.tobytes())

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_memmap_dataset_length(self):
        dataset = MemMapDataset(self.bin_file, sequence_length=self.seq_len)
        expected_seqs = (100 - 1) // self.seq_len
        self.assertEqual(len(dataset), expected_seqs)

    def test_input_target_causal_shift(self):
        dataset = MemMapDataset(self.bin_file, sequence_length=self.seq_len)
        input_ids, target_ids = dataset[0]

        # Input should be [0, 1, 2, 3, 4, 5, 6, 7]
        # Target should be [1, 2, 3, 4, 5, 6, 7, 8]
        self.assertEqual(input_ids.tolist(), list(range(0, 8)))
        self.assertEqual(target_ids.tolist(), list(range(1, 9)))

        # Verify next-token relationship: target[i] == input[i+1] for i in range(seq_len - 1)
        for i in range(self.seq_len - 1):
            self.assertEqual(target_ids[i].item(), input_ids[i + 1].item())


if __name__ == "__main__":
    unittest.main()
