import torch
import torch.nn as nn


class NextTokenLoss(nn.Module):
    """
    Cross-entropy loss for causal language modeling.

    Logits:
        [batch_size, sequence_length, vocab_size]

    Targets:
        [batch_size, sequence_length]
    """

    def __init__(self):
        super().__init__()

        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        batch_size, sequence_length, vocab_size = logits.shape

        logits = logits.reshape(
            batch_size * sequence_length,
            vocab_size,
        )

        targets = targets.reshape(
            batch_size * sequence_length,
        )

        loss = self.loss_function(
            logits,
            targets,
        )

        return loss
