"""
Binary Cross Entropy for DCGAN
"""

import torch
import torch.nn as nn


class Binary_Cross_Entropy(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.loss = nn.BCELoss()

    def forward(self, logits, labels):
        loss = self.loss(logits, labels)
        return loss
