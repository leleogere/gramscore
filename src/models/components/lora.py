"""
Code adapted from https://lightning.ai/docs/overview/finetune-models/llm-low-rank-adaption-of-large-language-models-lora
"""

import torch
from torch import nn


class LoRALayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.alpha = alpha

        std_dev = 1 / torch.sqrt(torch.tensor(self.rank).float())
        self.A = torch.nn.Parameter(torch.randn(self.in_dim, self.rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(self.rank, self.out_dim))

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x


class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)
