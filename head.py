import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """One causal self-attention head."""

    def __init__(self, head_size, n_embed, block_size, dropout):
        super().__init__()
        if min(head_size, n_embed, block_size) <= 0:
            raise ValueError("head_size, n_embed, and block_size must be positive")
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        if T > self.tril.size(0):
            raise ValueError("Sequence length exceeds block_size")
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        # Scale by the query/key dimension of this head.
        wei = q @ k.transpose(-2, -1) * k.size(-1)**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v  # (B, T, head_size)
        return out
