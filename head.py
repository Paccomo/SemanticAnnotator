import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    def __init__(self, head_size, n_embed, block_size, dropout):
        super().__init__()
        if min(head_size, n_embed, block_size) <= 0:
            raise ValueError("head_size, n_embed, and block_size must be positive")
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.block_size = block_size

    def forward(self, x, padding_mask=None):
        B, T, _ = x.shape
        if T > self.block_size:
            raise ValueError("Sequence length exceeds block_size")
        if padding_mask is not None:
            if padding_mask.shape != (B, T) or padding_mask.dtype != torch.bool:
                raise ValueError("padding_mask must be boolean with shape (batch_size, sequence_length)")
            # Broadcast the padding mask across all queries.
            padding_mask = padding_mask[:, None, :]

        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        # Scale by the query/key dimension of this head.
        wei = q @ k.transpose(-2, -1) * k.size(-1)**-0.5
        if padding_mask is not None:
            wei = wei.masked_fill(padding_mask, float("-inf"))
            # Keep softmax finite when a sequence has no real tokens.
            wei = wei.masked_fill(padding_mask.all(dim=-1, keepdim=True), 0.0)
        wei = F.softmax(wei, dim=-1)
        if padding_mask is not None:
            wei = wei.masked_fill(padding_mask, 0.0)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v  # (B, T, head_size)
        return out
