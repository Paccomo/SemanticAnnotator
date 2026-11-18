import torch
import torch.nn as nn

from head import Head


class MultiHeadAttention(nn.Module):
    """Causal attention heads followed by an output projection."""

    def __init__(self, num_heads, head_size, n_embed, block_size, dropout):
        super().__init__()
        if num_heads <= 0 or head_size <= 0 or num_heads * head_size != n_embed:
            raise ValueError("Positive num_heads * head_size must equal n_embed")
        self.heads = nn.ModuleList([
            Head(head_size, n_embed, block_size, dropout)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
