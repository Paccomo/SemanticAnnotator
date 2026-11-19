import torch.nn as nn

from feed_forward import FeedForward
from multi_head_attention import MultiHeadAttention


class Block(nn.Module):
    def __init__(self, n_embed, n_head, block_size, dropout):
        super().__init__()
        if n_embed <= 0 or n_head <= 0 or n_embed % n_head != 0:
            raise ValueError("n_embed and n_head must be positive, with n_embed divisible by n_head")
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embed, block_size, dropout)
        self.ff = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x, padding_mask=None):
        x = x + self.sa(self.ln1(x), padding_mask)
        x = x + self.ff(self.ln2(x))
        return x
