import torch
import torch.nn as nn

from block import Block


class SemanticAnnotationModel(nn.Module):
    def __init__(self, vocab_size, n_embed, n_head, n_layer, block_size, dropout, pad_id):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.block_size = block_size
        self.pad_id = pad_id

    def forward(self, idx):
        _, T = idx.shape
        if T > self.block_size:
            raise ValueError("Sequence length exceeds block_size")

        padding_mask = idx == self.pad_id
        positions = torch.arange(T, device=idx.device)

        x = self.token_embedding_table(idx) + self.position_embedding_table(positions)
        for block in self.blocks:
            x = block(x, padding_mask=padding_mask)

        return self.ln_f(x)