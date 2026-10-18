"""Example plugin: a tiny custom (non-transformers) language model. Used by tests/test_plugin.py.

Copy this file as a template for your own model: implement load(), blocks(), calib_batches().
"""
import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.up = nn.Linear(d, 4 * d)
        self.down = nn.Linear(4 * d, d)

    def forward(self, h):
        return h + self.down(torch.relu(self.up(self.norm(h))))


class TinyLM(nn.Module):
    def __init__(self, vocab=300, d=64, n=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList(Block(d) for _ in range(n))
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        h = self.emb(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(h)


def load():
    torch.manual_seed(0)
    return TinyLM().eval(), None


def blocks(model):
    return model.layers


def calib_batches(tokenizer, n_samples, seq_len, batch_size):
    g = torch.Generator().manual_seed(0)
    data = torch.randint(0, 300, (n_samples, seq_len), generator=g)
    for i in range(0, n_samples, batch_size):
        yield {"x": data[i:i + batch_size]}
