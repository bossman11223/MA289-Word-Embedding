"""Skip-gram style models: predict a context token from a center token.

The output layer only exists to give training a goal. After training we keep
the embedding matrix (model.embedding.weight) and throw the rest away.
"""
import torch
import torch.nn as nn


class EmbeddingFFNN(nn.Module):
    """center ID -> embedding (d) -> linear -> V logits."""

    def __init__(self, vocab_size, embedding_dim=24):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, token_ids):
        embeddings = self.embedding(token_ids)   # (batch, d)
        logits = self.output(embeddings)          # (batch, V)
        return logits


class DeeperEmbeddingFFNN(nn.Module):
    """center ID -> embedding (d) -> linear -> ReLU (hidden) -> linear -> V logits."""

    def __init__(self, vocab_size, embedding_dim=24, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden = nn.Linear(embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, token_ids):
        e = self.embedding(token_ids)       # (batch, d)
        h = torch.relu(self.hidden(e))      # (batch, hidden_dim)
        logits = self.output(h)             # (batch, V)
        return logits


def build_model(vocab_size, embedding_dim, hidden_dim=0):
    """hidden_dim == 0 gives the plain model; > 0 adds one hidden ReLU layer."""
    if hidden_dim and hidden_dim > 0:
        return DeeperEmbeddingFFNN(vocab_size, embedding_dim, hidden_dim)
    return EmbeddingFFNN(vocab_size, embedding_dim)
