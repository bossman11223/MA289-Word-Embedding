"""Load the competition files, build the vocabulary, and make DataLoaders."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_pairs(path):
    """Read a (center_token, context_token) CSV."""
    pairs = pd.read_csv(path)
    expected = ["center_token", "context_token"]
    if list(pairs.columns) != expected:
        raise ValueError(f"{path} must have columns {expected}, got {list(pairs.columns)}")
    return pairs


def load_test(path):
    """Read test.csv (pair_id, token1, token2)."""
    return pd.read_csv(path)


def build_vocab(pairs, test=None):
    """Map each token string to an integer ID, sorted so tok_0001 -> 0, tok_0002 -> 1, ...

    Test tokens are included so every test pair can be looked up. A test token
    that never appears in the training pairs keeps its random starting vector.
    """
    tokens = set(pairs["center_token"]) | set(pairs["context_token"])
    if test is not None:
        tokens |= set(test["token1"]) | set(test["token2"])

    all_tokens = sorted(tokens)
    token_to_idx = {token: i for i, token in enumerate(all_tokens)}
    idx_to_token = {i: token for token, i in token_to_idx.items()}
    return token_to_idx, idx_to_token


def encode_pairs(pairs, token_to_idx):
    """Convert token strings to long tensors (nn.Embedding needs integer indices)."""
    center_ids = torch.tensor(pairs["center_token"].map(token_to_idx).to_numpy(), dtype=torch.long)
    context_ids = torch.tensor(pairs["context_token"].map(token_to_idx).to_numpy(), dtype=torch.long)
    return center_ids, context_ids


def train_val_split(center_ids, context_ids, val_fraction, seed):
    """Randomly hold out val_fraction of the pairs for validation loss."""
    n = len(center_ids)
    rng = np.random.default_rng(seed)
    order = torch.from_numpy(rng.permutation(n))
    n_val = int(round(n * val_fraction))

    val_idx = order[:n_val]
    train_idx = order[n_val:]
    train = (center_ids[train_idx], context_ids[train_idx])
    val = (center_ids[val_idx], context_ids[val_idx])
    return train, val


def make_loader(center_ids, context_ids, batch_size, shuffle, seed=0):
    dataset = TensorDataset(center_ids, context_ids)
    # A seeded generator makes the shuffle order the same every run.
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)
