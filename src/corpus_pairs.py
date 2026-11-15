"""Rebuild (center, context) training pairs from train_corpus.txt.

For each token in a segment, every token within `window` positions on either
side becomes a context token. Example with window=1 and segment "a b c":
(a, b), (b, a), (b, c), (c, b).

Pairs never cross segment boundaries (each line of the corpus is one segment).
"""
import pandas as pd


def read_corpus(path):
    """Return a list of segments, each a list of token strings."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.split() for line in lines if line.strip()]


def build_pairs(segments, window):
    if window < 1:
        raise ValueError("window must be at least 1")

    centers, contexts = [], []
    for tokens in segments:
        n = len(tokens)
        for i, center in enumerate(tokens):
            lo = max(0, i - window)
            hi = min(n, i + window + 1)
            for j in range(lo, hi):
                if j != i:
                    centers.append(center)
                    contexts.append(tokens[j])

    return pd.DataFrame({"center_token": centers, "context_token": contexts})
