"""Work with the learned embedding matrix E (shape V x d)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def extract_embeddings(model):
    """Copy the embedding weights out of the model as a NumPy array."""
    return model.embedding.weight.detach().cpu().numpy()


def normalize_rows(E):
    """Scale each row to length 1. Then cosine similarity is just a dot product."""
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    return E / np.maximum(norms, 1e-12)   # avoid dividing by zero


def pair_cosines(E_norm, token_to_idx, tokens_a, tokens_b):
    """Cosine similarity for each (tokens_a[k], tokens_b[k]) pair."""
    ia = np.array([token_to_idx[t] for t in tokens_a])
    ib = np.array([token_to_idx[t] for t in tokens_b])
    # Row-wise dot product of the two sets of unit vectors.
    return np.sum(E_norm[ia] * E_norm[ib], axis=1)


def nearest_neighbors(token, E_norm, token_to_idx, idx_to_token, k=8):
    """The k tokens whose embeddings point most nearly in the same direction."""
    idx = token_to_idx[token]
    sims = E_norm @ E_norm[idx]
    order = np.argsort(-sims)

    rows = []
    for j in order:
        if j == idx:
            continue
        rows.append({"token": idx_to_token[int(j)], "cosine_similarity": float(sims[j])})
        if len(rows) == k:
            break
    return pd.DataFrame(rows)


def plot_pca(E_norm, idx_to_token, n_show=35, seed=289, save_path=None):
    """2D PCA projection of a random sample of tokens. Only a rough picture of d-dim space."""
    E_2d = PCA(n_components=2, random_state=seed).fit_transform(E_norm)
    rng = np.random.default_rng(seed)
    show_ids = rng.choice(len(E_norm), size=min(n_show, len(E_norm)), replace=False)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(E_2d[show_ids, 0], E_2d[show_ids, 1])
    for idx in show_ids:
        ax.annotate(idx_to_token[int(idx)], (E_2d[idx, 0], E_2d[idx, 1]), fontsize=7)
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.set_title("2D Projection of Selected Learned Embeddings")
    ax.grid(alpha=0.2)

    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig
