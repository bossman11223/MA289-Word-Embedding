"""Fast smoke tests on tiny synthetic data (no Kaggle files needed).

Run with:  pytest
"""
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.corpus_pairs import build_pairs
from src.data import build_vocab, encode_pairs, make_loader, train_val_split
from src.embeddings import nearest_neighbors, normalize_rows, pair_cosines
from src.model import DeeperEmbeddingFFNN, EmbeddingFFNN, build_model
from src.similarity import MAPPINGS, apply_mapping
from src.submission import make_submission, validate_submission
from src.train import evaluate, make_optimizer, train_one_epoch
from src.utils import set_seed

V = 6


@pytest.fixture
def pairs():
    return pd.DataFrame({
        "center_token": ["tok_0003", "tok_0001", "tok_0002", "tok_0005", "tok_0004", "tok_0001"] * 5,
        "context_token": ["tok_0001", "tok_0002", "tok_0003", "tok_0004", "tok_0005", "tok_0006"] * 5,
    })


@pytest.fixture
def test_df():
    return pd.DataFrame({
        "pair_id": ["pair_0001", "pair_0002", "pair_0003"],
        "token1": ["tok_0001", "tok_0002", "tok_0004"],
        "token2": ["tok_0002", "tok_0006", "tok_0005"],
    })


def test_vocab_mapping_is_sorted_and_complete(pairs, test_df):
    token_to_idx, idx_to_token = build_vocab(pairs, test_df)
    assert len(token_to_idx) == V
    assert token_to_idx["tok_0001"] == 0 and token_to_idx["tok_0006"] == 5
    assert all(idx_to_token[i] == t for t, i in token_to_idx.items())

    center_ids, context_ids = encode_pairs(pairs, token_to_idx)
    assert center_ids.dtype == torch.long
    assert center_ids[0].item() == 2  # tok_0003 -> 2


def test_train_val_split_sizes(pairs, test_df):
    token_to_idx, _ = build_vocab(pairs, test_df)
    c, t = encode_pairs(pairs, token_to_idx)
    (tr_c, tr_t), (va_c, va_t) = train_val_split(c, t, val_fraction=0.2, seed=0)
    assert len(tr_c) + len(va_c) == len(c)
    assert len(va_c) == 6


@pytest.mark.parametrize("hidden_dim", [0, 8])
def test_forward_shape(hidden_dim):
    model = build_model(V, embedding_dim=4, hidden_dim=hidden_dim)
    expected_cls = DeeperEmbeddingFFNN if hidden_dim else EmbeddingFFNN
    assert isinstance(model, expected_cls)

    batch = torch.tensor([0, 1, 2, 5], dtype=torch.long)
    logits = model(batch)
    assert logits.shape == (4, V)


@pytest.mark.parametrize("optimizer", ["adam", "sgd"])
def test_one_training_step_updates_embeddings(pairs, test_df, optimizer):
    set_seed(0)
    token_to_idx, _ = build_vocab(pairs, test_df)
    c, t = encode_pairs(pairs, token_to_idx)
    loader = make_loader(c, t, batch_size=8, shuffle=True, seed=0)

    model = build_model(V, embedding_dim=4)
    cfg = {"optimizer": optimizer, "lr": 0.1, "momentum": 0.0, "weight_decay": 0.0}
    opt = make_optimizer(model, cfg)
    before = model.embedding.weight.detach().clone()

    loss = train_one_epoch(model, loader, opt, nn.CrossEntropyLoss(), torch.device("cpu"))
    val_loss = evaluate(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"))

    assert np.isfinite(loss) and np.isfinite(val_loss)
    assert not torch.equal(before, model.embedding.weight)


def test_cosines_and_neighbors(pairs, test_df):
    token_to_idx, idx_to_token = build_vocab(pairs, test_df)
    E = np.random.default_rng(0).normal(size=(V, 4))
    E_norm = normalize_rows(E)
    assert np.allclose(np.linalg.norm(E_norm, axis=1), 1.0)

    cos = pair_cosines(E_norm, token_to_idx, test_df["token1"], test_df["token2"])
    assert cos.shape == (3,) and np.all(np.abs(cos) <= 1 + 1e-9)

    nn_df = nearest_neighbors("tok_0001", E_norm, token_to_idx, idx_to_token, k=3)
    assert len(nn_df) == 3 and "tok_0001" not in set(nn_df["token"])
    assert nn_df["cosine_similarity"].is_monotonic_decreasing


@pytest.mark.parametrize("name", sorted(MAPPINGS))
def test_every_mapping_outputs_values_in_unit_interval(name):
    cos = np.linspace(-1, 1, 21)
    cfg = {"name": name, "power": 0.5, "target_mean": 0.5, "target_std": 0.6}
    out = apply_mapping(cos, cfg)
    assert out.shape == cos.shape
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_mapping_formulas():
    cos = np.array([-1.0, 0.0, 0.5, 1.0])
    assert np.allclose(apply_mapping(cos, {"name": "shift"}), [0.0, 0.5, 0.75, 1.0])
    assert np.allclose(apply_mapping(cos, {"name": "clip"}), [0.0, 0.0, 0.5, 1.0])
    assert np.allclose(apply_mapping(cos, {"name": "power", "power": 2.0}), [0.0, 0.0, 0.25, 1.0])
    out = apply_mapping(cos, {"name": "rescale", "target_mean": 0.4, "target_std": 0.1})
    assert np.isclose(out.mean(), 0.4) and np.isclose(out.std(), 0.1)


def test_submission_validation(test_df):
    good = make_submission(test_df, np.array([0.1, 0.5, 0.9]))
    validate_submission(good, test_df)  # should not raise

    bad_cases = [
        good.rename(columns={"similarity": "score"}),                 # wrong column name
        good.iloc[:2],                                                 # wrong row count
        good.iloc[[1, 0, 2]].reset_index(drop=True),                   # wrong order
        good.assign(similarity=[0.1, np.nan, 0.9]),                    # NaN
        good.assign(similarity=[0.1, 1.2, 0.9]),                       # out of range
    ]
    for bad in bad_cases:
        with pytest.raises(AssertionError):
            validate_submission(bad, test_df)


def test_build_pairs_from_corpus_window():
    pairs = build_pairs([["a", "b", "c"]], window=1)
    got = list(zip(pairs["center_token"], pairs["context_token"]))
    assert got == [("a", "b"), ("b", "a"), ("b", "c"), ("c", "b")]
    # window=2 on 3 tokens: every token pairs with the other two.
    assert len(build_pairs([["a", "b", "c"]], window=2)) == 6
