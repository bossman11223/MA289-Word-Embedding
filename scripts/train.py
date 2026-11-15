"""Run one full experiment from a config file.

    python scripts/train.py --config configs/baseline.yaml

Steps: load pairs -> vocab -> train/val split -> train -> save embeddings ->
cosine for each test pair -> map to [0, 1] -> validated submission.csv.
"""
import argparse
import sys
from pathlib import Path

# Let "import src..." work when this file is run directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")  # save plots to files without opening windows

import matplotlib.pyplot as plt

from src.config import RUNS_DIR, load_config, resolve_path
from src.data import build_vocab, encode_pairs, load_pairs, load_test, make_loader, train_val_split
from src.embeddings import extract_embeddings, nearest_neighbors, normalize_rows, pair_cosines, plot_pca
from src.model import build_model
from src.runs import append_results_row, create_run_dir, save_run_artifacts
from src.similarity import apply_mapping, describe_mapping
from src.submission import make_submission, save_submission
from src.train import fit
from src.utils import get_device, set_seed


def main():
    parser = argparse.ArgumentParser(description="Train skip-gram embeddings and write a submission.")
    parser.add_argument("--config", required=True, help="path to a YAML file in configs/")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    print(f"Run '{cfg['name']}' on {device}")

    # ---- data ----
    pairs = load_pairs(resolve_path(cfg["pairs_file"]))
    test = load_test(resolve_path(cfg["test_file"]))
    token_to_idx, idx_to_token = build_vocab(pairs, test)
    V = len(token_to_idx)
    print(f"{len(pairs):,} pairs, vocabulary V = {V}")

    center_ids, context_ids = encode_pairs(pairs, token_to_idx)
    (train_c, train_t), (val_c, val_t) = train_val_split(center_ids, context_ids, cfg["val_fraction"], cfg["seed"])
    train_loader = make_loader(train_c, train_t, cfg["batch_size"], shuffle=True, seed=cfg["seed"])
    val_loader = make_loader(val_c, val_t, cfg["batch_size"], shuffle=False) if len(val_c) > 0 else None
    print(f"train pairs: {len(train_c):,} | validation pairs: {len(val_c):,}")

    # ---- model + training ----
    model = build_model(V, cfg["embedding_dim"], cfg["hidden_dim"]).to(device)
    print(model)
    history = fit(model, train_loader, val_loader, cfg, device)

    # ---- save run ----
    run_dir = create_run_dir(RUNS_DIR, cfg["name"])
    E = extract_embeddings(model)
    save_run_artifacts(run_dir, cfg, history, E, token_to_idx)

    E_norm = normalize_rows(E)
    fig = plot_pca(E_norm, idx_to_token, seed=cfg["seed"], save_path=run_dir / "pca.png")
    plt.close(fig)

    # ---- submission ----
    cosines = pair_cosines(E_norm, token_to_idx, test["token1"], test["token2"])
    similarity = apply_mapping(cosines, cfg["mapping"])
    submission = make_submission(test, similarity)
    save_submission(submission, test, run_dir / "submission.csv")

    mapping_label = describe_mapping(cfg["mapping"])
    append_results_row(RUNS_DIR / "results.csv", run_dir, cfg, history, mapping_label, "submission.csv")

    print(f"\nRaw test cosines:  mean {cosines.mean():.3f}, std {cosines.std():.3f}, "
          f"min {cosines.min():.3f}, max {cosines.max():.3f}")
    print(f"Mapped ({mapping_label}): mean {similarity.mean():.3f}, std {similarity.std():.3f}")
    query = idx_to_token[0]
    print(f"\nNearest neighbors of {query}:")
    print(nearest_neighbors(query, E_norm, token_to_idx, idx_to_token, k=5).to_string(index=False))
    print(f"\nSaved run to {run_dir}")


if __name__ == "__main__":
    main()
