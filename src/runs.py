"""Save everything about a run to runs/<timestamp>_<name>/ and log it in runs/results.csv."""
import csv
import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import save_config

RESULTS_COLUMNS = [
    "run_id", "name", "pairs_file", "embedding_dim", "hidden_dim", "epochs", "batch_size",
    "optimizer", "lr", "weight_decay", "seed", "final_train_loss", "final_val_loss",
    "best_val_loss", "best_epoch", "mapping", "submission_file", "kaggle_score",
]


def create_run_dir(runs_dir, name):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / f"{stamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_run_artifacts(run_dir, cfg, history, E, token_to_idx):
    """Config, loss history (CSV + plot), raw embedding matrix, and token map."""
    save_config(cfg, run_dir / "config.yaml")
    pd.DataFrame(history).to_csv(run_dir / "loss_history.csv", index=False)
    plot_loss(history, run_dir / "loss_curve.png")
    np.save(run_dir / "embeddings.npy", E)   # raw (not normalized) so any later step can choose
    with open(run_dir / "token_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(token_to_idx, f, indent=0)


def load_run(run_dir):
    """Read back the embedding matrix and token map saved by save_run_artifacts."""
    E = np.load(run_dir / "embeddings.npy")
    with open(run_dir / "token_to_idx.json", "r", encoding="utf-8") as f:
        token_to_idx = json.load(f)
    return E, token_to_idx


def plot_loss(history, path):
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [h["train_loss"] for h in history], marker="o", label="train")
    ax.plot(epochs, [h["val_loss"] for h in history], marker="o", label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean cross-entropy")
    ax.set_title("Training vs validation loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def append_results_row(results_path, run_dir, cfg, history, mapping_label, submission_file):
    """Add one row to results.csv. kaggle_score is left blank to fill in by hand."""
    val_losses = [h["val_loss"] for h in history]
    best = int(np.nanargmin(val_losses)) if not np.all(np.isnan(val_losses)) else None

    row = {
        "run_id": run_dir.name,
        "name": cfg["name"],
        "pairs_file": cfg["pairs_file"],
        "embedding_dim": cfg["embedding_dim"],
        "hidden_dim": cfg["hidden_dim"],
        "epochs": cfg["epochs"],
        "batch_size": cfg["batch_size"],
        "optimizer": cfg["optimizer"],
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "seed": cfg["seed"],
        "final_train_loss": round(history[-1]["train_loss"], 5),
        "final_val_loss": round(history[-1]["val_loss"], 5),
        "best_val_loss": round(val_losses[best], 5) if best is not None else "",
        "best_epoch": history[best]["epoch"] if best is not None else "",
        "mapping": mapping_label,
        "submission_file": submission_file,
        "kaggle_score": "",
    }

    write_header = not results_path.exists()
    with open(results_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
