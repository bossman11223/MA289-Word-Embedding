"""Project paths and experiment configs.

Every tunable setting lives in a YAML file in configs/. Any setting the YAML
leaves out falls back to DEFAULTS below, so a config only has to list what it
changes.
"""
import copy
from pathlib import Path

import yaml

# This file is src/config.py, so the project root is two levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RUNS_DIR = PROJECT_ROOT / "runs"

DEFAULTS = {
    "name": "baseline",
    # Training pairs: context_train.csv, or a file made by scripts/build_pairs.py.
    # Relative paths are relative to the project root.
    "pairs_file": "data/raw/context_train.csv",
    "test_file": "data/raw/test.csv",
    "val_fraction": 0.1,       # share of pairs held out to measure validation loss
    "embedding_dim": 24,
    "hidden_dim": 0,           # 0 = no hidden layer (embedding -> output)
    "epochs": 3,
    "batch_size": 1024,
    "optimizer": "adam",       # "adam" or "sgd"
    "lr": 0.01,
    "momentum": 0.0,           # only used by sgd
    "weight_decay": 0.0,
    "seed": 289,
    "device": "auto",          # "auto", "cpu", or "cuda"
    "mapping": {
        "name": "shift",       # see src/similarity.py: shift, clip, power, rescale
        "power": 1.0,          # used by "power"
        "target_mean": 0.5,    # used by "rescale"
        "target_std": 0.2,     # used by "rescale"
    },
}


def load_config(path):
    """Read a YAML config and fill in any missing keys from DEFAULTS."""
    with open(path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}

    unknown = set(user_cfg) - set(DEFAULTS)
    if unknown:
        # Catches typos like "learning_rate" that would otherwise be ignored silently.
        raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")

    cfg = copy.deepcopy(DEFAULTS)
    for key, value in user_cfg.items():
        if key == "mapping":
            cfg["mapping"].update(value or {})
        else:
            cfg[key] = value
    return cfg


def save_config(cfg, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def resolve_path(path_str):
    """Turn a config path into an absolute Path (relative = relative to project root)."""
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path
