"""Ways to turn cosine similarity (in [-1, 1]) into a Kaggle score in [0, 1].

We have no labels, so which mapping is best is an experiment: submit several
and compare their Kaggle RMSE. Each function takes a NumPy array of cosines
plus the mapping settings from the config. apply_mapping() always clips the
final result to [0, 1].
"""
import numpy as np


def shift(cos, params):
    """(cos + 1) / 2: maps -1 -> 0, 0 -> 0.5, 1 -> 1. The notebook baseline."""
    return (cos + 1.0) / 2.0


def clip(cos, params):
    """Treat negative cosine as 'unrelated' (0) and keep positive values as-is."""
    return np.clip(cos, 0.0, 1.0)


def power(cos, params):
    """clip(cos, 0, 1) ** p. p < 1 pushes scores up, p > 1 pushes them down."""
    return np.clip(cos, 0.0, 1.0) ** params["power"]


def rescale(cos, params):
    """Linearly rescale so the scores have the chosen mean and standard deviation.

    Standardize (z-scores), then stretch to target_std and shift to target_mean.
    Keeps the ordering of the pairs; only changes the spread and center.
    """
    std = cos.std()
    if std == 0:
        return np.full_like(cos, params["target_mean"])
    z = (cos - cos.mean()) / std
    return params["target_mean"] + params["target_std"] * z


MAPPINGS = {
    "shift": shift,
    "clip": clip,
    "power": power,
    "rescale": rescale,
}


def apply_mapping(cos, mapping_cfg):
    """Look up the mapping by name, apply it, and clip to [0, 1]."""
    name = mapping_cfg["name"]
    if name not in MAPPINGS:
        raise ValueError(f"Unknown mapping {name!r}. Choose from {sorted(MAPPINGS)}")
    scores = MAPPINGS[name](np.asarray(cos, dtype=float), mapping_cfg)
    return np.clip(scores, 0.0, 1.0)


def describe_mapping(mapping_cfg):
    """Short text label for results.csv, e.g. 'power(p=0.5)'."""
    name = mapping_cfg["name"]
    if name == "power":
        return f"power(p={mapping_cfg['power']})"
    if name == "rescale":
        return f"rescale(mean={mapping_cfg['target_mean']},std={mapping_cfg['target_std']})"
    return name
