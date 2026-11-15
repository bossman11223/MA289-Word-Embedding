"""Make a new submission from a saved run with a different cosine -> [0,1] mapping.

No retraining: it reloads embeddings.npy from the run folder.

    python scripts/make_submission.py --run runs/20261114_170500_baseline --mapping power --power 0.5
    python scripts/make_submission.py --run runs/20261114_170500_baseline --mapping rescale --target-mean 0.4 --target-std 0.15

Writes runs/<run>/submission_<mapping>.csv and adds a row to runs/results.csv.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import RUNS_DIR, load_config, resolve_path
from src.data import load_test
from src.embeddings import normalize_rows, pair_cosines
from src.runs import append_results_row, load_run
from src.similarity import MAPPINGS, apply_mapping, describe_mapping
from src.submission import make_submission, save_submission


def main():
    parser = argparse.ArgumentParser(description="Re-map a saved run's cosines and write a submission.")
    parser.add_argument("--run", required=True, help="run folder, e.g. runs/20261114_170500_baseline")
    parser.add_argument("--mapping", choices=sorted(MAPPINGS), help="defaults to the run's config")
    parser.add_argument("--power", type=float)
    parser.add_argument("--target-mean", type=float)
    parser.add_argument("--target-std", type=float)
    args = parser.parse_args()

    run_dir = resolve_path(args.run)
    cfg = load_config(run_dir / "config.yaml")

    # Start from the run's mapping settings and override whatever was passed on the command line.
    mapping_cfg = dict(cfg["mapping"])
    if args.mapping is not None:
        mapping_cfg["name"] = args.mapping
    if args.power is not None:
        mapping_cfg["power"] = args.power
    if args.target_mean is not None:
        mapping_cfg["target_mean"] = args.target_mean
    if args.target_std is not None:
        mapping_cfg["target_std"] = args.target_std

    E, token_to_idx = load_run(run_dir)
    test = load_test(resolve_path(cfg["test_file"]))
    cosines = pair_cosines(normalize_rows(E), token_to_idx, test["token1"], test["token2"])
    similarity = apply_mapping(cosines, mapping_cfg)

    label = describe_mapping(mapping_cfg)
    # Turn e.g. "power(p=0.5)" into a safe file name: submission_power_p0.5.csv
    safe = label.replace("(", "_").replace(")", "").replace("=", "").replace(",", "_")
    out_path = run_dir / f"submission_{safe}.csv"
    save_submission(make_submission(test, similarity), test, out_path)

    history = pd.read_csv(run_dir / "loss_history.csv").to_dict("records")
    append_results_row(RUNS_DIR / "results.csv", run_dir, cfg, history, label, out_path.name)

    print(f"Mapping {label}: mean {similarity.mean():.3f}, std {similarity.std():.3f}, "
          f"min {similarity.min():.3f}, max {similarity.max():.3f}")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
