"""Rebuild (center, context) pairs from train_corpus.txt with your own window size.

    python scripts/build_pairs.py --window 3

Writes data/processed/pairs_window3.csv. To train on it, set this in a config:
    pairs_file: data/processed/pairs_window3.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_PROCESSED, DATA_RAW
from src.corpus_pairs import build_pairs, read_corpus


def main():
    parser = argparse.ArgumentParser(description="Build skip-gram pairs from the corpus.")
    parser.add_argument("--window", type=int, default=2, help="tokens on each side of the center")
    args = parser.parse_args()

    segments = read_corpus(DATA_RAW / "train_corpus.txt")
    pairs = build_pairs(segments, args.window)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED / f"pairs_window{args.window}.csv"
    pairs.to_csv(out_path, index=False)

    n_tokens = sum(len(s) for s in segments)
    print(f"{len(segments)} segments, {n_tokens:,} tokens -> {len(pairs):,} pairs (window={args.window})")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
