"""Check any submission CSV before uploading it to Kaggle.

    python scripts/validate_submission.py runs/<run>/submission.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_RAW
from src.submission import validate_file


def main():
    parser = argparse.ArgumentParser(description="Validate a submission against test.csv.")
    parser.add_argument("submission", help="path to the submission CSV")
    parser.add_argument("--test", default=str(DATA_RAW / "test.csv"))
    args = parser.parse_args()

    validate_file(args.submission, args.test)
    print(f"OK: {args.submission} is a valid submission.")


if __name__ == "__main__":
    main()
