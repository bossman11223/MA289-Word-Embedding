"""Build and check the Kaggle submission file."""
import pandas as pd


def make_submission(test, similarity):
    """DataFrame with exactly [pair_id, similarity] in test.csv order."""
    submission = test[["pair_id"]].copy()
    submission["similarity"] = similarity
    return submission


def validate_submission(submission, test):
    """Raise AssertionError if the submission would be rejected or misaligned by Kaggle."""
    assert list(submission.columns) == ["pair_id", "similarity"], (
        f"columns must be ['pair_id', 'similarity'], got {list(submission.columns)}"
    )
    assert len(submission) == len(test), f"expected {len(test)} rows, got {len(submission)}"
    assert submission["pair_id"].tolist() == test["pair_id"].tolist(), (
        "pair_id values or order do not match test.csv"
    )
    assert not submission["similarity"].isna().any(), "similarity contains NaN"
    assert submission["similarity"].between(0.0, 1.0).all(), "similarity values must be in [0, 1]"


def save_submission(submission, test, path):
    """Validate first, then write, so an invalid file is never saved."""
    validate_submission(submission, test)
    submission.to_csv(path, index=False)


def validate_file(submission_path, test_path):
    """Check a saved CSV against test.csv."""
    validate_submission(pd.read_csv(submission_path), pd.read_csv(test_path))
