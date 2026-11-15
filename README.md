# MA289 Neural Network Word Embedding Competition

Learn embeddings for 900 anonymous tokens (`tok_0001` … `tok_0900`) with a skip-gram style
network in PyTorch, then score 1,200 test token pairs by cosine similarity mapped to [0, 1].
Kaggle scores the submission by RMSE against hidden targets. No pretrained embeddings or external
models are used.

## Approach

1. **Pairs.** Each training example is a (center, context) token pair: either `context_train.csv`
   (~300k pairs) or pairs rebuilt from `train_corpus.txt` with a chosen window size.
2. **Model.** `center ID -> nn.Embedding(V, d) -> nn.Linear(d, V)`, trained with cross-entropy to
   predict the context token. With `hidden_dim > 0` there is an extra `Linear -> ReLU` layer.
3. **Validation.** A random `val_fraction` of the pairs is held out. Train and validation
   cross-entropy are logged every epoch to spot overfitting without using Kaggle.
4. **Embeddings.** After training only the embedding matrix `E` is kept. Rows are L2-normalized,
   so the cosine of two tokens is a dot product.
5. **Mapping to [0, 1].** We have no labels, so how cosine becomes a score is an experiment
   (see `src/similarity.py`). The final output is always clipped to [0, 1].

| Mapping   | Formula                                                    |
|-----------|------------------------------------------------------------|
| `shift`   | `(cos + 1) / 2` (notebook baseline)                        |
| `clip`    | `clip(cos, 0, 1)`                                          |
| `power`   | `clip(cos, 0, 1) ** power`                                  |
| `rescale` | `target_mean + target_std * (cos - mean(cos)) / std(cos)`   |

`rescale` uses the mean and std of the 1,200 test cosines themselves.

## Setup (Windows, PowerShell)

```powershell
cd "C:\Users\zander.bos\Downloads\Github Repos\MA289-Word-Embedding"
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest
```

Put the four Kaggle files in `data/raw/`: `train_corpus.txt`, `context_train.csv`, `test.csv`,
`sample_submission.csv`.

## Usage

```powershell
# Train one experiment (also writes a validated submission.csv)
python scripts/train.py --config configs/baseline.yaml

# Try another mapping on the same embeddings, without retraining
python scripts/make_submission.py --run runs/<run_folder> --mapping power --power 0.5
python scripts/make_submission.py --run runs/<run_folder> --mapping rescale --target-mean 0.45 --target-std 0.15

# Check any submission file before uploading
python scripts/validate_submission.py runs/<run_folder>/submission.csv

# Build pairs from the corpus with your own window, then train on them
python scripts/build_pairs.py --window 3
python scripts/train.py --config configs/window3_power.yaml
```

To make a new experiment, copy a file in `configs/`, give it a new `name`, and change one thing.
Keys you leave out use the defaults in `src/config.py`.

## What each run saves

`runs/<YYYYMMDD_HHMMSS>_<name>/`

| File                  | Contents                                            |
|-----------------------|-----------------------------------------------------|
| `config.yaml`         | the full config used (defaults filled in)           |
| `loss_history.csv`    | epoch, train_loss, val_loss                         |
| `loss_curve.png`      | train vs validation cross-entropy                   |
| `embeddings.npy`      | raw embedding matrix E (V x d)                      |
| `token_to_idx.json`   | token string -> row of E                            |
| `pca.png`             | 2D PCA of 35 random tokens                          |
| `submission.csv`      | submission from the config's mapping                |
| `submission_<map>.csv`| extra submissions from `make_submission.py`         |

Every submission also adds a row to `runs/results.csv` (config summary, final and best
validation loss, mapping, file name). Fill in the `kaggle_score` column by hand after uploading.

## Project structure

```
├── configs/          one YAML per experiment
├── data/raw/         Kaggle files (gitignored)
├── data/processed/   pairs built from the corpus (gitignored)
├── notebooks/        exploration only
├── runs/             run outputs (gitignored except results.csv)
├── scripts/          train, make_submission, validate_submission, build_pairs
├── src/
│   ├── config.py        paths, defaults, YAML loading
│   ├── utils.py         seeding, device
│   ├── data.py          pairs, vocab, train/val split, DataLoaders
│   ├── corpus_pairs.py  windowed pairs from train_corpus.txt
│   ├── model.py         EmbeddingFFNN, DeeperEmbeddingFFNN
│   ├── train.py         optimizer, training loop, validation loss
│   ├── embeddings.py    extract/normalize E, cosines, nearest neighbors, PCA plot
│   ├── similarity.py    cosine -> [0, 1] mappings
│   ├── submission.py    build + validate submission
│   └── runs.py          run folders, loss plot, results.csv
└── tests/            pytest smoke tests on synthetic data
```
