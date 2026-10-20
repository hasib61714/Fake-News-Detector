# ml: model training

Trains the classifier the API serves: a scikit-learn `Pipeline` of `TfidfVectorizer` (1-2 grams, English stop words, sublinear TF) into `LogisticRegression` (class-balanced). It uses only plain scikit-learn objects, so the API can load the artifact without importing this package.

**No dataset is included in this repository and no trained model is committed.** Until you train one, the API answers `503 Model not trained yet`.

## Dataset format

Either form works. Labels must be the strings `fake` or `real`. Numeric labels are rejected on purpose, because 0/1 conventions differ between datasets and a silent flip would invert every prediction.

1. One CSV with `text` and `label` columns:
   ```csv
   text,label
   "Full article body ...",real
   "Another article ...",fake
   ```
2. Two CSVs, one per class, each with a `text` column (e.g. the Kaggle "Fake and Real News" `Fake.csv` / `True.csv`). The file determines the label.

Empty texts and exact duplicate texts are dropped. Each class needs at least 10 examples after cleaning.

Put files under `ml/data/` if you like; that directory is git-ignored. Check the license of any dataset you use.

## Train

From the repository root, after `make setup`:

```bash
make train DATA=ml/data/articles.csv
make train FAKE=ml/data/Fake.csv REAL=ml/data/True.csv
```

Or call the CLI directly (`.venv/bin/python -m fakenews_ml.train --help`): `--text-column`, `--output-dir`, `--test-size` (default 0.2), `--min-df`, `--seed` (default 42).

Training holds out a stratified test split, reports held-out accuracy, and writes:

| File | Content |
|---|---|
| `ml/artifacts/model.joblib` | The fitted pipeline; `classes_` are exactly `fake` and `real`. |
| `ml/artifacts/model.metadata.json` | `model_name`, `model_version`, `trained_at`, `labels`, `sklearn_version`, `seed`, dataset counts, held-out metrics. |

The model is written last and atomically, so a running API never sees a half-written file.

## Honest caveats

- Reported metrics are held-out accuracy on **the same dataset distribution**. They are not evidence the model detects misinformation in the wild. Some popular datasets contain shortcuts (for example, real articles beginning with a wire-service prefix such as "(Reuters)"), which inflate scores. Inspect your data.
- The classifier learns writing patterns, not facts. It is not a fact-checker.
- Only logistic regression is trained for now. The README's planned multi-algorithm comparison is not implemented yet.

## Security

`joblib` artifacts are pickles. Loading one can execute arbitrary code. Only serve artifacts you trained yourself or fully trust, and never let users control `MODEL_PATH`.

## Tests

```bash
make test-ml
```

The tests use a tiny synthetic corpus generated in code, purely to exercise the training code paths.
