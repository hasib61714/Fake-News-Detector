# ml: dataset validation, training, evaluation

Trains and evaluates the classifier the API serves. Two candidates are compared, the better one is
kept, and the result is saved as a **versioned artifact** together with its preprocessing and a full
record of how it was produced.

**No dataset and no trained model are included in this repository.** Until you train one, the API
answers `503 Model not trained yet`. Nothing in this directory reports metrics for you; the numbers
you get come from the data you provide.

## 1. Getting the dataset

The pipeline expects two CSV files, named exactly **`Fake.csv`** and **`True.csv`**, each with a
`text` column (other columns such as `title`, `subject`, `date` are ignored). The label comes from
the file: everything in `Fake.csv` is `fake`, everything in `True.csv` is `real`.

A common public choice is the Kaggle "Fake and Real News Dataset" (ISOT), which ships in exactly
this layout: <https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset>.
**Check the dataset's license and terms on its page before using it**; this repository does not
redistribute it and makes no claim about its license.

```bash
# Option A: download in the browser, unzip, and place the two files:
mkdir -p ml/data
mv ~/Downloads/Fake.csv ~/Downloads/True.csv ml/data/

# Option B: Kaggle CLI (needs your own Kaggle API token configured):
kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset -p ml/data --unzip
```

Final placement:

```
ml/data/Fake.csv
ml/data/True.csv
```

`ml/data/` is git-ignored. Never commit datasets or trained models.

Any other dataset works if you convert it to this layout, or pass one CSV with `text` and `label`
(`fake` | `real`) columns via `--data`. Numeric labels are rejected on purpose: 0/1 conventions
differ between datasets and a silent flip would invert every prediction.

## 2. Validate the files (no training)

```bash
make validate-data                    # reads ml/data/Fake.csv and ml/data/True.csv
make validate-data DATA_DIR=/other/dir
```

Validation is also the first step of every training run. It checks:

| Check | Outcome |
|---|---|
| Both files exist, are UTF-8 CSV, and have a `text` column | otherwise a clear error, nothing is trained |
| Empty / whitespace-only text | dropped and counted |
| Text under 50 characters after cleaning (the API's minimum) | dropped and counted |
| Duplicates within a class (ignoring case, HTML, URLs and a wire dateline) | first copy kept, rest dropped and counted |
| The same article present in **both** files | removed from both and counted (label conflict) |
| Fewer than 50 usable articles in a class | fatal error |
| Strong class imbalance, tiny dataset, or a wire-service marker (`(Reuters)`) concentrated in one class | warning |

The report lists every count per class, the SHA-256 of each source file, and a **dataset
fingerprint** (a hash of the cleaned, deduplicated content, independent of row order). All of it is
saved in the model metadata, so you can tell later exactly which data produced a model.

## 3. Train

```bash
make train                                   # ml/data/Fake.csv + ml/data/True.csv
make train DATA_DIR=/other/dir               # a directory with Fake.csv and True.csv
make train FAKE=path/to/Fake.csv REAL=path/to/True.csv
make train DATA=path/to/articles.csv         # one CSV with text,label columns
```

or `.venv/bin/python -m fakenews_ml.train --help` for all options (`--seed`, `--test-size`,
`--validation-size`, `--cv-folds`, `--min-df`, `--max-features`, `--target-accuracy`,
`--min-coverage`, `--min-threshold`, `--n-jobs`, `--output-dir`, `--no-activate`).

On the full ~45,000-article dataset expect several minutes (bigram TF-IDF, two models, 5-fold
cross-validation, plus calibration for the SVM). `--n-jobs` parallelizes the cross-validation.

### What training does, in order

1. **Split** (stratified, seeded): 60% train, 20% validation, 20% test by default.
2. **Compare** TF-IDF + Logistic Regression against TF-IDF + Linear SVM with stratified k-fold
   cross-validation on the **training split only**. The winner has the highest mean macro-F1; an
   exact tie goes to Logistic Regression. Hyperparameters are fixed defaults (`C=1.0`, balanced
   class weights), not tuned.
   Linear SVM has no probabilities of its own, so it is wrapped in sigmoid calibration, fitted
   on training data only.
3. **Fit** the winner on the training split. It is *not* refitted on train + validation, so the
   validation data stays clean for step 4.
4. **Derive the "uncertain" rule on the validation split** (next section).
5. **Evaluate on the test split exactly once**, after the model and the rule are final. Nothing is
   adjusted afterwards. Automated tests enforce this order and that the stages see only their own
   split.
6. **Write the artifact.**

### The "uncertain" category

The API can answer `fake`, `real` or `uncertain`. A prediction is `uncertain` when
`max(P(fake), P(real)) < threshold`.

The threshold is derived from the **validation split**, never from training or test data:

* candidates run from `--min-threshold` (default 0.60) to 0.99 in steps of 0.01;
* a candidate is admissible only if the model still answers decisively for at least
  `--min-coverage` (default 50%) of validation articles;
* the smallest admissible threshold whose decisive answers are at least `--target-accuracy`
  (default 95%) correct is chosen;
* if no admissible threshold reaches the target, the one with the highest decisive accuracy is
  chosen and the metadata records `"target_met": false`;
* `--min-threshold` is a floor: a prediction below 60% confidence is never called fake or real.

The defaults (95%, 50%, 60%) are policy choices, not learned values. They are recorded in the
metadata; change them if you want a different trade-off. On very clean data the derived threshold
will sit at the floor and few articles will be `uncertain`. That reflects the validation data, and
says nothing about how the model behaves on text unlike it.

## 4. Evaluation

`make train` prints, and the metadata stores, for the held-out **test** split: per-class precision,
recall and F1, macro averages, accuracy, the classification report, and the confusion matrix
(rows = true class, columns = predicted class, order `fake`, `real`). It also reports the effect of
the uncertainty rule: how many articles were decisive vs uncertain, and the same metrics on the
decisive subset only. Cross-validation scores for both candidates (mean ± std) and the validation
results are stored too.

```bash
make report        # re-print everything recorded for the active model
```

## 5. The artifact

Each run writes an immutable release and (unless `--no-activate`) copies it to the active location:

```
ml/artifacts/releases/<model_version>/model.joblib
ml/artifacts/releases/<model_version>/model.metadata.json
ml/artifacts/model.joblib              <- the API loads this
ml/artifacts/model.metadata.json
```

`model_version` is `<UTC timestamp>-<first 8 hex of the model's SHA-256>`. The `.joblib` file is one
scikit-learn `Pipeline`: **preprocessing → TF-IDF → classifier**, so cleaning and the model cannot
drift apart. The metadata records the SHA-256 of the model file, the preprocessing version and
description, Python / scikit-learn / numpy / joblib versions, the seed and all settings, the dataset
report and fingerprint, split sizes, cross-validation results, the uncertainty rule, and the
validation and test evaluations.

### Loading

`fakenews_ml.artifacts.load_artifact(path)` (used by the API) refuses an artifact when:

* the metadata file is missing, unreadable, or has an unsupported `schema_version`;
* the model file's SHA-256 differs from the one in the metadata (mismatched or tampered pair);
* the preprocessing version differs from the installed code (retrain after changing
  `preprocessing.py` and bump `PREPROCESSING_VERSION`);
* the labels are not exactly `fake` / `real`, the object is not a classifier, or the threshold is
  not a number in [0.5, 1].

A scikit-learn version different from the training one is logged as a warning, not an error;
`constraints.txt` pins the tested versions.

### Preprocessing

`fakenews_ml/preprocessing.py`: unescape HTML entities, NFKC-normalize, strip HTML tags, URLs and
e-mail addresses, remove a leading ALL-CAPS wire dateline such as `WASHINGTON (Reuters) -`, collapse
whitespace. Lowercasing, English stop words, 1-2-grams and TF-IDF weighting happen in the
vectorizer. The dateline rule exists because some public datasets contain that marker almost only
in real articles, which would teach the model to recognize a publisher instead of content.

## Honest caveats

* Test-set metrics describe **the same dataset distribution** you trained on. They are not evidence
  the model detects misinformation in the wild, and a model that scores very high on a public
  dataset can still fail on articles from other sources, topics or years. Inspect your data and
  your errors.
* The classifier learns writing patterns, not facts. It is not a fact-checker.
* Duplicate and dateline handling reduces known leakage but cannot remove every publisher-specific
  artifact (for example, `subject` columns and boilerplate phrases).
* Hyperparameters are not tuned and only two model families are compared.
* The validation set sets the uncertainty threshold; with a small validation split that threshold
  is noisy (the report warns on small datasets).

## Security

`joblib` artifacts are pickles. Loading one can execute arbitrary code. Only serve artifacts you
trained yourself or fully trust, and never let users control `MODEL_PATH`. The SHA-256 check
detects accidental mismatch, not a malicious attacker who can also edit the metadata.

## Tests

```bash
make test-ml
```

The tests build tiny templated corpora in code (disjoint vocabularies, no real news) purely to
exercise the code paths. They check dataset validation, preprocessing, splitting, selection,
the uncertainty rule, hand-computed metrics, artifact integrity checks, reproducibility with a fixed
seed, and that the test split is isolated from selection and threshold derivation.
