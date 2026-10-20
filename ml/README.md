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

**License: CC BY-NC-SA 4.0 (non-commercial, share-alike, attribution required), as stated by the
project owner from the Kaggle page. It was not independently verified by this repository's tooling.**
Do not use the data, or models trained on it, commercially. Source, license conditions, attribution
and citation, and the SHA-256 of the exact files used are recorded in
[`data/PROVENANCE.md`](data/PROVENANCE.md). This repository does not redistribute the dataset.

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

`ml/data/` is git-ignored (only `PROVENANCE.md` is tracked). Never commit datasets or trained models.

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
| Strong class imbalance, tiny dataset, or a wire-service dateline concentrated in one class | warning |
| Any monitored source marker (see "Remaining leakage") whose share differs by 20 percentage points or more between classes **after cleaning** | warning, plus a per-class table in the report |

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

### Preprocessing (version 2)

`fakenews_ml/preprocessing.py` is the **only** cleaning code. It is the first step of the saved
pipeline, so training, evaluation and the API run exactly the same function, and dataset
validation de-duplicates with it too. The artifact records `PREPROCESSING_VERSION` (currently
`"2"`) and the loader refuses any artifact with another version, so a model trained with
preprocessing v1 will not be served. Changing the output of any rule means bumping the version and
retraining.

Generic cleaning: unescape HTML entities, NFKC-normalize, strip HTML tags, URLs and e-mail
addresses, collapse whitespace. Lowercasing, English stop words, 1-2-grams and TF-IDF weighting
happen in the vectorizer.

Boilerplate removal (v2). Every rule is anchored to a specific form (start of text, end of text,
or an exact template) and none deletes a bare word:

| Rule | Removes | Anchoring |
|---|---|---|
| `wire_dateline` | `WASHINGTON (Reuters) -`, `XIAMEN, China (Reuters) -`, `NEW YORK/LONDON (Reuters) -`, `(AP) -` | start of text; place must start with an ALL-CAPS word |
| `wire_tag` | a stray `(Reuters) -` tag elsewhere (not `(AP)` / `(AFP)`, which are ordinary abbreviations) | the exact tag |
| `credit_line` | `(Reporting by Jane Doe; Editing by John Roe)` | parenthetical starting with a credit verb and a capitalized name |
| `sponsor_footer` | Reuters' two fixed "funded in part by SAP ... independently created by the Reuters editorial staff" footers | exact templates |
| `bare_link` | `pic.twitter.com/..`, `twitter.com/..`, `t.co/..`, `bit.ly/..` and similar links without `http` | domain followed by a path |
| `tweet_attribution` | `(@handle) March 4, 2017` after an embedded tweet (the display name and tweet text stay) | handle in parentheses followed by a full date |
| `image_credit` | trailing `Featured image via Mark Wilson/Getty Images`, `Photo by ...`, `Image via Twitter`, `Featured Image Credit: ...` | last credit in the final 160 characters; only the final credit is removed |
| `wire21_lead` / `wire21_footer` / `wire21_support` | `21st Century Wire says`, `READ MORE ... NEWS AT: 21st Century Wire ... Files`, `SUPPORT 21WIRE SUBSCRIBE ...` | start of text / the exact footer / final 160 characters |

Deliberately **kept**, because they are ordinary content: the word "Reuters" in sentences ("a letter
seen by Reuters", "Reuters/Ipsos poll"), `Getty Images` and `featured image` in running prose,
plain `(@handle)` mentions, `(AP)` / `(AFP)` abbreviations, dates, and "Reporting by ..." in prose.
Unit tests (`tests/test_preprocessing.py`) cover each rule, realistic article text that must come
out unchanged, idempotence, and that the saved pipeline equals "clean, then predict".

The rules were designed by inspecting the full dataset (including rows that later fall into the
test split) for publisher boilerplate. They are fixed, hand-written patterns and nothing is fitted
to labels, but you should know that this inspection happened. Measured effect on the dataset used
here (17,292 fake / 21,189 real articles after validation): the rules remove 2.2% of fake and
1.0% of real characters, the longest single deletion is 304 characters (a credit line), and
an audit of the most common and the three longest deleted spans of every rule found only
boilerplate (it did not read all ~60,000 deletions).

### Remaining leakage

The two classes come from different publishers, so cleaning cannot make them statistically
identical. `make validate-data` measures these markers per class **after** cleaning (shares of
articles) and warns when they differ by 20 points or more. For the Kaggle ISOT files with
preprocessing v2:

| Marker (after cleaning) | Fake | Real | Before v2 (fake / real) |
|---|---|---|---|
| mentions "Reuters" | 1.2% | **22.2%** | 1.3% / 28.0% |
| wire style "said on <weekday>" | 0.7% | **36.7%** | 0.7% / 36.7% |
| `featured image` / `image via` | 1.7% | 0.0% | at least 46.4% / 0.0% |
| `Getty Images` | 0.5% | 0.0% | 22.8% / 0.0% |
| tweet embeds (`pic.twitter`, `(@handle) date`) | 0.0% | 0.0% | at least 25% / 0.0% |
| 21st Century Wire / 21WIRE | 3.0% | 0.0% | 3.6% / 0.0% (mentions only) |
| "Read more" | 7.2% | 0.0% | 9.8% / 0.0% |
| `(Reuters)` tag | 0.0% | 0.0% | 0.0% / 7.6% |

What is still likely to help a model recognize the source rather than the content:

* **The word "Reuters" and wire-agency writing style** (real articles: "said on Monday",
  attribution phrasing, inverted-pyramid structure). These are ordinary content and are not
  removed, so a model can still partly separate the classes by style and by naming the agency.
* **Topic and `subject` differences.** Fake articles skew toward US politics commentary and
  opinion; real ones toward politics and world news. The `subject` column separates the classes
  perfectly and is never used, but topic words carry a similar signal.
* **Residual image and footer variants** that the anchored rules do not match (for example a
  lowercase `image via ...` after a colon, or credits longer than 160 characters): about 1.7% of
  fake articles; and "Read more" links (7.2% of fake).
* **Duplicated fake content.** About 23% of `Fake.csv` rows are duplicates. They are removed before
  splitting, so copies cannot sit on both sides of the split, but the remaining fake articles
  still come from a small number of sites.
* **Time and source coverage.** All articles are from 2015 to 2018 and from a handful of
  publishers, so scores say little about other years, outlets or topics.

Recommended after training (not done yet): inspect the largest-weight features of the model for
publisher words, and evaluate on a differently sourced dataset before drawing any conclusion about
real-world performance.

## Leakage experiments

The baseline's largest weights are mostly writing-style and attribution cues (`said`, weekdays,
`reuters`, contractions). Two tools measure how much that matters. Both are experiments: neither is
part of the served pipeline, and neither writes outside the git-ignored `ml/artifacts/` tree.

**Style-masking ablation** (`fakenews_ml/stylemask.py`, `fakenews_ml/ablation.py`). Retrains with
those tokens deleted, using the same data, split and seed, and compares with the baseline on the
same test rows (per-class precision/recall, confusion matrices, exact McNemar test, paired bootstrap).

```bash
.venv/bin/python -m fakenews_ml.ablation --level core        # or: extended, both
```

* The rules and the way results are read are fixed in
  [`experiments/style-mask-v1/PROTOCOL.md`](experiments/style-mask-v1/PROTOCOL.md) before any run;
  the runner records the protocol's SHA-256 in its results, refuses to run without it, and refuses to
  re-run a level that already has results.
* It aborts unless the dataset fingerprint matches the baseline and re-scoring the baseline on the
  rebuilt test split reproduces its saved confusion matrix. The baseline is never modified (its
  hashes are checked before and after). Ablation models are stored separately with a preprocessing
  version the API loader refuses, so they cannot be served by accident.
* The recorded run, with the verbatim output, is in
  [`experiments/style-mask-v1/RESULTS.md`](experiments/style-mask-v1/RESULTS.md). Short version:
  on this dataset, deleting the four named categories lowered macro-F1 by 0.45 points and the
  stricter extended mask by 1.03 points; the classes stay separable through naming conventions,
  topic and vocabulary. That is evidence about this dataset only, not about real-world performance.

**Contamination check** (`fakenews_ml/crosscheck.py`). Run this before evaluating on any other
dataset. It scores no model; it reports empty and duplicate articles, exact overlap with the
training data (after the pipeline's own cleaning), near-duplicates (TF-IDF cosine at or above
0.90) and shared publishers when a domain column is given.

```bash
.venv/bin/python -m fakenews_ml.crosscheck --external path/to/other.csv --text-column text \
    --domain-column domain
```

An external dataset needs its license and intended use checked first (as for ISOT, see
`data/PROVENANCE.md`) and substantial overlap means it cannot support a claim of external validity.

## Honest caveats

* Test-set metrics describe **the same dataset distribution** you trained on. They are not evidence
  the model detects misinformation in the wild, and a model that scores very high on a public
  dataset can still fail on articles from other sources, topics or years. Inspect your data and
  your errors.
* The classifier learns writing patterns, not facts. It is not a fact-checker.
* Preprocessing removes known boilerplate, but publisher and style signals remain (see "Remaining
  leakage"). Treat any metric on this dataset as optimistic.
* The dataset is licensed CC BY-NC-SA 4.0 (per the owner; see `data/PROVENANCE.md`): research use
  only, no commercial use, and trained models should be treated as subject to the same terms.
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
