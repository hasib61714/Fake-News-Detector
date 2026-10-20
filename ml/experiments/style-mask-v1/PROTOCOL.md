# Experiment protocol: style-masking ablation (style-mask-v1)

Written **before any ablation model was trained or scored**. The ablation runner records this
file's SHA-256 in every result, so an edit made after the fact is detectable. Changing anything
below after seeing results requires a new protocol version (`style-mask-v2`), not an edit.

## Question

The baseline model (TF-IDF + calibrated Linear SVM, preprocessing v2) scores 97.9% accuracy on a
held-out test split of the Kaggle ISOT data. Inspection of its weights showed that its largest
features are mostly writing-style and attribution cues, not claims: `said`, weekday names,
`reuters`, `told reuters`, formal naming (`president donald`), informal contractions (`isn`,
`didn`, `doesn`), outlet names and web-page words.

**How much of the baseline's performance depends on those cues?** The experiment deletes the
cues, retrains with everything else identical, and compares.

**Hypothesis (stated in advance):** deleting these tokens lowers held-out accuracy and macro-F1.
The size of the drop is unknown. The alternative outcome (little or no drop) would mean the
classes remain separable through other cues (topic, vocabulary, other source differences), and
would be reported just as plainly.

## Fixed design (identical for baseline and ablation)

| Item | Value |
|---|---|
| Data | `ml/data/Fake.csv`, `True.csv`; validation fingerprint `509386809ca6b554fa2fbd88b66bd2e60ff32756ad613832e72d224a018bd4ef` |
| Split | stratified 60 / 20 / 20 train / validation / test, seed 42: train 23,088, validation 7,696, test 7,697 |
| Preprocessing | version 2 cleaning (unchanged), then the style mask (ablation only) |
| Candidates | TF-IDF (1-2-grams, English stop words, `min_df=2`, 100,000 features, sublinear TF) + Logistic Regression, or + calibrated Linear SVM; same fixed hyperparameters |
| Selection | 5-fold stratified cross-validation on the **training split only**, highest mean macro-F1, ties to Logistic Regression |
| Uncertain rule | derived on the **validation split**: target decisive accuracy 0.95, minimum coverage 0.5, floor 0.60 |
| Test split | scored once per model after everything is fixed; no choice depends on it |
| Baseline | model `20261020T023658Z-c58382c4`, model SHA-256 `c58382c4019b585e98e5db96f34a04301d1d0e6143378e587d129d241fb6f8e6`, metadata SHA-256 `5fd879a1572e27d0632c66df9d9dee9fb3058e4776c6358338f01007139e3255`. It is never modified; hashes are checked before and after. |

Safeguards in the runner: it aborts unless the dataset fingerprint matches the baseline, and
unless re-scoring the baseline on the reconstructed test split reproduces the baseline's recorded
confusion matrix exactly (proof the split is the same). Each variant is run once; results are
never overwritten. Ablation models carry a preprocessing version the API loader refuses.

## Masking rules (frozen)

Tokens are **deleted** (not replaced by a placeholder, which would let the model learn how many
masked words an article had). Matching is whole-word and case-insensitive. Source:
`ml/fakenews_ml/stylemask.py`, `STYLE_MASK_VERSION = "1"`.

### Variant A1: `core` (exactly the four categories named for this experiment)

| Rule | Deletes |
|---|---|
| `said_family` | say, says, said, saying, told |
| `weekdays` | Monday ... Sunday (and plurals) |
| `reuters` | every mention of "Reuters" (covers "told Reuters", "Reuters/Ipsos", "Thomson Reuters") |
| `negative_contractions` | isn't, aren't, wasn't, weren't, doesn't, didn't, don't, hasn't, haven't, hadn't, couldn't, wouldn't, shouldn't, mustn't, needn't, ain't, can't, won't, cannot, with or without the apostrophe (`doesn t`, because the fake articles have apostrophes stripped) |
| `clitics` | 're, 've, 'll, 'd, 'm and the stripped forms `re`, `ve`, `ll` |

### Variant A2: `extended` (A1 plus further style and format categories)

| Rule | Deletes |
|---|---|
| `attribution_vocabulary` | according, report(s), reported, reportedly, reporter(s), spokesman, spokeswoman, spokesperson, statement, comment(s), citing, cited, quoted, added, stated, declined, confirmed |
| `months` | January ... December and Jan, Feb, Apr, Jun, Jul, Aug, Sep(t), Oct, Nov, Dec (not "May" or "March", which are ordinary words) |
| `outlet_names` | Breitbart, CNN, MSNBC, Politico, HuffPost, Huffington Post, BBC, NYPost, Fox News, NBC News, ABC News, CBS News, Daily Mail, New York Post, New York Times, Washington Post, Associated Press |
| `page_furniture` | read, watch, video(s), share, subscribe, via, image(s), photo(s), pic, twitter, facebook, youtube, instagram, com |
| `informal_intensifiers` | just, like, actually, apparently, really, literally, totally, basically, obviously, entire, entirely, simply |

### Where the rules came from, and what was and was not tuned

* A1's categories were named for this experiment (the baseline's leakage review); the word lists
  inside them are standard English forms of those categories.
* A2's extra categories were chosen from the baseline model's largest weights (a model fitted on
  the training split only, inspected on 2026-10-20) and general knowledge of newsroom style.
  Because the baseline was scored on the test split once before, the test split's existence was
  known, but **no ablation model had been trained and no ablation result of any kind had been
  seen** when these lists were frozen. No threshold below was chosen from results.
* The rules were run on eight hand-written sentences to check their mechanics. That found and
  fixed one bug (a plural weekday such as "Mondays" slipped through). No dataset text was used to
  adjust a list.
* A2 deliberately includes some words that are ordinary content (`report`, `like`, `read`). It is
  a stricter, blunter bound, and its drop will over-state the effect of pure style.

## What will be reported (all of it, whatever the outcome)

Both variants, baseline versus ablation, on the same 7,697 test rows: accuracy, macro-F1, per-class
precision / recall / F1, both confusion matrices, number of `uncertain` predictions, cross-validation
scores of both candidates, the exact McNemar test on the discordant pairs, a paired bootstrap 95%
interval (2,000 resamples, seed 0) for the accuracy and macro-F1 differences, the largest-weight
features of each ablation model, and the share of training articles containing each masked category
per class.

### How the result will be read (fixed in advance; the thresholds are conventions, not laws)

Using the drop in macro-F1 (baseline minus ablation):

| Drop | Reading |
|---|---|
| 1 point or less | these cues are **not** the main driver; the classes separate through other signals |
| more than 1 and up to 5 points | these cues **contribute**, but other signals carry most of the separation |
| more than 5 points | these cues are a **major** part of the baseline's apparent skill |

A statistically detectable difference (the bootstrap interval excludes 0) is reported separately
from its size. With 7,697 test rows even small differences can be detectable.

## Limitations stated in advance

1. **Same distribution.** The test split comes from the same two source groups and years as the
   training data. Ablation measures dependence on certain tokens, not real-world accuracy.
2. **Not a causal estimate.** Deleting tokens also removes some non-style content and changes
   the n-grams that remain (neighbors of a deleted word become adjacent).
3. **Remaining signals are not removed**: topic, vocabulary, formatting, sentence structure,
   other publisher habits. A model that stays accurate may simply be using those.
4. **A masked model is not a better model.** A lower score does not make a masked model worse, and a
   high score does not make it trustworthy.
5. **Two variants share one test split**, so the two comparisons are not independent and p-values
   are not corrected for multiple looks. One seed, one split.
6. **The comparison cannot separate "style" from "publisher identity"**: the two classes come from
   different publishers, so the two are confounded in this dataset by construction.
7. Nothing here verifies factual claims. The classifier never reads facts.

## Explicitly not done

No rule or threshold tuning after seeing results, no re-runs, no additional variants without a
new protocol version, no evaluation on any other dataset in this experiment (generalization is a
separate phase), no commit or push of datasets or model artifacts.

## Commands

```bash
cd ml/..                                  # repository root
.venv/bin/python -m fakenews_ml.ablation --level core
.venv/bin/python -m fakenews_ml.ablation --level extended
# results: ml/artifacts/experiments/style-mask-v1/<level>/{results.json,comparison.txt,...}
```

## Data terms

Same dataset and terms as the baseline: CC BY-NC-SA 4.0, non-commercial research and portfolio
use only; see `ml/data/PROVENANCE.md`. Ablation models are not distributed and stay in the
git-ignored `ml/artifacts/` tree.
