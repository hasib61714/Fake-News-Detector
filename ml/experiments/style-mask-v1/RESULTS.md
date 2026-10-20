# Results record: style-masking ablation (style-mask-v1)

This is a record of one run of the experiment specified in [`PROTOCOL.md`](PROTOCOL.md), written
after the results existed. The protocol was frozen first (SHA-256
`7e7369897d6e4c9fdf533e974b33431fd47a2e4a0a8b95f2d7836302c881b47d`, 2026-10-20 02:49:43 UTC) and is
unchanged. The result files themselves live in the git-ignored `ml/artifacts/experiments/` tree
together with the models; only aggregate numbers are reproduced here.

**These numbers come from a random split of one dataset (two source groups, 2015 to 2018). They
describe how much the model depends on the masked tokens within that data. They say nothing
about accuracy on other sources or about whether any article is true.**

## What was run

```bash
.venv/bin/python -m fakenews_ml.ablation --level both     # about 28 minutes, exit code 0
```

| Item | Value |
|---|---|
| Data | Kaggle ISOT `Fake.csv` / `True.csv`; fingerprint `509386809ca6b554fa2fbd88b66bd2e60ff32756ad613832e72d224a018bd4ef`; terms in `ml/data/PROVENANCE.md` (CC BY-NC-SA 4.0, non-commercial) |
| Split and seed | stratified 60 / 20 / 20, seed 42: train 23,088, validation 7,696, test 7,697 |
| Baseline | `20261020T023658Z-c58382c4` (calibrated Linear SVM); model SHA-256 `c58382c4019b585e98e5db96f34a04301d1d0e6143378e587d129d241fb6f8e6`, metadata SHA-256 `5fd879a1572e27d0632c66df9d9dee9fb3058e4776c6358338f01007139e3255`; byte-identical before and after the run |
| A1 `core` | model `20261020T030142Z-8b09f184` (calibrated Linear SVM), trained 2026-10-20 03:01:42 UTC |
| A2 `extended` | model `20261020T031806Z-59f5027e` (calibrated Linear SVM), trained 2026-10-20 03:18:06 UTC |
| Code | `stylemask.py` `89568231e093…`, `ablation.py` `e074ff4d21a9…`, `models.py` `acc67fec5e7d…`, `selection.py` `e354c2fd9ea4…`, `train.py` `251b7a28f3e0…` (SHA-256 prefixes at freeze time; identical when the run finished) |

Checks the runner made before training, for both levels: the dataset fingerprint matched the
baseline, and re-scoring the baseline on the reconstructed test split reproduced its recorded
confusion matrix exactly (so the split is the same). Each level was run exactly once.

### Transparency notes

* The test split is scored once per model **for selection purposes**. To compute the paired
  statistics the runner re-derived per-row predictions from the same frozen models (and re-scored
  the baseline once to verify the split). No choice depended on those predictions.
* The extended word lists were chosen before any ablation result existed, partly from the
  baseline's largest weights (see the protocol). Two variants share one test split and one seed,
  and the p-values below are not corrected for multiple looks.
* The rules were sanity-checked on eight hand-written sentences before freezing, which found and
  fixed one bug (plural weekdays). The tests added afterwards corrected three wrong expectations in
  the tests themselves, never a rule.

## Results (verbatim output of the runner)

### A1 `core`: said-family, weekdays, "Reuters", contractions

```text
STYLE-MASK ABLATION (core)   protocol sha256 7e7369897d6e4c9f…
Same data (fingerprint 509386809ca6b554…), seed 42, splits {'train': 23088, 'validation': 7696, 'test': 7697, 'stratified': True}
Baseline model 20261020T023658Z-c58382c4 (linear_svm); ablation model 20261020T030142Z-8b09f184 (linear_svm).

Cross-validation macro-F1 on the TRAINING split:
  logistic_regression  baseline 0.9771 ± 0.0023   ablation 0.9704 ± 0.0023
  linear_svm           baseline 0.9827 ± 0.0024   ablation 0.9768 ± 0.0018

HELD-OUT TEST SET (n=7697):     baseline   ablation      diff
  accuracy                        0.9792     0.9748   -0.0044
  macro F1                        0.9789     0.9745   -0.0045
  fake precision                  0.9902     0.9869   -0.0033
  fake recall                     0.9633     0.9566   -0.0066
  fake f1                         0.9766     0.9715   -0.0050
  real precision                  0.9707     0.9655   -0.0052
  real recall                     0.9922     0.9896   -0.0026
  real f1                         0.9813     0.9774   -0.0039

Confusion matrix, BASELINE (rows = true, columns = predicted):
                  pred fake  pred real
  true fake       3,332        127
  true real          33      4,205
Confusion matrix, ABLATION:
                  pred fake  pred real
  true fake       3,309        150
  true real          44      4,194

Uncertain rule on test: baseline 90 uncertain (threshold 0.60); ablation 161 uncertain (threshold 0.60).

Paired comparison on the same test rows:
  only baseline correct: 60   only ablation correct: 26   McNemar exact p = 0.000317
  accuracy_diff: -0.0044  (95% bootstrap CI -0.0069 to -0.0019, 2000 resamples)
  macro_f1_diff: -0.0045  (95% bootstrap CI -0.0070 to -0.0020, 2000 resamples)

Largest-weight features of the ABLATION model (positive = real):
  toward real: president donald (+4.12), nov (+2.25), reporters (+2.12), spokesman (+2.12), republican (+1.96), statement (+1.95), president barack (+1.94), news conference (+1.66), minister (+1.63), spokeswoman (+1.54), democratic (+1.52), capital (+1.43), president (+1.41), britain (+1.40), house representatives (+1.30)
  toward fake: just (-3.42), read (-3.24), gop (-2.93), watch (-2.67), president trump (-2.38), obama (-2.36), president obama (-2.34), america (-2.18), rep (-1.96), fact (-1.95), sen (-1.91), trump (-1.81), like (-1.81), video (-1.79), hillary (-1.71)

Baseline files unchanged during the run: True
```

### A2 `extended`: A1 plus attribution vocabulary, months, outlets, web-page words, intensifiers

```text
STYLE-MASK ABLATION (extended)   protocol sha256 7e7369897d6e4c9f…
Same data (fingerprint 509386809ca6b554…), seed 42, splits {'train': 23088, 'validation': 7696, 'test': 7697, 'stratified': True}
Baseline model 20261020T023658Z-c58382c4 (linear_svm); ablation model 20261020T031806Z-59f5027e (linear_svm).

Cross-validation macro-F1 on the TRAINING split:
  logistic_regression  baseline 0.9771 ± 0.0023   ablation 0.9641 ± 0.0016
  linear_svm           baseline 0.9827 ± 0.0024   ablation 0.9708 ± 0.0022

HELD-OUT TEST SET (n=7697):     baseline   ablation      diff
  accuracy                        0.9792     0.9691   -0.0101
  macro F1                        0.9789     0.9686   -0.0103
  fake precision                  0.9902     0.9838   -0.0064
  fake recall                     0.9633     0.9468   -0.0165
  fake f1                         0.9766     0.9649   -0.0116
  real precision                  0.9707     0.9579   -0.0128
  real recall                     0.9922     0.9873   -0.0050
  real f1                         0.9813     0.9723   -0.0090

Confusion matrix, BASELINE (rows = true, columns = predicted):
                  pred fake  pred real
  true fake       3,332        127
  true real          33      4,205
Confusion matrix, ABLATION:
                  pred fake  pred real
  true fake       3,275        184
  true real          54      4,184

Uncertain rule on test: baseline 90 uncertain (threshold 0.60); ablation 170 uncertain (threshold 0.60).

Paired comparison on the same test rows:
  only baseline correct: 112   only ablation correct: 34   McNemar exact p = 6.42e-11
  accuracy_diff: -0.0101  (95% bootstrap CI -0.0135 to -0.0073, 2000 resamples)
  macro_f1_diff: -0.0103  (95% bootstrap CI -0.0137 to -0.0074, 2000 resamples)

Largest-weight features of the ABLATION model (positive = real):
  toward real: president donald (+4.54), president barack (+2.13), republican (+2.11), news conference (+1.78), minister (+1.77), democratic (+1.74), capital (+1.51), president (+1.50), britain (+1.45), representatives (+1.45), representative (+1.44), house representatives (+1.40), ministry (+1.40), rival (+1.38), democrat hillary (+1.36)
  toward fake: gop (-3.24), president obama (-2.59), president trump (-2.57), obama (-2.51), america (-2.42), fact (-2.21), sen (-2.15), rep (-2.09), trump (-1.93), hillary (-1.84), mr (-1.73), american (-1.57), today (-1.49), know (-1.47), isis (-1.44)

Baseline files unchanged during the run: True
```

## How this reads against the thresholds fixed in the protocol

| Variant | Macro-F1 drop | Pre-registered reading |
|---|---|---|
| A1 `core` | 0.45 points | at most 1 point: these cues are **not** the main driver of the baseline's score |
| A2 `extended` | 1.03 points | more than 1 and up to 5 points: these cues **contribute**, but other signals carry most of the separation |

A2 is only 0.03 points past the 1-point line; the thresholds are conventions, so read it as
"a small contribution". Both differences are statistically detectable (the bootstrap intervals
exclude 0; McNemar p = 3.2e-4 and 6.4e-11), and both are small in size. After masking, the models
also become less confident: 161 and 170 test articles are called `uncertain`, against 90 for the
baseline.

## What the masked models still use

How often each masked category occurs in the **training** articles (descriptive only; computed
after the runs, no rule was changed because of it):

| Category | Fake | Real |
|---|---|---|
| said-family | 76.6% | 96.4% |
| weekdays | 41.1% | 93.5% |
| "Reuters" | 1.0% | 22.1% |
| negative contractions | 70.4% | 25.8% |
| clitics | 46.6% | 22.1% |
| attribution vocabulary (A2) | 59.5% | 77.2% |
| months (A2) | 26.5% | 54.2% |
| outlet names (A2) | 34.4% | 11.1% |
| page furniture (A2) | 69.7% | 21.7% |
| informal intensifiers (A2) | 80.9% | 34.3% |
| characters removed (core / extended) | 1.39% / 3.46% | 2.20% / 3.67% |

These categories differ sharply between the classes, yet deleting them costs only 0.5 to 1 point.
The classes remain separable through other, redundant signals. The strongest weights of the masked
models are:

* **toward real:** `president donald`, `president barack`, `republican`, `news conference`,
  `minister`, `britain`, `capital`, `representatives` (formal naming and international or
  institutional topics);
* **toward fake:** `gop`, `president trump`, `president obama`, `america`, `fact`, `sen`, `rep`,
  `mr`, `hillary` (abbreviated titles, US political-commentary vocabulary).

So the model mostly separates the two publisher groups by naming conventions, topic and vocabulary.
It does not detect false claims.

## Limits of this comparison

All limits in the protocol apply: same distribution, not a causal estimate, remaining signals not
removed, style and publisher confounded by construction, one split and one seed. In addition, a
masking ablation can show that removing a given set of cues costs little; it cannot show that the
model is free of publisher leakage, because many correlated cues were left in place.

## What this does not establish

External validity (no other dataset was evaluated; that is a separate phase that first needs a
dataset with a verified license and a contamination check, see `fakenews_ml/crosscheck.py`),
real-world performance, or truth verification of any article.
