"""Style-masking ablation: retrain with leakage-prone tokens deleted and compare to the baseline.

``python -m fakenews_ml.ablation --level core`` (or ``extended``, or ``both``).

Discipline enforced in code, so results cannot be quietly tuned:

* The protocol (``ml/experiments/style-mask-v1/PROTOCOL.md``) must exist; its SHA-256 is recorded
  in the results, so a protocol edited after the run is detectable.
* The dataset fingerprint and the baseline's recorded split/seed/config are reused. Before
  training, the baseline is re-scored on the reconstructed test split and its confusion matrix
  must equal the one saved at baseline time; otherwise the split is not the same and the run aborts.
* A level that already has results is never re-run (no repeated attempts).
* The baseline model and metadata are never written; their hashes are verified before and after.
* Ablation artifacts go to their own directory and carry a preprocessing version the API loader
  refuses, so an experimental model cannot be served by accident.
"""

import argparse
import hashlib
import io
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.stats import binomtest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

from fakenews_ml import __version__
from fakenews_ml.artifacts import LoadedArtifact, load_artifact, metadata_path_for
from fakenews_ml.data import LABELS, Dataset, DatasetError, load_data_dir
from fakenews_ml.evaluation import predict_top
from fakenews_ml.preprocessing import PREPROCESSING_VERSION, clean_text
from fakenews_ml.splits import split_indices, take
from fakenews_ml.stylemask import (
    LEVELS,
    RULES,
    STYLE_MASK_VERSION,
    StyleMaskLevel,
    mask_text,
    rule_hits,
)
from fakenews_ml.train import TrainingConfig, train

ML_DIR = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ML_DIR / "experiments" / "style-mask-v1" / "PROTOCOL.md"
DEFAULT_OUTPUT = ML_DIR / "artifacts" / "experiments" / "style-mask-v1"
DEFAULT_BASELINE = ML_DIR / "artifacts" / "model.joblib"
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0


class AblationError(RuntimeError):
    """A precondition of the experiment is not met; nothing was trained."""


# -- statistics ----------------------------------------------------------------------------------


def mcnemar_exact(baseline_only_correct: int, ablation_only_correct: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts."""
    n = baseline_only_correct + ablation_only_correct
    if n == 0:
        return 1.0
    return float(binomtest(min(baseline_only_correct, ablation_only_correct), n, 0.5).pvalue)


def _macro_f1(confusion: np.ndarray) -> float:
    f1s = []
    for k in (0, 1):
        tp = confusion[k, k]
        fp = confusion[1 - k, k]
        fn = confusion[k, 1 - k]
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s))


def paired_bootstrap(
    y_true: Sequence[str],
    pred_a: Sequence[str],
    pred_b: Sequence[str],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, dict[str, float]]:
    """Paired bootstrap of (B - A) for accuracy and macro-F1 over the SAME resampled test rows."""
    t = np.array([LABELS.index(v) for v in y_true])
    a = np.array([LABELS.index(v) for v in pred_a])
    b = np.array([LABELS.index(v) for v in pred_b])
    rng = np.random.default_rng(seed)
    n = len(t)
    acc_diffs, f1_diffs = [], []
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        ta, aa, bb = t[idx], a[idx], b[idx]
        conf_a = np.bincount(2 * ta + aa, minlength=4).reshape(2, 2)
        conf_b = np.bincount(2 * ta + bb, minlength=4).reshape(2, 2)
        acc_diffs.append((np.trace(conf_b) - np.trace(conf_a)) / n)
        f1_diffs.append(_macro_f1(conf_b) - _macro_f1(conf_a))

    def summarize(values: list[float], observed: float) -> dict[str, float]:
        lo, hi = np.percentile(values, [2.5, 97.5])
        return {"observed": observed, "ci95_low": float(lo), "ci95_high": float(hi)}

    conf_a_full = np.bincount(2 * t + a, minlength=4).reshape(2, 2)
    conf_b_full = np.bincount(2 * t + b, minlength=4).reshape(2, 2)
    return {
        "accuracy_diff": summarize(
            acc_diffs, float((np.trace(conf_b_full) - np.trace(conf_a_full)) / n)
        ),
        "macro_f1_diff": summarize(f1_diffs, _macro_f1(conf_b_full) - _macro_f1(conf_a_full)),
    }


def class_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, Any]:
    labels = list(LABELS)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    confusion = np.bincount(
        [2 * labels.index(t) + labels.index(q) for t, q in zip(y_true, y_pred, strict=True)],
        minlength=4,
    ).reshape(2, 2)
    return {
        "accuracy": float((np.trace(confusion)) / len(y_true)),
        "macro_f1": _macro_f1(confusion),
        "per_class": {
            lab: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(s[i]),
            }
            for i, lab in enumerate(labels)
        },
        "confusion_matrix": {"labels": labels, "matrix": confusion.tolist()},
    }


def top_weights(pipeline: Any, n: int = 25) -> dict[str, Any]:
    """Largest-weight features per class (positive weight = 'real'). Reads weights only."""
    tfidf, clf = pipeline.named_steps["tfidf"], pipeline.named_steps["clf"]
    names = np.asarray(tfidf.get_feature_names_out())
    if isinstance(clf, LogisticRegression):
        coef = clf.coef_[0]
    elif isinstance(clf, CalibratedClassifierCV):
        coef = np.mean([c.estimator.coef_[0] for c in clf.calibrated_classifiers_], axis=0)
    else:
        raise AblationError(f"Unsupported classifier {type(clf).__name__}")
    order = np.argsort(coef)
    return {
        "toward_real": [[str(names[i]), float(coef[i])] for i in order[::-1][:n]],
        "toward_fake": [[str(names[i]), float(coef[i])] for i in order[:n]],
    }


# -- helpers -------------------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _baseline_hashes(model_path: Path) -> dict[str, str]:
    return {
        "model_sha256": _sha256(model_path),
        "metadata_sha256": _sha256(metadata_path_for(model_path)),
    }


def _descriptive_rule_shares(
    texts: Sequence[str], labels: Sequence[str], level: StyleMaskLevel
) -> dict[str, Any]:
    """Share of TRAINING articles containing each masked category, per class (descriptive only)."""
    out: dict[str, Any] = {}
    for lab in LABELS:
        class_texts = [clean_text(t) for t, y in zip(texts, labels, strict=True) if y == lab]
        hits = [rule_hits(t, level) for t in class_texts]
        removed = sum(len(t) - len(mask_text(t, level)) for t in class_texts)
        total = sum(len(t) for t in class_texts)
        out[lab] = {
            "articles": len(class_texts),
            "share_of_articles_with_rule": {
                r.name: sum(h[r.name] > 0 for h in hits) / len(class_texts) for r in RULES[level]
            },
            "share_of_characters_removed": removed / total,
        }
    return out


# -- the experiment ------------------------------------------------------------------------------


def run_level(
    level: StyleMaskLevel,
    *,
    dataset: Dataset,
    baseline: LoadedArtifact,
    baseline_model_path: Path,
    output_dir: Path,
    protocol_path: Path,
    log: Any = print,
) -> dict[str, Any]:
    if not protocol_path.is_file():
        raise AblationError(f"Protocol not found at {protocol_path}; write it before running.")
    out = output_dir / level
    if (out / "results.json").exists():
        raise AblationError(
            f"{out / 'results.json'} already exists. Each level is run once; to repeat the "
            "experiment, move the old results aside deliberately and say why in the protocol."
        )

    before = _baseline_hashes(baseline_model_path)
    meta = baseline.metadata
    if dataset.report.fingerprint != meta["dataset"]["report"]["fingerprint"]:
        raise AblationError("Dataset fingerprint differs from the baseline's; not the same data.")

    base_cfg = dict(meta["config"])
    base_cfg.pop("style_mask", None)
    config = TrainingConfig(**{**base_cfg, "style_mask": level})
    splits = split_indices(dataset.labels, config.test_size, config.validation_size, config.seed)
    x_test, y_test = take(dataset.texts, splits.test), take(dataset.labels, splits.test)
    x_train, y_train = take(dataset.texts, splits.train), take(dataset.labels, splits.train)

    # The reconstructed split must reproduce the baseline's recorded test confusion matrix.
    base_pred, _ = predict_top(baseline.pipeline, x_test)
    recomputed = class_metrics(y_test, base_pred)["confusion_matrix"]
    recorded = meta["evaluation"]["test"]["all_predictions"]["confusion_matrix"]
    if recomputed != recorded:
        raise AblationError(
            "Re-scoring the baseline on the reconstructed test split does not reproduce its "
            f"recorded confusion matrix ({recomputed} vs {recorded}); the split differs."
        )
    log(f"[{level}] split verified: baseline test confusion matrix reproduced exactly")

    protocol_sha = _sha256(protocol_path)
    descriptive = _descriptive_rule_shares(x_train, y_train, level)
    log(f"[{level}] training ablation model (same seed {config.seed}, same split)...")
    artifact = train(dataset, config, log=log)

    pipeline = joblib.load(io.BytesIO(artifact.model_bytes))
    abl_pred, abl_conf = predict_top(pipeline, x_test)
    abl_metrics = class_metrics(y_test, abl_pred)
    recorded_abl = artifact.metadata["evaluation"]["test"]["all_predictions"]["confusion_matrix"]
    if abl_metrics["confusion_matrix"] != recorded_abl:
        raise AblationError("Ablation re-scoring disagrees with its own recorded test evaluation.")
    base_metrics = class_metrics(y_test, base_pred)

    base_correct = np.array([p == t for p, t in zip(base_pred, y_test, strict=True)])
    abl_correct = np.array([p == t for p, t in zip(abl_pred, y_test, strict=True)])
    only_base = int((base_correct & ~abl_correct).sum())
    only_abl = int((~base_correct & abl_correct).sum())

    results: dict[str, Any] = {
        "experiment": "style-mask-v1",
        "level": level,
        "style_mask_version": STYLE_MASK_VERSION,
        "protocol_sha256": protocol_sha,
        "dataset_fingerprint": dataset.report.fingerprint,
        "splits": artifact.metadata["dataset"]["splits"],
        "seed": config.seed,
        "baseline": {
            "model_version": baseline.version,
            "selected": meta["model_selection"]["selected"],
            "cv": {
                c["kind"]: c["metrics"]["f1_macro"] for c in meta["model_selection"]["candidates"]
            },
            "uncertainty_threshold": baseline.threshold,
            "test": base_metrics,
            "test_uncertainty": meta["evaluation"]["test"]["uncertainty"],
            **before,
        },
        "ablation": {
            "model_version": artifact.metadata["model_version"],
            "selected": artifact.metadata["model_selection"]["selected"],
            "cv": {
                c["kind"]: c["metrics"]["f1_macro"]
                for c in artifact.metadata["model_selection"]["candidates"]
            },
            "uncertainty_threshold": artifact.metadata["uncertainty"]["threshold"],
            "validation_uncertainty": artifact.metadata["uncertainty"],
            "test": abl_metrics,
            "test_uncertainty": artifact.metadata["evaluation"]["test"]["uncertainty"],
            "top_weights": top_weights(pipeline),
        },
        "paired_comparison": {
            "n_test": len(y_test),
            "discordant_baseline_only_correct": only_base,
            "discordant_ablation_only_correct": only_abl,
            "mcnemar_exact_p": mcnemar_exact(only_base, only_abl),
            "bootstrap": {
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                **paired_bootstrap(y_test, base_pred, abl_pred),
            },
        },
        "masked_content_descriptive_train_split": descriptive,
    }

    # Persist: own directory, and a preprocessing version the API loader refuses.
    out.mkdir(parents=True, exist_ok=True)
    meta_out = dict(artifact.metadata)
    meta_out["preprocessing"] = {
        **meta_out["preprocessing"],
        "version": f"{PREPROCESSING_VERSION}+style-mask-{level}-v{STYLE_MASK_VERSION}",
        "description": "EXPERIMENTAL ablation pipeline (clean, then style mask). Not servable.",
    }
    meta_out["experiment"] = {
        "name": "style-mask-v1",
        "level": level,
        "protocol_sha256": protocol_sha,
        "baseline_model_sha256": before["model_sha256"],
        "fakenews_ml": __version__,
    }
    (out / "model.joblib").write_bytes(artifact.model_bytes)
    (out / "metadata.json").write_text(json.dumps(meta_out, indent=2, allow_nan=False) + "\n")

    after = _baseline_hashes(baseline_model_path)
    results["baseline_unchanged"] = before == after
    if before != after:
        raise AblationError("The baseline files changed during the experiment!")
    (out / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    (out / "comparison.txt").write_text(format_comparison(results) + "\n")
    return results


# -- reporting -----------------------------------------------------------------------------------


def _fmt_matrix(m: list[list[int]]) -> str:
    return (
        "                  pred fake  pred real\n"
        f"  true fake  {m[0][0]:>10,} {m[0][1]:>10,}\n"
        f"  true real  {m[1][0]:>10,} {m[1][1]:>10,}"
    )


def format_comparison(r: dict[str, Any]) -> str:
    b, a, pc = r["baseline"], r["ablation"], r["paired_comparison"]
    bt, at = b["test"], a["test"]
    lines = [
        f"STYLE-MASK ABLATION ({r['level']})   protocol sha256 {r['protocol_sha256'][:16]}…",
        f"Same data (fingerprint {r['dataset_fingerprint'][:16]}…), seed {r['seed']}, "
        f"splits {r['splits']}",
        f"Baseline model {b['model_version']} ({b['selected']}); ablation model "
        f"{a['model_version']} ({a['selected']}).",
        "",
        "Cross-validation macro-F1 on the TRAINING split:",
    ]
    for kind in b["cv"]:
        lines.append(
            f"  {kind:<20} baseline {b['cv'][kind]['mean']:.4f} ± {b['cv'][kind]['std']:.4f}"
            f"   ablation {a['cv'][kind]['mean']:.4f} ± {a['cv'][kind]['std']:.4f}"
        )
    lines += [
        "",
        f"HELD-OUT TEST SET (n={pc['n_test']}):   {'baseline':>10} {'ablation':>10} {'diff':>9}",
        f"  accuracy                    {bt['accuracy']:>10.4f} {at['accuracy']:>10.4f} "
        f"{at['accuracy'] - bt['accuracy']:>+9.4f}",
        f"  macro F1                    {bt['macro_f1']:>10.4f} {at['macro_f1']:>10.4f} "
        f"{at['macro_f1'] - bt['macro_f1']:>+9.4f}",
    ]
    for lab in LABELS:
        for metric in ("precision", "recall", "f1"):
            x, y = bt["per_class"][lab][metric], at["per_class"][lab][metric]
            lines.append(f"  {lab + ' ' + metric:<27} {x:>10.4f} {y:>10.4f} {y - x:>+9.4f}")
    lines += [
        "",
        "Confusion matrix, BASELINE (rows = true, columns = predicted):",
        _fmt_matrix(bt["confusion_matrix"]["matrix"]),
        "Confusion matrix, ABLATION:",
        _fmt_matrix(at["confusion_matrix"]["matrix"]),
        "",
        f"Uncertain rule on test: baseline {b['test_uncertainty']['n_uncertain']} uncertain "
        f"(threshold {b['uncertainty_threshold']:.2f}); ablation "
        f"{a['test_uncertainty']['n_uncertain']} uncertain "
        f"(threshold {a['uncertainty_threshold']:.2f}).",
        "",
        "Paired comparison on the same test rows:",
        f"  only baseline correct: {pc['discordant_baseline_only_correct']}   "
        f"only ablation correct: {pc['discordant_ablation_only_correct']}   "
        f"McNemar exact p = {pc['mcnemar_exact_p']:.3g}",
    ]
    for name, d in pc["bootstrap"].items():
        if isinstance(d, dict):
            lines.append(
                f"  {name}: {d['observed']:+.4f}  (95% bootstrap CI {d['ci95_low']:+.4f} to "
                f"{d['ci95_high']:+.4f}, {pc['bootstrap']['resamples']} resamples)"
            )
    lines += ["", "Largest-weight features of the ABLATION model (positive = real):"]
    tw = a["top_weights"]
    lines.append(
        "  toward real: " + ", ".join(f"{n} ({w:+.2f})" for n, w in tw["toward_real"][:15])
    )
    lines.append(
        "  toward fake: " + ", ".join(f"{n} ({w:+.2f})" for n, w in tw["toward_fake"][:15])
    )
    lines.append(f"\nBaseline files unchanged during the run: {r['baseline_unchanged']}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--level", choices=[*LEVELS, "both"], required=True)
    parser.add_argument("--data-dir", type=Path, default=ML_DIR / "data")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    args = parser.parse_args(argv)
    try:
        dataset = load_data_dir(args.data_dir)
        baseline = load_artifact(args.baseline)
        levels: list[StyleMaskLevel] = list(LEVELS) if args.level == "both" else [args.level]
        for level in levels:
            results = run_level(
                level,
                dataset=dataset,
                baseline=baseline,
                baseline_model_path=args.baseline,
                output_dir=args.output_dir,
                protocol_path=args.protocol,
            )
            print()
            print(format_comparison(results))
            print(f"\nWritten to {args.output_dir / level}\n")
    except (AblationError, DatasetError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
