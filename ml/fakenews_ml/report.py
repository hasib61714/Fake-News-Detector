"""Human-readable summary of a trained artifact: ``python -m fakenews_ml.report [model.joblib]``."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fakenews_ml.artifacts import METADATA_FILENAME, metadata_path_for

DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _matrix(block: dict[str, Any]) -> str:
    labels = block["confusion_matrix"]["labels"]
    rows = block["confusion_matrix"]["matrix"]
    width = max(len(label) for label in labels) + 2
    head = " " * (width + 10) + "".join(f"pred {label}".rjust(width + 5) for label in labels)
    body = [
        f"  true {label}".ljust(width + 10) + "".join(str(v).rjust(width + 5) for v in row)
        for label, row in zip(labels, rows, strict=True)
    ]
    return "\n".join([head, *body])


def format_report(metadata: dict[str, Any]) -> str:
    out: list[str] = []
    out.append(f"Model {metadata['model_name']}  version {metadata['model_version']}")
    out.append(f"  trained_at {metadata['trained_at']}  seed {metadata['seed']}")
    libs = metadata["libraries"]
    out.append(f"  python {libs['python']}  scikit-learn {libs['scikit_learn']}")
    out.append(f"  artifact sha256 {metadata['artifact']['sha256']}")
    out.append(f"  preprocessing v{metadata['preprocessing']['version']}")

    dataset = metadata["dataset"]
    out.append("")
    out.append(f"Dataset fingerprint {dataset['report']['fingerprint']}")
    out.append(f"  splits: {dataset['splits']}")
    for warning in dataset["report"]["warnings"]:
        out.append(f"  WARNING: {warning}")

    selection = metadata["model_selection"]
    out.append("")
    out.append(
        f"Cross-validation on the TRAINING split only ({selection['cv_folds']}-fold, "
        f"selection metric {selection['metric']}):"
    )
    for c in selection["candidates"]:
        m = c["metrics"]
        marker = "  <- selected" if c["kind"] == selection["selected"] else ""
        out.append(
            f"  {c['name']:<30} f1_macro {m['f1_macro']['mean']:.4f} ± {m['f1_macro']['std']:.4f}"
            f"  precision {m['precision_macro']['mean']:.4f}"
            f"  recall {m['recall_macro']['mean']:.4f}"
            f"  accuracy {m['accuracy']['mean']:.4f}{marker}"
        )

    rule = metadata["uncertainty"]
    out.append("")
    out.append(
        f"Uncertainty rule (derived on VALIDATION, n={rule['validation_samples']}): predictions "
        f"with confidence < {rule['threshold']:.2f} are 'uncertain'"
    )
    out.append(
        f"  validation coverage {rule['validation_coverage']:.1%}, decisive accuracy "
        f"{_pct(rule['validation_decisive_accuracy'])}, target {rule['target_accuracy']:.0%} "
        f"met: {rule['target_met']}"
    )

    test = metadata["evaluation"]["test"]
    allp = test["all_predictions"]
    out.append("")
    out.append(f"FINAL TEST SET (n={allp['n']}), evaluated once after selection:")
    out.append(allp["classification_report_text"].rstrip())
    out.append("Confusion matrix (rows = true, columns = predicted):")
    out.append(_matrix(allp))
    unc = test["uncertainty"]
    out.append(
        f"With the uncertainty rule: {unc['n_decisive']} decisive, {unc['n_uncertain']} uncertain "
        f"(coverage {unc['coverage']:.1%})"
    )
    if test["decisive_only"]:
        d = test["decisive_only"]
        out.append(
            f"  decisive-only accuracy {d['accuracy']:.4f}, macro F1 {d['macro_avg']['f1']:.4f}"
        )
        out.append(_matrix(d))
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        nargs="?",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "model.joblib",
        help="model.joblib (its metadata is read from the sibling .metadata.json)",
    )
    args = parser.parse_args(argv)
    path = metadata_path_for(args.model)
    if not path.is_file():
        print(
            f"error: no {METADATA_FILENAME} found at {path}. Train a model first.", file=sys.stderr
        )
        return 1
    print(format_report(json.loads(path.read_text(encoding="utf-8"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
