"""Train, compare, calibrate the uncertainty rule, evaluate once, and write the artifact.

Stages (the order is the point; see ml/README.md):

1. Stratified split into train / validation / test (seeded).
2. Cross-validate Logistic Regression and Linear SVM on TRAIN only; pick the best macro-F1.
3. Fit the chosen pipeline on TRAIN.
4. Derive the 'uncertain' confidence threshold on VALIDATION.
5. Evaluate on TEST exactly once. Nothing is tuned afterwards.
6. Write a versioned artifact with all of the above recorded in its metadata.
"""

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fakenews_ml.artifacts import Artifact, build_artifact, write_release
from fakenews_ml.data import Dataset, DatasetError
from fakenews_ml.evaluation import evaluate, predict_top
from fakenews_ml.models import DISPLAY_NAMES, build_pipeline
from fakenews_ml.report import DEFAULT_ARTIFACT_DIR, format_report
from fakenews_ml.selection import SELECTION_METRIC, compare_candidates, select_best
from fakenews_ml.splits import split_indices, take
from fakenews_ml.stylemask import StyleMaskLevel
from fakenews_ml.uncertainty import UncertaintyRule, derive_rule
from fakenews_ml.validate import add_dataset_arguments, load_from_args


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    test_size: float = 0.2
    validation_size: float = 0.2
    cv_folds: int = 5
    min_df: int = 2
    max_features: int = 100_000
    target_accuracy: float = 0.95
    min_coverage: float = 0.5
    min_threshold: float = 0.6
    n_jobs: int = 1
    style_mask: StyleMaskLevel | None = None  # ablation experiment only; None = baseline


def _evaluate_split(
    name: str, pipeline: Any, texts: Sequence[str], labels: Sequence[str], threshold: float
) -> dict[str, Any]:
    """Score one split. Called once for 'validation' and once, last, for 'test'."""
    predicted, confidence = predict_top(pipeline, texts)
    return evaluate(labels, predicted, list(confidence), threshold)


def train(
    dataset: Dataset,
    config: TrainingConfig | None = None,
    *,
    log: Callable[[str], None] = lambda _: None,
) -> Artifact:
    """Run the full procedure and return the serialized artifact (not yet written to disk)."""
    config = config or TrainingConfig()
    splits = split_indices(dataset.labels, config.test_size, config.validation_size, config.seed)
    x_train, y_train = take(dataset.texts, splits.train), take(dataset.labels, splits.train)
    x_val, y_val = take(dataset.texts, splits.validation), take(dataset.labels, splits.validation)
    x_test, y_test = take(dataset.texts, splits.test), take(dataset.labels, splits.test)
    log(f"Split: train {len(x_train):,} / validation {len(x_val):,} / test {len(x_test):,}")

    log(f"Cross-validating candidates on the training split ({config.cv_folds}-fold)...")
    candidates = compare_candidates(
        x_train,
        y_train,
        cv_folds=config.cv_folds,
        seed=config.seed,
        min_df=config.min_df,
        max_features=config.max_features,
        n_jobs=config.n_jobs,
        style_mask=config.style_mask,
    )
    selected = select_best(candidates)
    log(f"Selected {selected.name} ({SELECTION_METRIC} {selected.score:.4f})")

    pipeline = build_pipeline(
        selected.kind,
        seed=config.seed,
        min_df=config.min_df,
        max_features=config.max_features,
        style_mask=config.style_mask,
    )
    pipeline.fit(x_train, y_train)

    predicted, confidence = predict_top(pipeline, x_val)
    rule: UncertaintyRule = derive_rule(
        list(confidence),
        [p == t for p, t in zip(predicted, y_val, strict=True)],
        target_accuracy=config.target_accuracy,
        min_coverage=config.min_coverage,
        min_threshold=config.min_threshold,
    )
    validation_eval = _evaluate_split("validation", pipeline, x_val, y_val, rule.threshold)

    log("Evaluating on the held-out test split (once)...")
    test_eval = _evaluate_split("test", pipeline, x_test, y_test, rule.threshold)

    return build_artifact(
        pipeline,
        {
            "model_name": DISPLAY_NAMES[selected.kind],
            "seed": config.seed,
            "config": asdict(config),
            "dataset": {
                "report": dataset.report.to_dict(),
                "splits": {
                    "train": len(x_train),
                    "validation": len(x_val),
                    "test": len(x_test),
                    "stratified": True,
                },
            },
            "model_selection": {
                "metric": SELECTION_METRIC,
                "data": "training split only",
                "cv_folds": config.cv_folds,
                "candidates": [c.to_dict() for c in candidates],
                "selected": selected.kind,
                "final_fit": "training split only (validation and test excluded)",
            },
            "uncertainty": rule.to_dict(),
            "evaluation": {"validation": validation_eval, "test": test_eval},
        },
    )


def _parse_args(argv: Sequence[str] | None) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    add_dataset_arguments(parser)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--no-activate", action="store_true", help="write the release only")
    defaults = TrainingConfig()
    for field, kind in (
        ("seed", int),
        ("test_size", float),
        ("validation_size", float),
        ("cv_folds", int),
        ("min_df", int),
        ("max_features", int),
        ("target_accuracy", float),
        ("min_coverage", float),
        ("min_threshold", float),
        ("n_jobs", int),
    ):
        parser.add_argument(
            f"--{field.replace('_', '-')}", type=kind, default=getattr(defaults, field)
        )
    return parser, parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    parser, args = _parse_args(argv)
    try:
        dataset = load_from_args(parser, args)
        print(dataset.report.format())
        # `style_mask` is an ablation-only option and is deliberately not exposed on this CLI.
        config = TrainingConfig(
            **{
                k: getattr(args, k)
                for k in TrainingConfig.__dataclass_fields__
                if k != "style_mask"
            }
        )
        artifact = train(dataset, config, log=print)
    except (DatasetError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    release = write_release(artifact, args.output_dir, activate=not args.no_activate)
    print()
    print(format_report(artifact.metadata))
    print()
    print(f"Release written to {release}")
    if not args.no_activate:
        print(f"Active model updated in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
