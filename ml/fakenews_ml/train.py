"""Train the classifier and write the artifact the API loads.

Outputs (in --output-dir, default ml/artifacts/):
    model.joblib            the fitted sklearn Pipeline (labels: "fake", "real")
    model.metadata.json     name, version, trained_at, metrics, dataset summary
"""

import argparse
import json
import os
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import sklearn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from fakenews_ml.data import Dataset, DatasetError, load_class_csvs, load_labeled_csv
from fakenews_ml.pipeline import MODEL_NAME, build_pipeline

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def train_and_save(
    dataset: Dataset,
    output_dir: Path,
    test_size: float = 0.2,
    seed: int = 42,
    min_df: int = 2,
) -> dict[str, Any]:
    """Fit on a stratified split, evaluate on the held-out part, write artifacts."""
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.texts,
        dataset.labels,
        test_size=test_size,
        random_state=seed,
        stratify=dataset.labels,
    )
    pipeline = build_pipeline(min_df=min_df, seed=seed)
    pipeline.fit(x_train, y_train)

    report = classification_report(
        y_test, pipeline.predict(x_test), output_dict=True, zero_division=0
    )

    trained_at = datetime.now(timezone.utc)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "model_name": MODEL_NAME,
        "model_version": trained_at.strftime("%Y%m%d%H%M%S"),
        "trained_at": trained_at.isoformat(timespec="seconds"),
        "labels": sorted(set(dataset.labels)),
        "sklearn_version": sklearn.__version__,
        "seed": seed,
        "dataset": {
            "train_samples": len(x_train),
            "test_samples": len(x_test),
            "dropped_empty": dataset.dropped_empty,
            "dropped_duplicates": dataset.dropped_duplicates,
        },
        "metrics": {
            "accuracy": report["accuracy"],
            "per_class": {k: report[k] for k in ("fake", "real")},
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"
    metadata_path = output_dir / "model.metadata.json"
    # Write via temp files and publish the model last: the API watches for model.joblib.
    metadata_tmp = metadata_path.with_suffix(".json.tmp")
    metadata_tmp.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    os.replace(metadata_tmp, metadata_path)
    model_tmp = model_path.with_suffix(".joblib.tmp")
    joblib.dump(pipeline, model_tmp)
    os.replace(model_tmp, model_path)
    return metadata


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    source = parser.add_argument_group("dataset (choose one form)")
    source.add_argument("--data", type=Path, help="CSV with 'text' and 'label' (fake|real) columns")
    source.add_argument("--fake-csv", type=Path, help="CSV of fake articles (e.g. Fake.csv)")
    source.add_argument("--real-csv", type=Path, help="CSV of real articles (e.g. True.csv)")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--min-df", type=int, default=2, help="TF-IDF minimum document frequency")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if bool(args.data) == bool(args.fake_csv or args.real_csv):
        parser.error("provide either --data, or both --fake-csv and --real-csv")
    if not args.data and not (args.fake_csv and args.real_csv):
        parser.error("--fake-csv and --real-csv must be used together")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.data:
            dataset = load_labeled_csv(args.data, text_column=args.text_column)
        else:
            dataset = load_class_csvs(args.fake_csv, args.real_csv, text_column=args.text_column)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    metadata = train_and_save(
        dataset, args.output_dir, test_size=args.test_size, seed=args.seed, min_df=args.min_df
    )
    print(f"Trained {metadata['model_name']} v{metadata['model_version']}")
    print(f"  samples: {metadata['dataset']}")
    print(f"  held-out accuracy: {metadata['metrics']['accuracy']:.4f}")
    print(f"  artifacts written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
