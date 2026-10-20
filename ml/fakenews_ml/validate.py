"""Validate dataset files without training: ``python -m fakenews_ml.validate``."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from fakenews_ml.data import Dataset, DatasetError, load_class_csvs, load_data_dir, load_labeled_csv

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_argument_group("dataset (default: Fake.csv and True.csv in --data-dir)")
    source.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    source.add_argument("--fake-csv", type=Path, help="explicit path to Fake.csv")
    source.add_argument("--real-csv", type=Path, help="explicit path to True.csv")
    source.add_argument("--data", type=Path, help="one CSV with 'text' and 'label' (fake|real)")
    parser.add_argument("--text-column", default="text")


def load_from_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> Dataset:
    if args.data and (args.fake_csv or args.real_csv):
        parser.error("use either --data, or --fake-csv with --real-csv, not both")
    if bool(args.fake_csv) != bool(args.real_csv):
        parser.error("--fake-csv and --real-csv must be used together")
    if args.data:
        return load_labeled_csv(args.data, text_column=args.text_column)
    if args.fake_csv:
        return load_class_csvs(args.fake_csv, args.real_csv, text_column=args.text_column)
    return load_data_dir(args.data_dir, text_column=args.text_column)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_arguments(parser)
    args = parser.parse_args(argv)
    try:
        dataset = load_from_args(parser, args)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(dataset.report.format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
