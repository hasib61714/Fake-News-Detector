"""Dataset loading. No dataset ships with this repository; see ml/README.md."""

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

LABELS = ("fake", "real")
MIN_PER_CLASS = 10

# Real-world article bodies exceed csv's default 128 KiB field limit.
csv.field_size_limit(2**31 - 1)


class DatasetError(ValueError):
    """The dataset is missing, malformed or too small to train on."""


@dataclass(frozen=True)
class Dataset:
    texts: list[str]
    labels: list[str]
    dropped_empty: int
    dropped_duplicates: int


def _read_column(path: Path, column: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise DatasetError(f"Dataset file not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise DatasetError(f"{path} has no '{column}' column (found: {reader.fieldnames}).")
        return list(reader)


def _normalize_label(raw: str, path: Path) -> str:
    label = raw.strip().lower()
    if label not in LABELS:
        raise DatasetError(
            f"{path}: label must be 'fake' or 'real', got {raw!r}. "
            "Numeric labels are rejected on purpose: 0/1 conventions differ between datasets."
        )
    return label


def load_labeled_csv(path: Path, text_column: str = "text", label_column: str = "label") -> Dataset:
    """Load one CSV with a text column and a label column containing 'fake' or 'real'."""
    rows = _read_column(path, text_column)
    if rows and label_column not in rows[0]:
        raise DatasetError(f"{path} has no '{label_column}' column.")
    pairs = [(row[text_column], _normalize_label(row[label_column], path)) for row in rows]
    return _build(pairs)


def load_class_csvs(fake_csv: Path, real_csv: Path, text_column: str = "text") -> Dataset:
    """Load one CSV per class (e.g. Fake.csv / True.csv); the file determines the label."""
    pairs = [(row[text_column], "fake") for row in _read_column(fake_csv, text_column)]
    pairs += [(row[text_column], "real") for row in _read_column(real_csv, text_column)]
    return _build(pairs)


def _build(pairs: Iterable[tuple[str, str]]) -> Dataset:
    seen: set[str] = set()
    texts: list[str] = []
    labels: list[str] = []
    empty = duplicates = 0
    for raw_text, label in pairs:
        text = raw_text.strip()
        if not text:
            empty += 1
        elif text in seen:
            duplicates += 1
        else:
            seen.add(text)
            texts.append(text)
            labels.append(label)

    for label in LABELS:
        count = labels.count(label)
        if count < MIN_PER_CLASS:
            raise DatasetError(
                f"Need at least {MIN_PER_CLASS} '{label}' examples after cleaning; got {count}."
            )
    return Dataset(texts, labels, dropped_empty=empty, dropped_duplicates=duplicates)
