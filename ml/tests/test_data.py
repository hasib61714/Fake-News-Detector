import csv
from pathlib import Path

import pytest

from fakenews_ml.data import DatasetError, load_class_csvs, load_labeled_csv


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def test_load_labeled_csv_cleans_and_dedupes(tmp_path: Path, rows: list[tuple[str, str]]) -> None:
    data = [[t, label] for t, label in rows] + [["", "fake"], [rows[0][0], "fake"]]
    path = write_csv(tmp_path / "d.csv", ["text", "label"], data)

    dataset = load_labeled_csv(path)

    assert len(dataset.texts) == len(rows)
    assert dataset.dropped_empty == 1
    assert dataset.dropped_duplicates == 1


def test_numeric_labels_are_rejected(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "d.csv", ["text", "label"], [["some text", "1"]])
    with pytest.raises(DatasetError, match="fake' or 'real"):
        load_labeled_csv(path)


def test_missing_column(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "d.csv", ["body", "label"], [["x", "fake"]])
    with pytest.raises(DatasetError, match="no 'text' column"):
        load_labeled_csv(path)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="not found"):
        load_labeled_csv(tmp_path / "nope.csv")


def test_too_few_examples(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "d.csv", ["text", "label"], [["a", "fake"], ["b", "real"]])
    with pytest.raises(DatasetError, match="at least"):
        load_labeled_csv(path)


def test_class_csvs(tmp_path: Path, rows: list[tuple[str, str]]) -> None:
    fake = write_csv(
        tmp_path / "Fake.csv", ["title", "text"], [["t", x] for x, y in rows if y == "fake"]
    )
    real = write_csv(
        tmp_path / "True.csv", ["title", "text"], [["t", x] for x, y in rows if y == "real"]
    )

    dataset = load_class_csvs(fake, real)

    assert dataset.labels.count("fake") == 30
    assert dataset.labels.count("real") == 30
