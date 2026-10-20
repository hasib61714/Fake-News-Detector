import json
from pathlib import Path

import joblib

from fakenews_ml.data import Dataset
from fakenews_ml.train import main, train_and_save
from tests.conftest import FAKE_WORDS, REAL_WORDS


def make_dataset(rows: list[tuple[str, str]]) -> Dataset:
    return Dataset([t for t, _ in rows], [label for _, label in rows], 0, 0)


def test_train_and_save_writes_working_artifacts(
    tmp_path: Path, rows: list[tuple[str, str]]
) -> None:
    metadata = train_and_save(make_dataset(rows), tmp_path, min_df=1)

    assert (tmp_path / "model.joblib").is_file()
    on_disk = json.loads((tmp_path / "model.metadata.json").read_text())
    assert on_disk["model_version"] == metadata["model_version"]
    assert on_disk["labels"] == ["fake", "real"]
    assert 0.0 <= on_disk["metrics"]["accuracy"] <= 1.0

    pipeline = joblib.load(tmp_path / "model.joblib")
    assert set(pipeline.classes_) == {"fake", "real"}
    assert pipeline.predict([FAKE_WORDS])[0] == "fake"
    assert pipeline.predict([REAL_WORDS])[0] == "real"


def test_cli_end_to_end(tmp_path: Path, rows: list[tuple[str, str]]) -> None:
    import csv

    data = tmp_path / "data.csv"
    with data.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "label"])
        writer.writerows(rows)
    out = tmp_path / "out"

    assert main(["--data", str(data), "--output-dir", str(out), "--min-df", "1"]) == 0
    assert (out / "model.joblib").is_file()


def test_cli_reports_dataset_errors(tmp_path: Path) -> None:
    assert main(["--data", str(tmp_path / "missing.csv"), "--output-dir", str(tmp_path)]) == 1
