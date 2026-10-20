import json
import re
from pathlib import Path
from typing import Any

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from fakenews_ml.artifacts import (
    ArtifactError,
    ArtifactNotFoundError,
    build_artifact,
    load_artifact,
    write_release,
)
from fakenews_ml.data import Dataset
from fakenews_ml.preprocessing import PREPROCESSING_VERSION
from fakenews_ml.report import format_report
from fakenews_ml.report import main as report_main
from fakenews_ml.train import TrainingConfig, train

FAST = TrainingConfig(cv_folds=3, min_df=1, seed=3)


@pytest.fixture
def active(dataset: Dataset, tmp_path: Path) -> Path:
    """An output dir holding a freshly trained, activated artifact."""
    write_release(train(dataset, FAST), tmp_path)
    return tmp_path


def rewrite_metadata(directory: Path, change: Any) -> None:
    path = directory / "model.metadata.json"
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


def test_release_is_versioned_immutable_and_matches_the_active_copy(active: Path) -> None:
    active_meta = json.loads((active / "model.metadata.json").read_text())
    release = active / "releases" / active_meta["model_version"]

    assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{8}", active_meta["model_version"])
    assert (release / "model.joblib").read_bytes() == (active / "model.joblib").read_bytes()
    assert (release / "model.metadata.json").read_text() == (
        active / "model.metadata.json"
    ).read_text()


def test_load_returns_pipeline_and_metadata_together(active: Path) -> None:
    loaded = load_artifact(active / "model.joblib")

    assert loaded.name in {"tfidf-logistic-regression", "tfidf-linear-svm-calibrated"}
    assert loaded.version == loaded.metadata["model_version"]
    assert loaded.pipeline.steps[0][0] == "clean"  # preprocessing travels inside the artifact
    assert loaded.metadata["preprocessing"]["version"] == PREPROCESSING_VERSION == "2"


def test_missing_model_is_reported_as_not_found(tmp_path: Path) -> None:
    with pytest.raises(ArtifactNotFoundError):
        load_artifact(tmp_path / "model.joblib")


def test_model_without_metadata_is_rejected(active: Path) -> None:
    (active / "model.metadata.json").unlink()
    with pytest.raises(ArtifactError, match="metadata is missing"):
        load_artifact(active / "model.joblib")


def test_model_file_that_does_not_match_its_recorded_hash_is_rejected(active: Path) -> None:
    other = Pipeline([("t", TfidfVectorizer()), ("c", LogisticRegression())]).fit(
        ["fake words here", "real words there"] * 3, ["fake", "real"] * 3
    )
    joblib.dump(other, active / "model.joblib")
    with pytest.raises(ArtifactError, match="SHA-256"):
        load_artifact(active / "model.joblib")


@pytest.mark.parametrize("stale", ["1", "0-old", "3"])
def test_preprocessing_version_mismatch_is_rejected(active: Path, stale: str) -> None:
    """An artifact trained with preprocessing v1 (or any other version) is refused, not served."""
    rewrite_metadata(active, lambda m: m["preprocessing"].update(version=stale))
    with pytest.raises(ArtifactError, match="preprocessing version"):
        load_artifact(active / "model.joblib")


def test_unsupported_schema_is_rejected(active: Path) -> None:
    rewrite_metadata(active, lambda m: m.update(schema_version=1))
    with pytest.raises(ArtifactError, match="Unsupported metadata schema"):
        load_artifact(active / "model.joblib")


@pytest.mark.parametrize("bad", [0.3, 1.5, "0.7", None, 1])
def test_invalid_threshold_is_rejected(active: Path, bad: object) -> None:
    rewrite_metadata(active, lambda m: m["uncertainty"].update(threshold=bad))
    with pytest.raises(ArtifactError, match="uncertainty threshold"):
        load_artifact(active / "model.joblib")


def test_wrong_labels_in_metadata_are_rejected(active: Path) -> None:
    rewrite_metadata(active, lambda m: m.update(labels=["spam", "ham"]))
    with pytest.raises(ArtifactError, match="labels"):
        load_artifact(active / "model.joblib")


def test_corrupt_metadata_json_is_rejected(active: Path) -> None:
    (active / "model.metadata.json").write_text("{not json")
    with pytest.raises(ArtifactError, match="not valid JSON"):
        load_artifact(active / "model.joblib")


def test_model_with_unexpected_classes_is_rejected(tmp_path: Path) -> None:
    pipeline = Pipeline([("t", TfidfVectorizer()), ("c", LogisticRegression())]).fit(
        ["aaa bbb", "ccc ddd"] * 3, ["spam", "ham"] * 3
    )
    body = {
        "model_name": "x",
        "uncertainty": {"threshold": 0.7},
    }
    write_release(build_artifact(pipeline, body), tmp_path)
    with pytest.raises(ArtifactError, match="classes"):
        load_artifact(tmp_path / "model.joblib")


def test_corrupt_model_bytes_are_rejected_even_with_a_matching_hash(tmp_path: Path) -> None:
    artifact = build_artifact(object(), {"model_name": "x", "uncertainty": {"threshold": 0.7}})
    write_release(artifact, tmp_path)
    with pytest.raises(ArtifactError, match="classifier"):
        load_artifact(tmp_path / "model.joblib")


def test_each_training_run_gets_its_own_release_directory(dataset: Dataset, tmp_path: Path) -> None:
    first = write_release(train(dataset, FAST), tmp_path)
    second = write_release(
        train(dataset, TrainingConfig(cv_folds=3, min_df=1, seed=99)), tmp_path, activate=False
    )

    assert first != second and first.is_dir() and second.is_dir()
    # --no-activate left the first release active.
    assert load_artifact(tmp_path / "model.joblib").version == first.name


def test_report_prints_selection_uncertainty_and_test_evaluation(
    active: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert report_main([str(active / "model.joblib")]) == 0
    text = capsys.readouterr().out
    metadata = json.loads((active / "model.metadata.json").read_text())

    assert text == format_report(metadata) + "\n"
    for expected in (
        "Cross-validation on the TRAINING split only",
        "<- selected",
        "Uncertainty rule",
        "FINAL TEST SET",
    ):
        assert expected in text


def test_report_without_a_model_explains_what_to_do(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert report_main([str(tmp_path / "model.joblib")]) == 1
    assert "Train a model first" in capsys.readouterr().err
