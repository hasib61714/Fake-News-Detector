import json
from pathlib import Path
from typing import Any

import pytest

import fakenews_ml.selection as selection
import fakenews_ml.train as train_module
import fakenews_ml.uncertainty as uncertainty
from fakenews_ml.artifacts import load_artifact, write_release
from fakenews_ml.data import Dataset
from fakenews_ml.splits import split_indices
from fakenews_ml.train import TrainingConfig, main, train
from tests.conftest import FAKE_WORDS, REAL_WORDS

FAST = TrainingConfig(cv_folds=3, min_df=1, seed=11)


def test_train_produces_a_complete_versioned_artifact(dataset: Dataset) -> None:
    artifact = train(dataset, FAST)
    m = artifact.metadata

    assert m["schema_version"] == 2
    assert m["model_version"].endswith(m["artifact"]["sha256"][:8])
    assert m["labels"] == ["fake", "real"]
    assert m["preprocessing"]["version"]
    assert m["seed"] == 11 and m["config"]["cv_folds"] == 3
    assert m["dataset"]["report"]["fingerprint"] == dataset.report.fingerprint
    assert m["dataset"]["splits"] == {"train": 96, "validation": 32, "test": 32, "stratified": True}

    selection = m["model_selection"]
    assert [c["kind"] for c in selection["candidates"]] == ["logistic_regression", "linear_svm"]
    assert selection["selected"] in {"logistic_regression", "linear_svm"}
    assert selection["data"] == "training split only"

    test = m["evaluation"]["test"]["all_predictions"]
    assert test["n"] == 32
    assert test["confusion_matrix"]["labels"] == ["fake", "real"]
    assert "classification_report" in test and "per_class" in test
    assert 0.5 <= m["uncertainty"]["threshold"] < 1.0
    json.dumps(m, allow_nan=False)


def test_written_artifact_loads_and_predicts_with_the_learned_signal(
    dataset: Dataset, tmp_path: Path
) -> None:
    artifact = train(dataset, FAST)
    release = write_release(artifact, tmp_path)

    loaded = load_artifact(release / "model.joblib")
    fake_p, real_p = loaded.pipeline.predict_proba(
        [" ".join(FAKE_WORDS * 2), " ".join(REAL_WORDS * 2)]
    )
    classes = list(loaded.pipeline.classes_)
    assert classes[fake_p.argmax()] == "fake"
    assert classes[real_p.argmax()] == "real"
    assert loaded.threshold == artifact.metadata["uncertainty"]["threshold"]


def test_same_seed_reproduces_selection_threshold_and_test_metrics(dataset: Dataset) -> None:
    a = train(dataset, FAST).metadata
    b = train(dataset, FAST).metadata

    assert a["model_selection"]["selected"] == b["model_selection"]["selected"]
    assert a["uncertainty"] == b["uncertainty"]
    assert a["evaluation"] == b["evaluation"]
    assert [c["metrics"] for c in a["model_selection"]["candidates"]] == [
        c["metrics"] for c in b["model_selection"]["candidates"]
    ]


def test_a_different_seed_changes_the_split(dataset: Dataset) -> None:
    a = train(dataset, FAST).metadata
    b = train(dataset, TrainingConfig(cv_folds=3, min_df=1, seed=12)).metadata
    assert a["seed"] != b["seed"]
    assert a["evaluation"]["test"] != b["evaluation"]["test"] or (
        a["model_selection"]["candidates"] != b["model_selection"]["candidates"]
    )


def test_final_test_set_is_isolated_from_selection_and_threshold_derivation(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spy on each stage: selection sees only train rows, the rule only validation, and the
    test split is scored exactly once, last."""
    events: list[tuple[str, Any]] = []
    real_compare = selection.compare_candidates
    real_derive = uncertainty.derive_rule
    real_eval = train_module._evaluate_split

    def compare(texts: Any, labels: Any, **kw: Any) -> Any:
        events.append(("compare", set(texts)))
        return real_compare(texts, labels, **kw)

    def derive(conf: Any, correct: Any, **kw: Any) -> Any:
        events.append(("derive", len(conf)))
        return real_derive(conf, correct, **kw)

    def evaluate_split(name: str, pipeline: Any, texts: Any, labels: Any, threshold: float) -> Any:
        events.append(("evaluate:" + name, set(texts)))
        return real_eval(name, pipeline, texts, labels, threshold)

    monkeypatch.setattr(train_module, "compare_candidates", compare)
    monkeypatch.setattr(train_module, "derive_rule", derive)
    monkeypatch.setattr(train_module, "_evaluate_split", evaluate_split)

    train(dataset, FAST)

    assert [name for name, _ in events] == [
        "compare",
        "derive",
        "evaluate:validation",
        "evaluate:test",
    ]
    by_name = dict(events)
    train_texts, test_texts = by_name["compare"], by_name["evaluate:test"]
    val_texts = by_name["evaluate:validation"]
    assert len(train_texts) == 96 and len(val_texts) == 32 and len(test_texts) == 32
    assert by_name["derive"] == 32
    assert train_texts.isdisjoint(test_texts) and train_texts.isdisjoint(val_texts)
    assert val_texts.isdisjoint(test_texts)


def test_test_labels_never_influence_the_trained_model(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hold the split fixed, flip ONLY the test rows' labels, and retrain: the fitted model,
    the model selection and the uncertainty rule must be unchanged; only the test score moves."""
    fixed = split_indices(dataset.labels, FAST.test_size, FAST.validation_size, FAST.seed)
    monkeypatch.setattr(train_module, "split_indices", lambda *args, **kwargs: fixed)

    flipped = list(dataset.labels)
    for i in fixed.test:
        flipped[i] = "real" if flipped[i] == "fake" else "fake"

    baseline = train(dataset, FAST)
    tampered = train(Dataset(dataset.texts, flipped, dataset.report), FAST)

    def candidate_scores(a: Any) -> list[Any]:
        return [c["metrics"] for c in a.metadata["model_selection"]["candidates"]]

    assert candidate_scores(baseline) == candidate_scores(tampered)
    assert (
        baseline.metadata["model_selection"]["selected"]
        == tampered.metadata["model_selection"]["selected"]
    )
    assert baseline.metadata["uncertainty"] == tampered.metadata["uncertainty"]
    assert (
        baseline.metadata["evaluation"]["validation"]
        == tampered.metadata["evaluation"]["validation"]
    )
    assert baseline.metadata["artifact"]["sha256"] == tampered.metadata["artifact"]["sha256"]
    assert baseline.metadata["evaluation"]["test"] != tampered.metadata["evaluation"]["test"]


def test_invalid_config_is_rejected(dataset: Dataset) -> None:
    with pytest.raises(ValueError, match="sum to less than 1"):
        train(dataset, TrainingConfig(test_size=0.6, validation_size=0.6, min_df=1))


# -- CLI ---------------------------------------------------------------------


def test_cli_trains_from_a_data_directory_and_activates(
    data_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "artifacts"
    code = main(
        ["--data-dir", str(data_dir), "--output-dir", str(out), "--cv-folds", "3", "--min-df", "1"]
    )
    text = capsys.readouterr().out

    assert code == 0
    assert (out / "model.joblib").is_file() and (out / "model.metadata.json").is_file()
    releases = list((out / "releases").iterdir())
    assert len(releases) == 1
    assert "Dataset validation report" in text and "FINAL TEST SET" in text
    assert "Confusion matrix" in text and "<- selected" in text
    assert load_artifact(out / "model.joblib").version == releases[0].name


def test_cli_no_activate_writes_only_the_release(data_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "artifacts"
    assert (
        main(
            [
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(out),
                "--no-activate",
                "--cv-folds",
                "3",
                "--min-df",
                "1",
            ]
        )
        == 0
    )
    assert not (out / "model.joblib").exists()
    assert len(list((out / "releases").iterdir())) == 1


def test_cli_reports_missing_dataset_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--data-dir", str(tmp_path), "--output-dir", str(tmp_path / "o")]) == 1
    assert "Expected Fake.csv and True.csv" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()


def test_cli_reports_invalid_numeric_options(
    data_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(tmp_path / "o"),
            "--test-size",
            "0.7",
            "--validation-size",
            "0.5",
        ]
    )
    assert code == 1
    assert "sum to less than 1" in capsys.readouterr().err


def test_cli_rejects_conflicting_dataset_arguments(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--fake-csv", str(tmp_path / "Fake.csv")])
    assert exc.value.code == 2
