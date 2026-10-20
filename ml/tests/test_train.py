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


# Fields that legitimately differ between two otherwise identical training runs.
VOLATILE_KEYS = {"total_seconds", "trained_at", "model_version"}


def without_volatile(value: Any) -> Any:
    """Deep copy of a metadata structure without wall-clock timings and timestamps."""
    if isinstance(value, dict):
        return {k: without_volatile(v) for k, v in value.items() if k not in VOLATILE_KEYS}
    if isinstance(value, list):
        return [without_volatile(v) for v in value]
    return value


def record_stages(
    monkeypatch: pytest.MonkeyPatch, dataset: Dataset, config: TrainingConfig
) -> list[tuple[str, Any]]:
    """Run ``train`` while recording what each stage receives, in call order.

    Events: ``("compare", train_texts)``, ``("derive", n_validation)``,
    ``("evaluate:validation", texts)`` and ``("evaluate:test", texts)``. Patches are undone
    when the helper returns, so it can be called repeatedly in one test.
    """
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

    with monkeypatch.context() as patch:
        patch.setattr(train_module, "compare_candidates", compare)
        patch.setattr(train_module, "derive_rule", derive)
        patch.setattr(train_module, "_evaluate_split", evaluate_split)
        train(dataset, config)
    return events


def test_same_seed_reproduces_the_whole_run(dataset: Dataset) -> None:
    """Everything except wall-clock timings and timestamps must be identical, including the
    SHA-256 of the serialized model (which depends on every fitted coefficient)."""
    a = train(dataset, FAST)
    b = train(dataset, FAST)

    assert a.metadata["artifact"]["sha256"] == b.metadata["artifact"]["sha256"]
    assert a.model_bytes == b.model_bytes
    assert without_volatile(a.metadata) == without_volatile(b.metadata)


def test_timings_are_the_only_run_to_run_difference_in_model_selection(dataset: Dataset) -> None:
    """Regression guard: comparisons of candidate records must never depend on elapsed time."""
    candidates = train(dataset, FAST).metadata["model_selection"]["candidates"]
    assert all(c["total_seconds"] >= 0 for c in candidates)
    assert all("total_seconds" not in without_volatile(c) for c in candidates)


def test_a_different_seed_changes_the_split(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compare the actual samples each stage receives under seed 11 and seed 12.

    Deterministic: depends only on the seeded permutation, not on metric values, model
    quality, elapsed time or how separable the fixture data is.
    """
    seeds = (11, 12)
    runs = {
        seed: dict(
            record_stages(monkeypatch, dataset, TrainingConfig(cv_folds=3, min_df=1, seed=seed))
        )
        for seed in seeds
    }
    stages = {"train": "compare", "validation": "evaluate:validation", "test": "evaluate:test"}

    for split, stage in stages.items():
        assert runs[11][stage] != runs[12][stage], f"{split} split is identical for seeds 11 and 12"

    everything = set(dataset.texts)
    for seed in seeds:
        parts = [runs[seed][stage] for stage in stages.values()]
        assert (len(parts[0]), len(parts[1]), len(parts[2])) == (96, 32, 32)
        assert set().union(*parts) == everything  # a partition: nothing lost...
        assert sum(len(part) for part in parts) == len(everything)  # ...and nothing duplicated


def test_final_test_set_is_isolated_from_selection_and_threshold_derivation(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spy on each stage: selection sees only train rows, the rule only validation, and the
    test split is scored exactly once, last."""
    events = record_stages(monkeypatch, dataset, FAST)

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
    # Only the test labels changed. Flipping every test label flips every correctness verdict, so
    # accuracy must be exactly complementary whatever the model's quality: this cannot pass by
    # luck, and it fails if the test score were computed from the original labels.
    base_test = baseline.metadata["evaluation"]["test"]["all_predictions"]
    flipped_test = tampered.metadata["evaluation"]["test"]["all_predictions"]
    assert flipped_test["n"] == base_test["n"]
    assert flipped_test["accuracy"] == pytest.approx(1.0 - base_test["accuracy"])


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
