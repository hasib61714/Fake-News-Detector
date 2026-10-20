import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from fakenews_ml.ablation import (
    AblationError,
    class_metrics,
    format_comparison,
    mcnemar_exact,
    paired_bootstrap,
    run_level,
    top_weights,
)
from fakenews_ml.artifacts import ArtifactError, load_artifact, write_release
from fakenews_ml.data import Dataset
from fakenews_ml.models import build_pipeline
from fakenews_ml.train import TrainingConfig, train

FAST = TrainingConfig(cv_folds=3, min_df=1, seed=11)

# -- statistics (hand-checkable) -----------------------------------------------------------------


def test_mcnemar_exact_matches_the_closed_form() -> None:
    assert mcnemar_exact(10, 0) == pytest.approx(2 * 0.5**10)
    assert mcnemar_exact(5, 5) == pytest.approx(1.0)
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(3, 9) == mcnemar_exact(9, 3)


def test_class_metrics_match_a_hand_calculation() -> None:
    y_true = ["fake"] * 4 + ["real"] * 6
    y_pred = ["fake", "fake", "fake", "real", "real", "real", "real", "real", "real", "fake"]
    m = class_metrics(y_true, y_pred)

    assert m["confusion_matrix"]["matrix"] == [[3, 1], [1, 5]]
    assert m["accuracy"] == pytest.approx(0.8)
    assert m["per_class"]["fake"]["precision"] == pytest.approx(3 / 4)
    assert m["per_class"]["real"]["recall"] == pytest.approx(5 / 6)
    assert m["macro_f1"] == pytest.approx((0.75 + 5 / 6) / 2)


def test_paired_bootstrap_is_deterministic_and_zero_for_identical_predictions() -> None:
    y = ["fake", "real"] * 50
    same = paired_bootstrap(y, y, y, resamples=200, seed=1)
    assert same["accuracy_diff"] == {"observed": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}

    worse = ["real" if i % 10 == 0 else v for i, v in enumerate(y)]
    first = paired_bootstrap(y, y, worse, resamples=300, seed=2)
    assert first == paired_bootstrap(y, y, worse, resamples=300, seed=2)
    assert first["accuracy_diff"]["observed"] < 0
    assert first["accuracy_diff"]["ci95_high"] < 0.0  # a clear 5-point loss is detected


# -- the experiment end to end on tiny in-code fixtures ------------------------------------------


@pytest.fixture
def setup(dataset: Dataset, tmp_path: Path) -> dict[str, Any]:
    base_dir = tmp_path / "baseline"
    write_release(train(dataset, FAST), base_dir)
    protocol = tmp_path / "PROTOCOL.md"
    protocol.write_text("pre-registered protocol (test fixture)\n")
    return {
        "dataset": dataset,
        "baseline_path": base_dir / "model.joblib",
        "protocol": protocol,
        "out": tmp_path / "experiments",
    }


def _run(setup: dict[str, Any], level: str = "core", **overrides: Any) -> dict[str, Any]:
    args = {
        "dataset": setup["dataset"],
        "baseline": load_artifact(setup["baseline_path"]),
        "baseline_model_path": setup["baseline_path"],
        "output_dir": setup["out"],
        "protocol_path": setup["protocol"],
        "log": lambda *_: None,
        **overrides,
    }
    return run_level(level, **args)  # type: ignore[arg-type]


def test_run_level_compares_on_the_same_split_and_leaves_the_baseline_untouched(
    setup: dict[str, Any],
) -> None:
    baseline_files = [
        setup["baseline_path"],
        setup["baseline_path"].with_name("model.metadata.json"),
    ]
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in baseline_files]

    results = _run(setup)

    assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in baseline_files] == before
    assert results["baseline_unchanged"] is True
    assert results["seed"] == 11
    assert results["splits"]["test"] == 32 == results["paired_comparison"]["n_test"]
    assert results["protocol_sha256"] == hashlib.sha256(setup["protocol"].read_bytes()).hexdigest()
    assert (
        results["baseline"]["test"]["confusion_matrix"]
        == load_artifact(setup["baseline_path"]).metadata["evaluation"]["test"]["all_predictions"][
            "confusion_matrix"
        ]
    )
    out = setup["out"] / "core"
    assert {p.name for p in out.iterdir()} == {
        "model.joblib",
        "metadata.json",
        "results.json",
        "comparison.txt",
    }
    assert json.loads((out / "results.json").read_text())["level"] == "core"
    assert "STYLE-MASK ABLATION (core)" in (out / "comparison.txt").read_text()


def test_the_ablation_artifact_cannot_be_served_by_the_api_loader(setup: dict[str, Any]) -> None:
    _run(setup)
    experiment = setup["out"] / "core"
    metadata = json.loads((experiment / "metadata.json").read_text())
    assert "style-mask-core" in metadata["preprocessing"]["version"]
    assert metadata["experiment"]["name"] == "style-mask-v1"

    # load_artifact wants "<model stem>.metadata.json"; the experiment stores "metadata.json".
    (experiment / "model.metadata.json").write_text((experiment / "metadata.json").read_text())
    with pytest.raises(ArtifactError, match="preprocessing version"):
        load_artifact(experiment / "model.joblib")


def test_a_finished_level_is_never_rerun(setup: dict[str, Any]) -> None:
    _run(setup)
    with pytest.raises(AblationError, match="already exists"):
        _run(setup)


def test_the_protocol_must_exist_before_running(setup: dict[str, Any]) -> None:
    setup["protocol"].unlink()
    with pytest.raises(AblationError, match="Protocol not found"):
        _run(setup)
    assert not (setup["out"] / "core").exists()  # nothing was trained or written


def test_different_data_is_refused(setup: dict[str, Any]) -> None:
    ds: Dataset = setup["dataset"]
    changed = Dataset(
        [t + " extra words" for t in ds.texts],
        ds.labels,
        type(ds.report)(
            sources=ds.report.sources,
            classes=ds.report.classes,
            total_kept=ds.report.total_kept,
            fingerprint="0" * 64,
            warnings=[],
        ),
    )
    with pytest.raises(AblationError, match="fingerprint"):
        _run(setup, dataset=changed)


def test_a_split_that_does_not_reproduce_the_baseline_is_refused(setup: dict[str, Any]) -> None:
    path = setup["baseline_path"].with_name("model.metadata.json")
    metadata = json.loads(path.read_text())
    metadata["evaluation"]["test"]["all_predictions"]["confusion_matrix"]["matrix"] = [
        [0, 0],
        [0, 0],
    ]
    path.write_text(json.dumps(metadata))
    with pytest.raises(AblationError, match="does not reproduce"):
        _run(setup)


def test_format_comparison_reports_every_requested_element(setup: dict[str, Any]) -> None:
    text = format_comparison(_run(setup))
    for expected in (
        "accuracy",
        "macro F1",
        "fake precision",
        "real recall",
        "Confusion matrix, BASELINE",
        "Confusion matrix, ABLATION",
        "McNemar exact",
        "bootstrap CI",
        "Largest-weight features of the ABLATION model",
        "Baseline files unchanged during the run: True",
    ):
        assert expected in text


@pytest.mark.parametrize("kind", ["logistic_regression", "linear_svm"])
def test_top_weights_lists_both_directions(kind: str) -> None:
    texts = [f"shocking miracle hoax secret {i}" for i in range(30)] + [
        f"ministry committee budget report {i}" for i in range(30)
    ]
    labels = ["fake"] * 30 + ["real"] * 30
    pipeline = build_pipeline(kind, min_df=1)  # type: ignore[arg-type]
    pipeline.fit(texts, labels)

    weights = top_weights(pipeline, n=5)
    assert all(w > 0 for _, w in weights["toward_real"])
    assert all(w < 0 for _, w in weights["toward_fake"])
    assert {"ministry", "committee"} & {name for name, _ in weights["toward_real"]}
    assert {"shocking", "miracle"} & {name for name, _ in weights["toward_fake"]}
