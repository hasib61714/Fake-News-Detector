import pytest

from fakenews_ml.selection import (
    CandidateResult,
    MetricSummary,
    compare_candidates,
    select_best,
)
from tests.conftest import make_rows


def result(kind: str, f1: float) -> CandidateResult:
    summary = MetricSummary(mean=f1, std=0.0)
    return CandidateResult(
        kind=kind,  # type: ignore[arg-type]
        name=kind,
        metrics={"f1_macro": summary},
        cv_folds=5,
        total_seconds=0.0,
    )


def test_select_best_picks_the_highest_macro_f1() -> None:
    assert (
        select_best([result("logistic_regression", 0.90), result("linear_svm", 0.95)]).kind
        == "linear_svm"
    )
    assert (
        select_best([result("logistic_regression", 0.97), result("linear_svm", 0.95)]).kind
        == "logistic_regression"
    )


def test_exact_tie_goes_to_the_simpler_logistic_regression() -> None:
    assert (
        select_best([result("logistic_regression", 0.9), result("linear_svm", 0.9)]).kind
        == "logistic_regression"
    )


def test_select_best_needs_candidates() -> None:
    with pytest.raises(ValueError):
        select_best([])


def test_compare_candidates_runs_both_models_with_cv_metrics() -> None:
    rows = make_rows(40)
    results = compare_candidates(
        [t for t, _ in rows], [y for _, y in rows], cv_folds=3, seed=1, min_df=1
    )

    assert [r.kind for r in results] == ["logistic_regression", "linear_svm"]
    for r in results:
        assert r.cv_folds == 3
        assert set(r.metrics) == {"accuracy", "precision_macro", "recall_macro", "f1_macro"}
        assert all(0.0 <= m.mean <= 1.0 for m in r.metrics.values())


def test_compare_candidates_is_reproducible() -> None:
    rows = make_rows(40)
    x, y = [t for t, _ in rows], [lab for _, lab in rows]
    first = compare_candidates(x, y, cv_folds=3, seed=4, min_df=1)
    second = compare_candidates(x, y, cv_folds=3, seed=4, min_df=1)
    assert [r.metrics for r in first] == [r.metrics for r in second]


def test_cv_needs_at_least_two_folds() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        compare_candidates(["a"], ["fake"], cv_folds=1)
