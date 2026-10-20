"""Model selection by stratified k-fold cross-validation on the training split only."""

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from sklearn.model_selection import StratifiedKFold, cross_validate

from fakenews_ml.models import CANDIDATES, DISPLAY_NAMES, CandidateKind, build_pipeline

SELECTION_METRIC = "f1_macro"
_SCORING = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]


@dataclass(frozen=True)
class MetricSummary:
    mean: float
    std: float


@dataclass(frozen=True)
class CandidateResult:
    kind: CandidateKind
    name: str
    metrics: dict[str, MetricSummary]
    cv_folds: int
    total_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def score(self) -> float:
        return self.metrics[SELECTION_METRIC].mean


def compare_candidates(
    texts: Sequence[str],
    labels: Sequence[str],
    *,
    cv_folds: int = 5,
    seed: int = 42,
    min_df: int = 2,
    max_features: int = 100_000,
    n_jobs: int = 1,
) -> list[CandidateResult]:
    """Cross-validate every candidate on the same folds. Pass training data only."""
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")
    folds = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    results = []
    for kind in CANDIDATES:
        started = time.perf_counter()
        scores = cross_validate(
            build_pipeline(kind, seed=seed, min_df=min_df, max_features=max_features),
            list(texts),
            list(labels),
            cv=folds,
            scoring=_SCORING,
            n_jobs=n_jobs,
            error_score="raise",
        )
        results.append(
            CandidateResult(
                kind=kind,
                name=DISPLAY_NAMES[kind],
                metrics={
                    m: MetricSummary(
                        mean=float(scores[f"test_{m}"].mean()), std=float(scores[f"test_{m}"].std())
                    )
                    for m in _SCORING
                },
                cv_folds=cv_folds,
                total_seconds=round(time.perf_counter() - started, 2),
            )
        )
    return results


def select_best(results: Sequence[CandidateResult]) -> CandidateResult:
    """Highest mean macro-F1; an exact tie goes to the earlier candidate in CANDIDATES."""
    if not results:
        raise ValueError("No candidates to select from.")
    best = results[0]
    for result in results[1:]:
        if result.score > best.score:
            best = result
    return best
