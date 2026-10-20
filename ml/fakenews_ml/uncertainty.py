"""The 'uncertain' category: a confidence threshold derived from the VALIDATION split.

Rule at prediction time: if ``max(P(fake), P(real)) < threshold`` the answer is "uncertain"
instead of "fake"/"real".

Deriving the threshold (never uses the training or test split):
1. Candidate thresholds run from ``min_threshold`` to 0.99 in steps of 0.01.
2. A threshold is *admissible* if it still gives a decisive answer for at least ``min_coverage``
   of validation articles (so the model cannot "win" by abstaining on everything).
3. Choose the smallest admissible threshold whose decisive answers are at least
   ``target_accuracy`` correct on validation. If none reaches the target, choose the admissible
   threshold with the highest decisive accuracy and record ``target_met = False``.
4. ``min_threshold`` is a floor: a prediction below that confidence (60% by default) is never
   presented as fake/real, whatever the validation data says.
"""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

GRID_STEP = 0.01
GRID_MAX = 0.99


@dataclass(frozen=True)
class UncertaintyRule:
    threshold: float
    min_threshold: float
    target_accuracy: float
    min_coverage: float
    target_met: bool
    validation_samples: int
    validation_coverage: float
    validation_decisive_accuracy: float | None

    def is_uncertain(self, confidence: float) -> bool:
        return confidence < self.threshold

    def to_dict(self) -> dict[str, Any]:
        return {"rule": "max_probability < threshold", **asdict(self)}


def derive_rule(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    target_accuracy: float = 0.95,
    min_coverage: float = 0.5,
    min_threshold: float = 0.6,
) -> UncertaintyRule:
    """``confidences[i]`` is the top-class probability and ``correct[i]`` whether it was right."""
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have the same length.")
    if len(confidences) == 0:
        raise ValueError("Cannot derive an uncertainty rule from an empty validation set.")
    if not 0.5 <= min_threshold < GRID_MAX:
        raise ValueError(f"min_threshold must be in [0.5, {GRID_MAX}); got {min_threshold}.")
    if not 0 < target_accuracy <= 1:
        raise ValueError(f"target_accuracy must be in (0, 1]; got {target_accuracy}.")
    if not 0 < min_coverage <= 1:
        raise ValueError(f"min_coverage must be in (0, 1]; got {min_coverage}.")

    conf = np.asarray(confidences, dtype=float)
    ok = np.asarray(correct, dtype=bool)
    n = len(conf)

    steps = int(round((GRID_MAX - min_threshold) / GRID_STEP))
    grid = [round(min_threshold + i * GRID_STEP, 4) for i in range(steps + 1)]

    def stats(threshold: float) -> tuple[float, float | None]:
        accepted = conf >= threshold
        count = int(accepted.sum())
        return count / n, (float(ok[accepted].mean()) if count else None)

    admissible = [(t, *stats(t)) for t in grid if stats(t)[0] >= min_coverage]
    if not admissible:
        chosen_t, coverage, accuracy = min_threshold, *stats(min_threshold)
        met = False
    else:
        meeting = [row for row in admissible if row[2] is not None and row[2] >= target_accuracy]
        if meeting:
            chosen_t, coverage, accuracy = meeting[0]
            met = True
        else:
            # Highest decisive accuracy; the smallest threshold wins ties (max keeps the first).
            chosen_t, coverage, accuracy = max(admissible, key=lambda row: row[2] or 0.0)
            met = False

    return UncertaintyRule(
        threshold=chosen_t,
        min_threshold=min_threshold,
        target_accuracy=target_accuracy,
        min_coverage=min_coverage,
        target_met=met,
        validation_samples=n,
        validation_coverage=coverage,
        validation_decisive_accuracy=accuracy,
    )
