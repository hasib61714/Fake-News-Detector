import pytest

from fakenews_ml.uncertainty import derive_rule


def test_perfect_validation_only_needs_the_floor() -> None:
    rule = derive_rule([0.99, 0.9, 0.8, 0.7, 0.65], [True] * 5, min_threshold=0.6)

    assert rule.threshold == 0.6
    assert rule.target_met
    assert rule.validation_coverage == 1.0
    assert rule.validation_decisive_accuracy == 1.0


def test_threshold_rises_until_decisive_answers_reach_the_target() -> None:
    # Everything below 0.80 is wrong, everything at or above it is right.
    confidences = [0.55 + 0.01 * i for i in range(45)]  # 0.55 .. 0.99
    correct = [c >= 0.80 for c in confidences]

    rule = derive_rule(
        confidences, correct, target_accuracy=1.0, min_coverage=0.3, min_threshold=0.5
    )

    assert rule.threshold == pytest.approx(0.80)
    assert rule.target_met
    assert rule.validation_decisive_accuracy == 1.0
    assert rule.validation_coverage == pytest.approx(20 / 45)


def test_min_coverage_stops_the_rule_from_abstaining_on_everything() -> None:
    confidences = [0.6, 0.6, 0.6, 0.6, 0.95]
    correct = [False, False, False, False, True]

    rule = derive_rule(
        confidences, correct, target_accuracy=1.0, min_coverage=0.5, min_threshold=0.6
    )

    # Reaching 100% needs coverage 20%, which is below the 50% minimum: target is not met.
    assert not rule.target_met
    assert rule.validation_coverage >= 0.5


def test_unreachable_target_picks_best_admissible_threshold_and_says_so() -> None:
    confidences = [0.7, 0.7, 0.9, 0.9]
    correct = [True, False, True, False]  # 50% right at every confidence

    rule = derive_rule(
        confidences, correct, target_accuracy=0.99, min_coverage=0.5, min_threshold=0.6
    )

    assert not rule.target_met
    assert rule.threshold == 0.6  # ties keep the smallest threshold
    assert rule.validation_decisive_accuracy == 0.5


def test_floor_is_never_undercut() -> None:
    rule = derive_rule([0.51, 0.52, 0.53, 0.99], [True] * 4, min_threshold=0.75)
    assert rule.threshold == 0.75
    assert rule.is_uncertain(0.74) and not rule.is_uncertain(0.75)


def test_is_uncertain_boundary() -> None:
    rule = derive_rule([0.9] * 5, [True] * 5, min_threshold=0.7)
    assert rule.is_uncertain(0.69999) and not rule.is_uncertain(0.7)


def test_rule_serializes_for_metadata() -> None:
    d = derive_rule([0.9, 0.8], [True, True]).to_dict()
    assert d["rule"] == "max_probability < threshold"
    assert {"threshold", "target_accuracy", "min_coverage", "validation_samples"} <= set(d)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_threshold": 0.4},
        {"min_threshold": 0.99},
        {"target_accuracy": 0},
        {"target_accuracy": 1.1},
        {"min_coverage": 0},
        {"min_coverage": 1.5},
    ],
)
def test_invalid_parameters(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        derive_rule([0.9], [True], **kwargs)


def test_empty_or_mismatched_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        derive_rule([], [])
    with pytest.raises(ValueError, match="same length"):
        derive_rule([0.9, 0.8], [True])
