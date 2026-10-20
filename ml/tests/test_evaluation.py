import pytest

from fakenews_ml.evaluation import evaluate

# Hand-computable case. true:  F F F F R R R R R R
#                       pred:  F F F R R R R R R F   (rows = true, cols = predicted)
Y_TRUE = ["fake"] * 4 + ["real"] * 6
Y_PRED = ["fake", "fake", "fake", "real", "real", "real", "real", "real", "real", "fake"]
CONF = [0.9, 0.9, 0.9, 0.55, 0.9, 0.9, 0.9, 0.9, 0.9, 0.55]


def test_confusion_matrix_and_per_class_metrics_match_hand_calculation() -> None:
    block = evaluate(Y_TRUE, Y_PRED, CONF, threshold=0.5)["all_predictions"]

    assert block["confusion_matrix"] == {"labels": ["fake", "real"], "matrix": [[3, 1], [1, 5]]}
    assert block["n"] == 10
    assert block["accuracy"] == pytest.approx(0.8)
    fake, real = block["per_class"]["fake"], block["per_class"]["real"]
    assert (fake["precision"], fake["recall"], fake["support"]) == (
        pytest.approx(3 / 4),
        pytest.approx(3 / 4),
        4,
    )
    assert real["precision"] == pytest.approx(5 / 6)
    assert real["recall"] == pytest.approx(5 / 6)
    assert fake["f1"] == pytest.approx(0.75)
    assert block["macro_avg"]["f1"] == pytest.approx((0.75 + 5 / 6) / 2)


def test_classification_report_is_present_in_both_forms() -> None:
    block = evaluate(Y_TRUE, Y_PRED, CONF, threshold=0.5)["all_predictions"]
    assert block["classification_report"]["fake"]["support"] == 4
    assert "precision" in block["classification_report_text"]


def test_uncertainty_splits_decisive_from_uncertain_samples() -> None:
    result = evaluate(Y_TRUE, Y_PRED, CONF, threshold=0.6)
    unc = result["uncertainty"]

    assert (unc["n_decisive"], unc["n_uncertain"]) == (8, 2)
    assert unc["coverage"] == pytest.approx(0.8)
    # Both set-aside articles (indices 3 and 9) were wrong, and both are 'uncertain'.
    assert unc["top_class_accuracy_on_uncertain"] == 0.0
    decisive = result["decisive_only"]
    assert decisive["n"] == 8
    assert decisive["accuracy"] == 1.0
    assert decisive["confusion_matrix"]["matrix"] == [[3, 0], [0, 5]]


def test_a_threshold_that_sets_everything_aside_has_no_decisive_block() -> None:
    result = evaluate(Y_TRUE, Y_PRED, CONF, threshold=0.99)
    assert result["decisive_only"] is None
    assert result["uncertainty"]["coverage"] == 0.0


def test_output_is_json_serializable() -> None:
    import json

    json.dumps(evaluate(Y_TRUE, Y_PRED, CONF, threshold=0.6), allow_nan=False)


def test_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="same length"):
        evaluate(["fake"], ["fake", "real"], [0.9], 0.6)
    with pytest.raises(ValueError, match="empty"):
        evaluate([], [], [], 0.6)
