"""Evaluation metrics. Everything returned is JSON-serializable and goes into the metadata."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from fakenews_ml.data import LABELS


def predict_top(pipeline: Any, texts: Sequence[str]) -> tuple[list[str], np.ndarray]:
    """Top class and its probability for each text, using the pipeline's own ``classes_``."""
    probabilities = pipeline.predict_proba(list(texts))
    classes = [str(c) for c in pipeline.classes_]
    top = probabilities.argmax(axis=1)
    return [classes[i] for i in top], probabilities.max(axis=1)


def _block(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, Any]:
    labels = list(LABELS)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
        "macro_avg": {
            "precision": float(macro[0]),
            "recall": float(macro[1]),
            "f1": float(macro[2]),
        },
        # Rows are the true class, columns the predicted class, both in `labels` order.
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        },
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
        "classification_report_text": classification_report(
            y_true, y_pred, labels=labels, zero_division=0, digits=4
        ),
    }


def evaluate(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    confidences: Sequence[float],
    threshold: float,
) -> dict[str, Any]:
    """Metrics for the model's top-class predictions, plus the effect of the uncertainty rule.

    ``all_predictions``: every sample scored by its top class (ignores the threshold).
    ``decisive_only``: only samples confident enough to be called fake/real.
    """
    if not (len(y_true) == len(y_pred) == len(confidences)):
        raise ValueError("y_true, y_pred and confidences must have the same length.")
    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty split.")

    truth = np.asarray(y_true)
    pred = np.asarray(y_pred)
    decisive = np.asarray(confidences, dtype=float) >= threshold
    uncertain = ~decisive

    result: dict[str, Any] = {
        "all_predictions": _block(list(truth), list(pred)),
        "uncertainty": {
            "threshold": threshold,
            "n_decisive": int(decisive.sum()),
            "n_uncertain": int(uncertain.sum()),
            "coverage": float(decisive.mean()),
            "uncertain_rate": float(uncertain.mean()),
            # How often the top class was right on the articles the rule set aside.
            "top_class_accuracy_on_uncertain": (
                float((truth[uncertain] == pred[uncertain]).mean()) if uncertain.any() else None
            ),
        },
        "decisive_only": (
            _block(list(truth[decisive]), list(pred[decisive])) if decisive.any() else None
        ),
    }
    return result
