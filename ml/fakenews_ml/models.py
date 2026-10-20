"""Candidate models. Every candidate is a full pipeline: cleaning -> TF-IDF -> classifier.

Both candidates expose ``predict_proba`` so the API contract and the uncertainty rule work the
same way for either. Linear SVM has no native probabilities, so it is wrapped in sigmoid
calibration, which is fitted on the training data only.
"""

from typing import Literal

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from fakenews_ml.preprocessing import clean_texts

CandidateKind = Literal["logistic_regression", "linear_svm"]
# Order matters: on an exact tie in the selection metric the earlier (simpler) one wins.
CANDIDATES: tuple[CandidateKind, ...] = ("logistic_regression", "linear_svm")

DISPLAY_NAMES: dict[CandidateKind, str] = {
    "logistic_regression": "tfidf-logistic-regression",
    "linear_svm": "tfidf-linear-svm-calibrated",
}


def build_pipeline(
    kind: CandidateKind, *, seed: int = 42, min_df: int = 2, max_features: int = 100_000
) -> Pipeline:
    classifier: LogisticRegression | CalibratedClassifierCV
    if kind == "logistic_regression":
        classifier = LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced", random_state=seed
        )
    elif kind == "linear_svm":
        svm = LinearSVC(C=1.0, class_weight="balanced", dual="auto", random_state=seed)
        classifier = CalibratedClassifierCV(svm, method="sigmoid", cv=3)
    else:
        raise ValueError(f"Unknown candidate {kind!r}")

    return Pipeline(
        [
            ("clean", FunctionTransformer(clean_texts, validate=False)),
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=min_df,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
            ("clf", classifier),
        ]
    )
