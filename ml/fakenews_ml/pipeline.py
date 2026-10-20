from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

MODEL_NAME = "tfidf-logistic-regression"


def build_pipeline(min_df: int = 2, seed: int = 42) -> Pipeline:
    """TF-IDF features into logistic regression.

    Deliberately plain scikit-learn objects only, so the API can unpickle the artifact
    without importing this package.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=min_df,
                    max_features=100_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
            ),
        ]
    )
