from collections.abc import Callable
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.config import Settings
from app.main import create_app

# Test-only toy corpus with disjoint vocabularies per class. It is used to fit a tiny real
# scikit-learn pipeline so the tests exercise genuine load/predict code paths.
FAKE_TEXT = "shocking miracle secret hoax conspiracy exposed unbelievable banned cure " * 2
REAL_TEXT = "ministry committee quarterly budget report parliament council statistics policy " * 2


def fit_toy_pipeline(labels: tuple[str, str] = ("fake", "real")) -> Pipeline:
    texts = [FAKE_TEXT, FAKE_TEXT + " extra", REAL_TEXT, REAL_TEXT + " extra"] * 3
    targets = [labels[0], labels[0], labels[1], labels[1]] * 3
    pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())])
    return pipeline.fit(texts, targets)


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    return tmp_path / "model.joblib"


@pytest.fixture
def write_model(model_path: Path) -> Callable[[], None]:
    def _write() -> None:
        joblib.dump(fit_toy_pipeline(), model_path)
        model_path.with_name("model.metadata.json").write_text(
            '{"model_name": "toy-test-model", "model_version": "t1", '
            '"trained_at": "2026-01-01T00:00:00+00:00"}'
        )

    return _write


@pytest.fixture
def make_client(model_path: Path) -> Callable[..., TestClient]:
    def _make(**overrides: object) -> TestClient:
        settings = Settings(
            _env_file=None,  # type: ignore[call-arg]
            app_env="test",
            model_path=model_path,
            **overrides,  # type: ignore[arg-type]
        )
        return TestClient(create_app(settings))

    return _make
