"""Test fixtures.

The API tests exercise the REAL artifact path: a model is trained with ``fakenews_ml`` on a
tiny, in-code corpus (templated nonsense with disjoint class vocabularies; it is NOT a dataset
and says nothing about real news), written with ``write_release`` and loaded by the service.
"""

import csv
import random
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
from fakenews_ml.artifacts import write_release
from fakenews_ml.data import load_data_dir
from fakenews_ml.train import TrainingConfig, train
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

FAKE_WORDS = [
    "shocking", "miracle", "secret", "hoax", "conspiracy",
    "exposed", "unbelievable", "banned", "cure", "elites",
]  # fmt: skip
REAL_WORDS = [
    "ministry", "committee", "quarterly", "budget", "report",
    "parliament", "council", "statistics", "policy", "officials",
]  # fmt: skip
COMMON = ["the", "a", "of", "and", "in", "to", "is", "on", "for", "with", "this", "that", "today"]

FAKE_TEXT = " ".join(FAKE_WORDS * 2) + " said today"
REAL_TEXT = " ".join(REAL_WORDS * 2) + " said today"
# Equal parts fake and real vocabulary: the model should be unable to choose confidently.
AMBIGUOUS_TEXT = " ".join(FAKE_WORDS[:5] + REAL_WORDS[:5] + FAKE_WORDS[:5] + REAL_WORDS[:5])


def _article(label: str, i: int, rng: random.Random) -> str:
    vocab = FAKE_WORDS if label == "fake" else REAL_WORDS
    words = rng.choices(vocab, k=12) + rng.choices(COMMON, k=14) + [f"item{label}{i}"]
    rng.shuffle(words)
    return " ".join(words)


@pytest.fixture(scope="session")
def trained_output(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train once per test session; tests copy the resulting files."""
    root = tmp_path_factory.mktemp("trained")
    data = root / "data"
    data.mkdir()
    rng = random.Random(5)
    for name, label in (("Fake.csv", "fake"), ("True.csv", "real")):
        with (data / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["text"])
            writer.writerows([[_article(label, i, rng)] for i in range(80)])

    artifact = train(load_data_dir(data), TrainingConfig(cv_folds=3, min_df=1, seed=5))
    out = root / "artifacts"
    write_release(artifact, out)
    return out


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    return tmp_path / "model.joblib"


@pytest.fixture
def write_model(model_path: Path, trained_output: Path) -> Callable[[], None]:
    def _write() -> None:
        # Metadata first, model last: the same order training uses when it activates a model.
        shutil.copyfile(
            trained_output / "model.metadata.json", model_path.with_name("model.metadata.json")
        )
        shutil.copyfile(trained_output / "model.joblib", model_path)

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
