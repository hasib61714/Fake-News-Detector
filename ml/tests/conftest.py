"""Tiny in-code fixtures. They only exercise the code paths and are NOT a dataset: the texts are
templated nonsense with disjoint class vocabularies and say nothing about real news."""

import csv
import random
from pathlib import Path

import pytest

from fakenews_ml.data import Dataset, load_class_csvs

FAKE_WORDS = [
    "shocking", "miracle", "secret", "hoax", "conspiracy",
    "exposed", "unbelievable", "banned", "cure", "elites",
]  # fmt: skip
REAL_WORDS = [
    "ministry", "committee", "quarterly", "budget", "report",
    "parliament", "council", "statistics", "policy", "officials",
]  # fmt: skip
COMMON = [
    "the", "a", "of", "and", "in", "to", "is", "on", "for",
    "with", "this", "that", "today", "week", "new", "said", "about",
]  # fmt: skip


def make_article(label: str, i: int, rng: random.Random) -> str:
    vocab = FAKE_WORDS if label == "fake" else REAL_WORDS
    words = rng.choices(vocab, k=12) + rng.choices(COMMON, k=14) + [f"item{label}{i}"]
    rng.shuffle(words)
    return " ".join(words)


def make_rows(n_per_class: int, seed: int = 7) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    return [
        (make_article(label, i, rng), label)
        for i in range(n_per_class)
        for label in ("fake", "real")
    ]


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


@pytest.fixture
def rows() -> list[tuple[str, str]]:
    return make_rows(80)


@pytest.fixture
def data_dir(tmp_path: Path, rows: list[tuple[str, str]]) -> Path:
    """A directory holding Fake.csv and True.csv in the layout developers are asked to use."""
    directory = tmp_path / "data"
    directory.mkdir()
    for name, label in (("Fake.csv", "fake"), ("True.csv", "real")):
        write_csv(
            directory / name,
            ["title", "text", "subject", "date"],
            [["t", text, "s", "d"] for text, lab in rows if lab == label],
        )
    return directory


@pytest.fixture
def dataset(data_dir: Path) -> Dataset:
    return load_class_csvs(data_dir / "Fake.csv", data_dir / "True.csv")
