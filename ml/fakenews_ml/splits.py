"""Stratified train / validation / test split, fixed by a seed.

* train: fit models and run cross-validation for model selection.
* validation: derive the 'uncertain' rule. Never used to fit or to choose a model.
* test: evaluated exactly once, after the model and rule are final.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

from sklearn.model_selection import train_test_split

T = TypeVar("T")


@dataclass(frozen=True)
class Splits:
    train: list[int]
    validation: list[int]
    test: list[int]


def split_indices(
    labels: Sequence[str], test_size: float, validation_size: float, seed: int
) -> Splits:
    """Sizes are fractions of the whole dataset, e.g. 0.2 / 0.2 leaves 0.6 for training."""
    if not (0 < test_size < 1 and 0 < validation_size < 1 and test_size + validation_size < 1):
        raise ValueError(
            "test_size and validation_size must be in (0, 1) and sum to less than 1; "
            f"got {test_size} and {validation_size}."
        )
    indices = list(range(len(labels)))
    rest, test = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=list(labels)
    )
    rest_labels = [labels[i] for i in rest]
    train, validation = train_test_split(
        rest,
        test_size=validation_size / (1 - test_size),
        random_state=seed,
        stratify=rest_labels,
    )
    return Splits(train=sorted(train), validation=sorted(validation), test=sorted(test))


def take(items: Sequence[T], indices: Sequence[int]) -> list[T]:
    return [items[i] for i in indices]
