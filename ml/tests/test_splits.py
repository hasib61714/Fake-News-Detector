import pytest

from fakenews_ml.splits import split_indices, take


def labels(n_fake: int = 100, n_real: int = 100) -> list[str]:
    return ["fake"] * n_fake + ["real"] * n_real


def test_splits_are_disjoint_and_cover_everything() -> None:
    y = labels()
    splits = split_indices(y, 0.2, 0.2, seed=1)

    parts = [set(splits.train), set(splits.validation), set(splits.test)]
    assert parts[0].isdisjoint(parts[1]) and parts[0].isdisjoint(parts[2])
    assert parts[1].isdisjoint(parts[2])
    assert set().union(*parts) == set(range(len(y)))


def test_split_sizes_follow_fractions_of_the_whole_dataset() -> None:
    splits = split_indices(labels(), 0.2, 0.2, seed=1)
    assert (len(splits.train), len(splits.validation), len(splits.test)) == (120, 40, 40)


def test_splits_are_stratified() -> None:
    y = labels(150, 50)
    splits = split_indices(y, 0.2, 0.2, seed=3)
    for part in (splits.train, splits.validation, splits.test):
        share = sum(y[i] == "fake" for i in part) / len(part)
        assert share == pytest.approx(0.75, abs=0.03)


def test_same_seed_same_split_different_seed_different_split() -> None:
    y = labels()
    assert split_indices(y, 0.2, 0.2, 5) == split_indices(y, 0.2, 0.2, 5)
    assert split_indices(y, 0.2, 0.2, 5).test != split_indices(y, 0.2, 0.2, 6).test


@pytest.mark.parametrize("test_size, validation_size", [(0, 0.2), (0.2, 0), (0.6, 0.4), (1.0, 0.1)])
def test_invalid_sizes_are_rejected(test_size: float, validation_size: float) -> None:
    with pytest.raises(ValueError, match="sum to less than 1"):
        split_indices(labels(), test_size, validation_size, seed=1)


def test_take() -> None:
    assert take(["a", "b", "c"], [2, 0]) == ["c", "a"]
