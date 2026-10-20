import pytest

# Synthetic, test-only corpus with disjoint vocabularies per class. It exists solely to
# exercise the training code; it is not a dataset and says nothing about real news.
FAKE_WORDS = "shocking miracle secret hoax conspiracy exposed unbelievable banned cure"
REAL_WORDS = "ministry committee quarterly budget report parliament council statistics policy"


def make_rows(n_per_class: int) -> list[tuple[str, str]]:
    rows = []
    for i in range(n_per_class):
        rows.append((f"{FAKE_WORDS} story number {i} {FAKE_WORDS.split()[i % 9]}", "fake"))
        rows.append((f"{REAL_WORDS} update number {i} {REAL_WORDS.split()[i % 9]}", "real"))
    return rows


@pytest.fixture
def rows() -> list[tuple[str, str]]:
    return make_rows(30)
