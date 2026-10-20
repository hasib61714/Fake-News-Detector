import pytest

from fakenews_ml.models import build_pipeline
from fakenews_ml.preprocessing import PREPROCESSING_VERSION, clean_text, clean_texts


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("<p>Hello <b>world</b></p>", "Hello world"),
        ("Fish &amp; chips &lt;3", "Fish & chips <3"),
        ("see https://example.com/a?b=1 and www.example.org now", "see and now"),
        ("write to press@example.org today", "write to today"),
        ("  lots \n\n of\t\twhitespace  ", "lots of whitespace"),
        ("ｆｕｌｌｗｉｄｔｈ", "fullwidth"),
        ("a < b and c > d", "a < b and c > d"),
        ("", ""),
    ],
)
def test_clean_text(raw: str, expected: str) -> None:
    assert clean_text(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("WASHINGTON (Reuters) - The committee met.", "The committee met."),
        ("NEW YORK/LONDON (Reuters) - Shares fell.", "Shares fell."),
        ("(AP) — Officials said.", "Officials said."),
        # Only a leading, upper-case dateline is removed.
        ("Officials said (Reuters) that it rained.", "Officials said (Reuters) that it rained."),
        (
            "Washington (Reuters) - mixed case is left alone",
            "Washington (Reuters) - mixed case is left alone",
        ),
    ],
)
def test_wire_dateline_is_stripped_only_at_the_start(raw: str, expected: str) -> None:
    assert clean_text(raw) == expected


def test_clean_text_is_idempotent() -> None:
    raw = "<p>WASHINGTON (Reuters) - Visit http://x.org &amp; read   more</p>"
    assert clean_text(clean_text(raw)) == clean_text(raw)


def test_clean_texts_maps_over_a_batch() -> None:
    assert clean_texts(["<i>a</i>", "b  c"]) == ["a", "b c"]


def test_version_is_a_nonempty_string() -> None:
    assert PREPROCESSING_VERSION


@pytest.mark.parametrize("kind", ["logistic_regression", "linear_svm"])
def test_cleaning_is_part_of_the_saved_pipeline(kind: str) -> None:
    """Raw HTML input must be treated like its cleaned form, because the pipeline cleans it."""
    texts = [f"shocking miracle hoax secret {i} story" for i in range(20)] + [
        f"ministry committee budget report {i} council" for i in range(20)
    ]
    labels = ["fake"] * 20 + ["real"] * 20
    pipeline = build_pipeline(kind, min_df=1)  # type: ignore[arg-type]
    pipeline.fit(texts, labels)

    plain = pipeline.predict_proba(["shocking miracle hoax secret story"])
    wrapped = pipeline.predict_proba(
        ["<div>SHOCKING miracle <a href='http://x.org'>hoax</a> secret &amp; story</div>"]
    )
    assert pipeline.steps[0][0] == "clean"
    assert plain.argmax() == wrapped.argmax()
