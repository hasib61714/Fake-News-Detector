import pickle

import pytest

from fakenews_ml.data import Dataset
from fakenews_ml.models import CANDIDATES, build_pipeline
from fakenews_ml.stylemask import LEVELS, RULES, mask_text, rule_hits
from fakenews_ml.train import TrainingConfig, main, train

# -- core rules ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Trump said on Tuesday that it rained.", "Trump on that it rained."),
        ("He says she is saying and they told us.", "He she is and they us."),
        ("Voters say no.", "Voters no."),
        ("A statement on Monday, Mondays and SUNDAY.", "A statement on , and ."),
        ("A Wednesday-night vote", "A -night vote"),  # a hyphen is a word boundary
        ("They told Reuters it was Thomson Reuters.", "They it was Thomson ."),
        ("The Reuters/Ipsos poll", "The /Ipsos poll"),
        ("It didn't, isn’t, can't, won't and cannot.", "It , , , and ."),
        ("It doesn t and wasn t and don t.", "It and and ."),  # apostrophe-stripped forms
        ("they're, we've, I'll and they re, we ve, I ll", "they , we , I and they , we , I"),
    ],
)
def test_core_masks_each_category(raw: str, expected: str) -> None:
    assert mask_text(raw, "core") == " ".join(expected.split())


@pytest.mark.parametrize(
    "text",
    [
        "The saidi family met Don Lemon and Don Jr. in Dover.",  # whole words only
        "Mondayish moods and Sundaes are not weekday names.",  # lookalikes without a word boundary
        "In an hour I can tell that you won the bet and that the cantaloupe is ripe.",
        "May and March are months, and the March on Washington was large.",
        "A Reuter in the story is a surname, not an agency.",
        "The words ill, vet, and red are ordinary words.",  # 'll', 've', 're' as substrings
        "According to the report the committee confirmed the figures in November.",  # extended-only
    ],
)
def test_core_leaves_ordinary_text_alone(text: str) -> None:
    assert mask_text(text, "core") == text


# -- extended rules ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("According to reports a spokesman declined to comment.", "to a to ."),
        ("It happened in November and in Sept. and Dec.", "It happened in and in . and ."),
        ("The Washington Post and CNN confirmed it.", "The and it."),
        ("Read more: watch the video via Twitter.", "more: the ."),
        ("He just actually really likes it, apparently.", "He likes it, ."),
    ],
)
def test_extended_adds_its_categories(raw: str, expected: str) -> None:
    assert mask_text(raw, "extended") == " ".join(expected.split())


def test_extended_keeps_may_march_and_other_ordinary_words() -> None:
    text = "May and March are months and the post office will deliver the mail."
    assert mask_text(text, "extended") == text


def test_core_is_a_strict_subset_of_extended() -> None:
    core_names = [r.name for r in RULES["core"]]
    assert [r.name for r in RULES["extended"]][: len(core_names)] == core_names
    assert len(RULES["extended"]) > len(core_names)


# -- invariants ----------------------------------------------------------------------------------


@pytest.mark.parametrize("level", LEVELS)
def test_masking_deletes_and_never_inserts_a_placeholder(level: str) -> None:
    out = mask_text("He said on Monday they didn't tell Reuters the truth.", level)  # type: ignore[arg-type]
    assert set(out.split()) <= {"He", "on", "they", "tell", "the", "truth."}
    assert "xx" not in out.lower() and "mask" not in out.lower()


@pytest.mark.parametrize("level", LEVELS)
def test_masking_is_idempotent_and_whitespace_clean(level: str) -> None:
    raw = "  He said   on Monday, didn't   he?  They told Reuters. "
    once = mask_text(raw, level)  # type: ignore[arg-type]
    assert mask_text(once, level) == once  # type: ignore[arg-type]
    assert once == " ".join(once.split())


def test_rule_hits_counts_matches_per_rule() -> None:
    hits = rule_hits("He said it. She said so on Monday and didn't ask Reuters.", "core")
    assert hits["said_family"] == 2
    assert hits["weekdays"] == 1
    assert hits["reuters"] == 1
    assert hits["negative_contractions"] == 1
    assert hits["clitics"] == 0


# -- pipeline integration ------------------------------------------------------------------------


@pytest.mark.parametrize("kind", CANDIDATES)
def test_baseline_pipeline_has_no_mask_step_unless_requested(kind: str) -> None:
    """The served model's structure must be exactly what it was before this experiment."""
    base = build_pipeline(kind, min_df=1)  # type: ignore[arg-type]
    assert [name for name, _ in base.steps] == ["clean", "tfidf", "clf"]
    masked = build_pipeline(kind, min_df=1, style_mask="core")  # type: ignore[arg-type]
    assert [name for name, _ in masked.steps] == ["clean", "style_mask", "tfidf", "clf"]


@pytest.mark.parametrize("level", LEVELS)
def test_masked_pipeline_masks_after_cleaning_and_survives_pickling(level: str) -> None:
    texts = [f"shocking miracle hoax secret {i} story said" for i in range(20)] + [
        f"ministry committee budget report {i} council told Reuters" for i in range(20)
    ]
    labels = ["fake"] * 20 + ["real"] * 20
    pipeline = build_pipeline("logistic_regression", min_df=1, style_mask=level)  # type: ignore[arg-type]
    pipeline.fit(texts, labels)

    vocab = set(pipeline.named_steps["tfidf"].get_feature_names_out())
    assert "said" not in vocab and "told" not in vocab and "reuters" not in vocab

    restored = pickle.loads(pickle.dumps(pipeline))
    probe = ["shocking miracle hoax said"]
    assert restored.predict_proba(probe).tolist() == pipeline.predict_proba(probe).tolist()


def test_baseline_training_config_has_no_mask_and_cli_cannot_set_one(dataset: Dataset) -> None:
    assert TrainingConfig().style_mask is None
    with pytest.raises(SystemExit) as exc:
        main(["--style-mask", "core"])
    assert exc.value.code == 2


def test_training_with_a_mask_records_it_and_trains_a_working_model(dataset: Dataset) -> None:
    config = TrainingConfig(cv_folds=3, min_df=1, seed=11, style_mask="core")
    artifact = train(dataset, config)
    assert artifact.metadata["config"]["style_mask"] == "core"

    baseline = train(dataset, TrainingConfig(cv_folds=3, min_df=1, seed=11))
    assert baseline.metadata["config"]["style_mask"] is None
