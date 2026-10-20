import pytest
from sklearn.preprocessing import FunctionTransformer

import fakenews_ml.preprocessing as preprocessing
from fakenews_ml.models import CANDIDATES, build_pipeline
from fakenews_ml.preprocessing import (
    BOILERPLATE_RULES,
    PREPROCESSING_VERSION,
    clean_text,
    clean_texts,
    explain,
)

# -- version -------------------------------------------------------------------------------------


def test_version_is_pinned_to_2() -> None:
    """Changing cleaning output requires a new version AND a deliberate edit of this test."""
    assert PREPROCESSING_VERSION == "2"


# -- generic cleaning ----------------------------------------------------------------------------


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
def test_generic_cleaning(raw: str, expected: str) -> None:
    assert clean_text(raw) == expected


def test_clean_texts_maps_over_a_batch() -> None:
    assert clean_texts(["<i>a</i>", "b  c"]) == ["a", "b c"]


# -- Reuters / wire datelines --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("WASHINGTON (Reuters) - The committee met.", "The committee met."),
        ("XIAMEN, China (Reuters) - Russian President spoke.", "Russian President spoke."),
        ("TAMPA, Fla. (Reuters) - An officer retired.", "An officer retired."),
        ("NEW YORK/LONDON (Reuters) - Shares fell.", "Shares fell."),
        ("CARLSBAD CAVERNS NATIONAL PARK, N.M. (Reuters) - A fire.", "A fire."),
        ("(AP) — Officials said.", "Officials said."),
        ("WASHINGTON (AP) - The vote passed.", "The vote passed."),
        ("PARIS (AFP) - Talks began.", "Talks began."),
        ("(Reuters) - Stocks rose.", "Stocks rose."),
    ],
)
def test_leading_datelines_are_removed(raw: str, expected: str) -> None:
    assert clean_text(raw) == expected


@pytest.mark.parametrize(
    "text",
    [
        "The Associated Press (AP) reported that the vote passed on Monday.",
        "Agence France-Presse (AFP) said the talks had ended on Friday night.",
        "President Obama (Reuters photo) arrived at the summit on Sunday morning.",
    ],
)
def test_ap_afp_abbreviations_and_other_parentheticals_are_kept(text: str) -> None:
    assert clean_text(text) == text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Not a leading dateline, but the "(Reuters)" wire tag itself is still removed.
        ("Officials said (Reuters) that it rained.", "Officials said that it rained."),
        # A mixed-case place is not an all-caps dateline, so only the tag is removed.
        ("Washington (Reuters) - A mixed-case place.", "Washington A mixed-case place."),
    ],
)
def test_reuters_tag_is_removed_even_when_it_is_not_a_leading_dateline(
    raw: str, expected: str
) -> None:
    assert clean_text(raw) == expected


def test_stray_reuters_tag_is_removed_but_ap_abbreviation_is_not() -> None:
    assert clean_text("Prices rose. LONDON (Reuters) - Another story.") == (
        "Prices rose. LONDON Another story."
    )
    assert clean_text("Sydney - (Reuters) - Australia will send troops.") == (
        "Sydney - Australia will send troops."
    )
    assert clean_text("The Associated Press (AP) said so.") == "The Associated Press (AP) said so."


# -- Reuters credit lines and sponsored footers --------------------------------------------------


def test_credit_line_parenthetical_is_removed() -> None:
    raw = "Lynch spoke on Monday. (Reporting by Eric Walsh; Editing by Mohammad Zargham)"
    assert clean_text(raw) == "Lynch spoke on Monday."
    raw = "Done. (Additional reporting by Megan Cassella in Detroit; Writing by Jo Doe) Next."
    assert clean_text(raw) == "Done. Next."


@pytest.mark.parametrize(
    "text",
    [
        "Analysts said (reporting by the Times, which first revealed the memo) it was final.",
        "The study (writing by hand improves recall) found strong effects across groups.",
        "He objected (editing by committee ruined the draft) during the long meeting today.",
    ],
)
def test_parentheticals_that_are_not_credits_are_kept(text: str) -> None:
    assert clean_text(text) == text


def test_sponsor_footers_are_removed_in_both_exact_templates() -> None:
    a = (
        "He agreed. (Editing by Caren Bohan) This article was funded in part by SAP. It was "
        "independently created by the Reuters editorial staff. SAP had no editorial involvement "
        "in its creation or production."
    )
    b = (
        "He agreed. SAP is the sponsor of this content. It was independently created by Reuters’ "
        "editorial staff and funded in part by SAP, which otherwise has no role in this coverage."
    )
    assert clean_text(a) == "He agreed."
    assert clean_text(b) == "He agreed."


def test_text_that_merely_resembles_the_sponsor_footer_is_kept() -> None:
    text = "The article was independently created by the editorial staff of the local paper."
    assert clean_text(text) == text


# -- tweet embeds --------------------------------------------------------------------------------


def test_tweet_embed_links_and_attributions_are_removed_but_names_and_text_stay() -> None:
    raw = (
        "Nothing found. pic.twitter.com/3vD5nFgE36 Senator Hatch Office (@senorrinhatch) "
        "August 10, 2017 Of course, Trump is not one of them."
    )
    assert (
        clean_text(raw)
        == "Nothing found. Senator Hatch Office Of course, Trump is not one of them."
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Watch bit.ly/2jBh4LU now", "Watch now"),
        ("see t.co/AbC123 here", "see here"),
        ("Watch this pic.twitter.com/fRb6ombu2x#Egypt now", "Watch this #Egypt now"),
        ("Trump (@realDonaldTrump) Sept. 17, 2012 said so", "Trump said so"),
    ],
)
def test_short_links_and_tweet_dates(raw: str, expected: str) -> None:
    assert clean_text(raw) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Rubio tweeted Madlaw (@madlaw1071) on Tuesday evening about the vote.",
        "He criticized twitter.com and the company in a long speech on Wednesday.",
        "The law passed on March 4, 2017 and took effect on May 1, 2018 across the state.",
        "Reporters filed pic and video requests with the mayor's office before the briefing.",
    ],
)
def test_twitter_mentions_dates_and_prose_are_kept(text: str) -> None:
    assert clean_text(text) == text


# -- image credits -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("He should resign.Featured image via Mark Wilson/Getty Images", "He should resign."),
        ("It is awful. Featured Image: Wikimedia", "It is awful."),
        ("Resistance is not futile. Photo by Alex Wong/Getty Images.", "Resistance is not futile."),
        ("Nixon resigned in 1974. Image via Twitter.", "Nixon resigned in 1974."),
        ("Sad.Featured image Shawn Thew Pool/Getty Images", "Sad."),
        ("Read on.Featured Image Credit: Gage Skidmore via Flickr", "Read on."),
        ("Terrible.Featured image is a screenshot", "Terrible."),
        ("Over.Featured image via video screen capture", "Over."),
    ],
)
def test_trailing_image_credits_are_removed(raw: str, expected: str) -> None:
    assert clean_text(raw) == expected


def test_multi_line_credit_block_at_the_end_is_removed() -> None:
    raw = (
        "Done.Featured Image: Photo of Donald Trump by Joe Raedle/Getty Images. Photo of "
        "Vladimir Putin by Thierry Chesnot/Getty Images"
    )
    assert clean_text(raw) == "Done."


def test_only_the_final_credit_is_removed_so_prose_between_two_credits_survives() -> None:
    raw = (
        "Image via TumblrAnd that is how workers fight back against blowhards."
        "Featured image via Flickr"
    )
    assert (
        clean_text(raw) == "Image via TumblrAnd that is how workers fight back against blowhards."
    )


@pytest.mark.parametrize(
    "text",
    [
        "The photo by the journalist won an award last year, and the editors praised it.",
        "Officials released the image from the incident, which was shared widely online.",
        "He changed the featured image on the site yesterday and nobody noticed at all.",
        "The agency Getty Images announced new licensing terms for editorial photographers.",
        "A photo by Reuters shows the scene, and the caption explains what happened there.",
        # Capitalized leads followed by ordinary lowercase prose are not credits.
        "Officials spoke later. Photo by the roadside stand owner showed the flood damage clearly.",
        "The crowd grew. Image from the incident spread quickly across social media that evening.",
        "Analysts disagree. Photo: the new design is bolder than the last one, they said on stage.",
    ],
)
def test_ordinary_prose_about_images_is_kept(text: str) -> None:
    assert clean_text(text) == text


def test_a_credit_that_is_not_at_the_end_is_kept() -> None:
    prose = " The council debated the proposal for many hours and then voted on each amendment."
    text = "Image via Twitter." + prose * 3  # more than 160 characters follow the credit
    assert clean_text(text) == text


# -- 21st Century Wire ---------------------------------------------------------------------------


def test_21wire_lead_footer_and_appeal_are_removed() -> None:
    assert clean_text("21st Century Wire says More evidence has surfaced.") == (
        "More evidence has surfaced."
    )
    raw = (
        "Story end.READ MORE OBAMA NEWS AT: 21st Century Wire Obama FilesSUPPORT 21WIRE "
        "SUBSCRIBE & BECOME A MEMBER @21WIRE.TV"
    )
    assert clean_text(raw) == "Story end."


def test_text_after_a_21wire_footer_is_not_deleted() -> None:
    raw = (
        "Great piece. READ MORE SHADOW GOVERNMENT NEWS AT: 21st Century Wire Shadow Government "
        "FilesTo hear Jay s podcasts, subscribe. Jay Dyer is the author."
    )
    assert clean_text(raw) == (
        "Great piece. To hear Jay s podcasts, subscribe. Jay Dyer is the author."
    )


def test_21st_century_in_prose_is_kept() -> None:
    text = "In the 21st century, wire services and the 21st Century Wire of old changed news."
    assert clean_text(text) == text


# -- realistic article text must not change ------------------------------------------------------

NORMAL_ARTICLES = [
    "The Senate voted 52-48 on Thursday to confirm the nominee, after a debate that lasted "
    "nearly eleven hours. Majority Leader Smith told reporters the vote reflected months of "
    "work, while the minority leader said Democrats would keep pressing for hearings.",
    "In a letter seen by Reuters on Friday, a group of lawmakers urged the agency to delay the "
    "rule. A Reuters/Ipsos poll in June found 50 percent of respondents supported the plan, "
    "up from 44 percent in March 2017.",
    'Prosecutors said the suspect, 34, was arrested on Monday. "We acted on a tip," police '
    "spokeswoman Kristy Campbell said. The photo by the bystander, shared widely, showed smoke "
    "rising over the harbor.",
    "Economists at Getty Images' parent company and elsewhere expect growth of 2.1 percent this "
    "year, according to a survey published on May 1, 2018 (see the footnote on page 12).",
    "She wrote on Twitter (@jane_doe) that the speech was long, and a user named Madlaw "
    "(@madlaw1071) replied on Tuesday evening. Neither posted a link to twitter.com in the end.",
    "The 21st century has changed newsrooms. Reporting by freelancers has grown, and editing by "
    "algorithm is now common at some outlets, according to a study.",
    "Der Minister sagte am Montag in Berlin, die Regierung werde das Gesetz nicht ändern. "
    "Café owners in São Paulo said prices rose 3.5% after the vote.",
]


@pytest.mark.parametrize("text", NORMAL_ARTICLES)
def test_normal_article_text_is_unchanged(text: str) -> None:
    assert clean_text(text) == text
    assert explain(text) == []  # and no boilerplate rule fired


# -- invariants ----------------------------------------------------------------------------------

BOILERPLATE_SAMPLES = [
    "WASHINGTON (Reuters) - Lynch spoke. (Reporting by Eric Walsh; Editing by Mohammad Zargham) "
    "This article was funded in part by SAP. It was independently created by the Reuters "
    "editorial staff. SAP had no editorial involvement in its creation or production.",
    "21st Century Wire says Nothing found. pic.twitter.com/3vD5nFgE36 Name (@abc) August 10, 2017 "
    "Trump spoke.Featured image via Mark Wilson/Getty Images",
    "Body text here.READ MORE TRUMP NEWS AT: 21st Century Wire Trump FilesSUPPORT 21WIRE "
    "SUBSCRIBE & BECOME A MEMBER @21WIRE.TV",
]


@pytest.mark.parametrize("text", [*BOILERPLATE_SAMPLES, *NORMAL_ARTICLES])
def test_cleaning_is_idempotent(text: str) -> None:
    once = clean_text(text)
    assert clean_text(once) == once


def test_explain_reports_the_rules_that_fired_and_agrees_with_clean_text() -> None:
    fired = {name for name, _ in explain(BOILERPLATE_SAMPLES[0])}
    assert fired == {"wire_dateline", "credit_line", "sponsor_footer"}
    fired = {name for name, _ in explain(BOILERPLATE_SAMPLES[1])}
    assert fired == {"wire21_lead", "bare_link", "tweet_attribution", "image_credit"}


def test_every_rule_is_named_uniquely_and_exercised_by_the_samples() -> None:
    names = [rule.name for rule in BOILERPLATE_RULES]
    assert len(names) == len(set(names))
    fired = {name for sample in BOILERPLATE_SAMPLES for name, _ in explain(sample)}
    fired |= {name for name, _ in explain("Prices rose. LONDON (Reuters) - Another story.")}
    assert fired == set(names)


def test_pathological_input_completes() -> None:
    """No catastrophic backtracking on long or adversarial inputs (a hang would fail CI)."""
    for text in (
        "A (" * 20_000,
        "Featured " * 20_000,
        "(Reporting by " * 5_000,
        "READ MORE " * 10_000,
    ):
        assert isinstance(clean_text(text), str)


# -- training and inference use exactly the same preprocessing -----------------------------------


@pytest.mark.parametrize("kind", CANDIDATES)
def test_pipeline_cleaning_step_is_the_shared_preprocessing_function(kind: str) -> None:
    pipeline = build_pipeline(kind, min_df=1)  # type: ignore[arg-type]
    name, step = pipeline.steps[0]
    assert name == "clean"
    assert isinstance(step, FunctionTransformer)
    assert step.func is preprocessing.clean_texts


@pytest.mark.parametrize("kind", CANDIDATES)
def test_pipeline_predictions_equal_predictions_on_manually_cleaned_text(kind: str) -> None:
    texts = [f"shocking miracle hoax secret {i} story" for i in range(30)] + [
        f"ministry committee budget report {i} council" for i in range(30)
    ]
    labels = ["fake"] * 30 + ["real"] * 30
    pipeline = build_pipeline(kind, min_df=1)  # type: ignore[arg-type]
    pipeline.fit(texts, labels)

    raw = [
        "WASHINGTON (Reuters) - ministry committee budget report council. "
        "(Reporting by Jo Doe; Editing by Al Roe)",
        "<p>shocking miracle hoax secret story</p> pic.twitter.com/abc123XYZ0 "
        "Featured image via Getty Images",
    ]
    without_cleaning = pipeline[1:]  # the same fitted TF-IDF + classifier, minus the cleaning step
    assert (
        pipeline.predict_proba(raw).tolist()
        == without_cleaning.predict_proba(clean_texts(raw)).tolist()
    )
