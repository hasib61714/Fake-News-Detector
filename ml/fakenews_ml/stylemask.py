"""Style masking for the leakage ablation (experiment only; not part of the served pipeline).

Some features a news classifier learns are about HOW a publisher writes rather than WHAT it says:
wire-agency attribution ("said", "told Reuters"), weekday datelines, informal contractions. This
module deletes those tokens so an "ablation" model can be trained and compared with the baseline.

Design rules (pre-registered in ml/experiments/style-mask-v1/PROTOCOL.md):

* Tokens are DELETED, not replaced by a placeholder. A placeholder such as "xxmask" would let the
  model learn how many masked words an article contained, which is itself a style signal.
* Matching is whole-word and case-insensitive; rules are fixed hand-written lists, not learned.
* Two levels. ``core`` is exactly the four categories named for the experiment (said-family,
  weekdays, "Reuters", contractions). ``extended`` adds attribution vocabulary, months, outlet
  names, web-page furniture and informal intensifiers. Both are fixed before any result is seen.

The functions are module-level so the sklearn pipeline that contains them can be pickled.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

STYLE_MASK_VERSION = "1"
StyleMaskLevel = Literal["core", "extended"]
LEVELS: tuple[StyleMaskLevel, ...] = ("core", "extended")


@dataclass(frozen=True)
class MaskRule:
    name: str
    category: str
    pattern: str  # a regex source; compiled case-insensitively


def _words(*words: str) -> str:
    return r"\b(?:" + "|".join(words) + r")\b"


# -- core: the four categories named for the experiment ------------------------------------------

_NEGATIVE_STEMS = (
    "isn|aren|wasn|weren|doesn|didn|don|hasn|haven|hadn|couldn|wouldn|shouldn|mustn|needn|ain|"
    "can|won"
)

CORE_RULES: tuple[MaskRule, ...] = (
    MaskRule("said_family", "reporting verb", _words("say", "says", "said", "saying", "told")),
    MaskRule(
        "weekdays",
        "weekday name",
        r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?\b",
    ),
    # Every mention of the agency: "told Reuters", "Reuters/Ipsos poll", "Thomson Reuters".
    MaskRule("reuters", "agency attribution", _words("reuters")),
    # "isn't" / "isn’t" / "isn t" (the dataset's fake articles have apostrophes stripped), "cannot".
    MaskRule(
        "negative_contractions",
        "contraction",
        rf"\b(?:{_NEGATIVE_STEMS})['’]?\s?t\b|\bcannot\b",
    ),
    # "they're", "we've", "I'll" and the apostrophe-stripped forms "they re", "we ve", "I ll".
    MaskRule("clitics", "contraction", r"['’](?:ll|ve|re|d|m)\b|\b(?:ll|ve|re)\b"),
)

# -- extended: stricter, adds further style/format categories ------------------------------------

EXTENDED_EXTRA_RULES: tuple[MaskRule, ...] = (
    MaskRule(
        "attribution_vocabulary",
        "attribution",
        _words(
            "according",
            "report",
            "reports",
            "reported",
            "reportedly",
            "reporter",
            "reporters",
            "spokesman",
            "spokeswoman",
            "spokesperson",
            "statement",
            "comment",
            "comments",
            "citing",
            "cited",
            "quoted",
            "added",
            "stated",
            "declined",
            "confirmed",
        ),
    ),  # fmt: skip
    MaskRule(
        "months",
        "date",
        _words(
            "january",
            "february",
            "april",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
            "jan",
            "feb",
            "apr",
            "jun",
            "jul",
            "aug",
            "sept?",
            "oct",
            "nov",
            "dec",
        ),
    ),  # fmt: skip; "may" and "march" are excluded on purpose: they are ordinary words
    MaskRule(
        "outlet_names",
        "outlet",
        _words(
            "breitbart",
            "cnn",
            "msnbc",
            "politico",
            "huffpost",
            "huffington post",
            "bbc",
            "nypost",
            "fox news",
            "nbc news",
            "abc news",
            "cbs news",
            "daily mail",
            "new york post",
            "new york times",
            "washington post",
            "associated press",
        ),
    ),  # fmt: skip
    MaskRule(
        "page_furniture",
        "web format",
        _words(
            "read",
            "watch",
            "video",
            "videos",
            "share",
            "subscribe",
            "via",
            "image",
            "images",
            "photo",
            "photos",
            "pic",
            "twitter",
            "facebook",
            "youtube",
            "instagram",
            "com",
        ),
    ),  # fmt: skip
    MaskRule(
        "informal_intensifiers",
        "informal style",
        _words(
            "just",
            "like",
            "actually",
            "apparently",
            "really",
            "literally",
            "totally",
            "basically",
            "obviously",
            "entire",
            "entirely",
            "simply",
        ),
    ),  # fmt: skip
)

RULES: dict[StyleMaskLevel, tuple[MaskRule, ...]] = {
    "core": CORE_RULES,
    "extended": CORE_RULES + EXTENDED_EXTRA_RULES,
}

_COMPILED: dict[StyleMaskLevel, re.Pattern[str]] = {
    level: re.compile("|".join(f"(?:{r.pattern})" for r in rules), re.IGNORECASE)
    for level, rules in RULES.items()
}
_PER_RULE: dict[str, re.Pattern[str]] = {
    r.name: re.compile(r.pattern, re.IGNORECASE) for r in RULES["extended"]
}
_WHITESPACE = re.compile(r"\s+")


def mask_text(text: str, level: StyleMaskLevel) -> str:
    """Delete every masked token (replaced by a space) and collapse whitespace."""
    return _WHITESPACE.sub(" ", _COMPILED[level].sub(" ", text)).strip()


def mask_texts_core(texts: Iterable[str]) -> list[str]:
    return [mask_text(t, "core") for t in texts]


def mask_texts_extended(texts: Iterable[str]) -> list[str]:
    return [mask_text(t, "extended") for t in texts]


def mask_function(level: StyleMaskLevel):  # type: ignore[no-untyped-def]
    """The module-level batch function for ``level`` (picklable, used by the sklearn step)."""
    return {"core": mask_texts_core, "extended": mask_texts_extended}[level]


def rule_hits(text: str, level: StyleMaskLevel) -> dict[str, int]:
    """Number of matches per rule in ``text`` (for descriptive reporting and tests)."""
    return {r.name: len(_PER_RULE[r.name].findall(text)) for r in RULES[level]}
