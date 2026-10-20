"""Text preprocessing shared by training and inference.

``clean_texts`` is the first step of the saved scikit-learn pipeline, so the exact cleaning used
in training is replayed at prediction time (and dataset validation de-duplicates with the same
function). The artifact records ``PREPROCESSING_VERSION`` and the loader refuses an artifact whose
version differs from the installed code. If you change anything here that alters output, bump the
version and retrain.

Version 2 removes publisher and image-embed boilerplate that would otherwise let a model "detect"
the source instead of the content. Every boilerplate rule is deliberately narrow: it is anchored
to a specific form (start of text, end of text, or an exact template) and never deletes a bare
word. What is NOT removed, because it is ordinary content, is listed in ``ml/README.md``.
"""

import html
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

PREPROCESSING_VERSION = "2"

DESCRIPTION = (
    "Unescape HTML entities, NFKC-normalize, strip HTML tags, URLs and e-mail addresses; remove "
    "anchored publisher/image-embed boilerplate (wire datelines and '(Reuters)' tags, Reuters "
    "credit lines and sponsored-content footers, trailing image credits such as 'Featured image "
    "via ...', tweet-embed links and '(@handle) Month D, YYYY' attributions, 21st Century Wire "
    "self-promotion); collapse whitespace. Lowercasing, stop-word removal and n-grams happen in "
    "TF-IDF."
)

_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December|"
    "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec"
)

# -- generic cleaning (not counted as "boilerplate" rules) ---------------------------------------

_TAG = re.compile(r"</?[A-Za-z][^>]*>")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_WHITESPACE = re.compile(r"\s+")

# -- boilerplate rules ----------------------------------------------------------------------------

# "WASHINGTON (Reuters) -", "XIAMEN, China (Reuters) -", "NEW YORK/LONDON (Reuters) -", "(AP) -".
# Only at the very start of the text, and the place must begin with an ALL-CAPS word, so an
# ordinary sentence that merely mentions a wire service is untouched.
_CAPS_WORD = r"[A-Z][A-Z.'’-]+"
DATELINE = re.compile(
    rf"^\s*(?:{_CAPS_WORD}(?: {_CAPS_WORD}){{0,3}}(?:/{_CAPS_WORD}(?: {_CAPS_WORD}){{0,2}})?"
    r"(?:,\s*[A-Z][A-Za-z.'’-]*(?: [A-Z][A-Za-z.'’-]*){0,2})?\s*[-–—]?\s*)?"
    r"\((?:Reuters|AP|AFP)\)\s*[-–—:]*\s*"
)

# A "(Reuters)" wire tag left anywhere else (second datelines, odd mixed-case places). "(AP)" and
# "(AFP)" are only removed as a leading dateline: elsewhere they are ordinary abbreviations.
_WIRE_TAG = re.compile(r"\s*\(Reuters\)\s*[-–—:]*\s*")

# "(Reporting by Jane Doe; Editing by John Roe)". The verb is case-insensitive but the credited
# name must be capitalized, so "(reporting by the Times showed ...)" is not a credit line.
_CREDIT_LINE = re.compile(
    r"\(\s*(?i:(?:additional\s+)?(?:reporting|writing|editing|compiled|compiling)"
    r"(?:\s+and\s+writing)?\s+by)\s+[A-Z][^()]{0,300}\)"
)

# Reuters' fixed sponsored-content footers (two exact templates).
_SPONSOR_FOOTER = re.compile(
    r"(?:This article was funded in part by SAP\.|SAP is the sponsor of this content\.)\s*"
    r"It was independently created by (?:the )?Reuters['’]?\s+editorial staff"
    r"(?:\.\s*SAP had no editorial involvement in its creation or production\.|"
    r" and funded in part by SAP, which otherwise has no role in this coverage\.)"
)

# "pic.twitter.com/Abc123", "t.co/xyz", "bit.ly/abc": links without a scheme (the URL rule needs
# one). A bare mention such as "twitter.com" without a path is left alone.
_BARE_LINK = re.compile(
    r"\b(?:pic\.twitter\.com|twitter\.com|t\.co|bit\.ly|youtu\.be|goo\.gl|ow\.ly|dlvr\.it|fb\.me)"
    r"/[\w./?=&%+~-]*",
    re.IGNORECASE,
)

# The attribution line of an embedded tweet: "(@realDonaldTrump) March 4, 2017". The display name
# before it and the tweet text are kept; plain "(@handle)" mentions in prose are not touched.
_TWEET_ATTRIBUTION = re.compile(rf"\(@\w{{1,15}}\)\s+(?:{_MONTHS})\.?\s+\d{{1,2}},\s+\d{{4}}")

# Image credit at the very END of the text: "Featured image via Mark Wilson/Getty Images",
# "Featured Image: Wikimedia", "Photo by Alex Wong/Getty Images.", "Image via Twitter.".
# Requires the capitalized lead word (or "Featured image"), a credit connector, a credited source
# (capitalized, or a known source word) and at most 160 characters to the end of the text.
# Two kinds of credit lead. A capital-F "Featured image/photo" is a credit marker by itself
# ("Featured image Shawn Thew/Getty Images", "Featured Image Credit: ...", "Featured image is a
# screenshot"). Other leads ("Image via", "Photo by", lower-case "featured image via") need a
# credit connector and a credited source, so ordinary prose such as "the featured image on the
# page" or "a photo by the author" is never matched.
_FEATURED_LEAD = r"Featured\s+(?:[Ii]mage|[Pp]hoto(?:graph)?)\b"
_CONNECTED_LEAD = (
    r"(?:[Ff]eatured\s+(?:[Ii]mage|[Pp]hoto(?:graph)?)|Image|Photo(?:graph)?)"
    r"(?:\s+(?:via|by|credit|courtesy of|from)\s+|\s*:\s*)"
)
_ANY_LEAD = rf"(?:{_FEATURED_LEAD}|{_CONNECTED_LEAD})"
_SOURCE_LOOKAHEAD = (
    r"(?=[A-Z@\"“]|(?:screen|video|file|flickr|wikimedia|facebook|twitter|youtube|instagram))"
)
# The tempered dot `(?:(?!lead).)` stops the match from starting at an earlier caption and
# swallowing the prose between two credits: only the final credit at the end is removed.
_IMAGE_CREDIT = re.compile(
    rf"(?:{_FEATURED_LEAD}|{_CONNECTED_LEAD}{_SOURCE_LOOKAHEAD})"
    rf"(?:(?!{_ANY_LEAD}).){{0,160}}$",
    re.DOTALL,
)

# 21st Century Wire self-promotion: a leading "21st Century Wire says", the footer link
# ("READ MORE OBAMA NEWS AT: 21st Century Wire Obama Files") and the closing appeal
# ("SUPPORT 21WIRE SUBSCRIBE ..."). Only the promotion itself is removed, never text after it.
_WIRE21_LEAD = re.compile(r"^\s*21st Century Wire says\s+")
_WIRE21_MORE = re.compile(
    r"READ MORE\b[A-Z0-9 ]{0,60}\bNEWS? AT:\s*21st Century Wire(?:\s+[\w' ]{0,40}?Files)?"
)
_WIRE21_SUPPORT = re.compile(r"SUPPORT 21WIRE\b.{0,160}$", re.DOTALL)


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]
    replacement: str = " "


# Order matters. Footers that end the text run first so that end-anchored rules see the real end.
BOILERPLATE_RULES: tuple[Rule, ...] = (
    Rule("bare_link", _BARE_LINK),
    Rule("tweet_attribution", _TWEET_ATTRIBUTION),
    Rule("wire21_footer", _WIRE21_MORE),
    Rule("wire21_support", _WIRE21_SUPPORT),
    Rule("wire21_lead", _WIRE21_LEAD, ""),
    Rule("wire_dateline", DATELINE, ""),
    Rule("wire_tag", _WIRE_TAG),
    Rule("credit_line", _CREDIT_LINE),
    Rule("sponsor_footer", _SPONSOR_FOOTER),
    Rule("image_credit", _IMAGE_CREDIT, ""),
)


def _generic(text: str) -> str:
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = _TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    return _EMAIL.sub(" ", text)


def _apply_rules(text: str, removed: list[tuple[str, str]] | None = None) -> str:
    """The single code path behind ``clean_text`` and ``explain``."""
    text = _generic(text)
    for rule in BOILERPLATE_RULES:
        if removed is not None:
            removed += [(rule.name, m.group(0).strip()) for m in rule.pattern.finditer(text)]
        # Trailing whitespace is trimmed after every rule so end-anchored rules see the real end.
        text = rule.pattern.sub(rule.replacement, text).rstrip()
    return text


def clean_text(text: str) -> str:
    return _WHITESPACE.sub(" ", _apply_rules(text)).strip()


def explain(text: str) -> list[tuple[str, str]]:
    """The spans each boilerplate rule removes from ``text``, as ``(rule_name, removed_text)``.

    For auditing and tests; it runs the exact same rules as ``clean_text``.
    """
    removed: list[tuple[str, str]] = []
    _apply_rules(text, removed)
    return removed


def clean_texts(texts: Iterable[str]) -> list[str]:
    """Batch form used inside the sklearn pipeline (a FunctionTransformer step)."""
    return [clean_text(text) for text in texts]
