"""Text preprocessing shared by training and inference.

``clean_texts`` is the first step of the saved scikit-learn pipeline, so the exact cleaning used
in training is replayed at prediction time. The artifact records ``PREPROCESSING_VERSION`` and
the loader refuses an artifact whose version differs from the installed code. If you change
anything here that alters output, bump the version and retrain.
"""

import html
import re
import unicodedata
from collections.abc import Iterable

PREPROCESSING_VERSION = "1"

DESCRIPTION = (
    "Unescape HTML entities, NFKC-normalize, strip HTML tags, URLs and e-mail addresses, "
    "remove a leading ALL-CAPS wire-service dateline such as 'WASHINGTON (Reuters) -', "
    "collapse whitespace. Lowercasing, stop-word removal and n-grams happen in TF-IDF."
)

_TAG = re.compile(r"</?[A-Za-z][^>]*>")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_WHITESPACE = re.compile(r"\s+")

# "WASHINGTON (Reuters) - ", "(AP) -", "NEW YORK/LONDON (Reuters) -". Several public fake-news
# datasets contain this marker almost only in the real class, which lets a model "detect" the
# publisher instead of the content. Upper-case only, so ordinary sentences are not touched.
DATELINE = re.compile(r"^\s*(?:[A-Z][A-Z.,'’/ -]{1,60}?\s*)?\((?:Reuters|AP|AFP)\)\s*[-–—:]*\s*")


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = _TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _EMAIL.sub(" ", text)
    text = DATELINE.sub("", text)
    return _WHITESPACE.sub(" ", text).strip()


def clean_texts(texts: Iterable[str]) -> list[str]:
    """Batch form used inside the sklearn pipeline (a FunctionTransformer step)."""
    return [clean_text(text) for text in texts]
