"""Publisher fingerprints that can still reveal the source instead of the content.

These patterns are only *measured*, never removed (removal rules live in ``preprocessing.py``).
After cleaning, dataset validation reports how often each appears per class and warns when one
class contains a marker far more often than the other. Some are ordinary content that happens to
correlate with a source (a wire service's habit of writing "said on Monday"), which is exactly why
they are monitored rather than stripped.
"""

import re

MARKERS: dict[str, re.Pattern[str]] = {
    "mentions 'Reuters'": re.compile(r"\breuters\b", re.IGNORECASE),
    "'(Reuters)' tag": re.compile(r"\(Reuters\)"),
    "wire style 'said on <weekday>'": re.compile(
        r"\bsaid on (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.IGNORECASE
    ),
    "'featured image' / 'image via'": re.compile(
        r"featured (?:image|photo)|image via", re.IGNORECASE
    ),
    "'Getty Images'": re.compile(r"getty images", re.IGNORECASE),
    "tweet embed (pic.twitter / @handle date)": re.compile(
        r"pic\.twitter\.com|\(@\w{1,15}\)\s+[A-Z][a-z]+\.?\s+\d{1,2},\s+\d{4}"
    ),
    "'21st Century Wire' / '21WIRE'": re.compile(r"21st century wire|21wire", re.IGNORECASE),
    "'Read more'": re.compile(r"\bread more\b", re.IGNORECASE),
}

# A marker is flagged when its share differs by at least this many percentage points per class.
GAP_WARN = 0.20


def marker_bits(cleaned_text: str) -> int:
    """Bitmask of which markers occur in ``cleaned_text`` (bit i = i-th marker in ``MARKERS``)."""
    bits = 0
    for i, pattern in enumerate(MARKERS.values()):
        if pattern.search(cleaned_text):
            bits |= 1 << i
    return bits
