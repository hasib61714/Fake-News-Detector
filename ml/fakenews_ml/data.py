"""Dataset loading and validation. No dataset ships with this repository; see ml/README.md.

Expected input is the pair of developer-provided files ``Fake.csv`` and ``True.csv`` (each with
a ``text`` column), or one CSV with ``text`` and ``label`` columns. Whatever the source, rows go
through the same validation, and every decision is counted in a :class:`DatasetReport` that is
stored in the model metadata.
"""

import csv
import hashlib
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fakenews_ml.leakage import GAP_WARN, MARKERS, marker_bits
from fakenews_ml.preprocessing import DATELINE, clean_text

LABELS = ("fake", "real")
FAKE_FILENAME = "Fake.csv"
REAL_FILENAME = "True.csv"

# Keep MIN_TEXT_CHARS in sync with MIN_ARTICLE_CHARS in services/api/app/schemas.py: the API
# rejects shorter input, so the model should not learn from it.
MIN_TEXT_CHARS = 50
MIN_PER_CLASS = 50
SMALL_DATASET_ROWS = 2_000
IMBALANCE_WARN_RATIO = 3.0
MARKER_GAP_WARN = 0.20

# Real-world article bodies exceed csv's default 128 KiB field limit.
csv.field_size_limit(2**31 - 1)


class DatasetError(ValueError):
    """The dataset is missing, malformed or too small to train on."""


@dataclass(frozen=True)
class SourceFile:
    name: str
    sha256: str
    rows: int
    label: str | None  # None when the file carries its own label column


@dataclass
class ClassCounts:
    rows_read: int = 0
    dropped_empty: int = 0
    dropped_too_short: int = 0
    dropped_duplicates: int = 0
    dropped_label_conflicts: int = 0
    kept: int = 0
    median_chars: float | None = None
    dateline_share: float | None = None  # raw text, before cleaning
    # Share of kept articles containing each monitored marker AFTER cleaning (see leakage.py).
    marker_shares: dict[str, float] = field(default_factory=dict)


@dataclass
class DatasetReport:
    sources: list[SourceFile]
    classes: dict[str, ClassCounts]
    total_kept: int
    fingerprint: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format(self) -> str:
        lines = ["Dataset validation report", "-------------------------"]
        for source in self.sources:
            lines.append(f"  {source.name}: {source.rows:,} rows, sha256 {source.sha256[:12]}…")
        for label, c in self.classes.items():
            lines.append(
                f"  [{label}] read {c.rows_read:,} | empty {c.dropped_empty:,} | "
                f"< {MIN_TEXT_CHARS} chars {c.dropped_too_short:,} | duplicates "
                f"{c.dropped_duplicates:,} | cross-class conflicts {c.dropped_label_conflicts:,} "
                f"| kept {c.kept:,} (median {c.median_chars:.0f} chars)"
                if c.median_chars is not None
                else f"  [{label}] read {c.rows_read:,} | kept 0"
            )
        lines.append(f"  total kept: {self.total_kept:,}   fingerprint: {self.fingerprint[:16]}…")
        shares = {label: c.marker_shares for label, c in self.classes.items()}
        if all(shares.values()):
            lines.append("  Residual source markers after cleaning (share of articles):")
            for name in MARKERS:
                f, r = shares["fake"][name], shares["real"][name]
                flag = "  <-- differs by class" if abs(f - r) >= GAP_WARN else ""
                lines.append(f"    {name:<42} fake {f:6.1%}   real {r:6.1%}{flag}")
        lines += [f"  WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


@dataclass(frozen=True)
class Dataset:
    """Raw (uncleaned) texts: the saved pipeline performs its own cleaning."""

    texts: list[str]
    labels: list[str]
    report: DatasetReport


@dataclass(frozen=True)
class _Source:
    file: SourceFile
    rows: list[tuple[str, str]]  # (raw text, label)


# -- reading -----------------------------------------------------------------


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path, required: list[str]) -> list[dict[str, str | None]]:
    if not path.is_file():
        raise DatasetError(f"Dataset file not found: {path}")
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            missing = [c for c in required if c not in (reader.fieldnames or [])]
            if missing:
                raise DatasetError(
                    f"{path.name} is missing column(s) {missing}; found {reader.fieldnames}."
                )
            return list(reader)
    except UnicodeDecodeError as exc:
        raise DatasetError(f"{path.name} is not valid UTF-8 text: {exc.reason}.") from exc
    except csv.Error as exc:
        raise DatasetError(f"{path.name} is not a readable CSV file: {exc}.") from exc


def _normalize_label(raw: str | None, path: Path) -> str:
    label = (raw or "").strip().lower()
    if label not in LABELS:
        raise DatasetError(
            f"{path.name}: label must be 'fake' or 'real', got {raw!r}. "
            "Numeric labels are rejected on purpose: 0/1 conventions differ between datasets."
        )
    return label


def _class_source(path: Path, label: str, text_column: str) -> _Source:
    rows = _read_csv(path, [text_column])
    pairs = [(row[text_column] or "", label) for row in rows]
    return _Source(SourceFile(path.name, _sha256_of(path), len(pairs), label), pairs)


# -- public loaders ----------------------------------------------------------


def load_class_csvs(fake_csv: Path, real_csv: Path, text_column: str = "text") -> Dataset:
    """Load one CSV per class (``Fake.csv`` / ``True.csv``); the file determines the label."""
    return validate_sources(
        [
            _class_source(Path(fake_csv), "fake", text_column),
            _class_source(Path(real_csv), "real", text_column),
        ]
    )


def load_data_dir(data_dir: Path, text_column: str = "text") -> Dataset:
    """Load ``Fake.csv`` and ``True.csv`` from ``data_dir`` (default ``ml/data``)."""
    data_dir = Path(data_dir)
    missing = [n for n in (FAKE_FILENAME, REAL_FILENAME) if not (data_dir / n).is_file()]
    if missing:
        raise DatasetError(
            f"Expected {FAKE_FILENAME} and {REAL_FILENAME} in {data_dir}, but {missing} "
            "not found. See 'Getting the dataset' in ml/README.md."
        )
    return load_class_csvs(data_dir / FAKE_FILENAME, data_dir / REAL_FILENAME, text_column)


def load_labeled_csv(path: Path, text_column: str = "text", label_column: str = "label") -> Dataset:
    """Load one CSV with a text column and a label column containing 'fake' or 'real'."""
    path = Path(path)
    rows = _read_csv(path, [text_column, label_column])
    pairs = [(row[text_column] or "", _normalize_label(row[label_column], path)) for row in rows]
    return validate_sources(
        [_Source(SourceFile(path.name, _sha256_of(path), len(pairs), None), pairs)]
    )


# -- validation --------------------------------------------------------------


def _key(cleaned: str) -> bytes:
    return hashlib.blake2b(cleaned.casefold().encode("utf-8"), digest_size=16).digest()


def validate_sources(sources: list[_Source]) -> Dataset:
    """Drop unusable rows, count every decision, and raise if the result cannot be trained on."""
    classes = {label: ClassCounts() for label in LABELS}
    candidates: list[tuple[str, str, bytes, int, int]] = []  # raw, label, key, length, markers

    for source in sources:
        for raw, label in source.rows:
            counts = classes[label]
            counts.rows_read += 1
            if not raw.strip():
                counts.dropped_empty += 1
                continue
            cleaned = clean_text(raw)
            if len(cleaned) < MIN_TEXT_CHARS:
                counts.dropped_too_short += 1
                continue
            candidates.append((raw, label, _key(cleaned), len(cleaned), marker_bits(cleaned)))

    labels_by_key: dict[bytes, set[str]] = {}
    for _, label, key, _, _ in candidates:
        labels_by_key.setdefault(key, set()).add(label)

    texts: list[str] = []
    labels: list[str] = []
    lengths: dict[str, list[int]] = {label: [] for label in LABELS}
    dateline_hits: Counter[str] = Counter()
    marker_hits: dict[str, list[int]] = {label: [0] * len(MARKERS) for label in LABELS}
    kept_keys: list[str] = []
    seen: set[bytes] = set()
    for raw, label, key, length, bits in candidates:
        if len(labels_by_key[key]) > 1:
            classes[label].dropped_label_conflicts += 1
        elif key in seen:
            classes[label].dropped_duplicates += 1
        else:
            seen.add(key)
            texts.append(raw)
            labels.append(label)
            lengths[label].append(length)
            dateline_hits[label] += bool(DATELINE.match(raw))
            for i in range(len(MARKERS)):
                marker_hits[label][i] += (bits >> i) & 1
            kept_keys.append(f"{label}\t{key.hex()}")

    for label in LABELS:
        counts = classes[label]
        counts.kept = len(lengths[label])
        if counts.kept:
            counts.median_chars = float(statistics.median(lengths[label]))
            counts.dateline_share = dateline_hits[label] / counts.kept
            counts.marker_shares = {
                name: marker_hits[label][i] / counts.kept for i, name in enumerate(MARKERS)
            }
        if counts.kept < MIN_PER_CLASS:
            raise DatasetError(
                f"Need at least {MIN_PER_CLASS} usable '{label}' articles after cleaning; "
                f"got {counts.kept} (read {counts.rows_read}, dropped empty "
                f"{counts.dropped_empty}, too short {counts.dropped_too_short}, duplicates "
                f"{counts.dropped_duplicates}, cross-class conflicts "
                f"{counts.dropped_label_conflicts})."
            )

    fingerprint = hashlib.sha256("\n".join(sorted(kept_keys)).encode("ascii")).hexdigest()
    report = DatasetReport(
        sources=[s.file for s in sources],
        classes=classes,
        total_kept=len(texts),
        fingerprint=fingerprint,
        warnings=_warnings(classes),
    )
    return Dataset(texts=texts, labels=labels, report=report)


def _warnings(classes: dict[str, ClassCounts]) -> list[str]:
    warnings: list[str] = []
    fake, real = classes["fake"], classes["real"]
    total = fake.kept + real.kept
    ratio = max(fake.kept, real.kept) / min(fake.kept, real.kept)
    if ratio > IMBALANCE_WARN_RATIO:
        warnings.append(
            f"Classes are imbalanced ({fake.kept:,} fake vs {real.kept:,} real); read per-class "
            "precision/recall rather than accuracy."
        )
    if total < SMALL_DATASET_ROWS:
        warnings.append(
            f"Only {total:,} usable articles; validation/test splits will be small and the "
            "reported metrics noisy."
        )
    if fake.dateline_share is not None and real.dateline_share is not None:
        gap = abs(fake.dateline_share - real.dateline_share)
        if gap > MARKER_GAP_WARN:
            warnings.append(
                f"{real.dateline_share:.0%} of real vs {fake.dateline_share:.0%} of fake articles "
                "start with a wire-service dateline such as '(Reuters)'. Preprocessing strips this "
                "marker, but check for other publisher-specific artifacts before trusting metrics."
            )
    if fake.marker_shares and real.marker_shares:
        for name in MARKERS:
            f, r = fake.marker_shares[name], real.marker_shares[name]
            if abs(f - r) >= GAP_WARN:
                warnings.append(
                    f"After cleaning, {name} appears in {r:.0%} of real vs {f:.0%} of fake "
                    "articles. A model can use this to recognize the source rather than the "
                    "content; treat high scores with suspicion."
                )
    return warnings
