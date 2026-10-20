"""Duplicate and contamination check between an external dataset and the training data.

``python -m fakenews_ml.crosscheck --external path/to/external.csv --text-column text``

Run this BEFORE evaluating on an external dataset. It scores no model and writes nothing. It
reports, for the external articles:

* empty texts and exact duplicates inside the external file;
* exact overlap with the reference (training) articles, after the pipeline's own cleaning and
  case-folding, so a copy with a different dateline or HTML wrapper still counts;
* near-duplicates: cosine similarity of TF-IDF vectors at or above a threshold (default 0.90),
  which catches reprints with small edits and different headlines;
* shared publishers, when a domain or URL column is given (a shared publisher means the external
  set is not independent even when no single article repeats).

An external set with substantial overlap cannot support a claim of external validity.
"""

import argparse
import csv
import hashlib
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from fakenews_ml.data import DatasetError, load_data_dir
from fakenews_ml.preprocessing import clean_text

NEAR_DUPLICATE_THRESHOLD = 0.90
MIN_REFERENCE_CHARS = 50
csv.field_size_limit(2**31 - 1)


@dataclass
class ContaminationReport:
    n_external: int
    n_empty_or_short: int
    n_duplicates_within_external: int
    n_exact_overlap_with_reference: int
    n_near_duplicate_of_reference: int  # includes exact overlaps
    near_duplicate_threshold: float
    max_similarity_percentiles: dict[str, float]
    shared_domains: list[str] = field(default_factory=list)
    domains_checked: bool = False

    @property
    def n_contaminated(self) -> int:
        return self.n_near_duplicate_of_reference

    @property
    def n_usable(self) -> int:
        """External articles left after dropping empties, internal duplicates and contamination."""
        return max(
            self.n_external
            - self.n_empty_or_short
            - self.n_duplicates_within_external
            - self.n_contaminated,
            0,
        )

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "n_usable": self.n_usable}

    def _pct(self, n: int) -> str:
        return f"{n / self.n_external:.1%}" if self.n_external else "n/a"

    def format(self) -> str:
        pct = self._pct
        lines = [
            "Cross-dataset contamination check (no model scored)",
            "---------------------------------------------------",
            f"  external articles                          {self.n_external:>8,}",
            f"  empty or under {MIN_REFERENCE_CHARS} chars"
            f"                   {self.n_empty_or_short:>8,}  ({pct(self.n_empty_or_short)})",
            f"  duplicates inside the external file        {self.n_duplicates_within_external:>8,}"
            f"  ({pct(self.n_duplicates_within_external)})",
            f"  exact overlap with the reference data     "
            f" {self.n_exact_overlap_with_reference:>8,}"
            f"  ({pct(self.n_exact_overlap_with_reference)})",
            f"  near-duplicates (cosine >= {self.near_duplicate_threshold:.2f})"
            f"           {self.n_near_duplicate_of_reference:>8,}"
            f"  ({pct(self.n_near_duplicate_of_reference)})",
            "  max similarity to any reference article, percentiles: "
            + ", ".join(f"{k} {v:.2f}" for k, v in self.max_similarity_percentiles.items()),
            f"  usable after removing all of the above     {self.n_usable:>8,}",
        ]
        if self.domains_checked:
            lines.append(
                f"  publishers shared with the reference data  {len(self.shared_domains):>8,}"
                + (f"  e.g. {', '.join(self.shared_domains[:8])}" if self.shared_domains else "")
            )
            lines.append(
                "  NOTE: only the reference publishers named on the command line can be matched; "
                "ISOT's fake-source sites are not recorded per article, so also review the "
                "external publishers by hand."
            )
        else:
            lines.append("  publishers: not checked (no domain/URL column given)")
        return "\n".join(lines)


def _key(text: str) -> bytes:
    return hashlib.blake2b(clean_text(text).casefold().encode("utf-8"), digest_size=16).digest()


def domain_of(value: str) -> str:
    """Registrable-looking host from a URL or bare domain, lower-cased, without 'www.'."""
    value = value.strip().lower()
    if not value:
        return ""
    host = urlparse(value if "//" in value else f"//{value}").netloc or value
    host = host.split("@")[-1].split(":")[0]
    return host.removeprefix("www.")


def check_contamination(
    external_texts: Sequence[str],
    reference_texts: Sequence[str],
    *,
    external_domains: Sequence[str] | None = None,
    reference_domains: Sequence[str] | None = None,
    threshold: float = NEAR_DUPLICATE_THRESHOLD,
    chunk: int = 500,
) -> ContaminationReport:
    cleaned_ext = [clean_text(t or "") for t in external_texts]
    usable = [i for i, t in enumerate(cleaned_ext) if len(t) >= MIN_REFERENCE_CHARS]
    n_short = len(cleaned_ext) - len(usable)

    seen: set[bytes] = set()
    first_copies: list[int] = []
    n_dupes = 0
    for i in usable:
        key = _key(external_texts[i])
        if key in seen:
            n_dupes += 1
        else:
            seen.add(key)
            first_copies.append(i)

    ref_keys = {_key(t) for t in reference_texts}
    exact = sum(_key(external_texts[i]) in ref_keys for i in first_copies)

    # Near duplicates: TF-IDF fitted on the reference only, external rows compared to it.
    vectorizer = TfidfVectorizer(
        stop_words="english", max_features=100_000, sublinear_tf=True, min_df=1
    )
    ref_matrix = vectorizer.fit_transform([clean_text(t) for t in reference_texts])
    max_sim = np.zeros(len(first_copies))
    ext_matrix = vectorizer.transform([cleaned_ext[i] for i in first_copies])
    for start in range(0, ext_matrix.shape[0], chunk):
        block = ext_matrix[start : start + chunk] @ ref_matrix.T
        max_sim[start : start + chunk] = block.max(axis=1).toarray().ravel()
    near = int((max_sim >= threshold).sum())
    percentiles = (
        {f"p{p}": float(np.percentile(max_sim, p)) for p in (50, 90, 99)} if len(max_sim) else {}
    )

    shared: list[str] = []
    checked = external_domains is not None and reference_domains is not None
    if checked:
        assert external_domains is not None and reference_domains is not None
        ext_set = {domain_of(d) for d in external_domains} - {""}
        ref_set = {domain_of(d) for d in reference_domains} - {""}
        shared = sorted(ext_set & ref_set)

    return ContaminationReport(
        n_external=len(external_texts),
        n_empty_or_short=n_short,
        n_duplicates_within_external=n_dupes,
        n_exact_overlap_with_reference=exact,
        n_near_duplicate_of_reference=max(near, exact),
        near_duplicate_threshold=threshold,
        max_similarity_percentiles=percentiles,
        shared_domains=shared,
        domains_checked=checked,
    )


def _read_column(path: Path, column: str) -> list[str]:
    if not path.is_file():
        raise DatasetError(f"File not found: {path}")
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if column not in (reader.fieldnames or []):
                raise DatasetError(f"{path.name} has no '{column}' column ({reader.fieldnames}).")
            return [row.get(column) or "" for row in reader]
    except UnicodeDecodeError as exc:
        raise DatasetError(f"{path.name} is not valid UTF-8: {exc.reason}.") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--external", type=Path, required=True, help="external CSV file")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--domain-column", help="column with each article's domain or URL")
    parser.add_argument(
        "--reference-domains",
        default="reuters.com",
        help="comma-separated publisher domains of the reference data. ISOT's real articles are "
        "from Reuters.com; its fake-source domains are NOT recorded in the files, so overlap with "
        "those has to be checked by hand",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data"
    )
    parser.add_argument("--threshold", type=float, default=NEAR_DUPLICATE_THRESHOLD)
    args = parser.parse_args(argv)
    try:
        reference = load_data_dir(args.data_dir)
        external_texts = _read_column(args.external, args.text_column)
        external_domains = (
            [domain_of(d) for d in _read_column(args.external, args.domain_column)]
            if args.domain_column
            else None
        )
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    report = check_contamination(
        external_texts,
        reference.texts,
        external_domains=external_domains,
        reference_domains=[d for d in args.reference_domains.split(",") if d.strip()],
        threshold=args.threshold,
    )
    print(report.format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
