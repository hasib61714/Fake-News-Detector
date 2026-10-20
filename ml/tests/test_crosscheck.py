import csv
from pathlib import Path

import pytest

from fakenews_ml.crosscheck import check_contamination, domain_of, main
from tests.conftest import make_rows, write_csv

REFERENCE = [t for t, _ in make_rows(40, seed=1)]


def distinct(prefix: str, n: int) -> list[str]:
    """Articles that share no vocabulary with the reference fixtures."""
    return [
        f"{prefix} alpha{i} beta{i} gamma{i} delta{i} epsilon{i} zeta{i} eta{i} theta{i} iota{i} "
        f"kappa{i} lambda{i} omega{i} sigma{i} tau{i} upsilon{i}"
        for i in range(n)
    ]


def test_clean_external_data_shows_no_contamination() -> None:
    report = check_contamination(distinct("novel", 30), REFERENCE)
    assert (report.n_exact_overlap_with_reference, report.n_near_duplicate_of_reference) == (0, 0)
    assert report.n_duplicates_within_external == 0
    assert report.n_usable == 30
    assert report.max_similarity_percentiles["p99"] < 0.5


def test_exact_copies_are_found_even_with_wrappers_and_datelines() -> None:
    external = [
        REFERENCE[0],
        f"<p>WASHINGTON (Reuters) - {REFERENCE[1]}</p>",
        REFERENCE[2].upper(),  # case-folded
        *distinct("novel", 7),
    ]
    report = check_contamination(external, REFERENCE)
    assert report.n_exact_overlap_with_reference == 3
    assert report.n_usable == 7


def test_near_duplicates_are_caught_but_unrelated_text_is_not() -> None:
    edited = REFERENCE[5] + " One extra closing sentence was added by a different outlet."
    reworded = " ".join(REFERENCE[6].split()[:-3]) + " and some other words entirely"
    report = check_contamination(
        [edited, reworded, *distinct("novel", 5)], REFERENCE, threshold=0.8
    )
    assert report.n_exact_overlap_with_reference == 0
    assert report.n_near_duplicate_of_reference == 2
    assert report.n_usable == 5


def test_duplicates_and_short_texts_inside_the_external_file_are_counted() -> None:
    a, b = distinct("novel", 2)
    report = check_contamination([a, a, a.upper(), b, "", "tiny"], REFERENCE)
    assert report.n_duplicates_within_external == 2
    assert report.n_empty_or_short == 2
    assert report.n_usable == 2


def test_shared_publishers_are_reported_only_when_domains_are_given() -> None:
    ext = distinct("novel", 3)
    without = check_contamination(ext, REFERENCE)
    assert without.domains_checked is False

    shared = check_contamination(
        ext,
        REFERENCE,
        external_domains=["https://www.Reuters.com/a", "example.org", "http://blog.test:8080/x"],
        reference_domains=["reuters.com"],
    )
    assert shared.domains_checked is True
    assert shared.shared_domains == ["reuters.com"]
    assert "reuters.com" in shared.format()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("https://www.Example.com/path?q=1", "example.com"),
        ("example.com", "example.com"),
        ("http://user@news.site:8080/x", "news.site"),
        ("", ""),
    ],
)
def test_domain_of(value: str, expected: str) -> None:
    assert domain_of(value) == expected


def test_cli_reports_without_scoring_any_model(
    data_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    external = write_csv(
        tmp_path / "ext.csv",
        ["text", "domain"],
        [[t, "news.example"] for t in distinct("novel", 10)]
        + [[make_rows(80)[0][0], "reuters.com"]],  # an article copied from the reference data
    )
    assert (
        main(
            ["--external", str(external), "--data-dir", str(data_dir), "--domain-column", "domain"]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "no model scored" in out
    assert "exact overlap with the reference data" in out
    assert "reuters.com" in out


def test_cli_errors_are_clean(
    data_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--external", str(tmp_path / "missing.csv"), "--data-dir", str(data_dir)]) == 1
    bad = tmp_path / "bad.csv"
    with bad.open("w", newline="") as f:
        csv.writer(f).writerow(["body"])
    assert main(["--external", str(bad), "--data-dir", str(data_dir)]) == 1
    assert capsys.readouterr().err.count("error:") == 2
