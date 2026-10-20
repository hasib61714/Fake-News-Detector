from pathlib import Path

import pytest

from fakenews_ml.data import (
    MIN_PER_CLASS,
    MIN_TEXT_CHARS,
    DatasetError,
    load_class_csvs,
    load_data_dir,
    load_labeled_csv,
)
from fakenews_ml.leakage import MARKERS
from tests.conftest import make_rows, write_csv


def long(text: str) -> str:
    return (text + " ") * 10


def test_loads_fake_and_true_csv_from_a_directory(data_dir: Path) -> None:
    dataset = load_data_dir(data_dir)

    assert dataset.labels.count("fake") == dataset.labels.count("real") == 80
    names = {s.name: s.label for s in dataset.report.sources}
    assert names == {"Fake.csv": "fake", "True.csv": "real"}
    assert all(len(s.sha256) == 64 for s in dataset.report.sources)
    assert dataset.report.classes["fake"].kept == 80


def test_label_comes_from_the_file_not_the_row(tmp_path: Path) -> None:
    fake = write_csv(tmp_path / "Fake.csv", ["text"], [[long(f"fake item {i}")] for i in range(60)])
    real = write_csv(tmp_path / "True.csv", ["text"], [[long(f"real item {i}")] for i in range(60)])
    dataset = load_class_csvs(fake, real)
    assert {t.split()[0]: lab for t, lab in zip(dataset.texts, dataset.labels, strict=True)} == {
        "fake": "fake",
        "real": "real",
    }


def test_missing_files_give_an_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match=r"Expected Fake.csv and True.csv.*README"):
        load_data_dir(tmp_path)


def test_one_missing_file(data_dir: Path) -> None:
    (data_dir / "True.csv").unlink()
    with pytest.raises(DatasetError, match="True.csv"):
        load_data_dir(data_dir)


def test_missing_text_column(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "Fake.csv", ["title", "body"], [["a", "b"]])
    ok = write_csv(tmp_path / "True.csv", ["text"], [["x"]])
    with pytest.raises(DatasetError, match=r"missing column\(s\) \['text'\]"):
        load_class_csvs(path, ok)


def test_non_utf8_file(tmp_path: Path) -> None:
    bad = tmp_path / "Fake.csv"
    bad.write_bytes(b"text\n\xff\xfe\xfa broken\n")
    ok = write_csv(tmp_path / "True.csv", ["text"], [["x"]])
    with pytest.raises(DatasetError, match="not valid UTF-8"):
        load_class_csvs(bad, ok)


def test_utf8_bom_is_accepted(tmp_path: Path, rows: list[tuple[str, str]]) -> None:
    fake = tmp_path / "Fake.csv"
    fake.write_bytes(
        b"\xef\xbb\xbftext\n" + "\n".join(f'"{t}"' for t, y in rows if y == "fake").encode()
    )
    real = write_csv(tmp_path / "True.csv", ["text"], [[t] for t, y in rows if y == "real"])
    assert load_class_csvs(fake, real).report.classes["fake"].kept == 80


def test_multiline_quoted_fields_are_read(tmp_path: Path) -> None:
    body = "first line of the article\nsecond line of the article " * 3
    fake = write_csv(tmp_path / "Fake.csv", ["text"], [[f"{body} {i}"] for i in range(60)])
    real = write_csv(tmp_path / "True.csv", ["text"], [[f"{body} real {i}"] for i in range(60)])
    assert "\n" in load_class_csvs(fake, real).texts[0]


def test_empty_short_and_duplicate_rows_are_dropped_and_counted(tmp_path: Path) -> None:
    unique = [[long(f"fake article number {i}")] for i in range(60)]
    junk = [[""], ["   "], ["too short"], [unique[0][0]], [unique[0][0].upper()]]
    fake = write_csv(tmp_path / "Fake.csv", ["text"], unique + junk)
    real = write_csv(
        tmp_path / "True.csv", ["text"], [[long(f"real article {i}")] for i in range(60)]
    )

    counts = load_class_csvs(fake, real).report.classes["fake"]

    assert counts.rows_read == 65
    assert counts.dropped_empty == 2
    assert counts.dropped_too_short == 1
    assert counts.dropped_duplicates == 2  # exact copy and a case-only variant
    assert counts.kept == 60


def test_html_and_dateline_variants_count_as_duplicates(tmp_path: Path) -> None:
    base = "the committee published its quarterly budget report to parliament today and "
    real_rows = [[f"{base}{i} " * 2] for i in range(60)]
    real_rows += [[f"WASHINGTON (Reuters) - {real_rows[0][0]}"], [f"<p>{real_rows[1][0]}</p>"]]
    real = write_csv(tmp_path / "True.csv", ["text"], real_rows)
    fake = write_csv(
        tmp_path / "Fake.csv", ["text"], [[long(f"hoax number {i}")] for i in range(60)]
    )

    assert load_class_csvs(fake, real).report.classes["real"].dropped_duplicates == 2


def test_texts_present_in_both_classes_are_removed_from_both(tmp_path: Path) -> None:
    shared = long("this exact article appears in both files")
    fake = write_csv(
        tmp_path / "Fake.csv", ["text"], [[shared]] + [[long(f"fake {i}")] for i in range(60)]
    )
    real = write_csv(
        tmp_path / "True.csv", ["text"], [[shared]] + [[long(f"real {i}")] for i in range(60)]
    )

    dataset = load_class_csvs(fake, real)

    assert shared not in dataset.texts
    assert dataset.report.classes["fake"].dropped_label_conflicts == 1
    assert dataset.report.classes["real"].dropped_label_conflicts == 1


def test_too_few_usable_articles_is_fatal(tmp_path: Path) -> None:
    fake = write_csv(
        tmp_path / "Fake.csv", ["text"], [[long(f"fake {i}")] for i in range(MIN_PER_CLASS - 1)]
    )
    real = write_csv(tmp_path / "True.csv", ["text"], [[long(f"real {i}")] for i in range(60)])
    with pytest.raises(DatasetError, match=rf"at least {MIN_PER_CLASS} usable 'fake'"):
        load_class_csvs(fake, real)


def test_text_shorter_than_the_api_minimum_is_not_trained_on(tmp_path: Path) -> None:
    short = "x" * (MIN_TEXT_CHARS - 1)
    fake = write_csv(
        tmp_path / "Fake.csv", ["text"], [[f"{short}{i}"[: MIN_TEXT_CHARS - 1]] for i in range(80)]
    )
    real = write_csv(tmp_path / "True.csv", ["text"], [[long(f"real {i}")] for i in range(60)])
    with pytest.raises(DatasetError, match="'fake'"):
        load_class_csvs(fake, real)


def test_warnings_for_imbalance_small_size_and_wire_marker(tmp_path: Path) -> None:
    real_rows = [[f"WASHINGTON (Reuters) - {long(f'real {i}')}"] for i in range(60)]
    fake_rows = [[long(f"fake {i}")] for i in range(300)]
    report = load_class_csvs(
        write_csv(tmp_path / "Fake.csv", ["text"], fake_rows),
        write_csv(tmp_path / "True.csv", ["text"], real_rows),
    ).report

    joined = " ".join(report.warnings)
    assert "imbalanced" in joined
    assert "Only 360 usable" in joined
    assert "wire-service dateline" in joined


def test_fingerprint_is_stable_and_order_independent(
    tmp_path: Path, rows: list[tuple[str, str]]
) -> None:
    def build(name: str, reverse: bool) -> str:
        directory = tmp_path / name
        directory.mkdir()
        for file, label in (("Fake.csv", "fake"), ("True.csv", "real")):
            subset = [[t] for t, y in rows if y == label]
            write_csv(directory / file, ["text"], subset[::-1] if reverse else subset)
        return load_data_dir(directory).report.fingerprint

    assert build("a", False) == build("b", False) == build("c", True)


def test_fingerprint_changes_when_the_data_changes(tmp_path: Path) -> None:
    def build(name: str, extra: str) -> str:
        directory = tmp_path / name
        directory.mkdir()
        rows = make_rows(60)
        write_csv(directory / "Fake.csv", ["text"], [[t + extra] for t, y in rows if y == "fake"])
        write_csv(directory / "True.csv", ["text"], [[t] for t, y in rows if y == "real"])
        return load_data_dir(directory).report.fingerprint

    assert build("a", "") != build("b", " changed")


def test_labeled_csv_form(tmp_path: Path, rows: list[tuple[str, str]]) -> None:
    path = write_csv(tmp_path / "d.csv", ["text", "label"], [[t, y.upper()] for t, y in rows])
    assert load_labeled_csv(path).labels.count("real") == 80


def test_numeric_labels_are_rejected(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "d.csv", ["text", "label"], [["some text", "1"]])
    with pytest.raises(DatasetError, match="fake' or 'real"):
        load_labeled_csv(path)


def test_report_is_json_serializable_and_printable(dataset) -> None:  # type: ignore[no-untyped-def]
    import json

    json.dumps(dataset.report.to_dict(), allow_nan=False)
    assert "Dataset validation report" in dataset.report.format()


# -- residual source markers (measured after cleaning) -------------------------------------------

STYLE = "wire style 'said on <weekday>'"
IMAGE = "'featured image' / 'image via'"


def _class_files(tmp_path: Path, fake_suffix: str, real_suffix: str) -> tuple[Path, Path]:
    fake = [[f"{long(f'fake report number {i}')} {fake_suffix}".strip()] for i in range(80)]
    real = [[f"{long(f'real report number {i}')} {real_suffix}".strip()] for i in range(80)]
    return (
        write_csv(tmp_path / "Fake.csv", ["text"], fake),
        write_csv(tmp_path / "True.csv", ["text"], real),
    )


def test_markers_are_measured_after_cleaning_so_removed_boilerplate_does_not_count(
    tmp_path: Path,
) -> None:
    # The image credit is removed by preprocessing, so it must NOT show up as a residual marker;
    # "said on Tuesday" is ordinary text that survives cleaning, so it must.
    fake, real = _class_files(
        tmp_path, "Featured image via Mark Wilson/Getty Images", "He said on Tuesday."
    )

    report = load_class_csvs(fake, real).report

    assert report.classes["fake"].marker_shares[IMAGE] == 0.0
    assert report.classes["real"].marker_shares[STYLE] == 1.0
    assert report.classes["fake"].marker_shares[STYLE] == 0.0
    assert set(report.classes["real"].marker_shares) == set(MARKERS)


def test_a_marker_concentrated_in_one_class_triggers_a_warning(tmp_path: Path) -> None:
    fake, real = _class_files(tmp_path, "", "He said on Tuesday.")
    report = load_class_csvs(fake, real).report

    matching = [w for w in report.warnings if STYLE in w]
    assert len(matching) == 1
    assert "100% of real vs 0% of fake" in matching[0]
    assert "<-- differs by class" in report.format()


def test_balanced_markers_trigger_no_marker_warning(tmp_path: Path) -> None:
    fake, real = _class_files(tmp_path, "She said on Monday.", "He said on Tuesday.")
    report = load_class_csvs(fake, real).report

    assert [w for w in report.warnings if "After cleaning" in w] == []
    assert "<-- differs by class" not in report.format()


def test_the_validation_report_lists_every_monitored_marker(dataset) -> None:  # type: ignore[no-untyped-def]
    text = dataset.report.format()
    assert "Residual source markers after cleaning" in text
    assert all(name in text for name in MARKERS)
