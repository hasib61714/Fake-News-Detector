from pathlib import Path

import pytest

from fakenews_ml.validate import main


def test_validate_prints_the_report_for_a_data_directory(
    data_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--data-dir", str(data_dir)]) == 0
    out = capsys.readouterr().out
    assert "Dataset validation report" in out and "Fake.csv" in out and "True.csv" in out


def test_validate_fails_with_exit_code_1_on_missing_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--data-dir", str(tmp_path)]) == 1
    assert "Expected Fake.csv and True.csv" in capsys.readouterr().err


def test_validate_accepts_explicit_paths(data_dir: Path) -> None:
    assert (
        main(["--fake-csv", str(data_dir / "Fake.csv"), "--real-csv", str(data_dir / "True.csv")])
        == 0
    )
