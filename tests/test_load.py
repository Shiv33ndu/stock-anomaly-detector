import pytest 
from pathlib import Path

from src.data.load import load_ticker_csv


# tests valid csv loads correctly 
def test_load_valid_csv(valid_csv):
    df = load_ticker_csv(valid_csv)

    assert not df.empty
    assert "date" in df.columns
    assert df["date"].dtype.kind == "M"  # datetime


# tests missing file 
def test_load_missing_file():
    with pytest.raises(FileNotFoundError):
        load_ticker_csv(Path("does_not_exist.csv"))

# Test: missing required column
def test_load_missing_column(fixtures_dir):
    path = fixtures_dir / "missing_column.csv"

    with pytest.raises(ValueError):
        load_ticker_csv(path)


# Test: unparseable date
def test_load_bad_date(fixtures_dir):
    path = fixtures_dir / "bad_date.csv"

    with pytest.raises(ValueError):
        load_ticker_csv(path)


