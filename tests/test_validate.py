import pytest 
from pathlib import Path

from src.data.load import load_ticker_csv
from src.data.validate import validate_ohlcv



# Test: valid data passes
def test_validate_valid_df(valid_csv):
    df = load_ticker_csv(valid_csv)
    validated = validate_ohlcv(df)

    assert validated is not None
    assert len(validated) == len(df)


# Test: duplicate dates
def test_validate_duplicate_dates(fixtures_dir):
    path = fixtures_dir / "duplicate_dates.csv"
    df = load_ticker_csv(path)

    with pytest.raises(ValueError):
        validate_ohlcv(df)



# Test: negative volume
def test_validate_negative_volume(fixtures_dir):
    path = fixtures_dir / "negative_volume.csv"
    df = load_ticker_csv(path)

    with pytest.raises(ValueError):
        validate_ohlcv(df)


# Test: invalid OHLC
def test_validate_bad_ohlc(fixtures_dir):
    path = fixtures_dir / "bad_ohlc.csv"
    df = load_ticker_csv(path)

    with pytest.raises(ValueError):
        validate_ohlcv(df)


# Test: unsorted dates auto-fix
def test_validate_unsorted_dates(fixtures_dir):
    path = fixtures_dir / "unsorted_dates.csv"
    df = load_ticker_csv(path)

    validated = validate_ohlcv(df)
    assert validated["date"].is_monotonic_increasing
