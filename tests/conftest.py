# tests/conftest.py
import pytest
from pathlib import Path

@pytest.fixture
def fixtures_dir():
    return Path(__file__).parent / "fixtures"

@pytest.fixture
def valid_csv(fixtures_dir):
    return fixtures_dir / "valid.csv"
