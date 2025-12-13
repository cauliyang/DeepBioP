"""Pytest configuration and shared fixtures for DeepBioP tests."""

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def small_fastq(test_data_dir: Path) -> Path:
    """Return path to small test FASTQ file."""
    return test_data_dir / "test.fastq"


@pytest.fixture(scope="session")
def labeled_fastq(test_data_dir: Path) -> Path:
    """Return path to labeled FASTQ file for supervised learning tests."""
    return test_data_dir / "labeled_small.fastq"


@pytest.fixture(scope="session")
def labels_csv(test_data_dir: Path) -> Path:
    """Return path to CSV labels file."""
    return test_data_dir / "labels.csv"


@pytest.fixture(scope="session")
def medium_fastq(test_data_dir: Path) -> Path:
    """Return path to medium-sized test FASTQ file."""
    return test_data_dir / "10000_records.fastq"


@pytest.fixture(scope="session")
def gzipped_fastq(test_data_dir: Path) -> Path:
    """Return path to gzipped FASTQ file."""
    return test_data_dir / "test.fastq.gz"


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> "Iterator[Path]":
    """Create temporary directory for cache tests."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir
    # Cleanup is automatic with tmp_path


@pytest.fixture
def sample_record() -> dict:
    """Return a sample FASTQ record for testing."""
    return {
        "id": b"@read_001",
        "sequence": b"ACGTACGTACGTACGT",
        "quality": b"IIIIIIIIIIIIIIII",
        "metadata": {},
    }


@pytest.fixture
def sample_records() -> list[dict]:
    """Return a list of sample FASTQ records for batch testing."""
    return [
        {
            "id": b"@read_001",
            "sequence": b"ACGTACGT",
            "quality": b"IIIIIIII",
            "metadata": {},
        },
        {
            "id": b"@read_002",
            "sequence": b"TGCATGCA",
            "quality": b"JJJJJJJJ",
            "metadata": {},
        },
        {
            "id": b"@read_003",
            "sequence": b"GGCCGGCC",
            "quality": b"KKKKKKKK",
            "metadata": {},
        },
    ]
