"""Pytest configuration and shared fixtures for DeepBioP tests."""

import random
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
def medium_fastq(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate a 10,000-record FASTQ file (generated once per session, not committed)."""
    path = tmp_path_factory.mktemp("data") / "10000_records.fastq"
    rng = random.Random(0)
    with path.open("w") as f:
        for i in range(10_000):
            length = rng.randint(50, 150)
            seq = "".join(rng.choices("ACGT", k=length))
            qual = "".join(chr(33 + rng.randint(2, 40)) for _ in range(length))
            f.write(f"@read_{i}\n{seq}\n+\n{qual}\n")
    return path


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
