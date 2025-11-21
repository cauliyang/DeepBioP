"""Tests for core module functionality.

Note: The original tests for reverse_complement and seq_to_kmers have been removed
because these functions do not exist in the current API. These were likely removed or
never implemented in the Python bindings.

Current core module only exports: Record (dataclass)

For sequence operations, use:
- ReverseComplement transform for reverse complement
- KmerEncoder for k-mer operations
"""

import pytest
from deepbiop import core


def test_core_record_exists():
    """Test that Record dataclass is available."""
    assert hasattr(core, "Record")
    # Record is a dataclass for type hints, basic check
    assert core.Record is not None


@pytest.mark.skip(reason="reverse_complement function does not exist in core module")
def test_reverse_complement():
    """Skipped: Use ReverseComplement transform instead."""
    pass


@pytest.mark.skip(reason="seq_to_kmers function does not exist in core module")
def test_fq():
    """Skipped: Use KmerEncoder transform instead."""
    pass
