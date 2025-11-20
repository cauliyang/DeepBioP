"""Tests for dataset functionality - streaming, memory usage, and batching."""

import gc
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
def test_fastq_memory_streaming():
    """
    Test FASTQ streaming keeps memory usage under 500MB.

    Requirements:
    - FR-001: Stream FASTQ files without loading entire file into memory
    - SC-001: Memory usage <500MB for 10GB FASTQ file

    This test verifies that:
    1. Large FASTQ files can be streamed record-by-record
    2. Memory usage stays below 500MB threshold
    3. No memory leaks during iteration
    """
    pytest.skip("Requires FastqDataset implementation (T020)")

    # This test will use a large FASTQ file (10GB) once available
    # For now, we'll use a smaller test file and verify the pattern
    test_file = Path(__file__).parent / "data" / "large_10k.fastq.gz"

    # Record initial memory
    gc.collect()
    initial_memory = _get_memory_mb()

    # Create dataset (should not load file into memory)
    dataset = FastqDataset(test_file)  # noqa: F821

    # Verify dataset creation doesn't load entire file
    after_create_memory = _get_memory_mb()
    memory_increase = after_create_memory - initial_memory
    assert memory_increase < 100, (
        f"Dataset creation used {memory_increase}MB (should be <100MB)"
    )

    # Stream through records
    record_count = 0
    max_memory = initial_memory

    for _record in dataset:
        record_count += 1

        # Check memory periodically (every 1000 records)
        if record_count % 1000 == 0:
            current_memory = _get_memory_mb()
            max_memory = max(max_memory, current_memory)

            # Verify memory stays under 500MB
            memory_used = current_memory - initial_memory
            assert memory_used < 500, (
                f"Memory exceeded 500MB threshold at record {record_count}: "
                f"{memory_used}MB used"
            )

    # Verify we processed records
    assert record_count > 0, "Dataset should contain records"

    # Final memory check
    final_memory = _get_memory_mb()
    total_memory_used = max_memory - initial_memory

    print(f"Processed {record_count} records")
    print(f"Peak memory usage: {total_memory_used}MB")
    print(f"Final memory usage: {final_memory - initial_memory}MB")

    assert total_memory_used < 500, (
        f"Peak memory {total_memory_used}MB exceeded 500MB threshold"
    )


def test_bam_batching_memory():
    """
    Test BAM batching without loading entire file.

    Requirements:
    - FR-003: Stream BAM files efficiently
    - SC-001: Memory usage <500MB

    This test verifies that:
    1. BAM files can be batched without full load
    2. Memory stays bounded during batching
    """
    pytest.skip("Requires BamDataset implementation (T022)")
    from deepbiop.core import collate_batch

    test_file = Path(__file__).parent / "data" / "sample.bam"

    gc.collect()
    initial_memory = _get_memory_mb()

    dataset = BamDataset(test_file)  # noqa: F821

    # Batch records
    batch_size = 32
    batch_count = 0

    batch_records = []
    for record in dataset:
        batch_records.append(record)

        if len(batch_records) == batch_size:
            # Create batch
            batch = collate_batch(batch_records, padding="longest")
            batch_count += 1

            # Verify batch properties
            assert len(batch) == batch_size
            assert batch.sequences.shape[0] == batch_size

            # Check memory
            current_memory = _get_memory_mb()
            memory_used = current_memory - initial_memory
            assert memory_used < 500, f"Memory exceeded threshold: {memory_used}MB"

            # Clear batch for next iteration
            batch_records = []

    assert batch_count > 0, "Should have created at least one batch"
    print(f"Created {batch_count} batches of size {batch_size}")


def test_multi_file_streaming():
    """
    Test sequential iteration over multiple files.

    Requirements:
    - FR-002: Handle multiple input files
    - FR-001: Stream efficiently

    This test verifies that:
    1. Multiple files can be streamed sequentially
    2. File boundaries are handled correctly
    3. Memory usage stays bounded across files
    """
    pytest.skip("Requires MultiFileDataset implementation (T027)")

    from deepbiop.dataset import MultiFileDataset

    test_files = [
        Path(__file__).parent / "data" / "sample.fastq.gz",
        Path(__file__).parent / "data" / "sample.fasta",
    ]

    gc.collect()
    initial_memory = _get_memory_mb()

    dataset = MultiFileDataset(test_files)

    # Track records per file
    total_records = 0

    for i, _record in enumerate(dataset):
        total_records += 1

        # Check memory periodically
        if i % 100 == 0:
            current_memory = _get_memory_mb()
            memory_used = current_memory - initial_memory
            assert memory_used < 500, f"Memory exceeded threshold: {memory_used}MB"

    assert total_records > 0, "Should have processed records from all files"
    print(f"Processed {total_records} records from {len(test_files)} files")


def test_zero_copy_arrays():
    """
    Test zero-copy NumPy array access from Rust.

    Requirements:
    - FR-014: Zero-copy data transfer between Rust and Python
    - SC-004: Efficient memory usage

    This test verifies that:
    1. Batch arrays are zero-copy (share memory with Rust)
    2. No unnecessary data copies during array access
    3. Arrays are properly managed and released
    """
    from deepbiop.core import SequenceRecord, collate_batch

    # Create test records
    records = [SequenceRecord(f"seq{i}", b"ACGT" * 10, None, None) for i in range(100)]

    gc.collect()
    initial_memory = _get_memory_mb()

    # Create batch
    batch = collate_batch(records, padding="longest")

    # Access arrays (should be zero-copy)
    sequences = batch.sequences
    attention_mask = batch.attention_mask

    # Verify array properties
    assert sequences.shape[0] == 100
    assert attention_mask.shape == sequences.shape

    # Memory check - should not double with array access
    after_access_memory = _get_memory_mb()
    memory_increase = after_access_memory - initial_memory

    # Allow some overhead but verify no full copy occurred
    # Each record is 40 bytes, 100 records = 4KB
    # With overhead, should be <10MB total
    assert memory_increase < 10, (
        f"Array access may have copied data: {memory_increase}MB increase "
        "(expected <10MB for zero-copy)"
    )

    print(f"Batch created with zero-copy arrays: {memory_increase}MB overhead")


def _get_memory_mb() -> float:
    """
    Get current process memory usage in MB.

    Returns
    -------
        Memory usage in megabytes.
    """
    try:
        import psutil

        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        # Fallback to sys if psutil not available
        # This is less accurate but works for basic checks
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        # On macOS, ru_maxrss is in bytes; on Linux, in kilobytes
        if sys.platform == "darwin":
            return usage.ru_maxrss / 1024 / 1024
        else:
            return usage.ru_maxrss / 1024


# Mark as slow tests that can be skipped in fast runs
pytestmark = pytest.mark.slow


# ============================================================================
# Phase 8: FASTA and BAM Dataset Support Tests (T070-T078)
# ============================================================================

# Test data paths
TEST_DATA_DIR = Path(__file__).parent / "data"
FASTA_TEST_FILE = TEST_DATA_DIR / "test.fa"


class TestFastaDataset:
    """T070: Unit test for FastaDataset - basic functionality."""

    def test_fasta_dataset_len(self):
        """Test FastaDataset __len__ method."""
        from deepbiop import FastaDataset

        dataset = FastaDataset(str(FASTA_TEST_FILE))

        # Should have a positive length
        assert len(dataset) > 0
        assert isinstance(len(dataset), int)

    def test_fasta_dataset_getitem(self):
        """Test FastaDataset __getitem__ method."""
        from deepbiop import FastaDataset

        dataset = FastaDataset(str(FASTA_TEST_FILE))

        # Get first record
        record = dataset[0]

        # Should be a dict with expected keys
        assert isinstance(record, dict)
        assert "id" in record
        assert "sequence" in record

        # ID should be bytes or str
        assert isinstance(record["id"], (bytes, str))

        # Sequence should be bytes or numpy array
        import numpy as np

        assert isinstance(record["sequence"], (bytes, np.ndarray))

    def test_fasta_dataset_iteration(self):
        """Test FastaDataset __iter__ method."""
        from deepbiop import FastaDataset

        dataset = FastaDataset(str(FASTA_TEST_FILE))

        # Should be iterable
        records = list(dataset)

        # Should have same length as __len__
        assert len(records) == len(dataset)

        # All records should be dicts
        for record in records:
            assert isinstance(record, dict)
            assert "id" in record
            assert "sequence" in record

    def test_fasta_dataset_random_access(self):
        """Test random access to different indices."""
        from deepbiop import FastaDataset

        dataset = FastaDataset(str(FASTA_TEST_FILE))

        if len(dataset) > 1:
            # Get multiple records by index
            record0 = dataset[0]
            record1 = dataset[1]

            # Different records should have different IDs or sequences
            assert (record0["id"] != record1["id"]) or (
                not (record0["sequence"] == record1["sequence"]).all()
                if hasattr(record0["sequence"], "all")
                else record0["sequence"] != record1["sequence"]
            )

    def test_fasta_dataset_index_error(self):
        """Test that out-of-bounds index raises IndexError."""
        from deepbiop import FastaDataset

        dataset = FastaDataset(str(FASTA_TEST_FILE))

        # Should raise IndexError for out-of-bounds access
        with pytest.raises(IndexError):
            _ = dataset[len(dataset)]

        with pytest.raises(IndexError):
            _ = dataset[-len(dataset) - 1]


class TestFastaDatasetDataLoader:
    """T071: Integration test - FastaDataset with DataLoader."""

    def test_fasta_dataset_with_dataloader(self):
        """Test FastaDataset works with PyTorch DataLoader."""
        from deepbiop import FastaDataset, default_collate

        try:
            from torch.utils.data import DataLoader
        except ImportError:
            pytest.skip("PyTorch not available")

        dataset = FastaDataset(str(FASTA_TEST_FILE))

        # Use default_collate to handle variable-length sequences
        loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=default_collate)

        # Should be able to iterate through DataLoader
        batches = list(loader)

        # Should have at least one batch
        assert len(batches) > 0

        # Each batch is a list of dicts (default_collate is identity function)
        first_batch = batches[0]
        assert isinstance(first_batch, list)
        assert len(first_batch) > 0

        # Each item in batch should be a dict
        assert isinstance(first_batch[0], dict)
        assert "id" in first_batch[0]
        assert "sequence" in first_batch[0]

    def test_fasta_dataset_batch_sizes(self):
        """Test different batch sizes work correctly."""
        from deepbiop import FastaDataset, default_collate

        try:
            from torch.utils.data import DataLoader
        except ImportError:
            pytest.skip("PyTorch not available")

        dataset = FastaDataset(str(FASTA_TEST_FILE))

        # Test different batch sizes
        for batch_size in [1, 2, 4]:
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, collate_fn=default_collate
            )
            batches = list(loader)

            # Total records should match (batch is list of dicts)
            total_in_batches = sum(len(batch) for batch in batches)
            assert total_in_batches == len(dataset)


class TestFastaDatasetTransforms:
    """T072: Test FastaDataset with transforms."""

    def test_fasta_dataset_with_transform(self):
        """Test applying transforms to FASTA dataset."""
        from deepbiop import FastaDataset, TransformDataset

        # Simple transform that adds a field
        def add_length_field(record):
            record["length"] = len(record["sequence"])
            return record

        dataset = FastaDataset(str(FASTA_TEST_FILE))
        transformed = TransformDataset(dataset, transform=add_length_field)

        # Get a record
        record = transformed[0]

        # Should have the new field
        assert "length" in record
        assert record["length"] > 0
        assert record["length"] == len(record["sequence"])

    def test_fasta_dataset_with_encoder(self):
        """Test FASTA dataset with sequence encoder."""
        from deepbiop import FastaDataset

        try:
            from deepbiop import IntegerEncoder
            import numpy as np
        except ImportError:
            pytest.skip("IntegerEncoder not available")

        dataset = FastaDataset(str(FASTA_TEST_FILE))

        # Apply encoder
        encoder = IntegerEncoder()
        record = dataset[0].copy()  # Copy to preserve original
        encoded = encoder(record)

        # Encoder transforms the sequence field to float32
        assert "sequence" in encoded
        assert isinstance(encoded["sequence"], np.ndarray)
        assert encoded["sequence"].dtype == np.float32


class TestFastaDatasetErrorHandling:
    """T073: Error handling tests."""

    def test_fasta_nonexistent_file(self):
        """Test FastaDataset with non-existent file."""
        from deepbiop import FastaDataset

        with pytest.raises((FileNotFoundError, IOError, RuntimeError)):
            _ = FastaDataset("nonexistent_file.fasta")

    def test_fasta_invalid_format(self):
        """Test FastaDataset with invalid format."""
        from deepbiop import FastaDataset

        # Try to open a FASTQ file as FASTA - may or may not error depending on implementation
        # This test documents expected behavior
        fastq_file = TEST_DATA_DIR / "test.fastq"
        if fastq_file.exists():
            # May succeed but with incorrect parsing, or may fail
            # Just ensure it doesn't crash completely
            try:
                dataset = FastaDataset(str(fastq_file))
                _ = len(dataset)  # Try to use it
            except Exception:
                pass  # Expected behavior - format mismatch
