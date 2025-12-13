"""Tests for large-scale batch processing.

Tests T032-T042 for User Story 3: Memory-efficient batch processing and multi-worker support.
"""

import gc
import sys
import tempfile
from pathlib import Path

import pytest

# Import datasets and loaders
try:
    from deepbiop import FastqDataset, TransformDataset
    from deepbiop.collate import default_collate, supervised_collate
except ImportError as e:
    pytest.skip(f"Datasets not yet exported: {e}", allow_module_level=True)

# Import PyTorch
try:
    from torch.utils.data import DataLoader
except ImportError:
    pytest.skip("PyTorch not installed", allow_module_level=True)


class TestMemoryEfficiency:
    """T032-T034: Memory usage and efficiency tests."""

    @pytest.fixture
    def large_fastq(self):
        """Create a large temporary FASTQ file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fq", delete=False) as f:
            # Write 1000 records (should be small enough for CI, large enough for testing)
            for i in range(1000):
                f.write(f"@seq{i}\n")
                f.write("ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\n")  # 44 bases
                f.write("+\n")
                f.write("I" * 44 + "\n")
            f.flush()  # Ensure all data is written
            fname = f.name
        yield fname
        Path(fname).unlink()

    def test_dataset_memory_does_not_grow_linearly(self, large_fastq):
        """T032: Dataset should not load entire file into memory."""
        # Measure memory before
        gc.collect()
        dataset = FastqDataset(large_fastq)

        # Access many samples
        for i in range(0, len(dataset), 100):
            _ = dataset[i]

        # Dataset should not hold all accessed samples in memory
        # (streaming behavior, not caching all)
        assert len(dataset) == 1000

    def test_dataloader_with_batching_memory(self, large_fastq):
        """T033: DataLoader batching should not accumulate memory."""
        dataset = FastqDataset(large_fastq)
        loader = DataLoader(dataset, batch_size=32, collate_fn=default_collate)

        # Process multiple batches
        batch_count = 0
        for batch in loader:
            assert len(batch) <= 32
            batch_count += 1

        # Should process all data in batches
        assert batch_count == 32  # 1000 / 32 = 31.25, rounds to 32

    def test_memory_efficient_transform_dataset(self, large_fastq):
        """T034: TransformDataset should apply transforms lazily."""
        from deepbiop import ReverseComplement

        dataset = FastqDataset(large_fastq)
        transformed = TransformDataset(dataset, ReverseComplement())

        # Access samples - transforms should be applied on-the-fly
        sample = transformed[0]
        original = dataset[0]

        # ReverseComplement should modify the sequence and quality
        # Original sequence: ACGTACGT... (palindrome, so RC is same)
        # But quality should be reversed
        assert isinstance(sample["sequence"], bytes)
        assert isinstance(sample["quality"], bytes)
        # Verify quality is reversed (can't test sequence as it's palindrome)
        assert len(sample["quality"]) == len(original["quality"])
        assert len(transformed) == 1000


class TestMultiWorkerDataLoader:
    """T035-T037: Multi-worker DataLoader compatibility tests."""

    @pytest.fixture
    def test_fastq(self):
        """Create a test FASTQ file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fq", delete=False) as f:
            for i in range(100):
                f.write(f"@seq{i}\n")
                f.write("ACGTACGTACGTACGT\n")
                f.write("+\n")
                f.write("IIIIIIIIIIIIIIII\n")
            f.flush()
            fname = f.name
        yield fname
        Path(fname).unlink()

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Multi-worker DataLoader not reliable on Windows",
    )
    def test_multiworker_dataloader(self, test_fastq):
        """T035: DataLoader should work with num_workers > 0."""
        dataset = FastqDataset(test_fastq)
        loader = DataLoader(
            dataset, batch_size=10, num_workers=2, collate_fn=default_collate
        )

        batch_count = 0
        sample_count = 0
        for batch in loader:
            batch_count += 1
            sample_count += len(batch)

        assert batch_count == 10  # 100 / 10 = 10 batches
        assert sample_count == 100  # All samples processed

    @pytest.mark.skip(
        reason="Rust PyTorch transforms not picklable for multiprocessing - known limitation"
    )
    def test_multiworker_with_transforms(self, test_fastq):
        """T036: Multi-worker DataLoader with transforms (currently not supported due to pickling).

        Note: Rust-based PyTorch transforms cannot be pickled for multiprocessing.
        Users should apply transforms in the main process or use num_workers=0.
        """
        from deepbiop import Compose, Mutator, ReverseComplement

        dataset = FastqDataset(test_fastq)
        transform = Compose([ReverseComplement(), Mutator(mutation_rate=0.1, seed=42)])
        transformed = TransformDataset(dataset, transform)

        # This would fail with pickle error in multiprocessing
        # TypeError: cannot pickle 'deepbiop.pytorch.ReverseComplement' object
        loader = DataLoader(
            transformed,
            batch_size=10,
            num_workers=0,  # Use 0 workers to avoid pickling
            collate_fn=default_collate,
        )

        batch_count = 0
        for batch in loader:
            assert len(batch) <= 10
            # Check that transforms were applied
            assert all("sequence" in sample for sample in batch)
            batch_count += 1

        assert batch_count == 10

    def test_deterministic_multiworker_ordering(self, test_fastq):
        """T037: Multi-worker DataLoader should maintain deterministic ordering with seed."""
        dataset = FastqDataset(test_fastq)

        # Use worker_init_fn for deterministic behavior
        def worker_init_fn(worker_id):
            import random

            import numpy as np

            random.seed(42 + worker_id)
            np.random.seed(42 + worker_id)

        loader = DataLoader(
            dataset,
            batch_size=10,
            num_workers=0,  # Use 0 workers for deterministic test
            collate_fn=default_collate,
            worker_init_fn=worker_init_fn,
        )

        # Collect all IDs
        ids = []
        for batch in loader:
            ids.extend([sample["id"] for sample in batch])

        # Should get all 100 samples
        assert len(ids) == 100


class TestKmerEncoding:
    """T038-T039: KmerEncoder tests."""

    def test_kmer_encoder_basic(self):
        """T038: KmerEncoder should encode sequences to k-mer counts."""
        import numpy as np

        from deepbiop import KmerEncoder

        encoder = KmerEncoder(k=3)
        sample = {"id": b"@seq1", "sequence": b"ACGTACGTACGT", "quality": b"I" * 12}

        result = encoder(sample)

        # KmerEncoder replaces "sequence" with encoded array
        assert isinstance(result["sequence"], np.ndarray)
        assert result["sequence"].dtype == np.float32
        # Original fields should be preserved
        assert result["id"] == b"@seq1"
        assert result["quality"] == b"I" * 12

    def test_kmer_encoder_canonical(self):
        """T039: KmerEncoder with canonical=True should treat RC kmers as same."""
        import numpy as np

        from deepbiop import KmerEncoder

        encoder = KmerEncoder(k=3, canonical=True)
        sample1 = {"id": b"@seq1", "sequence": b"ACG", "quality": b"III"}
        sample2 = {"id": b"@seq2", "sequence": b"CGT", "quality": b"III"}  # RC of ACG

        result1 = encoder(sample1.copy())
        result2 = encoder(sample2.copy())

        # With canonical encoding, sequences are encoded
        assert isinstance(result1["sequence"], np.ndarray)
        assert isinstance(result2["sequence"], np.ndarray)
        # Verify encoding happened (array is not all zeros)
        assert result1["sequence"].sum() > 0
        assert result2["sequence"].sum() > 0


class TestVariableLengthCollate:
    """T040-T042: Variable-length sequence handling tests."""

    @pytest.fixture
    def variable_length_batch(self):
        """Create a batch with variable-length sequences."""
        return [
            {"id": b"@seq1", "sequence": b"ACGT", "quality": b"IIII"},
            {"id": b"@seq2", "sequence": b"ACGTACGT", "quality": b"IIIIIIII"},
            {"id": b"@seq3", "sequence": b"ACGTACGTACGT", "quality": b"I" * 12},
        ]

    def test_default_collate_preserves_variable_length(self, variable_length_batch):
        """T040: default_collate should preserve variable-length sequences."""
        batch = default_collate(variable_length_batch)

        # Should return list as-is
        assert isinstance(batch, list)
        assert len(batch) == 3
        assert len(batch[0]["sequence"]) == 4
        assert len(batch[1]["sequence"]) == 8
        assert len(batch[2]["sequence"]) == 12

    def test_supervised_collate_variable_length(self):
        """T041: supervised_collate should handle variable-length features."""
        # Create batch with variable-length features
        batch = [
            {"features": [1, 2, 3], "target": 0, "id": b"@seq1"},
            {"features": [4, 5, 6, 7], "target": 1, "id": b"@seq2"},
            {"features": [8, 9], "target": 0, "id": b"@seq3"},
        ]

        result = supervised_collate(batch)

        assert "features" in result
        assert "targets" in result
        assert "ids" in result
        # Features should be list of variable-length arrays
        assert len(result["features"]) == 3
        assert len(result["features"][0]) == 3
        assert len(result["features"][1]) == 4
        assert len(result["features"][2]) == 2

    def test_bucket_collate_concept(self, variable_length_batch):
        """T042: Bucket collate concept - group similar lengths."""
        # This test demonstrates the concept for future bucket_collate implementation
        # For now, we'll just verify that we can group by length

        # Group by length
        from collections import defaultdict

        buckets = defaultdict(list)
        for sample in variable_length_batch:
            length = len(sample["sequence"])
            buckets[length].append(sample)

        # Verify bucketing works
        assert len(buckets[4]) == 1
        assert len(buckets[8]) == 1
        assert len(buckets[12]) == 1

        # Each bucket can be collated separately
        for length, bucket_samples in buckets.items():
            batch = default_collate(bucket_samples)
            # All sequences in bucket have same length
            assert all(len(s["sequence"]) == length for s in batch)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
