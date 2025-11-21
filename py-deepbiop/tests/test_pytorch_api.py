"""Tests for PyTorch-style Python API.

This test module validates the PyTorch-compatible Dataset, DataLoader,
and Transform classes for biological sequence data loading.

Test Structure:
- TestDataset: Dataset creation, indexing, transforms
- TestDataLoader: Batching, shuffling, iteration
- TestTransforms: Encoder and augmentation wrappers
- TestCollate: Collate function behavior
- TestCache: Cache save/load/validation
- TestIntegration: End-to-end workflows
"""

from pathlib import Path

# Tests will be added in subsequent tasks
# - T013-T016: Dataset tests
# - T026-T027: DataLoader tests
# - T019-T022: Transform tests
# - T030: Collate function tests
# - T054-T056: Cache tests
# - T034: Integration tests


class TestDataset:
    """Test Dataset class."""

    def test_placeholder(self):
        """Placeholder test to ensure pytest runs."""
        assert True

    def test_dataset_creation(self):
        """Test Dataset creation from FASTQ file."""
        # Import the Dataset class
        from deepbiop import pytorch

        # Path to test FASTQ file
        test_file = Path(__file__).parent / "data" / "test.fastq"
        assert test_file.exists(), f"Test file not found: {test_file}"

        # Create dataset
        dataset = pytorch.Dataset(str(test_file))

        # Check dataset length (test.fastq has 1000 sequences)
        assert len(dataset) == 1000, f"Expected 1000 sequences, got {len(dataset)}"

        # Check __getitem__ returns valid Sample
        sample = dataset[0]
        assert isinstance(sample, dict), "Sample should be a dict"
        assert "sequence" in sample, "Sample should have 'sequence' key"
        assert isinstance(sample["sequence"], bytes), "Sequence should be bytes"

        # Check valid indexing
        first_sample = dataset[0]
        last_sample = dataset[999]
        assert first_sample is not None
        assert last_sample is not None

        # Check __repr__
        repr_str = repr(dataset)
        assert "Dataset" in repr_str
        assert "1000" in repr_str or "num_samples=1000" in repr_str


class TestDataLoader:
    """Test DataLoader class."""

    def test_placeholder(self):
        """Placeholder test to ensure pytest runs."""
        assert True

    def test_dataloader_batching(self):
        """Test DataLoader basic batching functionality."""
        from pathlib import Path

        from deepbiop import pytorch

        # Create dataset
        test_file = Path(__file__).parent / "data" / "test.fastq"
        dataset = pytorch.Dataset(str(test_file))

        # Create data loader with batch_size=5
        loader = pytorch.DataLoader(dataset, batch_size=5, shuffle=False)

        # Check __len__ (1000 samples / 5 batch_size = 200 batches)
        assert len(loader) == 200, f"Expected 200 batches, got {len(loader)}"

        # Iterate and check batches
        batch_count = 0
        for batch in loader:
            batch_count += 1
            assert isinstance(batch, list), "Batch should be a list of samples"
            # All batches should have batch_size samples (1000 is evenly divisible by 5)
            assert len(batch) == 5, (
                f"Expected 5 samples in batch {batch_count}, got {len(batch)}"
            )

        assert batch_count == 200, f"Expected to iterate 200 batches, got {batch_count}"

        # Check __repr__
        repr_str = repr(loader)
        assert "DataLoader" in repr_str
        assert "batch_size=5" in repr_str


class TestTransforms:
    """Test Transform classes."""

    def test_placeholder(self):
        """Placeholder test to ensure pytest runs."""
        assert True

    def test_onehot_encoder(self):
        """Test OneHotEncoder transform wrapper."""
        import numpy as np

        from deepbiop import pytorch

        # Create encoder
        encoder = pytorch.OneHotEncoder(encoding_type="dna", unknown_strategy="skip")

        # Create sample with DNA sequence
        sample = {"sequence": b"ACGT", "quality": b"!!!!"}

        # Apply encoder
        transformed = encoder(sample)

        # Check sequence is encoded as NumPy array
        assert isinstance(transformed["sequence"], np.ndarray), (
            "Encoded sequence should be NumPy array"
        )
        assert transformed["sequence"].shape == (4, 4), (
            f"Expected shape (4, 4), got {transformed['sequence'].shape}"
        )
        assert transformed["sequence"].dtype == np.float32, (
            "Encoded sequence should be float32"
        )

        # Check encoding is correct (one-hot)
        # A=0, C=1, G=2, T=3
        expected = np.array(
            [
                [1, 0, 0, 0],  # A
                [0, 1, 0, 0],  # C
                [0, 0, 1, 0],  # G
                [0, 0, 0, 1],  # T
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(transformed["sequence"], expected)

        # Check quality is preserved
        assert transformed["quality"] == b"!!!!", "Quality should be preserved"

        # Check __repr__
        repr_str = repr(encoder)
        assert "OneHotEncoder" in repr_str

    def test_integer_encoder(self):
        """Test IntegerEncoder transform wrapper."""
        import numpy as np

        from deepbiop import pytorch

        # Create encoder
        encoder = pytorch.IntegerEncoder(encoding_type="dna")

        # Create sample with DNA sequence
        sample = {"sequence": b"ACGT", "quality": b"!!!!"}

        # Apply encoder
        transformed = encoder(sample)

        # Check sequence is encoded as NumPy array
        assert isinstance(transformed["sequence"], np.ndarray), (
            "Encoded sequence should be NumPy array"
        )
        assert transformed["sequence"].shape == (4,), (
            f"Expected shape (4,), got {transformed['sequence'].shape}"
        )
        assert transformed["sequence"].dtype == np.float32, (
            "Encoded sequence should be float32"
        )

        # Check encoding is correct (integer)
        # A=0, C=1, G=2, T=3
        expected = np.array([0, 1, 2, 3], dtype=np.float32)
        np.testing.assert_array_equal(transformed["sequence"], expected)

        # Check quality is preserved
        assert transformed["quality"] == b"!!!!", "Quality should be preserved"

        # Check __repr__
        repr_str = repr(encoder)
        assert "IntegerEncoder" in repr_str

    def test_kmer_encoder(self):
        """Test KmerEncoder transform wrapper."""
        import numpy as np

        from deepbiop import pytorch

        # Create encoder (k=3 for 3-mers)
        encoder = pytorch.KmerEncoder(k=3, canonical=False, encoding_type="dna")

        # Create sample with DNA sequence
        sample = {"sequence": b"ACGTACGT", "quality": b"!!!!!!!!"}

        # Apply encoder
        transformed = encoder(sample)

        # Check sequence is encoded as NumPy array
        assert isinstance(transformed["sequence"], np.ndarray), (
            "Encoded sequence should be NumPy array"
        )
        # For DNA with k=3, there are 4^3 = 64 possible k-mers
        assert transformed["sequence"].shape == (64,), (
            f"Expected shape (64,), got {transformed['sequence'].shape}"
        )
        assert transformed["sequence"].dtype == np.float32, (
            "Encoded sequence should be float32"
        )

        # Check that we have k-mer counts (total should equal number of k-mers)
        # For "ACGTACGT" (8 bases), we have 8-3+1 = 6 k-mers
        total_count = np.sum(transformed["sequence"])
        assert total_count == 6.0, f"Expected 6 k-mer counts, got {total_count}"

        # Check quality is preserved
        assert transformed["quality"] == b"!!!!!!!!", "Quality should be preserved"

        # Check __repr__
        repr_str = repr(encoder)
        assert "KmerEncoder" in repr_str

    def test_compose(self):
        """Test Compose transform for chaining transformations."""
        from deepbiop import pytorch

        # Create a pipeline: reverse complement → one-hot encoding
        transform = pytorch.Compose(
            [pytorch.ReverseComplement(), pytorch.OneHotEncoder(encoding_type="dna")]
        )

        # Create sample
        sample = {"sequence": b"ACGT", "quality": b"!!!!"}

        # Apply composed transform
        transformed = transform(sample)

        # Check that transformation was applied
        assert "sequence" in transformed
        assert transformed["sequence"].shape == (4, 4), (
            f"Expected shape (4, 4), got {transformed['sequence'].shape}"
        )

        # Quality should be preserved
        assert transformed["quality"] == b"!!!!", "Quality should be preserved"

    def test_reverse_complement(self):
        """Test ReverseComplement transform wrapper."""
        from deepbiop import pytorch

        # Create transform
        transform = pytorch.ReverseComplement()

        # Create sample
        sample = {"sequence": b"ACGT", "quality": b"!!!!"}

        # Apply transform
        transformed = transform(sample)

        # Check sequence was reverse complemented (ACGT → ACGT is palindrome)
        assert isinstance(transformed["sequence"], bytes), "Sequence should be bytes"
        assert len(transformed["sequence"]) == 4, "Length should be preserved"

        # Quality should be reversed to match sequence
        assert transformed["quality"] == b"!!!!", "Quality should be preserved"

        # Check __repr__
        repr_str = repr(transform)
        assert "ReverseComplement" in repr_str

    def test_mutator(self):
        """Test Mutator transform wrapper."""
        from deepbiop import pytorch

        # Create mutator with fixed seed for reproducibility
        transform = pytorch.Mutator(mutation_rate=0.5, seed=42)

        # Create sample
        sample = {"sequence": b"AAAA", "quality": b"!!!!"}

        # Apply transform
        transformed = transform(sample)

        # Check sequence is still bytes
        assert isinstance(transformed["sequence"], bytes), "Sequence should be bytes"
        assert len(transformed["sequence"]) == 4, "Length should be preserved"

        # Quality should be preserved
        assert transformed["quality"] == b"!!!!", "Quality should be preserved"

        # Check __repr__
        repr_str = repr(transform)
        assert "Mutator" in repr_str

    def test_sampler(self):
        """Test Sampler transform wrapper."""
        from deepbiop import pytorch

        # Create sampler (sample 4 bases from start)
        transform = pytorch.Sampler(length=4, strategy="start")

        # Create sample
        sample = {"sequence": b"ACGTACGT", "quality": b"!!!!!!!!"}

        # Apply transform
        transformed = transform(sample)

        # Check sequence was sampled
        assert isinstance(transformed["sequence"], bytes), "Sequence should be bytes"
        assert len(transformed["sequence"]) == 4, "Length should be 4"
        assert transformed["sequence"] == b"ACGT", "Should sample first 4 bases"

        # Quality should be sampled too
        assert transformed["quality"] == b"!!!!", (
            "Quality should match sampled sequence"
        )

        # Check __repr__
        repr_str = repr(transform)
        assert "Sampler" in repr_str


class TestCollate:
    """Test collate functions."""

    def test_placeholder(self):
        """Placeholder test to ensure pytest runs."""
        assert True

    def test_default_collate(self):
        """Test default_collate_fn stacks samples into batch dict."""
        import numpy as np

        from deepbiop import pytorch

        # Create samples with encoded sequences (simulating OneHotEncoder output)
        samples = [
            {
                "sequence": np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32),
                "quality": b"!!",
            },
            {
                "sequence": np.array(
                    [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]], dtype=np.float32
                ),
                "quality": b"!!!",
            },
            {"sequence": np.array([[0, 1, 0, 0]], dtype=np.float32), "quality": b"!"},
        ]

        # Collate samples into batch
        batch = pytorch.default_collate(samples)

        # Check batch structure
        assert isinstance(batch, dict), "Batch should be a dict"
        assert "sequences" in batch, "Batch should have 'sequences' key"
        assert "lengths" in batch, "Batch should have 'lengths' key"

        # Check sequences are padded and stacked
        sequences = batch["sequences"]
        assert isinstance(sequences, np.ndarray), "Sequences should be NumPy array"
        assert sequences.shape[0] == 3, (
            f"Expected batch size 3, got {sequences.shape[0]}"
        )
        assert sequences.shape[1] == 3, (
            f"Expected max length 3 (padded), got {sequences.shape[1]}"
        )
        assert sequences.shape[2] == 4, (
            f"Expected feature dim 4, got {sequences.shape[2]}"
        )

        # Check lengths
        lengths = batch["lengths"]
        assert isinstance(lengths, np.ndarray), "Lengths should be NumPy array"
        assert len(lengths) == 3, f"Expected 3 lengths, got {len(lengths)}"
        assert list(lengths) == [2, 3, 1], (
            f"Expected lengths [2, 3, 1], got {list(lengths)}"
        )

        # Check quality is preserved
        assert "quality" in batch, "Batch should have 'quality' key"
        assert batch["quality"] == [b"!!", b"!!!", b"!"], (
            "Quality should be list of bytes"
        )


class TestInspection:
    """Test dataset inspection and validation functionality."""

    def test_dataset_summary(self):
        """Test Dataset.summary() provides statistics about the dataset."""
        from pathlib import Path

        from deepbiop import pytorch

        # Load dataset
        test_file = Path(__file__).parent / "data" / "test.fastq"
        dataset = pytorch.Dataset(str(test_file))

        # Call summary
        summary = dataset.summary()

        # Verify summary structure
        assert isinstance(summary, dict), "Summary should be a dict"

        # Check required keys
        assert "num_samples" in summary, "Summary should have num_samples"
        assert "length_stats" in summary, "Summary should have length_stats"
        assert "memory_footprint" in summary, "Summary should have memory_footprint"

        # Verify num_samples
        assert summary["num_samples"] == 1000, (
            f"Expected 1000 samples, got {summary['num_samples']}"
        )

        # Verify length_stats structure
        length_stats = summary["length_stats"]
        assert isinstance(length_stats, dict), "length_stats should be a dict"
        assert "min" in length_stats, "Should have min length"
        assert "max" in length_stats, "Should have max length"
        assert "mean" in length_stats, "Should have mean length"
        assert "median" in length_stats, "Should have median length"
        assert all(length_stats[k] > 0 for k in ["min", "max", "mean", "median"]), (
            "All length stats should be positive"
        )

        # Verify memory_footprint
        assert isinstance(summary["memory_footprint"], int | float), (
            "memory_footprint should be numeric"
        )
        assert summary["memory_footprint"] > 0, "Memory footprint should be positive"

    def test_dataset_validation(self):
        """Test Dataset.validate() checks data quality."""
        from pathlib import Path

        from deepbiop import pytorch

        # Load dataset
        test_file = Path(__file__).parent / "data" / "test.fastq"
        dataset = pytorch.Dataset(str(test_file))

        # Call validate
        validation_result = dataset.validate()

        # Verify validation result structure
        assert isinstance(validation_result, dict), "Validation result should be a dict"

        # Check required keys
        assert "is_valid" in validation_result, "Should have is_valid flag"
        assert "warnings" in validation_result, "Should have warnings list"
        assert "errors" in validation_result, "Should have errors list"

        # Verify types
        assert isinstance(validation_result["is_valid"], bool), (
            "is_valid should be bool"
        )
        assert isinstance(validation_result["warnings"], list), (
            "warnings should be a list"
        )
        assert isinstance(validation_result["errors"], list), "errors should be a list"

        # For valid test data, should be valid with no errors
        assert validation_result["is_valid"] is True, "Test data should be valid"
        assert len(validation_result["errors"]) == 0, "Test data should have no errors"


class TestCache:
    """Test cache functionality."""

    def test_cache_save(self):
        """Test saving processed dataset to cache."""
        import tempfile
        from pathlib import Path

        from deepbiop import pytorch

        # Load dataset
        test_file = Path(__file__).parent / "data" / "test.fastq"
        dataset = pytorch.Dataset(str(test_file))

        # Create encoder
        encoder = pytorch.OneHotEncoder(encoding_type="dna")

        # Process samples
        processed_samples = []
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            encoded = encoder(sample)
            processed_samples.append(encoded)

        # Save to cache
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.npz"

            # Save processed data
            pytorch.save_cache(processed_samples, str(cache_path))

            # Verify cache file exists
            assert cache_path.exists(), "Cache file should exist"
            assert cache_path.stat().st_size > 0, "Cache file should not be empty"

            # Verify metadata file exists
            metadata_path = Path(str(cache_path) + ".meta.json")
            assert metadata_path.exists(), "Metadata file should exist"

    def test_cache_load(self):
        """Test loading processed dataset from cache."""
        import tempfile
        from pathlib import Path

        import numpy as np

        from deepbiop import pytorch

        # Load and process dataset
        test_file = Path(__file__).parent / "data" / "test.fastq"
        dataset = pytorch.Dataset(str(test_file))
        encoder = pytorch.OneHotEncoder(encoding_type="dna")

        # Process samples
        original_samples = []
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            encoded = encoder(sample)
            original_samples.append(encoded)

        # Save to cache
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.npz"
            pytorch.save_cache(original_samples, str(cache_path))

            # Load from cache
            loaded_samples = pytorch.load_cache(str(cache_path))

            # Verify loaded samples match original
            assert len(loaded_samples) == len(original_samples), (
                f"Expected {len(original_samples)} samples, got {len(loaded_samples)}"
            )

            # Check first sample
            assert "sequence" in loaded_samples[0], (
                "Loaded sample should have 'sequence' key"
            )
            assert isinstance(loaded_samples[0]["sequence"], np.ndarray), (
                "Loaded sequence should be NumPy array"
            )

            # Check shapes match
            np.testing.assert_array_equal(
                loaded_samples[0]["sequence"].shape,
                original_samples[0]["sequence"].shape,
            )

    def test_cache_invalidation(self):
        """Test cache invalidation when source file changes."""
        import tempfile
        import time
        from pathlib import Path

        from deepbiop import pytorch

        # Load dataset
        test_file = Path(__file__).parent / "data" / "test.fastq"
        dataset = pytorch.Dataset(str(test_file))
        encoder = pytorch.OneHotEncoder(encoding_type="dna")

        # Process samples
        samples = []
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            encoded = encoder(sample)
            samples.append(encoded)

        # Save to cache
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.npz"
            pytorch.save_cache(samples, str(cache_path), source_file=str(test_file))

            # Check cache is valid initially
            is_valid = pytorch.is_cache_valid(
                str(cache_path), source_file=str(test_file)
            )
            assert is_valid is True, "Cache should be valid initially"

            # Simulate file modification (update metadata timestamp)
            time.sleep(0.01)  # Ensure time difference

            # Touch the source file to change mtime
            # Note: We can't actually modify the test data, so we test with a different file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".fastq", delete=False
            ) as tmp:
                tmp.write("@seq1\nACGT\n+\n!!!!\n")
                tmp_path = tmp.name

            try:
                # Save cache with temp file reference
                cache_path2 = Path(tmpdir) / "test_cache2.npz"
                pytorch.save_cache(samples, str(cache_path2), source_file=tmp_path)

                # Verify cache is valid
                assert pytorch.is_cache_valid(str(cache_path2), source_file=tmp_path), (
                    "Cache should be valid for temp file"
                )

                # Modify the temp file (wait to ensure mtime changes)
                time.sleep(1.1)  # File mtime has 1-second resolution on many systems
                with Path(tmp_path).open("a") as f:
                    f.write("@seq2\nGGGG\n+\n!!!!\n")

                # Cache should now be invalid
                assert not pytorch.is_cache_valid(
                    str(cache_path2), source_file=tmp_path
                ), "Cache should be invalid after source file modification"
            finally:
                Path(tmp_path).unlink()  # Clean up temp file


class TestIntegration:
    """Test end-to-end workflows."""

    def test_placeholder(self):
        """Placeholder test to ensure pytest runs."""
        assert True

    def test_augmentation_pipeline(self):
        """Test augmentation pipeline with composed transforms."""
        from pathlib import Path

        import numpy as np

        from deepbiop import pytorch

        # 1. Load FASTQ file
        test_file = Path(__file__).parent / "data" / "test.fastq"
        dataset = pytorch.Dataset(str(test_file))

        # 2. Create augmentation pipeline: Sampler → Mutator → ReverseComplement
        augmentation = pytorch.Compose(
            [
                pytorch.Sampler(length=100, strategy="start"),  # Sample first 100 bases
                pytorch.Mutator(mutation_rate=0.1, seed=42),  # Mutate 10% of bases
                pytorch.ReverseComplement(),  # Reverse complement
            ]
        )

        # 3. Apply augmentation to a sample
        sample = dataset[0]
        augmented_sample = augmentation(sample)

        # Verify augmented sample structure
        assert isinstance(augmented_sample, dict), "Augmented sample should be a dict"
        assert "sequence" in augmented_sample, "Should have sequence"
        assert isinstance(augmented_sample["sequence"], bytes), (
            "Sequence should be bytes"
        )

        # Verify length is 100 (from Sampler)
        assert len(augmented_sample["sequence"]) == 100, (
            f"Expected length 100, got {len(augmented_sample['sequence'])}"
        )

        # 4. Create encoding pipeline: augmentation → encoding
        full_pipeline = pytorch.Compose(
            [
                augmentation,
                pytorch.OneHotEncoder(encoding_type="dna"),
            ]
        )

        # Apply full pipeline
        encoded_sample = full_pipeline(sample)

        # Verify encoded output
        assert isinstance(encoded_sample, dict), "Encoded sample should be a dict"
        assert "sequence" in encoded_sample, "Should have sequence"
        assert isinstance(encoded_sample["sequence"], np.ndarray), (
            "Sequence should be NumPy array"
        )
        assert encoded_sample["sequence"].shape == (100, 4), (
            f"Expected shape (100, 4), got {encoded_sample['sequence'].shape}"
        )

        # 5. Process multiple samples and create batch
        encoded_samples = []
        for i in range(5):
            sample = dataset[i]
            encoded = full_pipeline(sample)
            encoded_samples.append(encoded)

        # Collate into batch
        batch = pytorch.default_collate(encoded_samples)

        # Verify batch structure
        assert isinstance(batch, dict), "Batch should be a dict"
        assert "sequences" in batch, "Batch should have sequences"
        assert "lengths" in batch, "Batch should have lengths"
        assert batch["sequences"].shape == (5, 100, 4), (
            f"Expected shape (5, 100, 4), got {batch['sequences'].shape}"
        )

        # All lengths should be 100 (fixed by Sampler)
        assert all(length == 100 for length in batch["lengths"]), (
            "All sequences should have length 100"
        )

    def test_full_pipeline(self):
        """Test complete pipeline: load → transform → batch → ready for PyTorch."""
        from pathlib import Path

        import numpy as np

        from deepbiop import pytorch

        # 1. Load FASTQ file
        test_file = Path(__file__).parent / "data" / "test.fastq"
        dataset = pytorch.Dataset(str(test_file))

        # Verify dataset loaded correctly
        assert len(dataset) == 1000, f"Expected 1000 sequences, got {len(dataset)}"

        # 2. Apply transformation (OneHot encoding)
        encoder = pytorch.OneHotEncoder(encoding_type="dna", unknown_strategy="skip")

        # Transform first few samples manually to test
        sample = dataset[0]
        encoded_sample = encoder(sample)
        assert isinstance(encoded_sample["sequence"], np.ndarray), (
            "Encoded sequence should be NumPy array"
        )
        assert encoded_sample["sequence"].ndim == 2, "Encoded sequence should be 2D"

        # 3. Create DataLoader with batching
        pytorch.DataLoader(dataset, batch_size=5, shuffle=False)

        # 4. Iterate and collect batch - encode manually
        batch_samples = []
        for sample in dataset:
            encoded = encoder(sample)
            batch_samples.append(encoded)
            if len(batch_samples) == 5:
                break

        # 5. Collate samples into batch
        batch = pytorch.default_collate(batch_samples)

        # 6. Verify batch structure (ready for PyTorch)
        assert isinstance(batch, dict), "Batch should be a dict"
        assert "sequences" in batch, "Batch should have sequences"
        assert "lengths" in batch, "Batch should have lengths"

        # Verify batch shape [batch_size, max_len, features]
        sequences = batch["sequences"]
        assert sequences.shape[0] == 5, (
            f"Expected batch size 5, got {sequences.shape[0]}"
        )
        assert sequences.ndim == 3, f"Expected 3D tensor, got {sequences.ndim}D"
        assert sequences.shape[2] == 4, (
            f"Expected 4 features (ACGT), got {sequences.shape[2]}"
        )

        # Verify lengths are recorded
        lengths = batch["lengths"]
        assert len(lengths) == 5, f"Expected 5 lengths, got {len(lengths)}"
        assert all(length > 0 for length in lengths), "All lengths should be positive"

        # Verify quality metadata is preserved
        assert "quality" in batch, "Quality should be preserved"
        assert len(batch["quality"]) == 5, "Should have quality for all samples"


class TestPyTorchIntegration:
    """Test PyTorch integration and tensor compatibility."""

    def test_numpy_pytorch_conversion(self):
        """Test that NumPy arrays can be converted to PyTorch tensors."""
        from pathlib import Path

        import numpy as np

        from deepbiop import pytorch

        # Check if PyTorch is available
        try:
            import torch
        except ImportError:
            import pytest

            pytest.skip("PyTorch not installed")

        # Load data and create batch
        test_file = Path(__file__).parent / "data" / "test.fastq"
        dataset = pytorch.Dataset(str(test_file))
        encoder = pytorch.OneHotEncoder(encoding_type="dna")

        # Create batch
        encoded_samples = []
        for i in range(5):
            sample = dataset[i]
            encoded = encoder(sample)
            encoded_samples.append(encoded)

        batch = pytorch.default_collate(encoded_samples)

        # Verify NumPy array properties
        sequences = batch["sequences"]
        assert isinstance(sequences, np.ndarray), "Should be NumPy array"
        assert sequences.dtype == np.float32, f"Expected float32, got {sequences.dtype}"
        assert sequences.flags["C_CONTIGUOUS"], "Array should be C-contiguous"

        # Convert to PyTorch tensor
        tensor = torch.from_numpy(sequences)

        # Verify tensor properties
        assert tensor.dtype == torch.float32, "Tensor should be float32"
        assert tensor.shape == sequences.shape, "Shape should be preserved"
        assert tensor.is_contiguous(), "Tensor should be contiguous"

        # Verify data integrity after conversion
        np.testing.assert_array_equal(tensor.numpy(), sequences)

    def test_batch_indexing_slicing(self):
        """Test that batch indexing and slicing work correctly."""
        from pathlib import Path

        import numpy as np

        from deepbiop import pytorch

        # Check if PyTorch is available
        try:
            import torch
        except ImportError:
            import pytest

            pytest.skip("PyTorch not installed")

        # Load data and create batch
        test_file = Path(__file__).parent / "data" / "test.fastq"
        dataset = pytorch.Dataset(str(test_file))
        encoder = pytorch.OneHotEncoder(encoding_type="dna")

        # Create batch
        encoded_samples = []
        for i in range(10):
            sample = dataset[i]
            encoded = encoder(sample)
            encoded_samples.append(encoded)

        batch = pytorch.default_collate(encoded_samples)
        sequences = batch["sequences"]

        # Test indexing - get first sample
        first_sample = sequences[0]
        assert first_sample.shape == (sequences.shape[1], sequences.shape[2]), (
            f"Expected shape ({sequences.shape[1]}, {sequences.shape[2]}), got {first_sample.shape}"
        )

        # Test slicing - get first 3 samples
        first_three = sequences[:3]
        assert first_three.shape == (3, sequences.shape[1], sequences.shape[2]), (
            f"Expected shape (3, {sequences.shape[1]}, {sequences.shape[2]}), got {first_three.shape}"
        )

        # Convert to PyTorch and verify slicing still works
        tensor = torch.from_numpy(sequences)

        # Test PyTorch indexing
        torch_first = tensor[0]
        np.testing.assert_array_equal(torch_first.numpy(), first_sample)

        # Test PyTorch slicing
        torch_first_three = tensor[:3]
        np.testing.assert_array_equal(torch_first_three.numpy(), first_three)

    def test_pytorch_model_integration(self):
        """Test integration with a simple PyTorch model."""
        from pathlib import Path

        from deepbiop import pytorch

        # Check if PyTorch is available
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            import pytest

            pytest.skip("PyTorch not installed")

        # Load data and create batch
        test_file = Path(__file__).parent / "data" / "test.fastq"
        dataset = pytorch.Dataset(str(test_file))

        # Use Sampler to get fixed-length sequences
        transform = pytorch.Compose(
            [
                pytorch.Sampler(length=100, strategy="start"),
                pytorch.OneHotEncoder(encoding_type="dna"),
            ]
        )

        # Create batch
        encoded_samples = []
        for i in range(4):
            sample = dataset[i]
            encoded = transform(sample)
            encoded_samples.append(encoded)

        batch = pytorch.default_collate(encoded_samples)
        sequences = batch["sequences"]

        # Convert to PyTorch tensor
        tensor = torch.from_numpy(sequences)

        # Define a simple CNN model
        class SimpleCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv1d(
                    in_channels=4, out_channels=16, kernel_size=3, padding=1
                )
                self.pool = nn.MaxPool1d(kernel_size=2)
                self.fc = nn.Linear(16 * 50, 2)  # 100 / 2 = 50 after pooling

            def forward(self, x):
                # x shape: [batch, length, features] → need [batch, features, length]
                x = x.permute(0, 2, 1)
                x = torch.relu(self.conv1(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x

        # Create model
        model = SimpleCNN()
        model.eval()

        # Forward pass - should not raise any errors
        with torch.no_grad():
            output = model(tensor)

        # Verify output shape
        assert output.shape == (4, 2), f"Expected shape (4, 2), got {output.shape}"

        # Test with loss function
        labels = torch.tensor([0, 1, 0, 1])
        criterion = nn.CrossEntropyLoss()

        # This should not raise any errors
        loss = criterion(output, labels)
        assert isinstance(loss.item(), float), "Loss should be a float"
        assert loss.item() >= 0, "Loss should be non-negative"
