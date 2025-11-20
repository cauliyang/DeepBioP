"""
Tests for supervised learning features.

Tests target extraction, collate functions, and BiologicalDataModule
with supervised learning support.
"""

from pathlib import Path

import pytest


class TestTargetExtractor:
    """Test TargetExtractor class and factory methods."""

    def test_from_quality_mean(self):
        """Test extracting mean quality score."""
        from deepbiop.targets import TargetExtractor

        extractor = TargetExtractor.from_quality(stat="mean")

        record = {
            "id": b"@read_1",
            "sequence": b"ACGTACGT",
            "quality": [30, 32, 35, 38, 40, 42, 40, 38],
        }

        target = extractor(record)
        expected_mean = sum(record["quality"]) / len(record["quality"])

        assert abs(target - expected_mean) < 0.01, f"Expected {expected_mean}, got {target}"

    def test_from_quality_median(self):
        """Test extracting median quality score."""
        from deepbiop.targets import TargetExtractor

        extractor = TargetExtractor.from_quality(stat="median")

        record = {
            "id": b"@read_1",
            "sequence": b"ACGT",
            "quality": [30, 35, 40, 45],  # median should be (35+40)/2 = 37.5
        }

        target = extractor(record)
        assert target == 37.5

    def test_from_header_pattern(self):
        """Test extracting target from header using regex pattern."""
        from deepbiop.targets import TargetExtractor

        extractor = TargetExtractor.from_header(pattern=r"label=(\w+)")

        record = {
            "id": b"@read_1 label=positive score=0.95",
            "sequence": b"ACGT",
            "quality": None,
        }

        target = extractor(record)
        assert target == "positive"

    def test_from_header_key_value(self):
        """Test extracting target from header using key:value pairs."""
        from deepbiop.targets import TargetExtractor

        extractor = TargetExtractor.from_header(key="class", separator="|", converter=int)

        record = {
            "id": b"@read_1|class:1|score:0.95",
            "sequence": b"ACGT",
            "quality": None,
        }

        target = extractor(record)
        assert target == 1

    def test_from_sequence_gc_content(self):
        """Test extracting GC content from sequence."""
        from deepbiop.targets import TargetExtractor

        extractor = TargetExtractor.from_sequence(feature="gc_content")

        record = {
            "id": b"@read_1",
            "sequence": b"ACGTACGT",  # 4 GC out of 8 = 0.5
            "quality": None,
        }

        target = extractor(record)
        assert target == 0.5

    def test_from_sequence_length(self):
        """Test extracting sequence length."""
        from deepbiop.targets import TargetExtractor

        extractor = TargetExtractor.from_sequence(feature="length")

        record = {
            "id": b"@read_1",
            "sequence": b"ACGTACGT",
            "quality": None,
        }

        target = extractor(record)
        assert target == 8

    def test_custom_extractor(self):
        """Test custom extraction function."""
        from deepbiop.targets import TargetExtractor

        def custom_fn(record):
            """Extract first character of sequence."""
            return record["sequence"][0:1].decode()

        extractor = TargetExtractor(custom_fn)

        record = {
            "id": b"@read_1",
            "sequence": b"ACGT",
            "quality": None,
        }

        target = extractor(record)
        assert target == "A"

    def test_constant_extractor(self):
        """Test constant value extractor."""
        from deepbiop.targets import TargetExtractor

        extractor = TargetExtractor.constant(42)

        record = {"id": b"@read_1", "sequence": b"ACGT", "quality": None}

        target = extractor(record)
        assert target == 42


class TestClassificationExtractor:
    """Test classification-specific extractors."""

    def test_create_classification_extractor(self):
        """Test creating classification extractor."""
        from deepbiop.targets import create_classification_extractor

        extractor = create_classification_extractor(
            classes=["negative", "positive"],
            pattern=r"class=(\w+)",
        )

        record_neg = {
            "id": b"@read_1 class=negative",
            "sequence": b"ACGT",
            "quality": None,
        }

        record_pos = {
            "id": b"@read_2 class=positive",
            "sequence": b"ACGT",
            "quality": None,
        }

        assert extractor(record_neg) == 0  # negative -> 0
        assert extractor(record_pos) == 1  # positive -> 1


class TestCollateFunctions:
    """Test collate functions for batching."""

    def test_default_collate(self):
        """Test default (identity) collate function."""
        from deepbiop.collate import default_collate

        batch = [
            {"id": b"read_1", "sequence": b"ACGT"},
            {"id": b"read_2", "sequence": b"TTGG"},
        ]

        result = default_collate(batch)
        assert result == batch  # Identity function

    def test_supervised_collate(self):
        """Test supervised collate function."""
        from deepbiop.collate import supervised_collate

        batch = [
            {"features": [1, 2, 3], "target": 0.5, "id": b"read_1"},
            {"features": [4, 5, 6], "target": 0.8, "id": b"read_2"},
        ]

        result = supervised_collate(batch)

        assert "features" in result
        assert "targets" in result
        assert "ids" in result
        assert len(result["features"]) == 2
        assert len(result["targets"]) == 2

    def test_supervised_collate_tuples(self):
        """Test supervised collate with tuple inputs."""
        from deepbiop.collate import supervised_collate

        batch = [
            ([1, 2, 3], 0.5),
            ([4, 5, 6], 0.8),
        ]

        result = supervised_collate(batch)

        assert "features" in result
        assert "targets" in result
        assert len(result["features"]) == 2
        assert result["targets"] == [0.5, 0.8]


class TestTransformDataset:
    """Test TransformDataset with target extraction."""

    def test_transform_dataset_with_target(self):
        """Test TransformDataset with target extraction."""
        from deepbiop.transforms import TransformDataset
        from deepbiop.targets import TargetExtractor

        # Create mock dataset
        class MockDataset:
            def __iter__(self):
                yield {"id": b"read_1", "sequence": b"ACGTACGT", "quality": [30, 32, 35, 38]}
                yield {"id": b"read_2", "sequence": b"TTGGCCAA", "quality": [40, 42, 44, 46]}

        # Create target extractor
        target_fn = TargetExtractor.from_quality("mean")

        # Wrap dataset
        dataset = TransformDataset(
            MockDataset(),
            transform=None,
            target_fn=target_fn,
            return_dict=True,
        )

        # Iterate and check
        samples = list(dataset)
        assert len(samples) == 2

        # Check first sample
        assert "target" in samples[0]
        assert abs(samples[0]["target"] - 33.75) < 0.01  # Mean of [30, 32, 35, 38]

        # Check second sample
        assert "target" in samples[1]
        assert abs(samples[1]["target"] - 43.0) < 0.01  # Mean of [40, 42, 44, 46]

    def test_transform_dataset_tuple_return(self):
        """Test TransformDataset returning tuples."""
        from deepbiop.transforms import TransformDataset
        from deepbiop.targets import TargetExtractor

        class MockDataset:
            def __iter__(self):
                yield {"id": b"read_1", "sequence": b"ACGT", "quality": [30, 32]}

        target_fn = TargetExtractor.from_quality("mean")

        dataset = TransformDataset(
            MockDataset(),
            transform=None,
            target_fn=target_fn,
            return_dict=False,  # Return tuples
        )

        sample = next(iter(dataset))

        # Should return (features, target) tuple
        assert isinstance(sample, tuple)
        assert len(sample) == 2

        features, target = sample
        assert features == b"ACGT"  # No transform, so raw sequence
        assert abs(target - 31.0) < 0.01


class TestBiologicalDataModule:
    """Test BiologicalDataModule with supervised learning."""

    def test_data_module_with_transform_and_target(self):
        """Test BiologicalDataModule with transform and target extraction."""
        pytest.importorskip("pytorch_lightning", reason="PyTorch Lightning not installed")

        from deepbiop.lightning import BiologicalDataModule
        from deepbiop.targets import TargetExtractor

        test_file = Path(__file__).parent / "data" / "test.fastq"
        if not test_file.exists():
            pytest.skip(f"Test file not found: {test_file}")

        # Create data module with target extraction
        data_module = BiologicalDataModule(
            train_path=str(test_file),
            val_path=str(test_file),
            target_fn=TargetExtractor.from_quality("mean"),
            collate_mode="supervised",
            batch_size=4,
            num_workers=0,
        )

        # Setup
        data_module.setup(stage="fit")

        # Get train dataloader
        train_loader = data_module.train_dataloader()
        assert train_loader is not None

        # Get first batch
        batch = next(iter(train_loader))

        # Check batch structure
        # With supervised collate, batch should have "targets" key
        assert isinstance(batch, (dict, list))


class TestGetBuiltinExtractor:
    """Test convenience function for built-in extractors."""

    def test_get_builtin_quality_extractor(self):
        """Test getting built-in quality extractor."""
        from deepbiop.targets import get_builtin_extractor

        extractor = get_builtin_extractor("quality_mean")

        record = {"id": b"read_1", "sequence": b"ACGT", "quality": [30, 40]}

        target = extractor(record)
        assert target == 35.0  # Mean of [30, 40]

    def test_get_builtin_sequence_extractor(self):
        """Test getting built-in sequence extractor."""
        from deepbiop.targets import get_builtin_extractor

        extractor = get_builtin_extractor("gc_content")

        record = {"id": b"read_1", "sequence": b"GGCC", "quality": None}

        target = extractor(record)
        assert target == 1.0  # All GC

    def test_get_builtin_invalid_name(self):
        """Test error handling for invalid extractor name."""
        from deepbiop.targets import get_builtin_extractor

        with pytest.raises(ValueError, match="Unknown built-in extractor"):
            get_builtin_extractor("invalid_name")
