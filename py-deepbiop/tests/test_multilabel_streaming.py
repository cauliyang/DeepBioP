"""Tests for multi-label learning and streaming dataset features."""

import tempfile
from pathlib import Path

import pytest


class TestMultiLabelExtractor:
    """Test MultiLabelExtractor for multi-task learning."""

    def test_dict_extractors_dict_output(self):
        """Test MultiLabelExtractor with dict of extractors and dict output."""
        from deepbiop.targets import MultiLabelExtractor, TargetExtractor

        # Create sample record
        record = {
            "id": b"read1",
            "sequence": b"ACGTACGT",
            "quality": [30, 32, 35, 38],
        }

        # Create multi-label extractor
        extractor = MultiLabelExtractor(
            {
                "quality": TargetExtractor.from_quality(stat="mean"),
                "gc": TargetExtractor.from_sequence(feature="gc_content"),
                "length": TargetExtractor.from_sequence(feature="length"),
            }
        )

        # Extract targets
        targets = extractor(record)

        # Verify structure
        assert isinstance(targets, dict)
        assert set(targets.keys()) == {"quality", "gc", "length"}

        # Verify values
        assert targets["quality"] == pytest.approx(33.75, abs=0.01)
        assert targets["gc"] == pytest.approx(0.5, abs=0.01)
        assert targets["length"] == 8

    def test_dict_extractors_tuple_output(self):
        """Test MultiLabelExtractor with dict of extractors and tuple output."""
        from deepbiop.targets import MultiLabelExtractor, TargetExtractor

        record = {
            "id": b"read1",
            "sequence": b"ACGTACGT",
            "quality": [30, 32, 35, 38],
        }

        extractor = MultiLabelExtractor(
            {
                "quality": TargetExtractor.from_quality(stat="mean"),
                "gc": TargetExtractor.from_sequence(feature="gc_content"),
            },
            output_format="tuple",
        )

        targets = extractor(record)

        # Verify it's a tuple
        assert isinstance(targets, tuple)
        assert len(targets) == 2

        # Values should be in sorted key order: gc, quality
        assert targets[0] == pytest.approx(0.5, abs=0.01)  # gc
        assert targets[1] == pytest.approx(33.75, abs=0.01)  # quality

    def test_dict_extractors_array_output(self):
        """Test MultiLabelExtractor with dict of extractors and array output."""
        from deepbiop.targets import MultiLabelExtractor, TargetExtractor

        record = {
            "id": b"read1",
            "sequence": b"ACGTACGT",
            "quality": [30, 32, 35, 38],
        }

        extractor = MultiLabelExtractor(
            {
                "quality": TargetExtractor.from_quality(stat="mean"),
                "length": TargetExtractor.from_sequence(feature="length"),
            },
            output_format="array",
        )

        targets = extractor(record)

        # Verify it's a list
        assert isinstance(targets, list)
        assert len(targets) == 2

        # Values in sorted key order: length, quality
        assert targets[0] == 8  # length
        assert targets[1] == pytest.approx(33.75, abs=0.01)  # quality

    def test_list_extractors_tuple_output(self):
        """Test MultiLabelExtractor with list of extractors and tuple output."""
        from deepbiop.targets import MultiLabelExtractor, TargetExtractor

        record = {
            "id": b"read1",
            "sequence": b"ACGTACGT",
            "quality": [30, 32, 35, 38],
        }

        extractor = MultiLabelExtractor(
            [
                TargetExtractor.from_quality(stat="mean"),
                TargetExtractor.from_sequence(feature="gc_content"),
                TargetExtractor.from_sequence(feature="length"),
            ],
            output_format="tuple",
        )

        targets = extractor(record)

        # Verify structure
        assert isinstance(targets, tuple)
        assert len(targets) == 3

        # Verify values in order
        assert targets[0] == pytest.approx(33.75, abs=0.01)
        assert targets[1] == pytest.approx(0.5, abs=0.01)
        assert targets[2] == 8

    def test_list_extractors_array_output(self):
        """Test MultiLabelExtractor with list of extractors and array output."""
        from deepbiop.targets import MultiLabelExtractor, TargetExtractor

        record = {
            "id": b"read1",
            "sequence": b"ACGTACGT",
            "quality": [30, 32, 35, 38],
        }

        extractor = MultiLabelExtractor(
            [
                TargetExtractor.from_quality(stat="mean"),
                TargetExtractor.from_sequence(feature="gc_content"),
            ],
            output_format="array",
        )

        targets = extractor(record)

        # Verify structure
        assert isinstance(targets, list)
        assert len(targets) == 2

        # Verify values
        assert targets[0] == pytest.approx(33.75, abs=0.01)
        assert targets[1] == pytest.approx(0.5, abs=0.01)

    def test_list_extractors_dict_output(self):
        """Test MultiLabelExtractor with list of extractors and dict output."""
        from deepbiop.targets import MultiLabelExtractor, TargetExtractor

        record = {
            "id": b"read1",
            "sequence": b"ACGTACGT",
            "quality": [30, 32, 35, 38],
        }

        extractor = MultiLabelExtractor(
            [
                TargetExtractor.from_quality(stat="mean"),
                TargetExtractor.from_sequence(feature="gc_content"),
            ],
            output_format="dict",
        )

        targets = extractor(record)

        # Verify structure - should use target_0, target_1 keys
        assert isinstance(targets, dict)
        assert set(targets.keys()) == {"target_0", "target_1"}

        # Verify values
        assert targets["target_0"] == pytest.approx(33.75, abs=0.01)
        assert targets["target_1"] == pytest.approx(0.5, abs=0.01)

    def test_invalid_output_format(self):
        """Test MultiLabelExtractor with invalid output format."""
        from deepbiop.targets import MultiLabelExtractor, TargetExtractor

        with pytest.raises(ValueError, match="Invalid output_format"):
            MultiLabelExtractor(
                {"quality": TargetExtractor.from_quality(stat="mean")},
                output_format="invalid",
            )

    def test_single_extractor(self):
        """Test MultiLabelExtractor with single extractor."""
        from deepbiop.targets import MultiLabelExtractor, TargetExtractor

        record = {
            "id": b"read1",
            "sequence": b"ACGT",
            "quality": [30, 30, 30, 30],
        }

        extractor = MultiLabelExtractor(
            {"quality": TargetExtractor.from_quality(stat="mean")}
        )

        targets = extractor(record)

        assert isinstance(targets, dict)
        assert targets["quality"] == 30.0


class TestMultiLabelCollate:
    """Test multi-label collate functions."""

    def test_multi_label_collate_dict_targets(self):
        """Test multi_label_collate with dict targets."""
        from deepbiop.collate import multi_label_collate

        batch = [
            {"features": [1, 2], "target": {"quality": 30.0, "gc": 0.5}},
            {"features": [3, 4], "target": {"quality": 32.0, "gc": 0.6}},
            {"features": [5, 6], "target": {"quality": 28.0, "gc": 0.4}},
        ]

        result = multi_label_collate(batch)

        # Check structure
        assert "features" in result
        assert "targets" in result
        assert len(result["features"]) == 3

        # Check targets are restructured correctly
        assert isinstance(result["targets"], dict)
        assert set(result["targets"].keys()) == {"quality", "gc"}
        assert result["targets"]["quality"] == [30.0, 32.0, 28.0]
        assert result["targets"]["gc"] == [0.5, 0.6, 0.4]

    def test_multi_label_collate_tuple_targets(self):
        """Test multi_label_collate with tuple targets."""
        from deepbiop.collate import multi_label_collate

        batch = [
            {"features": [1, 2], "target": (30.0, 0.5)},
            {"features": [3, 4], "target": (32.0, 0.6)},
        ]

        result = multi_label_collate(batch)

        # Check targets remain as list of tuples
        assert isinstance(result["targets"], list)
        assert len(result["targets"]) == 2
        assert result["targets"][0] == (30.0, 0.5)
        assert result["targets"][1] == (32.0, 0.6)

    def test_multi_label_collate_list_targets(self):
        """Test multi_label_collate with list targets."""
        from deepbiop.collate import multi_label_collate

        batch = [
            {"features": [1, 2], "target": [30.0, 0.5]},
            {"features": [3, 4], "target": [32.0, 0.6]},
        ]

        result = multi_label_collate(batch)

        # Check targets remain as list of lists
        assert isinstance(result["targets"], list)
        assert len(result["targets"]) == 2
        assert result["targets"][0] == [30.0, 0.5]
        assert result["targets"][1] == [32.0, 0.6]

    def test_multi_label_collate_with_ids(self):
        """Test multi_label_collate preserves IDs."""
        from deepbiop.collate import multi_label_collate

        batch = [
            {
                "features": [1, 2],
                "target": {"quality": 30.0},
                "id": b"read1",
            },
            {
                "features": [3, 4],
                "target": {"quality": 32.0},
                "id": b"read2",
            },
        ]

        result = multi_label_collate(batch)

        # Check IDs are preserved
        assert "ids" in result
        assert result["ids"] == [b"read1", b"read2"]

    def test_multi_label_collate_with_quality_scores(self):
        """Test multi_label_collate preserves quality scores."""
        from deepbiop.collate import multi_label_collate

        batch = [
            {
                "features": [1, 2],
                "target": {"gc": 0.5},
                "quality": [30, 32],
            },
            {
                "features": [3, 4],
                "target": {"gc": 0.6},
                "quality": [35, 38],
            },
        ]

        result = multi_label_collate(batch)

        # Check quality scores are preserved
        assert "quality" in result
        assert result["quality"] == [[30, 32], [35, 38]]

    def test_multi_label_collate_empty_batch(self):
        """Test multi_label_collate with empty batch."""
        from deepbiop.collate import multi_label_collate

        result = multi_label_collate([])
        assert result == {}


class TestMultiLabelTensorCollate:
    """Test multi-label tensor collate function."""

    def test_multi_label_tensor_collate_dict_targets(self):
        """Test multi_label_tensor_collate with dict targets."""
        pytest.importorskip("torch")
        import torch

        from deepbiop.collate import multi_label_tensor_collate

        batch = [
            {
                "features": torch.tensor([1.0, 2.0]),
                "target": {"quality": 30.0, "gc": 0.5},
            },
            {
                "features": torch.tensor([3.0, 4.0]),
                "target": {"quality": 32.0, "gc": 0.6},
            },
        ]

        result = multi_label_tensor_collate(batch)

        # Check features are stacked
        assert isinstance(result["features"], torch.Tensor)
        assert result["features"].shape == (2, 2)

        # Check targets are converted to tensors
        assert isinstance(result["targets"], dict)
        assert isinstance(result["targets"]["quality"], torch.Tensor)
        assert isinstance(result["targets"]["gc"], torch.Tensor)
        assert torch.allclose(result["targets"]["quality"], torch.tensor([30.0, 32.0]))
        assert torch.allclose(result["targets"]["gc"], torch.tensor([0.5, 0.6]))

    def test_multi_label_tensor_collate_tuple_targets(self):
        """Test multi_label_tensor_collate with tuple targets."""
        pytest.importorskip("torch")
        import torch

        from deepbiop.collate import multi_label_tensor_collate

        batch = [
            {"features": torch.tensor([1.0, 2.0]), "target": (30.0, 0.5)},
            {"features": torch.tensor([3.0, 4.0]), "target": (32.0, 0.6)},
        ]

        result = multi_label_tensor_collate(batch)

        # Check targets are stacked into single tensor
        assert isinstance(result["targets"], torch.Tensor)
        assert result["targets"].shape == (2, 2)
        assert torch.allclose(
            result["targets"], torch.tensor([[30.0, 0.5], [32.0, 0.6]])
        )

    def test_multi_label_tensor_collate_empty_batch(self):
        """Test multi_label_tensor_collate with empty batch."""
        pytest.importorskip("torch")
        from deepbiop.collate import multi_label_tensor_collate

        result = multi_label_tensor_collate([])
        assert result == {}


class TestStreamingFastqDataset:
    """Test StreamingFastqDataset for memory-efficient loading."""

    @pytest.fixture
    def sample_fastq_file(self):
        """Create a temporary FASTQ file for testing."""
        content = """@read1
ACGTACGT
+
IIIIIIII
@read2
CGTAGCTA
+
JJJJJJJJ
@read3
GTATCGAT
+
KKKKKKKK
@read4
TACGATCG
+
LLLLLLLL
@read5
ATCGATCG
+
MMMMMMMM
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fastq", delete=False) as f:
            f.write(content)
            temp_path = f.name

        yield temp_path

        # Cleanup
        Path(temp_path).unlink()

    def test_streaming_dataset_basic_iteration(self, sample_fastq_file):
        """Test basic iteration through streaming dataset."""
        from deepbiop.fq import StreamingFastqDataset

        dataset = StreamingFastqDataset(sample_fastq_file)

        records = list(dataset)

        # Check we got all records
        assert len(records) == 5

        # Check first record structure
        assert "id" in records[0]
        assert "sequence" in records[0]
        assert "quality" in records[0]

        # Check first record content
        assert bytes(records[0]["id"]) == b"read1"
        assert bytes(records[0]["sequence"]) == b"ACGTACGT"
        assert bytes(records[0]["quality"]) == b"IIIIIIII"

    def test_streaming_dataset_with_shuffling(self, sample_fastq_file):
        """Test streaming dataset with shuffle buffer."""
        from deepbiop.fq import StreamingFastqDataset

        # Use small shuffle buffer
        dataset = StreamingFastqDataset(sample_fastq_file, shuffle_buffer_size=3)

        records = list(dataset)

        # Should still have all 5 records
        assert len(records) == 5

        # All record IDs should be present (order may differ)
        ids = [bytes(r["id"]) for r in records]
        expected_ids = {b"read1", b"read2", b"read3", b"read4", b"read5"}
        assert set(ids) == expected_ids

    def test_streaming_dataset_multiple_iterations(self, sample_fastq_file):
        """Test multiple iterations through same dataset."""
        from deepbiop.fq import StreamingFastqDataset

        dataset = StreamingFastqDataset(sample_fastq_file)

        # First iteration
        records1 = list(dataset)
        assert len(records1) == 5

        # Second iteration (creates new iterator)
        records2 = list(dataset)
        assert len(records2) == 5

        # Should get same records in same order (no shuffling)
        for r1, r2 in zip(records1, records2, strict=False):
            assert bytes(r1["id"]) == bytes(r2["id"])

    def test_streaming_dataset_file_not_found(self):
        """Test error handling for non-existent file."""
        from deepbiop.fq import StreamingFastqDataset

        with pytest.raises(FileNotFoundError, match="FASTQ file not found"):
            StreamingFastqDataset("nonexistent.fastq")

    def test_streaming_dataset_repr(self, sample_fastq_file):
        """Test string representation of dataset."""
        from deepbiop.fq import StreamingFastqDataset

        dataset = StreamingFastqDataset(sample_fastq_file, shuffle_buffer_size=1000)

        repr_str = repr(dataset)
        assert "StreamingFastqDataset" in repr_str
        assert "1000" in repr_str

    def test_streaming_dataset_no_shuffling(self, sample_fastq_file):
        """Test dataset with shuffle_buffer_size=0 (no shuffling)."""
        from deepbiop.fq import StreamingFastqDataset

        dataset = StreamingFastqDataset(sample_fastq_file, shuffle_buffer_size=0)

        records = list(dataset)

        # Should maintain original order
        assert bytes(records[0]["id"]) == b"read1"
        assert bytes(records[1]["id"]) == b"read2"
        assert bytes(records[2]["id"]) == b"read3"
        assert bytes(records[3]["id"]) == b"read4"
        assert bytes(records[4]["id"]) == b"read5"

    def test_streaming_dataset_iteration_protocol(self, sample_fastq_file):
        """Test that dataset follows Python iteration protocol."""
        from deepbiop.fq import StreamingFastqDataset

        dataset = StreamingFastqDataset(sample_fastq_file)

        # Test __iter__ returns iterator
        iterator = iter(dataset)
        assert iterator is not None

        # Test __next__ works
        first_record = next(iterator)
        assert bytes(first_record["id"]) == b"read1"

        second_record = next(iterator)
        assert bytes(second_record["id"]) == b"read2"

        # Consume rest
        remaining = list(iterator)
        assert len(remaining) == 3


class TestIntegration:
    """Integration tests combining multiple new features."""

    @pytest.fixture
    def sample_fastq_file(self):
        """Create a temporary FASTQ file for testing."""
        content = """@read1
ACGTACGT
+
IIIIIIII
@read2
CGTAGCTA
+
JJJJJJJJ
@read3
GTATCGAT
+
KKKKKKKK
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fastq", delete=False) as f:
            f.write(content)
            temp_path = f.name

        yield temp_path
        Path(temp_path).unlink()

    def test_streaming_with_multilabel_extraction(self, sample_fastq_file):
        """Test streaming dataset with multi-label extraction."""
        from deepbiop.fq import StreamingFastqDataset
        from deepbiop.targets import MultiLabelExtractor, TargetExtractor
        from deepbiop.transforms import TransformDataset

        # Create streaming dataset
        base_dataset = StreamingFastqDataset(sample_fastq_file)

        # Create multi-label extractor
        extractor = MultiLabelExtractor(
            {
                "quality": TargetExtractor.from_quality(stat="mean"),
                "gc": TargetExtractor.from_sequence(feature="gc_content"),
                "length": TargetExtractor.from_sequence(feature="length"),
            }
        )

        # Wrap with target extraction
        dataset = TransformDataset(base_dataset, target_fn=extractor)

        # Iterate and check
        records = list(dataset)
        assert len(records) == 3

        # Check multi-label targets
        for record in records:
            assert "target" in record
            assert isinstance(record["target"], dict)
            assert set(record["target"].keys()) == {"quality", "gc", "length"}

    def test_streaming_with_multilabel_collate(self, sample_fastq_file):
        """Test full pipeline: streaming + multi-label + collate."""
        from deepbiop.collate import multi_label_collate
        from deepbiop.fq import StreamingFastqDataset
        from deepbiop.targets import MultiLabelExtractor, TargetExtractor
        from deepbiop.transforms import TransformDataset

        # Create pipeline
        base_dataset = StreamingFastqDataset(sample_fastq_file)

        extractor = MultiLabelExtractor(
            {
                "quality": TargetExtractor.from_quality(stat="mean"),
                "gc": TargetExtractor.from_sequence(feature="gc_content"),
            }
        )

        dataset = TransformDataset(base_dataset, target_fn=extractor)

        # Collect batch
        batch = list(dataset)
        assert len(batch) == 3

        # Collate
        collated = multi_label_collate(batch)

        # Verify structure
        assert "targets" in collated
        assert isinstance(collated["targets"], dict)
        assert set(collated["targets"].keys()) == {"quality", "gc"}
        assert len(collated["targets"]["quality"]) == 3
        assert len(collated["targets"]["gc"]) == 3
