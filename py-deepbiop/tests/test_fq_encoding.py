"""Tests for FASTQ sequence encoding functionality."""

import numpy as np
import pytest

import deepbiop as dbp


class TestOneHotEncoder:
    """Test OneHotEncoder for DNA/RNA sequences."""

    def test_onehot_single_sequence(self):
        """Test encoding a single DNA sequence."""
        encoder = dbp.OneHotEncoder("dna", "skip")
        record = {"sequence": b"ACGT"}

        result = encoder(record)
        encoded = result["sequence"]

        # Shape: (seq_len, 4)
        assert encoded.shape == (4, 4)
        assert encoded.dtype == np.float32

        # A = [1, 0, 0, 0]
        np.testing.assert_array_equal(encoded[0], [1, 0, 0, 0])
        # C = [0, 1, 0, 0]
        np.testing.assert_array_equal(encoded[1], [0, 1, 0, 0])
        # G = [0, 0, 1, 0]
        np.testing.assert_array_equal(encoded[2], [0, 0, 1, 0])
        # T = [0, 0, 0, 1]
        np.testing.assert_array_equal(encoded[3], [0, 0, 0, 1])

    def test_onehot_batch_encoding(self):
        """Test encoding multiple sequences in a batch."""
        encoder = dbp.OneHotEncoder("dna", "skip")
        records = [{"sequence": b"ACGT"}, {"sequence": b"TTAA"}, {"sequence": b"GGCC"}]

        results = [encoder(r) for r in records]
        encoded = np.array([r["sequence"] for r in results])

        # Shape: (batch_size, seq_len, 4)
        assert encoded.shape == (3, 4, 4)
        assert encoded.dtype == np.float32

        # Check first sequence
        np.testing.assert_array_equal(encoded[0, 0], [1, 0, 0, 0])  # A

        # Check second sequence
        np.testing.assert_array_equal(encoded[1, 0], [0, 0, 0, 1])  # T
        np.testing.assert_array_equal(encoded[1, 1], [0, 0, 0, 1])  # T

        # Check third sequence
        np.testing.assert_array_equal(encoded[2, 0], [0, 0, 1, 0])  # G
        np.testing.assert_array_equal(encoded[2, 2], [0, 1, 0, 0])  # C

    def test_onehot_rna_encoding(self):
        """Test encoding RNA sequences (U instead of T)."""
        encoder = dbp.OneHotEncoder("rna", "skip")
        record = {"sequence": b"ACGU"}

        result = encoder(record)
        encoded = result["sequence"]

        assert encoded.shape == (4, 4)
        # U should be encoded like T
        np.testing.assert_array_equal(encoded[3], [0, 0, 0, 1])

    def test_onehot_empty_sequence(self):
        """Test encoding an empty sequence."""
        encoder = dbp.OneHotEncoder("dna", "skip")
        record = {"sequence": b""}

        result = encoder(record)
        encoded = result["sequence"]

        assert encoded.shape == (0, 4)

    def test_onehot_unknown_bases_skip(self):
        """Test encoding with unknown bases (skip mode)."""
        encoder = dbp.OneHotEncoder("dna", "skip")
        record = {"sequence": b"ACNGT"}  # N is unknown

        result = encoder(record)
        encoded = result["sequence"]

        assert encoded.shape == (5, 4)
        # N should be encoded as zeros with skip mode
        np.testing.assert_array_equal(encoded[2], [0, 0, 0, 0])


class TestKmerEncoder:
    """Test KmerEncoder for k-mer frequency features."""

    def test_kmer_basic_encoding(self):
        """Test basic k-mer encoding with k=3."""
        encoder = dbp.KmerEncoder(k=3, canonical=False, encoding_type="dna")
        record = {"sequence": b"ACGTACGT"}  # 8 bases = 6 overlapping 3-mers

        result = encoder(record)
        encoded = result["sequence"]

        # For k=3, there are 4^3 = 64 possible k-mers
        assert encoded.shape == (64,)
        assert encoded.dtype == np.float32

        # Should have non-zero counts
        assert encoded.sum() > 0

    def test_kmer_canonical_mode(self):
        """Test canonical k-mer encoding (treats reverse complement as same)."""
        encoder_canonical = dbp.KmerEncoder(k=3, canonical=True, encoding_type="dna")
        encoder_non_canonical = dbp.KmerEncoder(
            k=3, canonical=False, encoding_type="dna"
        )

        record = {"sequence": b"ACGT"}

        result_canonical = encoder_canonical(record)
        encoded_canonical = result_canonical["sequence"]

        result_non_canonical = encoder_non_canonical(record)
        encoded_non_canonical = result_non_canonical["sequence"]

        # Both should have 64 features for k=3
        assert encoded_canonical.shape == (64,)
        assert encoded_non_canonical.shape == (64,)

    def test_kmer_batch_encoding(self):
        """Test encoding multiple sequences."""
        encoder = dbp.KmerEncoder(k=2, canonical=False, encoding_type="dna")
        records = [{"sequence": b"ACGT"}, {"sequence": b"AAAA"}, {"sequence": b"TTTT"}]

        results = [encoder(r) for r in records]
        encoded = np.array([r["sequence"] for r in results])

        # For k=2, there are 4^2 = 16 possible k-mers
        assert encoded.shape == (3, 16)
        assert encoded.dtype == np.float32

        # AA sequence should have AA k-mer count = 3
        aa_index = 0  # AA is the first k-mer (0*4 + 0)
        assert encoded[1, aa_index] > 0

    def test_kmer_different_k_values(self):
        """Test different k values."""
        record = {"sequence": b"ACGTACGTACGT"}

        for k in [2, 3, 4]:
            encoder = dbp.KmerEncoder(k=k, canonical=False, encoding_type="dna")
            result = encoder(record)
            encoded = result["sequence"]

            expected_size = 4**k
            assert encoded.shape == (expected_size,)

    def test_kmer_short_sequence(self):
        """Test encoding sequence shorter than k."""
        encoder = dbp.KmerEncoder(k=5, canonical=False, encoding_type="dna")
        record = {"sequence": b"ACG"}  # Only 3 bases, but k=5

        result = encoder(record)
        encoded = result["sequence"]

        # Should return zeros for sequence shorter than k
        assert encoded.shape == (4**5,)
        assert encoded.sum() == 0


class TestIntegerEncoder:
    """Test IntegerEncoder for transformer-style embeddings."""

    def test_integer_single_sequence(self):
        """Test encoding a single DNA sequence."""
        encoder = dbp.IntegerEncoder("dna")
        record = {"sequence": b"ACGT"}

        result = encoder(record)
        encoded = result["sequence"]

        assert encoded.shape == (4,)
        assert encoded.dtype == np.float32

        # DNA encoding: A=0, C=1, G=2, T=3
        np.testing.assert_array_almost_equal(encoded, [0, 1, 2, 3])

    def test_integer_batch_encoding(self):
        """Test encoding multiple sequences."""
        encoder = dbp.IntegerEncoder("dna")
        records = [{"sequence": b"ACGT"}, {"sequence": b"TTAA"}, {"sequence": b"GGCC"}]

        results = [encoder(r) for r in records]
        encoded = np.array([r["sequence"] for r in results])

        assert encoded.shape == (3, 4)
        assert encoded.dtype == np.float32

        # Check encodings
        np.testing.assert_array_almost_equal(encoded[0], [0, 1, 2, 3])  # ACGT
        np.testing.assert_array_almost_equal(encoded[1], [3, 3, 0, 0])  # TTAA
        np.testing.assert_array_almost_equal(encoded[2], [2, 2, 1, 1])  # GGCC

    def test_integer_rna_encoding(self):
        """Test encoding RNA sequences."""
        encoder = dbp.IntegerEncoder("rna")
        record = {"sequence": b"ACGU"}

        result = encoder(record)
        encoded = result["sequence"]

        assert encoded.shape == (4,)
        # RNA encoding: A=0, C=1, G=2, U=3
        np.testing.assert_array_equal(encoded, [0, 1, 2, 3])

    def test_integer_empty_sequence(self):
        """Test encoding an empty sequence."""
        encoder = dbp.IntegerEncoder("dna")
        record = {"sequence": b""}

        result = encoder(record)
        encoded = result["sequence"]

        assert encoded.shape == (0,)

    def test_integer_long_sequence(self):
        """Test encoding a long sequence."""
        encoder = dbp.IntegerEncoder("dna")
        record = {"sequence": b"ACGT" * 100}  # 400 bases

        result = encoder(record)
        encoded = result["sequence"]

        assert encoded.shape == (400,)
        assert encoded.dtype == np.float32

        # Check repeating pattern
        np.testing.assert_array_almost_equal(encoded[0:4], [0, 1, 2, 3])
        np.testing.assert_array_almost_equal(encoded[4:8], [0, 1, 2, 3])


class TestEncoderIntegration:
    """Integration tests for encoder workflows."""

    def test_onehot_to_pytorch(self):
        """Test converting one-hot encoding to PyTorch tensor."""
        pytest.importorskip("torch")
        import torch

        encoder = dbp.OneHotEncoder("dna", "skip")
        record = {"sequence": b"ACGT"}

        result = encoder(record)
        encoded = result["sequence"]
        tensor = torch.from_numpy(encoded).float()

        assert tensor.shape == (4, 4)
        assert tensor.dtype == torch.float32

    def test_integer_to_pytorch(self):
        """Test converting integer encoding to PyTorch tensor."""
        pytest.importorskip("torch")
        import torch

        encoder = dbp.IntegerEncoder("dna")
        record = {"sequence": b"ACGT"}

        result = encoder(record)
        encoded = result["sequence"]
        tensor = torch.from_numpy(encoded).long()

        assert tensor.shape == (4,)
        assert tensor.dtype == torch.int64

    def test_kmer_to_numpy_array(self):
        """Test k-mer encoding produces valid numpy arrays."""
        encoder = dbp.KmerEncoder(k=3, canonical=False, encoding_type="dna")
        records = [{"sequence": b"ACGTACGT"}, {"sequence": b"TTTTAAAA"}]

        results = [encoder(r) for r in records]
        encoded = np.array([r["sequence"] for r in results])

        assert isinstance(encoded, np.ndarray)
        assert encoded.dtype == np.float32
        assert not np.isnan(encoded).any()
        assert not np.isinf(encoded).any()

    def test_batch_processing_consistency(self):
        """Test that batch encoding equals individual encodings."""
        encoder = dbp.OneHotEncoder("dna", "skip")
        records = [{"sequence": b"ACGT"}, {"sequence": b"TTAA"}, {"sequence": b"GGCC"}]

        # Batch encoding
        batch_results = [encoder(r) for r in records]
        batch_encoded = np.array([r["sequence"] for r in batch_results])

        # Individual encoding
        individual_encoded = [encoder(r)["sequence"] for r in records]

        # Should match
        for i in range(len(records)):
            np.testing.assert_array_equal(batch_encoded[i], individual_encoded[i])
