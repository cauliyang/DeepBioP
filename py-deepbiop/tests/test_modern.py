"""Tests for modern PyTorch-compatible API."""

from __future__ import annotations

import numpy as np

from deepbiop.modern import (
    DNA_ALPHABET_SIZE,
    PROTEIN_ALPHABET_SIZE,
    IntegerEncoder,
    OneHotEncoder,
    batch_sequences,
    create_attention_mask,
    create_padding_mask,
    pool_sequences,
    reverse_complement,
    to_channels_first,
    to_channels_last,
    unbatch_sequences,
)


class TestOneHotEncoder:
    """Test modern OneHotEncoder wrapper."""

    def test_encode_single_dna(self):
        """Test single DNA sequence encoding."""
        encoder = OneHotEncoder("dna")
        sequence = b"ACGT"
        encoded = encoder.encode(sequence)

        assert encoded.shape == (4, 4)
        assert encoded.dtype == np.float32
        # A should be [1,0,0,0], C=[0,1,0,0], G=[0,0,1,0], T=[0,0,0,1]
        np.testing.assert_array_equal(encoded[0], [1, 0, 0, 0])
        np.testing.assert_array_equal(encoded[1], [0, 1, 0, 0])
        np.testing.assert_array_equal(encoded[2], [0, 0, 1, 0])
        np.testing.assert_array_equal(encoded[3], [0, 0, 0, 1])

    def test_encode_single_protein(self):
        """Test single protein sequence encoding."""
        encoder = OneHotEncoder("protein")
        sequence = b"ACDE"
        encoded = encoder.encode(sequence)

        assert encoded.shape == (4, 20)
        assert encoded.dtype == np.float32

    def test_encode_batch_with_padding(self):
        """Test batch encoding with automatic padding."""
        encoder = OneHotEncoder("dna")
        sequences = [b"ACG", b"TGCA", b"AT"]
        batch = encoder.encode_batch(sequences)

        # Should pad to max length (4)
        assert batch.shape == (3, 4, 4)
        assert batch.dtype == np.float32

        # First sequence: ACG + padding
        assert np.sum(batch[0, :3]) == 3  # 3 valid positions
        assert np.sum(batch[0, 3]) == 0  # padding position

        # Second sequence: TGCA (full length)
        assert np.sum(batch[1]) == 4  # all valid

        # Third sequence: AT + padding
        assert np.sum(batch[2, :2]) == 2  # 2 valid positions
        assert np.sum(batch[2, 2:]) == 0  # padding positions

    def test_for_conv1d_rearrangement(self):
        """Test rearrangement for Conv1d architecture."""
        encoder = OneHotEncoder("dna")
        sequences = [b"ACGT", b"GCTA"]
        batch = encoder.encode_batch(sequences)

        # Original shape: [batch, length, alphabet]
        assert batch.shape == (2, 4, 4)

        # Rearranged: [batch, alphabet, length]
        conv_input = encoder.for_conv1d(batch)
        assert conv_input.shape == (2, 4, 4)

        # Verify data is correctly rearranged
        # batch[0, 1, 2] should equal conv_input[0, 2, 1]
        assert batch[0, 1, 2] == conv_input[0, 2, 1]

    def test_for_transformer_identity(self):
        """Test transformer format is identity operation."""
        encoder = OneHotEncoder("dna")
        sequences = [b"ACGT"]
        batch = encoder.encode_batch(sequences)

        transformer_input = encoder.for_transformer(batch)
        np.testing.assert_array_equal(batch, transformer_input)

    def test_for_rnn_with_lengths(self):
        """Test RNN preparation with sequence lengths."""
        encoder = OneHotEncoder("dna")
        sequences = [b"ACG", b"TGCA", b"AT"]
        batch = encoder.encode_batch(sequences)

        rnn_batch, lengths = encoder.for_rnn(batch)

        # Batch should be unchanged
        np.testing.assert_array_equal(rnn_batch, batch)

        # Lengths should be computed correctly
        expected_lengths = np.array([3, 4, 2], dtype=np.int64)
        np.testing.assert_array_equal(lengths, expected_lengths)

    def test_alphabet_size(self):
        """Test alphabet size properties."""
        dna_encoder = OneHotEncoder("dna")
        assert dna_encoder.alphabet_size == 4

        rna_encoder = OneHotEncoder("rna")
        assert rna_encoder.alphabet_size == 4

        protein_encoder = OneHotEncoder("protein")
        assert protein_encoder.alphabet_size == 20

    def test_repr(self):
        """Test string representation."""
        encoder = OneHotEncoder("dna", ambiguous_strategy="mask")
        repr_str = repr(encoder)

        assert "ModernOneHotEncoder" in repr_str
        assert "dna" in repr_str
        assert "alphabet_size=4" in repr_str
        assert "mask" in repr_str


class TestIntegerEncoder:
    """Test modern IntegerEncoder wrapper."""

    def test_encode_single_dna(self):
        """Test single DNA sequence encoding."""
        encoder = IntegerEncoder("dna")
        sequence = b"ACGT"
        encoded = encoder.encode(sequence)

        assert encoded.shape == (4,)
        assert encoded.dtype == np.float32
        # A=0, C=1, G=2, T=3
        np.testing.assert_array_equal(encoded, [0, 1, 2, 3])

    def test_encode_batch_with_padding(self):
        """Test batch encoding with padding."""
        encoder = IntegerEncoder("dna")
        sequences = [b"ACG", b"TGCA"]
        batch = encoder.encode_batch(sequences, pad_value=-1.0)

        # Should pad to max length (4)
        assert batch.shape == (2, 4)
        assert batch.dtype == np.float32

        # First sequence: ACG + padding
        expected_first = np.array([0, 1, 2, -1], dtype=np.float32)
        np.testing.assert_array_equal(batch[0], expected_first)

        # Second sequence: TGCA (full length)
        expected_second = np.array([3, 2, 1, 0], dtype=np.float32)
        np.testing.assert_array_equal(batch[1], expected_second)

    def test_repr(self):
        """Test string representation."""
        encoder = IntegerEncoder("protein")
        repr_str = repr(encoder)

        assert "ModernIntegerEncoder" in repr_str
        assert "protein" in repr_str


class TestTransforms:
    """Test tensor transformation utilities."""

    def test_to_channels_first(self):
        """Test channels-last to channels-first conversion."""
        # [batch=2, length=3, features=4]
        batch = np.random.randn(2, 3, 4).astype(np.float32)

        result = to_channels_first(batch)

        # Should be [batch=2, features=4, length=3]
        assert result.shape == (2, 4, 3)

        # Verify data integrity
        assert batch[0, 1, 2] == result[0, 2, 1]
        assert batch[1, 0, 3] == result[1, 3, 0]

    def test_to_channels_last(self):
        """Test channels-first to channels-last conversion."""
        # [batch=2, channels=4, length=3]
        batch = np.random.randn(2, 4, 3).astype(np.float32)

        result = to_channels_last(batch)

        # Should be [batch=2, length=3, channels=4]
        assert result.shape == (2, 3, 4)

        # Verify data integrity
        assert batch[0, 2, 1] == result[0, 1, 2]

    def test_round_trip_conversion(self):
        """Test channels_first <-> channels_last round trip."""
        original = np.random.randn(2, 5, 4).astype(np.float32)

        # channels_last -> channels_first -> channels_last
        converted = to_channels_first(original)
        restored = to_channels_last(converted)

        np.testing.assert_array_equal(original, restored)

    def test_pool_sequences_mean(self):
        """Test mean pooling over sequences."""
        # [batch=2, length=3, features=4]
        batch = np.array(
            [
                [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                [[2, 4, 6, 8], [10, 12, 14, 16], [18, 20, 22, 24]],
            ],
            dtype=np.float32,
        )

        pooled = pool_sequences(batch, method="mean")

        # Should be [batch=2, features=4]
        assert pooled.shape == (2, 4)

        # Verify mean calculation
        expected_first = np.array([5, 6, 7, 8], dtype=np.float32)
        np.testing.assert_array_almost_equal(pooled[0], expected_first)

    def test_pool_sequences_max(self):
        """Test max pooling over sequences."""
        batch = np.array(
            [[[1, 2], [5, 6], [3, 4]], [[10, 20], [5, 15], [8, 12]]], dtype=np.float32
        )

        pooled = pool_sequences(batch, method="max")

        # Should take max along sequence dimension
        expected = np.array([[5, 6], [10, 20]], dtype=np.float32)
        np.testing.assert_array_equal(pooled, expected)

    def test_pool_sequences_sum(self):
        """Test sum pooling over sequences."""
        batch = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)

        pooled = pool_sequences(batch, method="sum")

        expected = np.array([[4, 6], [12, 14]], dtype=np.float32)
        np.testing.assert_array_equal(pooled, expected)


class TestMasks:
    """Test mask creation utilities."""

    def test_create_padding_mask(self):
        """Test padding mask creation from lengths."""
        lengths = np.array([3, 5, 2])
        mask = create_padding_mask(lengths, max_length=6)

        assert mask.shape == (3, 6)
        assert mask.dtype == bool

        # First sequence (length=3): [1,1,1,0,0,0]
        expected_first = np.array([1, 1, 1, 0, 0, 0], dtype=bool)
        np.testing.assert_array_equal(mask[0], expected_first)

        # Second sequence (length=5): [1,1,1,1,1,0]
        expected_second = np.array([1, 1, 1, 1, 1, 0], dtype=bool)
        np.testing.assert_array_equal(mask[1], expected_second)

        # Third sequence (length=2): [1,1,0,0,0,0]
        expected_third = np.array([1, 1, 0, 0, 0, 0], dtype=bool)
        np.testing.assert_array_equal(mask[2], expected_third)

    def test_create_padding_mask_auto_max_length(self):
        """Test automatic max_length calculation."""
        lengths = np.array([3, 7, 2])
        mask = create_padding_mask(lengths)

        # Should use max(lengths) = 7
        assert mask.shape == (3, 7)

    def test_create_attention_mask_padding_only(self):
        """Test non-causal attention mask (padding only)."""
        lengths = np.array([3, 2])
        mask = create_attention_mask(lengths, max_length=4, causal=False)

        # Should be same as padding mask
        assert mask.shape == (2, 4)

        expected_first = np.array([1, 1, 1, 0], dtype=bool)
        np.testing.assert_array_equal(mask[0], expected_first)

    def test_create_attention_mask_causal(self):
        """Test causal attention mask."""
        lengths = np.array([3, 2])
        mask = create_attention_mask(lengths, max_length=3, causal=True)

        # Should be [batch, max_length, max_length]
        assert mask.shape == (2, 3, 3)

        # First sequence (length=3), position 0: can only attend to position 0
        expected_0_0 = np.array([1, 0, 0], dtype=bool)
        np.testing.assert_array_equal(mask[0, 0], expected_0_0)

        # First sequence, position 2: can attend to positions 0, 1, 2
        expected_0_2 = np.array([1, 1, 1], dtype=bool)
        np.testing.assert_array_equal(mask[0, 2], expected_0_2)

        # Second sequence (length=2), position 1: can attend to positions 0, 1 only
        expected_1_1 = np.array([1, 1, 0], dtype=bool)
        np.testing.assert_array_equal(mask[1, 1], expected_1_1)


class TestBatching:
    """Test batch/unbatch utilities."""

    def test_batch_sequences_left_align(self):
        """Test left-aligned batching."""
        seq1 = np.array([[1, 2], [3, 4]])  # length=2
        seq2 = np.array([[5, 6], [7, 8], [9, 10]])  # length=3

        batch, lengths = batch_sequences([seq1, seq2], align="left")

        assert batch.shape == (2, 3, 2)
        np.testing.assert_array_equal(lengths, [2, 3])

        # First sequence padded on right
        np.testing.assert_array_equal(batch[0, :2], seq1)
        np.testing.assert_array_equal(batch[0, 2], [0, 0])

        # Second sequence (full)
        np.testing.assert_array_equal(batch[1], seq2)

    def test_batch_sequences_right_align(self):
        """Test right-aligned batching."""
        seq1 = np.array([[1, 2], [3, 4]])  # length=2
        seq2 = np.array([[5, 6], [7, 8], [9, 10]])  # length=3

        batch, _lengths = batch_sequences([seq1, seq2], align="right")

        assert batch.shape == (2, 3, 2)

        # First sequence padded on left
        np.testing.assert_array_equal(batch[0, 0], [0, 0])
        np.testing.assert_array_equal(batch[0, 1:], seq1)

    def test_batch_sequences_custom_pad_value(self):
        """Test custom padding value."""
        seq1 = np.array([[1, 2]])
        seq2 = np.array([[3, 4], [5, 6]])

        batch, _lengths = batch_sequences([seq1, seq2], pad_value=-1.0)

        # Check padding uses custom value
        np.testing.assert_array_equal(batch[0, 1], [-1, -1])

    def test_unbatch_sequences(self):
        """Test unbatching back to variable-length sequences."""
        batch = np.array(
            [[[1, 2], [3, 4], [0, 0]], [[5, 6], [0, 0], [0, 0]]], dtype=np.float32
        )
        lengths = np.array([2, 1])

        sequences = unbatch_sequences(batch, lengths)

        assert len(sequences) == 2

        expected_first = np.array([[1, 2], [3, 4]], dtype=np.float32)
        np.testing.assert_array_equal(sequences[0], expected_first)

        expected_second = np.array([[5, 6]], dtype=np.float32)
        np.testing.assert_array_equal(sequences[1], expected_second)

    def test_batch_unbatch_round_trip(self):
        """Test batch -> unbatch round trip."""
        original_sequences = [
            np.array([[1, 2], [3, 4]]),
            np.array([[5, 6], [7, 8], [9, 10]]),
            np.array([[11, 12]]),
        ]

        batch, lengths = batch_sequences(original_sequences)
        restored = unbatch_sequences(batch, lengths)

        assert len(restored) == len(original_sequences)
        for orig, rest in zip(original_sequences, restored, strict=False):
            np.testing.assert_array_equal(orig, rest)


class TestReverseComplement:
    """Test reverse complement transformation."""

    def test_reverse_complement_dna(self):
        """Test reverse complement for DNA sequence."""
        # Create one-hot for "ACGT"
        # A=[1,0,0,0], C=[0,1,0,0], G=[0,0,1,0], T=[0,0,0,1]
        sequence = np.array(
            [
                [1, 0, 0, 0],  # A
                [0, 1, 0, 0],  # C
                [0, 0, 1, 0],  # G
                [0, 0, 0, 1],  # T
            ],
            dtype=np.float32,
        )

        # Add batch dimension
        batch = sequence[np.newaxis, ...]

        rc = reverse_complement(batch)

        # Reverse complement of "ACGT" is "ACGT" (palindrome)
        # Reverse: T,G,C,A -> Complement: A,C,G,T
        # T->A=[1,0,0,0], G->C=[0,1,0,0], C->G=[0,0,1,0], A->T=[0,0,0,1]
        expected = np.array(
            [
                [1, 0, 0, 0],  # A (from T)
                [0, 1, 0, 0],  # C (from G)
                [0, 0, 1, 0],  # G (from C)
                [0, 0, 0, 1],  # T (from A)
            ],
            dtype=np.float32,
        )

        np.testing.assert_array_equal(rc[0], expected)

    def test_reverse_complement_batch(self):
        """Test reverse complement on batch."""
        # "AT" sequence: A=[1,0,0,0], T=[0,0,0,1]
        seq1 = np.array([[1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
        # "GC" sequence: G=[0,0,1,0], C=[0,1,0,0]
        seq2 = np.array([[0, 0, 1, 0], [0, 1, 0, 0]], dtype=np.float32)

        batch = np.stack([seq1, seq2])
        rc = reverse_complement(batch)

        # RC of "AT" = "AT" (T->A, A->T reversed = A,T)
        expected_seq1 = np.array([[1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
        np.testing.assert_array_equal(rc[0], expected_seq1)

        # RC of "GC" = "GC" (C->G, G->C reversed = G,C)
        expected_seq2 = np.array([[0, 0, 1, 0], [0, 1, 0, 0]], dtype=np.float32)
        np.testing.assert_array_equal(rc[1], expected_seq2)


class TestConstants:
    """Test module constants."""

    def test_alphabet_sizes(self):
        """Test alphabet size constants."""
        assert DNA_ALPHABET_SIZE == 4
        assert PROTEIN_ALPHABET_SIZE == 20

    def test_constants_match_encoders(self):
        """Test constants match encoder alphabet sizes."""
        dna_encoder = OneHotEncoder("dna")
        assert dna_encoder.alphabet_size == DNA_ALPHABET_SIZE

        protein_encoder = OneHotEncoder("protein")
        assert protein_encoder.alphabet_size == PROTEIN_ALPHABET_SIZE
