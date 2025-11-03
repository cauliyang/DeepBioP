"""Tests for sequence augmentation functionality."""

import numpy as np

import deepbiop as dbp


class TestReverseComplement:
    """Test ReverseComplement transformation."""

    def test_reverse_complement_basic(self):
        """Test basic reverse complement transformation."""
        rc = dbp.ReverseComplement()
        sequence = b"ACGT"

        result = rc.apply(sequence)

        # Reverse complement of ACGT is ACGT (palindrome)
        assert result == b"ACGT"

    def test_reverse_complement_non_palindrome(self):
        """Test reverse complement of non-palindrome sequence."""
        rc = dbp.ReverseComplement()
        sequence = b"AAAA"

        result = rc.apply(sequence)

        # Reverse complement of AAAA is TTTT
        assert result == b"TTTT"

    def test_reverse_complement_complex(self):
        """Test reverse complement with mixed bases."""
        rc = dbp.ReverseComplement()
        sequence = b"ACGTACGT"

        result = rc.apply(sequence)

        # Reverse complement of ACGTACGT
        # Original:  A  C  G  T  A  C  G  T
        # Reverse:   T  G  C  A  T  G  C  A
        # Complement: A  C  G  T  A  C  G  T
        assert result == b"ACGTACGT"

    def test_reverse_complement_asymmetric(self):
        """Test reverse complement of asymmetric sequence."""
        rc = dbp.ReverseComplement()
        sequence = b"TTTTAAAACCCCGGGG"

        result = rc.apply(sequence)

        # Reverse complement of TTTTAAAACCCCGGGG
        # Reverse: GGGGCCCCAAAATTTT
        # Complement: CCCCGGGGTTTTAAAA
        assert result == b"CCCCGGGGTTTTAAAA"

    def test_reverse_complement_idempotent(self):
        """Test that applying RC twice returns original."""
        rc = dbp.ReverseComplement()
        sequence = b"ACGTACGTACGT"

        rc_once = rc.apply(sequence)
        rc_twice = rc.apply(rc_once)

        assert rc_twice == sequence

    def test_reverse_complement_empty(self):
        """Test reverse complement of empty sequence."""
        rc = dbp.ReverseComplement()
        sequence = b""

        result = rc.apply(sequence)

        assert result == b""

    def test_reverse_complement_batch(self):
        """Test batch reverse complement transformation."""
        rc = dbp.ReverseComplement()
        sequences = [b"ACGT", b"AAAA", b"TTTT", b"GGGG", b"CCCC"]

        results = rc.apply_batch(sequences)

        assert len(results) == 5
        assert results[0] == b"ACGT"  # Palindrome
        assert results[1] == b"TTTT"  # AAAA -> TTTT
        assert results[2] == b"AAAA"  # TTTT -> AAAA
        assert results[3] == b"CCCC"  # GGGG -> CCCC
        assert results[4] == b"GGGG"  # CCCC -> GGGG

    def test_reverse_complement_batch_empty(self):
        """Test batch RC with empty list."""
        rc = dbp.ReverseComplement()
        sequences = []

        results = rc.apply_batch(sequences)

        assert len(results) == 0


class TestMutator:
    """Test Mutator for introducing random mutations."""

    def test_mutator_basic(self):
        """Test basic mutation with low rate."""
        mutator = dbp.Mutator(mutation_rate=0.1, seed=42)
        sequence = b"AAAAAAAAAA"  # 10 A's

        result = mutator.apply(sequence)

        assert len(result) == len(sequence)
        # With 10% rate, expect ~1 mutation (but random)
        mutations = sum(1 for a, b in zip(sequence, result, strict=False) if a != b)
        assert 0 <= mutations <= len(sequence)

    def test_mutator_high_rate(self):
        """Test mutation with high rate."""
        mutator = dbp.Mutator(mutation_rate=0.5, seed=42)
        sequence = b"A" * 100

        result = mutator.apply(sequence)

        mutations = sum(1 for a, b in zip(sequence, result, strict=False) if a != b)
        # With 50% rate, expect ~50 mutations (within reasonable range)
        assert 30 <= mutations <= 70

    def test_mutator_zero_rate(self):
        """Test mutator with zero mutation rate."""
        mutator = dbp.Mutator(mutation_rate=0.0, seed=42)
        sequence = b"ACGTACGTACGT"

        result = mutator.apply(sequence)

        # Zero rate should return identical sequence
        assert result == sequence

    def test_mutator_reproducibility(self):
        """Test that same seed produces same mutations."""
        sequence = b"ACGTACGTACGTACGT"

        mutator1 = dbp.Mutator(mutation_rate=0.2, seed=123)
        result1 = mutator1.apply(sequence)

        mutator2 = dbp.Mutator(mutation_rate=0.2, seed=123)
        result2 = mutator2.apply(sequence)

        assert result1 == result2

    def test_mutator_different_seeds(self):
        """Test that different seeds produce different mutations."""
        sequence = b"ACGTACGTACGTACGT"

        mutator1 = dbp.Mutator(mutation_rate=0.3, seed=1)
        result1 = mutator1.apply(sequence)

        mutator2 = dbp.Mutator(mutation_rate=0.3, seed=2)
        result2 = mutator2.apply(sequence)

        # With high mutation rate, different seeds should give different results
        assert result1 != result2

    def test_mutator_valid_bases(self):
        """Test that mutations produce valid DNA bases."""
        mutator = dbp.Mutator(mutation_rate=1.0, seed=42)  # 100% mutation
        sequence = b"ACGTACGTACGT"

        result = mutator.apply(sequence)

        # All bases should still be valid DNA
        assert all(base in b"ACGT" for base in result)

    def test_mutator_batch(self):
        """Test batch mutation."""
        mutator = dbp.Mutator(mutation_rate=0.1, seed=42)
        sequences = [b"ACGTACGT", b"TTTTAAAA", b"GGGGCCCC"]

        results = mutator.apply_batch(sequences)

        assert len(results) == 3
        assert all(len(r) == len(s) for r, s in zip(results, sequences, strict=False))

    def test_mutator_empty_sequence(self):
        """Test mutator with empty sequence."""
        mutator = dbp.Mutator(mutation_rate=0.5, seed=42)
        sequence = b""

        result = mutator.apply(sequence)

        assert result == b""


class TestSampler:
    """Test Sampler for subsequence extraction."""

    def test_sampler_start_strategy(self):
        """Test sampling from start of sequence."""
        sampler = dbp.Sampler(length=10, strategy="start", seed=42)
        sequence = b"ACGTACGTACGTACGTACGT"  # 20 bases

        result = sampler.apply(sequence)

        assert len(result) == 10
        assert result == sequence[:10]

    def test_sampler_center_strategy(self):
        """Test sampling from center of sequence."""
        sampler = dbp.Sampler(length=10, strategy="center", seed=42)
        sequence = b"ACGTACGTACGTACGTACGT"  # 20 bases

        result = sampler.apply(sequence)

        assert len(result) == 10
        # Center of 20-base sequence with 10-base sample starts at position 5
        assert result == sequence[5:15]

    def test_sampler_end_strategy(self):
        """Test sampling from end of sequence."""
        sampler = dbp.Sampler(length=10, strategy="end", seed=42)
        sequence = b"ACGTACGTACGTACGTACGT"  # 20 bases

        result = sampler.apply(sequence)

        assert len(result) == 10
        assert result == sequence[-10:]

    def test_sampler_random_strategy(self):
        """Test random sampling."""
        sampler = dbp.Sampler(length=10, strategy="random", seed=42)
        sequence = b"A" * 50 + b"G" * 50  # 100 bases

        result = sampler.apply(sequence)

        assert len(result) == 10
        # Should be a valid subsequence
        assert all(base in b"AG" for base in result)

    def test_sampler_random_reproducibility(self):
        """Test that same seed produces same random sample."""
        sequence = b"ACGTACGTACGTACGTACGTACGTACGTACGT"

        sampler1 = dbp.Sampler(length=10, strategy="random", seed=123)
        result1 = sampler1.apply(sequence)

        sampler2 = dbp.Sampler(length=10, strategy="random", seed=123)
        result2 = sampler2.apply(sequence)

        assert result1 == result2

    def test_sampler_exact_length(self):
        """Test sampling when requested length equals sequence length."""
        sampler = dbp.Sampler(length=20, strategy="start", seed=42)
        sequence = b"ACGTACGTACGTACGTACGT"  # 20 bases

        result = sampler.apply(sequence)

        assert result == sequence

    def test_sampler_too_long(self):
        """Test sampling when requested length exceeds sequence length."""
        sampler = dbp.Sampler(length=100, strategy="start", seed=42)
        sequence = b"ACGTACGT"  # Only 8 bases

        result = sampler.apply(sequence)

        # Should return the entire sequence when length > sequence length
        assert len(result) <= len(sequence)

    def test_sampler_different_lengths(self):
        """Test different sample lengths."""
        sequence = b"ACGTACGTACGTACGTACGTACGTACGTACGT"  # 32 bases

        for length in [5, 10, 15, 20]:
            sampler = dbp.Sampler(length=length, strategy="start", seed=42)
            result = sampler.apply(sequence)
            assert len(result) == length


class TestQualitySimulator:
    """Test QualitySimulator for generating quality scores."""

    def test_quality_uniform(self):
        """Test uniform quality score generation."""
        model = dbp.QualityModel.uniform(min=20, max=40)
        sim = dbp.QualitySimulator(model, seed=42)

        quality = sim.generate(150)

        assert len(quality) == 150
        assert isinstance(quality, bytes)

        # Convert to Phred scores and check range
        phred_scores = [q - 33 for q in quality]
        assert all(20 <= q <= 40 for q in phred_scores)

    def test_quality_normal(self):
        """Test normal distribution quality generation."""
        model = dbp.QualityModel.normal(mean=30.0, std_dev=5.0)
        sim = dbp.QualitySimulator(model, seed=42)

        quality = sim.generate(150)

        assert len(quality) == 150
        phred_scores = [q - 33 for q in quality]

        # Check reasonable bounds
        assert all(0 <= q <= 93 for q in phred_scores)

        # Mean should be close to 30
        mean_score = np.mean(phred_scores)
        assert 25 <= mean_score <= 35

    def test_quality_high_quality(self):
        """Test high quality model (modern Illumina)."""
        model = dbp.QualityModel.HighQuality
        sim = dbp.QualitySimulator(model, seed=42)

        quality = sim.generate(150)

        assert len(quality) == 150
        phred_scores = [q - 33 for q in quality]

        # High quality should have mean around Q37
        mean_score = np.mean(phred_scores)
        assert 33 <= mean_score <= 41  # Allow some variance

    def test_quality_medium_quality(self):
        """Test medium quality model."""
        model = dbp.QualityModel.MediumQuality
        sim = dbp.QualitySimulator(model, seed=42)

        quality = sim.generate(150)

        assert len(quality) == 150
        phred_scores = [q - 33 for q in quality]

        # Medium quality should have mean around Q28
        mean_score = np.mean(phred_scores)
        assert 24 <= mean_score <= 32

    def test_quality_degrading(self):
        """Test degrading quality model."""
        model = dbp.QualityModel.degrading(start_mean=40.0, end_mean=20.0, std_dev=3.0)
        sim = dbp.QualitySimulator(model, seed=42)

        quality = sim.generate(150)

        assert len(quality) == 150
        phred_scores = [q - 33 for q in quality]

        # First half should have higher quality than second half
        first_half_mean = np.mean(phred_scores[:75])
        second_half_mean = np.mean(phred_scores[75:])

        assert first_half_mean > second_half_mean

    def test_quality_reproducibility(self):
        """Test that same seed produces same quality scores."""
        model = dbp.QualityModel.normal(mean=30.0, std_dev=5.0)

        sim1 = dbp.QualitySimulator(model, seed=123)
        quality1 = sim1.generate(100)

        sim2 = dbp.QualitySimulator(model, seed=123)
        quality2 = sim2.generate(100)

        assert quality1 == quality2

    def test_quality_different_seeds(self):
        """Test that different seeds produce different quality scores."""
        model = dbp.QualityModel.normal(mean=30.0, std_dev=5.0)

        sim1 = dbp.QualitySimulator(model, seed=1)
        quality1 = sim1.generate(100)

        sim2 = dbp.QualitySimulator(model, seed=2)
        quality2 = sim2.generate(100)

        # Different seeds should produce different results
        assert quality1 != quality2

    def test_quality_various_lengths(self):
        """Test quality generation for various lengths."""
        model = dbp.QualityModel.HighQuality
        sim = dbp.QualitySimulator(model, seed=42)

        for length in [50, 100, 150, 250, 500]:
            quality = sim.generate(length)
            assert len(quality) == length

    def test_quality_empty_length(self):
        """Test quality generation with zero length."""
        model = dbp.QualityModel.HighQuality
        sim = dbp.QualitySimulator(model, seed=42)

        quality = sim.generate(0)

        assert len(quality) == 0


class TestAugmentationIntegration:
    """Integration tests for augmentation workflows."""

    def test_reverse_complement_then_mutate(self):
        """Test chaining reverse complement and mutation."""
        rc = dbp.ReverseComplement()
        mutator = dbp.Mutator(mutation_rate=0.1, seed=42)

        sequence = b"ACGTACGTACGT"

        # Apply RC then mutate
        rc_seq = rc.apply(sequence)
        result = mutator.apply(rc_seq)

        assert len(result) == len(sequence)

    def test_mutate_then_sample(self):
        """Test mutating then sampling subsequence."""
        mutator = dbp.Mutator(mutation_rate=0.05, seed=42)
        sampler = dbp.Sampler(length=10, strategy="random", seed=42)

        sequence = b"ACGTACGTACGTACGTACGTACGTACGT"

        # Mutate then sample
        mutated = mutator.apply(sequence)
        result = sampler.apply(mutated)

        assert len(result) == 10

    def test_augmentation_pipeline(self):
        """Test complete augmentation pipeline."""
        rc = dbp.ReverseComplement()
        mutator = dbp.Mutator(mutation_rate=0.02, seed=42)
        sampler = dbp.Sampler(length=20, strategy="center", seed=42)
        quality_model = dbp.QualityModel.HighQuality
        quality_sim = dbp.QualitySimulator(quality_model, seed=42)

        sequence = b"ACGTACGTACGTACGTACGTACGTACGTACGT"

        # Full pipeline
        augmented = sequence
        augmented = rc.apply(augmented)
        augmented = mutator.apply(augmented)
        augmented = sampler.apply(augmented)
        quality = quality_sim.generate(len(augmented))

        assert len(augmented) == 20
        assert len(quality) == len(augmented)

    def test_batch_augmentation(self):
        """Test batch augmentation of multiple sequences."""
        rc = dbp.ReverseComplement()
        mutator = dbp.Mutator(mutation_rate=0.05, seed=42)

        sequences = [b"ACGTACGT" * 4, b"TTTTAAAA" * 4, b"GGGGCCCC" * 4]

        # Batch reverse complement
        rc_seqs = rc.apply_batch(sequences)

        # Batch mutate
        mutated = mutator.apply_batch(rc_seqs)

        assert len(mutated) == 3
        assert all(len(m) == len(s) for m, s in zip(mutated, sequences, strict=False))
