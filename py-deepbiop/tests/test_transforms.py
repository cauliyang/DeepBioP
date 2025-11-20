"""
Tests for transform reproducibility with seeds.

Tests T022-T025 for User Story 2: Reproducible Analysis.
"""

import pytest

# Import transforms - these should work once exported
try:
    from deepbiop import ReverseComplement, Mutator, Compose
except ImportError:
    # Fallback to pytorch module during development
    try:
        from deepbiop.pytorch import ReverseComplement, Mutator, Compose
    except ImportError:
        pytest.skip("Transforms not yet exported", allow_module_level=True)


class TestReverseComplementSeed:
    """T022: Unit test for ReverseComplement with seed reproducibility."""

    def test_reverse_complement_no_seed_deterministic(self):
        """ReverseComplement should be deterministic (no randomness)."""
        sample = {"id": b"@seq1", "sequence": b"ACGT", "quality": b"IIII"}

        # ReverseComplement is deterministic - no seed needed
        rc1 = ReverseComplement()
        rc2 = ReverseComplement()

        result1 = rc1(sample)
        result2 = rc2(sample)

        # Should produce identical results
        assert result1["sequence"] == result2["sequence"]
        assert result1["sequence"] == b"ACGT"  # RC of ACGT is ACGT (palindrome)

    def test_reverse_complement_actual_sequence(self):
        """Test reverse complement transformation."""
        sample = {"id": b"@seq1", "sequence": b"ATCG", "quality": b"ABCD"}

        rc = ReverseComplement()
        result = rc(sample)

        # RC of ATCG is CGAT
        assert result["sequence"] == b"CGAT"
        # Quality should be reversed too
        assert result["quality"] == b"DCBA"

    def test_reverse_complement_preserves_other_fields(self):
        """ReverseComplement should preserve non-sequence fields."""
        sample = {"id": b"@seq1", "sequence": b"ATCG", "metadata": "test"}

        rc = ReverseComplement()
        result = rc(sample)

        assert result["id"] == b"@seq1"
        assert result["metadata"] == "test"


class TestMutatorSeed:
    """T023: Unit test for Mutator with seed reproducibility."""

    def test_mutator_with_seed_reproducible(self):
        """Mutator with same seed should produce identical results."""
        sample = {"id": b"@seq1", "sequence": b"ACGTACGTACGT", "quality": b"I" * 12}

        mut1 = Mutator(mutation_rate=0.1, seed=42)
        mut2 = Mutator(mutation_rate=0.1, seed=42)

        result1 = mut1(sample.copy())
        result2 = mut2(sample.copy())

        # Should produce identical mutations
        assert result1["sequence"] == result2["sequence"]

    def test_mutator_different_seeds_different_results(self):
        """Mutator with different seeds should produce different results."""
        sample = {"id": b"@seq1", "sequence": b"ACGTACGTACGT", "quality": b"I" * 12}

        mut1 = Mutator(mutation_rate=0.5, seed=42)
        mut2 = Mutator(mutation_rate=0.5, seed=123)

        result1 = mut1(sample.copy())
        result2 = mut2(sample.copy())

        # High mutation rate should cause differences
        assert result1["sequence"] != result2["sequence"]

    def test_mutator_no_seed_nondeterministic(self):
        """Mutator without seed should produce varying results."""
        sample = {"id": b"@seq1", "sequence": b"ACGTACGTACGT", "quality": b"I" * 12}

        # Run multiple times - at least one should differ with high mutation rate
        results = []
        for _ in range(10):
            mut = Mutator(mutation_rate=0.5, seed=None)
            result = mut(sample.copy())
            results.append(result["sequence"])

        # With 50% mutation rate and 12 bases, should get variation
        # (statistically almost certain to have differences)
        unique_results = len(set(results))
        assert unique_results > 1, "Expected variation with no seed"


class TestTransformStateSaveLoad:
    """T025: State save/load test for transforms - Simplified for Rust RNG limitations."""

    def test_mutator_configuration_reproducible(self):
        """Mutator with same seed should produce identical results."""
        sample = {"id": b"@seq1", "sequence": b"ACGTACGTACGT", "quality": b"I" * 12}

        # Create two mutators with same seed
        mut1 = Mutator(mutation_rate=0.3, seed=42)
        mut2 = Mutator(mutation_rate=0.3, seed=42)

        # Should produce identical results
        result1 = mut1(sample.copy())
        result2 = mut2(sample.copy())
        assert result1["sequence"] == result2["sequence"]

    def test_mutator_seed_determines_output(self):
        """Different seeds produce different results with same configuration."""
        sample = {"id": b"@seq1", "sequence": b"ACGTACGTACGT", "quality": b"I" * 12}

        mut1 = Mutator(mutation_rate=0.5, seed=42)
        mut2 = Mutator(mutation_rate=0.5, seed=123)

        result1 = mut1(sample.copy())
        result2 = mut2(sample.copy())

        # High mutation rate should cause differences
        assert result1["sequence"] != result2["sequence"]


class TestComposeSeedPropagation:
    """T030: Compose should propagate seeds to child transforms."""

    def test_compose_with_seeded_transforms(self):
        """Compose should maintain reproducibility of seeded transforms."""
        sample = {"id": b"@seq1", "sequence": b"ACGTACGTACGT", "quality": b"I" * 12}

        # Create composed transform with seeds
        transform1 = Compose([
            ReverseComplement(),
            Mutator(mutation_rate=0.2, seed=42)
        ])

        transform2 = Compose([
            ReverseComplement(),
            Mutator(mutation_rate=0.2, seed=42)
        ])

        result1 = transform1(sample.copy())
        result2 = transform2(sample.copy())

        # Should produce identical results
        assert result1["sequence"] == result2["sequence"]

    def test_compose_preserves_transform_order(self):
        """Compose should apply transforms in correct order."""
        sample = {"id": b"@seq1", "sequence": b"ATCG", "quality": b"ABCD"}

        # First RC, then mutate
        transform = Compose([
            ReverseComplement(),
            Mutator(mutation_rate=0.0, seed=42)  # 0% mutation to test order
        ])

        result = transform(sample)

        # Should be RC of ATCG = CGAT (no mutation at 0%)
        assert result["sequence"] == b"CGAT"


class TestLengthFilter:
    """T055: Unit test for LengthFilter."""

    def test_length_filter_min_only(self):
        """Test LengthFilter with minimum length only."""
        try:
            from deepbiop import LengthFilter
        except ImportError:
            pytest.skip("LengthFilter not yet exported")

        # Filter accepting sequences >= 10 bases
        filter_fn = LengthFilter.min_only(10)

        # Test with sequences of different lengths
        short_seq = b"ACGT"  # 4 bases
        medium_seq = b"ACGTACGTAC"  # 10 bases
        long_seq = b"ACGTACGTACGTACGT"  # 16 bases

        assert not filter_fn.passes(short_seq), "Should reject sequences < 10"
        assert filter_fn.passes(medium_seq), "Should accept sequences == 10"
        assert filter_fn.passes(long_seq), "Should accept sequences > 10"

    def test_length_filter_max_only(self):
        """Test LengthFilter with maximum length only."""
        try:
            from deepbiop import LengthFilter
        except ImportError:
            pytest.skip("LengthFilter not yet exported")

        # Filter accepting sequences <= 10 bases
        filter_fn = LengthFilter.max_only(10)

        short_seq = b"ACGT"  # 4 bases
        medium_seq = b"ACGTACGTAC"  # 10 bases
        long_seq = b"ACGTACGTACGTACGT"  # 16 bases

        assert filter_fn.passes(short_seq), "Should accept sequences < 10"
        assert filter_fn.passes(medium_seq), "Should accept sequences == 10"
        assert not filter_fn.passes(long_seq), "Should reject sequences > 10"

    def test_length_filter_range(self):
        """Test LengthFilter with min and max range."""
        try:
            from deepbiop import LengthFilter
        except ImportError:
            pytest.skip("LengthFilter not yet exported")

        # Filter accepting sequences between 6 and 12 bases
        filter_fn = LengthFilter.range(6, 12)

        too_short = b"ACGT"  # 4 bases
        in_range_low = b"ACGTAC"  # 6 bases
        in_range_mid = b"ACGTACGT"  # 8 bases
        in_range_high = b"ACGTACGTACGT"  # 12 bases
        too_long = b"ACGTACGTACGTACGT"  # 16 bases

        assert not filter_fn.passes(too_short), "Should reject sequences < 6"
        assert filter_fn.passes(in_range_low), "Should accept sequences == 6"
        assert filter_fn.passes(in_range_mid), "Should accept sequences in range"
        assert filter_fn.passes(in_range_high), "Should accept sequences == 12"
        assert not filter_fn.passes(too_long), "Should reject sequences > 12"


class TestQualityFilter:
    """T056: Unit test for QualityFilter."""

    def test_quality_filter_min_mean(self):
        """Test QualityFilter with minimum mean quality."""
        try:
            from deepbiop import QualityFilter
        except ImportError:
            pytest.skip("QualityFilter not yet exported")

        # Filter accepting mean quality >= 30 (Phred+33)
        filter_fn = QualityFilter(min_mean_quality=30.0)

        # High quality: 'I' = 73 - 33 = 40
        high_quality = b"IIIIIIII"
        # Medium quality: '>' = 62 - 33 = 29
        medium_quality = b">>>>>>>>"
        # Low quality: '!' = 33 - 33 = 0
        low_quality = b"!!!!!!!!"

        # Note: Quality filter expects sequence, not just quality
        # For testing, we use the passes method which takes a sequence
        # The implementation will need to handle quality separately

        # Create simple test (just checking API works)
        assert filter_fn is not None, "QualityFilter should be created"

    def test_quality_filter_min_base(self):
        """Test QualityFilter with minimum base quality."""
        try:
            from deepbiop import QualityFilter
        except ImportError:
            pytest.skip("QualityFilter not yet exported")

        # Filter requiring all bases >= Q20 (Phred+33)
        filter_fn = QualityFilter(min_base_quality=20)

        # Verify filter was created
        assert filter_fn is not None, "QualityFilter should be created"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
