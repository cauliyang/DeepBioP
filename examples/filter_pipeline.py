"""
Example: Data Filtering Pipeline for Deep Learning

This script demonstrates how to use DeepBioP's filtering capabilities
to prepare sequencing data for machine learning training.

Workflow:
1. Filter by sequence length (remove too short/long reads)
2. Filter by quality score (remove low-quality reads)
3. Deduplicate (remove PCR duplicates)
4. Subsample (create train/val/test splits)
5. Encode for ML model
"""

import deepbiop as dbp


def basic_filtering_example():
    """Basic example of filtering sequences."""
    print("=" * 60)
    print("Basic Filtering Example")
    print("=" * 60)

    # Create filters
    length_filter = dbp.fq.LengthFilter.range(50, 500)
    quality_filter = dbp.fq.QualityFilter.mean_quality(20.0, quality_offset=33)
    dedup = dbp.fq.Deduplicator()

    # Example sequences
    sequences = [
        b"ACGTACGTACGT" * 5,  # 60 bp, good quality
        b"ACGT",  # Too short (4 bp)
        b"ACGTACGTACGT" * 100,  # Too long (1200 bp)
        b"ACGTACGTACGT" * 5,  # Duplicate of first
    ]

    qualities = [
        b"I" * 60,  # High quality (Q=40)
        b"I" * 4,  # High quality
        b"I" * 1200,  # High quality
        b"I" * 60,  # High quality
    ]

    print("\nFiltering sequences...")
    for i, (seq, qual) in enumerate(zip(sequences, qualities)):
        passes_length = length_filter.passes(seq)
        passes_quality = quality_filter.passes(seq, qual)
        passes_dedup = dedup.passes(seq)

        print(f"\nSequence {i + 1}:")
        print(f"  Length: {len(seq)} bp")
        print(f"  Passes length filter (50-500 bp): {passes_length}")
        print(f"  Passes quality filter (Q>=20): {passes_quality}")
        print(f"  Passes deduplication: {passes_dedup}")
        print(
            f"  OVERALL: {'✓ PASS' if (passes_length and passes_quality and passes_dedup) else '✗ FAIL'}"
        )

    print(f"\nUnique sequences seen: {dedup.unique_count()}")


def subsampling_example():
    """Example of subsampling for train/val/test splits."""
    print("\n" + "=" * 60)
    print("Subsampling Example - Train/Val/Test Split")
    print("=" * 60)

    # Simulate 100 sequences
    total_sequences = 100
    sequences = [f"SEQUENCE_{i}".encode() for i in range(total_sequences)]

    # Strategy 1: Random 80/10/10 split with reproducible seeds
    print("\nStrategy 1: Random 80/10/10 split (reproducible)")
    train_sampler = dbp.fq.Subsampler.random_fraction(0.8, seed=42)
    val_sampler = dbp.fq.Subsampler.random_fraction(0.1, seed=43)
    test_sampler = dbp.fq.Subsampler.random_fraction(0.1, seed=44)

    train_count = sum(1 for seq in sequences if train_sampler.passes(seq))
    val_count = sum(1 for seq in sequences if val_sampler.passes(seq))
    test_count = sum(1 for seq in sequences if test_sampler.passes(seq))

    print(
        f"  Train: {train_count}/{total_sequences} ({train_count / total_sequences * 100:.1f}%)"
    )
    print(
        f"  Val:   {val_count}/{total_sequences} ({val_count / total_sequences * 100:.1f}%)"
    )
    print(
        f"  Test:  {test_count}/{total_sequences} ({test_count / total_sequences * 100:.1f}%)"
    )

    # Strategy 2: First N for quick testing
    print("\nStrategy 2: First 10 sequences for quick test")
    quick_test_sampler = dbp.fq.Subsampler.first_n(10)
    test_count = sum(1 for seq in sequences if quick_test_sampler.passes(seq))
    print(f"  Selected: {test_count}/{total_sequences} sequences")

    # Strategy 3: Every Nth for systematic sampling
    print("\nStrategy 3: Every 5th sequence")
    systematic_sampler = dbp.fq.Subsampler.every_nth(5)
    selected_count = sum(1 for seq in sequences if systematic_sampler.passes(seq))
    print(f"  Selected: {selected_count}/{total_sequences} sequences")


def quality_analysis_example():
    """Example of quality score analysis."""
    print("\n" + "=" * 60)
    print("Quality Score Analysis Example")
    print("=" * 60)

    # Create quality filter
    filter = dbp.fq.QualityFilter(
        min_mean_quality=25.0, min_base_quality=20, quality_offset=33
    )

    # Example quality scores (ASCII Phred+33)
    examples = {
        "High quality (Q=40)": b"I" * 10,  # All Q=40
        "Good quality (Q=30)": b"?" * 10,  # All Q=30
        "Mixed quality": b"IIIII!!!!!",  # Half Q=40, half Q=0
        "Low quality (Q=10)": b"+" * 10,  # All Q=10
    }

    print("\nQuality threshold: mean >= 25, min base >= 20")
    print()

    for name, qual in examples.items():
        mean_qual = filter.calculate_mean_quality(list(qual))
        seq = b"A" * len(qual)
        passes = filter.passes(seq, list(qual))

        print(f"{name}:")
        print(f"  Mean quality: {mean_qual:.1f}")
        print(f"  Passes filter: {'✓ YES' if passes else '✗ NO'}")
        print()


def complete_ml_pipeline_example():
    """Complete pipeline: filter → encode → ready for ML."""
    print("\n" + "=" * 60)
    print("Complete ML Pipeline Example")
    print("=" * 60)

    # Simulate raw sequencing data
    raw_sequences = [
        (b"ACGTACGTACGT" * 5, b"I" * 60),  # Good: right length, high quality
        (b"ACGT", b"IIII"),  # Bad: too short
        (b"ACGTACGTACGT" * 5, b"!" * 60),  # Bad: low quality
        (b"ACGTACGTACGT" * 5, b"I" * 60),  # Duplicate
        (b"TTGGCCAATTGG" * 5, b"I" * 60),  # Good: different sequence
        (b"AAAACCCCGGGG" * 5, b"5" * 60),  # Good: medium quality (Q=20)
    ]

    print(f"\nRaw data: {len(raw_sequences)} sequences")

    # Step 1: Create filter pipeline
    length_filter = dbp.fq.LengthFilter.range(50, 500)
    quality_filter = dbp.fq.QualityFilter.mean_quality(20.0, quality_offset=33)
    dedup = dbp.fq.Deduplicator()

    # Step 2: Apply filters
    filtered_sequences = []
    for seq, qual in raw_sequences:
        if (
            length_filter.passes(seq)
            and quality_filter.passes(seq, list(qual))
            and dedup.passes(seq)
        ):
            filtered_sequences.append(seq)

    print(f"After filtering: {len(filtered_sequences)} sequences")
    print(f"  Removed: {len(raw_sequences) - len(filtered_sequences)} sequences")
    print(f"  Unique sequences: {dedup.unique_count()}")

    # Step 3: Subsample for train/test split (80/20)
    train_sampler = dbp.fq.Subsampler.random_fraction(0.8, seed=42)
    train_sequences = [seq for seq in filtered_sequences if train_sampler.passes(seq)]
    test_sequences = [seq for seq in filtered_sequences if seq not in train_sequences]

    print("\nTrain/test split:")
    print(f"  Train: {len(train_sequences)} sequences")
    print(f"  Test:  {len(test_sequences)} sequences")

    # Step 4: Encode for ML
    encoder = dbp.fq.OneHotEncoder("dna", "mask")

    if train_sequences:
        train_encoded = encoder.encode_batch(train_sequences)
        print(f"\nEncoded training data shape: {train_encoded.shape}")
        print("  Format: [num_sequences, seq_length, alphabet_size]")
        print("  Ready for PyTorch/TensorFlow!")

        # Example: convert to PyTorch (if available)
        try:
            import torch

            train_tensor = torch.from_numpy(train_encoded)
            print(f"\nPyTorch tensor shape: {train_tensor.shape}")
            print(f"  Device: {train_tensor.device}")
            print(f"  Dtype: {train_tensor.dtype}")
        except ImportError:
            print("\n(PyTorch not installed - skipping tensor conversion)")


def deduplication_statistics_example():
    """Example showing deduplication statistics."""
    print("\n" + "=" * 60)
    print("Deduplication Statistics Example")
    print("=" * 60)

    # Simulate dataset with PCR duplicates
    sequences = [
        b"ACGTACGT",  # Unique 1
        b"ACGTACGT",  # Duplicate
        b"ACGTACGT",  # Duplicate
        b"TTGGCCAA",  # Unique 2
        b"TTGGCCAA",  # Duplicate
        b"AAAACCCC",  # Unique 3
        b"ACGTACGT",  # Duplicate
        b"GGGGTTTT",  # Unique 4
    ]

    print(f"\nTotal sequences: {len(sequences)}")

    dedup = dbp.fq.Deduplicator()
    unique_sequences = []

    for seq in sequences:
        if dedup.passes(seq):
            unique_sequences.append(seq)

    print(f"Unique sequences: {len(unique_sequences)}")
    print(f"Duplicates removed: {len(sequences) - len(unique_sequences)}")
    print(
        f"Deduplication rate: {(1 - len(unique_sequences) / len(sequences)) * 100:.1f}%"
    )

    # Show which sequences were kept
    print("\nUnique sequences kept:")
    for i, seq in enumerate(unique_sequences, 1):
        print(f"  {i}. {seq.decode()}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("DeepBioP Filtering Examples")
    print("=" * 60)

    basic_filtering_example()
    subsampling_example()
    quality_analysis_example()
    complete_ml_pipeline_example()
    deduplication_statistics_example()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
