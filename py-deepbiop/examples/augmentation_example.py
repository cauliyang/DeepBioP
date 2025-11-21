#!/usr/bin/env python3
"""Example demonstrating data augmentation for biological sequences.

This script shows how to use DeepBioP's augmentation features to create
diverse training data for machine learning models.
"""

import deepbiop


def main():
    """Run augmentation examples demonstrating all features."""
    print("DeepBioP Data Augmentation Examples")
    print("=" * 60)

    # Example sequence
    original_seq = b"ACGTACGTACGTACGTACGTACGTACGTACGT"  # 32 bases
    print(f"\nOriginal sequence: {original_seq.decode()}")
    print(f"Length: {len(original_seq)} bases")

    # 1. Reverse Complement
    print("\n1. Reverse Complement Augmentation")
    print("-" * 60)
    rc = deepbiop.fq.ReverseComplement()
    rc_seq = rc.apply(original_seq)
    print(f"Original:  {original_seq.decode()}")
    print(f"Reverse:   {rc_seq.decode()}")
    print("Use case: Double training data by including reverse strand")

    # RNA mode
    rna_seq = b"ACGUACGUACGU"
    rc_rna = deepbiop.fq.ReverseComplement.for_rna()
    rna_result = rc_rna.apply(rna_seq)
    print(f"\nRNA mode: {rna_seq.decode()} -> {rna_result.decode()}")

    # 2. Random Mutations
    print("\n\n2. Random Mutation Augmentation")
    print("-" * 60)

    # Low mutation rate (1%)
    mutator_low = deepbiop.fq.Mutator(0.01, 42)
    mutated_low = mutator_low.apply(original_seq)
    mutations_low = sum(a != b for a, b in zip(original_seq, mutated_low, strict=False))
    print(f"Low rate (1%):  {mutated_low.decode()}")
    print(f"  Mutations: {mutations_low}/{len(original_seq)} bases changed")

    # Medium mutation rate (10%)
    mutator_med = deepbiop.fq.Mutator(0.10, 42)
    mutated_med = mutator_med.apply(original_seq)
    mutations_med = sum(a != b for a, b in zip(original_seq, mutated_med, strict=False))
    print(f"Medium rate (10%): {mutated_med.decode()}")
    print(f"  Mutations: {mutations_med}/{len(original_seq)} bases changed")

    # High mutation rate (20%)
    mutator_high = deepbiop.fq.Mutator(0.20, 42)
    mutated_high = mutator_high.apply(original_seq)
    mutations_high = sum(
        a != b for a, b in zip(original_seq, mutated_high, strict=False)
    )
    print(f"High rate (20%): {mutated_high.decode()}")
    print(f"  Mutations: {mutations_high}/{len(original_seq)} bases changed")

    print("\nUse case: Improve model robustness to sequencing errors")

    # 3. Subsequence Sampling
    print("\n\n3. Subsequence Sampling Augmentation")
    print("-" * 60)

    # Random sampling
    sampler_random = deepbiop.fq.Sampler.random(10, 42)
    sample1 = sampler_random.apply(original_seq)
    sample2 = sampler_random.apply(original_seq)
    print("Random (10bp):")
    print(f"  Sample 1: {sample1.decode()} (from random position)")
    print(f"  Sample 2: {sample2.decode()} (different position)")

    # Fixed position sampling
    sampler_start = deepbiop.fq.Sampler.from_start(10)
    sampler_center = deepbiop.fq.Sampler.from_center(10)
    sampler_end = deepbiop.fq.Sampler.from_end(10)

    print("\nFixed position sampling (10bp):")
    print(f"  Start:  {sampler_start.apply(original_seq).decode()}")
    print(f"  Center: {sampler_center.apply(original_seq).decode()}")
    print(f"  End:    {sampler_end.apply(original_seq).decode()}")

    print("\nUse case: Create fixed-length inputs for CNN/RNN models")

    # 4. Combined Augmentation Pipeline
    print("\n\n4. Combined Augmentation Pipeline")
    print("-" * 60)
    print("Combining multiple augmentations for robust training:\n")

    # Pipeline: Sample -> Mutate -> Maybe reverse complement
    sequences = []
    for i in range(5):
        # Random subsequence
        sampler = deepbiop.fq.Sampler.random(20, 100 + i)
        sampled = sampler.apply(original_seq)

        # Add mutations
        mutator = deepbiop.fq.Mutator(0.05, 200 + i)
        mutated = mutator.apply(sampled)

        # 50% chance of reverse complement
        if i % 2 == 0:
            rc = deepbiop.fq.ReverseComplement()
            final = rc.apply(mutated)
            label = "(RC)"
        else:
            final = mutated
            label = "    "

        sequences.append(final)
        print(f"Sample {i + 1} {label}: {final.decode()}")

    print("\n✓ Generated 5 diverse training samples from 1 sequence")

    # 5. Batch Processing Example
    print("\n\n5. Batch Processing Example")
    print("-" * 60)

    # Simulate multiple sequences
    test_sequences = [
        b"ACGTACGTACGT",
        b"GGGGCCCCAAAA",
        b"TTTTAAAACCCCGGGG",
    ]

    print("Processing batch of sequences:")
    augmenter = deepbiop.fq.Mutator(0.10, 42)
    sampler = deepbiop.fq.Sampler.random(10, 42)

    for i, seq in enumerate(test_sequences, 1):
        sampled = sampler.apply(seq)
        augmented = augmenter.apply(sampled)
        print(f"  Seq {i}: {seq.decode():20s} -> {augmented.decode()}")

    print("\n" + "=" * 60)
    print("For ML training pipelines, combine with FastqDataset for")
    print("efficient on-the-fly augmentation during batch loading.")
    print("=" * 60)


if __name__ == "__main__":
    main()
