"""PyTorch API Quickstart Example.

This example demonstrates the basic usage of DeepBioP's PyTorch-style API
for loading and preprocessing biological sequence data.

Example demonstrates:
- Dataset creation from FASTQ files
- Transform pipeline setup
- DataLoader configuration
- Batch processing workflow
"""

from pathlib import Path

from deepbiop import pytorch


def main():
    """Main function demonstrating PyTorch API usage."""
    print("PyTorch API Quickstart Example")
    print("=" * 70)
    print()

    # 1. Load FASTQ file
    print("Step 1: Loading FASTQ file")
    print("-" * 70)

    # Use test data from tests directory
    test_file = Path(__file__).parent.parent / "tests" / "data" / "test.fastq"

    if not test_file.exists():
        print(f"Error: Test file not found at {test_file}")
        print("Please ensure test data is available.")
        return

    dataset = pytorch.Dataset(str(test_file))
    print(f"✓ Loaded dataset with {len(dataset)} sequences")
    print()

    # 2. Inspect a sample
    print("Step 2: Inspecting a sample")
    print("-" * 70)
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Sequence: {sample['sequence'][:50]}... (first 50 bases)")
    print(f"Quality: {sample['quality'][:50]}... (first 50 scores)")
    print()

    # 3. Apply transformation
    print("Step 3: Applying OneHot encoding")
    print("-" * 70)
    encoder = pytorch.OneHotEncoder(encoding_type="dna", unknown_strategy="skip")
    encoded_sample = encoder(sample)
    print(f"Original sequence length: {len(sample['sequence'])} bases")
    print(f"Encoded shape: {encoded_sample['sequence'].shape}")
    print("Encoded shape format: [sequence_length, features (ACGT)]")
    print()

    # 4. Create DataLoader
    print("Step 4: Creating DataLoader")
    print("-" * 70)
    batch_size = 5
    loader = pytorch.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Set to True for training
    )
    print(f"✓ Created DataLoader with batch_size={batch_size}")
    print(f"✓ Number of batches: {len(loader)}")
    print()

    # 5. Process a batch
    print("Step 5: Processing batches")
    print("-" * 70)

    # Manually encode samples and collate
    batch_samples = []
    for i, sample in enumerate(dataset):
        if i >= batch_size:
            break
        encoded = encoder(sample)
        batch_samples.append(encoded)

    # Collate into batch
    batch = pytorch.default_collate(batch_samples)

    print(f"Batch keys: {list(batch.keys())}")
    print(f"Batch sequences shape: {batch['sequences'].shape}")
    print("Batch shape format: [batch_size, max_length, features]")
    print(f"Batch lengths: {batch['lengths']}")
    print(f"Max sequence length in batch: {batch['sequences'].shape[1]}")
    print()

    # 6. Summary
    print("Step 6: Summary")
    print("-" * 70)
    print("✓ Successfully loaded FASTQ data")
    print("✓ Applied one-hot encoding transformation")
    print("✓ Created batches with padding")
    print("✓ Batch is ready for PyTorch model input")
    print()
    print("Next steps:")
    print("- Use batch['sequences'] as input to PyTorch models")
    print("- Use batch['lengths'] for sequence length masking")
    print("- Try different encoders: IntegerEncoder, KmerEncoder")
    print("- Enable shuffling for training: shuffle=True")
    print("=" * 70)


if __name__ == "__main__":
    main()
