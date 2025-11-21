# Python API Usage Guide

This guide provides comprehensive examples for using DeepBioP's Python API for biological data preprocessing and machine learning workflows.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core API](#core-api)
4. [PyTorch-Style API](#pytorch-style-api)
5. [Encoding Strategies](#encoding-strategies)
6. [Data Augmentation](#data-augmentation)
7. [Advanced Usage](#advanced-usage)

## Installation

```bash
pip install deepbiop
```

For development installation:

```bash
git clone https://github.com/cauliyang/DeepBioP.git
cd DeepBioP/py-deepbiop
make build  # or: uv sync && uvx maturin develop
```

## Quick Start

### Encode DNA Sequences

```python
import deepbiop as dbp
import numpy as np

# One-hot encoding for CNNs/RNNs
encoder = dbp.OneHotEncoder("dna", "skip")
sequences = [b"ACGTACGT", b"TTGGCCAA", b"AAAACCCC"]
encoded = encoder.encode_batch(sequences)
print(encoded.shape)  # (3, 8, 4) - batch × seq_len × channels
```

### Load FASTQ Files

```python
import deepbiop as dbp

# Read FASTQ file
reader = dbp.FastqReader("data.fastq")
for record in reader:
    print(f"ID: {record.id}")
    print(f"Sequence: {record.sequence}")
    print(f"Quality: {record.quality}")
    break
```

## Core API

The core API provides direct access to biological file formats and encoding functions.

### FASTQ/FASTA Processing

#### Reading FASTQ Files

```python
import deepbiop as dbp

# Basic reading
reader = dbp.FastqReader("input.fastq")
for record in reader:
    print(f"{record.id}: {len(record.sequence)}bp")

# With compression support
reader = dbp.FastqReader("input.fastq.gz")  # Auto-detects gzip

# Batch processing
reader = dbp.FastqReader("input.fastq")
batch = reader.read_batch(1000)  # Read 1000 records at once
```

#### Writing FASTQ Files

```python
import deepbiop as dbp

# Create writer
writer = dbp.FastqWriter("output.fastq")

# Write records
for record in reader:
    # Process record...
    writer.write_record(record)

writer.close()

# With compression
writer = dbp.FastqWriter("output.fastq.gz", compression="gzip")
```

#### FASTA Files

```python
import deepbiop as dbp

# Read FASTA
reader = dbp.FastaReader("genome.fasta")
for record in reader:
    print(f">{record.id}\n{record.sequence[:50]}...")  # First 50bp

# Write FASTA
writer = dbp.FastaWriter("output.fasta")
writer.write_record(record)
writer.close()
```

### BAM/SAM Processing

```python
import deepbiop as dbp

# Read BAM file
reader = dbp.BamReader("alignments.bam")

# Count chimeric reads
chimeric_count = reader.count_chimeric()
print(f"Chimeric reads: {chimeric_count}")

# Convert BAM to FASTQ
bam_reader = dbp.BamReader("input.bam")
fastq_writer = dbp.FastqWriter("output.fastq")

for record in bam_reader.to_fastq_records():
    fastq_writer.write_record(record)

fastq_writer.close()
```

### VCF/GTF Processing

```python
import deepbiop as dbp

# Read VCF
reader = dbp.VcfReader("variants.vcf")

# Filter high-quality variants
variants = reader.filter_by_quality(30.0)
print(f"High-quality variants: {len(variants)}")

# Filter passing variants
passing = reader.filter_passing()

# Read GTF annotations
reader = dbp.GtfReader("annotations.gtf")
for record in reader:
    if record.feature_type == "gene":
        print(f"Gene: {record.gene_id}")
```

## PyTorch-Style API

The PyTorch-style API provides familiar Dataset and DataLoader patterns for ML workflows.

### Basic Dataset Usage

```python
from deepbiop.pytorch import Dataset, DataLoader

# Create dataset
dataset = Dataset("sequences.fastq")

# Access samples
print(f"Dataset size: {len(dataset)}")
sample = dataset[0]
print(f"Sequence: {sample['sequence']}")
print(f"Quality: {sample['quality']}")

# Iterate over dataset
for sample in dataset:
    # Process sample...
    pass
```

### DataLoader with Batching

```python
from deepbiop.pytorch import Dataset, DataLoader

# Create dataset and loader
dataset = Dataset("data.fastq")
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    drop_last=False
)

# Iterate over batches
for batch in loader:
    print(f"Batch size: {len(batch)}")
    for sample in batch:
        # Process each sample in batch
        pass
```

### With Transforms

```python
from deepbiop.pytorch import Dataset, DataLoader, OneHotEncoder, Compose, ReverseComplement

# Create transform pipeline
transform = Compose([
    ReverseComplement(probability=0.5),  # 50% chance of reverse complement
    OneHotEncoder(encoding_type="dna", unknown_strategy="skip")
])

# Create dataset with transform
dataset = Dataset("data.fastq", transform=transform)

# Create loader
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Iterate
for batch in loader:
    for sample in batch:
        # sample['sequence'] is now a NumPy array (seq_len, 4)
        encoded_seq = sample['sequence']
        assert encoded_seq.shape[1] == 4  # 4 channels for DNA
```

### Training Loop Example

```python
import torch
import torch.nn as nn
from deepbiop.pytorch import Dataset, DataLoader, OneHotEncoder, Compose, Mutator

# Define model
class DNAClassifier(nn.Module):
    def __init__(self, seq_length, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 32, kernel_size=7, padding=3)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.fc = nn.Linear(64 * (seq_length // 4), num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Prepare data
transform = Compose([
    Mutator(mutation_rate=0.01),  # Data augmentation
    OneHotEncoder(encoding_type="dna")
])

dataset = Dataset("train.fastq", transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
model = DNAClassifier(seq_length=150, num_classes=2)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Training loop
model.train()
for epoch in range(10):
    for batch in loader:
        # Convert to PyTorch tensors
        sequences = torch.stack([
            torch.from_numpy(sample['sequence'].T)  # (4, seq_len)
            for sample in batch
        ])

        # Get labels (assuming they're in metadata)
        labels = torch.tensor([
            sample.get('label', 0)  # Default label 0
            for sample in batch
        ])

        # Forward pass
        optimizer.zero_grad()
        outputs = model(sequences.float())
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

## Encoding Strategies

DeepBioP provides multiple encoding strategies optimized for different ML models.

### One-Hot Encoding (CNNs/RNNs)

```python
import deepbiop as dbp
import numpy as np

# DNA one-hot encoding
encoder = dbp.OneHotEncoder("dna", "skip")
sequence = b"ACGT"
encoded = encoder.encode(sequence)
print(encoded.shape)  # (4, 4) - seq_len × channels

# Expected output:
# [[1, 0, 0, 0],  # A
#  [0, 1, 0, 0],  # C
#  [0, 0, 1, 0],  # G
#  [0, 0, 0, 1]]  # T
```

### Integer Encoding (Transformers/Embeddings)

```python
import deepbiop as dbp

# Integer encoding for embeddings
encoder = dbp.IntegerEncoder("dna", "skip")
sequence = b"ACGT"
encoded = encoder.encode(sequence)
print(encoded)  # array([0, 1, 2, 3])

# Use with PyTorch embedding layer
import torch.nn as nn
embedding = nn.Embedding(num_embeddings=4, embedding_dim=128)
```

### K-mer Encoding (Feature-Based Models)

```python
import deepbiop as dbp

# 3-mer encoding
encoder = dbp.KmerEncoder(k=3, canonical=False, encoding_type="dna")
sequence = b"ACGTACGT"
encoded = encoder.encode(sequence)

# K-mer frequencies for Random Forest, XGBoost, etc.
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
# clf.fit(encoded_features, labels)
```

### Batch Encoding

```python
import deepbiop as dbp
import numpy as np

# Encode multiple sequences efficiently
encoder = dbp.OneHotEncoder("dna", "skip")
sequences = [b"ACGT", b"TTGG", b"AACC"]

# Batch encoding (parallelized with Rust)
batch_encoded = encoder.encode_batch(sequences)
print(batch_encoded.shape)  # (3, 4, 4) - batch × seq_len × channels
```

## Data Augmentation

DeepBioP provides biologically-relevant augmentation techniques.

### Reverse Complement

```python
from deepbiop.pytorch import ReverseComplement

# Augmentation for orientation-invariant training
augment = ReverseComplement(probability=0.5, seed=42)

sample = {"sequence": b"ACGT", "quality": b"IIII"}
augmented = augment(sample)
# 50% chance: sequence becomes b"ACGT" (reverse complement of ACGT)
```

### Random Mutations

```python
from deepbiop.pytorch import Mutator

# Simulate sequencing errors and SNPs
augment = Mutator(mutation_rate=0.01, seed=42)

sample = {"sequence": b"ACGTACGT" * 10}  # 80bp
mutated = augment(sample)
# ~0.8 bases will be randomly mutated
```

### Subsequence Sampling

```python
from deepbiop.pytorch import Sampler

# Extract fixed-length windows
sampler = Sampler(length=100, mode="random", seed=42)

sample = {"sequence": b"ACGT" * 50}  # 200bp
windowed = sampler(sample)
print(len(windowed['sequence']))  # 100bp

# Different modes
sampler_start = Sampler(length=100, mode="start")   # First 100bp
sampler_center = Sampler(length=100, mode="center") # Middle 100bp
sampler_end = Sampler(length=100, mode="end")       # Last 100bp
```

### Combining Augmentations

```python
from deepbiop.pytorch import Compose, ReverseComplement, Mutator, Sampler, OneHotEncoder

# Create augmentation pipeline
augmentation = Compose([
    Sampler(length=150, mode="random"),          # Random 150bp window
    ReverseComplement(probability=0.5),          # 50% reverse complement
    Mutator(mutation_rate=0.01),                 # 1% mutation rate
    OneHotEncoder(encoding_type="dna")           # Final encoding
])

sample = {"sequence": b"ACGT" * 100}  # 400bp
augmented = augmentation(sample)
print(augmented['sequence'].shape)  # (150, 4)
```

## Advanced Usage

### Caching Processed Data

```python
from deepbiop.pytorch import Dataset, save_cache, load_cache, is_cache_valid

cache_path = "processed_data.cache"
source_files = ["data.fastq"]

# Check cache validity
if is_cache_valid(cache_path, source_files, max_age_seconds=3600):
    # Load from cache
    data, metadata = load_cache(cache_path)
    print(f"Loaded from cache (version: {metadata['version']})")
else:
    # Process data
    dataset = Dataset("data.fastq")
    data = [dataset[i] for i in range(len(dataset))]

    # Save to cache
    save_cache(
        data,
        cache_path,
        metadata={"version": "1.0", "num_samples": len(data)}
    )
```

### Custom Collate Function

```python
from deepbiop.pytorch import DataLoader, Dataset
import numpy as np

def custom_collate(samples):
    """Collate samples with padding to max length."""
    # Find max sequence length
    max_len = max(len(s['sequence']) for s in samples)

    # Pad sequences
    padded = []
    for sample in samples:
        seq = sample['sequence']
        if len(seq) < max_len:
            # Pad with zeros
            pad_width = ((0, max_len - len(seq)), (0, 0))
            seq = np.pad(seq, pad_width, mode='constant')
        padded.append(seq)

    return {
        'sequences': np.stack(padded),
        'lengths': np.array([len(s['sequence']) for s in samples])
    }

# Use custom collate
dataset = Dataset("data.fastq")
loader = DataLoader(
    dataset,
    batch_size=16,
    collate_fn=custom_collate
)

for batch in loader:
    sequences = batch['sequences']  # Padded to same length
    lengths = batch['lengths']      # Original lengths
    break
```

### Dataset Statistics

```python
from deepbiop.pytorch import Dataset

dataset = Dataset("data.fastq")

# Get summary statistics
stats = dataset.summary()
print(f"Number of samples: {stats['num_samples']}")
print(f"Length statistics: {stats['length_stats']}")
print(f"Memory footprint: {stats['memory_footprint']} bytes")

# Validate dataset
validation = dataset.validate()
if validation['is_valid']:
    print("Dataset is valid!")
else:
    print(f"Warnings: {validation['warnings']}")
    print(f"Errors: {validation['errors']}")
```

### Integration with PyTorch DataLoader

DeepBioP Dataset is compatible with `torch.utils.data.DataLoader`:

```python
import torch.utils.data
from deepbiop.pytorch import Dataset, OneHotEncoder

# Create DeepBioP dataset
dataset = Dataset("data.fastq", transform=OneHotEncoder(encoding_type="dna"))

# Use with PyTorch DataLoader
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,  # Multi-process loading
    collate_fn=lambda x: x  # Custom collate if needed
)

for batch in loader:
    # Process with PyTorch
    pass
```

## Performance Tips

1. **Use batch operations**: `encoder.encode_batch()` is faster than looping with `encoder.encode()`

2. **Enable caching**: Cache preprocessed data for repeated experiments

3. **Streaming for large files**: Dataset uses streaming I/O for memory efficiency

4. **Parallel processing**: Use `num_workers > 0` in DataLoader for multi-process loading

5. **Rust-backed operations**: Encoding and augmentation are implemented in Rust for speed

## Next Steps

- See [API Reference](../api/index.md) for complete API documentation
- Check [Rust API Guide](rust-api.md) for using DeepBioP from Rust
- Explore [CLI Usage](../cli/cli.md) for command-line tools

## Troubleshooting

### Import errors

```python
# Make sure deepbiop is installed
pip install deepbiop

# Check installation
python -c "import deepbiop; print(deepbiop.__version__)"
```

### Memory issues with large files

```python
# Use streaming Dataset (lazy=True, default)
dataset = Dataset("large.fastq", lazy=True)

# Or process in chunks
reader = dbp.FastqReader("large.fastq")
chunk_size = 10000
for i in range(0, total_records, chunk_size):
    chunk = reader.read_batch(chunk_size)
    # Process chunk...
```

### Type hints not working

Make sure `.pyi` stub files are installed:

```bash
# They should be in: site-packages/deepbiop/*.pyi
python -c "import deepbiop; import os; print(os.path.dirname(deepbiop.__file__))"
```
