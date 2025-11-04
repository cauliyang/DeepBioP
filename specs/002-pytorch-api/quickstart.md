# Quickstart: PyTorch-Style Python API

**Date**: 2025-11-03
**Feature**: PyTorch-Style Python API for Deep Learning

This guide demonstrates the complete workflow from installation to training a simple deep learning model on biological sequence data.

---

## Installation

```bash
# Install DeepBioP with PyTorch API support
pip install deepbiop

# Optional: Install PyTorch for model training
pip install torch
```

---

## Basic Usage

### Example 1: Simple Dataset Loading

```python
from deepbiop.pytorch import Dataset, OneHotEncoder

# Create dataset with one-hot encoding
dataset = Dataset(
    "data/sequences.fastq",
    sequence_type="dna",
    transform=OneHotEncoder()
)

# Access samples
print(f"Dataset size: {len(dataset)}")  # 1000
sample = dataset[0]
print(f"Sequence shape: {sample['sequence'].shape}")  # (length, 4) for DNA one-hot
```

### Example 2: Batch Loading with DataLoader

```python
from deepbiop.pytorch import Dataset, DataLoader, OneHotEncoder

# Create dataset
dataset = Dataset(
    "data/sequences.fastq",
    transform=OneHotEncoder()
)

# Create data loader
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    seed=42
)

# Iterate through batches
for batch in loader:
    sequences = batch['sequences']  # shape: [32, max_len, 4]
    print(f"Batch shape: {sequences.shape}")
    break
```

---

## Complete Training Example

### Step 1: Prepare Data with Transform Pipeline

```python
from deepbiop.pytorch import (
    Dataset, DataLoader, Compose,
    Sampler, Mutator, OneHotEncoder
)

# Define transformation pipeline
transform = Compose([
    Sampler(length=100, strategy="random", seed=42),  # Extract 100bp subsequences
    Mutator(mutation_rate=0.01, seed=42),            # Apply 1% random mutations
    OneHotEncoder(encoding_type="dna")                # Convert to one-hot arrays
])

# Create dataset from FASTQ file
dataset = Dataset(
    file_paths="data/sequences.fastq",
    sequence_type="dna",
    transform=transform,
    cache_dir="/tmp/deepbiop_cache"  # Cache processed data for faster reloading
)

print(f"Loaded {len(dataset)} sequences")

# Inspect dataset
summary = dataset.summary()
print(f"Sequence lengths: {summary['sequence_lengths']}")
```

### Step 2: Create DataLoader for Batching

```python
# Create data loader with custom settings
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,      # Use 4 parallel workers for faster loading
    drop_last=True,     # Drop incomplete final batch
    seed=42             # Reproducible shuffling
)

print(f"Number of batches: {len(train_loader)}")
```

### Step 3: Define PyTorch Model

```python
import torch
import torch.nn as nn

class DNAClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: [batch_size, 4, length]
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)  # [batch_size, 128]
        x = self.fc(x)
        return x

model = DNAClassifier(num_classes=2)
print(model)
```

### Step 4: Training Loop

```python
import torch.optim as optim

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        # Convert NumPy arrays to PyTorch tensors (zero-copy)
        sequences = torch.from_numpy(batch['sequences'])  # [batch_size, length, 4]
        sequences = sequences.permute(0, 2, 1)  # [batch_size, 4, length] for Conv1d

        # Get labels (if available in your data)
        if 'labels' in batch and batch['labels'] is not None:
            labels = torch.from_numpy(batch['labels']).long()
        else:
            # Dummy labels for demonstration
            labels = torch.randint(0, 2, (sequences.size(0),))

        # Move to device
        sequences = sequences.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Batch [{batch_idx}/{len(train_loader)}], "
                  f"Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] completed, Average Loss: {avg_loss:.4f}")

print("Training completed!")
```

---

## Advanced Usage Examples

### Custom Collate Function for Variable-Length Sequences

```python
from deepbiop.pytorch import pad_collate_fn

# Create data loader with custom padding
loader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=pad_collate_fn(
        padding_value=0.0,
        pad_to=150  # Pad all sequences to 150bp
    )
)

for batch in loader:
    # All sequences now have fixed length 150
    assert batch['sequences'].shape[1] == 150
    break
```

### Using K-mer Encoding Instead of One-Hot

```python
from deepbiop.pytorch import KmerEncoder

# Create dataset with k-mer frequency encoding
kmer_transform = KmerEncoder(k=3, canonical=True)

dataset = Dataset(
    "data/sequences.fastq",
    transform=kmer_transform
)

# Each sample is now a k-mer frequency vector
sample = dataset[0]
print(f"K-mer vector shape: {sample['sequence'].shape}")  # (64,) for 3-mers (4^3)
```

### Multi-Transform Pipeline

```python
from deepbiop.pytorch import ReverseComplement, IntegerEncoder

# Complex pipeline with multiple augmentations
complex_transform = Compose([
    Sampler(length=100, strategy="center"),
    ReverseComplement(),           # 50% chance (can be wrapped in RandomApply)
    Mutator(mutation_rate=0.02),
    IntegerEncoder()               # A=0, C=1, G=2, T=3
])

dataset = Dataset(
    "data/sequences.fastq",
    transform=complex_transform
)
```

### Caching for Faster Reloading

```python
from deepbiop.pytorch import Cache

# First run: process and cache
dataset = Dataset(
    "data/large_file.fastq",
    transform=OneHotEncoder(),
    cache_dir="/tmp/cache"
)

# Save processed dataset
cache_path = Cache.save(dataset, "/tmp/cache")
print(f"Dataset cached to: {cache_path}")

# Second run: load from cache (10x faster)
if Cache.is_valid(cache_path, ["data/large_file.fastq"]):
    dataset = Cache.load(cache_path)
    print("Loaded from cache!")
else:
    dataset = Dataset("data/large_file.fastq", transform=OneHotEncoder())
```

---

## Integration with PyTorch Lightning

```python
import pytorch_lightning as pl
from torch.utils.data import random_split

class DNADataModule(pl.LightningDataModule):
    def __init__(self, file_path, batch_size=32):
        super().__init__()
        self.file_path = file_path
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Create full dataset
        full_dataset = Dataset(
            self.file_path,
            transform=Compose([
                Sampler(length=100, strategy="random"),
                OneHotEncoder()
            ])
        )

        # Split into train/val
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=4
        )

# Use with PyTorch Lightning Trainer
data_module = DNADataModule("data/sequences.fastq", batch_size=32)
trainer = pl.Trainer(max_epochs=10, accelerator="gpu", devices=1)
trainer.fit(model, data_module)
```

---

## Performance Tips

### 1. Use Multiple Workers for I/O-Bound Workloads

```python
# Single process (slow for large files)
loader_slow = DataLoader(dataset, batch_size=32, num_workers=0)

# Multi-process (faster, especially for many files)
loader_fast = DataLoader(dataset, batch_size=32, num_workers=4)
```

### 2. Enable Caching for Repeated Experiments

```python
# Cache encoded data to avoid re-encoding every run
dataset = Dataset(
    "data/sequences.fastq",
    transform=OneHotEncoder(),
    cache_dir="/tmp/cache"  # Speeds up subsequent runs by 10x
)
```

### 3. Use Lazy Loading for Large Datasets

```python
# Lazy loading (default): sequences loaded on-demand
dataset = Dataset("large_file.fastq", lazy=True)  # Low memory usage

# Eager loading: all sequences loaded into memory
dataset_eager = Dataset("small_file.fastq", lazy=False)  # Faster access
```

### 4. Pin Memory for GPU Training

```python
# When training on GPU, pin memory for faster CPU→GPU transfer
loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True  # Only useful when training on CUDA
)

for batch in loader:
    # Faster transfer to GPU
    sequences = torch.from_numpy(batch['sequences']).cuda(non_blocking=True)
```

---

## Troubleshooting

### Issue: "Index out of bounds" error

```python
# Check dataset size before indexing
print(f"Dataset size: {len(dataset)}")
sample = dataset[len(dataset) - 1]  # Last valid index
```

### Issue: "Invalid nucleotide" error

```python
# Validate your FASTQ file has correct format
dataset.summary()  # Check for data quality issues
```

### Issue: Slow batch generation

```python
# Use multiple workers and caching
loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,        # Parallel loading
    cache_dir="/tmp/cache"  # Cache processed data
)
```

### Issue: Out of memory

```python
# Reduce batch size or use lazy loading
loader = DataLoader(
    dataset,
    batch_size=16,  # Smaller batches
    num_workers=2   # Fewer workers
)

dataset = Dataset(file_path, lazy=True)  # On-demand loading
```

---

## Next Steps

1. **Explore transforms**: Try different encoders (one-hot, integer, k-mer) and augmentations
2. **Experiment with pipelines**: Combine multiple transforms with `Compose`
3. **Optimize performance**: Use caching and multi-worker loading
4. **Integrate with your model**: Drop-in replacement for `torch.utils.data.Dataset`

For more examples and advanced usage, see:
- **data-model.md**: Entity definitions and relationships
- **api-contracts.md**: Complete API reference with type annotations
- **research.md**: Technical decisions and implementation patterns

---

## Summary

The PyTorch-Style Python API provides:
- ✅ **Familiar PyTorch interface**: Dataset, DataLoader, Compose patterns
- ✅ **Biological sequence support**: FASTQ, FASTA files with zero manual parsing
- ✅ **Flexible transforms**: Encoding (one-hot, integer, k-mer) and augmentation (mutations, sampling)
- ✅ **High performance**: Parallel loading, caching, zero-copy NumPy↔PyTorch conversion
- ✅ **Easy integration**: Works seamlessly with existing PyTorch/Lightning code

**Code Example** (complete workflow in 15 lines):

```python
from deepbiop.pytorch import Dataset, DataLoader, Compose, Sampler, OneHotEncoder
import torch

# Load and transform data
dataset = Dataset("data.fastq", transform=Compose([
    Sampler(length=100), OneHotEncoder()
]))
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train model
model = torch.nn.Sequential(torch.nn.Conv1d(4, 64, 7), ...)
for batch in loader:
    sequences = torch.from_numpy(batch['sequences']).permute(0, 2, 1)
    outputs = model(sequences)
    # ... training logic
```

That's it! You're ready to use DeepBioP for deep learning on biological sequences.
