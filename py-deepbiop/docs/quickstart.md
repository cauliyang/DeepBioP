# Quick Start: DeepBioP Deep Learning API

Get started with DeepBioP for deep learning on biological sequence data in under 5 minutes.

## Installation

```bash
# Install DeepBioP
pip install deepbiop

# Or with uv (faster)
uv pip install deepbiop

# Optional: Install deep learning frameworks
pip install torch pytorch-lightning transformers
```

## Basic Usage

### Loading FASTQ Data

```python
from deepbiop.fq import FastqStreamDataset
from torch.utils.data import DataLoader

# Create streaming dataset
dataset = FastqStreamDataset("data/sample.fastq.gz")

# Wrap in PyTorch DataLoader
loader = DataLoader(dataset, batch_size=32, num_workers=4)

# Iterate through batches
for batch in loader:
    # Each batch is a list of dicts
    for record in batch:
        seq = record['sequence']   # numpy array of sequence bytes
        qual = record['quality']    # numpy array of quality scores
        id = record['id']           # sequence identifier
        print(f"Sequence {id}: {len(seq)} bases")
```

### Loading FASTA and BAM Data

```python
from deepbiop.fa import FastaStreamDataset
from deepbiop.bam import BamStreamDataset

# FASTA dataset
fasta_dataset = FastaStreamDataset("data/genome.fasta.gz")

# BAM dataset with threading
bam_dataset = BamStreamDataset("data/alignments.bam", threads=4)

# Use with DataLoader as above
loader = DataLoader(fasta_dataset, batch_size=16)
```

## Data Transformations

### Filtering

```python
from deepbiop import fq, Compose, FilterCompose

# Create filter pipeline
filters = FilterCompose([
    fq.QualityFilter(min_mean_quality=30.0),     # Filter by quality
    fq.LengthFilter(min_length=50, max_length=500)  # Filter by length
])

# Apply during iteration
for record in dataset:
    if filters.filter(record):
        # Process high-quality, properly-sized sequences
        pass
```

### Data Augmentation

```python
from deepbiop import fq, Compose

# Create augmentation pipeline
augmentations = Compose([
    fq.ReverseComplement(),              # Randomly reverse complement
    fq.Mutator(mutation_rate=0.1)        # Add random mutations
])

# Apply transformations
for record in dataset:
    augmented = augmentations(record)
    # augmented['sequence'] has transforms applied
```

### Sequence Encoding

```python
from deepbiop import fq

# One-hot encoding
encoder = fq.OneHotEncoder(encoding_type="dna", unknown_strategy="skip")

for record in dataset:
    encoded = encoder(record)
    # encoded['sequence'] is now shape (length, 4) one-hot array
    # [[1,0,0,0],  # A
    #  [0,1,0,0],  # C
    #  [0,0,1,0],  # G
    #  [0,0,0,1]]  # T
```

### Combining Filters and Transforms

```python
from deepbiop import TransformDataset, Compose, FilterCompose, fq

# Create combined pipeline
dataset = fq.FastqStreamDataset("data/reads.fastq.gz")

# Wrap with filters and transforms
processed = TransformDataset(
    dataset,
    transform=Compose([
        fq.ReverseComplement(),
        fq.Mutator(mutation_rate=0.05)
    ]),
    filter_fn=FilterCompose([
        fq.QualityFilter(min_mean_quality=25.0),
        fq.LengthFilter(min_length=100)
    ])
)

# Only high-quality, long sequences with augmentations
for record in processed:
    pass
```

## PyTorch DataLoader Integration

### Custom Collate Function

```python
import torch
from torch.utils.data import DataLoader

def bio_collate_fn(batch):
    """Collate function with padding for variable-length sequences."""
    sequences = [torch.from_numpy(item['sequence']).long() for item in batch]

    # Pad to max length in batch
    max_len = max(seq.shape[0] for seq in sequences)
    padded = torch.zeros(len(sequences), max_len, dtype=torch.long)

    for i, seq in enumerate(sequences):
        padded[i, :seq.shape[0]] = seq

    return {
        'sequences': padded,
        'lengths': torch.tensor([seq.shape[0] for seq in sequences])
    }

# Use custom collate
loader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=bio_collate_fn,
    num_workers=4
)
```

### Training Loop Example

```python
import torch.nn as nn
import torch.optim as optim

# Simple model
model = nn.Sequential(
    nn.Embedding(256, 64),
    nn.LSTM(64, 128, batch_first=True),
    nn.Linear(128, 2)
)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Training
model.train()
for batch in loader:
    sequences = batch['sequences']

    # Forward pass
    outputs = model(sequences)
    loss = criterion(outputs, labels)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## PyTorch Lightning Integration

### Using BiologicalDataModule

```python
from deepbiop.lightning import BiologicalDataModule
import pytorch_lightning as pl

# Create data module
dm = BiologicalDataModule(
    train_path="data/train.fastq.gz",
    val_path="data/val.fastq.gz",
    test_path="data/test.fastq.gz",
    batch_size=64,
    num_workers=8
)

# Setup datasets
dm.setup(stage='fit')

# Get dataloaders
train_loader = dm.train_dataloader()
val_loader = dm.val_dataloader()

# Use with Lightning Trainer
trainer = pl.Trainer(
    max_epochs=10,
    accelerator='auto',  # Automatically use GPU if available
    devices=1
)

trainer.fit(model, dm)
```

### Automatic File Type Detection

```python
# BiologicalDataModule auto-detects file types
dm = BiologicalDataModule(
    train_path="train.fastq",     # FASTQ detected
    val_path="val.fasta.gz",      # FASTA detected
    test_path="test.bam",         # BAM detected
    batch_size=32
)
```

## Performance Tips

### Multiprocessing

```python
# Use multiple workers for faster data loading
loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=8,           # Parallel data loading
    pin_memory=True,         # Faster GPU transfer
    prefetch_factor=2        # Pre-load batches
)
```

### Distributed Training

```python
from torch.utils.data import DistributedSampler

# For multi-GPU training
sampler = DistributedSampler(dataset, num_replicas=4, rank=0)
loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

### Memory-Efficient Streaming

```python
# Streaming datasets don't load entire file into memory
dataset = fq.FastqStreamDataset("large_100gb.fastq.gz")

# Processes one record at a time - constant memory usage
for record in dataset:
    process(record)
```

## Next Steps

### Documentation
- **[API Reference](api-reference.md)**: Complete API documentation
- **[Performance Benchmarks](performance.md)**: Throughput and memory metrics
- **[Troubleshooting](troubleshooting.md)**: Common errors and solutions

### Examples
- **[PyTorch Training](../examples/pytorch_training.ipynb)**: Complete PyTorch integration examples
- **[Lightning Module](../examples/lightning_module.ipynb)**: PyTorch Lightning workflows
- **[Transformers Integration](../examples/transformers_dna.ipynb)**: Hugging Face Transformers with biological data

### Advanced Topics
- **Transform Composition**: Build complex preprocessing pipelines
- **Custom Encoders**: Implement your own encoding schemes
- **Caching** (coming soon): Preprocess and cache for 5-10x speedup
- **Rust API** (coming soon): Low-level Rust interface for maximum performance

## Common Patterns

### DNA Classification

```python
from deepbiop import fq, Compose, FilterCompose, TransformDataset

# Load and preprocess
dataset = fq.FastqStreamDataset("dna_reads.fastq.gz")

processed = TransformDataset(
    dataset,
    transform=Compose([
        fq.OneHotEncoder(encoding_type="dna"),
        fq.Mutator(mutation_rate=0.05)  # Data augmentation
    ]),
    filter_fn=fq.QualityFilter(min_mean_quality=30.0)
)

# Train model
loader = DataLoader(processed, batch_size=128, num_workers=4)
for batch in loader:
    # Train your classifier
    pass
```

### Quality Control Pipeline

```python
# Filter and sample high-quality reads
filters = FilterCompose([
    fq.QualityFilter(min_mean_quality=35.0),
    fq.LengthFilter(min_length=150, max_length=300),
])

sampler = fq.Sampler.random_fraction(0.1)  # Sample 10%

for record in dataset:
    if filters.filter(record) and sampler.passes(record['sequence']):
        # High-quality, properly-sized, sampled record
        write_output(record)
```

## Getting Help

- **Issues**: https://github.com/cauliyang/DeepBioP/issues
- **Discussions**: https://github.com/cauliyang/DeepBioP/discussions
- **Documentation**: https://deepbiop.readthedocs.io

---

**Ready to start?** Try the [quickstart notebook](../examples/quickstart.ipynb) for an interactive introduction!
