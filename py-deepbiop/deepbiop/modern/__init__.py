"""Modern PyTorch-compatible API for DeepBioP.

This module provides a modern, shape-safe interface for biological sequence
encoding and transformation using einops and jaxtyping. It wraps the Rust-backed
encoders with convenient utilities for PyTorch/ML workflows.

Key Features
------------
- **Shape Safety**: Runtime shape validation with jaxtyping
- **Clear Transformations**: Self-documenting einops operations
- **Architecture Ready**: Helpers for Conv1d, Transformers, RNNs
- **Type Annotations**: Shape-aware type aliases for better IDE support

Examples:
--------
>>> from deepbiop.modern import OneHotEncoder, to_channels_first
>>> import numpy as np
>>>
>>> # Encode sequences with shape safety
>>> encoder = OneHotEncoder("dna")
>>> batch = encoder.encode_batch([b"ACGT", b"GCTA"])
>>> print(batch.shape)  # (2, 4, 4) - [batch, length, alphabet]
(2, 4, 4)
>>>
>>> # Rearrange for PyTorch Conv1d
>>> conv_input = encoder.for_conv1d(batch)
>>> print(conv_input.shape)  # (2, 4, 4) - [batch, channels, length]
(2, 4, 4)
>>>
>>> # Or use standalone transform
>>> conv_input = to_channels_first(batch)

Modules
-------
types
    Type aliases with shape annotations for sequences, batches, and metadata
encoders
    Modern encoder wrappers (OneHotEncoder, IntegerEncoder) with shape safety
transforms
    Tensor rearrangement utilities using einops

See Also:
--------
deepbiop.fq : Core FASTQ processing with Rust backend
deepbiop.fa : FASTA file operations
"""

from __future__ import annotations

# Import encoders
from deepbiop.modern.encoders import (
    IntegerEncoder,
    OneHotEncoder,
)

# Import transforms
from deepbiop.modern.transforms import (
    batch_sequences,
    create_attention_mask,
    create_padding_mask,
    pool_sequences,
    reverse_complement,
    to_channels_first,
    to_channels_last,
    unbatch_sequences,
)

# Import type aliases
from deepbiop.modern.types import (
    DEFAULT_PAD_VALUE,
    # Constants
    DNA_ALPHABET_SIZE,
    PROTEIN_ALPHABET_SIZE,
    RNA_ALPHABET_SIZE,
    AttentionMask,
    ChannelsFirstBatch,
    ClassificationTargets,
    # Sequence types
    DNASequence,
    # Batch types
    IntegerBatch,
    KmerBatch,
    MultiClassTargets,
    OneHotBatch,
    OneHotSequence,
    PaddingMask,
    ProteinSequence,
    QualityScores,
    # Target types
    RegressionTargets,
    # Metadata types
    SequenceLengths,
)

__all__ = [
    "DEFAULT_PAD_VALUE",
    # Constants
    "DNA_ALPHABET_SIZE",
    "PROTEIN_ALPHABET_SIZE",
    "RNA_ALPHABET_SIZE",
    "AttentionMask",
    "ChannelsFirstBatch",
    "ClassificationTargets",
    # Type aliases - Sequences
    "DNASequence",
    # Type aliases - Batches
    "IntegerBatch",
    "IntegerEncoder",
    "KmerBatch",
    "MultiClassTargets",
    "OneHotBatch",
    # Encoders
    "OneHotEncoder",
    "OneHotSequence",
    "PaddingMask",
    "ProteinSequence",
    "QualityScores",
    # Type aliases - Targets
    "RegressionTargets",
    # Type aliases - Metadata
    "SequenceLengths",
    "batch_sequences",
    "create_attention_mask",
    "create_padding_mask",
    "pool_sequences",
    "reverse_complement",
    # Transforms
    "to_channels_first",
    "to_channels_last",
    "unbatch_sequences",
]

# Version info
__version__ = "0.1.0"  # Initial modern API version
