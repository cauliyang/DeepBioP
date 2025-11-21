"""Modern type annotations for DeepBioP with shape semantics.

This module provides type aliases using jaxtyping for self-documenting
shape-annotated tensor types. These make the API more readable and help
catch shape mismatches early through runtime validation.

Examples:
--------
>>> from deepbiop.modern.types import DNASequence, OneHotBatch
>>> import numpy as np
>>>
>>> # Clear shape semantics in function signatures
>>> def process_batch(sequences: OneHotBatch) -> Float[np.ndarray, "batch features"]:
...     # Function knows input is [batch, length, alphabet=4]
...     pass
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Try to import jaxtyping for runtime type annotations
try:
    from jaxtyping import Bool, Float, Int

    HAS_JAXTYPING = True
except ImportError:
    # If jaxtyping not available, create dummy type aliases
    HAS_JAXTYPING = False

    # Create identity function that just returns the base type
    def _make_alias(base_type: Any, shape: str) -> Any:
        """Create a simple type alias without jaxtyping."""
        return base_type

    # Mock jaxtyping types
    class _MockJaxType:
        """Mock jaxtyping type for when jaxtyping not installed."""

        def __class_getitem__(cls, item: tuple[Any, str]) -> Any:
            """Return the base type without shape annotation."""
            if isinstance(item, tuple):
                return item[0]
            return item

    Int = _MockJaxType()  # type: ignore[assignment]
    Float = _MockJaxType()  # type: ignore[assignment]
    Bool = _MockJaxType()  # type: ignore[assignment]

# ==============================================================================
# Sequence Type Aliases
# ==============================================================================

#: Integer-encoded DNA/RNA sequence (0=A, 1=C, 2=G, 3=T/U)
DNASequence = Int[np.ndarray, "length"]

#: Integer-encoded protein sequence (alphabetically ordered amino acids)
ProteinSequence = Int[np.ndarray, "length"]

#: Quality scores for a sequence (Phred scores)
QualityScores = Int[np.ndarray, "length"]

#: One-hot encoded single sequence
OneHotSequence = Float[np.ndarray, "length alphabet"]

# ==============================================================================
# Batch Type Aliases
# ==============================================================================

#: Batch of integer-encoded sequences (padded to max_length)
IntegerBatch = Float[np.ndarray, "batch max_length"]

#: Batch of one-hot encoded sequences
#: For DNA/RNA: alphabet=4 (A,C,G,T/U)
#: For Protein: alphabet=20 (standard amino acids)
OneHotBatch = Float[np.ndarray, "batch max_length alphabet"]

#: Batch of sequences in channels-first format for Conv1d
#: Shape: [batch, alphabet/channels, max_length]
ChannelsFirstBatch = Float[np.ndarray, "batch channels max_length"]

#: K-mer frequency vectors for a batch
#: vocab_size = alphabet_size^k (e.g., 4^6=4096 for DNA 6-mers)
KmerBatch = Float[np.ndarray, "batch vocab_size"]

# ==============================================================================
# Metadata Type Aliases
# ==============================================================================

#: Sequence lengths before padding (for PackedSequence)
SequenceLengths = Int[np.ndarray, " batch"]

#: Binary mask indicating valid positions (1) vs padding (0)
PaddingMask = Bool[np.ndarray, "batch max_length"]

#: Attention mask for transformers (1=attend, 0=ignore)
AttentionMask = Bool[np.ndarray, "batch max_length"]

# ==============================================================================
# Target Type Aliases
# ==============================================================================

#: Regression targets (single value per sequence)
RegressionTargets = Float[np.ndarray, " batch"]

#: Classification targets (class indices)
ClassificationTargets = Int[np.ndarray, " batch"]

#: Multi-class probabilities or one-hot labels
MultiClassTargets = Float[np.ndarray, "batch num_classes"]

# ==============================================================================
# Constants
# ==============================================================================

#: Standard DNA alphabet size
DNA_ALPHABET_SIZE: int = 4

#: Standard RNA alphabet size
RNA_ALPHABET_SIZE: int = 4

#: Standard protein alphabet size (20 canonical amino acids)
PROTEIN_ALPHABET_SIZE: int = 20

#: Default padding value for numerical arrays
DEFAULT_PAD_VALUE: float = 0.0

__all__ = [
    "DEFAULT_PAD_VALUE",
    # Constants
    "DNA_ALPHABET_SIZE",
    "PROTEIN_ALPHABET_SIZE",
    "RNA_ALPHABET_SIZE",
    "AttentionMask",
    "ChannelsFirstBatch",
    "ClassificationTargets",
    # Sequence types
    "DNASequence",
    # Batch types
    "IntegerBatch",
    "KmerBatch",
    "MultiClassTargets",
    "OneHotBatch",
    "OneHotSequence",
    "PaddingMask",
    "ProteinSequence",
    "QualityScores",
    # Target types
    "RegressionTargets",
    # Metadata types
    "SequenceLengths",
]
