"""Modern encoders with shape safety and einops rearrangements.

This module provides modern wrappers around the Rust-backed encoders
with explicit shape annotations, runtime validation, and convenient
rearrangement utilities for different model architectures.

Examples:
--------
>>> from deepbiop.modern.encoders import OneHotEncoder
>>> import numpy as np
>>>
>>> # Create encoder with shape safety
>>> encoder = OneHotEncoder(encoding_type="dna")
>>>
>>> # Encode single sequence
>>> sequence = b"ACGTACGT"
>>> encoded = encoder.encode(sequence)  # Returns Float[Array, "8 4"]
>>>
>>> # Encode batch
>>> sequences = [b"ACGT", b"GCTA", b"TTAA"]
>>> batch = encoder.encode_batch(sequences)  # Returns Float[Array, "3 4 4"]
>>>
>>> # Rearrange for Conv1d (channels before length)
>>> conv_input = encoder.for_conv1d(batch)  # Returns Float[Array, "3 4 4"]
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np
from einops import rearrange

if TYPE_CHECKING:
    from jaxtyping import Float


# Try to import jaxtyping for runtime validation (optional)
try:
    from jaxtyping import jaxtyped
    from typeguard import typechecked

    HAS_JAXTYPING = True
except ImportError:
    HAS_JAXTYPING = False

    # Define dummy decorators if jaxtyping not available
    def jaxtyped(*, typechecker=None):
        """Dummy decorator when jaxtyping not available."""

        def decorator(func):
            return func

        return decorator

    def typechecked(func):
        """Dummy decorator when typeguard not available."""
        return func


# Import Rust-backed encoders
try:
    from deepbiop.fq import IntegerEncoder as RustIntegerEncoder
    from deepbiop.fq import OneHotEncoder as RustOneHotEncoder
except ImportError:
    RustOneHotEncoder = None
    RustIntegerEncoder = None
    warnings.warn(
        "Rust encoders not available. Install deepbiop with all features.",
        ImportWarning,
        stacklevel=2,
    )


class OneHotEncoder:
    """Modern one-hot encoder with shape safety and einops rearrangements.

    This class wraps the Rust-backed OneHotEncoder and adds:
    - Explicit shape annotations with jaxtyping
    - Runtime shape validation (when enabled)
    - Convenient rearrangement methods for different architectures
    - Self-documenting API through type hints

    Parameters
    ----------
    encoding_type : {"dna", "rna", "protein"}
        Type of biological sequence to encode
    ambiguous_strategy : {"skip", "mask", "random"}, default="mask"
        How to handle ambiguous bases:
        - "skip": Skip sequences with ambiguous bases
        - "mask": Replace ambiguous bases with zeros
        - "random": Replace with random valid bases
    validate_shapes : bool, default=False
        Enable runtime shape validation (requires jaxtyping + typeguard)
    seed : int | None, default=None
        Random seed for reproducible random replacements

    Examples:
    --------
    >>> encoder = OneHotEncoder("dna")
    >>> sequence = b"ACGT"
    >>> encoded = encoder.encode(sequence)
    >>> print(encoded.shape)  # (4, 4) - [length, alphabet]
    (4, 4)

    >>> # Encode batch with automatic padding
    >>> sequences = [b"ACG", b"TGCA", b"AT"]
    >>> batch = encoder.encode_batch(sequences)
    >>> print(batch.shape)  # (3, 4, 4) - [batch, max_length, alphabet]
    (3, 4, 4)

    >>> # Rearrange for PyTorch Conv1d
    >>> conv_input = encoder.for_conv1d(batch)
    >>> print(conv_input.shape)  # (3, 4, 4) - [batch, channels, length]
    (3, 4, 4)
    """

    def __init__(
        self,
        encoding_type: Literal["dna", "rna", "protein"] = "dna",
        ambiguous_strategy: Literal["skip", "mask", "random"] = "mask",
        *,
        validate_shapes: bool = False,
        seed: int | None = None,
    ):
        """Initialize modern one-hot encoder."""
        if RustOneHotEncoder is None:
            msg = (
                "Rust OneHotEncoder not available. Install deepbiop with all features."
            )
            raise ImportError(msg)

        self.encoding_type = encoding_type
        self.ambiguous_strategy = ambiguous_strategy
        self.validate_shapes = validate_shapes and HAS_JAXTYPING
        self.seed = seed

        # Create Rust-backed encoder
        self._rust_encoder = RustOneHotEncoder(
            encoding_type=encoding_type,
            ambiguous_strategy=ambiguous_strategy,
        )

        # Determine alphabet size
        if encoding_type in ("dna", "rna"):
            self.alphabet_size = 4
        elif encoding_type == "protein":
            self.alphabet_size = 20
        else:
            msg = f"Unknown encoding type: {encoding_type}"
            raise ValueError(msg)

    def encode(self, sequence: bytes) -> np.ndarray:
        """Encode single sequence to one-hot matrix.

        Parameters
        ----------
        sequence : bytes
            Biological sequence to encode

        Returns:
        -------
        OneHotSequence
            One-hot encoded matrix of shape [length, alphabet_size]

        Examples:
        --------
        >>> encoder = OneHotEncoder("dna")
        >>> encoded = encoder.encode(b"ACGT")
        >>> print(encoded.shape)
        (4, 4)
        """
        if self.validate_shapes and HAS_JAXTYPING:
            return self._encode_validated(sequence)
        return self._rust_encoder.encode(sequence)

    @jaxtyped(typechecker=typechecked)
    def _encode_validated(
        self, sequence: bytes
    ) -> Float[np.ndarray, "length alphabet"]:  # noqa: F722
        """Encode with runtime shape validation."""
        return self._rust_encoder.encode(sequence)

    def encode_batch(
        self, sequences: list[bytes], *, pad_value: float = 0.0
    ) -> np.ndarray:
        """Encode batch of sequences with automatic padding.

        Parameters
        ----------
        sequences : list[bytes]
            List of biological sequences to encode
        pad_value : float, default=0.0
            Value to use for padding shorter sequences

        Returns:
        -------
        OneHotBatch
            Batch of one-hot encoded sequences
            Shape: [batch_size, max_length, alphabet_size]

        Examples:
        --------
        >>> encoder = OneHotEncoder("dna")
        >>> batch = encoder.encode_batch([b"ACG", b"TGCA"])
        >>> print(batch.shape)
        (2, 4, 4)
        """
        if self.validate_shapes and HAS_JAXTYPING:
            return self._encode_batch_validated(sequences, pad_value=pad_value)

        # Encode all sequences
        encoded = [self._rust_encoder.encode(seq) for seq in sequences]

        # Find max length
        max_len = max(e.shape[0] for e in encoded)

        # Pad to max length
        batch = np.full(
            (len(sequences), max_len, self.alphabet_size), pad_value, dtype=np.float32
        )

        for i, enc in enumerate(encoded):
            length = enc.shape[0]
            batch[i, :length] = enc

        return batch

    @jaxtyped(typechecker=typechecked)
    def _encode_batch_validated(
        self, sequences: list[bytes], *, pad_value: float = 0.0
    ) -> Float[np.ndarray, "batch max_length alphabet"]:  # noqa: F722
        """Encode batch with runtime shape validation."""
        return self.encode_batch(sequences, pad_value=pad_value)

    def for_conv1d(self, batch: np.ndarray) -> np.ndarray:
        """Rearrange batch for PyTorch Conv1d (channels before length).

        Conv1d expects input shape: [batch, channels, length]
        This rearranges from: [batch, length, alphabet] -> [batch, alphabet, length]

        Parameters
        ----------
        batch : OneHotBatch
            Batch in default format [batch, length, alphabet]

        Returns:
        -------
        ChannelsFirstBatch
            Rearranged batch [batch, channels=alphabet, length]

        Examples:
        --------
        >>> encoder = OneHotEncoder("dna")
        >>> batch = encoder.encode_batch([b"ACGT", b"GCTA"])
        >>> conv_input = encoder.for_conv1d(batch)
        >>> print(conv_input.shape)  # (2, 4, 4) - batch, channels, length
        (2, 4, 4)
        """
        if self.validate_shapes and HAS_JAXTYPING:
            return self._for_conv1d_validated(batch)
        return rearrange(batch, "b l a -> b a l")

    @jaxtyped(typechecker=typechecked)
    def _for_conv1d_validated(
        self,
        batch: Float[np.ndarray, "batch length alphabet"],  # noqa: F722
    ) -> Float[np.ndarray, "batch alphabet length"]:  # noqa: F722
        """Rearrange with runtime shape validation."""
        return rearrange(batch, "b l a -> b a l")

    def for_transformer(self, batch: np.ndarray) -> np.ndarray:
        """Return batch as-is for transformer models.

        Transformers expect input shape: [batch, length, features]
        This is already the default format, so this is an identity operation.

        Parameters
        ----------
        batch : OneHotBatch
            Batch in default format [batch, length, alphabet]

        Returns:
        -------
        OneHotBatch
            Same batch (identity operation)

        Examples:
        --------
        >>> encoder = OneHotEncoder("dna")
        >>> batch = encoder.encode_batch([b"ACGT"])
        >>> transformer_input = encoder.for_transformer(batch)
        >>> assert np.array_equal(batch, transformer_input)
        """
        return batch

    def for_rnn(
        self, batch: np.ndarray, lengths: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Prepare batch for RNN models with sequence lengths.

        RNNs expect input shape: [batch, length, features]
        This returns the batch and length array for PackedSequence.

        Parameters
        ----------
        batch : OneHotBatch
            Batch in default format [batch, length, alphabet]
        lengths : np.ndarray | None
            Array of actual sequence lengths (before padding)
            If None, computes from non-zero positions

        Returns:
        -------
        tuple[OneHotBatch, SequenceLengths | None]
            (batch, lengths) tuple ready for RNN/PackedSequence

        Examples:
        --------
        >>> encoder = OneHotEncoder("dna")
        >>> sequences = [b"ACG", b"TGCA"]  # lengths: 3, 4
        >>> batch = encoder.encode_batch(sequences)
        >>> rnn_batch, lengths = encoder.for_rnn(batch)
        >>> print(lengths)  # [3, 4]
        [3 4]
        """
        if lengths is None:
            # Compute lengths from non-zero rows (sum over alphabet dimension)
            non_zero = (batch.sum(axis=-1) > 0).sum(axis=-1)
            lengths = non_zero.astype(np.int64)

        return batch, lengths

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ModernOneHotEncoder("
            f"type='{self.encoding_type}', "
            f"alphabet_size={self.alphabet_size}, "
            f"strategy='{self.ambiguous_strategy}', "
            f"validate_shapes={self.validate_shapes})"
        )


class IntegerEncoder:
    """Modern integer encoder with shape safety.

    This class wraps the Rust-backed IntegerEncoder and adds:
    - Explicit shape annotations with jaxtyping
    - Runtime shape validation (when enabled)
    - Batch encoding with padding

    Parameters
    ----------
    encoding_type : {"dna", "rna", "protein"}
        Type of biological sequence to encode
    validate_shapes : bool, default=False
        Enable runtime shape validation (requires jaxtyping + typeguard)

    Examples:
    --------
    >>> encoder = IntegerEncoder("dna")
    >>> sequence = b"ACGT"
    >>> encoded = encoder.encode(sequence)
    >>> print(encoded)  # [0, 1, 2, 3] (A=0, C=1, G=2, T=3)
    [0. 1. 2. 3.]
    """

    def __init__(
        self,
        encoding_type: Literal["dna", "rna", "protein"] = "dna",
        *,
        validate_shapes: bool = False,
    ):
        """Initialize modern integer encoder."""
        if RustIntegerEncoder is None:
            msg = (
                "Rust IntegerEncoder not available. Install deepbiop with all features."
            )
            raise ImportError(msg)

        self.encoding_type = encoding_type
        self.validate_shapes = validate_shapes and HAS_JAXTYPING

        # Create Rust-backed encoder
        self._rust_encoder = RustIntegerEncoder(encoding_type=encoding_type)

    def encode(self, sequence: bytes) -> np.ndarray:
        """Encode single sequence to integer array.

        Parameters
        ----------
        sequence : bytes
            Biological sequence to encode

        Returns:
        -------
        DNASequence | ProteinSequence
            Integer-encoded sequence of shape [length]

        Examples:
        --------
        >>> encoder = IntegerEncoder("dna")
        >>> encoded = encoder.encode(b"ACGT")
        >>> print(encoded)
        [0. 1. 2. 3.]
        """
        return self._rust_encoder.encode(sequence)

    def encode_batch(
        self, sequences: list[bytes], *, pad_value: float = -1.0
    ) -> np.ndarray:
        """Encode batch of sequences with padding.

        Parameters
        ----------
        sequences : list[bytes]
            List of biological sequences
        pad_value : float, default=-1.0
            Value for padding (negative to distinguish from valid indices)

        Returns:
        -------
        IntegerBatch
            Padded batch of shape [batch_size, max_length]

        Examples:
        --------
        >>> encoder = IntegerEncoder("dna")
        >>> batch = encoder.encode_batch([b"ACG", b"TGCA"])
        >>> print(batch)
        [[ 0.  1.  2. -1.]
         [ 3.  2.  1.  0.]]
        """
        # Encode all sequences
        encoded = [self._rust_encoder.encode(seq) for seq in sequences]

        # Find max length
        max_len = max(e.shape[0] for e in encoded)

        # Pad to max length
        batch = np.full((len(sequences), max_len), pad_value, dtype=np.float32)

        for i, enc in enumerate(encoded):
            length = enc.shape[0]
            batch[i, :length] = enc

        return batch

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ModernIntegerEncoder("
            f"type='{self.encoding_type}', "
            f"validate_shapes={self.validate_shapes})"
        )


__all__ = [
    "IntegerEncoder",
    "OneHotEncoder",
]
