"""Tensor rearrangement utilities using einops for clear transformations.

This module provides convenient functions for common tensor rearrangements
needed when working with different model architectures (CNNs, Transformers, RNNs).
All functions use einops for self-documenting, readable transformations.

Examples:
--------
>>> from deepbiop.modern.transforms import to_channels_first, pool_sequences
>>> import numpy as np
>>>
>>> # Create sample batch: [batch=2, length=4, features=4]
>>> batch = np.random.randn(2, 4, 4)
>>>
>>> # Convert for Conv1d
>>> conv_input = to_channels_first(batch)
>>> print(conv_input.shape)  # (2, 4, 4) - [batch, channels, length]
(2, 4, 4)
>>>
>>> # Pool over sequence dimension
>>> pooled = pool_sequences(batch, method="mean")
>>> print(pooled.shape)  # (2, 4) - [batch, features]
(2, 4)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from einops import rearrange, reduce, repeat


def to_channels_first(batch: np.ndarray) -> np.ndarray:
    """Convert batch from channels-last to channels-first format.

    Useful for PyTorch Conv1d which expects [batch, channels, length].

    Parameters
    ----------
    batch : OneHotBatch
        Input batch in format [batch, length, features]

    Returns:
    -------
    ChannelsFirstBatch
        Rearranged batch [batch, features, length]

    Examples:
    --------
    >>> import numpy as np
    >>> batch = np.random.randn(8, 100, 4)  # [batch, length, features]
    >>> conv_input = to_channels_first(batch)
    >>> print(conv_input.shape)
    (8, 4, 100)
    """
    return rearrange(batch, "b l f -> b f l")


def to_channels_last(batch: np.ndarray) -> np.ndarray:
    """Convert batch from channels-first to channels-last format.

    Useful for converting Conv1d output back to standard format.

    Parameters
    ----------
    batch : ChannelsFirstBatch
        Input batch in format [batch, channels, length]

    Returns:
    -------
    OneHotBatch
        Rearranged batch [batch, length, channels]

    Examples:
    --------
    >>> import numpy as np
    >>> batch = np.random.randn(8, 4, 100)  # [batch, channels, length]
    >>> standard = to_channels_last(batch)
    >>> print(standard.shape)
    (8, 100, 4)
    """
    return rearrange(batch, "b f l -> b l f")


def pool_sequences(
    batch: np.ndarray, method: Literal["mean", "max", "min", "sum"] = "mean"
) -> np.ndarray:
    """Pool over sequence dimension to get fixed-size representations.

    Parameters
    ----------
    batch : OneHotBatch
        Input batch [batch, length, features]
    method : {"mean", "max", "min", "sum"}
        Pooling method to use

    Returns:
    -------
    np.ndarray
        Pooled batch [batch, features]

    Examples:
    --------
    >>> import numpy as np
    >>> batch = np.random.randn(8, 100, 64)
    >>>
    >>> # Global average pooling
    >>> pooled = pool_sequences(batch, method="mean")
    >>> print(pooled.shape)
    (8, 64)
    >>>
    >>> # Global max pooling
    >>> pooled = pool_sequences(batch, method="max")
    >>> print(pooled.shape)
    (8, 64)
    """
    return reduce(batch, "b l f -> b f", method)


def create_padding_mask(
    lengths: np.ndarray, max_length: int | None = None
) -> np.ndarray:
    """Create binary padding mask from sequence lengths.

    Parameters
    ----------
    lengths : np.ndarray
        Array of actual sequence lengths [batch]
    max_length : int | None
        Maximum sequence length. If None, uses max(lengths)

    Returns:
    -------
    PaddingMask
        Binary mask [batch, max_length] where 1=valid, 0=padding

    Examples:
    --------
    >>> lengths = np.array([3, 5, 2])
    >>> mask = create_padding_mask(lengths, max_length=6)
    >>> print(mask)
    [[1 1 1 0 0 0]
     [1 1 1 1 1 0]
     [1 1 0 0 0 0]]
    """
    if max_length is None:
        max_length = int(lengths.max())

    batch_size = len(lengths)
    mask = np.zeros((batch_size, max_length), dtype=bool)

    for i, length in enumerate(lengths):
        mask[i, :length] = True

    return mask


def create_attention_mask(
    lengths: np.ndarray, max_length: int | None = None, *, causal: bool = False
) -> np.ndarray:
    """Create attention mask for transformer models.

    Parameters
    ----------
    lengths : np.ndarray
        Array of actual sequence lengths [batch]
    max_length : int | None
        Maximum sequence length. If None, uses max(lengths)
    causal : bool, default=False
        If True, creates causal mask (no attending to future positions)

    Returns:
    -------
    AttentionMask
        Attention mask. Shape depends on causal parameter:
        - causal=False: [batch, max_length] - padding mask
        - causal=True: [batch, max_length, max_length] - causal + padding

    Examples:
    --------
    >>> lengths = np.array([3, 2])
    >>>
    >>> # Standard attention mask (padding only)
    >>> mask = create_attention_mask(lengths, max_length=4)
    >>> print(mask)
    [[1 1 1 0]
     [1 1 0 0]]
    >>>
    >>> # Causal attention mask
    >>> mask = create_attention_mask(lengths, max_length=3, causal=True)
    >>> print(mask[0])  # First sequence (length=3)
    [[1 0 0]
     [1 1 0]
     [1 1 1]]
    """
    if not causal:
        # Standard padding mask
        return create_padding_mask(lengths, max_length)

    # Causal mask: combine padding and causality
    if max_length is None:
        max_length = int(lengths.max())

    batch_size = len(lengths)

    # Create causal mask (lower triangular)
    causal_mask = np.tril(np.ones((max_length, max_length), dtype=bool))

    # Expand to batch
    batch_causal = repeat(causal_mask, "i j -> b i j", b=batch_size)

    # Create padding mask and expand
    padding_mask = create_padding_mask(lengths, max_length)
    padding_expanded = repeat(padding_mask, "b j -> b i j", i=max_length)

    # Combine: valid only if both causal and not padding
    combined_mask = batch_causal & padding_expanded

    return combined_mask


def batch_sequences(
    sequences: list[np.ndarray],
    *,
    pad_value: float = 0.0,
    align: Literal["left", "right"] = "left",
) -> tuple[np.ndarray, np.ndarray]:
    """Batch variable-length sequences with padding.

    Parameters
    ----------
    sequences : list[np.ndarray]
        List of sequences with shape [length, features] each
    pad_value : float, default=0.0
        Value to use for padding
    align : {"left", "right"}
        Alignment of sequences:
        - "left": Pad on the right (default)
        - "right": Pad on the left

    Returns:
    -------
    tuple[np.ndarray, np.ndarray]
        (batched_sequences, lengths)
        - batched_sequences: [batch, max_length, features]
        - lengths: [batch] - original lengths

    Examples:
    --------
    >>> seq1 = np.array([[1, 2], [3, 4]])  # length=2
    >>> seq2 = np.array([[5, 6], [7, 8], [9, 10]])  # length=3
    >>> batch, lengths = batch_sequences([seq1, seq2])
    >>> print(batch.shape)
    (2, 3, 2)
    >>> print(lengths)
    [2 3]
    """
    # Get dimensions
    batch_size = len(sequences)
    lengths = np.array([len(seq) for seq in sequences])
    max_len = int(lengths.max())
    features = sequences[0].shape[1] if sequences[0].ndim > 1 else 1

    # Create batch array
    if sequences[0].ndim == 1:
        batch = np.full((batch_size, max_len), pad_value, dtype=sequences[0].dtype)
    else:
        batch = np.full(
            (batch_size, max_len, features), pad_value, dtype=sequences[0].dtype
        )

    # Fill sequences
    for i, (seq, length) in enumerate(zip(sequences, lengths, strict=False)):
        if align == "left":
            batch[i, :length] = seq
        else:  # align == "right"
            batch[i, -length:] = seq

    return batch, lengths


def unbatch_sequences(batch: np.ndarray, lengths: np.ndarray) -> list[np.ndarray]:
    """Unbatch padded sequences back to variable-length list.

    Parameters
    ----------
    batch : np.ndarray
        Batched sequences [batch, max_length, features]
    lengths : np.ndarray
        Original sequence lengths [batch]

    Returns:
    -------
    list[np.ndarray]
        List of sequences without padding

    Examples:
    --------
    >>> batch = np.array([[[1, 2], [3, 4], [0, 0]], [[5, 6], [0, 0], [0, 0]]])
    >>> lengths = np.array([2, 1])
    >>> sequences = unbatch_sequences(batch, lengths)
    >>> print(sequences[0])
    [[1 2]
     [3 4]]
    >>> print(sequences[1])
    [[5 6]]
    """
    return [batch[i, :length] for i, length in enumerate(lengths)]


def reverse_complement(batch: np.ndarray, *, reverse_axis: int = 1) -> np.ndarray:
    """Reverse complement transformation for DNA/RNA sequences.

    For one-hot encoded sequences, reverses the sequence and swaps:
    - A <-> T (indices 0 <-> 3)
    - C <-> G (indices 1 <-> 2)

    Parameters
    ----------
    batch : OneHotBatch
        One-hot encoded batch [batch, length, alphabet=4]
    reverse_axis : int, default=1
        Axis along which to reverse (typically the length axis)

    Returns:
    -------
    OneHotBatch
        Reverse complemented batch [batch, length, alphabet=4]

    Examples:
    --------
    >>> # One-hot for "ACGT" is [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
    >>> # Reverse complement "ACGT" -> "ACGT" (complement: T,G,C,A -> reverse: A,C,G,T)
    >>> batch = np.array([one_hot_encode("ACGT")])
    >>> rc_batch = reverse_complement(batch)
    """
    # Reverse sequence
    reversed_batch = np.flip(batch, axis=reverse_axis)

    # Swap A<->T (0<->3) and C<->G (1<->2) on alphabet dimension
    # Create index array for swapping
    swap_indices = [3, 2, 1, 0]  # A(0)->T(3), C(1)->G(2), G(2)->C(1), T(3)->A(0)

    # Apply swap on last axis (alphabet)
    complemented = reversed_batch[..., swap_indices]

    return complemented


__all__ = [
    "batch_sequences",
    "create_attention_mask",
    "create_padding_mask",
    "pool_sequences",
    "reverse_complement",
    "to_channels_first",
    "to_channels_last",
    "unbatch_sequences",
]
