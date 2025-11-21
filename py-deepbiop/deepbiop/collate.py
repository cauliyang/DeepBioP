"""Collate functions for batching biological sequence data with PyTorch.

This module provides collate functions compatible with PyTorch DataLoader
for handling variable-length sequences and supervised learning targets.
"""

from typing import Any


def default_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Default collate function for biological sequence batches.

    Preserves variable-length sequences as lists and stacks numpy arrays.
    This is the identity collate function - returns batch as-is for maximum flexibility.

    Args:
        batch (list[dict[str, Any]]): List of samples from dataset

    Returns:
        Batch as list of samples (identity function)

    Example:
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(dataset, batch_size=4, collate_fn=default_collate)
    """
    return batch


def supervised_collate(batch: list[dict[str, Any]]) -> dict[str, list | Any]:
    """Collate function for supervised learning with biological sequences.

    Separates features and targets for easy use with PyTorch training loops.
    Handles both dict-based samples and tuple-based samples.

    Args:
        batch (list[dict[str, Any]]): List of samples, each being either:
            - dict with keys like {"features": ..., "target": ..., "id": ...}
            - tuple of (features, target)

    Returns:
        Dictionary with structured batch:
        - "features": List of feature arrays/tensors
        - "targets": List of targets
        - "ids": List of sequence IDs (if available)
        - "sequences": List of raw sequences (if available)

    Example:
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(dataset, batch_size=32, collate_fn=supervised_collate)
        >>> for batch in loader:
        ...     features = batch["features"]  # List or stacked tensor
        ...     targets = batch["targets"]  # List of targets
        ...     loss = criterion(model(features), targets)
    """
    # Handle tuple-based samples (features, target)
    if isinstance(batch[0], tuple):
        features = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        return {
            "features": features,
            "targets": targets,
        }

    # Handle dict-based samples
    result = {}

    # Extract features if present
    if "features" in batch[0]:
        result["features"] = [item["features"] for item in batch]
    elif "sequence" in batch[0]:
        # Use raw sequence as features if no encoding applied
        result["sequences"] = [item["sequence"] for item in batch]

    # Extract targets
    if "target" in batch[0]:
        result["targets"] = [item["target"] for item in batch]

    # Extract IDs for tracking
    if "id" in batch[0]:
        result["ids"] = [item["id"] for item in batch]

    # Preserve quality scores if present
    if "quality" in batch[0]:
        result["quality"] = [item["quality"] for item in batch]

    # Preserve metadata if present
    if "metadata" in batch[0]:
        result["metadata"] = [item["metadata"] for item in batch]

    return result


def tensor_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate function that converts features and targets to PyTorch tensors.

    Requires numpy or PyTorch to be installed. Stacks features and targets
    into tensors for direct use with PyTorch models.

    Args:
        batch (list[dict[str, Any]]): List of samples with "features" and "target" keys

    Returns:
        Dictionary with:
        - "features": Stacked tensor of shape (batch_size, ...)
        - "targets": Tensor of targets of shape (batch_size,) or (batch_size, num_classes)
        - "ids": List of sequence IDs (if available)

    Example:
        >>> import torch
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(dataset, batch_size=32, collate_fn=tensor_collate)
        >>> for batch in loader:
        ...     features = batch["features"]  # torch.Tensor
        ...     targets = batch["targets"]  # torch.Tensor
        ...     outputs = model(features)
        ...     loss = criterion(outputs, targets)

    Raises:
        ImportError: If PyTorch is not installed
    """
    try:
        import torch
    except ImportError as e:
        msg = "PyTorch not installed. Install with: pip install torch"
        raise ImportError(msg) from e

    # Handle tuple-based samples
    if isinstance(batch[0], tuple):
        features = torch.stack([torch.as_tensor(item[0]) for item in batch])
        targets = torch.tensor([item[1] for item in batch])
        return {
            "features": features,
            "targets": targets,
        }

    # Handle dict-based samples
    result = {}

    # Stack features
    if "features" in batch[0]:
        features_list = [torch.as_tensor(item["features"]) for item in batch]
        result["features"] = torch.stack(features_list)

    # Stack targets
    if "target" in batch[0]:
        targets_list = [item["target"] for item in batch]
        # Handle different target types
        if isinstance(targets_list[0], list | tuple):
            # Multi-dimensional targets
            result["targets"] = torch.tensor(targets_list)
        else:
            # Scalar targets
            result["targets"] = torch.tensor(targets_list)

    # Keep IDs as list
    if "id" in batch[0]:
        result["ids"] = [item["id"] for item in batch]

    # Keep sequences as list (variable length)
    if "sequence" in batch[0]:
        result["sequences"] = [item["sequence"] for item in batch]

    return result


def get_collate_fn(mode: str = "default"):
    """Get a collate function by name.

    Args:
        mode (str): Collate mode
            - "default": Identity function, returns batch as-is
            - "supervised": Separates features and targets
            - "tensor": Converts to PyTorch tensors

    Returns:
        Collate function suitable for PyTorch DataLoader

    Example:
        >>> collate_fn = get_collate_fn("supervised")
        >>> loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
    """
    if mode == "default":
        return default_collate
    elif mode == "supervised":
        return supervised_collate
    elif mode == "tensor":
        return tensor_collate
    else:
        msg = f"Unknown collate mode: {mode}. Choose from: default, supervised, tensor"
        raise ValueError(msg)
