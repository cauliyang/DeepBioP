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


def multi_label_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate function for multi-label/multi-task learning.

    Handles batches where targets can be dict, tuple, or list (from MultiLabelExtractor).
    Intelligently structures the batch based on target format.

    Args:
        batch (list[dict[str, Any]]): List of samples with multi-label targets

    Returns:
        Dictionary with structured batch:
        - "features": List or stacked tensor of features
        - "targets": Structured targets based on format:
            * Dict targets → dict mapping names to lists of values
            * Tuple/list targets → list of tuples/lists
        - "ids": List of sequence IDs (if available)

    Example:
        >>> # With dict targets from MultiLabelExtractor
        >>> loader = DataLoader(dataset, batch_size=32, collate_fn=multi_label_collate)
        >>> for batch in loader:
        ...     features = batch["features"]
        ...     targets = batch["targets"]  # {"quality": [q1, q2, ...], "gc": [gc1, gc2, ...]}
        ...     # Use targets["quality"] for one task, targets["gc"] for another

        >>> # With tuple/array targets
        >>> for batch in loader:
        ...     features = batch["features"]
        ...     targets = batch["targets"]  # [(t1_1, t1_2), (t2_1, t2_2), ...]
    """
    # Handle empty batch
    if not batch:
        return {}

    result = {}

    # Extract features
    if "features" in batch[0]:
        result["features"] = [item["features"] for item in batch]
    elif "sequence" in batch[0]:
        result["sequences"] = [item["sequence"] for item in batch]

    # Extract and structure multi-label targets
    if "target" in batch[0]:
        first_target = batch[0]["target"]

        if isinstance(first_target, dict):
            # Dict targets: {"quality": 33.75, "gc": 0.5, ...}
            # Restructure to: {"quality": [33.75, 34.1, ...], "gc": [0.5, 0.48, ...]}
            target_dict = {}
            for sample in batch:
                for key, value in sample["target"].items():
                    if key not in target_dict:
                        target_dict[key] = []
                    target_dict[key].append(value)
            result["targets"] = target_dict

        elif isinstance(first_target, (tuple, list)):
            # Tuple/list targets: keep as list of tuples/lists
            result["targets"] = [item["target"] for item in batch]

        else:
            # Scalar targets: treat as single-label
            result["targets"] = [item["target"] for item in batch]

    # Extract IDs for tracking
    if "id" in batch[0]:
        result["ids"] = [item["id"] for item in batch]

    # Preserve quality scores if present
    if "quality" in batch[0]:
        result["quality"] = [item["quality"] for item in batch]

    return result


def multi_label_tensor_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate function for multi-label learning with tensor conversion.

    Converts multi-label targets to PyTorch tensors for direct use with
    multi-task learning models.

    Args:
        batch (list[dict[str, Any]]): List of samples with multi-label targets

    Returns:
        Dictionary with:
        - "features": Stacked tensor of shape (batch_size, ...)
        - "targets": Structured tensors based on format:
            * Dict targets → dict mapping names to tensors
            * Tuple/list targets → stacked tensor of shape (batch_size, num_targets)
        - "ids": List of sequence IDs (if available)

    Example:
        >>> import torch
        >>> loader = DataLoader(dataset, batch_size=32, collate_fn=multi_label_tensor_collate)
        >>> for batch in loader:
        ...     features = batch["features"]  # torch.Tensor
        ...     targets = batch["targets"]  # {"quality": torch.Tensor, "gc": torch.Tensor}
        ...     # Multi-task forward pass
        ...     quality_pred = model.quality_head(features)
        ...     gc_pred = model.gc_head(features)
        ...     loss = criterion_quality(quality_pred, targets["quality"]) + \\
        ...            criterion_gc(gc_pred, targets["gc"])

    Raises:
        ImportError: If PyTorch is not installed
    """
    try:
        import torch
    except ImportError as e:
        msg = "PyTorch not installed. Install with: pip install torch"
        raise ImportError(msg) from e

    # Handle empty batch
    if not batch:
        return {}

    result = {}

    # Stack features
    if "features" in batch[0]:
        features_list = [torch.as_tensor(item["features"]) for item in batch]
        result["features"] = torch.stack(features_list)

    # Convert multi-label targets to tensors
    if "target" in batch[0]:
        first_target = batch[0]["target"]

        if isinstance(first_target, dict):
            # Dict targets: convert each target type to tensor
            # {"quality": [33.75, 34.1], "gc": [0.5, 0.48]} →
            # {"quality": tensor([33.75, 34.1]), "gc": tensor([0.5, 0.48])}
            target_dict = {}
            for sample in batch:
                for key, value in sample["target"].items():
                    if key not in target_dict:
                        target_dict[key] = []
                    target_dict[key].append(value)

            # Convert each list to tensor
            result["targets"] = {
                key: torch.tensor(values) for key, values in target_dict.items()
            }

        elif isinstance(first_target, (tuple, list)):
            # Tuple/list targets: stack into (batch_size, num_targets) tensor
            targets_list = [torch.tensor(item["target"]) for item in batch]
            result["targets"] = torch.stack(targets_list)

        else:
            # Scalar targets
            result["targets"] = torch.tensor([item["target"] for item in batch])

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
            - "multi_label": Multi-label/multi-task learning
            - "multi_label_tensor": Multi-label with tensor conversion

    Returns:
        Collate function suitable for PyTorch DataLoader

    Example:
        >>> collate_fn = get_collate_fn("supervised")
        >>> loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
        >>>
        >>> # For multi-label learning
        >>> collate_fn = get_collate_fn("multi_label_tensor")
        >>> loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
    """
    if mode == "default":
        return default_collate
    elif mode == "supervised":
        return supervised_collate
    elif mode == "tensor":
        return tensor_collate
    elif mode == "multi_label":
        return multi_label_collate
    elif mode == "multi_label_tensor":
        return multi_label_tensor_collate
    else:
        msg = (
            f"Unknown collate mode: {mode}. "
            "Choose from: default, supervised, tensor, multi_label, multi_label_tensor"
        )
        raise ValueError(msg)
