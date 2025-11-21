r"""Target extraction utilities for supervised learning with biological sequences.

This module provides flexible target/label extraction from FASTQ, FASTA, and BAM files
to enable supervised learning tasks like classification and regression.

Examples:
--------
    >>> # Extract from header metadata
    >>> extractor = TargetExtractor.from_header(r"label=(\\w+)")
    >>>
    >>> # Extract quality statistics
    >>> extractor = TargetExtractor.from_quality(stat="mean")
    >>>
    >>> # Load from external CSV
    >>> extractor = TargetExtractor.from_file(
    ...     "labels.csv", id_col="read_id", label_col="class"
    ... )
    >>>
    >>> # Custom extraction function
    >>> def custom_fn(record):
    ...     return 1 if b"positive" in record["id"] else 0
    >>> extractor = TargetExtractor(custom_fn)
"""

import csv
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any


class TargetExtractor:
    """Base class for extracting targets/labels from biological sequence records.

    A TargetExtractor takes a record dict (with keys like "id", "sequence", "quality")
    and returns a target value suitable for supervised learning (e.g., class label,
    regression value).

    Args:
        fn: Callable that takes a record dict and returns a target value

    Example:
        >>> def get_gc_content(record):
        ...     seq = record["sequence"]
        ...     gc = seq.count(b"G") + seq.count(b"C")
        ...     return gc / len(seq)
        >>>
        >>> extractor = TargetExtractor(get_gc_content)
        >>> target = extractor({"sequence": b"ACGTACGT"})
    """

    def __init__(self, fn: Callable[[dict[str, Any]], Any]):
        """Initialize TargetExtractor with a function.

        Args:
            fn: Function that extracts target from a record
        """
        self.fn = fn

    def __call__(self, record: dict[str, Any]) -> Any:
        """Extract target from a record.

        Args:
            record: Record dict with keys like "id", "sequence", "quality"

        Returns:
            Target value (int, float, list, etc.)
        """
        return self.fn(record)

    @staticmethod
    def from_header(
        pattern: str | None = None,
        key: str | None = None,
        separator: str = "|",
        converter: Callable[[str], Any] = str,
    ) -> "TargetExtractor":
        r"""Create extractor that parses target from FASTQ/FASTA header.

        Supports two modes:
        1. Regex pattern: Extract first group from pattern match
        2. Key-value: Parse header like "id|key:value|other:data"

        Args:
            pattern (str | None): Regex pattern with one capture group (e.g., r"class=(\\w+)")
            key (str | None): Key to extract from key:value pairs (e.g., "label")
            separator (str): Separator for key:value pairs (default: "|")
            converter (Callable[[str], Any]): Function to convert extracted string to target type

        Returns:
            TargetExtractor instance

        Example:
            >>> # Extract from pattern
            >>> extractor = TargetExtractor.from_header(r"label=(\d+)", converter=int)
            >>> target = extractor({"id": b"@read1 label=5"})
            >>> assert target == 5
            >>>
            >>> # Extract from key:value pairs
            >>> extractor = TargetExtractor.from_header(key="class", converter=int)
            >>> target = extractor({"id": b"@read1|class:1|score:0.95"})
            >>> assert target == 1
        """
        if pattern is not None:
            # Regex-based extraction
            regex = re.compile(pattern)

            def extract_pattern(record: dict[str, Any]) -> Any:
                header = (
                    record["id"].decode()
                    if isinstance(record["id"], bytes)
                    else record["id"]
                )
                match = regex.search(header)
                if match:
                    return converter(match.group(1))
                msg = f"Pattern '{pattern}' not found in header: {header}"
                raise ValueError(msg)

            return TargetExtractor(extract_pattern)

        elif key is not None:
            # Key:value extraction
            def extract_key(record: dict[str, Any]) -> Any:
                header = (
                    record["id"].decode()
                    if isinstance(record["id"], bytes)
                    else record["id"]
                )
                parts = header.split(separator)
                for part in parts:
                    if ":" in part:
                        k, v = part.split(":", 1)
                        if k.strip() == key:
                            return converter(v.strip())
                msg = f"Key '{key}' not found in header: {header}"
                raise ValueError(msg)

            return TargetExtractor(extract_key)

        else:
            msg = "Must provide either 'pattern' or 'key' parameter"
            raise ValueError(msg)

    @staticmethod
    def from_quality(
        stat: str = "mean",
        min_quality: int = 0,
        max_quality: int = 60,
    ) -> "TargetExtractor":
        """Create extractor that computes statistics from quality scores.

        Useful for quality prediction tasks or as regression targets.

        Args:
            stat (str): Statistic to compute ("mean", "median", "min", "max", "std")
            min_quality (int): Minimum quality threshold (for filtering)
            max_quality (int): Maximum quality threshold (for clipping)

        Returns:
            TargetExtractor instance

        Example:
            >>> extractor = TargetExtractor.from_quality(stat="mean")
            >>> target = extractor({"quality": [30, 32, 35, 38]})
            >>> assert 30 <= target <= 38
        """

        def extract_quality(record: dict[str, Any]) -> float:
            quality = record.get("quality")
            if quality is None:
                msg = "Record has no quality scores (FASTA file?)"
                raise ValueError(msg)

            # Convert from bytes/ASCII to Phred scores if needed
            if isinstance(quality, bytes):
                # FASTQ quality scores are ASCII: Phred+33 encoding
                quality = [ord(q) - 33 for q in quality.decode()]
            elif isinstance(quality, str):
                # Handle string quality
                quality = [ord(q) - 33 for q in quality]
            else:
                # Already a list of integers
                quality = list(quality)

            # Clip quality values and convert to float to avoid uint8 overflow
            quality = [float(max(min_quality, min(q, max_quality))) for q in quality]

            if stat == "mean":
                return sum(quality) / len(quality)
            elif stat == "median":
                sorted_q = sorted(quality)
                n = len(sorted_q)
                return (
                    sorted_q[n // 2]
                    if n % 2
                    else (sorted_q[n // 2 - 1] + sorted_q[n // 2]) / 2
                )
            elif stat == "min":
                return min(quality)
            elif stat == "max":
                return max(quality)
            elif stat == "std":
                mean = sum(quality) / len(quality)
                variance = sum((q - mean) ** 2 for q in quality) / len(quality)
                return variance**0.5
            else:
                msg = f"Unknown stat: {stat}. Choose from: mean, median, min, max, std"
                raise ValueError(msg)

        return TargetExtractor(extract_quality)

    @staticmethod
    def from_sequence(
        feature: str = "gc_content",
    ) -> "TargetExtractor":
        """Create extractor that computes features from sequence.

        Args:
            feature (str): Feature to compute ("gc_content", "length", "complexity")

        Returns:
            TargetExtractor instance

        Example:
            >>> extractor = TargetExtractor.from_sequence(feature="gc_content")
            >>> target = extractor({"sequence": b"ACGTACGT"})
            >>> assert target == 0.5  # 4 GC out of 8 bases
        """

        def extract_gc_content(record: dict[str, Any]) -> float:
            seq = record["sequence"]
            if isinstance(seq, bytes):
                gc_count = (
                    seq.count(b"G")
                    + seq.count(b"C")
                    + seq.count(b"g")
                    + seq.count(b"c")
                )
            else:
                gc_count = (
                    seq.count("G") + seq.count("C") + seq.count("g") + seq.count("c")
                )
            return gc_count / len(seq) if len(seq) > 0 else 0.0

        def extract_length(record: dict[str, Any]) -> int:
            return len(record["sequence"])

        def extract_complexity(record: dict[str, Any]) -> float:
            """Sequence complexity using Shannon entropy."""
            seq = record["sequence"]
            if isinstance(seq, bytes):
                seq = seq.decode()

            # Count base frequencies
            counts = {}
            for base in seq:
                counts[base] = counts.get(base, 0) + 1

            # Calculate Shannon entropy
            length = len(seq)
            entropy = 0.0
            for count in counts.values():
                if count > 0:
                    p = count / length
                    entropy -= p * (p**0.5).bit_length()  # Simple complexity measure

            return entropy

        if feature == "gc_content":
            return TargetExtractor(extract_gc_content)
        elif feature == "length":
            return TargetExtractor(extract_length)
        elif feature == "complexity":
            return TargetExtractor(extract_complexity)
        else:
            msg = f"Unknown feature: {feature}. Choose from: gc_content, length, complexity"
            raise ValueError(msg)

    @staticmethod
    def from_file(
        filepath: str,
        id_column: str = "id",
        label_column: str = "label",
        converter: Callable[[str], Any] = float,
    ) -> "TargetExtractor":
        """Create extractor that loads labels from external CSV/JSON file.

        Args:
            filepath (str): Path to CSV or JSON file containing labels
            id_column (str): Column name for sequence IDs (CSV) or key (JSON)
            label_column (str): Column name for labels (CSV) or key (JSON)
            converter (Callable[[str], Any]): Function to convert label strings to target type

        Returns:
            TargetExtractor instance

        Example:
            >>> # labels.csv:
            >>> # read_id,class,score
            >>> # read_1,0,0.95
            >>> # read_2,1,0.88
            >>>
            >>> extractor = TargetExtractor.from_file(
            ...     "labels.csv",
            ...     id_column="read_id",
            ...     label_column="class",
            ...     converter=int,
            ... )
            >>> target = extractor({"id": b"read_1"})
            >>> assert target == 0
        """
        filepath = Path(filepath)

        # Load labels into memory
        labels = {}

        if filepath.suffix == ".csv":
            with filepath.open() as f:
                reader = csv.DictReader(f)
                for row in reader:
                    seq_id = row[id_column]
                    label = converter(row[label_column])
                    labels[seq_id] = label

        elif filepath.suffix == ".json":
            with filepath.open() as f:
                data = json.load(f)
                for item in data:
                    seq_id = item[id_column]
                    label = converter(item[label_column])
                    labels[seq_id] = label

        else:
            msg = f"Unsupported file format: {filepath.suffix}. Use .csv or .json"
            raise ValueError(msg)

        def extract_from_file(record: dict[str, Any]) -> Any:
            # Extract sequence ID from record
            seq_id = record["id"]
            if isinstance(seq_id, bytes):
                seq_id = seq_id.decode()

            # Remove @ prefix if present (FASTQ format)
            seq_id = seq_id.lstrip("@").split()[0]

            if seq_id not in labels:
                msg = f"Sequence ID '{seq_id}' not found in label file"
                raise ValueError(msg)

            return labels[seq_id]

        return TargetExtractor(extract_from_file)

    @staticmethod
    def constant(value: Any) -> "TargetExtractor":
        """Create extractor that returns a constant value for all records.

        Useful for testing or when all samples have the same label.

        Args:
            value (Any): Constant value to return

        Returns:
            TargetExtractor instance
        """
        return TargetExtractor(lambda record: value)


class MultiLabelExtractor:
    """Extractor for multiple targets per record (multi-task learning).

    Combines multiple TargetExtractor instances to extract multiple targets
    from a single record. Useful for multi-task learning scenarios where you
    need to predict multiple properties simultaneously.

    Args:
        extractors: Dictionary mapping target names to extractors, or list of extractors
        output_format: Format for returned targets ("dict", "tuple", or "array")

    Example:
        >>> # Named targets (dict output)
        >>> extractor = MultiLabelExtractor(
        ...     {
        ...         "quality": TargetExtractor.from_quality(stat="mean"),
        ...         "gc": TargetExtractor.from_sequence(feature="gc_content"),
        ...         "length": TargetExtractor.from_sequence(feature="length"),
        ...     }
        ... )
        >>> targets = extractor({"sequence": b"ACGT", "quality": [30, 32, 35, 38]})
        >>> # Returns: {"quality": 33.75, "gc": 0.5, "length": 4}
        >>>
        >>> # Positional targets (tuple output)
        >>> extractor = MultiLabelExtractor(
        ...     [
        ...         TargetExtractor.from_quality(stat="mean"),
        ...         TargetExtractor.from_sequence(feature="gc_content"),
        ...     ],
        ...     output_format="tuple",
        ... )
        >>> targets = extractor(record)
        >>> # Returns: (33.75, 0.5)
        >>>
        >>> # Array output for tensor conversion
        >>> extractor = MultiLabelExtractor([...], output_format="array")
        >>> targets = extractor(record)
        >>> # Returns: [33.75, 0.5]
    """

    def __init__(
        self,
        extractors: dict[str, TargetExtractor] | list[TargetExtractor],
        output_format: str = "dict",
    ):
        """Initialize MultiLabelExtractor.

        Args:
            extractors: Dictionary mapping names to extractors, or list of extractors
            output_format: "dict", "tuple", or "array"
        """
        if output_format not in ["dict", "tuple", "array"]:
            msg = f"Invalid output_format: {output_format}. Choose from: dict, tuple, array"
            raise ValueError(msg)

        self.extractors = extractors
        self.output_format = output_format

        # Validate extractors
        if isinstance(extractors, dict):
            if not extractors:
                msg = "extractors dict cannot be empty"
                raise ValueError(msg)
            for name, ext in extractors.items():
                if not callable(ext):
                    msg = f"Extractor '{name}' must be callable"
                    raise TypeError(msg)
        elif isinstance(extractors, list):
            if not extractors:
                msg = "extractors list cannot be empty"
                raise ValueError(msg)
            for i, ext in enumerate(extractors):
                if not callable(ext):
                    msg = f"Extractor at index {i} must be callable"
                    raise TypeError(msg)
        else:
            msg = "extractors must be dict or list"
            raise TypeError(msg)

    def __call__(self, record: dict[str, Any]) -> dict | tuple | list:
        """Extract multiple targets from a record.

        Args:
            record: Record dict with keys like "id", "sequence", "quality"

        Returns:
            Targets in format specified by output_format:
            - "dict": Dictionary mapping target names to values
            - "tuple": Tuple of target values
            - "array": List of target values
        """
        if isinstance(self.extractors, dict):
            # Named extractors
            results = {name: ext(record) for name, ext in self.extractors.items()}

            if self.output_format == "dict":
                return results
            elif self.output_format == "tuple":
                # Return values in consistent order (sorted by key)
                return tuple(results[k] for k in sorted(results.keys()))
            else:  # array
                # Return values in consistent order (sorted by key)
                return [results[k] for k in sorted(results.keys())]

        else:  # list
            # Positional extractors
            results = [ext(record) for ext in self.extractors]

            if self.output_format == "tuple":
                return tuple(results)
            elif self.output_format == "array":
                return results
            else:  # dict
                # Create dict with generic keys
                return {f"target_{i}": val for i, val in enumerate(results)}

    @property
    def target_names(self) -> list[str] | None:
        """Get names of targets (if using named extractors).

        Returns:
            List of target names, or None if using positional extractors
        """
        if isinstance(self.extractors, dict):
            return sorted(self.extractors.keys())
        return None

    @property
    def num_targets(self) -> int:
        """Get number of targets.

        Returns:
            Number of targets extracted
        """
        if isinstance(self.extractors, dict):
            return len(self.extractors)
        return len(self.extractors)


# Convenience functions for common use cases


def get_builtin_extractor(name: str, **kwargs) -> TargetExtractor:
    """Get a built-in target extractor by name.

    Args:
        name (str): Name of built-in extractor
        **kwargs: Additional arguments for the extractor

    Available extractors:
        - "quality_mean", "quality_median", "quality_min", "quality_max", "quality_std"
        - "gc_content", "length", "complexity"

    Returns:
        TargetExtractor instance

    Example:
        >>> extractor = get_builtin_extractor("quality_mean")
        >>> # Equivalent to: TargetExtractor.from_quality(stat="mean")
    """
    if name.startswith("quality_"):
        stat = name.replace("quality_", "")
        return TargetExtractor.from_quality(stat=stat, **kwargs)
    elif name in ["gc_content", "length", "complexity"]:
        return TargetExtractor.from_sequence(feature=name, **kwargs)
    else:
        msg = f"Unknown built-in extractor: {name}"
        raise ValueError(msg)


def create_classification_extractor(
    classes: list[str],
    pattern: str | None = None,
    key: str | None = None,
) -> TargetExtractor:
    r"""Create extractor for multi-class classification.

    Converts class names to integer indices.

    Args:
        classes (list[str]): List of class names (order defines indices)
        pattern (str | None): Regex pattern to extract class name from header
        key (str | None): Key to extract class name from header key:value pairs

    Returns:
        TargetExtractor that returns class indices

    Example:
        >>> extractor = create_classification_extractor(
        ...     classes=["negative", "positive"], pattern=r"class=(\\w+)"
        ... )
        >>> # Header "@read1 class=positive" → target = 1
    """
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    base_extractor = TargetExtractor.from_header(
        pattern=pattern, key=key, converter=str
    )

    def classify(record: dict[str, Any]) -> int:
        class_name = base_extractor(record)
        if class_name not in class_to_idx:
            msg = f"Unknown class: {class_name}. Valid classes: {classes}"
            raise ValueError(msg)
        return class_to_idx[class_name]

    return TargetExtractor(classify)
