import sys

# Make submodules importable (e.g., "from deepbiop.fq import X")
# The Rust extension already creates these as attributes, we just need to register them
from deepbiop import deepbiop as _deepbiop_ext

# Import everything from the Rust extension first
from deepbiop.deepbiop import *  # noqa: F403

# Now OVERRIDE PyTorch-compatible transforms with explicit assignment
# Note: We must use explicit assignment because Python import semantics don't
# allow later imports to override wildcard imports
# The pytorch module has the __call__ interface we need for transforms
try:
    from deepbiop import pytorch as _pytorch_module

    # Explicitly override the low-level transforms with PyTorch-compatible ones
    IntegerEncoder = _pytorch_module.IntegerEncoder
    KmerEncoder = _pytorch_module.KmerEncoder
    Mutator = _pytorch_module.Mutator
    OneHotEncoder = _pytorch_module.OneHotEncoder
    ReverseComplement = _pytorch_module.ReverseComplement
    Sampler = _pytorch_module.Sampler
    _PytorchCompose = _pytorch_module.Compose

    _TRANSFORMS_AVAILABLE = True
except (ImportError, AttributeError):
    # Transforms not built yet - will use Python fallback Compose
    _TRANSFORMS_AVAILABLE = False

# Import core data structures (pure Python)
from deepbiop.core import Record

# Import base abstractions (pure Python)
from deepbiop.dataset import Dataset

# Import Lightning module (pure Python, not from Rust)
from deepbiop.lightning import BiologicalDataModule

# Import transform composition utilities (pure Python)
from deepbiop.transforms import Compose, FilterCompose, Transform, TransformDataset

# Import target extraction utilities for supervised learning (pure Python)
from deepbiop.targets import (
    TargetExtractor,
    create_classification_extractor,
    get_builtin_extractor,
)

# Import collate functions for PyTorch DataLoader (pure Python)
from deepbiop.collate import default_collate, get_collate_fn, supervised_collate, tensor_collate

# Import PyTorch-compatible dataset wrappers (pure Python)
from deepbiop.datasets import FastaDataset, FastqDataset

# Register submodules in sys.modules so they can be imported with "from deepbiop.fq import"
# Only register modules that actually exist
for _module_name in ["fq", "fa", "bam", "core", "utils", "vcf", "gtf", "pytorch"]:
    if hasattr(_deepbiop_ext, _module_name):
        sys.modules[f"deepbiop.{_module_name}"] = getattr(_deepbiop_ext, _module_name)

# Import filter classes from Rust fq module
try:
    from deepbiop.fq import LengthFilter, QualityFilter
    _FILTERS_AVAILABLE = True
except (ImportError, AttributeError):
    _FILTERS_AVAILABLE = False

__all__ = [
    # Core data structures
    "Record",
    "Dataset",
    "Transform",
    # Dataset implementations
    "FastqDataset",
    "FastaDataset",
    # Lightning integration
    "BiologicalDataModule",
    # Transform composition
    "Compose",
    "FilterCompose",
    "TransformDataset",
    # Augmentation transforms (Rust-based)
    "ReverseComplement",
    "Mutator",
    "Sampler",
    # Encoders (Rust-based)
    "OneHotEncoder",
    "IntegerEncoder",
    "KmerEncoder",
    # Filters (Rust-based)
    "LengthFilter",
    "QualityFilter",
    # Target extraction for supervised learning
    "TargetExtractor",
    "get_builtin_extractor",
    "create_classification_extractor",
    # Collate functions
    "default_collate",
    "supervised_collate",
    "tensor_collate",
    "get_collate_fn",
]
