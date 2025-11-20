import sys

# Make submodules importable (e.g., "from deepbiop.fq import X")
# The Rust extension already creates these as attributes, we just need to register them
from deepbiop import deepbiop as _deepbiop_ext
from deepbiop.deepbiop import *  # noqa: F403

# Import Lightning module (pure Python, not from Rust)
from deepbiop.lightning import BiologicalDataModule

# Import transform composition utilities (pure Python)
from deepbiop.transforms import Compose, FilterCompose, TransformDataset

# Import target extraction utilities for supervised learning (pure Python)
from deepbiop.targets import (
    TargetExtractor,
    create_classification_extractor,
    get_builtin_extractor,
)

# Import collate functions for PyTorch DataLoader (pure Python)
from deepbiop.collate import default_collate, get_collate_fn, supervised_collate, tensor_collate

# Register submodules in sys.modules so they can be imported with "from deepbiop.fq import"
# Only register modules that actually exist
for _module_name in ["fq", "fa", "bam", "core", "utils", "vcf", "gtf", "pytorch"]:
    if hasattr(_deepbiop_ext, _module_name):
        sys.modules[f"deepbiop.{_module_name}"] = getattr(_deepbiop_ext, _module_name)

__all__ = [
    # Lightning integration
    "BiologicalDataModule",
    # Transform composition
    "Compose",
    "FilterCompose",
    "TransformDataset",
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
