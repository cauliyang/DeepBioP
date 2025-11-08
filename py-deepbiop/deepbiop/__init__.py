import sys

# Make submodules importable (e.g., "from deepbiop.fq import X")
# The Rust extension already creates these as attributes, we just need to register them
from deepbiop import deepbiop as _deepbiop_ext
from deepbiop.deepbiop import *  # noqa: F403

# Import Lightning module (pure Python, not from Rust)
from deepbiop.lightning import BiologicalDataModule

# Import transform composition utilities (pure Python)
from deepbiop.transforms import Compose, FilterCompose, TransformDataset

# Register submodules in sys.modules so they can be imported with "from deepbiop.fq import"
# Only register modules that actually exist
for _module_name in ["fq", "fa", "bam", "core", "utils", "vcf", "gtf", "pytorch"]:
    if hasattr(_deepbiop_ext, _module_name):
        sys.modules[f"deepbiop.{_module_name}"] = getattr(_deepbiop_ext, _module_name)

__all__ = ["BiologicalDataModule", "Compose", "FilterCompose", "TransformDataset"]
