"""DeepBioP: Deep Learning Preprocessing Library for Biological Data."""

# The Rust extension registers its submodules (fq, fa, bam, core, utils, vcf, gtf,
# pytorch) in sys.modules on import, so `from deepbiop.fq import ...` works.
from deepbiop.collate import (
    default_collate,
    get_collate_fn,
    multi_label_collate,
    multi_label_tensor_collate,
    supervised_collate,
    tensor_collate,
)
from deepbiop.dataset import Dataset
from deepbiop.datasets import BamDataset, FastaDataset, FastqDataset
from deepbiop.deepbiop import *  # noqa: F403
from deepbiop.deepbiop import fq, gtf, pytorch, vcf
from deepbiop.lightning import BiologicalDataModule
from deepbiop.record import Record
from deepbiop.targets import (
    MultiLabelExtractor,
    TargetExtractor,
    create_classification_extractor,
    get_builtin_extractor,
)
from deepbiop.transforms import Compose, FilterCompose, Transform, TransformDataset

# PyTorch-compatible transforms take precedence over the low-level fq/core
# encoders re-exported by the wildcard import above: they implement __call__.
IntegerEncoder = pytorch.IntegerEncoder
KmerEncoder = pytorch.KmerEncoder
Mutator = pytorch.Mutator
OneHotEncoder = pytorch.OneHotEncoder
ReverseComplement = pytorch.ReverseComplement
Sampler = pytorch.Sampler

LengthFilter = fq.LengthFilter
QualityFilter = fq.QualityFilter
GenomicFeature = gtf.GenomicFeature
GtfReader = gtf.GtfReader
Variant = vcf.Variant
VcfReader = vcf.VcfReader

__all__ = [
    "BamDataset",
    # Lightning integration
    "BiologicalDataModule",
    # Transform composition
    "Compose",
    "Dataset",
    "FastaDataset",
    # Dataset implementations
    "FastqDataset",
    "FilterCompose",
    # Genomic annotations (GTF)
    "GenomicFeature",
    "GtfReader",
    "IntegerEncoder",
    "KmerEncoder",
    # Filters (Rust-based)
    "LengthFilter",
    # Target extraction for supervised learning (multi-label)
    "MultiLabelExtractor",
    "Mutator",
    # Encoders (Rust-based)
    "OneHotEncoder",
    "QualityFilter",
    # Core data structures
    "Record",
    # Augmentation transforms (Rust-based)
    "ReverseComplement",
    "Sampler",
    # Target extraction for supervised learning
    "TargetExtractor",
    "Transform",
    "TransformDataset",
    # Genomic variants (VCF)
    "Variant",
    "VcfReader",
    "create_classification_extractor",
    # Collate functions
    "default_collate",
    "get_builtin_extractor",
    "get_collate_fn",
    "multi_label_collate",
    "multi_label_tensor_collate",
    "supervised_collate",
    "tensor_collate",
]
