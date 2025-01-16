# This file is automatically generated by pyo3_stub_gen
# ruff: noqa: E501, F401

import os
import pathlib
import typing

def count_chimeric_reads_for_path(bam,threads = ...) -> int:
    r"""Calculate the number of chimeric reads in a BAM file."""

def count_chimeric_reads_for_paths(bams,threads = ...) -> dict[str, int]:
    r"""Calculate the number of chimeric reads in multiple BAM files."""

def left_right_soft_clip(cigar_string:str) -> tuple[int, int]:
    r"""Calculate left and right soft clips from a cigar string."""

