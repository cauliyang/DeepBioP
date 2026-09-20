#!/usr/bin/env python3
"""Example demonstrating data export to ML formats (Parquet and NumPy).

This script shows how to use DeepBioP's export features to convert
biological sequence data into formats ready for machine learning:

- Parquet: `deepbiop.fq.encode_fq_path_to_parquet` writes a FASTQ file to a
  columnar Parquet file (SNAPPY-compressed) for pandas/polars/duckdb.
- NumPy: batch encoders return `numpy.ndarray`, so `.npy` output is one call
  to `numpy.save`.
"""

import tempfile
from pathlib import Path

import numpy as np

import deepbiop
from deepbiop import fq

SAMPLE_FASTQ = """@seq1
ACGTACGTACGT
+
IIIIIIIIIIII
@seq2
GGGGCCCCAAAA
+
!!!!!!!!!!!!
@seq3
TTTTAAAACCCCGGGG
+
###############
"""


def main():
    """Run export examples demonstrating the export formats."""
    print("DeepBioP Data Export Examples")
    print("=" * 60)

    ids = ["seq1", "seq2", "seq3"]
    sequences = [b"ACGTACGTACGT", b"GGGGCCCCAAAA", b"TTTTAAAACCCCGGGG"]
    print(f"\nSample Data: {len(ids)} sequences")
    for name, seq in zip(ids, sequences, strict=True):
        print(f"  {name}: {seq.decode()} ({len(seq)} bp)")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        fastq_path = tmpdir / "sample.fastq"
        fastq_path.write_text(SAMPLE_FASTQ)

        # 1. Export to Parquet
        print("\n\n1. Export to Parquet Format")
        print("-" * 60)
        parquet_path = tmpdir / "sample.parquet"
        fq.encode_fq_path_to_parquet(fastq_path, "ACGT", 33, parquet_path)
        print(f"Exported to Parquet: {parquet_path}")
        print(f"  File size: {parquet_path.stat().st_size} bytes")
        print("  Format: columnar storage with SNAPPY compression")
        print("  Use case: analytics with pandas/polars/duckdb")

        # 2. Export one-hot encoding to NumPy
        print("\n\n2. Export One-Hot Encoding to NumPy")
        print("-" * 60)
        onehot = fq.OneHotEncoder("dna", "skip").encode_batch(sequences)
        onehot_path = tmpdir / "onehot.npy"
        np.save(onehot_path, onehot)
        print(f"Saved one-hot array: {onehot_path}")
        print(f"  Shape: {onehot.shape} (sequences, max_length, 4)")
        print(f"  Dtype: {onehot.dtype}")

        # 3. Export integer encoding to NumPy
        print("\n\n3. Export Integer Encoding to NumPy")
        print("-" * 60)
        integer = fq.IntegerEncoder("dna").encode_batch(sequences)
        integer_path = tmpdir / "integer.npy"
        np.save(integer_path, integer)
        print(f"Saved integer array: {integer_path}")
        print(f"  Shape: {integer.shape} (sequences, max_length)")

        print("\n" + "=" * 60)
        print("All export examples completed!")
        print("=" * 60)


if __name__ == "__main__":
    main()
