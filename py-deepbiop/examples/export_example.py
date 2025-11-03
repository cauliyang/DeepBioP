#!/usr/bin/env python3
"""
Example demonstrating data export to ML formats (Parquet, NumPy, Arrow).

This script shows how to use DeepBioP's export features to convert
biological sequence data into formats ready for machine learning.
"""

import tempfile
from pathlib import Path

import deepbiop


def main():
    """Run export examples demonstrating all export formats."""
    print("DeepBioP Data Export Examples")
    print("=" * 60)

    # Sample data
    ids = ["seq1", "seq2", "seq3"]
    sequences = [b"ACGTACGTACGT", b"GGGGCCCCAAAA", b"TTTTAAAACCCCGGGG"]
    qualities = [b"IIIIIIIIIIII", b"!!!!!!!!!!", b"################"]

    print("\nSample Data:")
    print(f"  {len(ids)} sequences")
    for _i, (id, seq) in enumerate(zip(ids, sequences, strict=False)):
        print(f"  {id}: {seq.decode()} ({len(seq)} bp)")

    # 1. Export to Parquet
    print("\n\n1. Export to Parquet Format")
    print("-" * 60)

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_parquet:
        parquet_path = tmp_parquet.name

    try:
        # Export FASTQ data to Parquet
        deepbiop.utils.export_to_parquet(parquet_path, ids, sequences, qualities)

        file_size = Path(parquet_path).stat().st_size
        print(f"✓ Exported to Parquet: {parquet_path}")
        print(f"  File size: {file_size} bytes")
        print("  Format: Columnar storage with SNAPPY compression")
        print("  Use case: Efficient analytics with pandas/polars/duckdb")

        # Show how to read it back (requires pandas or polars)
        try:
            import pandas as pd

            df = pd.read_parquet(parquet_path)
            print("\n  Successfully read back with pandas:")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Rows: {len(df)}")
            print("\n  Sample row:")
            print(f"    ID: {df.iloc[0]['id']}")
            print(f"    Sequence length: {df.iloc[0]['length']}")
            print(f"    GC content: {df.iloc[0]['gc_content']:.2f}%")
        except ImportError:
            print("\n  (Install pandas to read Parquet files: pip install pandas)")

    finally:
        if Path(parquet_path).exists():
            Path(parquet_path).unlink()

    # 2. Export to NumPy (Integer Encoding)
    print("\n\n2. Export to NumPy Format (Integer Encoding)")
    print("-" * 60)

    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp_npy:
        npy_int_path = tmp_npy.name

    try:
        # Export with integer encoding: A=0, C=1, G=2, T=3
        deepbiop.utils.export_to_numpy_int(npy_int_path, sequences)

        file_size = Path(npy_int_path).stat().st_size
        print(f"✓ Exported to NumPy (integer): {npy_int_path}")
        print(f"  File size: {file_size} bytes")
        print("  Encoding: A=0, C=1, G=2, T=3")
        print("  Shape: (n_sequences, max_length)")
        print("  Use case: RNN/LSTM input (compact representation)")

        # Show how to read it back
        try:
            import numpy as np

            data = np.load(npy_int_path)
            print("\n  Successfully read back with NumPy:")
            print(f"  Array shape: {data.shape}")
            print(f"  Data type: {data.dtype}")
            print(f"  First sequence (encoded): {data[0][:12]}")
        except ImportError:
            print("\n  (Install numpy to read .npy files: pip install numpy)")

    finally:
        if Path(npy_int_path).exists():
            Path(npy_int_path).unlink()

    # 3. Export to NumPy (One-Hot Encoding)
    print("\n\n3. Export to NumPy Format (One-Hot Encoding)")
    print("-" * 60)

    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp_npy:
        npy_onehot_path = tmp_npy.name

    try:
        # Export with one-hot encoding
        deepbiop.utils.export_to_numpy_onehot(npy_onehot_path, sequences)

        file_size = Path(npy_onehot_path).stat().st_size
        print(f"✓ Exported to NumPy (one-hot): {npy_onehot_path}")
        print(f"  File size: {file_size} bytes")
        print("  Encoding: Binary matrix [A, C, G, T]")
        print("  Shape: (n_sequences, max_length, 4)")
        print("  Use case: CNN input (spatial features)")

        try:
            import numpy as np

            data = np.load(npy_onehot_path)
            print("\n  Successfully read back with NumPy:")
            print(f"  Array shape: {data.shape}")
            print(f"  Data type: {data.dtype}")
            print(f"  First base (one-hot): {data[0][0]}")
        except ImportError:
            pass

    finally:
        if Path(npy_onehot_path).exists():
            Path(npy_onehot_path).unlink()

    # 4. Export FASTQ to Paired NumPy Files
    print("\n\n4. Export FASTQ to Paired NumPy Files")
    print("-" * 60)

    with tempfile.NamedTemporaryFile(suffix="_seq.npy", delete=False) as tmp_seq:
        seq_path = tmp_seq.name
    with tempfile.NamedTemporaryFile(suffix="_qual.npy", delete=False) as tmp_qual:
        qual_path = tmp_qual.name

    try:
        # Export sequences and qualities together
        deepbiop.utils.export_fastq_to_numpy(seq_path, qual_path, sequences, qualities)

        seq_size = Path(seq_path).stat().st_size
        qual_size = Path(qual_path).stat().st_size
        print("✓ Exported FASTQ to paired NumPy files:")
        print(f"  Sequences: {seq_path} ({seq_size} bytes)")
        print(f"  Qualities: {qual_path} ({qual_size} bytes)")
        print("  Use case: ML models with quality-aware learning")

        try:
            import numpy as np

            seqs = np.load(seq_path)
            quals = np.load(qual_path)
            print("\n  Successfully read back both files:")
            print(f"  Sequences shape: {seqs.shape}")
            print(f"  Qualities shape: {quals.shape}")
            print("  Quality scores: Phred values (0-40+)")
            print(f"  Example quality: {quals[0][:12]}")
        except ImportError:
            pass

    finally:
        if Path(seq_path).exists():
            Path(seq_path).unlink()
        if Path(qual_path).exists():
            Path(qual_path).unlink()

    # 5. Export Quality Scores Only
    print("\n\n5. Export Quality Scores to NumPy")
    print("-" * 60)

    with tempfile.NamedTemporaryFile(suffix="_quality.npy", delete=False) as tmp_qual:
        qual_only_path = tmp_qual.name

    try:
        deepbiop.utils.export_quality_to_numpy(
            qual_only_path,
            qualities,
            33,  # Phred+33 offset
        )

        file_size = Path(qual_only_path).stat().st_size
        print(f"✓ Exported quality scores: {qual_only_path}")
        print(f"  File size: {file_size} bytes")
        print("  Phred offset: 33 (Illumina 1.8+)")
        print("  Use case: Quality-aware base calling, error correction")

    finally:
        if Path(qual_only_path).exists():
            Path(qual_only_path).unlink()

    print("\n" + "=" * 60)
    print("All export formats demonstrated successfully!")
    print("\nNext steps:")
    print("  • Use Parquet for fast analytics (pandas/polars/duckdb)")
    print("  • Use NumPy integer encoding for RNN/LSTM models")
    print("  • Use NumPy one-hot encoding for CNN models")
    print("  • Combine with deepbiop.fq for on-the-fly data loading")
    print("=" * 60)


if __name__ == "__main__":
    main()
