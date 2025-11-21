#!/usr/bin/env python3
"""Example demonstrating format conversion between FASTA and FASTQ.

This script shows how to use DeepBioP's conversion features to convert
between FASTA and FASTQ formats.
"""

import tempfile
from pathlib import Path

import deepbiop


def main():
    """Run conversion examples demonstrating format conversions."""
    print("DeepBioP Format Conversion Examples")
    print("=" * 60)

    # Test data file
    fasta_file = Path(__file__).parent / "test_sample.fa"

    if not fasta_file.exists():
        print(f"Error: Test file not found: {fasta_file}")
        return

    # 1. FASTA to FASTQ conversion
    print("\n1. FASTA to FASTQ Conversion")
    print("-" * 60)

    with tempfile.NamedTemporaryFile(suffix=".fq", delete=False) as tmp_fq:
        tmp_fq_path = tmp_fq.name

    try:
        # Convert FASTA to FASTQ (assigns default Q40 quality scores)
        deepbiop.fa.fasta_to_fastq(fasta_file, tmp_fq_path)
        print("✓ Converted FASTA to FASTQ")
        print(f"  Input:  {fasta_file}")
        print(f"  Output: {tmp_fq_path}")
        print("  Quality: All bases assigned Q40 (99.99% accuracy)")

        # Read and display first record
        with Path(tmp_fq_path).open() as f:
            lines = f.readlines()
            if len(lines) >= 4:
                print("\nFirst FASTQ record:")
                print(f"  ID:      {lines[0].strip()}")
                print(f"  Seq:     {lines[1].strip()[:50]}...")
                print(f"  Quality: {lines[3].strip()[:50]}...")

        # 2. FASTQ to FASTA conversion (round-trip)
        print("\n\n2. FASTQ to FASTA Conversion")
        print("-" * 60)

        with tempfile.NamedTemporaryFile(suffix=".fa", delete=False) as tmp_fa:
            tmp_fa_path = tmp_fa.name

        try:
            # Convert FASTQ back to FASTA (quality scores are discarded)
            deepbiop.fq.fastq_to_fasta(tmp_fq_path, tmp_fa_path)
            print("✓ Converted FASTQ to FASTA")
            print(f"  Input:  {tmp_fq_path}")
            print(f"  Output: {tmp_fa_path}")
            print("  Note: Quality scores discarded")

            # Read and display first record
            with Path(tmp_fa_path).open() as f:
                lines = f.readlines()
                if len(lines) >= 2:
                    print("\nFirst FASTA record:")
                    print(f"  ID:  {lines[0].strip()}")
                    print(f"  Seq: {lines[1].strip()[:50]}...")

            print("\n" + "=" * 60)
            print("Format conversions completed successfully!")
            print("=" * 60)

        finally:
            if Path(tmp_fa_path).exists():
                Path(tmp_fa_path).unlink()

    finally:
        if Path(tmp_fq_path).exists():
            Path(tmp_fq_path).unlink()


if __name__ == "__main__":
    main()
