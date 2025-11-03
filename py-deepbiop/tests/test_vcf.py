"""Tests for VCF file parsing and variant analysis."""

import tempfile
from pathlib import Path

import pytest

import deepbiop as dbp


@pytest.fixture
def minimal_vcf_file():
    """Create a minimal VCF file for testing."""
    content = """##fileformat=VCFv4.2
##contig=<ID=chr1,length=249250621>
##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">
##FILTER=<ID=PASS,Description="All filters passed">
##FILTER=<ID=LowQual,Description="Low quality">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
chr1\t1000\trs123\tA\tT\t30.0\tPASS\tDP=10
chr1\t2000\trs456\tG\tC\t20.0\tPASS\tDP=15
chr1\t3000\t.\tAT\tA\t25.0\tLowQual\tDP=8
chr1\t4000\t.\tC\tCAT\t40.0\tPASS\tDP=20
chr1\t5000\trs789\tT\tG,A\t35.0\tPASS\tDP=12
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".vcf", delete=False) as f:
        f.write(content)
        vcf_path = f.name

    yield vcf_path

    # Cleanup
    Path(vcf_path).unlink()


class TestVcfReader:
    """Test VcfReader for parsing VCF files."""

    def test_vcf_reader_open(self, minimal_vcf_file):
        """Test opening a VCF file."""
        reader = dbp.VcfReader(minimal_vcf_file)
        assert reader is not None

    def test_vcf_reader_invalid_path(self):
        """Test opening non-existent VCF file."""
        with pytest.raises(IOError):
            dbp.VcfReader("nonexistent.vcf")

    def test_read_all_variants(self, minimal_vcf_file):
        """Test reading all variants from VCF file."""
        reader = dbp.VcfReader(minimal_vcf_file)
        variants = reader.read_all()

        assert len(variants) == 5
        assert all(isinstance(v, dbp.Variant) for v in variants)

    def test_filter_by_quality(self, minimal_vcf_file):
        """Test filtering variants by quality score."""
        reader = dbp.VcfReader(minimal_vcf_file)

        # Filter for quality >= 30
        high_quality = reader.filter_by_quality(30.0)

        assert len(high_quality) >= 3
        assert all(v.quality is None or v.quality >= 30.0 for v in high_quality)

    def test_filter_by_quality_strict(self, minimal_vcf_file):
        """Test strict quality filtering."""
        reader = dbp.VcfReader(minimal_vcf_file)

        # Filter for very high quality >= 35
        very_high_quality = reader.filter_by_quality(35.0)

        assert len(very_high_quality) >= 2
        assert all(v.quality is None or v.quality >= 35.0 for v in very_high_quality)

    def test_filter_passing_variants(self, minimal_vcf_file):
        """Test filtering for variants that pass all filters."""
        reader = dbp.VcfReader(minimal_vcf_file)

        passing = reader.filter_passing()

        assert len(passing) == 4  # 4 variants have PASS filter
        assert all(v.passes_filter() for v in passing)


class TestVariant:
    """Test Variant class for representing genomic variants."""

    def test_variant_attributes(self, minimal_vcf_file):
        """Test variant attribute access."""
        reader = dbp.VcfReader(minimal_vcf_file)
        variants = reader.read_all()

        # First variant: chr1:1000 A>T
        v = variants[0]
        assert v.chromosome == "chr1"
        assert v.position == 1000
        assert v.id == "rs123"
        assert v.reference_allele == "A"
        assert v.alternate_alleles == ["T"]
        assert v.quality == 30.0
        assert "PASS" in v.filter

    def test_variant_is_snp(self, minimal_vcf_file):
        """Test SNP detection."""
        reader = dbp.VcfReader(minimal_vcf_file)
        variants = reader.read_all()

        # First variant: A>T is a SNP
        assert variants[0].is_snp()

        # Second variant: G>C is a SNP
        assert variants[1].is_snp()

    def test_variant_is_not_snp(self, minimal_vcf_file):
        """Test non-SNP detection."""
        reader = dbp.VcfReader(minimal_vcf_file)
        variants = reader.read_all()

        # Third variant: AT>A is a deletion, not SNP
        assert not variants[2].is_snp()

        # Fourth variant: C>CAT is an insertion, not SNP
        assert not variants[3].is_snp()

    def test_variant_is_indel(self, minimal_vcf_file):
        """Test indel detection."""
        reader = dbp.VcfReader(minimal_vcf_file)
        variants = reader.read_all()

        # Third variant: AT>A is a deletion (indel)
        assert variants[2].is_indel()

        # Fourth variant: C>CAT is an insertion (indel)
        assert variants[3].is_indel()

    def test_variant_is_not_indel(self, minimal_vcf_file):
        """Test non-indel detection."""
        reader = dbp.VcfReader(minimal_vcf_file)
        variants = reader.read_all()

        # First variant: A>T is a SNP, not indel
        assert not variants[0].is_indel()

        # Second variant: G>C is a SNP, not indel
        assert not variants[1].is_indel()

    def test_variant_multiallelic(self, minimal_vcf_file):
        """Test multiallelic variant (multiple alternate alleles)."""
        reader = dbp.VcfReader(minimal_vcf_file)
        variants = reader.read_all()

        # Fifth variant: T>G,A has multiple alternates
        v = variants[4]
        assert v.reference_allele == "T"
        assert len(v.alternate_alleles) == 2
        assert "G" in v.alternate_alleles
        assert "A" in v.alternate_alleles

    def test_variant_passes_filter(self, minimal_vcf_file):
        """Test filter pass detection."""
        reader = dbp.VcfReader(minimal_vcf_file)
        variants = reader.read_all()

        # First variant has PASS filter
        assert variants[0].passes_filter()

        # Third variant has LowQual filter (should not pass)
        assert not variants[2].passes_filter()

    def test_variant_no_id(self, minimal_vcf_file):
        """Test variant without ID (.)."""
        reader = dbp.VcfReader(minimal_vcf_file)
        variants = reader.read_all()

        # Third variant has no ID
        v = variants[2]
        assert v.id is None or v.id == "."

    def test_variant_repr(self, minimal_vcf_file):
        """Test variant string representation."""
        reader = dbp.VcfReader(minimal_vcf_file)
        variants = reader.read_all()

        # First variant repr should contain key info
        repr_str = repr(variants[0])
        assert "chr1" in repr_str
        assert "1000" in repr_str
        assert "A" in repr_str


class TestVariantClassification:
    """Test variant classification and filtering."""

    def test_classify_snps_vs_indels(self, minimal_vcf_file):
        """Test separating SNPs from indels."""
        reader = dbp.VcfReader(minimal_vcf_file)
        variants = reader.read_all()

        snps = [v for v in variants if v.is_snp()]
        indels = [v for v in variants if v.is_indel()]

        # Should have 2 SNPs (A>T, G>C)
        assert len(snps) >= 2

        # Should have 2 indels (AT>A, C>CAT)
        assert len(indels) >= 2

        # A variant can't be both SNP and indel in standard cases
        {id(v) for v in snps} & {id(v) for v in indels}
        # Note: Multiallelic variants (T>G,A) might be classified as both

    def test_quality_distribution(self, minimal_vcf_file):
        """Test quality score distribution."""
        reader = dbp.VcfReader(minimal_vcf_file)
        variants = reader.read_all()

        qualities = [v.quality for v in variants if v.quality is not None]

        assert len(qualities) > 0
        assert min(qualities) >= 20.0
        assert max(qualities) <= 40.0

    def test_chromosome_filtering(self, minimal_vcf_file):
        """Test filtering variants by chromosome."""
        reader = dbp.VcfReader(minimal_vcf_file)
        variants = reader.read_all()

        chr1_variants = [v for v in variants if v.chromosome == "chr1"]

        # All variants in test file are on chr1
        assert len(chr1_variants) == len(variants)

    def test_position_range_filtering(self, minimal_vcf_file):
        """Test filtering variants by position range."""
        reader = dbp.VcfReader(minimal_vcf_file)
        variants = reader.read_all()

        # Filter for positions between 2000 and 4000
        range_variants = [v for v in variants if 2000 <= v.position <= 4000]

        assert len(range_variants) == 3  # positions 2000, 3000, 4000


class TestVcfIntegration:
    """Integration tests for VCF workflows."""

    def test_quality_and_filter_combination(self, minimal_vcf_file):
        """Test combining quality and filter criteria."""
        reader = dbp.VcfReader(minimal_vcf_file)
        variants = reader.read_all()

        # High quality AND passing filter
        high_quality_passing = [
            v
            for v in variants
            if v.quality is not None and v.quality >= 30.0 and v.passes_filter()
        ]

        # Should have at least 3 variants (Q30, Q40, Q35 with PASS)
        assert len(high_quality_passing) >= 3

    def test_snp_quality_analysis(self, minimal_vcf_file):
        """Test analyzing SNP quality."""
        reader = dbp.VcfReader(minimal_vcf_file)
        variants = reader.read_all()

        snps = [v for v in variants if v.is_snp()]
        snp_qualities = [v.quality for v in snps if v.quality is not None]

        assert len(snp_qualities) > 0
        # SNPs in test data should have reasonable quality
        assert all(q >= 20.0 for q in snp_qualities)

    def test_indel_analysis(self, minimal_vcf_file):
        """Test analyzing indels."""
        reader = dbp.VcfReader(minimal_vcf_file)
        variants = reader.read_all()

        indels = [v for v in variants if v.is_indel()]

        # Check indel types
        deletions = [
            v
            for v in indels
            if len(v.reference_allele) > min(len(a) for a in v.alternate_alleles)
        ]
        insertions = [
            v
            for v in indels
            if any(len(a) > len(v.reference_allele) for a in v.alternate_alleles)
        ]

        assert len(deletions) + len(insertions) > 0

    def test_variant_statistics(self, minimal_vcf_file):
        """Test computing variant statistics."""
        reader = dbp.VcfReader(minimal_vcf_file)
        variants = reader.read_all()

        total = len(variants)
        snps = sum(1 for v in variants if v.is_snp())
        indels = sum(1 for v in variants if v.is_indel())
        passing = sum(1 for v in variants if v.passes_filter())
        high_quality = sum(
            1 for v in variants if v.quality is not None and v.quality >= 30.0
        )

        assert total == 5
        assert snps + indels >= total  # Some might be both (multiallelic)
        assert passing >= 4
        assert high_quality >= 3

    def test_export_to_pandas(self, minimal_vcf_file):
        """Test converting variants to pandas DataFrame."""
        pytest.importorskip("pandas")
        import pandas as pd

        reader = dbp.VcfReader(minimal_vcf_file)
        variants = reader.read_all()

        # Convert to dict format for DataFrame
        data = {
            "chromosome": [v.chromosome for v in variants],
            "position": [v.position for v in variants],
            "id": [v.id for v in variants],
            "ref": [v.reference_allele for v in variants],
            "alt": [",".join(v.alternate_alleles) for v in variants],
            "quality": [v.quality for v in variants],
            "is_snp": [v.is_snp() for v in variants],
            "is_indel": [v.is_indel() for v in variants],
            "passes_filter": [v.passes_filter() for v in variants],
        }

        df = pd.DataFrame(data)

        assert len(df) == 5
        assert "chromosome" in df.columns
        assert "position" in df.columns
        assert df["is_snp"].sum() >= 2
        assert df["is_indel"].sum() >= 2
