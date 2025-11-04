"""Tests for GTF file parsing and genomic feature analysis."""

import sys
import tempfile
from pathlib import Path

import pytest

import deepbiop as dbp

# Skip all GTF tests on Windows due to SIMD/CPU instruction compatibility issues
# Error: Windows fatal exception: code 0xc000001d (Illegal instruction)
# This is a known issue with noodles GTF parsing on Windows CI runners
pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="GTF tests fail on Windows CI with illegal instruction error",
)


@pytest.fixture
def minimal_gtf_file():
    """Create a minimal GTF file for testing."""
    content = """##description: Test GTF file
##provider: Test
##format: gtf
chr1\ttest\tgene\t1000\t5000\t.\t+\t.\tgene_id "GENE001"; gene_name "TestGene1"; gene_type "protein_coding";
chr1\ttest\ttranscript\t1000\t5000\t.\t+\t.\tgene_id "GENE001"; transcript_id "TRANS001"; gene_name "TestGene1";
chr1\ttest\texon\t1000\t2000\t.\t+\t.\tgene_id "GENE001"; transcript_id "TRANS001"; exon_number "1";
chr1\ttest\tCDS\t1100\t1900\t.\t+\t0\tgene_id "GENE001"; transcript_id "TRANS001"; exon_number "1";
chr1\ttest\texon\t3000\t5000\t.\t+\t.\tgene_id "GENE001"; transcript_id "TRANS001"; exon_number "2";
chr1\ttest\tCDS\t3000\t4900\t.\t+\t1\tgene_id "GENE001"; transcript_id "TRANS001"; exon_number "2";
chr2\ttest\tgene\t10000\t15000\t.\t-\t.\tgene_id "GENE002"; gene_name "TestGene2"; gene_type "lncRNA";
chr2\ttest\ttranscript\t10000\t15000\t.\t-\t.\tgene_id "GENE002"; transcript_id "TRANS002"; gene_name "TestGene2";
chr2\ttest\texon\t10000\t12000\t.\t-\t.\tgene_id "GENE002"; transcript_id "TRANS002"; exon_number "1";
chr2\ttest\texon\t13000\t15000\t.\t-\t.\tgene_id "GENE002"; transcript_id "TRANS002"; exon_number "2";
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".gtf", delete=False) as f:
        f.write(content)
        gtf_path = f.name

    yield gtf_path

    # Cleanup
    Path(gtf_path).unlink()


class TestGtfReader:
    """Test GtfReader for parsing GTF files."""

    def test_gtf_reader_open(self, minimal_gtf_file):
        """Test opening a GTF file."""
        reader = dbp.GtfReader(minimal_gtf_file)
        assert reader is not None

    def test_gtf_reader_invalid_path(self):
        """Test opening non-existent GTF file."""
        with pytest.raises(IOError):
            dbp.GtfReader("nonexistent.gtf")

    def test_read_all_features(self, minimal_gtf_file):
        """Test reading all features from GTF file."""
        reader = dbp.GtfReader(minimal_gtf_file)
        features = reader.read_all()

        assert len(features) == 10
        assert all(isinstance(f, dbp.GenomicFeature) for f in features)

    def test_filter_by_type_gene(self, minimal_gtf_file):
        """Test filtering for gene features."""
        reader = dbp.GtfReader(minimal_gtf_file)
        genes = reader.filter_by_type("gene")

        assert len(genes) == 2
        assert all(f.feature_type == "gene" for f in genes)

    def test_filter_by_type_exon(self, minimal_gtf_file):
        """Test filtering for exon features."""
        reader = dbp.GtfReader(minimal_gtf_file)
        exons = reader.filter_by_type("exon")

        assert len(exons) == 4
        assert all(f.feature_type == "exon" for f in exons)

    def test_filter_by_type_cds(self, minimal_gtf_file):
        """Test filtering for CDS features."""
        reader = dbp.GtfReader(minimal_gtf_file)
        cds_features = reader.filter_by_type("CDS")

        assert len(cds_features) == 2
        assert all(f.feature_type == "CDS" for f in cds_features)

    def test_filter_by_type_transcript(self, minimal_gtf_file):
        """Test filtering for transcript features."""
        reader = dbp.GtfReader(minimal_gtf_file)
        transcripts = reader.filter_by_type("transcript")

        assert len(transcripts) == 2
        assert all(f.feature_type == "transcript" for f in transcripts)

    def test_build_gene_index(self, minimal_gtf_file):
        """Test building gene index."""
        reader = dbp.GtfReader(minimal_gtf_file)
        gene_index = reader.build_gene_index()

        # Should have 2 genes
        assert len(gene_index) == 2
        assert "GENE001" in gene_index
        assert "GENE002" in gene_index

        # GENE001 should have 6 features (gene, transcript, 2 exons, 2 CDS)
        assert len(gene_index["GENE001"]) == 6

        # GENE002 should have 4 features (gene, transcript, 2 exons)
        assert len(gene_index["GENE002"]) == 4


class TestGenomicFeature:
    """Test GenomicFeature class."""

    def test_feature_attributes(self, minimal_gtf_file):
        """Test feature attribute access."""
        reader = dbp.GtfReader(minimal_gtf_file)
        features = reader.read_all()

        # First feature: gene on chr1
        f = features[0]
        assert f.seqname == "chr1"
        assert f.source == "test"
        assert f.feature_type == "gene"
        assert f.start == 1000
        assert f.end == 5000
        assert f.strand == "+"

    def test_feature_gene_id(self, minimal_gtf_file):
        """Test extracting gene ID from attributes."""
        reader = dbp.GtfReader(minimal_gtf_file)
        features = reader.read_all()

        # First feature should have GENE001
        assert features[0].gene_id() == "GENE001"

        # Seventh feature should have GENE002
        assert features[6].gene_id() == "GENE002"

    def test_feature_transcript_id(self, minimal_gtf_file):
        """Test extracting transcript ID from attributes."""
        reader = dbp.GtfReader(minimal_gtf_file)
        features = reader.read_all()

        # Second feature (transcript) should have TRANS001
        assert features[1].transcript_id() == "TRANS001"

        # Gene features don't have transcript ID
        assert features[0].transcript_id() is None

    def test_feature_gene_name(self, minimal_gtf_file):
        """Test extracting gene name from attributes."""
        reader = dbp.GtfReader(minimal_gtf_file)
        features = reader.read_all()

        # First feature has gene_name
        assert features[0].gene_name() == "TestGene1"

        # Seventh feature has different gene_name
        assert features[6].gene_name() == "TestGene2"

    def test_feature_length(self, minimal_gtf_file):
        """Test calculating feature length."""
        reader = dbp.GtfReader(minimal_gtf_file)
        features = reader.read_all()

        # First gene: 1000-5000 = 4001 bp (inclusive)
        assert features[0].length() == 4001

        # First exon: 1000-2000 = 1001 bp
        assert features[2].length() == 1001

        # Second exon: 3000-5000 = 2001 bp
        assert features[4].length() == 2001

    def test_feature_strand_forward(self, minimal_gtf_file):
        """Test forward strand features."""
        reader = dbp.GtfReader(minimal_gtf_file)
        genes = reader.filter_by_type("gene")

        # First gene is on forward strand
        assert genes[0].strand == "+"

    def test_feature_strand_reverse(self, minimal_gtf_file):
        """Test reverse strand features."""
        reader = dbp.GtfReader(minimal_gtf_file)
        genes = reader.filter_by_type("gene")

        # Second gene is on reverse strand
        assert genes[1].strand == "-"

    def test_feature_frame(self, minimal_gtf_file):
        """Test CDS frame attribute."""
        reader = dbp.GtfReader(minimal_gtf_file)
        cds_features = reader.filter_by_type("CDS")

        # First CDS has frame 0
        assert cds_features[0].frame == 0

        # Second CDS has frame 1
        assert cds_features[1].frame == 1

    def test_feature_attributes_dict(self, minimal_gtf_file):
        """Test accessing raw attributes dictionary."""
        reader = dbp.GtfReader(minimal_gtf_file)
        features = reader.read_all()

        # First feature should have attributes
        attrs = features[0].attributes
        assert isinstance(attrs, dict)
        assert "gene_id" in attrs
        assert "gene_name" in attrs
        assert attrs["gene_id"] == "GENE001"

    def test_feature_repr(self, minimal_gtf_file):
        """Test feature string representation."""
        reader = dbp.GtfReader(minimal_gtf_file)
        features = reader.read_all()

        # First feature repr should contain key info
        repr_str = repr(features[0])
        assert "gene" in repr_str
        assert "chr1" in repr_str
        assert "1000" in repr_str
        assert "5000" in repr_str
        assert "+" in repr_str


class TestGtfAnalysis:
    """Test GTF analysis workflows."""

    def test_count_features_by_type(self, minimal_gtf_file):
        """Test counting features by type."""
        reader = dbp.GtfReader(minimal_gtf_file)
        features = reader.read_all()

        from collections import Counter

        type_counts = Counter(f.feature_type for f in features)

        assert type_counts["gene"] == 2
        assert type_counts["transcript"] == 2
        assert type_counts["exon"] == 4
        assert type_counts["CDS"] == 2

    def test_chromosome_distribution(self, minimal_gtf_file):
        """Test analyzing features by chromosome."""
        reader = dbp.GtfReader(minimal_gtf_file)
        features = reader.read_all()

        chr1_features = [f for f in features if f.seqname == "chr1"]
        chr2_features = [f for f in features if f.seqname == "chr2"]

        # chr1 has 6 features, chr2 has 4 features
        assert len(chr1_features) == 6
        assert len(chr2_features) == 4

    def test_gene_structure_analysis(self, minimal_gtf_file):
        """Test analyzing gene structure."""
        reader = dbp.GtfReader(minimal_gtf_file)
        gene_index = reader.build_gene_index()

        # Analyze GENE001
        gene001_features = gene_index["GENE001"]
        exons = [f for f in gene001_features if f.feature_type == "exon"]
        cds = [f for f in gene001_features if f.feature_type == "CDS"]

        assert len(exons) == 2
        assert len(cds) == 2

        # Calculate total exonic length
        total_exon_length = sum(e.length() for e in exons)
        assert total_exon_length == 1001 + 2001  # 3002 bp

    def test_coding_vs_noncoding(self, minimal_gtf_file):
        """Test distinguishing coding from non-coding genes."""
        reader = dbp.GtfReader(minimal_gtf_file)
        genes = reader.filter_by_type("gene")

        # Check gene_type attribute
        gene1_type = genes[0].attributes.get("gene_type")
        gene2_type = genes[1].attributes.get("gene_type")

        assert gene1_type == "protein_coding"
        assert gene2_type == "lncRNA"

    def test_strand_specific_analysis(self, minimal_gtf_file):
        """Test strand-specific feature analysis."""
        reader = dbp.GtfReader(minimal_gtf_file)
        features = reader.read_all()

        forward_features = [f for f in features if f.strand == "+"]
        reverse_features = [f for f in features if f.strand == "-"]

        # chr1 features are forward, chr2 features are reverse
        assert len(forward_features) == 6
        assert len(reverse_features) == 4

    def test_exon_coordinates(self, minimal_gtf_file):
        """Test extracting exon coordinates."""
        reader = dbp.GtfReader(minimal_gtf_file)
        exons = reader.filter_by_type("exon")

        # Sort by start position
        sorted_exons = sorted(exons, key=lambda e: (e.seqname, e.start))

        # First exon on chr1
        assert sorted_exons[0].start == 1000
        assert sorted_exons[0].end == 2000

        # Second exon on chr1
        assert sorted_exons[1].start == 3000
        assert sorted_exons[1].end == 5000


class TestGtfIntegration:
    """Integration tests for GTF workflows."""

    def test_export_to_pandas(self, minimal_gtf_file):
        """Test converting features to pandas DataFrame."""
        pytest.importorskip("pandas")
        import pandas as pd

        reader = dbp.GtfReader(minimal_gtf_file)
        features = reader.read_all()

        # Convert to dict format for DataFrame
        data = {
            "seqname": [f.seqname for f in features],
            "feature_type": [f.feature_type for f in features],
            "start": [f.start for f in features],
            "end": [f.end for f in features],
            "strand": [f.strand for f in features],
            "length": [f.length() for f in features],
            "gene_id": [f.gene_id() for f in features],
            "transcript_id": [f.transcript_id() for f in features],
            "gene_name": [f.gene_name() for f in features],
        }

        df = pd.DataFrame(data)

        assert len(df) == 10
        assert "seqname" in df.columns
        assert "feature_type" in df.columns
        assert df["feature_type"].nunique() == 4  # gene, transcript, exon, CDS

    def test_gene_statistics(self, minimal_gtf_file):
        """Test computing gene-level statistics."""
        reader = dbp.GtfReader(minimal_gtf_file)
        gene_index = reader.build_gene_index()

        stats = {}
        for gene_id, gene_features in gene_index.items():
            exons = [f for f in gene_features if f.feature_type == "exon"]
            cds = [f for f in gene_features if f.feature_type == "CDS"]

            stats[gene_id] = {
                "num_exons": len(exons),
                "num_cds": len(cds),
                "total_exon_length": sum(e.length() for e in exons),
                "total_cds_length": sum(c.length() for c in cds),
            }

        # GENE001 has 2 exons, 2 CDS
        assert stats["GENE001"]["num_exons"] == 2
        assert stats["GENE001"]["num_cds"] == 2
        assert stats["GENE001"]["total_exon_length"] == 3002

        # GENE002 has 2 exons, 0 CDS (non-coding)
        assert stats["GENE002"]["num_exons"] == 2
        assert stats["GENE002"]["num_cds"] == 0

    def test_transcript_to_gene_mapping(self, minimal_gtf_file):
        """Test mapping transcripts to genes."""
        reader = dbp.GtfReader(minimal_gtf_file)
        transcripts = reader.filter_by_type("transcript")

        transcript_to_gene = {f.transcript_id(): f.gene_id() for f in transcripts}

        assert transcript_to_gene["TRANS001"] == "GENE001"
        assert transcript_to_gene["TRANS002"] == "GENE002"

    def test_genomic_range_query(self, minimal_gtf_file):
        """Test querying features in genomic range."""
        reader = dbp.GtfReader(minimal_gtf_file)
        features = reader.read_all()

        # Query chr1:1500-3500
        chr1_range = [
            f
            for f in features
            if f.seqname == "chr1" and not (f.end < 1500 or f.start > 3500)
        ]

        # Should include: gene, transcript, exon1, CDS1, exon2
        assert len(chr1_range) >= 5

    def test_feature_hierarchy(self, minimal_gtf_file):
        """Test gene -> transcript -> exon hierarchy."""
        reader = dbp.GtfReader(minimal_gtf_file)
        gene_index = reader.build_gene_index()

        # For GENE001, check hierarchy
        gene001 = gene_index["GENE001"]

        genes = [f for f in gene001 if f.feature_type == "gene"]
        transcripts = [f for f in gene001 if f.feature_type == "transcript"]
        exons = [f for f in gene001 if f.feature_type == "exon"]

        # Should have 1 gene, 1 transcript, 2 exons
        assert len(genes) == 1
        assert len(transcripts) == 1
        assert len(exons) == 2

        # All should share the same gene_id
        assert all(f.gene_id() == "GENE001" for f in gene001)

    def test_cds_coverage(self, minimal_gtf_file):
        """Test calculating CDS coverage of exons."""
        reader = dbp.GtfReader(minimal_gtf_file)
        gene_index = reader.build_gene_index()

        gene001 = gene_index["GENE001"]
        exons = [f for f in gene001 if f.feature_type == "exon"]
        cds = [f for f in gene001 if f.feature_type == "CDS"]

        total_exon = sum(e.length() for e in exons)
        total_cds = sum(c.length() for c in cds)

        # CDS should be shorter than exons (5' and 3' UTRs)
        assert total_cds < total_exon

        # Calculate CDS coverage
        coverage = total_cds / total_exon
        assert 0 < coverage < 1
