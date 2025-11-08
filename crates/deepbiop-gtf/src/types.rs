use ahash::HashMap;
use serde::{Deserialize, Serialize};

/// Strand orientation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Strand {
    /// Forward strand (+)
    Forward,
    /// Reverse strand (-)
    Reverse,
    /// Unstranded (.)
    Unstranded,
}

impl From<char> for Strand {
    fn from(c: char) -> Self {
        match c {
            '+' => Strand::Forward,
            '-' => Strand::Reverse,
            _ => Strand::Unstranded,
        }
    }
}

impl std::fmt::Display for Strand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Strand::Forward => write!(f, "+"),
            Strand::Reverse => write!(f, "-"),
            Strand::Unstranded => write!(f, "."),
        }
    }
}

/// Represents annotated genomic elements from GTF/GFF files
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenomicFeature {
    /// Chromosome/contig name
    pub seqname: String,
    /// Annotation source (e.g., "ENSEMBL", "RefSeq")
    pub source: String,
    /// Feature type (gene, exon, CDS, transcript, etc.)
    pub feature_type: String,
    /// Start position (1-based, inclusive)
    pub start: u64,
    /// End position (1-based, inclusive)
    pub end: u64,
    /// Feature score
    pub score: Option<f32>,
    /// Strand (+, -, or unstranded)
    pub strand: Strand,
    /// Reading frame (0, 1, or 2 for CDS)
    pub frame: Option<u8>,
    /// GTF attributes (gene_id, transcript_id, etc.)
    pub attributes: HashMap<String, String>,
}

impl GenomicFeature {
    /// Create a new GenomicFeature
    pub fn new(
        seqname: String,
        source: String,
        feature_type: String,
        start: u64,
        end: u64,
        strand: Strand,
    ) -> Self {
        Self {
            seqname,
            source,
            feature_type,
            start,
            end,
            score: None,
            strand,
            frame: None,
            attributes: HashMap::default(),
        }
    }

    /// Check if two features overlap
    pub fn overlaps(&self, other: &GenomicFeature) -> bool {
        self.seqname == other.seqname && self.start <= other.end && self.end >= other.start
    }

    /// Get feature length in base pairs
    pub fn length(&self) -> u64 {
        self.end - self.start + 1
    }

    /// Get gene_id attribute if present
    pub fn gene_id(&self) -> Option<&String> {
        self.attributes.get("gene_id")
    }

    /// Get transcript_id attribute if present
    pub fn transcript_id(&self) -> Option<&String> {
        self.attributes.get("transcript_id")
    }

    /// Get gene_name attribute if present
    pub fn gene_name(&self) -> Option<&String> {
        self.attributes.get("gene_name")
    }

    /// Check if feature is on forward strand
    pub fn is_forward_strand(&self) -> bool {
        self.strand == Strand::Forward
    }

    /// Check if feature is on reverse strand
    pub fn is_reverse_strand(&self) -> bool {
        self.strand == Strand::Reverse
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_creation() {
        let feature = GenomicFeature::new(
            "chr1".to_string(),
            "ENSEMBL".to_string(),
            "gene".to_string(),
            1000,
            2000,
            Strand::Forward,
        );

        assert_eq!(feature.seqname, "chr1");
        assert_eq!(feature.length(), 1001);
        assert!(feature.is_forward_strand());
    }

    #[test]
    fn test_overlaps() {
        let f1 = GenomicFeature::new(
            "chr1".to_string(),
            "ENSEMBL".to_string(),
            "gene".to_string(),
            1000,
            2000,
            Strand::Forward,
        );

        let f2 = GenomicFeature::new(
            "chr1".to_string(),
            "ENSEMBL".to_string(),
            "exon".to_string(),
            1500,
            1800,
            Strand::Forward,
        );

        let f3 = GenomicFeature::new(
            "chr1".to_string(),
            "ENSEMBL".to_string(),
            "gene".to_string(),
            3000,
            4000,
            Strand::Forward,
        );

        assert!(f1.overlaps(&f2));
        assert!(f2.overlaps(&f1));
        assert!(!f1.overlaps(&f3));
    }

    #[test]
    fn test_strand() {
        assert_eq!(Strand::from('+'), Strand::Forward);
        assert_eq!(Strand::from('-'), Strand::Reverse);
        assert_eq!(Strand::from('.'), Strand::Unstranded);
    }
}
