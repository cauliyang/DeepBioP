use ahash::HashMap;
use serde::{Deserialize, Serialize};

/// Represents a genomic variant from VCF file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Variant {
    /// Chromosome name
    pub chromosome: String,
    /// Genomic position (1-based)
    pub position: u64,
    /// Variant ID (e.g., dbSNP ID)
    pub id: Option<String>,
    /// Reference base(s)
    pub reference_allele: String,
    /// Alternate allele(s)
    pub alternate_alleles: Vec<String>,
    /// Variant quality score
    pub quality: Option<f32>,
    /// Filter status (PASS, or filter IDs)
    pub filter: Vec<String>,
    /// INFO field annotations
    pub info: HashMap<String, String>,
}

impl Variant {
    /// Create a new Variant
    pub fn new(
        chromosome: String,
        position: u64,
        reference_allele: String,
        alternate_alleles: Vec<String>,
    ) -> Self {
        Self {
            chromosome,
            position,
            id: None,
            reference_allele,
            alternate_alleles,
            quality: None,
            filter: Vec::new(),
            info: HashMap::default(),
        }
    }

    /// Check if variant is a SNP (single nucleotide polymorphism)
    pub fn is_snp(&self) -> bool {
        self.reference_allele.len() == 1
            && self
                .alternate_alleles
                .iter()
                .all(|alt| alt.len() == 1 && alt != "*")
    }

    /// Check if variant is an indel (insertion or deletion)
    pub fn is_indel(&self) -> bool {
        let ref_len = self.reference_allele.len();
        self.alternate_alleles
            .iter()
            .any(|alt| alt.len() != ref_len || alt == "*")
    }

    /// Check if variant passed all filters
    pub fn passes_filter(&self) -> bool {
        self.filter.is_empty() || self.filter.contains(&"PASS".to_string())
    }

    /// Get variant type as string
    pub fn variant_type(&self) -> &str {
        if self.is_snp() {
            "SNP"
        } else if self.is_indel() {
            "INDEL"
        } else {
            "OTHER"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variant_snp() {
        let variant = Variant::new(
            "chr1".to_string(),
            100,
            "A".to_string(),
            vec!["G".to_string()],
        );
        assert!(variant.is_snp());
        assert!(!variant.is_indel());
        assert_eq!(variant.variant_type(), "SNP");
    }

    #[test]
    fn test_variant_indel() {
        let variant = Variant::new(
            "chr1".to_string(),
            100,
            "AT".to_string(),
            vec!["A".to_string()],
        );
        assert!(!variant.is_snp());
        assert!(variant.is_indel());
        assert_eq!(variant.variant_type(), "INDEL");
    }

    #[test]
    fn test_passes_filter() {
        let mut variant = Variant::new(
            "chr1".to_string(),
            100,
            "A".to_string(),
            vec!["G".to_string()],
        );

        // No filters = pass
        assert!(variant.passes_filter());

        // PASS filter
        variant.filter.push("PASS".to_string());
        assert!(variant.passes_filter());

        // Failed filter
        variant.filter = vec!["LowQual".to_string()];
        assert!(!variant.passes_filter());
    }
}
