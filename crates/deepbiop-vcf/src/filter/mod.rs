use crate::types::Variant;

/// Filter variants based on various criteria
pub struct VariantFilter {
    min_quality: Option<f32>,
    pass_only: bool,
    variant_types: Vec<String>,
}

impl VariantFilter {
    /// Create a new variant filter with default settings
    pub fn new() -> Self {
        Self {
            min_quality: None,
            pass_only: false,
            variant_types: Vec::new(),
        }
    }

    /// Set minimum quality score threshold
    pub fn with_min_quality(mut self, quality: f32) -> Self {
        self.min_quality = Some(quality);
        self
    }

    /// Filter to only variants that pass all filters
    pub fn pass_only(mut self) -> Self {
        self.pass_only = true;
        self
    }

    /// Filter by variant type (SNP, INDEL, etc.)
    pub fn with_variant_types(mut self, types: Vec<String>) -> Self {
        self.variant_types = types;
        self
    }

    /// Apply filter to a list of variants
    ///
    /// # Example
    ///
    /// ```
    /// use deepbiop_vcf::filter::VariantFilter;
    /// use deepbiop_vcf::types::Variant;
    ///
    /// let variants = vec![
    ///     Variant::new("chr1".to_string(), 100, "A".to_string(), vec!["G".to_string()]),
    /// ];
    ///
    /// let filter = VariantFilter::new()
    ///     .with_min_quality(30.0)
    ///     .pass_only();
    ///
    /// let filtered = filter.apply(&variants);
    /// ```
    pub fn apply(&self, variants: &[Variant]) -> Vec<Variant> {
        variants
            .iter()
            .filter(|v| self.passes(v))
            .cloned()
            .collect()
    }

    /// Check if a single variant passes the filter
    fn passes(&self, variant: &Variant) -> bool {
        // Check quality threshold
        if let Some(min_q) = self.min_quality {
            if variant.quality.is_none_or(|q| q < min_q) {
                return false;
            }
        }

        // Check pass-only filter
        if self.pass_only && !variant.passes_filter() {
            return false;
        }

        // Check variant type filter
        if !self.variant_types.is_empty() {
            let vtype = variant.variant_type();
            if !self.variant_types.iter().any(|t| t == vtype) {
                return false;
            }
        }

        true
    }
}

impl Default for VariantFilter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_variant(quality: Option<f32>, filter: Vec<String>) -> Variant {
        let mut variant = Variant::new(
            "chr1".to_string(),
            100,
            "A".to_string(),
            vec!["G".to_string()],
        );
        variant.quality = quality;
        variant.filter = filter;
        variant
    }

    #[test]
    fn test_quality_filter() {
        let variants = vec![
            create_test_variant(Some(40.0), vec![]),
            create_test_variant(Some(20.0), vec![]),
            create_test_variant(None, vec![]),
        ];

        let filter = VariantFilter::new().with_min_quality(30.0);
        let filtered = filter.apply(&variants);

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].quality, Some(40.0));
    }

    #[test]
    fn test_pass_only_filter() {
        let variants = vec![
            create_test_variant(Some(40.0), vec!["PASS".to_string()]),
            create_test_variant(Some(40.0), vec!["LowQual".to_string()]),
            create_test_variant(Some(40.0), vec![]),
        ];

        let filter = VariantFilter::new().pass_only();
        let filtered = filter.apply(&variants);

        assert_eq!(filtered.len(), 2); // PASS and empty (both pass)
    }

    #[test]
    fn test_variant_type_filter() {
        let snp = Variant::new(
            "chr1".to_string(),
            100,
            "A".to_string(),
            vec!["G".to_string()],
        );
        let indel = Variant::new(
            "chr1".to_string(),
            200,
            "AT".to_string(),
            vec!["A".to_string()],
        );

        let variants = vec![snp, indel];

        let filter = VariantFilter::new().with_variant_types(vec!["SNP".to_string()]);
        let filtered = filter.apply(&variants);

        assert_eq!(filtered.len(), 1);
        assert!(filtered[0].is_snp());
    }
}
