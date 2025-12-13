use crate::types::Variant;
use ahash::HashMap;

/// Extract and manipulate INFO field annotations from variants
pub struct InfoExtractor;

impl InfoExtractor {
    /// Extract specific INFO fields from variants
    ///
    /// # Arguments
    ///
    /// * `variants` - List of variants to extract from
    /// * `fields` - INFO field names to extract
    ///
    /// # Returns
    ///
    /// A map of variant positions to their INFO field values
    ///
    /// # Example
    ///
    /// ```
    /// use deepbiop_vcf::annotate::InfoExtractor;
    /// use deepbiop_vcf::types::Variant;
    ///
    /// let variant = Variant::new("chr1".to_string(), 100, "A".to_string(), vec!["G".to_string()]);
    /// let variants = vec![variant];
    /// let fields = vec!["AF".to_string(), "DP".to_string()];
    ///
    /// let extracted = InfoExtractor::extract_fields(&variants, &fields);
    /// ```
    pub fn extract_fields(
        variants: &[Variant],
        fields: &[String],
    ) -> HashMap<u64, HashMap<String, String>> {
        let mut result = HashMap::default();

        for variant in variants {
            let mut info_map = HashMap::default();

            for field in fields {
                if let Some(value) = variant.info.get(field) {
                    info_map.insert(field.clone(), value.clone());
                }
            }

            if !info_map.is_empty() {
                result.insert(variant.position, info_map);
            }
        }

        result
    }

    /// Get all unique INFO field names from a set of variants
    ///
    /// # Example
    ///
    /// ```
    /// use deepbiop_vcf::annotate::InfoExtractor;
    /// use deepbiop_vcf::types::Variant;
    ///
    /// let variant = Variant::new("chr1".to_string(), 100, "A".to_string(), vec!["G".to_string()]);
    /// let variants = vec![variant];
    ///
    /// let fields = InfoExtractor::list_info_fields(&variants);
    /// ```
    pub fn list_info_fields(variants: &[Variant]) -> Vec<String> {
        let mut fields = ahash::HashSet::default();

        for variant in variants {
            for key in variant.info.keys() {
                fields.insert(key.clone());
            }
        }

        let mut sorted: Vec<String> = fields.into_iter().collect();
        sorted.sort();
        sorted
    }

    /// Extract a single INFO field as a vector of values
    ///
    /// # Example
    ///
    /// ```
    /// use deepbiop_vcf::annotate::InfoExtractor;
    /// use deepbiop_vcf::types::Variant;
    ///
    /// let variant = Variant::new("chr1".to_string(), 100, "A".to_string(), vec!["G".to_string()]);
    /// let variants = vec![variant];
    ///
    /// let depths = InfoExtractor::extract_field(&variants, "DP");
    /// ```
    pub fn extract_field(variants: &[Variant], field: &str) -> Vec<Option<String>> {
        variants
            .iter()
            .map(|v| v.info.get(field).cloned())
            .collect()
    }

    /// Count variants by INFO field presence
    ///
    /// # Example
    ///
    /// ```
    /// use deepbiop_vcf::annotate::InfoExtractor;
    /// use deepbiop_vcf::types::Variant;
    ///
    /// let variant = Variant::new("chr1".to_string(), 100, "A".to_string(), vec!["G".to_string()]);
    /// let variants = vec![variant];
    ///
    /// let counts = InfoExtractor::count_by_field(&variants, "DP");
    /// println!("Variants with DP: {}, without: {}", counts.0, counts.1);
    /// ```
    pub fn count_by_field(variants: &[Variant], field: &str) -> (usize, usize) {
        let with_field = variants
            .iter()
            .filter(|v| v.info.contains_key(field))
            .count();
        let without_field = variants.len() - with_field;
        (with_field, without_field)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_variant_with_info(position: u64, info: Vec<(&str, &str)>) -> Variant {
        let mut variant = Variant::new(
            "chr1".to_string(),
            position,
            "A".to_string(),
            vec!["G".to_string()],
        );
        for (key, value) in info {
            variant.info.insert(key.to_string(), value.to_string());
        }
        variant
    }

    #[test]
    fn test_extract_fields() {
        let variants = vec![
            create_test_variant_with_info(100, vec![("AF", "0.5"), ("DP", "30")]),
            create_test_variant_with_info(200, vec![("AF", "0.3")]),
        ];

        let fields = vec!["AF".to_string(), "DP".to_string()];
        let extracted = InfoExtractor::extract_fields(&variants, &fields);

        assert_eq!(extracted.len(), 2);
        assert!(extracted.get(&100).unwrap().contains_key("AF"));
        assert!(extracted.get(&100).unwrap().contains_key("DP"));
        assert!(extracted.get(&200).unwrap().contains_key("AF"));
        assert!(!extracted.get(&200).unwrap().contains_key("DP"));
    }

    #[test]
    fn test_list_info_fields() {
        let variants = vec![
            create_test_variant_with_info(100, vec![("AF", "0.5"), ("DP", "30")]),
            create_test_variant_with_info(200, vec![("AF", "0.3"), ("AC", "2")]),
        ];

        let fields = InfoExtractor::list_info_fields(&variants);
        assert_eq!(fields.len(), 3);
        assert!(fields.contains(&"AF".to_string()));
        assert!(fields.contains(&"DP".to_string()));
        assert!(fields.contains(&"AC".to_string()));
    }

    #[test]
    fn test_count_by_field() {
        let variants = vec![
            create_test_variant_with_info(100, vec![("DP", "30")]),
            create_test_variant_with_info(200, vec![("AF", "0.3")]),
            create_test_variant_with_info(300, vec![("DP", "40")]),
        ];

        let (with_dp, without_dp) = InfoExtractor::count_by_field(&variants, "DP");
        assert_eq!(with_dp, 2);
        assert_eq!(without_dp, 1);
    }
}
