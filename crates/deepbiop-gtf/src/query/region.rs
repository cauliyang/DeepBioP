use crate::types::GenomicFeature;

/// Query features by genomic region
pub struct RegionQuery;

impl RegionQuery {
    /// Find all features that overlap a genomic region
    ///
    /// # Arguments
    ///
    /// * `features` - List of genomic features to search
    /// * `chromosome` - Chromosome name
    /// * `start` - Region start position (1-based, inclusive)
    /// * `end` - Region end position (1-based, inclusive)
    ///
    /// # Example
    ///
    /// ```
    /// use deepbiop_gtf::query::region::RegionQuery;
    /// use deepbiop_gtf::types::GenomicFeature;
    ///
    /// let features = vec![];
    /// let overlapping = RegionQuery::query(&features, "chr1", 1000, 2000);
    /// ```
    pub fn query<'a>(
        features: &'a [GenomicFeature],
        chromosome: &str,
        start: u64,
        end: u64,
    ) -> Vec<&'a GenomicFeature> {
        features
            .iter()
            .filter(|f| f.seqname == chromosome && f.start <= end && f.end >= start)
            .collect()
    }

    /// Find features that are completely contained within a region
    ///
    /// # Example
    ///
    /// ```
    /// use deepbiop_gtf::query::region::RegionQuery;
    /// use deepbiop_gtf::types::GenomicFeature;
    ///
    /// let features = vec![];
    /// let contained = RegionQuery::query_contained(&features, "chr1", 1000, 2000);
    /// ```
    pub fn query_contained<'a>(
        features: &'a [GenomicFeature],
        chromosome: &str,
        start: u64,
        end: u64,
    ) -> Vec<&'a GenomicFeature> {
        features
            .iter()
            .filter(|f| f.seqname == chromosome && f.start >= start && f.end <= end)
            .collect()
    }

    /// Find features by type within a region
    ///
    /// # Example
    ///
    /// ```
    /// use deepbiop_gtf::query::region::RegionQuery;
    /// use deepbiop_gtf::types::GenomicFeature;
    ///
    /// let features = vec![];
    /// let genes = RegionQuery::query_by_type(&features, "chr1", 1000, 2000, "gene");
    /// ```
    pub fn query_by_type<'a>(
        features: &'a [GenomicFeature],
        chromosome: &str,
        start: u64,
        end: u64,
        feature_type: &str,
    ) -> Vec<&'a GenomicFeature> {
        features
            .iter()
            .filter(|f| {
                f.seqname == chromosome
                    && f.start <= end
                    && f.end >= start
                    && f.feature_type == feature_type
            })
            .collect()
    }

    /// Count features in a region
    pub fn count_in_region(
        features: &[GenomicFeature],
        chromosome: &str,
        start: u64,
        end: u64,
    ) -> usize {
        Self::query(features, chromosome, start, end).len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Strand;

    fn create_test_feature(chr: &str, start: u64, end: u64, ftype: &str) -> GenomicFeature {
        GenomicFeature::new(
            chr.to_string(),
            "TEST".to_string(),
            ftype.to_string(),
            start,
            end,
            Strand::Forward,
        )
    }

    #[test]
    fn test_region_query() {
        let features = vec![
            create_test_feature("chr1", 1000, 2000, "gene"),
            create_test_feature("chr1", 1500, 1800, "exon"),
            create_test_feature("chr1", 3000, 4000, "gene"),
            create_test_feature("chr2", 1000, 2000, "gene"),
        ];

        let overlapping = RegionQuery::query(&features, "chr1", 1200, 1600);
        assert_eq!(overlapping.len(), 2); // gene and exon overlap

        let chr2 = RegionQuery::query(&features, "chr2", 1000, 2000);
        assert_eq!(chr2.len(), 1);
    }

    #[test]
    fn test_contained_query() {
        let features = vec![
            create_test_feature("chr1", 1000, 2000, "gene"),
            create_test_feature("chr1", 1200, 1800, "exon"),
            create_test_feature("chr1", 3000, 4000, "gene"),
        ];

        let contained = RegionQuery::query_contained(&features, "chr1", 1000, 2000);
        assert_eq!(contained.len(), 2); // gene and exon are within bounds
    }

    #[test]
    fn test_query_by_type() {
        let features = vec![
            create_test_feature("chr1", 1000, 2000, "gene"),
            create_test_feature("chr1", 1200, 1800, "exon"),
            create_test_feature("chr1", 1500, 2500, "gene"),
        ];

        let genes = RegionQuery::query_by_type(&features, "chr1", 1000, 2000, "gene");
        assert_eq!(genes.len(), 2);

        let exons = RegionQuery::query_by_type(&features, "chr1", 1000, 2000, "exon");
        assert_eq!(exons.len(), 1);
    }
}
