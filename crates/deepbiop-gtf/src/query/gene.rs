use crate::types::GenomicFeature;
use ahash::HashMap;

/// Query features by gene ID
pub struct GeneQuery {
    index: HashMap<String, Vec<GenomicFeature>>,
}

impl GeneQuery {
    /// Create a new GeneQuery from an index
    pub fn new(index: HashMap<String, Vec<GenomicFeature>>) -> Self {
        Self { index }
    }

    /// Get all features for a specific gene
    ///
    /// # Example
    ///
    /// ```
    /// use deepbiop_gtf::query::gene::GeneQuery;
    /// use deepbiop_gtf::types::GenomicFeature;
    /// use ahash::HashMap;
    ///
    /// let index = HashMap::default();
    /// let query = GeneQuery::new(index);
    /// let features = query.get_gene("ENSG00000000001");
    /// ```
    pub fn get_gene(&self, gene_id: &str) -> Option<&Vec<GenomicFeature>> {
        self.index.get(gene_id)
    }

    /// Get all gene IDs in the index
    pub fn list_genes(&self) -> Vec<String> {
        let mut genes: Vec<String> = self.index.keys().cloned().collect();
        genes.sort();
        genes
    }

    /// Count total number of indexed genes
    pub fn gene_count(&self) -> usize {
        self.index.len()
    }

    /// Get features by gene name (searches attributes)
    ///
    /// # Example
    ///
    /// ```
    /// # use deepbiop_gtf::query::gene::GeneQuery;
    /// # use ahash::HashMap;
    /// let index = HashMap::default();
    /// let query = GeneQuery::new(index);
    /// let features = query.get_by_name("TP53");
    /// ```
    pub fn get_by_name(&self, gene_name: &str) -> Vec<&GenomicFeature> {
        self.index
            .values()
            .flatten()
            .filter(|f| f.gene_name().is_some_and(|name| name == gene_name))
            .collect()
    }

    /// Get all exons for a gene
    pub fn get_exons(&self, gene_id: &str) -> Vec<&GenomicFeature> {
        self.get_gene(gene_id)
            .map(|features| {
                features
                    .iter()
                    .filter(|f| f.feature_type == "exon")
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get all transcripts for a gene
    pub fn get_transcripts(&self, gene_id: &str) -> Vec<&GenomicFeature> {
        self.get_gene(gene_id)
            .map(|features| {
                features
                    .iter()
                    .filter(|f| f.feature_type == "transcript")
                    .collect()
            })
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Strand;

    fn create_test_feature(gene_id: &str, feature_type: &str) -> GenomicFeature {
        let mut feature = GenomicFeature::new(
            "chr1".to_string(),
            "TEST".to_string(),
            feature_type.to_string(),
            1000,
            2000,
            Strand::Forward,
        );
        feature
            .attributes
            .insert("gene_id".to_string(), gene_id.to_string());
        feature
    }

    #[test]
    fn test_gene_query() {
        let mut index = HashMap::default();
        index.insert(
            "GENE1".to_string(),
            vec![
                create_test_feature("GENE1", "gene"),
                create_test_feature("GENE1", "exon"),
            ],
        );

        let query = GeneQuery::new(index);

        assert_eq!(query.gene_count(), 1);
        assert!(query.get_gene("GENE1").is_some());
        assert_eq!(query.get_gene("GENE1").unwrap().len(), 2);
    }

    #[test]
    fn test_get_exons() {
        let mut index = HashMap::default();
        index.insert(
            "GENE1".to_string(),
            vec![
                create_test_feature("GENE1", "gene"),
                create_test_feature("GENE1", "exon"),
                create_test_feature("GENE1", "exon"),
            ],
        );

        let query = GeneQuery::new(index);
        let exons = query.get_exons("GENE1");

        assert_eq!(exons.len(), 2);
    }
}
