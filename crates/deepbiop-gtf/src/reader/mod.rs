use crate::types::{GenomicFeature, Strand};
use anyhow::{Context, Result};
use noodles::gff;
use noodles::gff::feature::record::Attributes;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// GTF file reader for streaming genomic features
pub struct GtfReader {
    reader: gff::io::Reader<BufReader<File>>,
}

impl GtfReader {
    /// Open a GTF file for reading
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the GTF file (can be gzipped)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use deepbiop_gtf::reader::GtfReader;
    /// use std::path::Path;
    ///
    /// let reader = GtfReader::open(Path::new("annotations.gtf")).unwrap();
    /// ```
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("Failed to open GTF file: {:?}", path.as_ref()))?;

        let buf_reader = BufReader::new(file);
        let reader = gff::io::Reader::new(buf_reader);

        Ok(Self { reader })
    }

    /// Read all features from the file
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use deepbiop_gtf::reader::GtfReader;
    /// # use std::path::Path;
    /// let mut reader = GtfReader::open(Path::new("annotations.gtf")).unwrap();
    /// let features = reader.read_all().unwrap();
    /// println!("Found {} features", features.len());
    /// ```
    pub fn read_all(&mut self) -> Result<Vec<GenomicFeature>> {
        let mut features = Vec::new();

        // Use lazy records to avoid parsing issues with GTF attributes
        use std::io::BufRead;
        let lines: Vec<String> = self
            .reader
            .get_mut()
            .lines()
            .collect::<std::io::Result<Vec<_>>>()
            .context("Failed to read GTF lines")?;

        for line in lines {
            // Skip comments and empty lines
            if line.trim().is_empty() || line.starts_with('#') {
                continue;
            }

            let feature = Self::parse_gtf_line(&line)?;
            features.push(feature);
        }

        Ok(features)
    }

    /// Parse a single GTF line into a GenomicFeature
    fn parse_gtf_line(line: &str) -> Result<GenomicFeature> {
        let parts: Vec<&str> = line.split('\t').collect();

        if parts.len() != 9 {
            anyhow::bail!(
                "Invalid GTF line: expected 9 tab-separated fields, got {}",
                parts.len()
            );
        }

        let seqname = parts[0].to_string();
        let source = parts[1].to_string();
        let feature_type = parts[2].to_string();

        let start: u64 = parts[3]
            .parse()
            .with_context(|| format!("Invalid start position: {}", parts[3]))?;
        let end: u64 = parts[4]
            .parse()
            .with_context(|| format!("Invalid end position: {}", parts[4]))?;

        let score = if parts[5] == "." {
            None
        } else {
            Some(
                parts[5]
                    .parse::<f32>()
                    .with_context(|| format!("Invalid score: {}", parts[5]))?,
            )
        };

        let strand_char = parts[6].chars().next().unwrap_or('.');
        let strand = Strand::from(strand_char);

        let frame = if parts[7] == "." {
            None
        } else {
            Some(
                parts[7]
                    .parse::<u8>()
                    .with_context(|| format!("Invalid frame: {}", parts[7]))?,
            )
        };

        // Parse GTF attributes (format: key "value"; key "value";)
        let attributes = Self::parse_gtf_attributes(parts[8])?;

        Ok(GenomicFeature {
            seqname,
            source,
            feature_type,
            start,
            end,
            score,
            strand,
            frame,
            attributes,
        })
    }

    /// Parse GTF-style attributes
    fn parse_gtf_attributes(attr_str: &str) -> Result<ahash::HashMap<String, String>> {
        let mut attributes = ahash::HashMap::default();

        // Split by semicolon, trim whitespace
        for pair in attr_str.split(';') {
            let pair = pair.trim();
            if pair.is_empty() {
                continue;
            }

            // GTF format: key "value" or key value
            let parts: Vec<&str> = pair.splitn(2, ' ').collect();
            if parts.len() == 2 {
                let key = parts[0].trim().to_string();
                let value = parts[1].trim().trim_matches('"').to_string();
                attributes.insert(key, value);
            }
        }

        Ok(attributes)
    }

    /// Convert a noodles GFF record to our GenomicFeature type
    fn record_to_feature(record: &gff::feature::RecordBuf) -> Result<GenomicFeature> {
        let seqname = record.reference_sequence_name().to_string();
        let source = record.source().to_string();
        let feature_type = record.ty().to_string();

        // Get start and end positions - RecordBuf returns Position directly
        let start: u64 = usize::from(record.start())
            .try_into()
            .context("Start position overflow")?;

        let end: u64 = usize::from(record.end())
            .try_into()
            .context("End position overflow")?;

        // Get score - returns Option<f32> directly for RecordBuf
        let score = record.score();

        // Parse strand - RecordBuf returns Strand directly
        let strand_char = match record.strand() {
            gff::feature::record::Strand::Forward => '+',
            gff::feature::record::Strand::Reverse => '-',
            gff::feature::record::Strand::None | gff::feature::record::Strand::Unknown => '.',
        };
        let strand = Strand::from(strand_char);

        // Parse frame/phase - RecordBuf returns Option<Phase>
        let frame = match record.phase() {
            Some(gff::feature::record::Phase::Zero) => Some(0),
            Some(gff::feature::record::Phase::One) => Some(1),
            Some(gff::feature::record::Phase::Two) => Some(2),
            None => None,
        };

        // Parse attributes - GTF attributes are stored in an IndexMap
        let mut attributes = ahash::HashMap::default();

        // Iterate through attributes and convert to our format
        for (key, _value) in record.attributes().iter().flatten() {
            // Convert BString key to String
            let key_str = String::from_utf8_lossy(key.as_ref());
            // For now, store a placeholder for the value
            // TODO: Properly extract value based on its enum variant
            attributes.insert(key_str.to_string(), "value".to_string());
        }

        Ok(GenomicFeature {
            seqname,
            source,
            feature_type,
            start,
            end,
            score,
            strand,
            frame,
            attributes,
        })
    }

    /// Build an index of features by gene ID for fast lookups
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use deepbiop_gtf::reader::GtfReader;
    /// # use std::path::Path;
    /// let mut reader = GtfReader::open(Path::new("annotations.gtf")).unwrap();
    /// let index = reader.build_gene_index().unwrap();
    /// ```
    pub fn build_gene_index(&mut self) -> Result<ahash::HashMap<String, Vec<GenomicFeature>>> {
        let features = self.read_all()?;
        let mut index = ahash::HashMap::default();

        for feature in features {
            if let Some(gene_id) = feature.gene_id() {
                index
                    .entry(gene_id.clone())
                    .or_insert_with(Vec::new)
                    .push(feature);
            }
        }

        Ok(index)
    }

    /// Filter features by type (e.g., "gene", "exon", "CDS")
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use deepbiop_gtf::reader::GtfReader;
    /// # use std::path::Path;
    /// let mut reader = GtfReader::open(Path::new("annotations.gtf")).unwrap();
    /// let genes = reader.filter_by_type("gene").unwrap();
    /// ```
    pub fn filter_by_type(&mut self, feature_type: &str) -> Result<Vec<GenomicFeature>> {
        let all_features = self.read_all()?;
        Ok(all_features
            .into_iter()
            .filter(|f| f.feature_type == feature_type)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

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
        assert_eq!(feature.feature_type, "gene");
    }

    #[test]
    fn test_read_gtf_file() {
        // Create a temporary GTF file
        let content = "chr1\ttest\tgene\t1000\t5000\t.\t+\t.\tgene_id \"GENE001\"; gene_name \"TestGene1\";\n";

        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        temp_file.write_all(content.as_bytes()).unwrap();
        temp_file.flush().unwrap(); // Ensure data is written
        let path = temp_file.path();

        println!("Created temp file at: {:?}", path);
        println!("File content:\n{}", std::fs::read_to_string(path).unwrap());

        // Try to read it with our new parse method
        let mut reader = GtfReader::open(path).unwrap();
        let features_result = reader.read_all();

        match features_result {
            Ok(features) => {
                println!("Successfully read {} features", features.len());
                assert_eq!(features.len(), 1);

                let feature = &features[0];
                println!("Feature: {:?}", feature);
                assert_eq!(feature.seqname, "chr1");
                assert_eq!(feature.feature_type, "gene");
                assert_eq!(feature.start, 1000);
                assert_eq!(feature.end, 5000);
                assert_eq!(feature.gene_id(), Some(&"GENE001".to_string()));
                assert_eq!(feature.gene_name(), Some(&"TestGene1".to_string()));
            }
            Err(e) => {
                eprintln!("Error reading GTF: {}", e);
                panic!("Failed to read GTF: {}", e);
            }
        }
    }
}
