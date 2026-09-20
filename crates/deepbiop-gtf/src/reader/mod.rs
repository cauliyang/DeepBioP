use crate::types::{GenomicFeature, Strand};
use anyhow::{Context, Result};
use deepbiop_utils::io::create_reader_for_compressed_file;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

/// GTF file reader.
///
/// The file is parsed lazily on the first query and the features are kept, so
/// every query method can be called any number of times on one reader.
pub struct GtfReader {
    reader: Option<BufReader<Box<dyn Read + Send + Sync>>>,
    features: Vec<GenomicFeature>,
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
        let reader = create_reader_for_compressed_file(path.as_ref())
            .with_context(|| format!("Failed to open GTF file: {:?}", path.as_ref()))?;

        Ok(Self {
            reader: Some(BufReader::new(reader)),
            features: Vec::new(),
        })
    }

    /// Parse the file on first use; subsequent calls reuse the parsed features.
    fn features(&mut self) -> Result<&[GenomicFeature]> {
        if let Some(reader) = self.reader.take() {
            let mut features = Vec::new();
            for line in reader.lines() {
                let line = line.context("Failed to read GTF line")?;
                if line.trim().is_empty() || line.starts_with('#') {
                    continue;
                }
                features.push(Self::parse_gtf_line(&line)?);
            }
            self.features = features;
        }
        Ok(&self.features)
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
        Ok(self.features()?.to_vec())
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

    /// Parse GTF-style attributes (`key "value"; key "value";`).
    ///
    /// Semicolons inside quoted values do not split fields. Keys that occur
    /// more than once (e.g. GENCODE `tag`) keep every value, comma-joined in
    /// file order.
    fn parse_gtf_attributes(attr_str: &str) -> Result<ahash::HashMap<String, String>> {
        let mut attributes: ahash::HashMap<String, String> = ahash::HashMap::default();

        for pair in split_unquoted(attr_str, ';') {
            let pair = pair.trim();
            if pair.is_empty() {
                continue;
            }

            // GTF format: key "value" or key value
            let Some((key, value)) = pair.split_once(' ') else {
                continue;
            };
            let key = key.trim();
            let value = value.trim().trim_matches('"');
            match attributes.get_mut(key) {
                Some(existing) => {
                    existing.push(',');
                    existing.push_str(value);
                }
                None => {
                    attributes.insert(key.to_owned(), value.to_owned());
                }
            }
        }

        Ok(attributes)
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
        let mut index: ahash::HashMap<String, Vec<GenomicFeature>> = ahash::HashMap::default();

        for feature in self.features()? {
            if let Some(gene_id) = feature.gene_id() {
                index
                    .entry(gene_id.clone())
                    .or_default()
                    .push(feature.clone());
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
        Ok(self
            .features()?
            .iter()
            .filter(|f| f.feature_type == feature_type)
            .cloned()
            .collect())
    }
}

/// Split `s` on `sep` occurrences that are outside double quotes.
fn split_unquoted(s: &str, sep: char) -> impl Iterator<Item = &str> {
    let mut in_quotes = false;
    s.split(move |c: char| {
        if c == '"' {
            in_quotes = !in_quotes;
        }
        c == sep && !in_quotes
    })
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
        let content = "chr1\ttest\tgene\t1000\t5000\t.\t+\t.\tgene_id \"GENE001\"; gene_name \"TestGene1\";\n\
                       chr1\ttest\texon\t1000\t1200\t.\t+\t.\tgene_id \"GENE001\"; tag \"basic\"; tag \"CCDS\"; note \"a; b\";\n";

        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        temp_file.write_all(content.as_bytes()).unwrap();
        temp_file.flush().unwrap();

        let mut reader = GtfReader::open(temp_file.path()).unwrap();
        let features = reader.read_all().unwrap();
        assert_eq!(features.len(), 2);

        let gene = &features[0];
        assert_eq!(gene.seqname, "chr1");
        assert_eq!(gene.feature_type, "gene");
        assert_eq!((gene.start, gene.end), (1000, 5000));
        assert_eq!(gene.gene_id(), Some(&"GENE001".to_string()));
        assert_eq!(gene.gene_name(), Some(&"TestGene1".to_string()));

        // Repeated keys are kept, quoted semicolons do not split fields.
        let exon = &features[1];
        assert_eq!(
            exon.attributes.get("tag").map(String::as_str),
            Some("basic,CCDS")
        );
        assert_eq!(
            exon.attributes.get("note").map(String::as_str),
            Some("a; b")
        );

        // The reader is reusable: querying again after read_all still sees the data.
        assert_eq!(reader.filter_by_type("exon").unwrap().len(), 1);
        assert_eq!(reader.build_gene_index().unwrap()["GENE001"].len(), 2);
        assert_eq!(reader.read_all().unwrap().len(), 2);
    }
}
