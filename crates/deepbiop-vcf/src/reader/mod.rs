use crate::types::Variant;
use anyhow::{Context, Result};
use noodles::vcf;
use noodles::vcf::variant::record::{Filters, Ids};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// VCF file reader for streaming variant records
pub struct VcfReader {
    reader: vcf::io::Reader<BufReader<File>>,
    header: vcf::Header,
}

impl VcfReader {
    /// Open a VCF file for reading
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the VCF file (can be gzipped)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use deepbiop_vcf::reader::VcfReader;
    /// use std::path::Path;
    ///
    /// let reader = VcfReader::open(Path::new("variants.vcf")).unwrap();
    /// ```
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("Failed to open VCF file: {:?}", path.as_ref()))?;

        let buf_reader = BufReader::new(file);
        let mut reader = vcf::io::Reader::new(buf_reader);

        let header = reader.read_header().context("Failed to read VCF header")?;

        Ok(Self { reader, header })
    }

    /// Get the VCF header
    pub fn header(&self) -> &vcf::Header {
        &self.header
    }

    /// Read all variants from the file
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use deepbiop_vcf::reader::VcfReader;
    /// # use std::path::Path;
    /// let mut reader = VcfReader::open(Path::new("variants.vcf")).unwrap();
    /// let variants = reader.read_all().unwrap();
    /// println!("Found {} variants", variants.len());
    /// ```
    pub fn read_all(&mut self) -> Result<Vec<Variant>> {
        let mut variants = Vec::new();

        // Collect records to avoid borrow checker issue
        let records: Vec<vcf::Record> = self
            .reader
            .records()
            .collect::<std::io::Result<Vec<_>>>()
            .context("Failed to read VCF records")?;

        for record in records {
            let variant = Self::record_to_variant(&record)?;
            variants.push(variant);
        }

        Ok(variants)
    }

    /// Convert a noodles VCF record to our Variant type
    fn record_to_variant(record: &vcf::Record) -> Result<Variant> {
        let chromosome = record.reference_sequence_name().to_string();

        // Get position - VCF is 1-based, variant_start() returns Option<Result<Position>>
        let position = match record.variant_start() {
            Some(Ok(pos)) => usize::from(pos) as u64,
            Some(Err(e)) => anyhow::bail!("Failed to get variant position: {}", e),
            None => anyhow::bail!("No variant start position"),
        };

        // Get ID - may be empty
        let id = if record.ids().is_empty() {
            None
        } else {
            // Get first ID
            record
                .ids()
                .as_ref()
                .split(',')
                .next()
                .map(|s| s.to_string())
        };

        let reference_allele = record.reference_bases().to_string();

        // Get alternate alleles
        let alternate_alleles: Vec<String> = record
            .alternate_bases()
            .as_ref()
            .split(',')
            .map(|s| s.to_string())
            .collect();

        // Get quality score - returns Option<Result<f32>>
        let quality = match record.quality_score() {
            Some(Ok(q)) => Some(q),
            _ => None,
        };

        // Get filters
        let filter: Vec<String> = if record.filters().is_empty() {
            vec![]
        } else {
            record
                .filters()
                .as_ref()
                .split(';')
                .map(|s| s.to_string())
                .collect()
        };

        // Extract INFO fields - simplified version
        let mut info = ahash::HashMap::default();

        // For now, store the raw INFO string
        // In a full implementation, you'd parse each field according to the header
        info.insert("raw_info".to_string(), format!("{:?}", record.info()));

        Ok(Variant {
            chromosome,
            position,
            id,
            reference_allele,
            alternate_alleles,
            quality,
            filter,
            info,
        })
    }

    /// Filter variants by quality score
    ///
    /// # Arguments
    ///
    /// * `min_quality` - Minimum quality score threshold
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use deepbiop_vcf::reader::VcfReader;
    /// # use std::path::Path;
    /// let mut reader = VcfReader::open(Path::new("variants.vcf")).unwrap();
    /// let high_quality = reader.filter_by_quality(30.0).unwrap();
    /// ```
    pub fn filter_by_quality(&mut self, min_quality: f32) -> Result<Vec<Variant>> {
        let all_variants = self.read_all()?;
        Ok(all_variants
            .into_iter()
            .filter(|v| v.quality.is_some_and(|q| q >= min_quality))
            .collect())
    }

    /// Filter variants that pass all filters
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use deepbiop_vcf::reader::VcfReader;
    /// # use std::path::Path;
    /// let mut reader = VcfReader::open(Path::new("variants.vcf")).unwrap();
    /// let passing = reader.filter_passing().unwrap();
    /// ```
    pub fn filter_passing(&mut self) -> Result<Vec<Variant>> {
        let all_variants = self.read_all()?;
        Ok(all_variants
            .into_iter()
            .filter(|v| v.passes_filter())
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variant_creation() {
        let variant = Variant {
            chromosome: "chr1".to_string(),
            position: 100,
            id: Some("rs123".to_string()),
            reference_allele: "A".to_string(),
            alternate_alleles: vec!["G".to_string()],
            quality: Some(30.0),
            filter: vec!["PASS".to_string()],
            info: ahash::HashMap::default(),
        };

        assert_eq!(variant.chromosome, "chr1");
        assert!(variant.passes_filter());
    }
}
