use crate::types::Variant;
use anyhow::{Context, Result};
use noodles::vcf;
use noodles::vcf::variant::record::info::field::value::Array as InfoArray;
use noodles::vcf::variant::record::info::field::Value as InfoValue;
use noodles::vcf::variant::record::{Filters, Ids};
use std::io::{BufReader, Read};
use std::path::Path;

/// VCF file reader.
///
/// Records are parsed lazily on the first query and kept, so every query
/// method can be called any number of times on one reader.
pub struct VcfReader {
    reader: Option<vcf::io::Reader<BufReader<Box<dyn Read + Send + Sync>>>>,
    header: vcf::Header,
    variants: Vec<Variant>,
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
        let file = deepbiop_utils::io::create_reader_for_compressed_file(path.as_ref())
            .with_context(|| format!("Failed to open VCF file: {:?}", path.as_ref()))?;

        let mut reader = vcf::io::Reader::new(BufReader::new(file));
        let header = reader.read_header().context("Failed to read VCF header")?;

        Ok(Self {
            reader: Some(reader),
            header,
            variants: Vec::new(),
        })
    }

    /// Parse the file on first use; subsequent calls reuse the parsed variants.
    fn variants(&mut self) -> Result<&[Variant]> {
        if let Some(mut reader) = self.reader.take() {
            let mut variants = Vec::new();
            for record in reader.records() {
                let record = record.context("Failed to read VCF record")?;
                variants.push(Self::record_to_variant(&record, &self.header)?);
            }
            self.variants = variants;
        }
        Ok(&self.variants)
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
        Ok(self.variants()?.to_vec())
    }

    /// Convert a noodles VCF record to our Variant type
    fn record_to_variant(record: &vcf::Record, header: &vcf::Header) -> Result<Variant> {
        let chromosome = record.reference_sequence_name().to_string();

        // Get position - VCF is 1-based, variant_start() returns Option<Result<Position>>
        let position = match record.variant_start() {
            Some(Ok(pos)) => usize::from(pos) as u64,
            Some(Err(e)) => anyhow::bail!("Failed to get variant position: {}", e),
            None => anyhow::bail!("No variant start position"),
        };

        // Get ID - may be empty; the VCF spec separates multiple IDs with ';'
        let id = if record.ids().is_empty() {
            None
        } else {
            // Get first ID
            record
                .ids()
                .as_ref()
                .split(';')
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

        // Extract INFO fields, parsed per the header's field-type definitions
        let mut info = ahash::HashMap::default();
        for result in record.info().iter(header) {
            let (key, value) = result.with_context(|| {
                format!("Failed to parse INFO field for variant at {chromosome}:{position}")
            })?;
            info.insert(key.to_string(), format_info_value(value)?);
        }

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
        Ok(self
            .variants()?
            .iter()
            .filter(|v| v.quality.is_some_and(|q| q >= min_quality))
            .cloned()
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
        Ok(self
            .variants()?
            .iter()
            .filter(|v| v.passes_filter())
            .cloned()
            .collect())
    }
}

/// Render a single INFO field value as its VCF-style display string.
///
/// A missing typed value (an explicit `.` in the VCF text) is rendered as
/// `.`; a bare flag is rendered as `true`.
fn format_info_value(value: Option<InfoValue<'_>>) -> Result<String> {
    match value {
        None => Ok(".".to_string()),
        Some(InfoValue::Integer(n)) => Ok(n.to_string()),
        Some(InfoValue::Float(n)) => Ok(n.to_string()),
        Some(InfoValue::Flag) => Ok("true".to_string()),
        Some(InfoValue::Character(c)) => Ok(c.to_string()),
        Some(InfoValue::String(s)) => Ok(s.to_string()),
        Some(InfoValue::Array(array)) => format_info_array(array),
    }
}

/// Render an INFO array value as a comma-joined string, with missing
/// elements rendered as `.` per the VCF spec.
fn format_info_array(array: InfoArray<'_>) -> Result<String> {
    let joined = match array {
        InfoArray::Integer(values) => values
            .iter()
            .map(|item| item.map(|opt| opt.map_or(".".to_string(), |n| n.to_string())))
            .collect::<std::io::Result<Vec<_>>>()
            .context("Failed to parse INFO integer array value")?
            .join(","),
        InfoArray::Float(values) => values
            .iter()
            .map(|item| item.map(|opt| opt.map_or(".".to_string(), |n| n.to_string())))
            .collect::<std::io::Result<Vec<_>>>()
            .context("Failed to parse INFO float array value")?
            .join(","),
        InfoArray::Character(values) => values
            .iter()
            .map(|item| item.map(|opt| opt.map_or(".".to_string(), |c| c.to_string())))
            .collect::<std::io::Result<Vec<_>>>()
            .context("Failed to parse INFO character array value")?
            .join(","),
        InfoArray::String(values) => values
            .iter()
            .map(|item| item.map(|opt| opt.map_or(".".to_string(), |s| s.to_string())))
            .collect::<std::io::Result<Vec<_>>>()
            .context("Failed to parse INFO string array value")?
            .join(","),
    };

    Ok(joined)
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

    fn write_test_vcf() -> tempfile::NamedTempFile {
        use std::io::Write;

        let mut file = tempfile::NamedTempFile::new().expect("Failed to create temp VCF file");
        write!(
            file,
            "##fileformat=VCFv4.3\n\
             ##INFO=<ID=DP,Number=1,Type=Integer,Description=\"Depth\">\n\
             ##INFO=<ID=AF,Number=A,Type=Float,Description=\"Allele Frequency\">\n\
             ##INFO=<ID=DB,Number=0,Type=Flag,Description=\"dbSNP membership\">\n\
             #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n\
             chr1\t100\t.\tA\tG\t30\tPASS\tDP=30;AF=0.5;DB\n\
             chr1\t200\trs1;rs2\tA\tG,T\t30\tPASS\tDP=30;AF=0.5,0.25\n"
        )
        .expect("Failed to write temp VCF file");
        file
    }

    #[test]
    fn test_record_to_variant_parses_info_fields() {
        let file = write_test_vcf();
        let mut reader = VcfReader::open(file.path()).expect("Failed to open temp VCF file");
        let variants = reader.read_all().expect("Failed to read variants");

        assert_eq!(variants.len(), 2);

        let first = &variants[0];
        assert_eq!(first.info.get("DP").map(String::as_str), Some("30"));
        assert_eq!(first.info.get("AF").map(String::as_str), Some("0.5"));
        assert_eq!(first.info.get("DB").map(String::as_str), Some("true"));
        assert_eq!(first.info.len(), 3);
    }

    #[test]
    fn test_record_to_variant_parses_multiallelic_info_array() {
        let file = write_test_vcf();
        let mut reader = VcfReader::open(file.path()).expect("Failed to open temp VCF file");
        let variants = reader.read_all().expect("Failed to read variants");

        let second = &variants[1];
        assert_eq!(second.info.get("AF").map(String::as_str), Some("0.5,0.25"));
        assert_eq!(second.id, Some("rs1".to_string()));
    }
}
