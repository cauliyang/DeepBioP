//! Feature extraction from BAM alignment records for machine learning

use ahash::HashMap;
use anyhow::{Context, Result};
use noodles::sam;
use noodles::sam::alignment::record::Cigar;

/// Features extracted from an alignment record for ML analysis
#[derive(Debug, Clone)]
pub struct AlignmentFeatures {
    /// Mapping quality score (0-255, 0=unmapped, 255=highest quality)
    pub mapping_quality: u8,

    /// Is the read mapped to the reference?
    pub is_mapped: bool,

    /// Is this read part of a paired-end sequencing run?
    pub is_paired: bool,

    /// Is this a supplementary alignment?
    pub is_supplementary: bool,

    /// Is this a secondary alignment?
    pub is_secondary: bool,

    /// Is the mate read mapped (for paired reads)?
    pub is_mate_mapped: Option<bool>,

    /// Template length (insert size for paired reads)
    pub template_length: i32,

    /// Number of aligned bases (from CIGAR)
    pub aligned_length: usize,

    /// Number of matches (M operations in CIGAR)
    pub num_matches: usize,

    /// Number of insertions (I operations in CIGAR)
    pub num_insertions: usize,

    /// Number of deletions (D operations in CIGAR)
    pub num_deletions: usize,

    /// Number of soft clips (S operations in CIGAR)
    pub num_soft_clips: usize,

    /// Number of hard clips (H operations in CIGAR)
    pub num_hard_clips: usize,

    /// Edit distance to reference (from NM tag if present)
    pub edit_distance: Option<u32>,

    /// Additional SAM tags
    pub tags: HashMap<String, String>,
}

impl AlignmentFeatures {
    /// Extract features from a SAM/BAM record
    ///
    /// # Example
    ///
    /// ```no_run
    /// use deepbiop_bam::features::AlignmentFeatures;
    /// use noodles::bam;
    ///
    /// # fn example(record: &bam::Record) -> anyhow::Result<()> {
    /// let features = AlignmentFeatures::from_record(record)?;
    /// println!("Mapping quality: {}", features.mapping_quality);
    /// println!("Is mapped: {}", features.is_mapped);
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_record(record: &impl sam::alignment::Record) -> Result<Self> {
        // Extract flags
        let flags = record.flags()?;
        let is_mapped = !flags.is_unmapped();
        let is_paired = flags.is_segmented();
        let is_supplementary = flags.is_supplementary();
        let is_secondary = flags.is_secondary();

        // Mate mapping status (only for paired reads)
        let is_mate_mapped = if is_paired {
            Some(!flags.is_mate_unmapped())
        } else {
            None
        };

        // Mapping quality
        let mapping_quality = match record.mapping_quality() {
            Some(mq_result) => {
                let mq = mq_result.context("Failed to read mapping quality")?;
                u8::from(mq)
            }
            None => 0,
        };

        // Template length
        let template_length = record.template_length()?;

        // Parse CIGAR to extract alignment statistics
        let (
            aligned_length,
            num_matches,
            num_insertions,
            num_deletions,
            num_soft_clips,
            num_hard_clips,
        ) = Self::parse_cigar(record)?;

        // Extract edit distance from NM tag
        let edit_distance = Self::extract_edit_distance(record);

        // Extract additional tags as strings
        let tags = Self::extract_tags(record);

        Ok(Self {
            mapping_quality,
            is_mapped,
            is_paired,
            is_supplementary,
            is_secondary,
            is_mate_mapped,
            template_length,
            aligned_length,
            num_matches,
            num_insertions,
            num_deletions,
            num_soft_clips,
            num_hard_clips,
            edit_distance,
            tags,
        })
    }

    /// Parse CIGAR string to extract alignment statistics
    fn parse_cigar(
        record: &impl sam::alignment::Record,
    ) -> Result<(usize, usize, usize, usize, usize, usize)> {
        let cigar = record.cigar();

        let mut aligned_length = 0;
        let mut num_matches = 0;
        let mut num_insertions = 0;
        let mut num_deletions = 0;
        let mut num_soft_clips = 0;
        let mut num_hard_clips = 0;

        for result in cigar.iter() {
            let op = result.context("Failed to parse CIGAR operation")?;
            let len = op.len();

            use noodles::sam::alignment::record::cigar::op::Kind;
            match op.kind() {
                Kind::Match | Kind::SequenceMatch | Kind::SequenceMismatch => {
                    num_matches += len;
                    aligned_length += len;
                }
                Kind::Insertion => {
                    num_insertions += len;
                }
                Kind::Deletion => {
                    num_deletions += len;
                    aligned_length += len;
                }
                Kind::SoftClip => {
                    num_soft_clips += len;
                }
                Kind::HardClip => {
                    num_hard_clips += len;
                }
                Kind::Skip => {
                    aligned_length += len;
                }
                Kind::Pad => {
                    // Padding doesn't consume reference or query
                }
            }
        }

        Ok((
            aligned_length,
            num_matches,
            num_insertions,
            num_deletions,
            num_soft_clips,
            num_hard_clips,
        ))
    }

    /// Extract edit distance from NM tag
    fn extract_edit_distance(record: &impl sam::alignment::Record) -> Option<u32> {
        // Try to get NM tag (edit distance)
        let data = record.data();

        // Iterate through tags to find NM
        for (tag, value) in data.iter().flatten() {
            if tag.as_ref() == b"NM" {
                // Try to extract integer value from Value enum
                // Value types can be Int8, UInt8, Int16, etc.
                use noodles::sam::alignment::record::data::field::Value;
                match value {
                    Value::Int8(n) => return Some(n as u32),
                    Value::UInt8(n) => return Some(n as u32),
                    Value::Int16(n) => return Some(n as u32),
                    Value::UInt16(n) => return Some(n as u32),
                    Value::Int32(n) => return Some(n as u32),
                    Value::UInt32(n) => return Some(n),
                    _ => {}
                }
            }
        }

        None
    }

    /// Extract SAM tags as string map
    fn extract_tags(record: &impl sam::alignment::Record) -> HashMap<String, String> {
        let mut tags = HashMap::default();
        let data = record.data();

        for (tag, value) in data.iter().flatten() {
            let tag_str = String::from_utf8_lossy(tag.as_ref()).to_string();
            // Convert Value enum to string representation
            let value_str = format!("{:?}", value);
            tags.insert(tag_str, value_str);
        }

        tags
    }

    /// Calculate alignment identity (matches / aligned_length)
    pub fn identity(&self) -> f32 {
        if self.aligned_length == 0 {
            0.0
        } else {
            self.num_matches as f32 / self.aligned_length as f32
        }
    }

    /// Calculate indel rate ((insertions + deletions) / aligned_length)
    pub fn indel_rate(&self) -> f32 {
        if self.aligned_length == 0 {
            0.0
        } else {
            (self.num_insertions + self.num_deletions) as f32 / self.aligned_length as f32
        }
    }

    /// Check if this is a high-quality alignment (mapping quality >= threshold)
    pub fn is_high_quality(&self, min_quality: u8) -> bool {
        self.is_mapped && self.mapping_quality >= min_quality
    }

    /// Check if this is a proper pair (both mates mapped, reasonable insert size)
    pub fn is_proper_pair(&self, max_insert_size: i32) -> bool {
        self.is_paired
            && self.is_mapped
            && self.is_mate_mapped.unwrap_or(false)
            && self.template_length.abs() <= max_insert_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alignment_features_calculations() {
        let features = AlignmentFeatures {
            mapping_quality: 60,
            is_mapped: true,
            is_paired: true,
            is_supplementary: false,
            is_secondary: false,
            is_mate_mapped: Some(true),
            template_length: 300,
            aligned_length: 100,
            num_matches: 95,
            num_insertions: 2,
            num_deletions: 3,
            num_soft_clips: 5,
            num_hard_clips: 0,
            edit_distance: Some(5),
            tags: HashMap::default(),
        };

        assert_eq!(features.identity(), 0.95);
        assert_eq!(features.indel_rate(), 0.05);
        assert!(features.is_high_quality(30));
        assert!(features.is_proper_pair(500));
    }
}
