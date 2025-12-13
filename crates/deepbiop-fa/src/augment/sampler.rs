//! Random subsequence sampling for FASTA records.

use anyhow::Result;
use noodles::fasta::record::Definition;
use noodles::fasta::Record as FastaRecord;
use rand::Rng;

/// Randomly cuts FASTA sequences to create augmented sequences
///
/// # Arguments
///
/// * `records` - Vector of FASTA records to augment
/// * `min_length` - Minimum length of cut sequences
/// * `max_length` - Maximum length of cut sequences
/// * `num_cuts` - Number of cuts to make per sequence
///
/// # Returns
///
/// Returns a vector of new FASTA records containing the cut sequences
///
/// # Example
///
/// ```no_run
/// use deepbiop_fa::augment::cut_fa_randomly;
/// use noodles::fasta::Record;
///
/// # fn example(records: &[Record]) -> anyhow::Result<()> {
/// let augmented = cut_fa_randomly(records, 100, 500, 3)?;
/// println!("Generated {} augmented sequences", augmented.len());
/// # Ok(())
/// # }
/// ```
pub fn cut_fa_randomly(
    records: &[FastaRecord],
    min_length: usize,
    max_length: usize,
    num_cuts: usize,
) -> Result<Vec<FastaRecord>> {
    let mut rng = rand::rng();
    let mut augmented_records = Vec::new();

    for record in records {
        let sequence = record.sequence().as_ref();
        let seq_len = sequence.len();

        // Skip sequences shorter than min_length
        if seq_len < min_length {
            continue;
        }

        for i in 0..num_cuts {
            // Generate random cut length between min and max
            let cut_len = rng.random_range(min_length..=max_length.min(seq_len));

            // Generate random start position
            let max_start = seq_len - cut_len;
            let start = rng.random_range(0..=max_start);

            // Create new sequence from cut region
            let cut_seq = sequence[start..start + cut_len].to_vec();

            // Create new record name with cut information
            let new_name = format!(
                "{}_cut{}_{}_{}",
                String::from_utf8_lossy(record.name()),
                i + 1,
                start,
                start + cut_len
            );
            // Create new record
            let new_record =
                FastaRecord::new(Definition::new(new_name.as_bytes(), None), cut_seq.into());
            augmented_records.push(new_record);
        }
    }

    Ok(augmented_records)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cut_fa_randomly() {
        use noodles::fasta::record::Sequence;

        // Create a test sequence
        let sequence = b"ACGTACGTACGTACGTACGTACGTACGTACGT".to_vec(); // 32 bases
        let record = FastaRecord::new(Definition::new(b"test_seq", None), Sequence::from(sequence));

        let records = vec![record];
        let result = cut_fa_randomly(&records, 10, 20, 2);

        assert!(result.is_ok());
        let augmented = result.unwrap();
        assert_eq!(augmented.len(), 2); // 2 cuts from 1 sequence

        // Verify cut lengths are within bounds
        for aug_record in augmented {
            let len = aug_record.sequence().len();
            assert!((10..=20).contains(&len));
        }
    }
}
