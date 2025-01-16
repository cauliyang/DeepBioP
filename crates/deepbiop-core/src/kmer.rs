use itertools::Itertools;
use rayon::prelude::*;
use std::ops::Range;

use crate::{
    error::DPError,
    types::{Element, Id2KmerTable, Kmer2IdTable},
};
use anyhow::Error;
use anyhow::Result;

/// Convert a sequence of k-mer IDs back into a DNA sequence.
///
/// This function takes a slice of k-mer IDs and a lookup table mapping IDs to k-mers,
/// and reconstructs the original DNA sequence by converting each ID back to its k-mer
/// and joining them together.
///
/// # Arguments
///
/// * `kmer_ids` - A slice of integer IDs representing k-mers
/// * `id2kmer_table` - A HashMap mapping k-mer IDs to their byte sequences
///
/// # Returns
///
/// The reconstructed DNA sequence as a vector of bytes, wrapped in a Result.
/// Returns an error if any k-mer ID is not found in the lookup table.
///
/// # Errors
///
/// Returns `DPError::InvalidKmerId` if a k-mer ID is not found in the lookup table
pub fn kmerids_to_seq(kmer_ids: &[Element], id2kmer_table: Id2KmerTable) -> Result<Vec<u8>> {
    let result = kmer_ids
        .par_iter()
        .map(|&id| {
            id2kmer_table
                .get(&id)
                .ok_or(Error::new(DPError::InvalidKmerId))
                .map(|kmer| kmer.as_ref())
        })
        .collect::<Result<Vec<_>>>()?;

    kmers_to_seq(result)
}

/// Convert a k-mer target region back to the original sequence target region.
///
/// This function takes a target region that was adjusted for k-mer calculations and converts it
/// back to the corresponding region in the original sequence by reversing the k-mer adjustments.
///
/// # Arguments
///
/// * `kmer_target` - The target region adjusted for k-mers as a Range<usize>
/// * `k` - The length of k-mers used
///
/// # Returns
///
/// A Range<usize> representing the target region in the original sequence
///
/// # Example
///
/// ```
/// use std::ops::Range;
/// use deepbiop_core::kmer::to_original_target_region;
///
/// let kmer_target = 0..3;  // A k-mer target region
/// let k = 4;  // k-mer length
/// let original_target = to_original_target_region(&kmer_target, k);
/// assert_eq!(original_target, 0..6);  // Original sequence region
/// ```
pub fn to_original_target_region(kmer_target: &Range<usize>, k: usize) -> Range<usize> {
    // The start of the target region remains the same
    let original_start = kmer_target.start;

    // Attempt to reverse the end adjustment by adding k - 1, assuming the adjustment was due to k-mer calculation
    let original_end = if kmer_target.end > original_start {
        kmer_target.end + k - 1
    } else {
        kmer_target.end
    };

    original_start..original_end
}

/// Convert an original sequence target region to a k-mer target region.
///
/// This function takes a target region from the original sequence and converts it to the corresponding
/// region for k-mer calculations by adjusting for k-mer length and sequence boundaries.
///
/// # Arguments
///
/// * `original_target` - The target region in the original sequence as a Range<usize>
/// * `k` - The length of k-mers to use
/// * `seq_len` - Optional sequence length to validate target region bounds
///
/// # Returns
///
/// A Result containing a Range<usize> representing the adjusted k-mer target region
///
/// # Errors
///
/// Returns `DPError::TargetRegionInvalid` if:
/// - The target region is invalid (start >= end)
/// - k is 0
/// - The target region extends beyond sequence length (if seq_len provided)
///
/// # Example
///
/// ```
/// use std::ops::Range;
/// use deepbiop_core::kmer::to_kmer_target_region;
///
/// let original_target = 0..6;  // Original sequence region
/// let k = 4;  // k-mer length
/// let kmer_target = to_kmer_target_region(&original_target, k, Some(10)).unwrap();
/// assert_eq!(kmer_target, 0..3);  // Adjusted k-mer region
/// ```
pub fn to_kmer_target_region(
    original_target: &Range<usize>,
    k: usize,
    seq_len: Option<usize>,
) -> Result<Range<usize>> {
    if original_target.start >= original_target.end || k == 0 {
        return Err(Error::from(DPError::TargetRegionInvalid));
    }

    if let Some(seq_len) = seq_len {
        // Ensure the target region is valid.
        if original_target.end > seq_len {
            return Err(Error::new(DPError::TargetRegionInvalid));
        }
    }

    // Calculate how many k-mers can be formed starting within the original target region.
    let num_kmers_in_target = if original_target.end - original_target.start >= k {
        original_target.end - original_target.start - k + 1
    } else {
        0
    };

    // The new target region starts at the same position as the original target region.
    let new_start = original_target.start;

    // The end of the new target region needs to be adjusted based on the number of k-mers.
    // It is the start position of the last k-mer that can be formed within the original target region.
    let new_end = if num_kmers_in_target > 0 {
        new_start + num_kmers_in_target
    } else {
        original_target.end
    };

    Ok(new_start..new_end)
}

/// Convert a DNA sequence into k-mers.
///
/// This function takes a DNA sequence and splits it into k-mers of specified length.
/// The k-mers can be either overlapping or non-overlapping based on the `overlap` parameter.
///
/// # Arguments
///
/// * `seq` - A byte slice containing the DNA sequence
/// * `k` - The length of each k-mer
/// * `overlap` - Whether to generate overlapping k-mers
///
/// # Returns
///
/// A vector of k-mer byte slices
///
/// # Example
///
/// ```
/// use deepbiop_core::kmer::seq_to_kmers;
///
/// let seq = b"ATCGATCG";
///
/// // Overlapping k-mers
/// let kmers = seq_to_kmers(seq, 3, true);
/// assert_eq!(kmers, vec![b"ATC", b"TCG", b"CGA", b"GAT", b"ATC", b"TCG"]);
///
/// // Non-overlapping k-mers
/// let kmers = seq_to_kmers(seq, 3, false);
/// assert_eq!(kmers, vec![b"ATC".as_slice(), b"GAT".as_slice(), b"CG".as_slice()]);
/// ```
pub fn seq_to_kmers(seq: &[u8], k: usize, overlap: bool) -> Vec<&[u8]> {
    if overlap {
        seq.par_windows(k).collect()
    } else {
        seq.par_chunks(k).collect()
    }
}
/// Convert k-mers back into a DNA sequence.
///
/// This function takes a vector of k-mers and reconstructs the original DNA sequence.
/// The k-mers are assumed to be in order and overlapping by k-1 bases.
///
/// # Arguments
///
/// * `kmers` - A vector of k-mer byte slices
///
/// # Returns
///
/// A Result containing the reconstructed DNA sequence as a byte vector
///
/// # Errors
///
/// Returns an error if any k-mer is invalid (empty)
///
/// # Example
///
/// ```
/// use deepbiop_core::kmer::kmers_to_seq;
///
/// let kmers = vec![b"ATC".as_slice(), b"TCG".as_slice(), b"CGA".as_slice()];
/// let seq = kmers_to_seq(kmers).unwrap();
/// assert_eq!(seq, b"ATCGA");
/// ```
pub fn kmers_to_seq(kmers: Vec<&[u8]>) -> Result<Vec<u8>> {
    // Early return for empty input
    if kmers.is_empty() {
        return Ok(Vec::new());
    }

    // Validate first kmer
    let first_kmer = kmers[0];
    if first_kmer.is_empty() {
        return Err(DPError::InvalidKmerId.into());
    }

    // Initialize result with first kmer
    let mut result = Vec::with_capacity(first_kmer.len() + kmers.len() - 1);
    result.extend_from_slice(first_kmer);

    // Process remaining kmers in parallel
    let remaining_bases: Result<Vec<u8>> = kmers
        .into_par_iter()
        .skip(1)
        .map(|kmer| {
            kmer.last()
                .ok_or_else(|| DPError::InvalidKmerId.into())
                .copied()
        })
        .collect();

    // Extend result with remaining bases
    result.extend(remaining_bases?);

    Ok(result)
}

/// Convert a DNA sequence into k-mers with their positions in the original sequence.
///
/// This function takes a DNA sequence and splits it into k-mers of specified length,
/// returning both the k-mers and their start/end positions in the original sequence.
///
/// # Arguments
///
/// * `seq` - A DNA sequence as a byte slice
/// * `kmer_size` - The length of each k-mer
/// * `overlap` - Whether to generate overlapping k-mers
///
/// # Returns
///
/// A Result containing a vector of tuples, where each tuple contains:
/// - A k-mer as a byte slice
/// - A tuple of (start_position, end_position) indicating the k-mer's location in the original sequence
///
/// # Errors
///
/// Returns an error if:
/// - `kmer_size` is 0
/// - `kmer_size` is greater than the sequence length
///
/// # Example
///
/// ```
/// use deepbiop_core::kmer::seq_to_kmers_and_offset;
///
/// let seq = b"ATCGA";
/// let result = seq_to_kmers_and_offset(seq, 3, true).unwrap();
/// assert_eq!(result.len(), 3);
/// assert_eq!(result[0], (b"ATC".as_slice(), (0, 3)));
/// assert_eq!(result[1], (b"TCG".as_slice(), (1, 4)));
/// assert_eq!(result[2], (b"CGA".as_slice(), (2, 5)));
/// ```
#[allow(clippy::type_complexity)]
pub fn seq_to_kmers_and_offset(
    seq: &[u8],
    kmer_size: usize,
    overlap: bool,
) -> Result<Vec<(&[u8], (usize, usize))>> {
    // Check for invalid kmer_size
    if kmer_size == 0 || kmer_size > seq.len() {
        return Err(DPError::SeqShorterThanKmer.into());
    }

    if seq.is_empty() {
        return Ok(Vec::new());
    }

    if overlap {
        // Overlapping case: use .windows() with step of 1 (default behavior of .windows())
        Ok(seq
            .par_windows(kmer_size)
            .enumerate()
            .map(|(i, kmer)| (kmer, (i, i + kmer_size)))
            .collect())
    } else {
        // Non-overlapping case: iterate with steps of kmer_size
        Ok(seq
            .par_chunks(kmer_size)
            .enumerate()
            .filter_map(|(i, chunk)| {
                if chunk.len() == kmer_size {
                    Some((chunk, (i * kmer_size, i * kmer_size + kmer_size)))
                } else {
                    // ignore the last chunk if it's shorter than kmer_size
                    None
                }
            })
            .collect())
    }
}

/// Generate a lookup table mapping k-mers to unique IDs.
///
/// This function takes a slice of base characters and a k-mer length,
/// and generates a HashMap mapping each possible k-mer to a unique integer ID.
///
/// # Arguments
///
/// * `base` - A slice containing the base characters to use (e.g. b"ATCG")
/// * `k` - The length of k-mers to generate
///
/// # Returns
///
/// A HashMap mapping k-mer byte sequences to integer IDs
///
/// # Example
///
/// ```
/// use deepbiop_core::kmer::generate_kmers_table;
///
/// let bases = b"AC";
/// let k = 2;
/// let table = generate_kmers_table(bases, k);
/// assert_eq!(table.len(), 4); // AA, AC, CA, CC
/// ```
pub fn generate_kmers_table(base: &[u8], k: u8) -> Kmer2IdTable {
    generate_kmers(base, k)
        .into_par_iter()
        .enumerate()
        .map(|(id, kmer)| (kmer, id as Element))
        .collect()
}

/// Generate all possible k-mers from a set of base characters.
///
/// This function takes a slice of base characters and a k-mer length,
/// and generates all possible k-mer combinations of that length.
///
/// # Arguments
///
/// * `bases` - A slice containing the base characters to use (e.g. b"ATCG")
/// * `k` - The length of k-mers to generate
///
/// # Returns
///
/// A vector containing all possible k-mer combinations as byte vectors
///
/// # Example
///
/// ```
/// use deepbiop_core::kmer::generate_kmers;
///
/// let bases = b"AC";
/// let k = 2;
/// let kmers = generate_kmers(bases, 2);
/// assert_eq!(kmers.len(), 4); // AA, AC, CA, CC
/// ```
pub fn generate_kmers(bases: &[u8], k: u8) -> Vec<Vec<u8>> {
    // Convert u8 slice to char Vec directly where needed
    (0..k)
        .map(|_| bases.iter().map(|&c| c as char)) // Direct conversion to char iter
        .multi_cartesian_product()
        .map(|combo| combo.into_iter().map(|c| c as u8).collect::<Vec<_>>())
        .collect::<Vec<_>>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use bio::utils::Interval;

    #[test]
    fn test_seq_to_kmers() {
        let seq1 = b"ATCGT";
        let k = 2;
        let kmers = seq_to_kmers(seq1, k, true);
        assert_eq!(kmers.len(), seq1.len() - k + 1);

        let seq2 = b"AT";
        let k = 3;
        let kmers = seq_to_kmers(seq2, k, true);
        println!("{:?}", kmers);
        assert_eq!(kmers.len(), 0);
    }

    #[test]
    fn test_generate_kmers() {
        // Test case 1: bases = ['A', 'C', 'G', 'T'], k = 2
        let bases1 = b"ACGT";
        let k1 = 2;
        let expected1 = vec![
            "AA", "AC", "AG", "AT", "CA", "CC", "CG", "CT", "GA", "GC", "GG", "GT", "TA", "TC",
            "TG", "TT",
        ]
        .into_iter()
        .map(|s| s.chars().map(|c| c as u8).collect::<Vec<_>>())
        .collect::<Vec<_>>();

        assert_eq!(generate_kmers(bases1, k1), expected1);

        // Test case 2: bases = ['A', 'C'], k = 3
        let bases2 = b"AC";
        let k2 = 3;
        let expected2 = vec!["AAA", "AAC", "ACA", "ACC", "CAA", "CAC", "CCA", "CCC"]
            .into_iter()
            .map(|s| s.chars().map(|c| c as u8).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        assert_eq!(generate_kmers(bases2, k2), expected2);
    }

    #[test]
    fn test_generate_kmers_table() {
        let base = b"ACGT";
        let k = 2;
        let expected_table: Kmer2IdTable = [
            ("AA", 0),
            ("GC", 9),
            ("GT", 11),
            ("CA", 4),
            ("TA", 12),
            ("TC", 13),
            ("CG", 6),
            ("CT", 7),
            ("GA", 8),
            ("AG", 2),
            ("AC", 1),
            ("AT", 3),
            ("CC", 5),
            ("GG", 10),
            ("TG", 14),
            ("TT", 15),
        ]
        .iter()
        .map(|&(kmer, id)| (kmer.chars().map(|c| c as u8).collect(), id))
        .collect();

        assert_eq!(generate_kmers_table(base, k), expected_table);
    }

    #[test]
    fn test_generate_kmers_table_empty_base() {
        use ahash::HashMap;
        use ahash::HashMapExt;

        let base = b"";
        let k = 2;
        let expected_table: Kmer2IdTable = HashMap::new();
        assert_eq!(generate_kmers_table(base, k), expected_table);
    }

    #[test]
    fn test_construct_seq_from_kmers() {
        let k = 3;
        let seq = b"AAACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let kmers = seq_to_kmers(seq, k, true);
        let kmers_as_bytes: Vec<&[u8]> = kmers.into_iter().collect();
        let result = kmers_to_seq(kmers_as_bytes).unwrap();
        assert_eq!(seq.to_vec(), result);
    }

    #[test]
    fn test_update_target_region() {
        let original_target: Interval<usize> = (2..6).into(); // Target region [2, 6)
        let k = 3; // K-mer size
        let new_target_region = to_kmer_target_region(&original_target, k, None).unwrap();
        assert_eq!(new_target_region, 2..4);
    }

    #[test]
    fn test_update_target_region_valid() {
        let original_target = Interval::new(0..10).unwrap();
        let k = 3;
        let seq_len = Some(20);

        let result = to_kmer_target_region(&original_target, k, seq_len);

        assert!(result.is_ok());
        let new_target = result.unwrap();

        assert_eq!(new_target.start, original_target.start);
        assert_eq!(new_target.end, original_target.start + 8);
    }

    #[test]
    fn test_update_target_region_invalid_start_greater_than_end() {
        let original_target = Interval::new(10..10).unwrap();
        let k = 3;
        let seq_len = Some(20);

        let result = to_kmer_target_region(&original_target, k, seq_len);
        assert!(result.is_err());

        assert_eq!(
            result.unwrap_err().to_string(),
            DPError::TargetRegionInvalid.to_string()
        );
    }

    #[test]
    fn test_update_target_region_invalid_end_greater_than_seq_len() {
        let original_target = Interval::new(0..25).unwrap();
        let k = 3;
        let seq_len = Some(20);

        let result = to_kmer_target_region(&original_target, k, seq_len);

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            DPError::TargetRegionInvalid.to_string()
        );
    }

    #[test]
    fn test_to_original_target_region() {
        // Test case 1: kmer_target.end > original_start
        let kmer_target = 2..5;
        let k = 3;
        let expected = 2..7;

        assert_eq!(
            to_kmer_target_region(&expected, k, None).unwrap(),
            kmer_target
        );
        assert_eq!(to_original_target_region(&kmer_target, k), expected);

        // Test case 3: kmer_target.end == original_start
        let kmer_target = 5..5;
        let k = 3;
        let expected = 5..5;
        assert_eq!(to_original_target_region(&kmer_target, k), expected);
    }

    #[test]
    fn test_seq_to_kmers_and_offset_overlap() {
        let seq = b"ATCGATCGATCG";
        let kmer_size = 4;
        let overlap = true;
        let result = seq_to_kmers_and_offset(seq, kmer_size, overlap).unwrap();
        assert_eq!(result.len(), seq.len() - kmer_size + 1);
        assert_eq!(result[0], (&b"ATCG"[..], (0, 4)));
        assert_eq!(result[1], (&b"TCGA"[..], (1, 5)));
        assert_eq!(result[result.len() - 1], (&b"ATCG"[..], (8, 12)));
    }

    #[test]
    fn test_seq_to_kmers_and_offset_non_overlap() {
        let seq = b"ATCGATCGATCG";
        let kmer_size = 4;
        let overlap = false;
        let result = seq_to_kmers_and_offset(seq, kmer_size, overlap).unwrap();
        assert_eq!(result.len(), seq.len() / kmer_size);
        assert_eq!(result[0], (&b"ATCG"[..], (0, 4)));
        assert_eq!(result[1], (&b"ATCG"[..], (4, 8)));
    }
}
