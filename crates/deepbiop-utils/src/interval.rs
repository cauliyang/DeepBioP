use anyhow::anyhow;
use anyhow::Result;
use bstr::BStr;

use rayon::prelude::*;
use std::ops::Range;

pub fn generate_unmaped_intervals(
    input: &[Range<usize>],
    total_length: usize,
) -> Vec<Range<usize>> {
    // Assuming the input ranges are sorted and non-overlapping
    let mut result = Vec::new();

    if input.is_empty() {
        result.push(0..total_length);
        return result;
    }

    // Initial start for the very first interval
    let mut current_start = 0;

    for range in input.iter() {
        // Check if there's a gap between current_start and the start of the current range
        if current_start < range.start {
            result.push(current_start..range.start);
        }
        // Update current_start to the end of the current range
        current_start = range.end;
    }

    // Optionally handle the case after the last interval if necessary,
    // For example, if you know the total length and want to add an interval up to that length

    if current_start < total_length - 1 {
        result.push(current_start..total_length - 1);
    }

    result
}

// Function to remove intervals from a sequence and keep the remaining parts
pub fn remove_intervals_and_keep_left<'a>(
    seq: &'a [u8],
    intervals: &[Range<usize>],
) -> Result<(Vec<&'a BStr>, Vec<Range<usize>>)> {
    let mut intervals = intervals.to_vec();
    intervals.par_sort_by(|a: &Range<usize>, b: &Range<usize>| a.start.cmp(&b.start));
    let selected_intervals = generate_unmaped_intervals(&intervals, seq.len());

    let selected_seq = selected_intervals
        .par_iter()
        .map(|interval| {
            // Check if the interval is valid and starts after the current start point
            if interval.start < seq.len() {
                // Add the segment before the current interval
                Ok(seq[interval.start..interval.end].as_ref())
            } else {
                Err(anyhow!(format!("invalid {:?}", interval)))
            }
        })
        .collect::<Result<Vec<_>>>()?;

    Ok((selected_seq, selected_intervals))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remove_intervals_and_keep_left() {
        let seq = b"abcdefghijklmnopqrstuvwxyz";
        // |a| bcde |fghij| klmno |pqrst| uvwxyz

        let intervals = vec![1..5, 10..15, 20..25];
        let (seq, _inters) = remove_intervals_and_keep_left(seq, &intervals).unwrap();
        assert_eq!(seq, vec!["a", "fghij", "pqrst"]);

        let seq = b"abcdefghijklmnopqrstuvwxyz";
        let intervals = vec![5..10, 15..20];
        let (seq, _inters) = remove_intervals_and_keep_left(seq, &intervals).unwrap();
        assert_eq!(seq, vec!["abcde", "klmno", "uvwxy"]);
    }

    #[allow(clippy::single_range_in_vec_init)]
    #[test]
    fn test_generate_unmaped_intervals() {
        let intervals = vec![8100..8123];
        let seq_len = 32768;
        let selected_intervals = generate_unmaped_intervals(&intervals, seq_len);
        assert_eq!(selected_intervals, vec![0..8100, 8123..32767]);
    }
}
