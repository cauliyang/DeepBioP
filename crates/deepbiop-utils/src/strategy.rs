use ahash::HashMap;
use ahash::HashMapExt;
use rayon::prelude::*;

pub fn majority_voting(labels: &[i8], window_size: usize) -> Vec<i8> {
    // Adjust window size to be odd
    let window_size = if window_size % 2 == 0 {
        window_size + 1
    } else {
        window_size
    };

    let half_window = window_size / 2;
    let len = labels.len();

    labels
        .par_iter()
        .enumerate()
        .map(|(i, _)| {
            let mut start = i.saturating_sub(half_window);
            let end = usize::min(len, i + half_window + 1);

            if end == len && (end - start) < window_size {
                start = end.saturating_sub(window_size);
            }

            let window: Vec<_> = labels[start..end].to_vec();

            // if end == len {
            //     window.extend(vec![1; i + half_window + 1 - end]); // This assumes '1' is the label to extend
            // }

            // Compute the most common element in the window
            let mut counts = HashMap::new();
            for label in window {
                *counts.entry(label).or_insert(0) += 1;
            }

            // if counts has two key and the value is the same, return the current label
            if counts.len() == 2 {
                let values: Vec<_> = counts.values().collect();
                if values[0] == values[1] {
                    return labels[i];
                }
            }

            counts
                .iter()
                .max_by_key(|(_, &count)| count)
                .map(|(&label, _)| label)
                .unwrap()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_majority_voting() {
        let labels = vec![1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1];
        let window_size = 3;
        let result = majority_voting(&labels, window_size);
        println!("{:?}", result);
        let expected = vec![1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_majority_voting2() {
        let labels = vec![1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1];
        let window_size = 3;
        let result = majority_voting(&labels, window_size);
        println!("{:?}", result);
        let expected = vec![1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_majority_voting_empty_labels() {
        let labels = vec![];
        let window_size = 3;
        let result = majority_voting(&labels, window_size);
        println!("{:?}", result);
    }

    #[test]
    fn test_majority_voting_small_window() {
        let labels = vec![1, 0, 0, 1, 1, 0, 1, 0, 0, 0];
        let window_size = 1;
        let result = majority_voting(&labels, window_size);
        assert_eq!(result, vec![1, 0, 0, 1, 1, 0, 1, 0, 0, 0]);
    }
}
