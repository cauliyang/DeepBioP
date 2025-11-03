//! Reverse complement transformation for DNA/RNA sequences.

use super::Augmentation;

/// Reverse complement transformation for DNA/RNA sequences.
///
/// This augmentation reverses the sequence and complements each base:
/// - DNA: A↔T, C↔G
/// - RNA: A↔U, C↔G
///
/// # Examples
///
/// ```
/// use deepbiop_fq::augment::{ReverseComplement, Augmentation};
///
/// let rc = ReverseComplement::new();
/// let sequence = b"ACGT";
/// let result = rc.apply(sequence);
/// assert_eq!(result, b"ACGT"); // Palindrome!
///
/// let sequence = b"AACCGGTT";
/// let result = rc.apply(sequence);
/// assert_eq!(result, b"AACCGGTT"); // Also palindrome!
/// ```
#[derive(Debug, Clone, Default)]
pub struct ReverseComplement {
    /// Whether to treat sequences as RNA (use U instead of T)
    is_rna: bool,
}

impl ReverseComplement {
    /// Create a new reverse complement transformer for DNA.
    pub fn new() -> Self {
        Self { is_rna: false }
    }

    /// Create a new reverse complement transformer for RNA.
    pub fn for_rna() -> Self {
        Self { is_rna: true }
    }

    /// Get the complement of a single base.
    #[inline]
    fn complement_base(&self, base: u8) -> u8 {
        match base {
            b'A' => {
                if self.is_rna {
                    b'U'
                } else {
                    b'T'
                }
            }
            b'a' => {
                if self.is_rna {
                    b'u'
                } else {
                    b't'
                }
            }
            b'T' => b'A',
            b't' => b'a',
            b'U' => b'A',
            b'u' => b'a',
            b'C' => b'G',
            b'c' => b'g',
            b'G' => b'C',
            b'g' => b'c',
            // Ambiguity codes (uppercase)
            b'R' => b'Y', // A/G -> T/C
            b'Y' => b'R', // C/T -> G/A
            b'S' => b'S', // G/C (self-complement)
            b'W' => b'W', // A/T (self-complement)
            b'K' => b'M', // G/T -> C/A
            b'M' => b'K', // A/C -> T/G
            b'B' => b'V', // C/G/T -> G/C/A
            b'D' => b'H', // A/G/T -> T/C/A
            b'H' => b'D', // A/C/T -> T/G/A
            b'V' => b'B', // A/C/G -> T/G/C
            b'N' => b'N', // Any base
            // Ambiguity codes (lowercase)
            b'r' => b'y',
            b'y' => b'r',
            b's' => b's',
            b'w' => b'w',
            b'k' => b'm',
            b'm' => b'k',
            b'b' => b'v',
            b'd' => b'h',
            b'h' => b'd',
            b'v' => b'b',
            b'n' => b'n',
            _ => base, // Unknown bases pass through
        }
    }
}

impl Augmentation for ReverseComplement {
    fn apply(&mut self, sequence: &[u8]) -> Vec<u8> {
        sequence
            .iter()
            .rev()
            .map(|&base| self.complement_base(base))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_reverse_complement() {
        let mut rc = ReverseComplement::new();
        assert_eq!(rc.apply(b"ACGT"), b"ACGT"); // Palindrome
        assert_eq!(rc.apply(b"A"), b"T");
        assert_eq!(rc.apply(b"T"), b"A");
        assert_eq!(rc.apply(b"C"), b"G");
        assert_eq!(rc.apply(b"G"), b"C");
    }

    #[test]
    fn test_reverse_complement_sequence() {
        let mut rc = ReverseComplement::new();
        assert_eq!(rc.apply(b"AAAAA"), b"TTTTT");
        assert_eq!(rc.apply(b"ACGTACGT"), b"ACGTACGT"); // Palindrome
        assert_eq!(rc.apply(b"ATCGATCG"), b"CGATCGAT");
        assert_eq!(rc.apply(b"GCGC"), b"GCGC"); // Palindrome
    }

    #[test]
    fn test_lowercase() {
        let mut rc = ReverseComplement::new();
        assert_eq!(rc.apply(b"acgt"), b"acgt");
        assert_eq!(rc.apply(b"AtCg"), b"cGaT");
    }

    #[test]
    fn test_rna_mode() {
        let mut rc = ReverseComplement::for_rna();
        assert_eq!(rc.apply(b"ACGU"), b"ACGU"); // RNA palindrome
        assert_eq!(rc.apply(b"A"), b"U");
        assert_eq!(rc.apply(b"U"), b"A");
    }

    #[test]
    fn test_ambiguity_codes() {
        let mut rc = ReverseComplement::new();
        // R (A/G) -> Y (T/C)
        assert_eq!(rc.apply(b"R"), b"Y");
        // Y (C/T) -> R (G/A)
        assert_eq!(rc.apply(b"Y"), b"R");
        // S (G/C) is self-complement
        assert_eq!(rc.apply(b"S"), b"S");
        // W (A/T) is self-complement
        assert_eq!(rc.apply(b"W"), b"W");
        // N (any) is self-complement
        assert_eq!(rc.apply(b"N"), b"N");
    }

    #[test]
    fn test_empty_sequence() {
        let mut rc = ReverseComplement::new();
        assert_eq!(rc.apply(b""), b"");
    }

    #[test]
    fn test_unknown_bases() {
        let mut rc = ReverseComplement::new();
        // Unknown bases should pass through
        assert_eq!(rc.apply(b"AXZ"), b"ZXT");
    }
}
