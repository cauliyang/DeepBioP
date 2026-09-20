//! Encoding scheme types for biological sequence representations.

use crate::error::DPError;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

// TODO: consider add N for dna and rna

/// Encoding scheme enum defining different types of sequence encodings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EncodingScheme {
    /// One-hot encoding with specified alphabet size (e.g., 4 for DNA, 20 for protein)
    OneHot { alphabet_size: usize },
    /// Integer encoding (A=0, C=1, G=2, T=3)
    Integer { max_value: usize },
    /// K-mer encoding with k-mer length and canonical flag
    Kmer { k: usize, canonical: bool },
    /// Custom user-defined encoding
    Custom { name: String },
}

/// Encoding type specifies the biological sequence type being encoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EncodingType {
    /// DNA sequence (A, C, G, T)
    DNA,
    /// RNA sequence (A, C, G, U)
    RNA,
    /// Protein sequence (20 amino acids)
    Protein,
}

impl EncodingType {
    /// Get the alphabet size for this encoding type.
    pub fn alphabet_size(&self) -> usize {
        match self {
            EncodingType::DNA | EncodingType::RNA => 4,
            EncodingType::Protein => 20,
        }
    }

    /// Get the alphabet characters for this encoding type.
    pub fn alphabet(&self) -> &'static [u8] {
        match self {
            EncodingType::DNA => b"ACGT",
            EncodingType::RNA => b"ACGU",
            EncodingType::Protein => b"ACDEFGHIKLMNPQRSTVWY",
        }
    }

    /// Check if a character is valid for this encoding type.
    pub fn is_valid_char(&self, c: u8) -> bool {
        self.alphabet().contains(&c.to_ascii_uppercase())
    }
}

impl FromStr for EncodingType {
    type Err = DPError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "dna" => Ok(EncodingType::DNA),
            "rna" => Ok(EncodingType::RNA),
            "protein" => Ok(EncodingType::Protein),
            _ => Err(DPError::InvalidValue(format!(
                "Invalid encoding type: '{}'. Expected 'dna', 'rna', or 'protein'",
                s
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoding_type_alphabet_size() {
        assert_eq!(EncodingType::DNA.alphabet_size(), 4);
        assert_eq!(EncodingType::RNA.alphabet_size(), 4);
        assert_eq!(EncodingType::Protein.alphabet_size(), 20);
    }

    #[test]
    fn test_encoding_type_is_valid_char() {
        assert!(EncodingType::DNA.is_valid_char(b'A'));
        assert!(EncodingType::DNA.is_valid_char(b'a'));
        assert!(EncodingType::DNA.is_valid_char(b'C'));
        assert!(EncodingType::DNA.is_valid_char(b'G'));
        assert!(EncodingType::DNA.is_valid_char(b'T'));
        assert!(!EncodingType::DNA.is_valid_char(b'U'));
        assert!(!EncodingType::DNA.is_valid_char(b'N'));
    }

    #[test]
    fn test_encoding_scheme_equality() {
        let scheme1 = EncodingScheme::OneHot { alphabet_size: 4 };
        let scheme2 = EncodingScheme::OneHot { alphabet_size: 4 };
        let scheme3 = EncodingScheme::OneHot { alphabet_size: 20 };

        assert_eq!(scheme1, scheme2);
        assert_ne!(scheme1, scheme3);
    }
}
