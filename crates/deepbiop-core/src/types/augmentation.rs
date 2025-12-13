//! Data augmentation types for biological sequences.

use serde::{Deserialize, Serialize};

/// Augmentation transform enum defining different types of data augmentation operations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AugmentationTransform {
    /// Reverse complement transformation
    ReverseComplement,
    /// Random point mutations with specified rate (0.0-1.0)
    Mutation { rate: f64, seed: Option<u64> },
    /// Extract random subsequence
    Subsample { length: usize, seed: Option<u64> },
    /// Combination of multiple transforms
    Composite {
        transforms: Vec<AugmentationTransform>,
    },
}

/// Transform type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransformType {
    /// Deterministic transformation (always produces same output for same input)
    Deterministic,
    /// Stochastic transformation (output varies with randomness)
    Stochastic,
}

impl AugmentationTransform {
    /// Get the transform type (deterministic or stochastic).
    pub fn transform_type(&self) -> TransformType {
        match self {
            AugmentationTransform::ReverseComplement => TransformType::Deterministic,
            AugmentationTransform::Mutation { .. } => TransformType::Stochastic,
            AugmentationTransform::Subsample { .. } => TransformType::Stochastic,
            AugmentationTransform::Composite { transforms } => {
                // Composite is stochastic if any component is stochastic
                if transforms
                    .iter()
                    .any(|t| t.transform_type() == TransformType::Stochastic)
                {
                    TransformType::Stochastic
                } else {
                    TransformType::Deterministic
                }
            }
        }
    }

    /// Check if this transform is reproducible (has a seed for stochastic transforms).
    pub fn is_reproducible(&self) -> bool {
        match self {
            AugmentationTransform::ReverseComplement => true,
            AugmentationTransform::Mutation { seed, .. } => seed.is_some(),
            AugmentationTransform::Subsample { seed, .. } => seed.is_some(),
            AugmentationTransform::Composite { transforms } => {
                transforms.iter().all(|t| t.is_reproducible())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transform_type() {
        assert_eq!(
            AugmentationTransform::ReverseComplement.transform_type(),
            TransformType::Deterministic
        );
        assert_eq!(
            AugmentationTransform::Mutation {
                rate: 0.05,
                seed: None
            }
            .transform_type(),
            TransformType::Stochastic
        );
    }

    #[test]
    fn test_is_reproducible() {
        assert!(AugmentationTransform::ReverseComplement.is_reproducible());
        assert!(AugmentationTransform::Mutation {
            rate: 0.05,
            seed: Some(42)
        }
        .is_reproducible());
        assert!(!AugmentationTransform::Mutation {
            rate: 0.05,
            seed: None
        }
        .is_reproducible());
    }

    #[test]
    fn test_composite_transform_type() {
        let composite = AugmentationTransform::Composite {
            transforms: vec![
                AugmentationTransform::ReverseComplement,
                AugmentationTransform::Mutation {
                    rate: 0.05,
                    seed: Some(42),
                },
            ],
        };
        assert_eq!(composite.transform_type(), TransformType::Stochastic);
    }
}
