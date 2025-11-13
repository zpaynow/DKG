use ark_ed_on_bn254::Fr;
use ark_ff::Field;
use ark_std::UniformRand;
use rand::Rng;
use std::ops::{Add, Mul};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DkgError {
    #[error("Threshold must be at least 1")]
    InvalidThreshold,
    #[error("Number of shares must be at least the threshold")]
    InvalidShareCount,
    #[error("Not enough shares to reconstruct secret (need at least {threshold}, got {actual})")]
    InsufficientShares { threshold: usize, actual: usize },
    #[error("Duplicate share indices detected")]
    DuplicateIndices,
    #[error("Share index cannot be zero")]
    ZeroIndex,
}

/// Represents a share in Shamir's Secret Sharing scheme
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Share {
    /// The x-coordinate (participant index, must be non-zero)
    pub index: u32,
    /// The y-coordinate (share value)
    pub value: Fr,
}

/// Polynomial used for Shamir's Secret Sharing
#[derive(Debug, Clone)]
pub struct Polynomial {
    /// Coefficients of the polynomial, where coefficients[0] is the constant term (the secret)
    coefficients: Vec<Fr>,
}

impl Polynomial {
    /// Generate a random polynomial of degree (threshold - 1) with a given secret as constant term
    pub fn random<R: Rng>(secret: Fr, threshold: usize, rng: &mut R) -> Self {
        let mut coefficients = vec![secret];

        // Generate (threshold - 1) random coefficients
        for _ in 1..threshold {
            coefficients.push(Fr::rand(rng));
        }

        Polynomial { coefficients }
    }

    /// Evaluate the polynomial at a given x value
    pub fn evaluate(&self, x: Fr) -> Fr {
        // Use Horner's method for efficient polynomial evaluation
        let mut result = self.coefficients[self.coefficients.len() - 1];

        for i in (0..self.coefficients.len() - 1).rev() {
            result = result.mul(x).add(self.coefficients[i]);
        }

        result
    }

    /// Get the secret (constant term)
    pub fn secret(&self) -> Fr {
        self.coefficients[0]
    }

    /// Get the degree of the polynomial
    pub fn degree(&self) -> usize {
        self.coefficients.len() - 1
    }
}

/// Distributed Key Generation using Shamir's Secret Sharing
pub struct Dkg {
    /// Threshold (minimum number of shares needed to reconstruct)
    threshold: usize,
    /// Total number of shares to generate
    num_shares: usize,
}

impl Dkg {
    /// Create a new DKG instance
    ///
    /// # Arguments
    /// * `threshold` - Minimum number of shares needed to reconstruct the secret (m)
    /// * `num_shares` - Total number of shares to generate (n)
    pub fn new(threshold: usize, num_shares: usize) -> Result<Self, DkgError> {
        if threshold == 0 {
            return Err(DkgError::InvalidThreshold);
        }
        if num_shares < threshold {
            return Err(DkgError::InvalidShareCount);
        }

        Ok(Dkg {
            threshold,
            num_shares,
        })
    }

    /// Generate a random secret and distribute shares
    pub fn generate_shares<R: Rng>(&self, rng: &mut R) -> (Fr, Vec<Share>) {
        let secret = Fr::rand(rng);
        let shares = self.share_secret(secret, rng);
        (secret, shares)
    }

    /// Generate shares from a given secret
    pub fn share_secret<R: Rng>(&self, secret: Fr, rng: &mut R) -> Vec<Share> {
        let polynomial = Polynomial::random(secret, self.threshold, rng);

        let mut shares = Vec::with_capacity(self.num_shares);
        for i in 1..=self.num_shares {
            let x = Fr::from(i as u32);
            let y = polynomial.evaluate(x);
            shares.push(Share {
                index: i as u32,
                value: y,
            });
        }

        shares
    }

    /// Reconstruct the secret from shares using Lagrange interpolation
    pub fn reconstruct_secret(&self, shares: &[Share]) -> Result<Fr, DkgError> {
        if shares.len() < self.threshold {
            return Err(DkgError::InsufficientShares {
                threshold: self.threshold,
                actual: shares.len(),
            });
        }

        // Check for zero indices
        for share in shares {
            if share.index == 0 {
                return Err(DkgError::ZeroIndex);
            }
        }

        // Check for duplicate indices
        let mut indices = shares.iter().map(|s| s.index).collect::<Vec<_>>();
        indices.sort();
        for i in 1..indices.len() {
            if indices[i] == indices[i - 1] {
                return Err(DkgError::DuplicateIndices);
            }
        }

        // Use only the first 'threshold' shares
        let shares_to_use = &shares[..self.threshold];

        // Lagrange interpolation at x = 0 to get the constant term (the secret)
        let mut secret = Fr::from(0u32);

        for i in 0..shares_to_use.len() {
            let x_i = Fr::from(shares_to_use[i].index);
            let y_i = shares_to_use[i].value;

            // Calculate Lagrange basis polynomial L_i(0)
            let mut numerator = Fr::from(1u32);
            let mut denominator = Fr::from(1u32);

            for j in 0..shares_to_use.len() {
                if i != j {
                    let x_j = Fr::from(shares_to_use[j].index);
                    // L_i(0) = prod_{j!=i} (0 - x_j) / (x_i - x_j)
                    //        = prod_{j!=i} (-x_j) / (x_i - x_j)
                    numerator = numerator.mul(-x_j);
                    denominator = denominator.mul(x_i - x_j);
                }
            }

            let lagrange_basis = numerator.mul(denominator.inverse().unwrap());
            secret = secret.add(y_i.mul(lagrange_basis));
        }

        Ok(secret)
    }

    /// Get the threshold
    pub fn threshold(&self) -> usize {
        self.threshold
    }

    /// Get the total number of shares
    pub fn num_shares(&self) -> usize {
        self.num_shares
    }
}

/// Convenience function to generate a secret from seeds
/// This can be used for deterministic generation in the future
pub fn generate_from_seeds() -> Fr {
    let mut rng = rand::thread_rng();
    Fr::rand(&mut rng)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_std::test_rng;

    #[test]
    fn test_polynomial_evaluation() {
        let mut rng = test_rng();
        let secret = Fr::from(42u32);
        let poly = Polynomial::random(secret, 3, &mut rng);

        assert_eq!(poly.secret(), secret);
        assert_eq!(poly.degree(), 2);

        // Evaluate at x=0 should give the secret
        assert_eq!(poly.evaluate(Fr::from(0u32)), secret);
    }

    #[test]
    fn test_basic_secret_sharing() {
        let mut rng = test_rng();
        let dkg = Dkg::new(3, 5).unwrap();

        let secret = Fr::from(12345u32);
        let shares = dkg.share_secret(secret, &mut rng);

        assert_eq!(shares.len(), 5);

        // Reconstruct with exactly threshold shares
        let reconstructed = dkg.reconstruct_secret(&shares[..3]).unwrap();
        assert_eq!(reconstructed, secret);
    }

    #[test]
    fn test_reconstruction_with_different_shares() {
        let mut rng = test_rng();
        let dkg = Dkg::new(3, 5).unwrap();

        let secret = Fr::from(98765u32);
        let shares = dkg.share_secret(secret, &mut rng);

        // Try different combinations of threshold shares
        let reconstructed1 = dkg.reconstruct_secret(&shares[0..3]).unwrap();
        let reconstructed2 = dkg.reconstruct_secret(&shares[1..4]).unwrap();
        let reconstructed3 = dkg.reconstruct_secret(&shares[2..5]).unwrap();

        assert_eq!(reconstructed1, secret);
        assert_eq!(reconstructed2, secret);
        assert_eq!(reconstructed3, secret);
    }

    #[test]
    fn test_reconstruction_with_more_than_threshold() {
        let mut rng = test_rng();
        let dkg = Dkg::new(2, 5).unwrap();

        let secret = Fr::from(555u32);
        let shares = dkg.share_secret(secret, &mut rng);

        // Reconstruct with more than threshold shares
        let reconstructed = dkg.reconstruct_secret(&shares).unwrap();
        assert_eq!(reconstructed, secret);
    }

    #[test]
    fn test_insufficient_shares() {
        let mut rng = test_rng();
        let dkg = Dkg::new(3, 5).unwrap();

        let secret = Fr::from(777u32);
        let shares = dkg.share_secret(secret, &mut rng);

        // Try to reconstruct with fewer than threshold shares
        let result = dkg.reconstruct_secret(&shares[..2]);
        assert!(matches!(result, Err(DkgError::InsufficientShares { .. })));
    }

    #[test]
    fn test_invalid_threshold() {
        let result = Dkg::new(0, 5);
        assert!(matches!(result, Err(DkgError::InvalidThreshold)));
    }

    #[test]
    fn test_invalid_share_count() {
        let result = Dkg::new(5, 3);
        assert!(matches!(result, Err(DkgError::InvalidShareCount)));
    }

    #[test]
    fn test_generate_shares() {
        let mut rng = test_rng();
        let dkg = Dkg::new(3, 5).unwrap();

        let (secret, shares) = dkg.generate_shares(&mut rng);

        assert_eq!(shares.len(), 5);

        let reconstructed = dkg.reconstruct_secret(&shares[..3]).unwrap();
        assert_eq!(reconstructed, secret);
    }

    #[test]
    fn test_share_indices() {
        let mut rng = test_rng();
        let dkg = Dkg::new(2, 4).unwrap();

        let shares = dkg.share_secret(Fr::from(100u32), &mut rng);

        // Check that indices are 1, 2, 3, 4
        for (i, share) in shares.iter().enumerate() {
            assert_eq!(share.index, (i + 1) as u32);
        }
    }

    #[test]
    fn test_large_threshold() {
        let mut rng = test_rng();
        let dkg = Dkg::new(10, 15).unwrap();

        let secret = Fr::from(123456789u32);
        let shares = dkg.share_secret(secret, &mut rng);

        let reconstructed = dkg.reconstruct_secret(&shares[..10]).unwrap();
        assert_eq!(reconstructed, secret);
    }

    #[test]
    fn test_random_secret_generation() {
        let mut rng = test_rng();
        let dkg = Dkg::new(3, 5).unwrap();

        let (secret1, shares1) = dkg.generate_shares(&mut rng);
        let (secret2, shares2) = dkg.generate_shares(&mut rng);

        // Secrets should be different (with overwhelming probability)
        assert_ne!(secret1, secret2);

        // Each set of shares should reconstruct to its own secret
        assert_eq!(dkg.reconstruct_secret(&shares1[..3]).unwrap(), secret1);
        assert_eq!(dkg.reconstruct_secret(&shares2[..3]).unwrap(), secret2);
    }

    #[test]
    fn test_edge_case_threshold_equals_shares() {
        let mut rng = test_rng();
        let dkg = Dkg::new(5, 5).unwrap();

        let secret = Fr::from(999u32);
        let shares = dkg.share_secret(secret, &mut rng);

        let reconstructed = dkg.reconstruct_secret(&shares).unwrap();
        assert_eq!(reconstructed, secret);
    }

    #[test]
    fn test_zero_index_error() {
        let dkg = Dkg::new(2, 3).unwrap();
        let shares = vec![
            Share {
                index: 0,
                value: Fr::from(1u32),
            },
            Share {
                index: 2,
                value: Fr::from(2u32),
            },
        ];

        let result = dkg.reconstruct_secret(&shares);
        assert!(matches!(result, Err(DkgError::ZeroIndex)));
    }

    #[test]
    fn test_duplicate_indices_error() {
        let dkg = Dkg::new(2, 3).unwrap();
        let shares = vec![
            Share {
                index: 1,
                value: Fr::from(1u32),
            },
            Share {
                index: 1,
                value: Fr::from(2u32),
            },
        ];

        let result = dkg.reconstruct_secret(&shares);
        assert!(matches!(result, Err(DkgError::DuplicateIndices)));
    }
}
