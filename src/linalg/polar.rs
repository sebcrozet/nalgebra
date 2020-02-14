#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use approx::AbsDiffEq;

use crate::{DMatrix};
use crate::linalg::SVD;
use alga::general::{ComplexField};

/// Polar Decomposition of a general matrix.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(
            serialize = "DMatrix<N>: Serialize,
                        DMatrix<N>: Serialize,
                        DMatrix<N>: Serialize"
    ))
)]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(
            deserialize = "DMatrix<N>: Deserialize<'de>,
                        DMatrix<N>: Deserialize<'de>,
                        DMatrix<N>: Deserialize<'de>"
    ))
)]
#[derive(Clone, Debug)]
pub struct Polar<N: ComplexField>
{
    /// The rotation matrix
    pub r: Option<DMatrix<N>>,
    /// The left hermitian matrix (A = PR)
    pub p_l: Option<DMatrix<N>>,
    /// The right hermitian matrix (A = RP)
    pub p_r: Option<DMatrix<N>>
}

impl<N: ComplexField> Polar<N>
where
{
    /// Computes the Polar Decomposition of the matrix using its SVD
    pub fn new(matrix: DMatrix<N>) -> Self {
        Self::try_new(
            matrix,
            N::RealField::default_epsilon(),
            0
        ).unwrap()
    }

    /// Attempts to compute the Polar Decomposition using the SVD of the matrix
    ///
    /// # Arguments
    ///
    /// * `eps`           − tolerance used to determine when a value converged to 0.
    /// * `max_niter`     − maximum total number of iterations performed by the algorithm. If this
    /// number of iteration is exceeded, `None` is returned. If `niter == 0`, then the algorithm
    /// continues indefinitely until convergence.
    pub fn try_new(
        matrix: DMatrix<N>,
        eps: N::RealField,
        max_niter: usize
    ) -> Option<Self> {

        let svd = SVD::try_new(matrix, true, true, eps, max_niter)?;

        let r: Option<DMatrix<N>> =
            if let (Some(u), Some(v_t)) = (&svd.u, &svd.v_t) {
                Some(u*v_t)
            } else {
                None
            };

        let sigma: DMatrix<N> = DMatrix::from_diagonal(&svd.singular_values.map(|e| N::from_real(e)));;
        let p_r = svd.v_t.as_ref().map(|v_t| v_t.adjoint() * &sigma * v_t);
        let p_l = svd.u.as_ref().map(|u| u * &sigma * u.adjoint());

        Some(Self {
            r,
            p_l,
            p_r,
        })
    } 

    /// Rebuild the original matrix usign the left decompositon (A=PR)
    ///
    /// This is useful if some of the values have been manually modified.
    /// Returns `Err` if the right- and left- singular vectors have not been
    /// computed at construction-time.
    pub fn recompose_left(self) -> Result<DMatrix<N>, &'static str> {
        match (&self.r, &self.p_l) {
            (Some(r), Some(p_l)) => {
                Ok(p_l * r)
            }
            (None, None) => Err("Polar recomposition: P and R have not been computed."),
            (None, _) => Err("Polar recomposition: P has not been computed."),
            (_, None) => Err("Polar recomposition: R has not been computed."),
        }
    }

    /// Rebuild the original matrix usign the right decompositon (A=RP)
    ///
    /// This is useful if some of the values have been manually modified.
    /// Returns `Err` if the right- and left- singular vectors have not been
    /// computed at construction-time.
    pub fn recompose_right(self) -> Result<DMatrix<N>, &'static str> {
        match (&self.r, &self.p_r) {
            (Some(r), Some(p_r)) => {
                Ok(r * p_r)
            }
            (None, None) => Err("Polar recomposition: P and R have not been computed."),
            (None, _) => Err("Polar recomposition: P has not been computed."),
            (_, None) => Err("Polar recomposition: R has not been computed."),
        }
    }
}

impl<N: ComplexField> DMatrix<N>
where
{
    /// Computes the Polar Decomposition of the matrix using its SVD
    pub fn polar(self) -> Polar<N> {
        Polar::new(self.into_owned())
    }

    /// Attempts to compute the Polar Decomposition using the SVD of the matrix
    ///
    /// # Arguments
    ///
    /// * `eps`       − tolerance used to determine when a value converged to 0.
    /// * `max_niter` − maximum total number of iterations performed by the algorithm. If this
    /// number of iteration is exceeded, `None` is returned. If `niter == 0`, then the algorithm
    /// continues indefinitely until convergence.
    pub fn try_polar(
        self,
        eps: N::RealField,
        max_niter: usize,
    ) -> Option<Polar<N>> {
        Polar::try_new(self.into_owned(), eps, max_niter)
    }
}
