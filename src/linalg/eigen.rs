#[cfg(feature = "serde-serialize")]
use serde::{Serialize, Deserialize};

use alga::general::ComplexField;
use num_complex::Complex;
use std::cmp;
use std::fmt::Display;
use std::ops::MulAssign;

use crate::allocator::Allocator;
use crate::{Dim, DimDiff, DimSub, Dynamic, U1, U2, U3};
use crate::base::storage::Storage;
use crate::base::{DefaultAllocator, Hessenberg, MatrixN, SquareMatrix, Unit, Vector2, Vector3, VectorN};
use crate::constraint::{DimEq, ShapeConstraint};

use crate::geometry::{Reflection, UnitComplex};
use crate::linalg::householder;
use crate::linalg::Schur;

/// Eigendecomposition of a real matrix with real eigenvalues (or complex eigen values for complex matrices).
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize",
    serde(
        bound(
            serialize = "DefaultAllocator: Allocator<N, D>,
         VectorN<N, D>: Serialize,
         MatrixN<N, D>: Serialize"
        )
    )
)]
#[cfg_attr(
    feature = "serde-serialize",
    serde(
        bound(
            deserialize = "DefaultAllocator: Allocator<N, D>,
         VectorN<N, D>: Serialize,
         MatrixN<N, D>: Deserialize<'de>"
        )
    )
)]
#[derive(Clone, Debug)]
pub struct Eigen<N: ComplexField, D: Dim>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    pub eigenvectors: MatrixN<N, D>,
    pub eigenvalues: VectorN<N, D>,
}

impl<N: ComplexField, D: Dim> Copy for Eigen<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
    MatrixN<N, D>: Copy,
    VectorN<N, D>: Copy,
{
}

impl<N: ComplexField, D: Dim> Eigen<N, D>
where
    D: DimSub<U1>,                                   // For Hessenberg.
    ShapeConstraint: DimEq<Dynamic, DimDiff<D, U1>>, // For Hessenberg.
    DefaultAllocator: Allocator<N, D, DimDiff<D, U1>>
        + Allocator<N, DimDiff<D, U1>>
        + Allocator<N, D, D>
        + Allocator<N, D>,
    // XXX: for debug
    DefaultAllocator: Allocator<usize, D, D>,
    MatrixN<N, D>: Display,
{
    /// Computes the eigendecomposition of a diagonalizable matrix with Complex eigenvalues.
    pub fn new(m: MatrixN<N, D>) -> Option<Eigen<N, D>> {
        assert!(
            m.is_square(),
            "Unable to compute the eigendecomposition of a non-square matrix."
        );

        let dim = m.nrows();
        let (mut eigenvectors, mut eigenvalues) = Schur::new(m, 0).unwrap().unpack();

        println!("Schur eigenvalues: {}", eigenvalues);

        // Check that the eigenvalues are all Complex.
        for i in 0..dim - 1 {
            if !eigenvalues[(i + 1, i)].is_zero() {
                return None;
            }
        }

        for j in 1..dim {
            for i in 0..j {
                let diff = eigenvalues[(i, i)] - eigenvalues[(j, j)];

                if diff.is_zero() && !eigenvalues[(i, j)].is_zero() {
                    return None;
                }

                let z = -eigenvalues[(i, j)] / diff;

                for k in j + 1..dim {
                    eigenvalues[(i, k)] -= z * eigenvalues[(j, k)];
                }

                for k in 0..dim {
                    eigenvectors[(k, j)] += z * eigenvectors[(k, i)];
                }
            }
        }

        // Normalize the eigenvector basis.
        for i in 0..dim {
            let _ = eigenvectors.column_mut(i).normalize_mut();
        }

        Some(Eigen {
            eigenvectors,
            eigenvalues: eigenvalues.diagonal(),
        })
    }
}
