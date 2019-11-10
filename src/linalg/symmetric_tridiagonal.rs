#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use alga::general::ComplexField;
use crate::allocator::Allocator;
use crate::base::{DefaultAllocator, MatrixMN, MatrixN, SquareMatrix, VectorN};
use crate::{DimDiff, DimSub, U1};
use crate::storage::Storage;

use crate::linalg::householder;

/// Tridiagonalization of a symmetric matrix.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(
        serialize = "DefaultAllocator: Allocator<N, D, D> +
                           Allocator<N, DimDiff<D, U1>>,
         MatrixN<N, D>: Serialize,
         VectorN<N, DimDiff<D, U1>>: Serialize"
    ))
)]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(
        deserialize = "DefaultAllocator: Allocator<N, D, D> +
                           Allocator<N, DimDiff<D, U1>>,
         MatrixN<N, D>: Deserialize<'de>,
         VectorN<N, DimDiff<D, U1>>: Deserialize<'de>"
    ))
)]
#[derive(Clone, Debug)]
pub struct SymmetricTridiagonal<N: ComplexField, D: DimSub<U1>>
where DefaultAllocator: Allocator<N, D, D> + Allocator<N, DimDiff<D, U1>>
{
    tri: MatrixN<N, D>,
    off_diagonal: VectorN<N, DimDiff<D, U1>>,
}

impl<N: ComplexField, D: DimSub<U1>> Copy for SymmetricTridiagonal<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, DimDiff<D, U1>>,
    MatrixN<N, D>: Copy,
    VectorN<N, DimDiff<D, U1>>: Copy,
{}

impl<N: ComplexField, D: DimSub<U1>> SymmetricTridiagonal<N, D>
where DefaultAllocator: Allocator<N, D, D> + Allocator<N, DimDiff<D, U1>>
{
    /// Computes the tridiagonalization of the symmetric matrix `m`.
    ///
    /// Only the lower-triangular part (including the diagonal) of `m` is read.
    pub fn new(mut m: MatrixN<N, D>) -> Self {
        let dim = m.data.shape().0;

        assert!(
            m.is_square(),
            "Unable to compute the symmetric tridiagonal decomposition of a non-square matrix."
        );
        assert!(
            dim.value() != 0,
            "Unable to compute the symmetric tridiagonal decomposition of an empty matrix."
        );

        let mut off_diagonal = unsafe { MatrixMN::new_uninitialized_generic(dim.sub(U1), U1) };
        let mut p = unsafe { MatrixMN::new_uninitialized_generic(dim.sub(U1), U1) };

        for i in 0..dim.value() - 1 {
            let mut m = m.rows_range_mut(i + 1..);
            let (mut axis, mut m) = m.columns_range_pair_mut(i, i + 1..);

            let (norm, not_zero) = householder::reflection_axis_mut(&mut axis);
            off_diagonal[i] = norm;

            if not_zero {
                let mut p = p.rows_range_mut(i..);

                p.hegemv(crate::convert(2.0), &m, &axis, N::zero());

                let dot = axis.dotc(&p);
                m.hegerc(-N::one(), &p, &axis, N::one());
                m.hegerc(-N::one(), &axis, &p, N::one());
                m.hegerc(dot * crate::convert(2.0), &axis, &axis, N::one());
            }
        }

        Self {
            tri: m,
            off_diagonal,
        }
    }

    #[doc(hidden)]
    // For debugging.
    pub fn internal_tri(&self) -> &MatrixN<N, D> {
        &self.tri
    }

    /// Retrieve the orthogonal transformation, diagonal, and off diagonal elements of this
    /// decomposition.
    pub fn unpack(self) -> (MatrixN<N, D>, VectorN<N::RealField, D>, VectorN<N::RealField, DimDiff<D, U1>>)
    where DefaultAllocator: Allocator<N::RealField, D>
                          + Allocator<N::RealField, DimDiff<D, U1>> {
        let diag = self.diagonal();
        let q = self.q();

        (q, diag, self.off_diagonal.map(N::modulus))
    }

    /// Retrieve the diagonal, and off diagonal elements of this decomposition.
    pub fn unpack_tridiagonal(self) -> (VectorN<N::RealField, D>, VectorN<N::RealField, DimDiff<D, U1>>)
        where DefaultAllocator: Allocator<N::RealField, D>
                              + Allocator<N::RealField, DimDiff<D, U1>> {
        (self.diagonal(), self.off_diagonal.map(N::modulus))
    }

    /// The diagonal components of this decomposition.
    pub fn diagonal(&self) -> VectorN<N::RealField, D>
    where DefaultAllocator: Allocator<N::RealField, D> { self.tri.map_diagonal(|e| e.real()) }

    /// The off-diagonal components of this decomposition.
    pub fn off_diagonal(&self) -> VectorN<N::RealField, DimDiff<D, U1>>
    where DefaultAllocator: Allocator<N::RealField, DimDiff<D, U1>> {
        self.off_diagonal.map(N::modulus)
    }

    /// Computes the orthogonal matrix `Q` of this decomposition.
    pub fn q(&self) -> MatrixN<N, D> {
        householder::assemble_q(&self.tri, self.off_diagonal.as_slice())
    }

    /// Recomputes the original symmetric matrix.
    pub fn recompose(mut self) -> MatrixN<N, D> {
        let q = self.q();
        self.tri.fill_lower_triangle(N::zero(), 2);
        self.tri.fill_upper_triangle(N::zero(), 2);

        for i in 0..self.off_diagonal.len() {
            let val = N::from_real(self.off_diagonal[i].modulus());
            self.tri[(i + 1, i)] = val;
            self.tri[(i, i + 1)] = val;
        }

        &q * self.tri * q.adjoint()
    }
}

impl<N: ComplexField, D: DimSub<U1>, S: Storage<N, D, D>> SquareMatrix<N, D, S>
where DefaultAllocator: Allocator<N, D, D> + Allocator<N, DimDiff<D, U1>>
{
    /// Computes the tridiagonalization of this symmetric matrix.
    ///
    /// Only the lower-triangular part (including the diagonal) of `m` is read.
    pub fn symmetric_tridiagonalize(self) -> SymmetricTridiagonal<N, D> {
        SymmetricTridiagonal::new(self.into_owned())
    }
}
