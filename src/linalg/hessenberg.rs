#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use alga::general::ComplexField;
use crate::allocator::Allocator;
use crate::base::{DefaultAllocator, MatrixMN, MatrixN, SquareMatrix, VectorN};
use crate::{DimDiff, DimSub, U1};
use crate::storage::Storage;

use crate::linalg::householder;

/// Hessenberg decomposition of a general matrix.
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
pub struct Hessenberg<N: ComplexField, D: DimSub<U1>>
where DefaultAllocator: Allocator<N, D, D> + Allocator<N, DimDiff<D, U1>>
{
    hess: MatrixN<N, D>,
    subdiag: VectorN<N, DimDiff<D, U1>>,
}

impl<N: ComplexField, D: DimSub<U1>> Copy for Hessenberg<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, DimDiff<D, U1>>,
    MatrixN<N, D>: Copy,
    VectorN<N, DimDiff<D, U1>>: Copy,
{}

impl<N: ComplexField, D: DimSub<U1>> Hessenberg<N, D>
where DefaultAllocator: Allocator<N, D, D> + Allocator<N, D> + Allocator<N, DimDiff<D, U1>>
{
    /// Computes the Hessenberg decomposition using householder reflections.
    pub fn new(hess: MatrixN<N, D>) -> Self {
        let mut work = unsafe { MatrixMN::new_uninitialized_generic(hess.data.shape().0, U1) };
        Self::new_with_workspace(hess, &mut work)
    }

    /// Computes the Hessenberg decomposition using householder reflections.
    ///
    /// The workspace containing `D` elements must be provided but its content does not have to be
    /// initialized.
    pub fn new_with_workspace(mut hess: MatrixN<N, D>, work: &mut VectorN<N, D>) -> Self {
        assert!(
            hess.is_square(),
            "Cannot compute the hessenberg decomposition of a non-square matrix."
        );

        let dim = hess.data.shape().0;

        assert!(
            dim.value() != 0,
            "Cannot compute the hessenberg decomposition of an empty matrix."
        );
        assert_eq!(
            dim.value(),
            work.len(),
            "Hessenberg: invalid workspace size."
        );

        let mut subdiag = unsafe { MatrixMN::new_uninitialized_generic(dim.sub(U1), U1) };

        if dim.value() == 0 {
            return Hessenberg { hess, subdiag };
        }

        for ite in 0..dim.value() - 1 {
            householder::clear_column_unchecked(&mut hess, &mut subdiag[ite], ite, 1, Some(work));
        }

        Hessenberg { hess, subdiag }
    }

    /// Retrieves `(q, h)` with `q` the orthogonal matrix of this decomposition and `h` the
    /// hessenberg matrix.
    #[inline]
    pub fn unpack(self) -> (MatrixN<N, D>, MatrixN<N, D>) {
        let q = self.q();

        (q, self.unpack_h())
    }

    /// Retrieves the upper trapezoidal submatrix `H` of this decomposition.
    #[inline]
    pub fn unpack_h(mut self) -> MatrixN<N, D> {
        let dim = self.hess.nrows();
        self.hess.fill_lower_triangle(N::zero(), 2);
        self.hess
            .slice_mut((1, 0), (dim - 1, dim - 1))
            .set_partial_diagonal(self.subdiag.iter().map(|e| N::from_real(e.modulus())));
        self.hess
    }

    // FIXME: add a h that moves out of self.
    /// Retrieves the upper trapezoidal submatrix `H` of this decomposition.
    ///
    /// This is less efficient than `.unpack_h()` as it allocates a new matrix.
    #[inline]
    pub fn h(&self) -> MatrixN<N, D> {
        let dim = self.hess.nrows();
        let mut res = self.hess.clone();
        res.fill_lower_triangle(N::zero(), 2);
        res.slice_mut((1, 0), (dim - 1, dim - 1))
            .set_partial_diagonal(self.subdiag.iter().map(|e| N::from_real(e.modulus())));
        res
    }

    /// Computes the orthogonal matrix `Q` of this decomposition.
    pub fn q(&self) -> MatrixN<N, D> {
        householder::assemble_q(&self.hess, self.subdiag.as_slice())
    }

    #[doc(hidden)]
    pub fn hess_internal(&self) -> &MatrixN<N, D> {
        &self.hess
    }
}

impl<N: ComplexField, D: DimSub<U1>, S: Storage<N, D, D>> SquareMatrix<N, D, S>
where DefaultAllocator: Allocator<N, D, D> + Allocator<N, D> + Allocator<N, DimDiff<D, U1>>
{
    /// Computes the Hessenberg decomposition of this matrix using householder reflections.
    pub fn hessenberg(self) -> Hessenberg<N, D> {
        Hessenberg::new(self.into_owned())
    }
}
