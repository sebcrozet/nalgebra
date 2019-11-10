#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};
use num::Zero;

use alga::general::ComplexField;
use crate::allocator::{Allocator, Reallocator};
use crate::base::{DefaultAllocator, Matrix, MatrixMN, MatrixN, Unit, VectorN};
use crate::constraint::{SameNumberOfRows, ShapeConstraint};
use crate::{Dim, DimMin, DimMinimum, U1};
use crate::storage::{Storage, StorageMut};

use crate::geometry::Reflection;
use crate::linalg::householder;

/// The QR decomposition of a general matrix.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(
        serialize = "DefaultAllocator: Allocator<N, R, C> +
                           Allocator<N, DimMinimum<R, C>>,
         MatrixMN<N, R, C>: Serialize,
         VectorN<N, DimMinimum<R, C>>: Serialize"
    ))
)]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(
        deserialize = "DefaultAllocator: Allocator<N, R, C> +
                           Allocator<N, DimMinimum<R, C>>,
         MatrixMN<N, R, C>: Deserialize<'de>,
         VectorN<N, DimMinimum<R, C>>: Deserialize<'de>"
    ))
)]
#[derive(Clone, Debug)]
pub struct QR<N: ComplexField, R: DimMin<C>, C: Dim>
where DefaultAllocator: Allocator<N, R, C> + Allocator<N, DimMinimum<R, C>>
{
    qr: MatrixMN<N, R, C>,
    diag: VectorN<N, DimMinimum<R, C>>,
}

impl<N: ComplexField, R: DimMin<C>, C: Dim> Copy for QR<N, R, C>
where
    DefaultAllocator: Allocator<N, R, C> + Allocator<N, DimMinimum<R, C>>,
    MatrixMN<N, R, C>: Copy,
    VectorN<N, DimMinimum<R, C>>: Copy,
{}

impl<N: ComplexField, R: DimMin<C>, C: Dim> QR<N, R, C>
where DefaultAllocator: Allocator<N, R, C> + Allocator<N, R> + Allocator<N, DimMinimum<R, C>>
{
    /// Computes the QR decomposition using householder reflections.
    pub fn new(mut matrix: MatrixMN<N, R, C>) -> Self {
        let (nrows, ncols) = matrix.data.shape();
        let min_nrows_ncols = nrows.min(ncols);

        let mut diag = unsafe { MatrixMN::new_uninitialized_generic(min_nrows_ncols, U1) };

        if min_nrows_ncols.value() == 0 {
            return QR {
                qr: matrix,
                diag: diag,
            };
        }

        for ite in 0..min_nrows_ncols.value() {
            householder::clear_column_unchecked(&mut matrix, &mut diag[ite], ite, 0, None);
        }

        QR {
            qr: matrix,
            diag: diag,
        }
    }

    /// Retrieves the upper trapezoidal submatrix `R` of this decomposition.
    #[inline]
    pub fn r(&self) -> MatrixMN<N, DimMinimum<R, C>, C>
    where
        DefaultAllocator: Allocator<N, DimMinimum<R, C>, C>,
    {
        let (nrows, ncols) = self.qr.data.shape();
        let mut res = self.qr.rows_generic(0, nrows.min(ncols)).upper_triangle();
        res.set_partial_diagonal(self.diag.iter().map(|e| N::from_real(e.modulus())));
        res
    }

    /// Retrieves the upper trapezoidal submatrix `R` of this decomposition.
    ///
    /// This is usually faster than `r` but consumes `self`.
    #[inline]
    pub fn unpack_r(self) -> MatrixMN<N, DimMinimum<R, C>, C>
    where
        DefaultAllocator: Reallocator<N, R, C, DimMinimum<R, C>, C>,
    {
        let (nrows, ncols) = self.qr.data.shape();
        let mut res = self.qr.resize_generic(nrows.min(ncols), ncols, N::zero());
        res.fill_lower_triangle(N::zero(), 1);
        res.set_partial_diagonal(self.diag.iter().map(|e| N::from_real(e.modulus())));
        res
    }

    /// Computes the orthogonal matrix `Q` of this decomposition.
    pub fn q(&self) -> MatrixMN<N, R, DimMinimum<R, C>>
    where DefaultAllocator: Allocator<N, R, DimMinimum<R, C>> {
        let (nrows, ncols) = self.qr.data.shape();

        // NOTE: we could build the identity matrix and call q_mul on it.
        // Instead we don't so that we take in account the matrix sparseness.
        let mut res = Matrix::identity_generic(nrows, nrows.min(ncols));
        let dim = self.diag.len();

        for i in (0..dim).rev() {
            let axis = self.qr.slice_range(i.., i);
            // FIXME: sometimes, the axis might have a zero magnitude.
            let refl = Reflection::new(Unit::new_unchecked(axis), N::zero());

            let mut res_rows = res.slice_range_mut(i.., i..);
            refl.reflect_with_sign(&mut res_rows, self.diag[i].signum());
        }

        res
    }

    /// Unpacks this decomposition into its two matrix factors.
    pub fn unpack(
        self,
    ) -> (
        MatrixMN<N, R, DimMinimum<R, C>>,
        MatrixMN<N, DimMinimum<R, C>, C>,
    )
    where
        DimMinimum<R, C>: DimMin<C, Output = DimMinimum<R, C>>,
        DefaultAllocator:
            Allocator<N, R, DimMinimum<R, C>> + Reallocator<N, R, C, DimMinimum<R, C>, C>,
    {
        (self.q(), self.unpack_r())
    }

    #[doc(hidden)]
    pub fn qr_internal(&self) -> &MatrixMN<N, R, C> {
        &self.qr
    }

    /// Multiplies the provided matrix by the transpose of the `Q` matrix of this decomposition.
    pub fn q_tr_mul<R2: Dim, C2: Dim, S2>(&self, rhs: &mut Matrix<N, R2, C2, S2>)
    // FIXME: do we need a static constraint on the number of rows of rhs?
    where S2: StorageMut<N, R2, C2> {
        let dim = self.diag.len();

        for i in 0..dim {
            let axis = self.qr.slice_range(i.., i);
            let refl = Reflection::new(Unit::new_unchecked(axis), N::zero());

            let mut rhs_rows = rhs.rows_range_mut(i..);
            refl.reflect_with_sign(&mut rhs_rows, self.diag[i].signum().conjugate());
        }
    }
}

impl<N: ComplexField, D: DimMin<D, Output = D>> QR<N, D, D>
where DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
{
    /// Solves the linear system `self * x = b`, where `x` is the unknown to be determined.
    ///
    /// Returns `None` if `self` is not invertible.
    pub fn solve<R2: Dim, C2: Dim, S2>(
        &self,
        b: &Matrix<N, R2, C2, S2>,
    ) -> Option<MatrixMN<N, R2, C2>>
    where
        S2: Storage<N, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
        DefaultAllocator: Allocator<N, R2, C2>,
    {
        let mut res = b.clone_owned();

        if self.solve_mut(&mut res) {
            Some(res)
        } else {
            None
        }
    }

    /// Solves the linear system `self * x = b`, where `x` is the unknown to be determined.
    ///
    /// If the decomposed matrix is not invertible, this returns `false` and its input `b` is
    /// overwritten with garbage.
    pub fn solve_mut<R2: Dim, C2: Dim, S2>(&self, b: &mut Matrix<N, R2, C2, S2>) -> bool
    where
        S2: StorageMut<N, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        assert_eq!(
            self.qr.nrows(),
            b.nrows(),
            "QR solve matrix dimension mismatch."
        );
        assert!(
            self.qr.is_square(),
            "QR solve: unable to solve a non-square system."
        );

        self.q_tr_mul(b);
        self.solve_upper_triangular_mut(b)
    }

    // FIXME: duplicate code from the `solve` module.
    fn solve_upper_triangular_mut<R2: Dim, C2: Dim, S2>(
        &self,
        b: &mut Matrix<N, R2, C2, S2>,
    ) -> bool
    where
        S2: StorageMut<N, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let dim = self.qr.nrows();

        for k in 0..b.ncols() {
            let mut b = b.column_mut(k);
            for i in (0..dim).rev() {
                let coeff;

                unsafe {
                    let diag = self.diag.vget_unchecked(i).modulus();

                    if diag.is_zero() {
                        return false;
                    }

                    coeff = b.vget_unchecked(i).unscale(diag);
                    *b.vget_unchecked_mut(i) = coeff;
                }

                b.rows_range_mut(..i)
                    .axpy(-coeff, &self.qr.slice_range(..i, i), N::one());
            }
        }

        true
    }

    /// Computes the inverse of the decomposed matrix.
    ///
    /// Returns `None` if the decomposed matrix is not invertible.
    pub fn try_inverse(&self) -> Option<MatrixN<N, D>> {
        assert!(
            self.qr.is_square(),
            "QR inverse: unable to compute the inverse of a non-square matrix."
        );

        // FIXME: is there a less naive method ?
        let (nrows, ncols) = self.qr.data.shape();
        let mut res = MatrixN::identity_generic(nrows, ncols);

        if self.solve_mut(&mut res) {
            Some(res)
        } else {
            None
        }
    }

    /// Indicates if the decomposed matrix is invertible.
    pub fn is_invertible(&self) -> bool {
        assert!(
            self.qr.is_square(),
            "QR: unable to test the invertibility of a non-square matrix."
        );

        for i in 0..self.diag.len() {
            if self.diag[i].is_zero() {
                return false;
            }
        }

        true
    }

    // /// Computes the determinant of the decomposed matrix.
    // pub fn determinant(&self) -> N {
    //     let dim = self.qr.nrows();
    //     assert!(self.qr.is_square(), "QR determinant: unable to compute the determinant of a non-square matrix.");

    //     let mut res = N::one();
    //     for i in 0 .. dim {
    //         res *= unsafe { *self.diag.vget_unchecked(i) };
    //     }

    //     res self.q_determinant()
    // }
}

impl<N: ComplexField, R: DimMin<C>, C: Dim, S: Storage<N, R, C>> Matrix<N, R, C, S>
where DefaultAllocator: Allocator<N, R, C> + Allocator<N, R> + Allocator<N, DimMinimum<R, C>>
{
    /// Computes the QR decomposition of this matrix.
    pub fn qr(self) -> QR<N, R, C> {
        QR::new(self.into_owned())
    }
}
