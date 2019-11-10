#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use alga::general::ComplexField;
use crate::allocator::Allocator;
use crate::base::{DefaultAllocator, Matrix, MatrixMN, MatrixN, Unit, VectorN};
use crate::{Dim, DimDiff, DimMin, DimMinimum, DimSub, U1};
use crate::storage::Storage;

use crate::geometry::Reflection;
use crate::linalg::householder;

/// The bidiagonalization of a general matrix.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(
        serialize = "DimMinimum<R, C>: DimSub<U1>,
         DefaultAllocator: Allocator<N, R, C>             +
                           Allocator<N, DimMinimum<R, C>> +
                           Allocator<N, DimDiff<DimMinimum<R, C>, U1>>,
         MatrixMN<N, R, C>: Serialize,
         VectorN<N, DimMinimum<R, C>>: Serialize,
         VectorN<N, DimDiff<DimMinimum<R, C>, U1>>: Serialize"
    ))
)]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(
        deserialize = "DimMinimum<R, C>: DimSub<U1>,
         DefaultAllocator: Allocator<N, R, C>             +
                           Allocator<N, DimMinimum<R, C>> +
                           Allocator<N, DimDiff<DimMinimum<R, C>, U1>>,
         MatrixMN<N, R, C>: Deserialize<'de>,
         VectorN<N, DimMinimum<R, C>>: Deserialize<'de>,
         VectorN<N, DimDiff<DimMinimum<R, C>, U1>>: Deserialize<'de>"
    ))
)]
#[derive(Clone, Debug)]
pub struct Bidiagonal<N: ComplexField, R: DimMin<C>, C: Dim>
where
    DimMinimum<R, C>: DimSub<U1>,
    DefaultAllocator: Allocator<N, R, C>
        + Allocator<N, DimMinimum<R, C>>
        + Allocator<N, DimDiff<DimMinimum<R, C>, U1>>,
{
    // FIXME: perhaps we should pack the axises into different vectors so that axises for `v_t` are
    // contiguous. This prevents some useless copies.
    uv: MatrixMN<N, R, C>,
    /// The diagonal elements of the decomposed matrix.
    diagonal: VectorN<N, DimMinimum<R, C>>,
    /// The off-diagonal elements of the decomposed matrix.
    off_diagonal: VectorN<N, DimDiff<DimMinimum<R, C>, U1>>,
    upper_diagonal: bool,
}

impl<N: ComplexField, R: DimMin<C>, C: Dim> Copy for Bidiagonal<N, R, C>
where
    DimMinimum<R, C>: DimSub<U1>,
    DefaultAllocator: Allocator<N, R, C>
        + Allocator<N, DimMinimum<R, C>>
        + Allocator<N, DimDiff<DimMinimum<R, C>, U1>>,
    MatrixMN<N, R, C>: Copy,
    VectorN<N, DimMinimum<R, C>>: Copy,
    VectorN<N, DimDiff<DimMinimum<R, C>, U1>>: Copy,
{}

impl<N: ComplexField, R: DimMin<C>, C: Dim> Bidiagonal<N, R, C>
where
    DimMinimum<R, C>: DimSub<U1>,
    DefaultAllocator: Allocator<N, R, C>
        + Allocator<N, C>
        + Allocator<N, R>
        + Allocator<N, DimMinimum<R, C>>
        + Allocator<N, DimDiff<DimMinimum<R, C>, U1>>,
{
    /// Computes the Bidiagonal decomposition using householder reflections.
    pub fn new(mut matrix: MatrixMN<N, R, C>) -> Self {
        let (nrows, ncols) = matrix.data.shape();
        let min_nrows_ncols = nrows.min(ncols);
        let dim = min_nrows_ncols.value();
        assert!(
            dim != 0,
            "Cannot compute the bidiagonalization of an empty matrix."
        );

        let mut diagonal = unsafe { MatrixMN::new_uninitialized_generic(min_nrows_ncols, U1) };
        let mut off_diagonal =
            unsafe { MatrixMN::new_uninitialized_generic(min_nrows_ncols.sub(U1), U1) };
        let mut axis_packed = unsafe { MatrixMN::new_uninitialized_generic(ncols, U1) };
        let mut work = unsafe { MatrixMN::new_uninitialized_generic(nrows, U1) };

        let upper_diagonal = nrows.value() >= ncols.value();
        if upper_diagonal {
            for ite in 0..dim - 1 {
                householder::clear_column_unchecked(&mut matrix, &mut diagonal[ite], ite, 0, None);
                householder::clear_row_unchecked(
                    &mut matrix,
                    &mut off_diagonal[ite],
                    &mut axis_packed,
                    &mut work,
                    ite,
                    1,
                );
            }

            householder::clear_column_unchecked(
                &mut matrix,
                &mut diagonal[dim - 1],
                dim - 1,
                0,
                None,
            );
        } else {
            for ite in 0..dim - 1 {
                householder::clear_row_unchecked(
                    &mut matrix,
                    &mut diagonal[ite],
                    &mut axis_packed,
                    &mut work,
                    ite,
                    0,
                );
                householder::clear_column_unchecked(
                    &mut matrix,
                    &mut off_diagonal[ite],
                    ite,
                    1,
                    None,
                );
            }

            householder::clear_row_unchecked(
                &mut matrix,
                &mut diagonal[dim - 1],
                &mut axis_packed,
                &mut work,
                dim - 1,
                0,
            );
        }

        Bidiagonal {
            uv: matrix,
            diagonal,
            off_diagonal,
            upper_diagonal,
        }
    }

    /// Indicates whether this decomposition contains an upper-diagonal matrix.
    #[inline]
    pub fn is_upper_diagonal(&self) -> bool {
        self.upper_diagonal
    }

    #[inline]
    fn axis_shift(&self) -> (usize, usize) {
        if self.upper_diagonal {
            (0, 1)
        } else {
            (1, 0)
        }
    }

    /// Unpacks this decomposition into its three matrix factors `(U, D, V^t)`.
    ///
    /// The decomposed matrix `M` is equal to `U * D * V^t`.
    #[inline]
    pub fn unpack(
        self,
    ) -> (
        MatrixMN<N, R, DimMinimum<R, C>>,
        MatrixN<N, DimMinimum<R, C>>,
        MatrixMN<N, DimMinimum<R, C>, C>,
    )
    where
        DefaultAllocator: Allocator<N, DimMinimum<R, C>, DimMinimum<R, C>>
            + Allocator<N, R, DimMinimum<R, C>>
            + Allocator<N, DimMinimum<R, C>, C>,
    {
        // FIXME: optimize by calling a reallocator.
        (self.u(), self.d(), self.v_t())
    }

    /// Retrieves the upper trapezoidal submatrix `R` of this decomposition.
    #[inline]
    pub fn d(&self) -> MatrixN<N, DimMinimum<R, C>>
    where
        DefaultAllocator: Allocator<N, DimMinimum<R, C>, DimMinimum<R, C>>,
    {
        let (nrows, ncols) = self.uv.data.shape();

        let d = nrows.min(ncols);
        let mut res = MatrixN::identity_generic(d, d);
        res.set_partial_diagonal(self.diagonal.iter().map(|e| N::from_real(e.modulus())));

        let start = self.axis_shift();
        res.slice_mut(start, (d.value() - 1, d.value() - 1))
            .set_partial_diagonal(self.off_diagonal.iter().map(|e| N::from_real(e.modulus())));
        res
    }

    /// Computes the orthogonal matrix `U` of this `U * D * V` decomposition.
    // FIXME: code duplication with householder::assemble_q.
    // Except that we are returning a rectangular matrix here.
    pub fn u(&self) -> MatrixMN<N, R, DimMinimum<R, C>>
    where DefaultAllocator: Allocator<N, R, DimMinimum<R, C>> {
        let (nrows, ncols) = self.uv.data.shape();

        let mut res = Matrix::identity_generic(nrows, nrows.min(ncols));
        let dim = self.diagonal.len();
        let shift = self.axis_shift().0;

        for i in (0..dim - shift).rev() {
            let axis = self.uv.slice_range(i + shift.., i);
            // FIXME: sometimes, the axis might have a zero magnitude.
            let refl = Reflection::new(Unit::new_unchecked(axis), N::zero());

            let mut res_rows = res.slice_range_mut(i + shift.., i..);

            let sign = if self.upper_diagonal {
                self.diagonal[i].signum()
            } else {
                self.off_diagonal[i].signum()
            };

            refl.reflect_with_sign(&mut res_rows, sign);
        }

        res
    }

    /// Computes the orthogonal matrix `V_t` of this `U * D * V_t` decomposition.
    pub fn v_t(&self) -> MatrixMN<N, DimMinimum<R, C>, C>
    where DefaultAllocator: Allocator<N, DimMinimum<R, C>, C> {
        let (nrows, ncols) = self.uv.data.shape();
        let min_nrows_ncols = nrows.min(ncols);

        let mut res = Matrix::identity_generic(min_nrows_ncols, ncols);
        let mut work = unsafe { MatrixMN::new_uninitialized_generic(min_nrows_ncols, U1) };
        let mut axis_packed = unsafe { MatrixMN::new_uninitialized_generic(ncols, U1) };

        let shift = self.axis_shift().1;

        for i in (0..min_nrows_ncols.value() - shift).rev() {
            let axis = self.uv.slice_range(i, i + shift..);
            let mut axis_packed = axis_packed.rows_range_mut(i + shift..);
            axis_packed.tr_copy_from(&axis);
            // FIXME: sometimes, the axis might have a zero magnitude.
            let refl = Reflection::new(Unit::new_unchecked(axis_packed), N::zero());

            let mut res_rows = res.slice_range_mut(i.., i + shift..);

            let sign = if self.upper_diagonal {
                self.off_diagonal[i].signum()
            } else {
                self.diagonal[i].signum()
            };

            refl.reflect_rows_with_sign(&mut res_rows, &mut work.rows_range_mut(i..), sign);
        }

        res
    }

    /// The diagonal part of this decomposed matrix.
    pub fn diagonal(&self) -> VectorN<N::RealField, DimMinimum<R, C>>
        where DefaultAllocator: Allocator<N::RealField, DimMinimum<R, C>> {
        self.diagonal.map(|e| e.modulus())
    }

    /// The off-diagonal part of this decomposed matrix.
    pub fn off_diagonal(&self) -> VectorN<N::RealField, DimDiff<DimMinimum<R, C>, U1>>
        where DefaultAllocator: Allocator<N::RealField, DimDiff<DimMinimum<R, C>, U1>> {
        self.off_diagonal.map(|e| e.modulus())
    }

    #[doc(hidden)]
    pub fn uv_internal(&self) -> &MatrixMN<N, R, C> {
        &self.uv
    }
}

// impl<N: ComplexField, D: DimMin<D, Output = D> + DimSub<Dynamic>> Bidiagonal<N, D, D>
//     where DefaultAllocator: Allocator<N, D, D> +
//                             Allocator<N, D> {
//     /// Solves the linear system `self * x = b`, where `x` is the unknown to be determined.
//     pub fn solve<R2: Dim, C2: Dim, S2>(&self, b: &Matrix<N, R2, C2, S2>) -> MatrixMN<N, R2, C2>
//         where S2: StorageMut<N, R2, C2>,
//               ShapeConstraint: SameNumberOfRows<R2, D>,
//               DefaultAllocator: Allocator<N, R2, C2> {
//         let mut res = b.clone_owned();
//         self.solve_mut(&mut res);
//         res
//     }
//
//     /// Solves the linear system `self * x = b`, where `x` is the unknown to be determined.
//     pub fn solve_mut<R2: Dim, C2: Dim, S2>(&self, b: &mut Matrix<N, R2, C2, S2>)
//         where S2: StorageMut<N, R2, C2>,
//               ShapeConstraint: SameNumberOfRows<R2, D> {
//
//         assert_eq!(self.uv.nrows(), b.nrows(), "Bidiagonal solve matrix dimension mismatch.");
//         assert!(self.uv.is_square(), "Bidiagonal solve: unable to solve a non-square system.");
//
//         self.q_tr_mul(b);
//         self.solve_upper_triangular_mut(b);
//     }
//
//     // FIXME: duplicate code from the `solve` module.
//     fn solve_upper_triangular_mut<R2: Dim, C2: Dim, S2>(&self, b: &mut Matrix<N, R2, C2, S2>)
//         where S2: StorageMut<N, R2, C2>,
//               ShapeConstraint: SameNumberOfRows<R2, D> {
//
//         let dim  = self.uv.nrows();
//
//         for k in 0 .. b.ncols() {
//             let mut b = b.column_mut(k);
//             for i in (0 .. dim).rev() {
//                 let coeff;
//
//                 unsafe {
//                     let diag = *self.diag.vget_unchecked(i);
//                     coeff = *b.vget_unchecked(i) / diag;
//                     *b.vget_unchecked_mut(i) = coeff;
//                 }
//
//                 b.rows_range_mut(.. i).axpy(-coeff, &self.uv.slice_range(.. i, i), N::one());
//             }
//         }
//     }
//
//     /// Computes the inverse of the decomposed matrix.
//     pub fn inverse(&self) -> MatrixN<N, D> {
//         assert!(self.uv.is_square(), "Bidiagonal inverse: unable to compute the inverse of a non-square matrix.");
//
//         // FIXME: is there a less naive method ?
//         let (nrows, ncols) = self.uv.data.shape();
//         let mut res = MatrixN::identity_generic(nrows, ncols);
//         self.solve_mut(&mut res);
//         res
//     }
//
//     // /// Computes the determinant of the decomposed matrix.
//     // pub fn determinant(&self) -> N {
//     //     let dim = self.uv.nrows();
//     //     assert!(self.uv.is_square(), "Bidiagonal determinant: unable to compute the determinant of a non-square matrix.");
//
//     //     let mut res = N::one();
//     //     for i in 0 .. dim {
//     //         res *= unsafe { *self.diag.vget_unchecked(i) };
//     //     }
//
//     //     res self.q_determinant()
//     // }
// }

impl<N: ComplexField, R: DimMin<C>, C: Dim, S: Storage<N, R, C>> Matrix<N, R, C, S>
where
    DimMinimum<R, C>: DimSub<U1>,
    DefaultAllocator: Allocator<N, R, C>
        + Allocator<N, C>
        + Allocator<N, R>
        + Allocator<N, DimMinimum<R, C>>
        + Allocator<N, DimDiff<DimMinimum<R, C>, U1>>,
{
    /// Computes the bidiagonalization using householder reflections.
    pub fn bidiagonalize(self) -> Bidiagonal<N, R, C> {
        Bidiagonal::new(self.into_owned())
    }
}
