#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use alga::general::ComplexField;
use crate::allocator::Allocator;
use crate::base::{DefaultAllocator, Matrix, MatrixMN, MatrixN};
use crate::constraint::{SameNumberOfRows, ShapeConstraint};
use crate::{Dim, DimMin, DimMinimum};
use crate::storage::{Storage, StorageMut};

use crate::linalg::lu;
use crate::linalg::PermutationSequence;

/// LU decomposition with full row and column pivoting.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(
        serialize = "DefaultAllocator: Allocator<N, R, C> +
                           Allocator<(usize, usize), DimMinimum<R, C>>,
         MatrixMN<N, R, C>: Serialize,
         PermutationSequence<DimMinimum<R, C>>: Serialize"
    ))
)]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(
        deserialize = "DefaultAllocator: Allocator<N, R, C> +
                           Allocator<(usize, usize), DimMinimum<R, C>>,
         MatrixMN<N, R, C>: Deserialize<'de>,
         PermutationSequence<DimMinimum<R, C>>: Deserialize<'de>"
    ))
)]
#[derive(Clone, Debug)]
pub struct FullPivLU<N: ComplexField, R: DimMin<C>, C: Dim>
where DefaultAllocator: Allocator<N, R, C> + Allocator<(usize, usize), DimMinimum<R, C>>
{
    lu: MatrixMN<N, R, C>,
    p: PermutationSequence<DimMinimum<R, C>>,
    q: PermutationSequence<DimMinimum<R, C>>,
}

impl<N: ComplexField, R: DimMin<C>, C: Dim> Copy for FullPivLU<N, R, C>
where
    DefaultAllocator: Allocator<N, R, C> + Allocator<(usize, usize), DimMinimum<R, C>>,
    MatrixMN<N, R, C>: Copy,
    PermutationSequence<DimMinimum<R, C>>: Copy,
{}

impl<N: ComplexField, R: DimMin<C>, C: Dim> FullPivLU<N, R, C>
where DefaultAllocator: Allocator<N, R, C> + Allocator<(usize, usize), DimMinimum<R, C>>
{
    /// Computes the LU decomposition with full pivoting of `matrix`.
    ///
    /// This effectively computes `P, L, U, Q` such that `P * matrix * Q = LU`.
    pub fn new(mut matrix: MatrixMN<N, R, C>) -> Self {
        let (nrows, ncols) = matrix.data.shape();
        let min_nrows_ncols = nrows.min(ncols);

        let mut p = PermutationSequence::identity_generic(min_nrows_ncols);
        let mut q = PermutationSequence::identity_generic(min_nrows_ncols);

        if min_nrows_ncols.value() == 0 {
            return Self {
                lu: matrix,
                p: p,
                q: q,
            };
        }

        for i in 0..min_nrows_ncols.value() {
            let piv = matrix.slice_range(i.., i..).icamax_full();
            let row_piv = piv.0 + i;
            let col_piv = piv.1 + i;
            let diag = matrix[(row_piv, col_piv)];

            if diag.is_zero() {
                // The remaining of the matrix is zero.
                break;
            }

            matrix.swap_columns(i, col_piv);
            q.append_permutation(i, col_piv);

            if row_piv != i {
                p.append_permutation(i, row_piv);
                matrix.columns_range_mut(..i).swap_rows(i, row_piv);
                lu::gauss_step_swap(&mut matrix, diag, i, row_piv);
            } else {
                lu::gauss_step(&mut matrix, diag, i);
            }
        }

        Self {
            lu: matrix,
            p: p,
            q: q,
        }
    }

    #[doc(hidden)]
    pub fn lu_internal(&self) -> &MatrixMN<N, R, C> {
        &self.lu
    }

    /// The lower triangular matrix of this decomposition.
    #[inline]
    pub fn l(&self) -> MatrixMN<N, R, DimMinimum<R, C>>
    where DefaultAllocator: Allocator<N, R, DimMinimum<R, C>> {
        let (nrows, ncols) = self.lu.data.shape();
        let mut m = self.lu.columns_generic(0, nrows.min(ncols)).into_owned();
        m.fill_upper_triangle(N::zero(), 1);
        m.fill_diagonal(N::one());
        m
    }

    /// The upper triangular matrix of this decomposition.
    #[inline]
    pub fn u(&self) -> MatrixMN<N, DimMinimum<R, C>, C>
    where DefaultAllocator: Allocator<N, DimMinimum<R, C>, C> {
        let (nrows, ncols) = self.lu.data.shape();
        self.lu.rows_generic(0, nrows.min(ncols)).upper_triangle()
    }

    /// The row permutations of this decomposition.
    #[inline]
    pub fn p(&self) -> &PermutationSequence<DimMinimum<R, C>> {
        &self.p
    }

    /// The column permutations of this decomposition.
    #[inline]
    pub fn q(&self) -> &PermutationSequence<DimMinimum<R, C>> {
        &self.q
    }

    /// The two matrices of this decomposition and the row and column permutations: `(P, L, U, Q)`.
    #[inline]
    pub fn unpack(
        self,
    ) -> (
        PermutationSequence<DimMinimum<R, C>>,
        MatrixMN<N, R, DimMinimum<R, C>>,
        MatrixMN<N, DimMinimum<R, C>, C>,
        PermutationSequence<DimMinimum<R, C>>,
    )
    where DefaultAllocator: Allocator<N, R, DimMinimum<R, C>> + Allocator<N, DimMinimum<R, C>, C>
    {
        // Use reallocation for either l or u.
        let l = self.l();
        let u = self.u();
        let p = self.p;
        let q = self.q;

        (p, l, u, q)
    }
}

impl<N: ComplexField, D: DimMin<D, Output = D>> FullPivLU<N, D, D>
where DefaultAllocator: Allocator<N, D, D> + Allocator<(usize, usize), D>
{
    /// Solves the linear system `self * x = b`, where `x` is the unknown to be determined.
    ///
    /// Returns `None` if the decomposed matrix is not invertible.
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
    /// If the decomposed matrix is not invertible, this returns `false` and its input `b` may
    /// be overwritten with garbage.
    pub fn solve_mut<R2: Dim, C2: Dim, S2>(&self, b: &mut Matrix<N, R2, C2, S2>) -> bool
    where
        S2: StorageMut<N, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        assert_eq!(
            self.lu.nrows(),
            b.nrows(),
            "FullPivLU solve matrix dimension mismatch."
        );
        assert!(
            self.lu.is_square(),
            "FullPivLU solve: unable to solve a non-square system."
        );

        if self.is_invertible() {
            self.p.permute_rows(b);
            let _ = self.lu.solve_lower_triangular_with_diag_mut(b, N::one());
            let _ = self.lu.solve_upper_triangular_mut(b);
            self.q.inv_permute_rows(b);

            true
        } else {
            false
        }
    }

    /// Computes the inverse of the decomposed matrix.
    ///
    /// Returns `None` if the decomposed matrix is not invertible.
    pub fn try_inverse(&self) -> Option<MatrixN<N, D>> {
        assert!(
            self.lu.is_square(),
            "FullPivLU inverse: unable to compute the inverse of a non-square matrix."
        );

        let (nrows, ncols) = self.lu.data.shape();

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
            self.lu.is_square(),
            "FullPivLU: unable to test the invertibility of a non-square matrix."
        );

        let dim = self.lu.nrows();
        !self.lu[(dim - 1, dim - 1)].is_zero()
    }

    /// Computes the determinant of the decomposed matrix.
    pub fn determinant(&self) -> N {
        assert!(
            self.lu.is_square(),
            "FullPivLU determinant: unable to compute the determinant of a non-square matrix."
        );

        let dim = self.lu.nrows();
        let mut res = self.lu[(dim - 1, dim - 1)];
        if !res.is_zero() {
            for i in 0..dim - 1 {
                res *= unsafe { *self.lu.get_unchecked((i, i)) };
            }

            res * self.p.determinant() * self.q.determinant()
        } else {
            N::zero()
        }
    }
}

impl<N: ComplexField, R: DimMin<C>, C: Dim, S: Storage<N, R, C>> Matrix<N, R, C, S>
where DefaultAllocator: Allocator<N, R, C> + Allocator<(usize, usize), DimMinimum<R, C>>
{
    /// Computes the LU decomposition with full pivoting of `matrix`.
    ///
    /// This effectively computes `P, L, U, Q` such that `P * matrix * Q = LU`.
    pub fn full_piv_lu(self) -> FullPivLU<N, R, C> {
        FullPivLU::new(self.into_owned())
    }
}
