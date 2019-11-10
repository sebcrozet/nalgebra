use alga::general::ComplexField;

use crate::base::allocator::Allocator;
use crate::base::constraint::{SameNumberOfRows, ShapeConstraint};
use crate::{Dim, U1};
use crate::base::storage::{Storage, StorageMut};
use crate::base::{DefaultAllocator, Matrix, MatrixMN, SquareMatrix, Vector, DVectorSlice};

impl<N: ComplexField, D: Dim, S: Storage<N, D, D>> SquareMatrix<N, D, S> {
    /// Computes the solution of the linear system `self . x = b` where `x` is the unknown and only
    /// the lower-triangular part of `self` (including the diagonal) is considered not-zero.
    #[inline]
    pub fn solve_lower_triangular<R2: Dim, C2: Dim, S2>(
        &self,
        b: &Matrix<N, R2, C2, S2>,
    ) -> Option<MatrixMN<N, R2, C2>>
    where
        S2: StorageMut<N, R2, C2>,
        DefaultAllocator: Allocator<N, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let mut res = b.clone_owned();
        if self.solve_lower_triangular_mut(&mut res) {
            Some(res)
        } else {
            None
        }
    }

    /// Computes the solution of the linear system `self . x = b` where `x` is the unknown and only
    /// the upper-triangular part of `self` (including the diagonal) is considered not-zero.
    #[inline]
    pub fn solve_upper_triangular<R2: Dim, C2: Dim, S2>(
        &self,
        b: &Matrix<N, R2, C2, S2>,
    ) -> Option<MatrixMN<N, R2, C2>>
    where
        S2: StorageMut<N, R2, C2>,
        DefaultAllocator: Allocator<N, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let mut res = b.clone_owned();
        if self.solve_upper_triangular_mut(&mut res) {
            Some(res)
        } else {
            None
        }
    }

    /// Solves the linear system `self . x = b` where `x` is the unknown and only the
    /// lower-triangular part of `self` (including the diagonal) is considered not-zero.
    pub fn solve_lower_triangular_mut<R2: Dim, C2: Dim, S2>(
        &self,
        b: &mut Matrix<N, R2, C2, S2>,
    ) -> bool
    where
        S2: StorageMut<N, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let cols = b.ncols();

        for i in 0..cols {
            if !self.solve_lower_triangular_vector_mut(&mut b.column_mut(i)) {
                return false;
            }
        }

        true
    }

    fn solve_lower_triangular_vector_mut<R2: Dim, S2>(&self, b: &mut Vector<N, R2, S2>) -> bool
    where
        S2: StorageMut<N, R2, U1>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let dim = self.nrows();

        for i in 0..dim {
            let coeff;

            unsafe {
                let diag = *self.get_unchecked((i, i));

                if diag.is_zero() {
                    return false;
                }

                coeff = *b.vget_unchecked(i) / diag;
                *b.vget_unchecked_mut(i) = coeff;
            }

            b.rows_range_mut(i + 1..)
                .axpy(-coeff, &self.slice_range(i + 1.., i), N::one());
        }

        true
    }

    // FIXME: add the same but for solving upper-triangular.
    /// Solves the linear system `self . x = b` where `x` is the unknown and only the
    /// lower-triangular part of `self` is considered not-zero. The diagonal is never read as it is
    /// assumed to be equal to `diag`. Returns `false` and does not modify its inputs if `diag` is zero.
    pub fn solve_lower_triangular_with_diag_mut<R2: Dim, C2: Dim, S2>(
        &self,
        b: &mut Matrix<N, R2, C2, S2>,
        diag: N,
    ) -> bool
    where
        S2: StorageMut<N, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        if diag.is_zero() {
            return false;
        }

        let dim = self.nrows();
        let cols = b.ncols();

        for k in 0..cols {
            let mut bcol = b.column_mut(k);

            for i in 0..dim - 1 {
                let coeff = unsafe { *bcol.vget_unchecked(i) } / diag;
                bcol.rows_range_mut(i + 1..)
                    .axpy(-coeff, &self.slice_range(i + 1.., i), N::one());
            }
        }

        true
    }

    /// Solves the linear system `self . x = b` where `x` is the unknown and only the
    /// upper-triangular part of `self` (including the diagonal) is considered not-zero.
    pub fn solve_upper_triangular_mut<R2: Dim, C2: Dim, S2>(
        &self,
        b: &mut Matrix<N, R2, C2, S2>,
    ) -> bool
    where
        S2: StorageMut<N, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let cols = b.ncols();

        for i in 0..cols {
            if !self.solve_upper_triangular_vector_mut(&mut b.column_mut(i)) {
                return false;
            }
        }

        true
    }

    fn solve_upper_triangular_vector_mut<R2: Dim, S2>(&self, b: &mut Vector<N, R2, S2>) -> bool
    where
        S2: StorageMut<N, R2, U1>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let dim = self.nrows();

        for i in (0..dim).rev() {
            let coeff;

            unsafe {
                let diag = *self.get_unchecked((i, i));

                if diag.is_zero() {
                    return false;
                }

                coeff = *b.vget_unchecked(i) / diag;
                *b.vget_unchecked_mut(i) = coeff;
            }

            b.rows_range_mut(..i)
                .axpy(-coeff, &self.slice_range(..i, i), N::one());
        }

        true
    }

    /*
     *
     * Transpose and adjoint versions
     *
     */
    /// Computes the solution of the linear system `self.transpose() . x = b` where `x` is the unknown and only
    /// the lower-triangular part of `self` (including the diagonal) is considered not-zero.
    #[inline]
    pub fn tr_solve_lower_triangular<R2: Dim, C2: Dim, S2>(
        &self,
        b: &Matrix<N, R2, C2, S2>,
    ) -> Option<MatrixMN<N, R2, C2>>
        where
            S2: StorageMut<N, R2, C2>,
            DefaultAllocator: Allocator<N, R2, C2>,
            ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let mut res = b.clone_owned();
        if self.tr_solve_lower_triangular_mut(&mut res) {
            Some(res)
        } else {
            None
        }
    }

    /// Computes the solution of the linear system `self.transpose() . x = b` where `x` is the unknown and only
    /// the upper-triangular part of `self` (including the diagonal) is considered not-zero.
    #[inline]
    pub fn tr_solve_upper_triangular<R2: Dim, C2: Dim, S2>(
        &self,
        b: &Matrix<N, R2, C2, S2>,
    ) -> Option<MatrixMN<N, R2, C2>>
        where
            S2: StorageMut<N, R2, C2>,
            DefaultAllocator: Allocator<N, R2, C2>,
            ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let mut res = b.clone_owned();
        if self.tr_solve_upper_triangular_mut(&mut res) {
            Some(res)
        } else {
            None
        }
    }

    /// Solves the linear system `self.transpose() . x = b` where `x` is the unknown and only the
    /// lower-triangular part of `self` (including the diagonal) is considered not-zero.
    pub fn tr_solve_lower_triangular_mut<R2: Dim, C2: Dim, S2>(
        &self,
        b: &mut Matrix<N, R2, C2, S2>,
    ) -> bool
        where
            S2: StorageMut<N, R2, C2>,
            ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let cols = b.ncols();

        for i in 0..cols {
            if !self.xx_solve_lower_triangular_vector_mut(&mut b.column_mut(i), |e| e, |a, b| a.dot(b)) {
                return false;
            }
        }

        true
    }

    /// Solves the linear system `self.transpose() . x = b` where `x` is the unknown and only the
    /// upper-triangular part of `self` (including the diagonal) is considered not-zero.
    pub fn tr_solve_upper_triangular_mut<R2: Dim, C2: Dim, S2>(
        &self,
        b: &mut Matrix<N, R2, C2, S2>,
    ) -> bool
        where
            S2: StorageMut<N, R2, C2>,
            ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let cols = b.ncols();

        for i in 0..cols {
            if !self.xx_solve_upper_triangular_vector_mut(&mut b.column_mut(i), |e| e, |a, b| a.dot(b)) {
                return false;
            }
        }

        true
    }

    /// Computes the solution of the linear system `self.adjoint() . x = b` where `x` is the unknown and only
    /// the lower-triangular part of `self` (including the diagonal) is considered not-zero.
    #[inline]
    pub fn ad_solve_lower_triangular<R2: Dim, C2: Dim, S2>(
        &self,
        b: &Matrix<N, R2, C2, S2>,
    ) -> Option<MatrixMN<N, R2, C2>>
        where
            S2: StorageMut<N, R2, C2>,
            DefaultAllocator: Allocator<N, R2, C2>,
            ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let mut res = b.clone_owned();
        if self.ad_solve_lower_triangular_mut(&mut res) {
            Some(res)
        } else {
            None
        }
    }

    /// Computes the solution of the linear system `self.adjoint() . x = b` where `x` is the unknown and only
    /// the upper-triangular part of `self` (including the diagonal) is considered not-zero.
    #[inline]
    pub fn ad_solve_upper_triangular<R2: Dim, C2: Dim, S2>(
        &self,
        b: &Matrix<N, R2, C2, S2>,
    ) -> Option<MatrixMN<N, R2, C2>>
        where
            S2: StorageMut<N, R2, C2>,
            DefaultAllocator: Allocator<N, R2, C2>,
            ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let mut res = b.clone_owned();
        if self.ad_solve_upper_triangular_mut(&mut res) {
            Some(res)
        } else {
            None
        }
    }

    /// Solves the linear system `self.adjoint() . x = b` where `x` is the unknown and only the
    /// lower-triangular part of `self` (including the diagonal) is considered not-zero.
    pub fn ad_solve_lower_triangular_mut<R2: Dim, C2: Dim, S2>(
        &self,
        b: &mut Matrix<N, R2, C2, S2>,
    ) -> bool
        where
            S2: StorageMut<N, R2, C2>,
            ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let cols = b.ncols();

        for i in 0..cols {
            if !self.xx_solve_lower_triangular_vector_mut(&mut b.column_mut(i), |e| e.conjugate(), |a, b| a.dotc(b)) {
                return false;
            }
        }

        true
    }

    /// Solves the linear system `self.adjoint() . x = b` where `x` is the unknown and only the
    /// upper-triangular part of `self` (including the diagonal) is considered not-zero.
    pub fn ad_solve_upper_triangular_mut<R2: Dim, C2: Dim, S2>(
        &self,
        b: &mut Matrix<N, R2, C2, S2>,
    ) -> bool
        where
            S2: StorageMut<N, R2, C2>,
            ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let cols = b.ncols();

        for i in 0..cols {
            if !self.xx_solve_upper_triangular_vector_mut(&mut b.column_mut(i), |e| e.conjugate(), |a, b| a.dotc(b)) {
                return false;
            }
        }

        true
    }


    #[inline(always)]
    fn xx_solve_lower_triangular_vector_mut<R2: Dim, S2>(
        &self,
        b: &mut Vector<N, R2, S2>,
        conjugate: impl Fn(N) -> N,
        dot: impl Fn(&DVectorSlice<N, S::RStride, S::CStride>, &DVectorSlice<N, S2::RStride, S2::CStride>) -> N,
    ) -> bool
        where
            S2: StorageMut<N, R2, U1>,
            ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let dim = self.nrows();

        for i in (0..dim).rev() {
            let dot = dot(&self.slice_range(i + 1.., i), &b.slice_range(i + 1.., 0));

            unsafe {
                let b_i = b.vget_unchecked_mut(i);

                let diag = conjugate(*self.get_unchecked((i, i)));

                if diag.is_zero() {
                    return false;
                }

                *b_i = (*b_i - dot) / diag;
            }
        }

        true
    }

    #[inline(always)]
    fn xx_solve_upper_triangular_vector_mut<R2: Dim, S2>(
        &self,
        b: &mut Vector<N, R2, S2>,
        conjugate: impl Fn(N) -> N,
        dot: impl Fn(&DVectorSlice<N, S::RStride, S::CStride>, &DVectorSlice<N, S2::RStride, S2::CStride>) -> N,
    ) -> bool
        where
            S2: StorageMut<N, R2, U1>,
            ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let dim = self.nrows();

        for i in 0..dim {
            let dot = dot(&self.slice_range(..i, i), &b.slice_range(..i, 0));

            unsafe {
                let b_i = b.vget_unchecked_mut(i);
                let diag = conjugate(*self.get_unchecked((i, i)));

                if diag.is_zero() {
                    return false;
                }

                *b_i = (*b_i - dot) / diag;
            }
        }

        true
    }
}
