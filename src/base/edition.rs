use num::{One, Zero};
use std::cmp;
#[cfg(any(feature = "std", feature = "alloc"))]
use std::iter::ExactSizeIterator;
#[cfg(any(feature = "std", feature = "alloc"))]
use std::mem;
use std::ptr;

use crate::base::allocator::{Allocator, Reallocator};
use crate::base::constraint::{DimEq, SameNumberOfColumns, SameNumberOfRows, ShapeConstraint};
#[cfg(any(feature = "std", feature = "alloc"))]
use crate::Dynamic;
use crate::{
    Dim, DimAdd, DimDiff, DimMin, DimMinimum, DimName, DimSub, DimSum, U1,
};
use crate::base::storage::{Storage, StorageMut};
#[cfg(any(feature = "std", feature = "alloc"))]
use crate::base::DMatrix;
use crate::base::{DefaultAllocator, Matrix, MatrixMN, RowVector, Scalar, Vector};

impl<N: Scalar + Zero, R: Dim, C: Dim, S: Storage<N, R, C>> Matrix<N, R, C, S> {
    /// Extracts the upper triangular part of this matrix (including the diagonal).
    #[inline]
    pub fn upper_triangle(&self) -> MatrixMN<N, R, C>
    where DefaultAllocator: Allocator<N, R, C> {
        let mut res = self.clone_owned();
        res.fill_lower_triangle(N::zero(), 1);

        res
    }

    /// Extracts the lower triangular part of this matrix (including the diagonal).
    #[inline]
    pub fn lower_triangle(&self) -> MatrixMN<N, R, C>
    where DefaultAllocator: Allocator<N, R, C> {
        let mut res = self.clone_owned();
        res.fill_upper_triangle(N::zero(), 1);

        res
    }

    /// Creates a new matrix by extracting the given set of rows from `self`.
    #[cfg(any(feature = "std", feature = "alloc"))]
    pub fn select_rows<'a, I>(&self, irows: I) -> MatrixMN<N, Dynamic, C>
    where
        I: IntoIterator<Item = &'a usize>,
        I::IntoIter: ExactSizeIterator + Clone,
        DefaultAllocator: Allocator<N, Dynamic, C>,
    {
        let irows = irows.into_iter();
        let ncols = self.data.shape().1;
        let mut res =
            unsafe { MatrixMN::new_uninitialized_generic(Dynamic::new(irows.len()), ncols) };

        // First, check that all the indices from irows are valid.
        // This will allow us to use unchecked access in the inner loop.
        for i in irows.clone() {
            assert!(*i < self.nrows(), "Row index out of bounds.")
        }

        for j in 0..ncols.value() {
            // FIXME: use unchecked column indexing
            let mut res = res.column_mut(j);
            let src = self.column(j);

            for (destination, source) in irows.clone().enumerate() {
                unsafe { *res.vget_unchecked_mut(destination) = *src.vget_unchecked(*source) }
            }
        }

        res
    }

    /// Creates a new matrix by extracting the given set of columns from `self`.
    #[cfg(any(feature = "std", feature = "alloc"))]
    pub fn select_columns<'a, I>(&self, icols: I) -> MatrixMN<N, R, Dynamic>
    where
        I: IntoIterator<Item = &'a usize>,
        I::IntoIter: ExactSizeIterator,
        DefaultAllocator: Allocator<N, R, Dynamic>,
    {
        let icols = icols.into_iter();
        let nrows = self.data.shape().0;
        let mut res =
            unsafe { MatrixMN::new_uninitialized_generic(nrows, Dynamic::new(icols.len())) };

        for (destination, source) in icols.enumerate() {
            res.column_mut(destination).copy_from(&self.column(*source))
        }

        res
    }
}

impl<N: Scalar, R: Dim, C: Dim, S: StorageMut<N, R, C>> Matrix<N, R, C, S> {
    /// Sets all the elements of this matrix to `val`.
    #[inline]
    pub fn fill(&mut self, val: N) {
        for e in self.iter_mut() {
            *e = val
        }
    }

    /// Fills `self` with the identity matrix.
    #[inline]
    pub fn fill_with_identity(&mut self)
    where N: Zero + One {
        self.fill(N::zero());
        self.fill_diagonal(N::one());
    }

    /// Sets all the diagonal elements of this matrix to `val`.
    #[inline]
    pub fn fill_diagonal(&mut self, val: N) {
        let (nrows, ncols) = self.shape();
        let n = cmp::min(nrows, ncols);

        for i in 0..n {
            unsafe { *self.get_unchecked_mut((i, i)) = val }
        }
    }

    /// Sets all the elements of the selected row to `val`.
    #[inline]
    pub fn fill_row(&mut self, i: usize, val: N) {
        assert!(i < self.nrows(), "Row index out of bounds.");
        for j in 0..self.ncols() {
            unsafe { *self.get_unchecked_mut((i, j)) = val }
        }
    }

    /// Sets all the elements of the selected column to `val`.
    #[inline]
    pub fn fill_column(&mut self, j: usize, val: N) {
        assert!(j < self.ncols(), "Row index out of bounds.");
        for i in 0..self.nrows() {
            unsafe { *self.get_unchecked_mut((i, j)) = val }
        }
    }

    /// Fills the diagonal of this matrix with the content of the given vector.
    #[inline]
    pub fn set_diagonal<R2: Dim, S2>(&mut self, diag: &Vector<N, R2, S2>)
    where
        R: DimMin<C>,
        S2: Storage<N, R2>,
        ShapeConstraint: DimEq<DimMinimum<R, C>, R2>,
    {
        let (nrows, ncols) = self.shape();
        let min_nrows_ncols = cmp::min(nrows, ncols);
        assert_eq!(diag.len(), min_nrows_ncols, "Mismatched dimensions.");

        for i in 0..min_nrows_ncols {
            unsafe { *self.get_unchecked_mut((i, i)) = *diag.vget_unchecked(i) }
        }
    }

    /// Fills the diagonal of this matrix with the content of the given iterator.
    ///
    /// This will fill as many diagonal elements as the iterator yields, up to the
    /// minimum of the number of rows and columns of `self`, and starting with the
    /// diagonal element at index (0, 0).
    #[inline]
    pub fn set_partial_diagonal(&mut self, diag: impl Iterator<Item = N>) {
        let (nrows, ncols) = self.shape();
        let min_nrows_ncols = cmp::min(nrows, ncols);

        for (i, val) in diag.enumerate().take(min_nrows_ncols) {
            unsafe { *self.get_unchecked_mut((i, i)) = val }
        }
    }

    /// Fills the selected row of this matrix with the content of the given vector.
    #[inline]
    pub fn set_row<C2: Dim, S2>(&mut self, i: usize, row: &RowVector<N, C2, S2>)
    where
        S2: Storage<N, U1, C2>,
        ShapeConstraint: SameNumberOfColumns<C, C2>,
    {
        self.row_mut(i).copy_from(row);
    }

    /// Fills the selected column of this matrix with the content of the given vector.
    #[inline]
    pub fn set_column<R2: Dim, S2>(&mut self, i: usize, column: &Vector<N, R2, S2>)
    where
        S2: Storage<N, R2, U1>,
        ShapeConstraint: SameNumberOfRows<R, R2>,
    {
        self.column_mut(i).copy_from(column);
    }

    /// Sets all the elements of the lower-triangular part of this matrix to `val`.
    ///
    /// The parameter `shift` allows some subdiagonals to be left untouched:
    /// * If `shift = 0` then the diagonal is overwritten as well.
    /// * If `shift = 1` then the diagonal is left untouched.
    /// * If `shift > 1`, then the diagonal and the first `shift - 1` subdiagonals are left
    /// untouched.
    #[inline]
    pub fn fill_lower_triangle(&mut self, val: N, shift: usize) {
        for j in 0..self.ncols() {
            for i in (j + shift)..self.nrows() {
                unsafe { *self.get_unchecked_mut((i, j)) = val }
            }
        }
    }

    /// Sets all the elements of the lower-triangular part of this matrix to `val`.
    ///
    /// The parameter `shift` allows some superdiagonals to be left untouched:
    /// * If `shift = 0` then the diagonal is overwritten as well.
    /// * If `shift = 1` then the diagonal is left untouched.
    /// * If `shift > 1`, then the diagonal and the first `shift - 1` superdiagonals are left
    /// untouched.
    #[inline]
    pub fn fill_upper_triangle(&mut self, val: N, shift: usize) {
        for j in shift..self.ncols() {
            // FIXME: is there a more efficient way to avoid the min ?
            // (necessary for rectangular matrices)
            for i in 0..cmp::min(j + 1 - shift, self.nrows()) {
                unsafe { *self.get_unchecked_mut((i, j)) = val }
            }
        }
    }

    /// Swaps two rows in-place.
    #[inline]
    pub fn swap_rows(&mut self, irow1: usize, irow2: usize) {
        assert!(irow1 < self.nrows() && irow2 < self.nrows());

        if irow1 != irow2 {
            // FIXME: optimize that.
            for i in 0..self.ncols() {
                unsafe { self.swap_unchecked((irow1, i), (irow2, i)) }
            }
        }
        // Otherwise do nothing.
    }

    /// Swaps two columns in-place.
    #[inline]
    pub fn swap_columns(&mut self, icol1: usize, icol2: usize) {
        assert!(icol1 < self.ncols() && icol2 < self.ncols());

        if icol1 != icol2 {
            // FIXME: optimize that.
            for i in 0..self.nrows() {
                unsafe { self.swap_unchecked((i, icol1), (i, icol2)) }
            }
        }
        // Otherwise do nothing.
    }
}

impl<N: Scalar, D: Dim, S: StorageMut<N, D, D>> Matrix<N, D, D, S> {
    /// Copies the upper-triangle of this matrix to its lower-triangular part.
    ///
    /// This makes the matrix symmetric. Panics if the matrix is not square.
    pub fn fill_lower_triangle_with_upper_triangle(&mut self) {
        assert!(self.is_square(), "The input matrix should be square.");

        let dim = self.nrows();
        for j in 0..dim {
            for i in j + 1..dim {
                unsafe {
                    *self.get_unchecked_mut((i, j)) = *self.get_unchecked((j, i));
                }
            }
        }
    }

    /// Copies the upper-triangle of this matrix to its upper-triangular part.
    ///
    /// This makes the matrix symmetric. Panics if the matrix is not square.
    pub fn fill_upper_triangle_with_lower_triangle(&mut self) {
        assert!(self.is_square(), "The input matrix should be square.");

        for j in 1..self.ncols() {
            for i in 0..j {
                unsafe {
                    *self.get_unchecked_mut((i, j)) = *self.get_unchecked((j, i));
                }
            }
        }
    }
}

/*
 *
 * FIXME: specialize all the following for slices.
 *
 */
impl<N: Scalar, R: Dim, C: Dim, S: Storage<N, R, C>> Matrix<N, R, C, S> {
    /*
     *
     * Column removal.
     *
     */
    /// Removes the `i`-th column from this matrix.
    #[inline]
    pub fn remove_column(self, i: usize) -> MatrixMN<N, R, DimDiff<C, U1>>
    where
        C: DimSub<U1>,
        DefaultAllocator: Reallocator<N, R, C, R, DimDiff<C, U1>>,
    {
        self.remove_fixed_columns::<U1>(i)
    }

    /// Removes all columns in `indices`   
    #[cfg(any(feature = "std", feature = "alloc"))]
    pub fn remove_columns_at(self, indices: &[usize]) -> MatrixMN<N, R, Dynamic>
    where
        C: DimSub<Dynamic, Output = Dynamic>,
        DefaultAllocator: Reallocator<N, R, C, R, Dynamic>,
    {
        let mut m = self.into_owned();
        let (nrows, ncols) = m.data.shape();
        let mut offset: usize = 0;
        let mut target: usize = 0;
        while offset + target < ncols.value() {
            if indices.contains(&(target + offset)) {
                offset += 1;
            } else {
                unsafe {
                    let ptr_source = m
                        .data
                        .ptr()
                        .offset(((target + offset) * nrows.value()) as isize);
                    let ptr_target = m.data.ptr_mut().offset((target * nrows.value()) as isize);

                    ptr::copy(ptr_source, ptr_target, nrows.value());
                    target += 1;
                }
            }
        }

        unsafe {
            Matrix::from_data(DefaultAllocator::reallocate_copy(
                nrows,
                ncols.sub(Dynamic::from_usize(offset)),
                m.data,
            ))
        }
    }

    /// Removes all rows in `indices`   
    #[cfg(any(feature = "std", feature = "alloc"))]
    pub fn remove_rows_at(self, indices: &[usize]) -> MatrixMN<N, Dynamic, C>
    where
        R: DimSub<Dynamic, Output = Dynamic>,
        DefaultAllocator: Reallocator<N, R, C, Dynamic, C>,
    {        
        let mut m = self.into_owned();
        let (nrows, ncols) = m.data.shape();
        let mut offset: usize = 0;
        let mut target: usize = 0;
        while offset + target < nrows.value() * ncols.value() {
            if indices.contains(&((target + offset) % nrows.value())) {
                offset += 1;
            } else {
                unsafe {
                    let ptr_source = m.data.ptr().offset((target + offset) as isize);
                    let ptr_target = m.data.ptr_mut().offset(target as isize);

                    ptr::copy(ptr_source, ptr_target, 1);
                    target += 1;
                }
            }
        }

        unsafe {
            Matrix::from_data(DefaultAllocator::reallocate_copy(
                nrows.sub(Dynamic::from_usize(offset / ncols.value ())),
                ncols,
                m.data,
            ))
        }
    }

    /// Removes `D::dim()` consecutive columns from this matrix, starting with the `i`-th
    /// (included).
    #[inline]
    pub fn remove_fixed_columns<D>(self, i: usize) -> MatrixMN<N, R, DimDiff<C, D>>
    where
        D: DimName,
        C: DimSub<D>,
        DefaultAllocator: Reallocator<N, R, C, R, DimDiff<C, D>>,
    {
        self.remove_columns_generic(i, D::name())
    }

    /// Removes `n` consecutive columns from this matrix, starting with the `i`-th (included).
    #[inline]
    #[cfg(any(feature = "std", feature = "alloc"))]
    pub fn remove_columns(self, i: usize, n: usize) -> MatrixMN<N, R, Dynamic>
    where
        C: DimSub<Dynamic, Output = Dynamic>,
        DefaultAllocator: Reallocator<N, R, C, R, Dynamic>,
    {
        self.remove_columns_generic(i, Dynamic::new(n))
    }

    /// Removes `nremove.value()` columns from this matrix, starting with the `i`-th (included).
    ///
    /// This is the generic implementation of `.remove_columns(...)` and
    /// `.remove_fixed_columns(...)` which have nicer API interfaces.
    #[inline]
    pub fn remove_columns_generic<D>(self, i: usize, nremove: D) -> MatrixMN<N, R, DimDiff<C, D>>
    where
        D: Dim,
        C: DimSub<D>,
        DefaultAllocator: Reallocator<N, R, C, R, DimDiff<C, D>>,
    {
        let mut m = self.into_owned();
        let (nrows, ncols) = m.data.shape();
        assert!(
            i + nremove.value() <= ncols.value(),
            "Column index out of range."
        );

        if nremove.value() != 0 && i + nremove.value() < ncols.value() {
            // The first `deleted_i * nrows` are left untouched.
            let copied_value_start = i + nremove.value();

            unsafe {
                let ptr_in = m
                    .data
                    .ptr()
                    .offset((copied_value_start * nrows.value()) as isize);
                let ptr_out = m.data.ptr_mut().offset((i * nrows.value()) as isize);

                ptr::copy(
                    ptr_in,
                    ptr_out,
                    (ncols.value() - copied_value_start) * nrows.value(),
                );
            }
        }

        unsafe {
            Matrix::from_data(DefaultAllocator::reallocate_copy(
                nrows,
                ncols.sub(nremove),
                m.data,
            ))
        }
    }

    /*
     *
     * Row removal.
     *
     */
    /// Removes the `i`-th row from this matrix.
    #[inline]
    pub fn remove_row(self, i: usize) -> MatrixMN<N, DimDiff<R, U1>, C>
    where
        R: DimSub<U1>,
        DefaultAllocator: Reallocator<N, R, C, DimDiff<R, U1>, C>,
    {
        self.remove_fixed_rows::<U1>(i)
    }

    /// Removes `D::dim()` consecutive rows from this matrix, starting with the `i`-th (included).
    #[inline]
    pub fn remove_fixed_rows<D>(self, i: usize) -> MatrixMN<N, DimDiff<R, D>, C>
    where
        D: DimName,
        R: DimSub<D>,
        DefaultAllocator: Reallocator<N, R, C, DimDiff<R, D>, C>,
    {
        self.remove_rows_generic(i, D::name())
    }

    /// Removes `n` consecutive rows from this matrix, starting with the `i`-th (included).
    #[inline]
    #[cfg(any(feature = "std", feature = "alloc"))]
    pub fn remove_rows(self, i: usize, n: usize) -> MatrixMN<N, Dynamic, C>
    where
        R: DimSub<Dynamic, Output = Dynamic>,
        DefaultAllocator: Reallocator<N, R, C, Dynamic, C>,
    {
        self.remove_rows_generic(i, Dynamic::new(n))
    }

    /// Removes `nremove.value()` rows from this matrix, starting with the `i`-th (included).
    ///
    /// This is the generic implementation of `.remove_rows(...)` and `.remove_fixed_rows(...)`
    /// which have nicer API interfaces.
    #[inline]
    pub fn remove_rows_generic<D>(self, i: usize, nremove: D) -> MatrixMN<N, DimDiff<R, D>, C>
    where
        D: Dim,
        R: DimSub<D>,
        DefaultAllocator: Reallocator<N, R, C, DimDiff<R, D>, C>,
    {
        let mut m = self.into_owned();
        let (nrows, ncols) = m.data.shape();
        assert!(
            i + nremove.value() <= nrows.value(),
            "Row index out of range."
        );

        if nremove.value() != 0 {
            unsafe {
                compress_rows(
                    &mut m.data.as_mut_slice(),
                    nrows.value(),
                    ncols.value(),
                    i,
                    nremove.value(),
                );
            }
        }

        unsafe {
            Matrix::from_data(DefaultAllocator::reallocate_copy(
                nrows.sub(nremove),
                ncols,
                m.data,
            ))
        }
    }

    /*
     *
     * Columns insertion.
     *
     */
    /// Inserts a column filled with `val` at the `i-th` position.
    #[inline]
    pub fn insert_column(self, i: usize, val: N) -> MatrixMN<N, R, DimSum<C, U1>>
    where
        C: DimAdd<U1>,
        DefaultAllocator: Reallocator<N, R, C, R, DimSum<C, U1>>,
    {
        self.insert_fixed_columns::<U1>(i, val)
    }

    /// Inserts `D::dim()` columns filled with `val` starting at the `i-th` position.
    #[inline]
    pub fn insert_fixed_columns<D>(self, i: usize, val: N) -> MatrixMN<N, R, DimSum<C, D>>
    where
        D: DimName,
        C: DimAdd<D>,
        DefaultAllocator: Reallocator<N, R, C, R, DimSum<C, D>>,
    {
        let mut res = unsafe { self.insert_columns_generic_uninitialized(i, D::name()) };
        res.fixed_columns_mut::<D>(i).fill(val);
        res
    }

    /// Inserts `n` columns filled with `val` starting at the `i-th` position.
    #[inline]
    #[cfg(any(feature = "std", feature = "alloc"))]
    pub fn insert_columns(self, i: usize, n: usize, val: N) -> MatrixMN<N, R, Dynamic>
    where
        C: DimAdd<Dynamic, Output = Dynamic>,
        DefaultAllocator: Reallocator<N, R, C, R, Dynamic>,
    {
        let mut res = unsafe { self.insert_columns_generic_uninitialized(i, Dynamic::new(n)) };
        res.columns_mut(i, n).fill(val);
        res
    }

    /// Inserts `ninsert.value()` columns starting at the `i-th` place of this matrix.
    ///
    /// The added column values are not initialized.
    #[inline]
    pub unsafe fn insert_columns_generic_uninitialized<D>(
        self,
        i: usize,
        ninsert: D,
    ) -> MatrixMN<N, R, DimSum<C, D>>
    where
        D: Dim,
        C: DimAdd<D>,
        DefaultAllocator: Reallocator<N, R, C, R, DimSum<C, D>>,
    {
        let m = self.into_owned();
        let (nrows, ncols) = m.data.shape();
        let mut res = Matrix::from_data(DefaultAllocator::reallocate_copy(
            nrows,
            ncols.add(ninsert),
            m.data,
        ));

        assert!(i <= ncols.value(), "Column insertion index out of range.");

        if ninsert.value() != 0 && i != ncols.value() {
            let ptr_in = res.data.ptr().offset((i * nrows.value()) as isize);
            let ptr_out = res
                .data
                .ptr_mut()
                .offset(((i + ninsert.value()) * nrows.value()) as isize);

            ptr::copy(ptr_in, ptr_out, (ncols.value() - i) * nrows.value())
        }

        res
    }

    /*
     *
     * Rows insertion.
     *
     */
    /// Inserts a row filled with `val` at the `i-th` position.
    #[inline]
    pub fn insert_row(self, i: usize, val: N) -> MatrixMN<N, DimSum<R, U1>, C>
    where
        R: DimAdd<U1>,
        DefaultAllocator: Reallocator<N, R, C, DimSum<R, U1>, C>,
    {
        self.insert_fixed_rows::<U1>(i, val)
    }

    /// Inserts `D::dim()` rows filled with `val` starting at the `i-th` position.
    #[inline]
    pub fn insert_fixed_rows<D>(self, i: usize, val: N) -> MatrixMN<N, DimSum<R, D>, C>
    where
        D: DimName,
        R: DimAdd<D>,
        DefaultAllocator: Reallocator<N, R, C, DimSum<R, D>, C>,
    {
        let mut res = unsafe { self.insert_rows_generic_uninitialized(i, D::name()) };
        res.fixed_rows_mut::<D>(i).fill(val);
        res
    }

    /// Inserts `n` rows filled with `val` starting at the `i-th` position.
    #[inline]
    #[cfg(any(feature = "std", feature = "alloc"))]
    pub fn insert_rows(self, i: usize, n: usize, val: N) -> MatrixMN<N, Dynamic, C>
    where
        R: DimAdd<Dynamic, Output = Dynamic>,
        DefaultAllocator: Reallocator<N, R, C, Dynamic, C>,
    {
        let mut res = unsafe { self.insert_rows_generic_uninitialized(i, Dynamic::new(n)) };
        res.rows_mut(i, n).fill(val);
        res
    }

    /// Inserts `ninsert.value()` rows at the `i-th` place of this matrix.
    ///
    /// The added rows values are not initialized.
    /// This is the generic implementation of `.insert_rows(...)` and
    /// `.insert_fixed_rows(...)` which have nicer API interfaces.
    #[inline]
    pub unsafe fn insert_rows_generic_uninitialized<D>(
        self,
        i: usize,
        ninsert: D,
    ) -> MatrixMN<N, DimSum<R, D>, C>
    where
        D: Dim,
        R: DimAdd<D>,
        DefaultAllocator: Reallocator<N, R, C, DimSum<R, D>, C>,
    {
        let m = self.into_owned();
        let (nrows, ncols) = m.data.shape();
        let mut res = Matrix::from_data(DefaultAllocator::reallocate_copy(
            nrows.add(ninsert),
            ncols,
            m.data,
        ));

        assert!(i <= nrows.value(), "Row insertion index out of range.");

        if ninsert.value() != 0 {
            extend_rows(
                &mut res.data.as_mut_slice(),
                nrows.value(),
                ncols.value(),
                i,
                ninsert.value(),
            );
        }

        res
    }

    /*
     *
     * Resizing.
     *
     */

    /// Resizes this matrix so that it contains `new_nrows` rows and `new_ncols` columns.
    ///
    /// The values are copied such that `self[(i, j)] == result[(i, j)]`. If the result has more
    /// rows and/or columns than `self`, then the extra rows or columns are filled with `val`.
    #[cfg(any(feature = "std", feature = "alloc"))]
    pub fn resize(self, new_nrows: usize, new_ncols: usize, val: N) -> DMatrix<N>
    where DefaultAllocator: Reallocator<N, R, C, Dynamic, Dynamic> {
        self.resize_generic(Dynamic::new(new_nrows), Dynamic::new(new_ncols), val)
    }

    /// Resizes this matrix vertically, i.e., so that it contains `new_nrows` rows while keeping the same number of columns.
    ///
    /// The values are copied such that `self[(i, j)] == result[(i, j)]`. If the result has more
    /// rows than `self`, then the extra rows are filled with `val`.
    #[cfg(any(feature = "std", feature = "alloc"))]
    pub fn resize_vertically(self, new_nrows: usize, val: N) -> MatrixMN<N, Dynamic, C>
    where DefaultAllocator: Reallocator<N, R, C, Dynamic, C> {
        let ncols = self.data.shape().1;
        self.resize_generic(Dynamic::new(new_nrows), ncols, val)
    }

    /// Resizes this matrix horizontally, i.e., so that it contains `new_ncolumns` columns while keeping the same number of columns.
    ///
    /// The values are copied such that `self[(i, j)] == result[(i, j)]`. If the result has more
    /// columns than `self`, then the extra columns are filled with `val`.
    #[cfg(any(feature = "std", feature = "alloc"))]
    pub fn resize_horizontally(self, new_ncols: usize, val: N) -> MatrixMN<N, R, Dynamic>
    where DefaultAllocator: Reallocator<N, R, C, R, Dynamic> {
        let nrows = self.data.shape().0;
        self.resize_generic(nrows, Dynamic::new(new_ncols), val)
    }

    /// Resizes this matrix so that it contains `R2::value()` rows and `C2::value()` columns.
    ///
    /// The values are copied such that `self[(i, j)] == result[(i, j)]`. If the result has more
    /// rows and/or columns than `self`, then the extra rows or columns are filled with `val`.
    pub fn fixed_resize<R2: DimName, C2: DimName>(self, val: N) -> MatrixMN<N, R2, C2>
    where DefaultAllocator: Reallocator<N, R, C, R2, C2> {
        self.resize_generic(R2::name(), C2::name(), val)
    }

    /// Resizes `self` such that it has dimensions `new_nrows × now_ncols`.
    ///
    /// The values are copied such that `self[(i, j)] == result[(i, j)]`. If the result has more
    /// rows and/or columns than `self`, then the extra rows or columns are filled with `val`.
    #[inline]
    pub fn resize_generic<R2: Dim, C2: Dim>(
        self,
        new_nrows: R2,
        new_ncols: C2,
        val: N,
    ) -> MatrixMN<N, R2, C2>
    where
        DefaultAllocator: Reallocator<N, R, C, R2, C2>,
    {
        let (nrows, ncols) = self.shape();
        let mut data = self.data.into_owned();

        if new_nrows.value() == nrows {
            let res = unsafe { DefaultAllocator::reallocate_copy(new_nrows, new_ncols, data) };
            let mut res = Matrix::from_data(res);
            if new_ncols.value() > ncols {
                res.columns_range_mut(ncols..).fill(val);
            }

            res
        } else {
            let mut res;

            unsafe {
                if new_nrows.value() < nrows {
                    compress_rows(
                        &mut data.as_mut_slice(),
                        nrows,
                        ncols,
                        new_nrows.value(),
                        nrows - new_nrows.value(),
                    );
                    res = Matrix::from_data(DefaultAllocator::reallocate_copy(
                        new_nrows, new_ncols, data,
                    ));
                } else {
                    res = Matrix::from_data(DefaultAllocator::reallocate_copy(
                        new_nrows, new_ncols, data,
                    ));
                    extend_rows(
                        &mut res.data.as_mut_slice(),
                        nrows,
                        new_ncols.value(),
                        nrows,
                        new_nrows.value() - nrows,
                    );
                }
            }

            if new_ncols.value() > ncols {
                res.columns_range_mut(ncols..).fill(val);
            }

            if new_nrows.value() > nrows {
                res.slice_range_mut(nrows.., ..cmp::min(ncols, new_ncols.value()))
                    .fill(val);
            }

            res
        }
    }
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl<N: Scalar> DMatrix<N> {
    /// Resizes this matrix in-place.
    ///
    /// The values are copied such that `self[(i, j)] == result[(i, j)]`. If the result has more
    /// rows and/or columns than `self`, then the extra rows or columns are filled with `val`.
    ///
    /// Defined only for owned fully-dynamic matrices, i.e., `DMatrix`.
    pub fn resize_mut(&mut self, new_nrows: usize, new_ncols: usize, val: N)
    where DefaultAllocator: Reallocator<N, Dynamic, Dynamic, Dynamic, Dynamic> {
        let placeholder = unsafe { Self::new_uninitialized(0, 0) };
        let old = mem::replace(self, placeholder);
        let new = old.resize(new_nrows, new_ncols, val);
        let _ = mem::replace(self, new);
    }
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl<N: Scalar, C: Dim> MatrixMN<N, Dynamic, C>
where DefaultAllocator: Allocator<N, Dynamic, C>
{
    /// Changes the number of rows of this matrix in-place.
    ///
    /// The values are copied such that `self[(i, j)] == result[(i, j)]`. If the result has more
    /// rows than `self`, then the extra rows are filled with `val`.
    ///
    /// Defined only for owned matrices with a dynamic number of rows (for example, `DVector`).
    #[cfg(any(feature = "std", feature = "alloc"))]
    pub fn resize_vertically_mut(&mut self, new_nrows: usize, val: N)
    where DefaultAllocator: Reallocator<N, Dynamic, C, Dynamic, C> {
        let placeholder =
            unsafe { Self::new_uninitialized_generic(Dynamic::new(0), self.data.shape().1) };
        let old = mem::replace(self, placeholder);
        let new = old.resize_vertically(new_nrows, val);
        let _ = mem::replace(self, new);
    }
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl<N: Scalar, R: Dim> MatrixMN<N, R, Dynamic>
where DefaultAllocator: Allocator<N, R, Dynamic>
{
    /// Changes the number of column of this matrix in-place.
    ///
    /// The values are copied such that `self[(i, j)] == result[(i, j)]`. If the result has more
    /// columns than `self`, then the extra columns are filled with `val`.
    ///
    /// Defined only for owned matrices with a dynamic number of columns (for example, `DVector`).
    #[cfg(any(feature = "std", feature = "alloc"))]
    pub fn resize_horizontally_mut(&mut self, new_ncols: usize, val: N)
    where DefaultAllocator: Reallocator<N, R, Dynamic, R, Dynamic> {
        let placeholder =
            unsafe { Self::new_uninitialized_generic(self.data.shape().0, Dynamic::new(0)) };
        let old = mem::replace(self, placeholder);
        let new = old.resize_horizontally(new_ncols, val);
        let _ = mem::replace(self, new);
    }
}

unsafe fn compress_rows<N: Scalar>(
    data: &mut [N],
    nrows: usize,
    ncols: usize,
    i: usize,
    nremove: usize,
)
{
    let new_nrows = nrows - nremove;

    if new_nrows == 0 || ncols == 0 {
        return; // Nothing to do as the output matrix is empty.
    }

    let ptr_in = data.as_ptr();
    let ptr_out = data.as_mut_ptr();

    let mut curr_i = i;

    for k in 0..ncols - 1 {
        ptr::copy(
            ptr_in.offset((curr_i + (k + 1) * nremove) as isize),
            ptr_out.offset(curr_i as isize),
            new_nrows,
        );

        curr_i += new_nrows;
    }

    // Deal with the last column from which less values have to be copied.
    let remaining_len = nrows - i - nremove;
    ptr::copy(
        ptr_in.offset((nrows * ncols - remaining_len) as isize),
        ptr_out.offset(curr_i as isize),
        remaining_len,
    );
}

// Moves entries of a matrix buffer to make place for `ninsert` emty rows starting at the `i-th` row index.
// The `data` buffer is assumed to contained at least `(nrows + ninsert) * ncols` elements.
unsafe fn extend_rows<N: Scalar>(
    data: &mut [N],
    nrows: usize,
    ncols: usize,
    i: usize,
    ninsert: usize,
)
{
    let new_nrows = nrows + ninsert;

    if new_nrows == 0 || ncols == 0 {
        return; // Nothing to do as the output matrix is empty.
    }

    let ptr_in = data.as_ptr();
    let ptr_out = data.as_mut_ptr();

    let remaining_len = nrows - i;
    let mut curr_i = new_nrows * ncols - remaining_len;

    // Deal with the last column from which less values have to be copied.
    ptr::copy(
        ptr_in.offset((nrows * ncols - remaining_len) as isize),
        ptr_out.offset(curr_i as isize),
        remaining_len,
    );

    for k in (0..ncols - 1).rev() {
        curr_i -= new_nrows;

        ptr::copy(
            ptr_in.offset((k * nrows + i) as isize),
            ptr_out.offset(curr_i as isize),
            nrows,
        );
    }
}

/// Extend the number of columns of the `Matrix` with elements from
/// a given iterator.
#[cfg(any(feature = "std", feature = "alloc"))]
impl<N, R, S> Extend<N> for Matrix<N, R, Dynamic, S>
where
    N: Scalar,
    R: Dim,
    S: Extend<N>,
{
    /// Extend the number of columns of the `Matrix` with elements
    /// from the given iterator.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{DMatrix, Dynamic, Matrix, MatrixMN, Matrix3};
    ///
    /// let data = vec![0, 1, 2,      // column 1
    ///                 3, 4, 5];     // column 2
    ///
    /// let mut matrix = DMatrix::from_vec(3, 2, data);
    ///
    /// matrix.extend(vec![6, 7, 8]); // column 3
    ///
    /// assert!(matrix.eq(&Matrix3::new(0, 3, 6,
    ///                                 1, 4, 7,
    ///                                 2, 5, 8)));
    /// ```
    ///
    /// # Panics
    /// This function panics if the number of elements yielded by the
    /// given iterator is not a multiple of the number of rows of the
    /// `Matrix`.
    ///
    /// ```should_panic
    /// # use nalgebra::{DMatrix, Dynamic, MatrixMN};
    /// let data = vec![0, 1, 2,  // column 1
    ///                 3, 4, 5]; // column 2
    ///
    /// let mut matrix = DMatrix::from_vec(3, 2, data);
    ///
    /// // The following panics because the vec length is not a multiple of 3.
    /// matrix.extend(vec![6, 7, 8, 9]);
    /// ```
    fn extend<I: IntoIterator<Item = N>>(&mut self, iter: I) {
        self.data.extend(iter);
    }
}

/// Extend the number of rows of the `Vector` with elements from
/// a given iterator.
#[cfg(any(feature = "std", feature = "alloc"))]
impl<N, S> Extend<N> for Matrix<N, Dynamic, U1, S>
where
    N: Scalar,
    S: Extend<N>,
{
    /// Extend the number of rows of a `Vector` with elements
    /// from the given iterator.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::DVector;
    /// let mut vector = DVector::from_vec(vec![0, 1, 2]);
    /// vector.extend(vec![3, 4, 5]);
    /// assert!(vector.eq(&DVector::from_vec(vec![0, 1, 2, 3, 4, 5])));
    /// ```
    fn extend<I: IntoIterator<Item = N>>(&mut self, iter: I) {
        self.data.extend(iter);
    }
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl<N, R, S, RV, SV> Extend<Vector<N, RV, SV>> for Matrix<N, R, Dynamic, S>
where
    N: Scalar,
    R: Dim,
    S: Extend<Vector<N, RV, SV>>,
    RV: Dim,
    SV: Storage<N, RV>,
    ShapeConstraint: SameNumberOfRows<R, RV>,
{
    /// Extends the number of columns of a `Matrix` with `Vector`s
    /// from a given iterator.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{DMatrix, Vector3, Matrix3x4};
    ///
    /// let data = vec![0, 1, 2,          // column 1
    ///                 3, 4, 5];         // column 2
    ///
    /// let mut matrix = DMatrix::from_vec(3, 2, data);
    ///
    /// matrix.extend(
    ///   vec![Vector3::new(6,  7,  8),   // column 3
    ///        Vector3::new(9, 10, 11)]); // column 4
    ///
    /// assert!(matrix.eq(&Matrix3x4::new(0, 3, 6,  9,
    ///                                   1, 4, 7, 10,
    ///                                   2, 5, 8, 11)));
    /// ```
    ///
    /// # Panics
    /// This function panics if the dimension of each `Vector` yielded
    /// by the given iterator is not equal to the number of rows of
    /// this `Matrix`.
    ///
    /// ```should_panic
    /// # use nalgebra::{DMatrix, Vector2, Matrix3x4};
    /// let mut matrix =
    ///   DMatrix::from_vec(3, 2,
    ///                     vec![0, 1, 2,   // column 1
    ///                          3, 4, 5]); // column 2
    ///
    /// // The following panics because this matrix can only be extended with 3-dimensional vectors.
    /// matrix.extend(
    ///   vec![Vector2::new(6,  7)]); // too few dimensions!
    /// ```
    ///
    /// ```should_panic
    /// # use nalgebra::{DMatrix, Vector4, Matrix3x4};
    /// let mut matrix =
    ///   DMatrix::from_vec(3, 2,
    ///                     vec![0, 1, 2,   // column 1
    ///                          3, 4, 5]); // column 2
    ///
    /// // The following panics because this matrix can only be extended with 3-dimensional vectors.
    /// matrix.extend(
    ///   vec![Vector4::new(6, 7, 8, 9)]); // too few dimensions!
    /// ```
    fn extend<I: IntoIterator<Item = Vector<N, RV, SV>>>(&mut self, iter: I) {
        self.data.extend(iter);
    }
}
