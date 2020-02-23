use num::{One, Zero};
#[cfg(feature = "abomonation-serialize")]
use std::io::{Result as IOResult, Write};

use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use std::any::TypeId;
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::mem;

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[cfg(feature = "abomonation-serialize")]
use abomonation::Abomonation;

use alga::general::{ClosedAdd, ClosedMul, ClosedSub, RealField, Ring, ComplexField, Field};

use crate::base::allocator::{Allocator, SameShapeAllocator, SameShapeC, SameShapeR};
use crate::base::constraint::{DimEq, SameNumberOfColumns, SameNumberOfRows, ShapeConstraint};
use crate::base::dimension::{Dim, DimAdd, DimSum, IsNotStaticOne, U1, U2, U3};
use crate::base::iter::{MatrixIter, MatrixIterMut, RowIter, RowIterMut, ColumnIter, ColumnIterMut};
use crate::base::storage::{
    ContiguousStorage, ContiguousStorageMut, Owned, SameShapeStorage, Storage, StorageMut,
};
use crate::base::{DefaultAllocator, MatrixMN, MatrixN, Scalar, Unit, VectorN};

/// A square matrix.
pub type SquareMatrix<N, D, S> = Matrix<N, D, D, S>;

/// A matrix with one column and `D` rows.
pub type Vector<N, D, S> = Matrix<N, D, U1, S>;

/// A matrix with one row and `D` columns .
pub type RowVector<N, D, S> = Matrix<N, U1, D, S>;

/// The type of the result of a matrix sum.
pub type MatrixSum<N, R1, C1, R2, C2> =
    Matrix<N, SameShapeR<R1, R2>, SameShapeC<C1, C2>, SameShapeStorage<N, R1, C1, R2, C2>>;

/// The type of the result of a matrix sum.
pub type VectorSum<N, R1, R2> =
    Matrix<N, SameShapeR<R1, R2>, U1, SameShapeStorage<N, R1, U1, R2, U1>>;

/// The type of the result of a matrix cross product.
pub type MatrixCross<N, R1, C1, R2, C2> =
    Matrix<N, SameShapeR<R1, R2>, SameShapeC<C1, C2>, SameShapeStorage<N, R1, C1, R2, C2>>;

/// The most generic column-major matrix (and vector) type.
///
/// It combines four type parameters:
/// - `N`: for the matrix components scalar type.
/// - `R`: for the matrix number of rows.
/// - `C`: for the matrix number of columns.
/// - `S`: for the matrix data storage, i.e., the buffer that actually contains the matrix
/// components.
///
/// The matrix dimensions parameters `R` and `C` can either be:
/// - type-level unsigned integer constants (e.g. `U1`, `U124`) from the `nalgebra::` root module.
/// All numbers from 0 to 127 are defined that way.
/// - type-level unsigned integer constants (e.g. `U1024`, `U10000`) from the `typenum::` crate.
/// Using those, you will not get error messages as nice as for numbers smaller than 128 defined on
/// the `nalgebra::` module.
/// - the special value `Dynamic` from the `nalgebra::` root module. This indicates that the
/// specified dimension is not known at compile-time. Note that this will generally imply that the
/// matrix data storage `S` performs a dynamic allocation and contains extra metadata for the
/// matrix shape.
///
/// Note that mixing `Dynamic` with type-level unsigned integers is allowed. Actually, a
/// dynamically-sized column vector should be represented as a `Matrix<N, Dynamic, U1, S>` (given
/// some concrete types for `N` and a compatible data storage type `S`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Matrix<N: Scalar, R: Dim, C: Dim, S> {
    /// The data storage that contains all the matrix components and informations about its number
    /// of rows and column (if needed).
    pub data: S,

    _phantoms: PhantomData<(N, R, C)>,
}

impl<N: Scalar, R: Dim, C: Dim, S: fmt::Debug> fmt::Debug for Matrix<N, R, C, S> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        formatter
            .debug_struct("Matrix")
            .field("data", &self.data)
            .finish()
    }
}

#[cfg(feature = "serde-serialize")]
impl<N, R, C, S> Serialize for Matrix<N, R, C, S>
where
    N: Scalar,
    R: Dim,
    C: Dim,
    S: Serialize,
{
    fn serialize<T>(&self, serializer: T) -> Result<T::Ok, T::Error>
    where T: Serializer {
        self.data.serialize(serializer)
    }
}

#[cfg(feature = "serde-serialize")]
impl<'de, N, R, C, S> Deserialize<'de> for Matrix<N, R, C, S>
where
    N: Scalar,
    R: Dim,
    C: Dim,
    S: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: Deserializer<'de> {
        S::deserialize(deserializer).map(|x| Matrix {
            data: x,
            _phantoms: PhantomData,
        })
    }
}

#[cfg(feature = "abomonation-serialize")]
impl<N: Scalar, R: Dim, C: Dim, S: Abomonation> Abomonation for Matrix<N, R, C, S> {
    unsafe fn entomb<W: Write>(&self, writer: &mut W) -> IOResult<()> {
        self.data.entomb(writer)
    }

    unsafe fn exhume<'a, 'b>(&'a mut self, bytes: &'b mut [u8]) -> Option<&'b mut [u8]> {
        self.data.exhume(bytes)
    }

    fn extent(&self) -> usize {
        self.data.extent()
    }
}

impl<N: Scalar, R: Dim, C: Dim, S> Matrix<N, R, C, S> {
    /// Creates a new matrix with the given data without statically checking that the matrix
    /// dimension matches the storage dimension.
    #[inline]
    pub unsafe fn from_data_statically_unchecked(data: S) -> Matrix<N, R, C, S> {
        Matrix {
            data: data,
            _phantoms: PhantomData,
        }
    }
}

impl<N: Scalar, R: Dim, C: Dim, S: Storage<N, R, C>> Matrix<N, R, C, S> {
    /// Creates a new matrix with the given data.
    #[inline]
    pub fn from_data(data: S) -> Self {
        unsafe { Self::from_data_statically_unchecked(data) }
    }

    /// The total number of elements of this matrix.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::Matrix3x4;
    /// let mat = Matrix3x4::<f32>::zeros();
    /// assert_eq!(mat.len(), 12);
    #[inline]
    pub fn len(&self) -> usize {
        let (nrows, ncols) = self.shape();
        nrows * ncols
    }

    /// The shape of this matrix returned as the tuple (number of rows, number of columns).
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::Matrix3x4;
    /// let mat = Matrix3x4::<f32>::zeros();
    /// assert_eq!(mat.shape(), (3, 4));
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        let (nrows, ncols) = self.data.shape();
        (nrows.value(), ncols.value())
    }

    /// The number of rows of this matrix.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::Matrix3x4;
    /// let mat = Matrix3x4::<f32>::zeros();
    /// assert_eq!(mat.nrows(), 3);
    #[inline]
    pub fn nrows(&self) -> usize {
        self.shape().0
    }

    /// The number of columns of this matrix.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::Matrix3x4;
    /// let mat = Matrix3x4::<f32>::zeros();
    /// assert_eq!(mat.ncols(), 4);
    #[inline]
    pub fn ncols(&self) -> usize {
        self.shape().1
    }

    /// The strides (row stride, column stride) of this matrix.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::DMatrix;
    /// let mat = DMatrix::<f32>::zeros(10, 10);
    /// let slice = mat.slice_with_steps((0, 0), (5, 3), (1, 2));
    /// // The column strides is the number of steps (here 2) multiplied by the corresponding dimension.
    /// assert_eq!(mat.strides(), (1, 10));
    #[inline]
    pub fn strides(&self) -> (usize, usize) {
        let (srows, scols) = self.data.strides();
        (srows.value(), scols.value())
    }

    /// Iterates through this matrix coordinates in column-major order.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// let mat = Matrix2x3::new(11, 12, 13,
    ///                          21, 22, 23);
    /// let mut it = mat.iter();
    /// assert_eq!(*it.next().unwrap(), 11);
    /// assert_eq!(*it.next().unwrap(), 21);
    /// assert_eq!(*it.next().unwrap(), 12);
    /// assert_eq!(*it.next().unwrap(), 22);
    /// assert_eq!(*it.next().unwrap(), 13);
    /// assert_eq!(*it.next().unwrap(), 23);
    /// assert!(it.next().is_none());
    #[inline]
    pub fn iter(&self) -> MatrixIter<N, R, C, S> {
        MatrixIter::new(&self.data)
    }

    /// Iterate through the rows of this matrix.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// let mut a = Matrix2x3::new(1, 2, 3,
    ///                            4, 5, 6);
    /// for (i, row) in a.row_iter().enumerate() {
    ///     assert_eq!(row, a.row(i))
    /// }
    /// ```
    #[inline]
    pub fn row_iter(&self) -> RowIter<N, R, C, S> {
        RowIter::new(self)
    }

    /// Iterate through the columns of this matrix.
    /// # Example
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// let mut a = Matrix2x3::new(1, 2, 3,
    ///                            4, 5, 6);
    /// for (i, column) in a.column_iter().enumerate() {
    ///     assert_eq!(column, a.column(i))
    /// }
    /// ```
    #[inline]
    pub fn column_iter(&self) -> ColumnIter<N, R, C, S> {
        ColumnIter::new(self)
    }

    /// Computes the row and column coordinates of the i-th element of this matrix seen as a
    /// vector.
    #[inline]
    pub fn vector_to_matrix_index(&self, i: usize) -> (usize, usize) {
        let (nrows, ncols) = self.shape();

        // Two most common uses that should be optimized by the compiler for statically-sized
        // matrices.
        if nrows == 1 {
            (0, i)
        } else if ncols == 1 {
            (i, 0)
        } else {
            (i % nrows, i / nrows)
        }
    }

    /// Returns a pointer to the start of the matrix.
    ///
    /// If the matrix is not empty, this pointer is guaranteed to be aligned
    /// and non-null.
    #[inline]
    pub fn as_ptr(&self) -> *const N {
        self.data.ptr()
    }

    /// Tests whether `self` and `rhs` are equal up to a given epsilon.
    ///
    /// See `relative_eq` from the `RelativeEq` trait for more details.
    #[inline]
    pub fn relative_eq<R2, C2, SB>(
        &self,
        other: &Matrix<N, R2, C2, SB>,
        eps: N::Epsilon,
        max_relative: N::Epsilon,
    ) -> bool
    where
        N: RelativeEq,
        R2: Dim,
        C2: Dim,
        SB: Storage<N, R2, C2>,
        N::Epsilon: Copy,
        ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2>,
    {
        assert!(self.shape() == other.shape());
        self.iter()
            .zip(other.iter())
            .all(|(a, b)| a.relative_eq(b, eps, max_relative))
    }

    /// Tests whether `self` and `rhs` are exactly equal.
    #[inline]
    pub fn eq<R2, C2, SB>(&self, other: &Matrix<N, R2, C2, SB>) -> bool
    where
        N: PartialEq,
        R2: Dim,
        C2: Dim,
        SB: Storage<N, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2>,
    {
        assert!(self.shape() == other.shape());
        self.iter().zip(other.iter()).all(|(a, b)| *a == *b)
    }

    /// Moves this matrix into one that owns its data.
    #[inline]
    pub fn into_owned(self) -> MatrixMN<N, R, C>
    where DefaultAllocator: Allocator<N, R, C> {
        Matrix::from_data(self.data.into_owned())
    }

    // FIXME: this could probably benefit from specialization.
    // XXX: bad name.
    /// Moves this matrix into one that owns its data. The actual type of the result depends on
    /// matrix storage combination rules for addition.
    #[inline]
    pub fn into_owned_sum<R2, C2>(self) -> MatrixSum<N, R, C, R2, C2>
    where
        R2: Dim,
        C2: Dim,
        DefaultAllocator: SameShapeAllocator<N, R, C, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2>,
    {
        if TypeId::of::<SameShapeStorage<N, R, C, R2, C2>>() == TypeId::of::<Owned<N, R, C>>() {
            // We can just return `self.into_owned()`.

            unsafe {
                // FIXME: check that those copies are optimized away by the compiler.
                let owned = self.into_owned();
                let res = mem::transmute_copy(&owned);
                mem::forget(owned);
                res
            }
        } else {
            self.clone_owned_sum()
        }
    }

    /// Clones this matrix to one that owns its data.
    #[inline]
    pub fn clone_owned(&self) -> MatrixMN<N, R, C>
    where DefaultAllocator: Allocator<N, R, C> {
        Matrix::from_data(self.data.clone_owned())
    }

    /// Clones this matrix into one that owns its data. The actual type of the result depends on
    /// matrix storage combination rules for addition.
    #[inline]
    pub fn clone_owned_sum<R2, C2>(&self) -> MatrixSum<N, R, C, R2, C2>
    where
        R2: Dim,
        C2: Dim,
        DefaultAllocator: SameShapeAllocator<N, R, C, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2>,
    {
        let (nrows, ncols) = self.shape();
        let nrows: SameShapeR<R, R2> = Dim::from_usize(nrows);
        let ncols: SameShapeC<C, C2> = Dim::from_usize(ncols);

        let mut res: MatrixSum<N, R, C, R2, C2> =
            unsafe { Matrix::new_uninitialized_generic(nrows, ncols) };

        // FIXME: use copy_from
        for j in 0..res.ncols() {
            for i in 0..res.nrows() {
                unsafe {
                    *res.get_unchecked_mut((i, j)) = self.get_unchecked((i, j)).inlined_clone();
                }
            }
        }

        res
    }

    /// Returns a matrix containing the result of `f` applied to each of its entries.
    #[inline]
    pub fn map<N2: Scalar, F: FnMut(N) -> N2>(&self, mut f: F) -> MatrixMN<N2, R, C>
    where DefaultAllocator: Allocator<N2, R, C> {
        let (nrows, ncols) = self.data.shape();

        let mut res = unsafe { MatrixMN::new_uninitialized_generic(nrows, ncols) };

        for j in 0..ncols.value() {
            for i in 0..nrows.value() {
                unsafe {
                    let a = self.data.get_unchecked(i, j).inlined_clone();
                    *res.data.get_unchecked_mut(i, j) = f(a)
                }
            }
        }

        res
    }

    /// Returns a matrix containing the result of `f` applied to each of its entries. Unlike `map`,
    /// `f` also gets passed the row and column index, i.e. `f(row, col, value)`.
    #[inline]
    pub fn map_with_location<N2: Scalar, F: FnMut(usize, usize, N) -> N2>(
        &self,
        mut f: F,
    ) -> MatrixMN<N2, R, C>
    where
        DefaultAllocator: Allocator<N2, R, C>,
    {
        let (nrows, ncols) = self.data.shape();

        let mut res = unsafe { MatrixMN::new_uninitialized_generic(nrows, ncols) };

        for j in 0..ncols.value() {
            for i in 0..nrows.value() {
                unsafe {
                    let a = self.data.get_unchecked(i, j).inlined_clone();
                    *res.data.get_unchecked_mut(i, j) = f(i, j, a)
                }
            }
        }

        res
    }

    /// Returns a matrix containing the result of `f` applied to each entries of `self` and
    /// `rhs`.
    #[inline]
    pub fn zip_map<N2, N3, S2, F>(&self, rhs: &Matrix<N2, R, C, S2>, mut f: F) -> MatrixMN<N3, R, C>
    where
        N2: Scalar,
        N3: Scalar,
        S2: Storage<N2, R, C>,
        F: FnMut(N, N2) -> N3,
        DefaultAllocator: Allocator<N3, R, C>,
    {
        let (nrows, ncols) = self.data.shape();

        let mut res = unsafe { MatrixMN::new_uninitialized_generic(nrows, ncols) };

        assert!(
            (nrows.value(), ncols.value()) == rhs.shape(),
            "Matrix simultaneous traversal error: dimension mismatch."
        );

        for j in 0..ncols.value() {
            for i in 0..nrows.value() {
                unsafe {
                    let a = self.data.get_unchecked(i, j).inlined_clone();
                    let b = rhs.data.get_unchecked(i, j).inlined_clone();
                    *res.data.get_unchecked_mut(i, j) = f(a, b)
                }
            }
        }

        res
    }

    /// Returns a matrix containing the result of `f` applied to each entries of `self` and
    /// `b`, and `c`.
    #[inline]
    pub fn zip_zip_map<N2, N3, N4, S2, S3, F>(
        &self,
        b: &Matrix<N2, R, C, S2>,
        c: &Matrix<N3, R, C, S3>,
        mut f: F,
    ) -> MatrixMN<N4, R, C>
    where
        N2: Scalar,
        N3: Scalar,
        N4: Scalar,
        S2: Storage<N2, R, C>,
        S3: Storage<N3, R, C>,
        F: FnMut(N, N2, N3) -> N4,
        DefaultAllocator: Allocator<N4, R, C>,
    {
        let (nrows, ncols) = self.data.shape();

        let mut res = unsafe { MatrixMN::new_uninitialized_generic(nrows, ncols) };

        assert!(
            (nrows.value(), ncols.value()) == b.shape()
                && (nrows.value(), ncols.value()) == c.shape(),
            "Matrix simultaneous traversal error: dimension mismatch."
        );

        for j in 0..ncols.value() {
            for i in 0..nrows.value() {
                unsafe {
                    let a = self.data.get_unchecked(i, j).inlined_clone();
                    let b = b.data.get_unchecked(i, j).inlined_clone();
                    let c = c.data.get_unchecked(i, j).inlined_clone();
                    *res.data.get_unchecked_mut(i, j) = f(a, b, c)
                }
            }
        }

        res
    }

    /// Folds a function `f` on each entry of `self`.
    #[inline]
    pub fn fold<Acc>(&self, init: Acc, mut f: impl FnMut(Acc, N) -> Acc) -> Acc {
        let (nrows, ncols) = self.data.shape();

        let mut res = init;

        for j in 0..ncols.value() {
            for i in 0..nrows.value() {
                unsafe {
                    let a = self.data.get_unchecked(i, j).inlined_clone();
                    res = f(res, a)
                }
            }
        }

        res
    }

    /// Folds a function `f` on each pairs of entries from `self` and `rhs`.
    #[inline]
    pub fn zip_fold<N2, R2, C2, S2, Acc>(&self, rhs: &Matrix<N2, R2, C2, S2>, init: Acc, mut f: impl FnMut(Acc, N, N2) -> Acc) -> Acc
        where
            N2: Scalar,
            R2: Dim,
            C2: Dim,
            S2: Storage<N2, R2, C2>,
            ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2>
    {
        let (nrows, ncols) = self.data.shape();

        let mut res = init;

        assert!(
            (nrows.value(), ncols.value()) == rhs.shape(),
            "Matrix simultaneous traversal error: dimension mismatch."
        );

        for j in 0..ncols.value() {
            for i in 0..nrows.value() {
                unsafe {
                    let a = self.data.get_unchecked(i, j).inlined_clone();
                    let b = rhs.data.get_unchecked(i, j).inlined_clone();
                    res = f(res, a, b)
                }
            }
        }

        res
    }

    /// Transposes `self` and store the result into `out`.
    #[inline]
    pub fn transpose_to<R2, C2, SB>(&self, out: &mut Matrix<N, R2, C2, SB>)
    where
        R2: Dim,
        C2: Dim,
        SB: StorageMut<N, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, C2> + SameNumberOfColumns<C, R2>,
    {
        let (nrows, ncols) = self.shape();
        assert!(
            (ncols, nrows) == out.shape(),
            "Incompatible shape for transpose-copy."
        );

        // FIXME: optimize that.
        for i in 0..nrows {
            for j in 0..ncols {
                unsafe {
                    *out.get_unchecked_mut((j, i)) = self.get_unchecked((i, j)).inlined_clone();
                }
            }
        }
    }

    /// Transposes `self`.
    #[inline]
    #[must_use = "Did you mean to use transpose_mut()?"]
    pub fn transpose(&self) -> MatrixMN<N, C, R>
    where DefaultAllocator: Allocator<N, C, R> {
        let (nrows, ncols) = self.data.shape();

        unsafe {
            let mut res = Matrix::new_uninitialized_generic(ncols, nrows);
            self.transpose_to(&mut res);

            res
        }
    }
}

impl<N: Scalar, R: Dim, C: Dim, S: StorageMut<N, R, C>> Matrix<N, R, C, S> {
    /// Mutably iterates through this matrix coordinates.
    #[inline]
    pub fn iter_mut(&mut self) -> MatrixIterMut<N, R, C, S> {
        MatrixIterMut::new(&mut self.data)
    }

    /// Returns a mutable pointer to the start of the matrix.
    ///
    /// If the matrix is not empty, this pointer is guaranteed to be aligned
    /// and non-null.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut N {
        self.data.ptr_mut()
    }

    /// Mutably iterates through this matrix rows.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// let mut a = Matrix2x3::new(1, 2, 3,
    ///                            4, 5, 6);
    /// for (i, mut row) in a.row_iter_mut().enumerate() {
    ///     row *= (i + 1) * 10;
    /// }
    ///
    /// let expected = Matrix2x3::new(10, 20, 30,
    ///                               80, 100, 120);
    /// assert_eq!(a, expected);
    /// ```
    #[inline]
    pub fn row_iter_mut(&mut self) -> RowIterMut<N, R, C, S> {
        RowIterMut::new(self)
    }

    /// Mutably iterates through this matrix columns.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// let mut a = Matrix2x3::new(1, 2, 3,
    ///                            4, 5, 6);
    /// for (i, mut col) in a.column_iter_mut().enumerate() {
    ///     col *= (i + 1) * 10;
    /// }
    ///
    /// let expected = Matrix2x3::new(10, 40, 90,
    ///                               40, 100, 180);
    /// assert_eq!(a, expected);
    /// ```
    #[inline]
    pub fn column_iter_mut(&mut self) -> ColumnIterMut<N, R, C, S> {
        ColumnIterMut::new(self)
    }

    /// Swaps two entries without bound-checking.
    #[inline]
    pub unsafe fn swap_unchecked(&mut self, row_cols1: (usize, usize), row_cols2: (usize, usize)) {
        debug_assert!(row_cols1.0 < self.nrows() && row_cols1.1 < self.ncols());
        debug_assert!(row_cols2.0 < self.nrows() && row_cols2.1 < self.ncols());
        self.data.swap_unchecked(row_cols1, row_cols2)
    }

    /// Swaps two entries.
    #[inline]
    pub fn swap(&mut self, row_cols1: (usize, usize), row_cols2: (usize, usize)) {
        let (nrows, ncols) = self.shape();
        assert!(
            row_cols1.0 < nrows && row_cols1.1 < ncols,
            "Matrix elements swap index out of bounds."
        );
        assert!(
            row_cols2.0 < nrows && row_cols2.1 < ncols,
            "Matrix elements swap index out of bounds."
        );
        unsafe { self.swap_unchecked(row_cols1, row_cols2) }
    }

    /// Fills this matrix with the content of a slice. Both must hold the same number of elements.
    ///
    /// The components of the slice are assumed to be ordered in column-major order.
    #[inline]
    pub fn copy_from_slice(&mut self, slice: &[N]) {
        let (nrows, ncols) = self.shape();

        assert!(
            nrows * ncols == slice.len(),
            "The slice must contain the same number of elements as the matrix."
        );

        for j in 0..ncols {
            for i in 0..nrows {
                unsafe {
                    *self.get_unchecked_mut((i, j)) = slice.get_unchecked(i + j * nrows).inlined_clone();
                }
            }
        }
    }

    /// Fills this matrix with the content of another one. Both must have the same shape.
    #[inline]
    pub fn copy_from<R2, C2, SB>(&mut self, other: &Matrix<N, R2, C2, SB>)
    where
        R2: Dim,
        C2: Dim,
        SB: Storage<N, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2>,
    {
        assert!(
            self.shape() == other.shape(),
            "Unable to copy from a matrix with a different shape."
        );

        for j in 0..self.ncols() {
            for i in 0..self.nrows() {
                unsafe {
                    *self.get_unchecked_mut((i, j)) = other.get_unchecked((i, j)).inlined_clone();
                }
            }
        }
    }

    /// Fills this matrix with the content of the transpose another one.
    #[inline]
    pub fn tr_copy_from<R2, C2, SB>(&mut self, other: &Matrix<N, R2, C2, SB>)
    where
        R2: Dim,
        C2: Dim,
        SB: Storage<N, R2, C2>,
        ShapeConstraint: DimEq<R, C2> + SameNumberOfColumns<C, R2>,
    {
        let (nrows, ncols) = self.shape();
        assert!(
            (ncols, nrows) == other.shape(),
            "Unable to copy from a matrix with incompatible shape."
        );

        for j in 0..ncols {
            for i in 0..nrows {
                unsafe {
                    *self.get_unchecked_mut((i, j)) = other.get_unchecked((j, i)).inlined_clone();
                }
            }
        }
    }

    // FIXME: rename `apply` to `apply_mut` and `apply_into` to `apply`?
    /// Returns `self` with each of its components replaced by the result of a closure `f` applied on it.
    #[inline]
    pub fn apply_into<F: FnMut(N) -> N>(mut self, f: F) -> Self{
        self.apply(f);
        self
    }

    /// Replaces each component of `self` by the result of a closure `f` applied on it.
    #[inline]
    pub fn apply<F: FnMut(N) -> N>(&mut self, mut f: F) {
        let (nrows, ncols) = self.shape();

        for j in 0..ncols {
            for i in 0..nrows {
                unsafe {
                    let e = self.data.get_unchecked_mut(i, j);
                    *e = f(e.inlined_clone())
                }
            }
        }
    }

    /// Replaces each component of `self` by the result of a closure `f` applied on its components
    /// joined with the components from `rhs`.
    #[inline]
    pub fn zip_apply<N2, R2, C2, S2>(&mut self, rhs: &Matrix<N2, R2, C2, S2>, mut f: impl FnMut(N, N2) -> N)
        where N2: Scalar,
              R2: Dim,
              C2: Dim,
              S2: Storage<N2, R2, C2>,
              ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2> {
        let (nrows, ncols) = self.shape();

        assert!(
            (nrows, ncols) == rhs.shape(),
            "Matrix simultaneous traversal error: dimension mismatch."
        );

        for j in 0..ncols {
            for i in 0..nrows {
                unsafe {
                    let e = self.data.get_unchecked_mut(i, j);
                    let rhs = rhs.get_unchecked((i, j)).inlined_clone();
                    *e = f(e.inlined_clone(), rhs)
                }
            }
        }
    }


    /// Replaces each component of `self` by the result of a closure `f` applied on its components
    /// joined with the components from `b` and `c`.
    #[inline]
    pub fn zip_zip_apply<N2, R2, C2, S2, N3, R3, C3, S3>(&mut self, b: &Matrix<N2, R2, C2, S2>, c: &Matrix<N3, R3, C3, S3>, mut f: impl FnMut(N, N2, N3) -> N)
        where N2: Scalar,
              R2: Dim,
              C2: Dim,
              S2: Storage<N2, R2, C2>,
              N3: Scalar,
              R3: Dim,
              C3: Dim,
              S3: Storage<N3, R3, C3>,
              ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2>,
              ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2> {
        let (nrows, ncols) = self.shape();

        assert!(
            (nrows, ncols) == b.shape(),
            "Matrix simultaneous traversal error: dimension mismatch."
        );
        assert!(
            (nrows, ncols) == c.shape(),
            "Matrix simultaneous traversal error: dimension mismatch."
        );

        for j in 0..ncols {
            for i in 0..nrows {
                unsafe {
                    let e = self.data.get_unchecked_mut(i, j);
                    let b = b.get_unchecked((i, j)).inlined_clone();
                    let c = c.get_unchecked((i, j)).inlined_clone();
                    *e = f(e.inlined_clone(), b, c)
                }
            }
        }
    }
}

impl<N: Scalar, D: Dim, S: Storage<N, D>> Vector<N, D, S> {
    /// Gets a reference to the i-th element of this column vector without bound checking.
    #[inline]
    pub unsafe fn vget_unchecked(&self, i: usize) -> &N {
        debug_assert!(i < self.nrows(), "Vector index out of bounds.");
        let i = i * self.strides().0;
        self.data.get_unchecked_linear(i)
    }
}

impl<N: Scalar, D: Dim, S: StorageMut<N, D>> Vector<N, D, S> {
    /// Gets a mutable reference to the i-th element of this column vector without bound checking.
    #[inline]
    pub unsafe fn vget_unchecked_mut(&mut self, i: usize) -> &mut N {
        debug_assert!(i < self.nrows(), "Vector index out of bounds.");
        let i = i * self.strides().0;
        self.data.get_unchecked_linear_mut(i)
    }
}

impl<N: Scalar, R: Dim, C: Dim, S: ContiguousStorage<N, R, C>> Matrix<N, R, C, S> {
    /// Extracts a slice containing the entire matrix entries ordered column-by-columns.
    #[inline]
    pub fn as_slice(&self) -> &[N] {
        self.data.as_slice()
    }
}

impl<N: Scalar, R: Dim, C: Dim, S: ContiguousStorageMut<N, R, C>> Matrix<N, R, C, S> {
    /// Extracts a mutable slice containing the entire matrix entries ordered column-by-columns.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [N] {
        self.data.as_mut_slice()
    }
}

impl<N: Scalar, D: Dim, S: StorageMut<N, D, D>> Matrix<N, D, D, S> {
    /// Transposes the square matrix `self` in-place.
    pub fn transpose_mut(&mut self) {
        assert!(
            self.is_square(),
            "Unable to transpose a non-square matrix in-place."
        );

        let dim = self.shape().0;

        for i in 1..dim {
            for j in 0..i {
                unsafe { self.swap_unchecked((i, j), (j, i)) }
            }
        }
    }
}

impl<N: ComplexField, R: Dim, C: Dim, S: Storage<N, R, C>> Matrix<N, R, C, S> {
    /// Takes the adjoint (aka. conjugate-transpose) of `self` and store the result into `out`.
    #[inline]
    pub fn adjoint_to<R2, C2, SB>(&self, out: &mut Matrix<N, R2, C2, SB>)
    where
        R2: Dim,
        C2: Dim,
        SB: StorageMut<N, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, C2> + SameNumberOfColumns<C, R2>,
    {
        let (nrows, ncols) = self.shape();
        assert!(
            (ncols, nrows) == out.shape(),
            "Incompatible shape for transpose-copy."
        );

        // FIXME: optimize that.
        for i in 0..nrows {
            for j in 0..ncols {
                unsafe {
                    *out.get_unchecked_mut((j, i)) = self.get_unchecked((i, j)).conjugate();
                }
            }
        }
    }

    /// The adjoint (aka. conjugate-transpose) of `self`.
    #[inline]
    #[must_use = "Did you mean to use adjoint_mut()?"]
    pub fn adjoint(&self) -> MatrixMN<N, C, R>
    where DefaultAllocator: Allocator<N, C, R> {
        let (nrows, ncols) = self.data.shape();

        unsafe {
            let mut res: MatrixMN<_, C, R> = Matrix::new_uninitialized_generic(ncols, nrows);
            self.adjoint_to(&mut res);

            res
        }
    }

    /// Takes the conjugate and transposes `self` and store the result into `out`.
    #[deprecated(note = "Renamed `self.adjoint_to(out)`.")]
    #[inline]
    pub fn conjugate_transpose_to<R2, C2, SB>(&self, out: &mut Matrix<N, R2, C2, SB>)
        where
            R2: Dim,
            C2: Dim,
            SB: StorageMut<N, R2, C2>,
            ShapeConstraint: SameNumberOfRows<R, C2> + SameNumberOfColumns<C, R2>,
    {
        self.adjoint_to(out)
    }

    /// The conjugate transposition of `self`.
    #[deprecated(note = "Renamed `self.adjoint()`.")]
    #[inline]
    pub fn conjugate_transpose(&self) -> MatrixMN<N, C, R>
        where DefaultAllocator: Allocator<N, C, R> {
        self.adjoint()
    }

    /// The conjugate of `self`.
    #[inline]
    #[must_use = "Did you mean to use conjugate_mut()?"]
    pub fn conjugate(&self) -> MatrixMN<N, R, C>
        where DefaultAllocator: Allocator<N, R, C> {
        self.map(|e| e.conjugate())
    }

    /// Divides each component of the complex matrix `self` by the given real.
    #[inline]
    #[must_use = "Did you mean to use unscale_mut()?"]
    pub fn unscale(&self, real: N::RealField) -> MatrixMN<N, R, C>
        where DefaultAllocator: Allocator<N, R, C> {
        self.map(|e| e.unscale(real))
    }

    /// Multiplies each component of the complex matrix `self` by the given real.
    #[inline]
    #[must_use = "Did you mean to use scale_mut()?"]
    pub fn scale(&self, real: N::RealField) -> MatrixMN<N, R, C>
        where DefaultAllocator: Allocator<N, R, C> {
        self.map(|e| e.scale(real))
    }
}

impl<N: ComplexField, R: Dim, C: Dim, S: StorageMut<N, R, C>> Matrix<N, R, C, S> {
    /// The conjugate of the complex matrix `self` computed in-place.
    #[inline]
    pub fn conjugate_mut(&mut self) {
        self.apply(|e| e.conjugate())
    }

    /// Divides each component of the complex matrix `self` by the given real.
    #[inline]
    pub fn unscale_mut(&mut self, real: N::RealField) {
        self.apply(|e| e.unscale(real))
    }

    /// Multiplies each component of the complex matrix `self` by the given real.
    #[inline]
    pub fn scale_mut(&mut self, real: N::RealField) {
        self.apply(|e| e.scale(real))
    }
}

impl<N: ComplexField, D: Dim, S: StorageMut<N, D, D>> Matrix<N, D, D, S> {
    /// Sets `self` to its adjoint.
    #[deprecated(note = "Renamed to `self.adjoint_mut()`.")]
    pub fn conjugate_transform_mut(&mut self) {
        self.adjoint_mut()
    }

    /// Sets `self` to its adjoint (aka. conjugate-transpose).
    pub fn adjoint_mut(&mut self) {
        assert!(
            self.is_square(),
            "Unable to transpose a non-square matrix in-place."
        );

        let dim = self.shape().0;

        for i in 0..dim {
            for j in 0..i {
                unsafe {
                    let ref_ij = self.get_unchecked_mut((i, j)) as *mut N;
                    let ref_ji = self.get_unchecked_mut((j, i)) as *mut N;
                    let conj_ij = (*ref_ij).conjugate();
                    let conj_ji = (*ref_ji).conjugate();
                    *ref_ij = conj_ji;
                    *ref_ji = conj_ij;
                }
            }

            {
                let diag = unsafe { self.get_unchecked_mut((i, i)) };
                *diag = diag.conjugate();
            }
        }
    }
}

impl<N: Scalar, D: Dim, S: Storage<N, D, D>> SquareMatrix<N, D, S> {
    /// The diagonal of this matrix.
    #[inline]
    pub fn diagonal(&self) -> VectorN<N, D>
        where DefaultAllocator: Allocator<N, D> {
        self.map_diagonal(|e| e)
    }

    /// Apply the given function to this matrix's diagonal and returns it.
    ///
    /// This is a more efficient version of `self.diagonal().map(f)` since this
    /// allocates only once.
    pub fn map_diagonal<N2: Scalar>(&self, mut f: impl FnMut(N) -> N2) -> VectorN<N2, D>
        where DefaultAllocator: Allocator<N2, D> {
        assert!(
            self.is_square(),
            "Unable to get the diagonal of a non-square matrix."
        );

        let dim = self.data.shape().0;
        let mut res = unsafe { VectorN::new_uninitialized_generic(dim, U1) };

        for i in 0..dim.value() {
            unsafe {
                *res.vget_unchecked_mut(i) = f(self.get_unchecked((i, i)).inlined_clone());
            }
        }

        res
    }

    /// Computes a trace of a square matrix, i.e., the sum of its diagonal elements.
    #[inline]
    pub fn trace(&self) -> N
        where N: Ring {
        assert!(
            self.is_square(),
            "Cannot compute the trace of non-square matrix."
        );

        let dim = self.data.shape().0;
        let mut res = N::zero();

        for i in 0..dim.value() {
            res += unsafe { self.get_unchecked((i, i)).inlined_clone() };
        }

        res
    }
}

impl<N: ComplexField, D: Dim, S: Storage<N, D, D>> SquareMatrix<N, D, S> {
    /// The symmetric part of `self`, i.e., `0.5 * (self + self.transpose())`.
    #[inline]
    pub fn symmetric_part(&self) -> MatrixMN<N, D, D>
        where DefaultAllocator: Allocator<N, D, D> {
        assert!(self.is_square(), "Cannot compute the symmetric part of a non-square matrix.");
        let mut tr = self.transpose();
        tr += self;
        tr *= crate::convert::<_, N>(0.5);
        tr
    }

    /// The hermitian part of `self`, i.e., `0.5 * (self + self.adjoint())`.
    #[inline]
    pub fn hermitian_part(&self) -> MatrixMN<N, D, D>
        where DefaultAllocator: Allocator<N, D, D> {
        assert!(self.is_square(), "Cannot compute the hermitian part of a non-square matrix.");

        let mut tr = self.adjoint();
        tr += self;
        tr *= crate::convert::<_, N>(0.5);
        tr
    }
}

impl<N: Scalar + Zero + One, D: DimAdd<U1> + IsNotStaticOne, S: Storage<N, D, D>> Matrix<N, D, D, S> {

    /// Yields the homogeneous matrix for this matrix, i.e., appending an additional dimension and
    /// and setting the diagonal element to `1`.
    #[inline]
    pub fn to_homogeneous(&self) -> MatrixN<N, DimSum<D, U1>>
    where DefaultAllocator: Allocator<N, DimSum<D, U1>, DimSum<D, U1>> {
        assert!(self.is_square(), "Only square matrices can currently be transformed to homogeneous coordinates.");
        let dim = DimSum::<D, U1>::from_usize(self.nrows() + 1);
        let mut res = MatrixN::identity_generic(dim, dim);
        res.generic_slice_mut::<D, D>((0, 0), self.data.shape()).copy_from(&self);
        res
    }

}

impl<N: Scalar + Zero, D: DimAdd<U1>, S: Storage<N, D>> Vector<N, D, S> {
    /// Computes the coordinates in projective space of this vector, i.e., appends a `0` to its
    /// coordinates.
    #[inline]
    pub fn to_homogeneous(&self) -> VectorN<N, DimSum<D, U1>>
    where DefaultAllocator: Allocator<N, DimSum<D, U1>> {
        self.push(N::zero())
    }

    /// Constructs a vector from coordinates in projective space, i.e., removes a `0` at the end of
    /// `self`. Returns `None` if this last component is not zero.
    #[inline]
    pub fn from_homogeneous<SB>(v: Vector<N, DimSum<D, U1>, SB>) -> Option<VectorN<N, D>>
    where
        SB: Storage<N, DimSum<D, U1>>,
        DefaultAllocator: Allocator<N, D>,
    {
        if v[v.len() - 1].is_zero() {
            let nrows = D::from_usize(v.len() - 1);
            Some(v.generic_slice((0, 0), (nrows, U1)).into_owned())
        } else {
            None
        }
    }
}

impl<N: Scalar + Zero, D: DimAdd<U1>, S: Storage<N, D>> Vector<N, D, S> {
    /// Constructs a new vector of higher dimension by appending `element` to the end of `self`.
    #[inline]
    pub fn push(&self, element: N) -> VectorN<N, DimSum<D, U1>>
    where DefaultAllocator: Allocator<N, DimSum<D, U1>> {
        let len = self.len();
        let hnrows = DimSum::<D, U1>::from_usize(len + 1);
        let mut res = unsafe { VectorN::<N, _>::new_uninitialized_generic(hnrows, U1) };
        res.generic_slice_mut((0, 0), self.data.shape())
            .copy_from(self);
        res[(len, 0)] = element;

        res
    }
}

impl<N, R: Dim, C: Dim, S> AbsDiffEq for Matrix<N, R, C, S>
where
    N: Scalar + AbsDiffEq,
    S: Storage<N, R, C>,
    N::Epsilon: Copy,
{
    type Epsilon = N::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        N::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.iter()
            .zip(other.iter())
            .all(|(a, b)| a.abs_diff_eq(b, epsilon))
    }
}

impl<N, R: Dim, C: Dim, S> RelativeEq for Matrix<N, R, C, S>
where
    N: Scalar + RelativeEq,
    S: Storage<N, R, C>,
    N::Epsilon: Copy,
{
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        N::default_max_relative()
    }

    #[inline]
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool
    {
        self.relative_eq(other, epsilon, max_relative)
    }
}

impl<N, R: Dim, C: Dim, S> UlpsEq for Matrix<N, R, C, S>
where
    N: Scalar + UlpsEq,
    S: Storage<N, R, C>,
    N::Epsilon: Copy,
{
    #[inline]
    fn default_max_ulps() -> u32 {
        N::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        assert!(self.shape() == other.shape());
        self.iter()
            .zip(other.iter())
            .all(|(a, b)| a.ulps_eq(b, epsilon, max_ulps))
    }
}

impl<N, R: Dim, C: Dim, S> PartialOrd for Matrix<N, R, C, S>
where
    N: Scalar + PartialOrd,
    S: Storage<N, R, C>,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.shape() != other.shape() {
            return None;
        }

        if self.nrows() == 0 || self.ncols() == 0 {
            return Some(Ordering::Equal);
        }

        let mut first_ord = unsafe {
            self.data
                .get_unchecked_linear(0)
                .partial_cmp(other.data.get_unchecked_linear(0))
        };

        if let Some(first_ord) = first_ord.as_mut() {
            let mut it = self.iter().zip(other.iter());
            let _ = it.next(); // Drop the first elements (we already tested it).

            for (left, right) in it {
                if let Some(ord) = left.partial_cmp(right) {
                    match ord {
                        Ordering::Equal => { /* Does not change anything. */ }
                        Ordering::Less => {
                            if *first_ord == Ordering::Greater {
                                return None;
                            }
                            *first_ord = ord
                        }
                        Ordering::Greater => {
                            if *first_ord == Ordering::Less {
                                return None;
                            }
                            *first_ord = ord
                        }
                    }
                } else {
                    return None;
                }
            }
        }

        first_ord
    }

    #[inline]
    fn lt(&self, right: &Self) -> bool {
        assert!(
            self.shape() == right.shape(),
            "Matrix comparison error: dimensions mismatch."
        );
        self.iter().zip(right.iter()).all(|(a, b)| a.lt(b))
    }

    #[inline]
    fn le(&self, right: &Self) -> bool {
        assert!(
            self.shape() == right.shape(),
            "Matrix comparison error: dimensions mismatch."
        );
        self.iter().zip(right.iter()).all(|(a, b)| a.le(b))
    }

    #[inline]
    fn gt(&self, right: &Self) -> bool {
        assert!(
            self.shape() == right.shape(),
            "Matrix comparison error: dimensions mismatch."
        );
        self.iter().zip(right.iter()).all(|(a, b)| a.gt(b))
    }

    #[inline]
    fn ge(&self, right: &Self) -> bool {
        assert!(
            self.shape() == right.shape(),
            "Matrix comparison error: dimensions mismatch."
        );
        self.iter().zip(right.iter()).all(|(a, b)| a.ge(b))
    }
}

impl<N, R: Dim, C: Dim, S> Eq for Matrix<N, R, C, S>
where
    N: Scalar + Eq,
    S: Storage<N, R, C>,
{}

impl<N, R, R2, C, C2, S, S2> PartialEq<Matrix<N, R2, C2, S2>> for Matrix<N, R, C, S>
where
    N: Scalar + PartialEq,
    C: Dim,
    C2: Dim,
    R: Dim,
    R2: Dim,
    S: Storage<N, R, C>,
    S2: Storage<N, R2, C2>
{
    #[inline]
    fn eq(&self, right: &Matrix<N, R2, C2, S2>) -> bool {
        self.shape() == right.shape() && self.iter().zip(right.iter()).all(|(l, r)| l == r)
    }
}

macro_rules! impl_fmt {
    ($trait: path, $fmt_str_without_precision: expr, $fmt_str_with_precision: expr) => {
        impl<N, R: Dim, C: Dim, S> $trait for Matrix<N, R, C, S>
        where
            N: Scalar + $trait,
            S: Storage<N, R, C>,
            DefaultAllocator: Allocator<usize, R, C>,
        {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                #[cfg(feature = "std")]
                fn val_width<N: Scalar + $trait>(val: &N, f: &mut fmt::Formatter) -> usize {
                    match f.precision() {
                        Some(precision) => format!($fmt_str_with_precision, val, precision).chars().count(),
                        None => format!($fmt_str_without_precision, val).chars().count(),
                    }
                }

                #[cfg(not(feature = "std"))]
                fn val_width<N: Scalar + $trait>(_: &N, _: &mut fmt::Formatter) -> usize {
                    4
                }

                let (nrows, ncols) = self.data.shape();

                if nrows.value() == 0 || ncols.value() == 0 {
                    return write!(f, "[ ]");
                }

                let mut max_length = 0;
                let mut lengths: MatrixMN<usize, R, C> = Matrix::zeros_generic(nrows, ncols);
                let (nrows, ncols) = self.shape();

                for i in 0..nrows {
                    for j in 0..ncols {
                        lengths[(i, j)] = val_width(&self[(i, j)], f);
                        max_length = crate::max(max_length, lengths[(i, j)]);
                    }
                }

                let max_length_with_space = max_length + 1;

                writeln!(f)?;
                writeln!(
                    f,
                    "  ┌ {:>width$} ┐",
                    "",
                    width = max_length_with_space * ncols - 1
                )?;

                for i in 0..nrows {
                    write!(f, "  │")?;
                    for j in 0..ncols {
                        let number_length = lengths[(i, j)] + 1;
                        let pad = max_length_with_space - number_length;
                        write!(f, " {:>thepad$}", "", thepad = pad)?;
                        match f.precision() {
                            Some(precision) => write!(f, $fmt_str_with_precision, (*self)[(i, j)], precision)?,
                            None => write!(f, $fmt_str_without_precision, (*self)[(i, j)])?,
                        }
                    }
                    writeln!(f, " │")?;
                }

                writeln!(
                    f,
                    "  └ {:>width$} ┘",
                    "",
                    width = max_length_with_space * ncols - 1
                )?;
                writeln!(f)
            }
        }
    };
}
impl_fmt!(fmt::Display, "{}", "{:.1$}");
impl_fmt!(fmt::LowerExp, "{:e}", "{:.1$e}");
impl_fmt!(fmt::UpperExp, "{:E}", "{:.1$E}");
impl_fmt!(fmt::Octal, "{:o}", "{:1$o}");
impl_fmt!(fmt::LowerHex, "{:x}", "{:1$x}");
impl_fmt!(fmt::UpperHex, "{:X}", "{:1$X}");
impl_fmt!(fmt::Binary, "{:b}", "{:.1$b}");
impl_fmt!(fmt::Pointer, "{:p}", "{:.1$p}");

#[test]
fn lower_exp() {
    let test = crate::Matrix2::new(1e6, 2e5, 2e-5, 1.);
    assert_eq!(format!("{:e}", test), r"
  ┌           ┐
  │  1e6  2e5 │
  │ 2e-5  1e0 │
  └           ┘

")
}

impl<N: Scalar + Ring, R: Dim, C: Dim, S: Storage<N, R, C>> Matrix<N, R, C, S> {
    /// The perpendicular product between two 2D column vectors, i.e. `a.x * b.y - a.y * b.x`.
    #[inline]
    pub fn perp<R2, C2, SB>(&self, b: &Matrix<N, R2, C2, SB>) -> N
    where
        R2: Dim,
        C2: Dim,
        SB: Storage<N, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, U2>
            + SameNumberOfColumns<C, U1>
            + SameNumberOfRows<R2, U2>
            + SameNumberOfColumns<C2, U1>,
    {
        assert!(self.shape() == (2, 1), "2D perpendicular product ");

        unsafe {
            self.get_unchecked((0, 0)).inlined_clone() * b.get_unchecked((1, 0)).inlined_clone()
                - self.get_unchecked((1, 0)).inlined_clone() * b.get_unchecked((0, 0)).inlined_clone()
        }
    }

    // FIXME: use specialization instead of an assertion.
    /// The 3D cross product between two vectors.
    ///
    /// Panics if the shape is not 3D vector. In the future, this will be implemented only for
    /// dynamically-sized matrices and statically-sized 3D matrices.
    #[inline]
    pub fn cross<R2, C2, SB>(&self, b: &Matrix<N, R2, C2, SB>) -> MatrixCross<N, R, C, R2, C2>
    where
        R2: Dim,
        C2: Dim,
        SB: Storage<N, R2, C2>,
        DefaultAllocator: SameShapeAllocator<N, R, C, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2>,
    {
        let shape = self.shape();
        assert!(
            shape == b.shape(),
            "Vector cross product dimension mismatch."
        );
        assert!(
            (shape.0 == 3 && shape.1 == 1) || (shape.0 == 1 && shape.1 == 3),
            "Vector cross product dimension mismatch."
        );

        if shape.0 == 3 {
            unsafe {
                // FIXME: soooo ugly!
                let nrows = SameShapeR::<R, R2>::from_usize(3);
                let ncols = SameShapeC::<C, C2>::from_usize(1);
                let mut res = Matrix::new_uninitialized_generic(nrows, ncols);

                let ax = self.get_unchecked((0, 0));
                let ay = self.get_unchecked((1, 0));
                let az = self.get_unchecked((2, 0));

                let bx = b.get_unchecked((0, 0));
                let by = b.get_unchecked((1, 0));
                let bz = b.get_unchecked((2, 0));

                *res.get_unchecked_mut((0, 0)) = ay.inlined_clone() * bz.inlined_clone() - az.inlined_clone() * by.inlined_clone();
                *res.get_unchecked_mut((1, 0)) = az.inlined_clone() * bx.inlined_clone() - ax.inlined_clone() * bz.inlined_clone();
                *res.get_unchecked_mut((2, 0)) = ax.inlined_clone() * by.inlined_clone() - ay.inlined_clone() * bx.inlined_clone();

                res
            }
        } else {
            unsafe {
                // FIXME: ugly!
                let nrows = SameShapeR::<R, R2>::from_usize(1);
                let ncols = SameShapeC::<C, C2>::from_usize(3);
                let mut res = Matrix::new_uninitialized_generic(nrows, ncols);

                let ax = self.get_unchecked((0, 0));
                let ay = self.get_unchecked((0, 1));
                let az = self.get_unchecked((0, 2));

                let bx = b.get_unchecked((0, 0));
                let by = b.get_unchecked((0, 1));
                let bz = b.get_unchecked((0, 2));

                *res.get_unchecked_mut((0, 0)) = ay.inlined_clone() * bz.inlined_clone() - az.inlined_clone() * by.inlined_clone();
                *res.get_unchecked_mut((0, 1)) = az.inlined_clone() * bx.inlined_clone() - ax.inlined_clone() * bz.inlined_clone();
                *res.get_unchecked_mut((0, 2)) = ax.inlined_clone() * by.inlined_clone() - ay.inlined_clone() * bx.inlined_clone();

                res
            }
        }
    }
}

impl<N: Scalar + Field, S: Storage<N, U3>> Vector<N, U3, S>
where DefaultAllocator: Allocator<N, U3>
{
    /// Computes the matrix `M` such that for all vector `v` we have `M * v == self.cross(&v)`.
    #[inline]
    pub fn cross_matrix(&self) -> MatrixN<N, U3> {
        MatrixN::<N, U3>::new(
            N::zero(),
            -self[2].inlined_clone(),
            self[1].inlined_clone(),
            self[2].inlined_clone(),
            N::zero(),
            -self[0].inlined_clone(),
            -self[1].inlined_clone(),
            self[0].inlined_clone(),
            N::zero(),
        )
    }
}

impl<N: ComplexField, R: Dim, C: Dim, S: Storage<N, R, C>> Matrix<N, R, C, S> {
    /// The smallest angle between two vectors.
    #[inline]
    pub fn angle<R2: Dim, C2: Dim, SB>(&self, other: &Matrix<N, R2, C2, SB>) -> N::RealField
    where
        SB: Storage<N, R2, C2>,
        ShapeConstraint: DimEq<R, R2> + DimEq<C, C2>,
    {
        let prod = self.dotc(other);
        let n1 = self.norm();
        let n2 = other.norm();

        if n1.is_zero() || n2.is_zero() {
            N::RealField::zero()
        } else {
            let cang = prod.real() / (n1 * n2);

            if cang > N::RealField::one() {
                N::RealField::zero()
            } else if cang < -N::RealField::one() {
                N::RealField::pi()
            } else {
                cang.acos()
            }
        }
    }
}

impl<N: Scalar + Zero + One + ClosedAdd + ClosedSub + ClosedMul, D: Dim, S: Storage<N, D>>
    Vector<N, D, S>
{
    /// Returns `self * (1.0 - t) + rhs * t`, i.e., the linear blend of the vectors x and y using the scalar value a.
    ///
    /// The value for a is not restricted to the range `[0, 1]`.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let x = Vector3::new(1.0, 2.0, 3.0);
    /// let y = Vector3::new(10.0, 20.0, 30.0);
    /// assert_eq!(x.lerp(&y, 0.1), Vector3::new(1.9, 3.8, 5.7));
    /// ```
    pub fn lerp<S2: Storage<N, D>>(&self, rhs: &Vector<N, D, S2>, t: N) -> VectorN<N, D>
    where DefaultAllocator: Allocator<N, D> {
        let mut res = self.clone_owned();
        res.axpy(t.inlined_clone(), rhs, N::one() - t);
        res
    }
}

impl<N: RealField, D: Dim, S: Storage<N, D>> Unit<Vector<N, D, S>> {
    /// Computes the spherical linear interpolation between two unit vectors.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::{Unit, Vector2};
    ///
    /// let v1 = Unit::new_normalize(Vector2::new(1.0, 2.0));
    /// let v2 = Unit::new_normalize(Vector2::new(2.0, -3.0));
    ///
    /// let v = v1.slerp(&v2, 1.0);
    ///
    /// assert_eq!(v, v2);
    /// ```
    pub fn slerp<S2: Storage<N, D>>(
        &self,
        rhs: &Unit<Vector<N, D, S2>>,
        t: N,
    ) -> Unit<VectorN<N, D>>
    where
        DefaultAllocator: Allocator<N, D>,
    {
        // FIXME: the result is wrong when self and rhs are collinear with opposite direction.
        self.try_slerp(rhs, t, N::default_epsilon())
            .unwrap_or(Unit::new_unchecked(self.clone_owned()))
    }

    /// Computes the spherical linear interpolation between two unit vectors.
    ///
    /// Returns `None` if the two vectors are almost collinear and with opposite direction
    /// (in this case, there is an infinity of possible results).
    pub fn try_slerp<S2: Storage<N, D>>(
        &self,
        rhs: &Unit<Vector<N, D, S2>>,
        t: N,
        epsilon: N,
    ) -> Option<Unit<VectorN<N, D>>>
    where
        DefaultAllocator: Allocator<N, D>,
    {
        let c_hang = self.dot(rhs);

        // self == other
        if c_hang >= N::one() {
            return Some(Unit::new_unchecked(self.clone_owned()));
        }

        let hang = c_hang.acos();
        let s_hang = (N::one() - c_hang * c_hang).sqrt();

        // FIXME: what if s_hang is 0.0 ? The result is not well-defined.
        if relative_eq!(s_hang, N::zero(), epsilon = epsilon) {
            None
        } else {
            let ta = ((N::one() - t) * hang).sin() / s_hang;
            let tb = (t * hang).sin() / s_hang;
            let mut res = self.scale(ta);
            res.axpy(tb, &**rhs, N::one());

            Some(Unit::new_unchecked(res))
        }
    }
}

impl<N, R: Dim, C: Dim, S> AbsDiffEq for Unit<Matrix<N, R, C, S>>
where
    N: Scalar + AbsDiffEq,
    S: Storage<N, R, C>,
    N::Epsilon: Copy,
{
    type Epsilon = N::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        N::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.as_ref().abs_diff_eq(other.as_ref(), epsilon)
    }
}

impl<N, R: Dim, C: Dim, S> RelativeEq for Unit<Matrix<N, R, C, S>>
where
    N: Scalar + RelativeEq,
    S: Storage<N, R, C>,
    N::Epsilon: Copy,
{
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        N::default_max_relative()
    }

    #[inline]
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool
    {
        self.as_ref()
            .relative_eq(other.as_ref(), epsilon, max_relative)
    }
}

impl<N, R: Dim, C: Dim, S> UlpsEq for Unit<Matrix<N, R, C, S>>
where
    N: Scalar + UlpsEq,
    S: Storage<N, R, C>,
    N::Epsilon: Copy,
{
    #[inline]
    fn default_max_ulps() -> u32 {
        N::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.as_ref().ulps_eq(other.as_ref(), epsilon, max_ulps)
    }
}

impl<N, R, C, S> Hash for Matrix<N, R, C, S>
where
    N: Scalar + Hash,
    R: Dim,
    C: Dim,
    S: Storage<N, R, C>,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        let (nrows, ncols) = self.shape();
        (nrows, ncols).hash(state);

        for j in 0..ncols {
            for i in 0..nrows {
                unsafe {
                     self.get_unchecked((i, j)).hash(state);
                }
            }
        }
    }
}
