#[cfg(feature = "arbitrary")]
use crate::base::storage::Owned;
#[cfg(feature = "arbitrary")]
use quickcheck::{Arbitrary, Gen};

use num::{Bounded, One, Zero};
use rand::distributions::{Distribution, Standard};
use rand::Rng;
#[cfg(feature = "std")]
use rand;
#[cfg(feature = "std")]
use rand_distr::StandardNormal;
use std::iter;
use typenum::{self, Cmp, Greater};

#[cfg(feature = "std")]
use alga::general::RealField;
use alga::general::{ClosedAdd, ClosedMul};

use crate::base::allocator::Allocator;
use crate::{Dim, DimName, Dynamic, U1, U2, U3, U4, U5, U6};
use crate::base::storage::Storage;
use crate::base::{DefaultAllocator, Matrix, MatrixMN, MatrixN, Scalar, Unit, Vector, VectorN};

/*
 *
 * Generic constructors.
 *
 */
impl<N: Scalar, R: Dim, C: Dim> MatrixMN<N, R, C>
where DefaultAllocator: Allocator<N, R, C>
{
    /// Creates a new uninitialized matrix. If the matrix has a compile-time dimension, this panics
    /// if `nrows != R::to_usize()` or `ncols != C::to_usize()`.
    #[inline]
    pub unsafe fn new_uninitialized_generic(nrows: R, ncols: C) -> Self {
        Self::from_data(DefaultAllocator::allocate_uninitialized(nrows, ncols))
    }

    /// Creates a matrix with all its elements set to `elem`.
    #[inline]
    pub fn from_element_generic(nrows: R, ncols: C, elem: N) -> Self {
        let len = nrows.value() * ncols.value();
        Self::from_iterator_generic(nrows, ncols, iter::repeat(elem).take(len))
    }

    /// Creates a matrix with all its elements set to `elem`.
    ///
    /// Same as `from_element_generic`.
    #[inline]
    pub fn repeat_generic(nrows: R, ncols: C, elem: N) -> Self {
        let len = nrows.value() * ncols.value();
        Self::from_iterator_generic(nrows, ncols, iter::repeat(elem).take(len))
    }

    /// Creates a matrix with all its elements set to 0.
    #[inline]
    pub fn zeros_generic(nrows: R, ncols: C) -> Self
    where N: Zero {
        Self::from_element_generic(nrows, ncols, N::zero())
    }

    /// Creates a matrix with all its elements filled by an iterator.
    #[inline]
    pub fn from_iterator_generic<I>(nrows: R, ncols: C, iter: I) -> Self
    where I: IntoIterator<Item = N> {
        Self::from_data(DefaultAllocator::allocate_from_iterator(nrows, ncols, iter))
    }

    /// Creates a matrix with its elements filled with the components provided by a slice in
    /// row-major order.
    ///
    /// The order of elements in the slice must follow the usual mathematic writing, i.e.,
    /// row-by-row.
    #[inline]
    pub fn from_row_slice_generic(nrows: R, ncols: C, slice: &[N]) -> Self {
        assert!(
            slice.len() == nrows.value() * ncols.value(),
            "Matrix init. error: the slice did not contain the right number of elements."
        );

        let mut res = unsafe { Self::new_uninitialized_generic(nrows, ncols) };
        let mut iter = slice.iter();

        for i in 0..nrows.value() {
            for j in 0..ncols.value() {
                unsafe { *res.get_unchecked_mut((i, j)) = *iter.next().unwrap() }
            }
        }

        res
    }

    /// Creates a matrix with its elements filled with the components provided by a slice. The
    /// components must have the same layout as the matrix data storage (i.e. column-major).
    #[inline]
    pub fn from_column_slice_generic(nrows: R, ncols: C, slice: &[N]) -> Self {
        Self::from_iterator_generic(nrows, ncols, slice.iter().cloned())
    }

    /// Creates a matrix filled with the results of a function applied to each of its component
    /// coordinates.
    #[inline]
    pub fn from_fn_generic<F>(nrows: R, ncols: C, mut f: F) -> Self
    where F: FnMut(usize, usize) -> N {
        let mut res = unsafe { Self::new_uninitialized_generic(nrows, ncols) };

        for j in 0..ncols.value() {
            for i in 0..nrows.value() {
                unsafe { *res.get_unchecked_mut((i, j)) = f(i, j) }
            }
        }

        res
    }

    /// Creates a new identity matrix.
    ///
    /// If the matrix is not square, the largest square submatrix starting at index `(0, 0)` is set
    /// to the identity matrix. All other entries are set to zero.
    #[inline]
    pub fn identity_generic(nrows: R, ncols: C) -> Self
    where N: Zero + One {
        Self::from_diagonal_element_generic(nrows, ncols, N::one())
    }

    /// Creates a new matrix with its diagonal filled with copies of `elt`.
    ///
    /// If the matrix is not square, the largest square submatrix starting at index `(0, 0)` is set
    /// to the identity matrix. All other entries are set to zero.
    #[inline]
    pub fn from_diagonal_element_generic(nrows: R, ncols: C, elt: N) -> Self
    where N: Zero + One {
        let mut res = Self::zeros_generic(nrows, ncols);

        for i in 0..crate::min(nrows.value(), ncols.value()) {
            unsafe { *res.get_unchecked_mut((i, i)) = elt }
        }

        res
    }

    /// Creates a new matrix that may be rectangular. The first `elts.len()` diagonal elements are
    /// filled with the content of `elts`. Others are set to 0.
    ///
    /// Panics if `elts.len()` is larger than the minimum among `nrows` and `ncols`.
    #[inline]
    pub fn from_partial_diagonal_generic(nrows: R, ncols: C, elts: &[N]) -> Self
    where N: Zero {
        let mut res = Self::zeros_generic(nrows, ncols);
        assert!(
            elts.len() <= crate::min(nrows.value(), ncols.value()),
            "Too many diagonal elements provided."
        );

        for (i, elt) in elts.iter().enumerate() {
            unsafe { *res.get_unchecked_mut((i, i)) = *elt }
        }

        res
    }

    /// Builds a new matrix from its rows.
    ///
    /// Panics if not enough rows are provided (for statically-sized matrices), or if all rows do
    /// not have the same dimensions.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{RowVector3, Matrix3};
    /// # use std::iter;
    ///
    /// let m = Matrix3::from_rows(&[ RowVector3::new(1.0, 2.0, 3.0),  RowVector3::new(4.0, 5.0, 6.0),  RowVector3::new(7.0, 8.0, 9.0) ]);
    ///
    /// assert!(m.m11 == 1.0 && m.m12 == 2.0 && m.m13 == 3.0 &&
    ///         m.m21 == 4.0 && m.m22 == 5.0 && m.m23 == 6.0 &&
    ///         m.m31 == 7.0 && m.m32 == 8.0 && m.m33 == 9.0);
    /// ```
    #[inline]
    pub fn from_rows<SB>(rows: &[Matrix<N, U1, C, SB>]) -> Self
    where SB: Storage<N, U1, C> {
        assert!(rows.len() > 0, "At least one row must be given.");
        let nrows = R::try_to_usize().unwrap_or(rows.len());
        let ncols = rows[0].len();
        assert!(
            rows.len() == nrows,
            "Invalid number of rows provided to build this matrix."
        );

        if C::try_to_usize().is_none() {
            assert!(
                rows.iter().all(|r| r.len() == ncols),
                "The provided rows must all have the same dimension."
            );
        }

        // FIXME: optimize that.
        Self::from_fn_generic(R::from_usize(nrows), C::from_usize(ncols), |i, j| {
            rows[i][(0, j)]
        })
    }

    /// Builds a new matrix from its columns.
    ///
    /// Panics if not enough columns are provided (for statically-sized matrices), or if all
    /// columns do not have the same dimensions.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{Vector3, Matrix3};
    /// # use std::iter;
    ///
    /// let m = Matrix3::from_columns(&[ Vector3::new(1.0, 2.0, 3.0),  Vector3::new(4.0, 5.0, 6.0),  Vector3::new(7.0, 8.0, 9.0) ]);
    ///
    /// assert!(m.m11 == 1.0 && m.m12 == 4.0 && m.m13 == 7.0 &&
    ///         m.m21 == 2.0 && m.m22 == 5.0 && m.m23 == 8.0 &&
    ///         m.m31 == 3.0 && m.m32 == 6.0 && m.m33 == 9.0);
    /// ```
    #[inline]
    pub fn from_columns<SB>(columns: &[Vector<N, R, SB>]) -> Self
    where SB: Storage<N, R> {
        assert!(columns.len() > 0, "At least one column must be given.");
        let ncols = C::try_to_usize().unwrap_or(columns.len());
        let nrows = columns[0].len();
        assert!(
            columns.len() == ncols,
            "Invalid number of columns provided to build this matrix."
        );

        if R::try_to_usize().is_none() {
            assert!(
                columns.iter().all(|r| r.len() == nrows),
                "The columns provided must all have the same dimension."
            );
        }

        // FIXME: optimize that.
        Self::from_fn_generic(R::from_usize(nrows), C::from_usize(ncols), |i, j| {
            columns[j][i]
        })
    }

    /// Creates a matrix filled with random values.
    #[inline]
    #[cfg(feature = "std")]
    pub fn new_random_generic(nrows: R, ncols: C) -> Self
    where Standard: Distribution<N> {
        Self::from_fn_generic(nrows, ncols, |_, _| rand::random())
    }

    /// Creates a matrix filled with random values from the given distribution.
    #[inline]
    pub fn from_distribution_generic<Distr: Distribution<N> + ?Sized, G: Rng + ?Sized>(
        nrows: R,
        ncols: C,
        distribution: &Distr,
        rng: &mut G,
    ) -> Self
    {
        Self::from_fn_generic(nrows, ncols, |_, _| distribution.sample(rng))
    }

    /// Creates a matrix backed by a given `Vec`.
    ///
    /// The output matrix is filled column-by-column.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{Dynamic, DMatrix, Matrix, U1};
    ///
    /// let vec = vec![0, 1, 2, 3, 4, 5];
    /// let vec_ptr = vec.as_ptr();
    ///
    /// let matrix = Matrix::from_vec_generic(Dynamic::new(vec.len()), U1, vec);
    /// let matrix_storage_ptr = matrix.data.as_vec().as_ptr();
    ///
    /// // `matrix` is backed by exactly the same `Vec` as it was constructed from.
    /// assert_eq!(matrix_storage_ptr, vec_ptr);
    /// ```
    #[inline]
    #[cfg(feature = "std")]
    pub fn from_vec_generic(nrows: R, ncols: C, data: Vec<N>) -> Self {
        Self::from_iterator_generic(nrows, ncols, data)
    }
}

impl<N, D: Dim> MatrixN<N, D>
where
    N: Scalar,
    DefaultAllocator: Allocator<N, D, D>,
{
    /// Creates a square matrix with its diagonal set to `diag` and all other entries set to 0.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{Vector3, DVector, Matrix3, DMatrix};
    /// # use std::iter;
    ///
    /// let m = Matrix3::from_diagonal(&Vector3::new(1.0, 2.0, 3.0));
    /// // The two additional arguments represent the matrix dimensions.
    /// let dm = DMatrix::from_diagonal(&DVector::from_row_slice(&[1.0, 2.0, 3.0]));
    ///
    /// assert!(m.m11 == 1.0 && m.m12 == 0.0 && m.m13 == 0.0 &&
    ///         m.m21 == 0.0 && m.m22 == 2.0 && m.m23 == 0.0 &&
    ///         m.m31 == 0.0 && m.m32 == 0.0 && m.m33 == 3.0);
    /// assert!(dm[(0, 0)] == 1.0 && dm[(0, 1)] == 0.0 && dm[(0, 2)] == 0.0 &&
    ///         dm[(1, 0)] == 0.0 && dm[(1, 1)] == 2.0 && dm[(1, 2)] == 0.0 &&
    ///         dm[(2, 0)] == 0.0 && dm[(2, 1)] == 0.0 && dm[(2, 2)] == 3.0);
    /// ```
    #[inline]
    pub fn from_diagonal<SB: Storage<N, D>>(diag: &Vector<N, D, SB>) -> Self
    where N: Zero {
        let (dim, _) = diag.data.shape();
        let mut res = Self::zeros_generic(dim, dim);

        for i in 0..diag.len() {
            unsafe {
                *res.get_unchecked_mut((i, i)) = *diag.vget_unchecked(i);
            }
        }

        res
    }
}

/*
 *
 * Generate constructors with varying number of arguments, depending on the object type.
 *
 */
macro_rules! impl_constructors(
    ($($Dims: ty),*; $(=> $DimIdent: ident: $DimBound: ident),*; $($gargs: expr),*; $($args: ident),*) => {
        impl<N: Scalar, $($DimIdent: $DimBound, )*> MatrixMN<N $(, $Dims)*>
            where DefaultAllocator: Allocator<N $(, $Dims)*> {

            /// Creates a new uninitialized matrix or vector.
            #[inline]
            pub unsafe fn new_uninitialized($($args: usize),*) -> Self {
                Self::new_uninitialized_generic($($gargs),*)
            }

            /// Creates a matrix or vector with all its elements set to `elem`.
            ///
            /// # Example
            /// ```
            /// # use nalgebra::{Matrix2x3, Vector3, DVector, DMatrix};
            ///
            /// let v = Vector3::from_element(2.0);
            /// // The additional argument represents the vector dimension.
            /// let dv = DVector::from_element(3, 2.0);
            /// let m = Matrix2x3::from_element(2.0);
            /// // The two additional arguments represent the matrix dimensions.
            /// let dm = DMatrix::from_element(2, 3, 2.0);
            ///
            /// assert!(v.x == 2.0 && v.y == 2.0 && v.z == 2.0);
            /// assert!(dv[0] == 2.0 && dv[1] == 2.0 && dv[2] == 2.0);
            /// assert!(m.m11 == 2.0 && m.m12 == 2.0 && m.m13 == 2.0 &&
            ///         m.m21 == 2.0 && m.m22 == 2.0 && m.m23 == 2.0);
            /// assert!(dm[(0, 0)] == 2.0 && dm[(0, 1)] == 2.0 && dm[(0, 2)] == 2.0 &&
            ///         dm[(1, 0)] == 2.0 && dm[(1, 1)] == 2.0 && dm[(1, 2)] == 2.0);
            /// ```
            #[inline]
            pub fn from_element($($args: usize,)* elem: N) -> Self {
                Self::from_element_generic($($gargs, )* elem)
            }

            /// Creates a matrix or vector with all its elements set to `elem`.
            ///
            /// Same as `.from_element`.
            ///
            /// # Example
            /// ```
            /// # use nalgebra::{Matrix2x3, Vector3, DVector, DMatrix};
            ///
            /// let v = Vector3::repeat(2.0);
            /// // The additional argument represents the vector dimension.
            /// let dv = DVector::repeat(3, 2.0);
            /// let m = Matrix2x3::repeat(2.0);
            /// // The two additional arguments represent the matrix dimensions.
            /// let dm = DMatrix::repeat(2, 3, 2.0);
            ///
            /// assert!(v.x == 2.0 && v.y == 2.0 && v.z == 2.0);
            /// assert!(dv[0] == 2.0 && dv[1] == 2.0 && dv[2] == 2.0);
            /// assert!(m.m11 == 2.0 && m.m12 == 2.0 && m.m13 == 2.0 &&
            ///         m.m21 == 2.0 && m.m22 == 2.0 && m.m23 == 2.0);
            /// assert!(dm[(0, 0)] == 2.0 && dm[(0, 1)] == 2.0 && dm[(0, 2)] == 2.0 &&
            ///         dm[(1, 0)] == 2.0 && dm[(1, 1)] == 2.0 && dm[(1, 2)] == 2.0);
            /// ```
            #[inline]
            pub fn repeat($($args: usize,)* elem: N) -> Self {
                Self::repeat_generic($($gargs, )* elem)
            }

            /// Creates a matrix or vector with all its elements set to `0`.
            ///
            /// # Example
            /// ```
            /// # use nalgebra::{Matrix2x3, Vector3, DVector, DMatrix};
            ///
            /// let v = Vector3::<f32>::zeros();
            /// // The argument represents the vector dimension.
            /// let dv = DVector::<f32>::zeros(3);
            /// let m = Matrix2x3::<f32>::zeros();
            /// // The two arguments represent the matrix dimensions.
            /// let dm = DMatrix::<f32>::zeros(2, 3);
            ///
            /// assert!(v.x == 0.0 && v.y == 0.0 && v.z == 0.0);
            /// assert!(dv[0] == 0.0 && dv[1] == 0.0 && dv[2] == 0.0);
            /// assert!(m.m11 == 0.0 && m.m12 == 0.0 && m.m13 == 0.0 &&
            ///         m.m21 == 0.0 && m.m22 == 0.0 && m.m23 == 0.0);
            /// assert!(dm[(0, 0)] == 0.0 && dm[(0, 1)] == 0.0 && dm[(0, 2)] == 0.0 &&
            ///         dm[(1, 0)] == 0.0 && dm[(1, 1)] == 0.0 && dm[(1, 2)] == 0.0);
            /// ```
            #[inline]
            pub fn zeros($($args: usize),*) -> Self
                where N: Zero {
                Self::zeros_generic($($gargs),*)
            }

            /// Creates a matrix or vector with all its elements filled by an iterator.
            ///
            /// The output matrix is filled column-by-column.
            ///
            /// # Example
            /// ```
            /// # use nalgebra::{Matrix2x3, Vector3, DVector, DMatrix};
            /// # use std::iter;
            ///
            /// let v = Vector3::from_iterator((0..3).into_iter());
            /// // The additional argument represents the vector dimension.
            /// let dv = DVector::from_iterator(3, (0..3).into_iter());
            /// let m = Matrix2x3::from_iterator((0..6).into_iter());
            /// // The two additional arguments represent the matrix dimensions.
            /// let dm = DMatrix::from_iterator(2, 3, (0..6).into_iter());
            ///
            /// assert!(v.x == 0 && v.y == 1 && v.z == 2);
            /// assert!(dv[0] == 0 && dv[1] == 1 && dv[2] == 2);
            /// assert!(m.m11 == 0 && m.m12 == 2 && m.m13 == 4 &&
            ///         m.m21 == 1 && m.m22 == 3 && m.m23 == 5);
            /// assert!(dm[(0, 0)] == 0 && dm[(0, 1)] == 2 && dm[(0, 2)] == 4 &&
            ///         dm[(1, 0)] == 1 && dm[(1, 1)] == 3 && dm[(1, 2)] == 5);
            /// ```
            #[inline]
            pub fn from_iterator<I>($($args: usize,)* iter: I) -> Self
                where I: IntoIterator<Item = N> {
                Self::from_iterator_generic($($gargs, )* iter)
            }

            /// Creates a matrix or vector filled with the results of a function applied to each of its
            /// component coordinates.
            ///
            /// # Example
            /// ```
            /// # use nalgebra::{Matrix2x3, Vector3, DVector, DMatrix};
            /// # use std::iter;
            ///
            /// let v = Vector3::from_fn(|i, _| i);
            /// // The additional argument represents the vector dimension.
            /// let dv = DVector::from_fn(3, |i, _| i);
            /// let m = Matrix2x3::from_fn(|i, j| i * 3 + j);
            /// // The two additional arguments represent the matrix dimensions.
            /// let dm = DMatrix::from_fn(2, 3, |i, j| i * 3 + j);
            ///
            /// assert!(v.x == 0 && v.y == 1 && v.z == 2);
            /// assert!(dv[0] == 0 && dv[1] == 1 && dv[2] == 2);
            /// assert!(m.m11 == 0 && m.m12 == 1 && m.m13 == 2 &&
            ///         m.m21 == 3 && m.m22 == 4 && m.m23 == 5);
            /// assert!(dm[(0, 0)] == 0 && dm[(0, 1)] == 1 && dm[(0, 2)] == 2 &&
            ///         dm[(1, 0)] == 3 && dm[(1, 1)] == 4 && dm[(1, 2)] == 5);
            /// ```
            #[inline]
            pub fn from_fn<F>($($args: usize,)* f: F) -> Self
                where F: FnMut(usize, usize) -> N {
                Self::from_fn_generic($($gargs, )* f)
            }

            /// Creates an identity matrix. If the matrix is not square, the largest square
            /// submatrix (starting at the first row and column) is set to the identity while all
            /// other entries are set to zero.
            ///
            /// # Example
            /// ```
            /// # use nalgebra::{Matrix2x3, DMatrix};
            /// # use std::iter;
            ///
            /// let m = Matrix2x3::<f32>::identity();
            /// // The two additional arguments represent the matrix dimensions.
            /// let dm = DMatrix::<f32>::identity(2, 3);
            ///
            /// assert!(m.m11 == 1.0 && m.m12 == 0.0 && m.m13 == 0.0 &&
            ///         m.m21 == 0.0 && m.m22 == 1.0 && m.m23 == 0.0);
            /// assert!(dm[(0, 0)] == 1.0 && dm[(0, 1)] == 0.0 && dm[(0, 2)] == 0.0 &&
            ///         dm[(1, 0)] == 0.0 && dm[(1, 1)] == 1.0 && dm[(1, 2)] == 0.0);
            /// ```
            #[inline]
            pub fn identity($($args: usize,)*) -> Self
                where N: Zero + One {
                Self::identity_generic($($gargs),* )
            }

            /// Creates a matrix filled with its diagonal filled with `elt` and all other
            /// components set to zero.
            ///
            /// # Example
            /// ```
            /// # use nalgebra::{Matrix2x3, DMatrix};
            /// # use std::iter;
            ///
            /// let m = Matrix2x3::from_diagonal_element(5.0);
            /// // The two additional arguments represent the matrix dimensions.
            /// let dm = DMatrix::from_diagonal_element(2, 3, 5.0);
            ///
            /// assert!(m.m11 == 5.0 && m.m12 == 0.0 && m.m13 == 0.0 &&
            ///         m.m21 == 0.0 && m.m22 == 5.0 && m.m23 == 0.0);
            /// assert!(dm[(0, 0)] == 5.0 && dm[(0, 1)] == 0.0 && dm[(0, 2)] == 0.0 &&
            ///         dm[(1, 0)] == 0.0 && dm[(1, 1)] == 5.0 && dm[(1, 2)] == 0.0);
            /// ```
            #[inline]
            pub fn from_diagonal_element($($args: usize,)* elt: N) -> Self
                where N: Zero + One {
                Self::from_diagonal_element_generic($($gargs, )* elt)
            }

            /// Creates a new matrix that may be rectangular. The first `elts.len()` diagonal
            /// elements are filled with the content of `elts`. Others are set to 0.
            ///
            /// Panics if `elts.len()` is larger than the minimum among `nrows` and `ncols`.
            ///
            /// # Example
            /// ```
            /// # use nalgebra::{Matrix3, DMatrix};
            /// # use std::iter;
            ///
            /// let m = Matrix3::from_partial_diagonal(&[1.0, 2.0]);
            /// // The two additional arguments represent the matrix dimensions.
            /// let dm = DMatrix::from_partial_diagonal(3, 3, &[1.0, 2.0]);
            ///
            /// assert!(m.m11 == 1.0 && m.m12 == 0.0 && m.m13 == 0.0 &&
            ///         m.m21 == 0.0 && m.m22 == 2.0 && m.m23 == 0.0 &&
            ///         m.m31 == 0.0 && m.m32 == 0.0 && m.m33 == 0.0);
            /// assert!(dm[(0, 0)] == 1.0 && dm[(0, 1)] == 0.0 && dm[(0, 2)] == 0.0 &&
            ///         dm[(1, 0)] == 0.0 && dm[(1, 1)] == 2.0 && dm[(1, 2)] == 0.0 &&
            ///         dm[(2, 0)] == 0.0 && dm[(2, 1)] == 0.0 && dm[(2, 2)] == 0.0);
            /// ```
            #[inline]
            pub fn from_partial_diagonal($($args: usize,)* elts: &[N]) -> Self
                where N: Zero {
                Self::from_partial_diagonal_generic($($gargs, )* elts)
            }

            /// Creates a matrix or vector filled with random values from the given distribution.
            #[inline]
            pub fn from_distribution<Distr: Distribution<N> + ?Sized, G: Rng + ?Sized>(
                $($args: usize,)*
                distribution: &Distr,
                rng: &mut G,
            ) -> Self {
                Self::from_distribution_generic($($gargs, )* distribution, rng)
            }
        }

        impl<N: Scalar, $($DimIdent: $DimBound, )*> MatrixMN<N $(, $Dims)*>
            where
            DefaultAllocator: Allocator<N $(, $Dims)*>,
            Standard: Distribution<N> {

            /// Creates a matrix filled with random values.
            #[inline]
            #[cfg(feature = "std")]
            pub fn new_random($($args: usize),*) -> Self {
                Self::new_random_generic($($gargs),*)
            }
        }
    }
);

// FIXME: this is not very pretty. We could find a better call syntax.
impl_constructors!(R, C;                         // Arguments for Matrix<N, ..., S>
                   => R: DimName, => C: DimName; // Type parameters for impl<N, ..., S>
                   R::name(), C::name();         // Arguments for `_generic` constructors.
                   ); // Arguments for non-generic constructors.

impl_constructors!(R, Dynamic;
                   => R: DimName;
                   R::name(), Dynamic::new(ncols);
                   ncols);

impl_constructors!(Dynamic, C;
                   => C: DimName;
                   Dynamic::new(nrows), C::name();
                   nrows);

impl_constructors!(Dynamic, Dynamic;
                   ;
                   Dynamic::new(nrows), Dynamic::new(ncols);
                   nrows, ncols);

/*
 *
 * Constructors that don't necessarily require all dimensions
 * to be specified whon one dimension is already known.
 *
 */
macro_rules! impl_constructors_from_data(
    ($data: ident; $($Dims: ty),*; $(=> $DimIdent: ident: $DimBound: ident),*; $($gargs: expr),*; $($args: ident),*) => {
        impl<N: Scalar, $($DimIdent: $DimBound, )*> MatrixMN<N $(, $Dims)*>
        where DefaultAllocator: Allocator<N $(, $Dims)*> {
            /// Creates a matrix with its elements filled with the components provided by a slice
            /// in row-major order.
            ///
            /// The order of elements in the slice must follow the usual mathematic writing, i.e.,
            /// row-by-row.
            ///
            /// # Example
            /// ```
            /// # use nalgebra::{Matrix2x3, Vector3, DVector, DMatrix};
            /// # use std::iter;
            ///
            /// let v = Vector3::from_row_slice(&[0, 1, 2]);
            /// // The additional argument represents the vector dimension.
            /// let dv = DVector::from_row_slice(&[0, 1, 2]);
            /// let m = Matrix2x3::from_row_slice(&[0, 1, 2, 3, 4, 5]);
            /// // The two additional arguments represent the matrix dimensions.
            /// let dm = DMatrix::from_row_slice(2, 3, &[0, 1, 2, 3, 4, 5]);
            ///
            /// assert!(v.x == 0 && v.y == 1 && v.z == 2);
            /// assert!(dv[0] == 0 && dv[1] == 1 && dv[2] == 2);
            /// assert!(m.m11 == 0 && m.m12 == 1 && m.m13 == 2 &&
            ///         m.m21 == 3 && m.m22 == 4 && m.m23 == 5);
            /// assert!(dm[(0, 0)] == 0 && dm[(0, 1)] == 1 && dm[(0, 2)] == 2 &&
            ///         dm[(1, 0)] == 3 && dm[(1, 1)] == 4 && dm[(1, 2)] == 5);
            /// ```
            #[inline]
            pub fn from_row_slice($($args: usize,)* $data: &[N]) -> Self {
                Self::from_row_slice_generic($($gargs, )* $data)
            }

            /// Creates a matrix with its elements filled with the components provided by a slice
            /// in column-major order.
            ///
            /// # Example
            /// ```
            /// # use nalgebra::{Matrix2x3, Vector3, DVector, DMatrix};
            /// # use std::iter;
            ///
            /// let v = Vector3::from_column_slice(&[0, 1, 2]);
            /// // The additional argument represents the vector dimension.
            /// let dv = DVector::from_column_slice(&[0, 1, 2]);
            /// let m = Matrix2x3::from_column_slice(&[0, 1, 2, 3, 4, 5]);
            /// // The two additional arguments represent the matrix dimensions.
            /// let dm = DMatrix::from_column_slice(2, 3, &[0, 1, 2, 3, 4, 5]);
            ///
            /// assert!(v.x == 0 && v.y == 1 && v.z == 2);
            /// assert!(dv[0] == 0 && dv[1] == 1 && dv[2] == 2);
            /// assert!(m.m11 == 0 && m.m12 == 2 && m.m13 == 4 &&
            ///         m.m21 == 1 && m.m22 == 3 && m.m23 == 5);
            /// assert!(dm[(0, 0)] == 0 && dm[(0, 1)] == 2 && dm[(0, 2)] == 4 &&
            ///         dm[(1, 0)] == 1 && dm[(1, 1)] == 3 && dm[(1, 2)] == 5);
            /// ```
            #[inline]
            pub fn from_column_slice($($args: usize,)* $data: &[N]) -> Self {
                Self::from_column_slice_generic($($gargs, )* $data)
            }

            /// Creates a matrix backed by a given `Vec`.
            ///
            /// The output matrix is filled column-by-column.
            ///
            /// # Example
            /// ```
            /// # use nalgebra::{DMatrix, Matrix2x3};
            ///
            /// let m = Matrix2x3::from_vec(vec![0, 1, 2, 3, 4, 5]);
            ///
            /// assert!(m.m11 == 0 && m.m12 == 2 && m.m13 == 4 &&
            ///         m.m21 == 1 && m.m22 == 3 && m.m23 == 5);
            ///
            ///
            /// // The two additional arguments represent the matrix dimensions.
            /// let dm = DMatrix::from_vec(2, 3, vec![0, 1, 2, 3, 4, 5]);
            ///
            /// assert!(dm[(0, 0)] == 0 && dm[(0, 1)] == 2 && dm[(0, 2)] == 4 &&
            ///         dm[(1, 0)] == 1 && dm[(1, 1)] == 3 && dm[(1, 2)] == 5);
            /// ```
            #[inline]
            #[cfg(feature = "std")]
            pub fn from_vec($($args: usize,)* $data: Vec<N>) -> Self {
                Self::from_vec_generic($($gargs, )* $data)
            }
        }
    }
);

// FIXME: this is not very pretty. We could find a better call syntax.
impl_constructors_from_data!(data; R, C;                  // Arguments for Matrix<N, ..., S>
                            => R: DimName, => C: DimName; // Type parameters for impl<N, ..., S>
                            R::name(), C::name();         // Arguments for `_generic` constructors.
                            ); // Arguments for non-generic constructors.

impl_constructors_from_data!(data; R, Dynamic;
                            => R: DimName;
                            R::name(), Dynamic::new(data.len() / R::dim());
                            );

impl_constructors_from_data!(data; Dynamic, C;
                            => C: DimName;
                            Dynamic::new(data.len() / C::dim()), C::name();
                            );

impl_constructors_from_data!(data; Dynamic, Dynamic;
                            ;
                            Dynamic::new(nrows), Dynamic::new(ncols);
                            nrows, ncols);



/*
 *
 * Zero, One, Rand traits.
 *
 */
impl<N, R: DimName, C: DimName> Zero for MatrixMN<N, R, C>
where
    N: Scalar + Zero + ClosedAdd,
    DefaultAllocator: Allocator<N, R, C>,
{
    #[inline]
    fn zero() -> Self {
        Self::from_element(N::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.iter().all(|e| e.is_zero())
    }
}

impl<N, D: DimName> One for MatrixN<N, D>
where
    N: Scalar + Zero + One + ClosedMul + ClosedAdd,
    DefaultAllocator: Allocator<N, D, D>,
{
    #[inline]
    fn one() -> Self {
        Self::identity()
    }
}

impl<N, R: DimName, C: DimName> Bounded for MatrixMN<N, R, C>
where
    N: Scalar + Bounded,
    DefaultAllocator: Allocator<N, R, C>,
{
    #[inline]
    fn max_value() -> Self {
        Self::from_element(N::max_value())
    }

    #[inline]
    fn min_value() -> Self {
        Self::from_element(N::min_value())
    }
}

impl<N: Scalar, R: Dim, C: Dim> Distribution<MatrixMN<N, R, C>> for Standard
where
    DefaultAllocator: Allocator<N, R, C>,
    Standard: Distribution<N>,
{
    #[inline]
    fn sample<'a, G: Rng + ?Sized>(&self, rng: &'a mut G) -> MatrixMN<N, R, C> {
        let nrows = R::try_to_usize().unwrap_or(rng.gen_range(0, 10));
        let ncols = C::try_to_usize().unwrap_or(rng.gen_range(0, 10));

        MatrixMN::from_fn_generic(R::from_usize(nrows), C::from_usize(ncols), |_, _| rng.gen())
    }
}

#[cfg(feature = "arbitrary")]
impl<N, R, C> Arbitrary for MatrixMN<N, R, C>
where
    R: Dim,
    C: Dim,
    N: Scalar + Arbitrary + Send,
    DefaultAllocator: Allocator<N, R, C>,
    Owned<N, R, C>: Clone + Send,
{
    #[inline]
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        let nrows = R::try_to_usize().unwrap_or(g.gen_range(0, 10));
        let ncols = C::try_to_usize().unwrap_or(g.gen_range(0, 10));

        Self::from_fn_generic(R::from_usize(nrows), C::from_usize(ncols), |_, _| {
            N::arbitrary(g)
        })
    }
}

#[cfg(feature = "std")]
impl<N: RealField, D: DimName> Distribution<Unit<VectorN<N, D>>> for Standard
where
    DefaultAllocator: Allocator<N, D>,
    StandardNormal: Distribution<N>,
{
    /// Generate a uniformly distributed random unit vector.
    #[inline]
    fn sample<'a, G: Rng + ?Sized>(&self, rng: &'a mut G) -> Unit<VectorN<N, D>> {
        Unit::new_normalize(VectorN::from_distribution_generic(
            D::name(),
            U1,
            &StandardNormal,
            rng,
        ))
    }
}

/*
 *
 * Constructors for small matrices and vectors.
 *
 */
macro_rules! componentwise_constructors_impl(
    ($($R: ty, $C: ty, $($args: ident:($irow: expr,$icol: expr)),*);* $(;)*) => {$(
        impl<N> MatrixMN<N, $R, $C>
            where N: Scalar,
                  DefaultAllocator: Allocator<N, $R, $C> {
            /// Initializes this matrix from its components.
            #[inline]
            pub fn new($($args: N),*) -> Self {
                unsafe {
                    let mut res = Self::new_uninitialized();
                    $( *res.get_unchecked_mut(($irow, $icol)) = $args; )*

                    res
                }
            }
        }
    )*}
);

componentwise_constructors_impl!(
    /*
     * Square matrices 1 .. 6.
     */
    U2, U2, m11:(0,0), m12:(0,1),
            m21:(1,0), m22:(1,1);
    U3, U3, m11:(0,0), m12:(0,1), m13:(0,2),
            m21:(1,0), m22:(1,1), m23:(1,2),
            m31:(2,0), m32:(2,1), m33:(2,2);
    U4, U4, m11:(0,0), m12:(0,1), m13:(0,2), m14:(0,3),
            m21:(1,0), m22:(1,1), m23:(1,2), m24:(1,3),
            m31:(2,0), m32:(2,1), m33:(2,2), m34:(2,3),
            m41:(3,0), m42:(3,1), m43:(3,2), m44:(3,3);
    U5, U5, m11:(0,0), m12:(0,1), m13:(0,2), m14:(0,3), m15:(0,4),
            m21:(1,0), m22:(1,1), m23:(1,2), m24:(1,3), m25:(1,4),
            m31:(2,0), m32:(2,1), m33:(2,2), m34:(2,3), m35:(2,4),
            m41:(3,0), m42:(3,1), m43:(3,2), m44:(3,3), m45:(3,4),
            m51:(4,0), m52:(4,1), m53:(4,2), m54:(4,3), m55:(4,4);
    U6, U6, m11:(0,0), m12:(0,1), m13:(0,2), m14:(0,3), m15:(0,4), m16:(0,5),
            m21:(1,0), m22:(1,1), m23:(1,2), m24:(1,3), m25:(1,4), m26:(1,5),
            m31:(2,0), m32:(2,1), m33:(2,2), m34:(2,3), m35:(2,4), m36:(2,5),
            m41:(3,0), m42:(3,1), m43:(3,2), m44:(3,3), m45:(3,4), m46:(3,5),
            m51:(4,0), m52:(4,1), m53:(4,2), m54:(4,3), m55:(4,4), m56:(4,5),
            m61:(5,0), m62:(5,1), m63:(5,2), m64:(5,3), m65:(5,4), m66:(5,5);

    /*
     * Rectangular matrices with 2 rows.
     */
    U2, U3, m11:(0,0), m12:(0,1), m13:(0,2),
            m21:(1,0), m22:(1,1), m23:(1,2);
    U2, U4, m11:(0,0), m12:(0,1), m13:(0,2), m14:(0,3),
            m21:(1,0), m22:(1,1), m23:(1,2), m24:(1,3);
    U2, U5, m11:(0,0), m12:(0,1), m13:(0,2), m14:(0,3), m15:(0,4),
            m21:(1,0), m22:(1,1), m23:(1,2), m24:(1,3), m25:(1,4);
    U2, U6, m11:(0,0), m12:(0,1), m13:(0,2), m14:(0,3), m15:(0,4), m16:(0,5),
            m21:(1,0), m22:(1,1), m23:(1,2), m24:(1,3), m25:(1,4), m26:(1,5);

    /*
     * Rectangular matrices with 3 rows.
     */
    U3, U2, m11:(0,0), m12:(0,1),
            m21:(1,0), m22:(1,1),
            m31:(2,0), m32:(2,1);
    U3, U4, m11:(0,0), m12:(0,1), m13:(0,2), m14:(0,3),
            m21:(1,0), m22:(1,1), m23:(1,2), m24:(1,3),
            m31:(2,0), m32:(2,1), m33:(2,2), m34:(2,3);
    U3, U5, m11:(0,0), m12:(0,1), m13:(0,2), m14:(0,3), m15:(0,4),
            m21:(1,0), m22:(1,1), m23:(1,2), m24:(1,3), m25:(1,4),
            m31:(2,0), m32:(2,1), m33:(2,2), m34:(2,3), m35:(2,4);
    U3, U6, m11:(0,0), m12:(0,1), m13:(0,2), m14:(0,3), m15:(0,4), m16:(0,5),
            m21:(1,0), m22:(1,1), m23:(1,2), m24:(1,3), m25:(1,4), m26:(1,5),
            m31:(2,0), m32:(2,1), m33:(2,2), m34:(2,3), m35:(2,4), m36:(2,5);

    /*
     * Rectangular matrices with 4 rows.
     */
    U4, U2, m11:(0,0), m12:(0,1),
            m21:(1,0), m22:(1,1),
            m31:(2,0), m32:(2,1),
            m41:(3,0), m42:(3,1);
    U4, U3, m11:(0,0), m12:(0,1), m13:(0,2),
            m21:(1,0), m22:(1,1), m23:(1,2),
            m31:(2,0), m32:(2,1), m33:(2,2),
            m41:(3,0), m42:(3,1), m43:(3,2);
    U4, U5, m11:(0,0), m12:(0,1), m13:(0,2), m14:(0,3), m15:(0,4),
            m21:(1,0), m22:(1,1), m23:(1,2), m24:(1,3), m25:(1,4),
            m31:(2,0), m32:(2,1), m33:(2,2), m34:(2,3), m35:(2,4),
            m41:(3,0), m42:(3,1), m43:(3,2), m44:(3,3), m45:(3,4);
    U4, U6, m11:(0,0), m12:(0,1), m13:(0,2), m14:(0,3), m15:(0,4), m16:(0,5),
            m21:(1,0), m22:(1,1), m23:(1,2), m24:(1,3), m25:(1,4), m26:(1,5),
            m31:(2,0), m32:(2,1), m33:(2,2), m34:(2,3), m35:(2,4), m36:(2,5),
            m41:(3,0), m42:(3,1), m43:(3,2), m44:(3,3), m45:(3,4), m46:(3,5);

    /*
     * Rectangular matrices with 5 rows.
     */
    U5, U2, m11:(0,0), m12:(0,1),
            m21:(1,0), m22:(1,1),
            m31:(2,0), m32:(2,1),
            m41:(3,0), m42:(3,1),
            m51:(4,0), m52:(4,1);
    U5, U3, m11:(0,0), m12:(0,1), m13:(0,2),
            m21:(1,0), m22:(1,1), m23:(1,2),
            m31:(2,0), m32:(2,1), m33:(2,2),
            m41:(3,0), m42:(3,1), m43:(3,2),
            m51:(4,0), m52:(4,1), m53:(4,2);
    U5, U4, m11:(0,0), m12:(0,1), m13:(0,2), m14:(0,3),
            m21:(1,0), m22:(1,1), m23:(1,2), m24:(1,3),
            m31:(2,0), m32:(2,1), m33:(2,2), m34:(2,3),
            m41:(3,0), m42:(3,1), m43:(3,2), m44:(3,3),
            m51:(4,0), m52:(4,1), m53:(4,2), m54:(4,3);
    U5, U6, m11:(0,0), m12:(0,1), m13:(0,2), m14:(0,3), m15:(0,4), m16:(0,5),
            m21:(1,0), m22:(1,1), m23:(1,2), m24:(1,3), m25:(1,4), m26:(1,5),
            m31:(2,0), m32:(2,1), m33:(2,2), m34:(2,3), m35:(2,4), m36:(2,5),
            m41:(3,0), m42:(3,1), m43:(3,2), m44:(3,3), m45:(3,4), m46:(3,5),
            m51:(4,0), m52:(4,1), m53:(4,2), m54:(4,3), m55:(4,4), m56:(4,5);

    /*
     * Rectangular matrices with 6 rows.
     */
    U6, U2, m11:(0,0), m12:(0,1),
            m21:(1,0), m22:(1,1),
            m31:(2,0), m32:(2,1),
            m41:(3,0), m42:(3,1),
            m51:(4,0), m52:(4,1),
            m61:(5,0), m62:(5,1);
    U6, U3, m11:(0,0), m12:(0,1), m13:(0,2),
            m21:(1,0), m22:(1,1), m23:(1,2),
            m31:(2,0), m32:(2,1), m33:(2,2),
            m41:(3,0), m42:(3,1), m43:(3,2),
            m51:(4,0), m52:(4,1), m53:(4,2),
            m61:(5,0), m62:(5,1), m63:(5,2);
    U6, U4, m11:(0,0), m12:(0,1), m13:(0,2), m14:(0,3),
            m21:(1,0), m22:(1,1), m23:(1,2), m24:(1,3),
            m31:(2,0), m32:(2,1), m33:(2,2), m34:(2,3),
            m41:(3,0), m42:(3,1), m43:(3,2), m44:(3,3),
            m51:(4,0), m52:(4,1), m53:(4,2), m54:(4,3),
            m61:(5,0), m62:(5,1), m63:(5,2), m64:(5,3);
    U6, U5, m11:(0,0), m12:(0,1), m13:(0,2), m14:(0,3), m15:(0,4),
            m21:(1,0), m22:(1,1), m23:(1,2), m24:(1,3), m25:(1,4),
            m31:(2,0), m32:(2,1), m33:(2,2), m34:(2,3), m35:(2,4),
            m41:(3,0), m42:(3,1), m43:(3,2), m44:(3,3), m45:(3,4),
            m51:(4,0), m52:(4,1), m53:(4,2), m54:(4,3), m55:(4,4),
            m61:(5,0), m62:(5,1), m63:(5,2), m64:(5,3), m65:(5,4);

    /*
     * Row vectors 1 .. 6.
     */
    U1, U1, x:(0,0);
    U1, U2, x:(0,0), y:(0,1);
    U1, U3, x:(0,0), y:(0,1), z:(0,2);
    U1, U4, x:(0,0), y:(0,1), z:(0,2), w:(0,3);
    U1, U5, x:(0,0), y:(0,1), z:(0,2), w:(0,3), a:(0,4);
    U1, U6, x:(0,0), y:(0,1), z:(0,2), w:(0,3), a:(0,4), b:(0,5);

    /*
     * Column vectors 1 .. 6.
     */
    U2, U1, x:(0,0), y:(1,0);
    U3, U1, x:(0,0), y:(1,0), z:(2,0);
    U4, U1, x:(0,0), y:(1,0), z:(2,0), w:(3,0);
    U5, U1, x:(0,0), y:(1,0), z:(2,0), w:(3,0), a:(4,0);
    U6, U1, x:(0,0), y:(1,0), z:(2,0), w:(3,0), a:(4,0), b:(5,0);
);

/*
 *
 * Axis constructors.
 *
 */
impl<N, R: DimName> VectorN<N, R>
where
    N: Scalar + Zero + One,
    DefaultAllocator: Allocator<N, R>,
{
    /// The column vector with a 1 as its first component, and zero elsewhere.
    #[inline]
    pub fn x() -> Self
    where R::Value: Cmp<typenum::U0, Output = Greater> {
        let mut res = Self::zeros();
        unsafe {
            *res.vget_unchecked_mut(0) = N::one();
        }

        res
    }

    /// The column vector with a 1 as its second component, and zero elsewhere.
    #[inline]
    pub fn y() -> Self
    where R::Value: Cmp<typenum::U1, Output = Greater> {
        let mut res = Self::zeros();
        unsafe {
            *res.vget_unchecked_mut(1) = N::one();
        }

        res
    }

    /// The column vector with a 1 as its third component, and zero elsewhere.
    #[inline]
    pub fn z() -> Self
    where R::Value: Cmp<typenum::U2, Output = Greater> {
        let mut res = Self::zeros();
        unsafe {
            *res.vget_unchecked_mut(2) = N::one();
        }

        res
    }

    /// The column vector with a 1 as its fourth component, and zero elsewhere.
    #[inline]
    pub fn w() -> Self
    where R::Value: Cmp<typenum::U3, Output = Greater> {
        let mut res = Self::zeros();
        unsafe {
            *res.vget_unchecked_mut(3) = N::one();
        }

        res
    }

    /// The column vector with a 1 as its fifth component, and zero elsewhere.
    #[inline]
    pub fn a() -> Self
    where R::Value: Cmp<typenum::U4, Output = Greater> {
        let mut res = Self::zeros();
        unsafe {
            *res.vget_unchecked_mut(4) = N::one();
        }

        res
    }

    /// The column vector with a 1 as its sixth component, and zero elsewhere.
    #[inline]
    pub fn b() -> Self
    where R::Value: Cmp<typenum::U5, Output = Greater> {
        let mut res = Self::zeros();
        unsafe {
            *res.vget_unchecked_mut(5) = N::one();
        }

        res
    }

    /// The unit column vector with a 1 as its first component, and zero elsewhere.
    #[inline]
    pub fn x_axis() -> Unit<Self>
    where R::Value: Cmp<typenum::U0, Output = Greater> {
        Unit::new_unchecked(Self::x())
    }

    /// The unit column vector with a 1 as its second component, and zero elsewhere.
    #[inline]
    pub fn y_axis() -> Unit<Self>
    where R::Value: Cmp<typenum::U1, Output = Greater> {
        Unit::new_unchecked(Self::y())
    }

    /// The unit column vector with a 1 as its third component, and zero elsewhere.
    #[inline]
    pub fn z_axis() -> Unit<Self>
    where R::Value: Cmp<typenum::U2, Output = Greater> {
        Unit::new_unchecked(Self::z())
    }

    /// The unit column vector with a 1 as its fourth component, and zero elsewhere.
    #[inline]
    pub fn w_axis() -> Unit<Self>
    where R::Value: Cmp<typenum::U3, Output = Greater> {
        Unit::new_unchecked(Self::w())
    }

    /// The unit column vector with a 1 as its fifth component, and zero elsewhere.
    #[inline]
    pub fn a_axis() -> Unit<Self>
    where R::Value: Cmp<typenum::U4, Output = Greater> {
        Unit::new_unchecked(Self::a())
    }

    /// The unit column vector with a 1 as its sixth component, and zero elsewhere.
    #[inline]
    pub fn b_axis() -> Unit<Self>
    where R::Value: Cmp<typenum::U5, Output = Greater> {
        Unit::new_unchecked(Self::b())
    }
}
