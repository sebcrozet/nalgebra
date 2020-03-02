use crate::{Scalar, Dim, Matrix, VectorN, RowVectorN, DefaultAllocator, U1, VectorSliceN};
use alga::general::{AdditiveMonoid, Field, SupersetOf};
use crate::storage::Storage;
use crate::allocator::Allocator;

impl<N: Scalar, R: Dim, C: Dim, S: Storage<N, R, C>> Matrix<N, R, C, S> {
    /// Returns a row vector where each element is the result of the application of `f` on the
    /// corresponding column of the original matrix.
    #[inline]
    pub fn compress_rows(&self, f: impl Fn(VectorSliceN<N, R, S::RStride, S::CStride>) -> N) -> RowVectorN<N, C>
        where DefaultAllocator: Allocator<N, U1, C> {

        let ncols = self.data.shape().1;
        let mut res = unsafe { RowVectorN::new_uninitialized_generic(U1, ncols) };

        for i in 0..ncols.value() {
            // FIXME: avoid bound checking of column.
            unsafe { *res.get_unchecked_mut((0, i)) = f(self.column(i)); }
        }

        res
    }

    /// Returns a column vector where each element is the result of the application of `f` on the
    /// corresponding column of the original matrix.
    ///
    /// This is the same as `self.compress_rows(f).transpose()`.
    #[inline]
    pub fn compress_rows_tr(&self, f: impl Fn(VectorSliceN<N, R, S::RStride, S::CStride>) -> N) -> VectorN<N, C>
        where DefaultAllocator: Allocator<N, C> {

        let ncols = self.data.shape().1;
        let mut res = unsafe { VectorN::new_uninitialized_generic(ncols, U1) };

        for i in 0..ncols.value() {
            // FIXME: avoid bound checking of column.
            unsafe { *res.vget_unchecked_mut(i) = f(self.column(i)); }
        }

        res
    }

    /// Returns a column vector resulting from the folding of `f` on each column of this matrix.
    #[inline]
    pub fn compress_columns(&self, init: VectorN<N, R>, f: impl Fn(&mut VectorN<N, R>, VectorSliceN<N, R, S::RStride, S::CStride>)) -> VectorN<N, R>
        where DefaultAllocator: Allocator<N, R> {
        let mut res = init;

        for i in 0..self.ncols() {
            f(&mut res, self.column(i))
        }

        res
    }
}

impl<N: Scalar + AdditiveMonoid, R: Dim, C: Dim, S: Storage<N, R, C>> Matrix<N, R, C, S> {
    /*
     *
     * Sum computation.
     *
     */
    /// The sum of all the elements of this matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::Matrix2x3;
    ///
    /// let m = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                        4.0, 5.0, 6.0);
    /// assert_eq!(m.sum(), 21.0);
    /// ```
    #[inline]
    pub fn sum(&self) -> N {
        self.iter().cloned().fold(N::zero(), |a, b| a + b)
    }

    /// The sum of all the rows of this matrix.
    ///
    /// Use `.row_variance_tr` if you need the result in a column vector instead.
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, Matrix3x2};
    /// # use nalgebra::{RowVector2, RowVector3};
    ///
    /// let m = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                        4.0, 5.0, 6.0);
    /// assert_eq!(m.row_sum(), RowVector3::new(5.0, 7.0, 9.0));
    ///
    /// let mint = Matrix3x2::new(1,2,3,4,5,6);
    /// assert_eq!(mint.row_sum(), RowVector2::new(9,12));
    /// ```
    #[inline]
    pub fn row_sum(&self) -> RowVectorN<N, C>
        where DefaultAllocator: Allocator<N, U1, C> {
        self.compress_rows(|col| col.sum())
    }

    /// The sum of all the rows of this matrix. The result is transposed and returned as a column vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, Matrix3x2};
    /// # use nalgebra::{Vector2, Vector3};
    ///
    /// let m = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                        4.0, 5.0, 6.0);
    /// assert_eq!(m.row_sum_tr(), Vector3::new(5.0, 7.0, 9.0));
    ///
    /// let mint = Matrix3x2::new(1,2,3,4,5,6);
    /// assert_eq!(mint.row_sum_tr(), Vector2::new(9,12));
    /// ```
    #[inline]
    pub fn row_sum_tr(&self) -> VectorN<N, C>
        where DefaultAllocator: Allocator<N, C> {
        self.compress_rows_tr(|col| col.sum())
    }

    /// The sum of all the columns of this matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, Matrix3x2};
    /// # use nalgebra::{Vector2, Vector3};
    ///
    /// let m = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                        4.0, 5.0, 6.0);
    /// assert_eq!(m.column_sum(), Vector2::new(6.0, 15.0));
    ///
    /// let mint = Matrix3x2::new(1,2,3,4,5,6);
    /// assert_eq!(mint.column_sum(), Vector3::new(3,7,11));
    /// ```
    #[inline]
    pub fn column_sum(&self) -> VectorN<N, R>
        where DefaultAllocator: Allocator<N, R> {
        let nrows = self.data.shape().0;
        self.compress_columns(VectorN::zeros_generic(nrows, U1), |out, col| {
            *out += col;
        })
    }
}

impl<N: Scalar + Field + SupersetOf<f64>, R: Dim, C: Dim, S: Storage<N, R, C>> Matrix<N, R, C, S> {
    /*
     *
     * Variance computation.
     *
     */
    /// The variance of all the elements of this matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Matrix2x3;
    ///
    /// let m = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                        4.0, 5.0, 6.0);
    /// assert_relative_eq!(m.variance(), 35.0 / 12.0, epsilon = 1.0e-8);
    /// ```
    #[inline]
    pub fn variance(&self) -> N {
        if self.len() == 0 {
            N::zero()
        } else {
            let val = self.iter().cloned().fold((N::zero(), N::zero()), |a, b| (a.0 + b.inlined_clone() * b.inlined_clone(), a.1 + b));
            let denom = N::one() / crate::convert::<_, N>(self.len() as f64);
            let vd = val.1 * denom.inlined_clone();
            val.0 * denom - vd.inlined_clone() * vd
        }
    }

    /// The variance of all the rows of this matrix.
    ///
    /// Use `.row_variance_tr` if you need the result in a column vector instead.
    /// # Example
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, RowVector3};
    ///
    /// let m = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                        4.0, 5.0, 6.0);
    /// assert_eq!(m.row_variance(), RowVector3::new(2.25, 2.25, 2.25));
    /// ```
    #[inline]
    pub fn row_variance(&self) -> RowVectorN<N, C>
        where DefaultAllocator: Allocator<N, U1, C> {
        self.compress_rows(|col| col.variance())
    }

    /// The variance of all the rows of this matrix. The result is transposed and returned as a column vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, Vector3};
    ///
    /// let m = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                        4.0, 5.0, 6.0);
    /// assert_eq!(m.row_variance_tr(), Vector3::new(2.25, 2.25, 2.25));
    /// ```
    #[inline]
    pub fn row_variance_tr(&self) -> VectorN<N, C>
        where DefaultAllocator: Allocator<N, C> {
        self.compress_rows_tr(|col| col.variance())
    }

    /// The variance of all the columns of this matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Matrix2x3, Vector2};
    ///
    /// let m = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                        4.0, 5.0, 6.0);
    /// assert_relative_eq!(m.column_variance(), Vector2::new(2.0 / 3.0, 2.0 / 3.0), epsilon = 1.0e-8);
    /// ```
    #[inline]
    pub fn column_variance(&self) -> VectorN<N, R>
        where DefaultAllocator: Allocator<N, R> {
        let (nrows, ncols) = self.data.shape();

        let mut mean = self.column_mean();
        mean.apply(|e| -(e.inlined_clone() * e));

        let denom = N::one() / crate::convert::<_, N>(ncols.value() as f64);
        self.compress_columns(mean, |out, col| {
            for i in 0..nrows.value() {
                unsafe {
                    let val = col.vget_unchecked(i);
                    *out.vget_unchecked_mut(i) += denom.inlined_clone() * val.inlined_clone() * val.inlined_clone()
                }
            }
        })
    }

    /*
     *
     * Mean computation.
     *
     */
    /// The mean of all the elements of this matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::Matrix2x3;
    ///
    /// let m = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                        4.0, 5.0, 6.0);
    /// assert_eq!(m.mean(), 3.5);
    /// ```
    #[inline]
    pub fn mean(&self) -> N {
        if self.len() == 0 {
            N::zero()
        } else {
            self.sum() / crate::convert(self.len() as f64)
        }
    }

    /// The mean of all the rows of this matrix.
    ///
    /// Use `.row_mean_tr` if you need the result in a column vector instead.
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, RowVector3};
    ///
    /// let m = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                        4.0, 5.0, 6.0);
    /// assert_eq!(m.row_mean(), RowVector3::new(2.5, 3.5, 4.5));
    /// ```
    #[inline]
    pub fn row_mean(&self) -> RowVectorN<N, C>
        where DefaultAllocator: Allocator<N, U1, C> {
        self.compress_rows(|col| col.mean())
    }

    /// The mean of all the rows of this matrix. The result is transposed and returned as a column vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, Vector3};
    ///
    /// let m = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                        4.0, 5.0, 6.0);
    /// assert_eq!(m.row_mean_tr(), Vector3::new(2.5, 3.5, 4.5));
    /// ```
    #[inline]
    pub fn row_mean_tr(&self) -> VectorN<N, C>
        where DefaultAllocator: Allocator<N, C> {
        self.compress_rows_tr(|col| col.mean())
    }

    /// The mean of all the columns of this matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, Vector2};
    ///
    /// let m = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                        4.0, 5.0, 6.0);
    /// assert_eq!(m.column_mean(), Vector2::new(2.0, 5.0));
    /// ```
    #[inline]
    pub fn column_mean(&self) -> VectorN<N, R>
        where DefaultAllocator: Allocator<N, R> {
        let (nrows, ncols) = self.data.shape();
        let denom = N::one() / crate::convert::<_, N>(ncols.value() as f64);
        self.compress_columns(VectorN::zeros_generic(nrows, U1), |out, col| {
            out.axpy(denom.inlined_clone(), &col, N::one())
        })
    }
}
