use alga::general::{ClosedAdd, ClosedMul, ComplexField};
#[cfg(feature = "std")]
use matrixmultiply;
use num::{One, Signed, Zero};
#[cfg(feature = "std")]
use std::mem;

use crate::base::allocator::Allocator;
use crate::base::constraint::{
    AreMultipliable, DimEq, SameNumberOfColumns, SameNumberOfRows, ShapeConstraint,
};
use crate::{Dim, Dynamic, U1, U2, U3, U4};
use crate::base::storage::{Storage, StorageMut};
use crate::base::{DefaultAllocator, Matrix, Scalar, SquareMatrix, Vector, DVectorSlice, VectorSliceN};


// FIXME: find a way to avoid code duplication just for complex number support.
impl<N: ComplexField, D: Dim, S: Storage<N, D>> Vector<N, D, S> {
    /// Computes the index of the vector component with the largest complex or real absolute value.
    ///
    /// # Examples:
    ///
    /// ```
    /// # extern crate num_complex;
    /// # extern crate nalgebra;
    /// # use num_complex::Complex;
    /// # use nalgebra::Vector3;
    /// let vec = Vector3::new(Complex::new(11.0, 3.0), Complex::new(-15.0, 0.0), Complex::new(13.0, 5.0));
    /// assert_eq!(vec.icamax(), 2);
    /// ```
    #[inline]
    pub fn icamax(&self) -> usize {
        assert!(!self.is_empty(), "The input vector must not be empty.");

        let mut the_max = unsafe { self.vget_unchecked(0).norm1() };
        let mut the_i = 0;

        for i in 1..self.nrows() {
            let val = unsafe { self.vget_unchecked(i).norm1() };

            if val > the_max {
                the_max = val;
                the_i = i;
            }
        }

        the_i
    }
}

impl<N: Scalar + PartialOrd, D: Dim, S: Storage<N, D>> Vector<N, D, S> {
    /// Computes the index and value of the vector component with the largest value.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let vec = Vector3::new(11, -15, 13);
    /// assert_eq!(vec.argmax(), (2, 13));
    /// ```
    #[inline]
    pub fn argmax(&self) -> (usize, N) {
        assert!(!self.is_empty(), "The input vector must not be empty.");

        let mut the_max = unsafe { self.vget_unchecked(0) };
        let mut the_i = 0;

        for i in 1..self.nrows() {
            let val = unsafe { self.vget_unchecked(i) };

            if val > the_max {
                the_max = val;
                the_i = i;
            }
        }

        (the_i, *the_max)
    }

    /// Computes the index of the vector component with the largest value.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let vec = Vector3::new(11, -15, 13);
    /// assert_eq!(vec.imax(), 2);
    /// ```
    #[inline]
    pub fn imax(&self) -> usize {
        self.argmax().0
    }

    /// Computes the index of the vector component with the largest absolute value.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let vec = Vector3::new(11, -15, 13);
    /// assert_eq!(vec.iamax(), 1);
    /// ```
    #[inline]
    pub fn iamax(&self) -> usize
        where N: Signed {
        assert!(!self.is_empty(), "The input vector must not be empty.");

        let mut the_max = unsafe { self.vget_unchecked(0).abs() };
        let mut the_i = 0;

        for i in 1..self.nrows() {
            let val = unsafe { self.vget_unchecked(i).abs() };

            if val > the_max {
                the_max = val;
                the_i = i;
            }
        }

        the_i
    }

    /// Computes the index and value of the vector component with the smallest value.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let vec = Vector3::new(11, -15, 13);
    /// assert_eq!(vec.argmin(), (1, -15));
    /// ```
    #[inline]
    pub fn argmin(&self) -> (usize, N) {
        assert!(!self.is_empty(), "The input vector must not be empty.");

        let mut the_min = unsafe { self.vget_unchecked(0) };
        let mut the_i = 0;

        for i in 1..self.nrows() {
            let val = unsafe { self.vget_unchecked(i) };

            if val < the_min {
                the_min = val;
                the_i = i;
            }
        }

        (the_i, *the_min)
    }

    /// Computes the index of the vector component with the smallest value.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let vec = Vector3::new(11, -15, 13);
    /// assert_eq!(vec.imin(), 1);
    /// ```
    #[inline]
    pub fn imin(&self) -> usize {
        self.argmin().0
    }

    /// Computes the index of the vector component with the smallest absolute value.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let vec = Vector3::new(11, -15, 13);
    /// assert_eq!(vec.iamin(), 0);
    /// ```
    #[inline]
    pub fn iamin(&self) -> usize
        where N: Signed {
        assert!(!self.is_empty(), "The input vector must not be empty.");

        let mut the_min = unsafe { self.vget_unchecked(0).abs() };
        let mut the_i = 0;

        for i in 1..self.nrows() {
            let val = unsafe { self.vget_unchecked(i).abs() };

            if val < the_min {
                the_min = val;
                the_i = i;
            }
        }

        the_i
    }
}

// FIXME: find a way to avoid code duplication just for complex number support.
impl<N: ComplexField, R: Dim, C: Dim, S: Storage<N, R, C>> Matrix<N, R, C, S> {
    /// Computes the index of the matrix component with the largest absolute value.
    ///
    /// # Examples:
    ///
    /// ```
    /// # extern crate num_complex;
    /// # extern crate nalgebra;
    /// # use num_complex::Complex;
    /// # use nalgebra::Matrix2x3;
    /// let mat = Matrix2x3::new(Complex::new(11.0, 1.0), Complex::new(-12.0, 2.0), Complex::new(13.0, 3.0),
    ///                          Complex::new(21.0, 43.0), Complex::new(22.0, 5.0), Complex::new(-23.0, 0.0));
    /// assert_eq!(mat.icamax_full(), (1, 0));
    /// ```
    #[inline]
    pub fn icamax_full(&self) -> (usize, usize) {
        assert!(!self.is_empty(), "The input matrix must not be empty.");

        let mut the_max = unsafe { self.get_unchecked((0, 0)).norm1() };
        let mut the_ij = (0, 0);

        for j in 0..self.ncols() {
            for i in 0..self.nrows() {
                let val = unsafe { self.get_unchecked((i, j)).norm1() };

                if val > the_max {
                    the_max = val;
                    the_ij = (i, j);
                }
            }
        }

        the_ij
    }
}


impl<N: Scalar + PartialOrd + Signed, R: Dim, C: Dim, S: Storage<N, R, C>> Matrix<N, R, C, S> {
    /// Computes the index of the matrix component with the largest absolute value.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// let mat = Matrix2x3::new(11, -12, 13,
    ///                          21, 22, -23);
    /// assert_eq!(mat.iamax_full(), (1, 2));
    /// ```
    #[inline]
    pub fn iamax_full(&self) -> (usize, usize) {
        assert!(!self.is_empty(), "The input matrix must not be empty.");

        let mut the_max = unsafe { self.get_unchecked((0, 0)).abs() };
        let mut the_ij = (0, 0);

        for j in 0..self.ncols() {
            for i in 0..self.nrows() {
                let val = unsafe { self.get_unchecked((i, j)).abs() };

                if val > the_max {
                    the_max = val;
                    the_ij = (i, j);
                }
            }
        }

        the_ij
    }
}

impl<N, R: Dim, C: Dim, S: Storage<N, R, C>> Matrix<N, R, C, S>
where N: Scalar + Zero + ClosedAdd + ClosedMul
{
    #[inline(always)]
    fn dotx<R2: Dim, C2: Dim, SB>(&self, rhs: &Matrix<N, R2, C2, SB>, conjugate: impl Fn(N) -> N) -> N
        where
            SB: Storage<N, R2, C2>,
            ShapeConstraint: DimEq<R, R2> + DimEq<C, C2>,
    {
        assert!(
            self.nrows() == rhs.nrows(),
            "Dot product dimensions mismatch."
        );

        // So we do some special cases for common fixed-size vectors of dimension lower than 8
        // because the `for` loop below won't be very efficient on those.
        if (R::is::<U2>() || R2::is::<U2>()) && (C::is::<U1>() || C2::is::<U1>()) {
            unsafe {
                let a = conjugate(*self.get_unchecked((0, 0))) * *rhs.get_unchecked((0, 0));
                let b = conjugate(*self.get_unchecked((1, 0))) * *rhs.get_unchecked((1, 0));

                return a + b;
            }
        }
        if (R::is::<U3>() || R2::is::<U3>()) && (C::is::<U1>() || C2::is::<U1>()) {
            unsafe {
                let a = conjugate(*self.get_unchecked((0, 0))) * *rhs.get_unchecked((0, 0));
                let b = conjugate(*self.get_unchecked((1, 0))) * *rhs.get_unchecked((1, 0));
                let c = conjugate(*self.get_unchecked((2, 0))) * *rhs.get_unchecked((2, 0));

                return a + b + c;
            }
        }
        if (R::is::<U4>() || R2::is::<U4>()) && (C::is::<U1>() || C2::is::<U1>()) {
            unsafe {
                let mut a = conjugate(*self.get_unchecked((0, 0))) * *rhs.get_unchecked((0, 0));
                let mut b = conjugate(*self.get_unchecked((1, 0))) * *rhs.get_unchecked((1, 0));
                let c = conjugate(*self.get_unchecked((2, 0))) * *rhs.get_unchecked((2, 0));
                let d = conjugate(*self.get_unchecked((3, 0))) * *rhs.get_unchecked((3, 0));

                a += c;
                b += d;

                return a + b;
            }
        }

        // All this is inspired from the "unrolled version" discussed in:
        // https://blog.theincredibleholk.org/blog/2012/12/10/optimizing-dot-product/
        //
        // And this comment from bluss:
        // https://users.rust-lang.org/t/how-to-zip-two-slices-efficiently/2048/12
        let mut res = N::zero();

        // We have to define them outside of the loop (and not inside at first assignment)
        // otherwise vectorization won't kick in for some reason.
        let mut acc0;
        let mut acc1;
        let mut acc2;
        let mut acc3;
        let mut acc4;
        let mut acc5;
        let mut acc6;
        let mut acc7;

        for j in 0..self.ncols() {
            let mut i = 0;

            acc0 = N::zero();
            acc1 = N::zero();
            acc2 = N::zero();
            acc3 = N::zero();
            acc4 = N::zero();
            acc5 = N::zero();
            acc6 = N::zero();
            acc7 = N::zero();

            while self.nrows() - i >= 8 {
                acc0 += unsafe { conjugate(*self.get_unchecked((i + 0, j))) * *rhs.get_unchecked((i + 0, j)) };
                acc1 += unsafe { conjugate(*self.get_unchecked((i + 1, j))) * *rhs.get_unchecked((i + 1, j)) };
                acc2 += unsafe { conjugate(*self.get_unchecked((i + 2, j))) * *rhs.get_unchecked((i + 2, j)) };
                acc3 += unsafe { conjugate(*self.get_unchecked((i + 3, j))) * *rhs.get_unchecked((i + 3, j)) };
                acc4 += unsafe { conjugate(*self.get_unchecked((i + 4, j))) * *rhs.get_unchecked((i + 4, j)) };
                acc5 += unsafe { conjugate(*self.get_unchecked((i + 5, j))) * *rhs.get_unchecked((i + 5, j)) };
                acc6 += unsafe { conjugate(*self.get_unchecked((i + 6, j))) * *rhs.get_unchecked((i + 6, j)) };
                acc7 += unsafe { conjugate(*self.get_unchecked((i + 7, j))) * *rhs.get_unchecked((i + 7, j)) };
                i += 8;
            }

            res += acc0 + acc4;
            res += acc1 + acc5;
            res += acc2 + acc6;
            res += acc3 + acc7;

            for k in i..self.nrows() {
                res += unsafe { conjugate(*self.get_unchecked((k, j))) * *rhs.get_unchecked((k, j)) }
            }
        }

        res
    }


    /// The dot product between two vectors or matrices (seen as vectors).
    ///
    /// This is equal to `self.transpose() * rhs`. For the sesquilinear complex dot product, use
    /// `self.dotc(rhs)`.
    ///
    /// Note that this is **not** the matrix multiplication as in, e.g., numpy. For matrix
    /// multiplication, use one of: `.gemm`, `.mul_to`, `.mul`, the `*` operator.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::{Vector3, Matrix2x3};
    /// let vec1 = Vector3::new(1.0, 2.0, 3.0);
    /// let vec2 = Vector3::new(0.1, 0.2, 0.3);
    /// assert_eq!(vec1.dot(&vec2), 1.4);
    ///
    /// let mat1 = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                           4.0, 5.0, 6.0);
    /// let mat2 = Matrix2x3::new(0.1, 0.2, 0.3,
    ///                           0.4, 0.5, 0.6);
    /// assert_eq!(mat1.dot(&mat2), 9.1);
    /// ```
    ///
    #[inline]
    pub fn dot<R2: Dim, C2: Dim, SB>(&self, rhs: &Matrix<N, R2, C2, SB>) -> N
    where
        SB: Storage<N, R2, C2>,
        ShapeConstraint: DimEq<R, R2> + DimEq<C, C2>,
    {
        self.dotx(rhs, |e| e)
    }

    /// The conjugate-linear dot product between two vectors or matrices (seen as vectors).
    ///
    /// This is equal to `self.adjoint() * rhs`.
    /// For real vectors, this is identical to `self.dot(&rhs)`.
    /// Note that this is **not** the matrix multiplication as in, e.g., numpy. For matrix
    /// multiplication, use one of: `.gemm`, `.mul_to`, `.mul`, the `*` operator.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::{Vector2, Complex};
    /// let vec1 = Vector2::new(Complex::new(1.0, 2.0), Complex::new(3.0, 4.0));
    /// let vec2 = Vector2::new(Complex::new(0.4, 0.3), Complex::new(0.2, 0.1));
    /// assert_eq!(vec1.dotc(&vec2), Complex::new(2.0, -1.0));
    ///
    /// // Note that for complex vectors, we generally have:
    /// // vec1.dotc(&vec2) != vec2.dot(&vec2)
    /// assert_ne!(vec1.dotc(&vec2), vec1.dot(&vec2));
    /// ```
    #[inline]
    pub fn dotc<R2: Dim, C2: Dim, SB>(&self, rhs: &Matrix<N, R2, C2, SB>) -> N
        where
            N: ComplexField,
            SB: Storage<N, R2, C2>,
            ShapeConstraint: DimEq<R, R2> + DimEq<C, C2>,
    {
        self.dotx(rhs, ComplexField::conjugate)
    }

    /// The dot product between the transpose of `self` and `rhs`.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::{Vector3, RowVector3, Matrix2x3, Matrix3x2};
    /// let vec1 = Vector3::new(1.0, 2.0, 3.0);
    /// let vec2 = RowVector3::new(0.1, 0.2, 0.3);
    /// assert_eq!(vec1.tr_dot(&vec2), 1.4);
    ///
    /// let mat1 = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                           4.0, 5.0, 6.0);
    /// let mat2 = Matrix3x2::new(0.1, 0.4,
    ///                           0.2, 0.5,
    ///                           0.3, 0.6);
    /// assert_eq!(mat1.tr_dot(&mat2), 9.1);
    /// ```
    #[inline]
    pub fn tr_dot<R2: Dim, C2: Dim, SB>(&self, rhs: &Matrix<N, R2, C2, SB>) -> N
    where
        SB: Storage<N, R2, C2>,
        ShapeConstraint: DimEq<C, R2> + DimEq<R, C2>,
    {
        let (nrows, ncols) = self.shape();
        assert!(
            (ncols, nrows) == rhs.shape(),
            "Transposed dot product dimension mismatch."
        );

        let mut res = N::zero();

        for j in 0..self.nrows() {
            for i in 0..self.ncols() {
                res += unsafe { *self.get_unchecked((j, i)) * *rhs.get_unchecked((i, j)) }
            }
        }

        res
    }
}

fn array_axpy<N>(y: &mut [N], a: N, x: &[N], beta: N, stride1: usize, stride2: usize, len: usize)
where N: Scalar + Zero + ClosedAdd + ClosedMul {
    for i in 0..len {
        unsafe {
            let y = y.get_unchecked_mut(i * stride1);
            *y = a * *x.get_unchecked(i * stride2) + beta * *y;
        }
    }
}

fn array_ax<N>(y: &mut [N], a: N, x: &[N], stride1: usize, stride2: usize, len: usize)
where N: Scalar + Zero + ClosedAdd + ClosedMul {
    for i in 0..len {
        unsafe {
            *y.get_unchecked_mut(i * stride1) = a * *x.get_unchecked(i * stride2);
        }
    }
}

impl<N, D: Dim, S> Vector<N, D, S>
where
    N: Scalar + Zero + ClosedAdd + ClosedMul,
    S: StorageMut<N, D>,
{
    /// Computes `self = a * x + b * self`.
    ///
    /// If `b` is zero, `self` is never read from.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let mut vec1 = Vector3::new(1.0, 2.0, 3.0);
    /// let vec2 = Vector3::new(0.1, 0.2, 0.3);
    /// vec1.axpy(10.0, &vec2, 5.0);
    /// assert_eq!(vec1, Vector3::new(6.0, 12.0, 18.0));
    /// ```
    #[inline]
    pub fn axpy<D2: Dim, SB>(&mut self, a: N, x: &Vector<N, D2, SB>, b: N)
    where
        SB: Storage<N, D2>,
        ShapeConstraint: DimEq<D, D2>,
    {
        assert_eq!(self.nrows(), x.nrows(), "Axpy: mismatched vector shapes.");

        let rstride1 = self.strides().0;
        let rstride2 = x.strides().0;

        let y = self.data.as_mut_slice();
        let x = x.data.as_slice();

        if !b.is_zero() {
            array_axpy(y, a, x, b, rstride1, rstride2, x.len());
        } else {
            array_ax(y, a, x, rstride1, rstride2, x.len());
        }
    }

    /// Computes `self = alpha * a * x + beta * self`, where `a` is a matrix, `x` a vector, and
    /// `alpha, beta` two scalars.
    ///
    /// If `beta` is zero, `self` is never read.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::{Matrix2, Vector2};
    /// let mut vec1 = Vector2::new(1.0, 2.0);
    /// let vec2 = Vector2::new(0.1, 0.2);
    /// let mat = Matrix2::new(1.0, 2.0,
    ///                        3.0, 4.0);
    /// vec1.gemv(10.0, &mat, &vec2, 5.0);
    /// assert_eq!(vec1, Vector2::new(10.0, 21.0));
    /// ```
    #[inline]
    pub fn gemv<R2: Dim, C2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: N,
        a: &Matrix<N, R2, C2, SB>,
        x: &Vector<N, D3, SC>,
        beta: N,
    ) where
        N: One,
        SB: Storage<N, R2, C2>,
        SC: Storage<N, D3>,
        ShapeConstraint: DimEq<D, R2> + AreMultipliable<R2, C2, D3, U1>,
    {
        let dim1 = self.nrows();
        let (nrows2, ncols2) = a.shape();
        let dim3 = x.nrows();

        assert!(
            ncols2 == dim3 && dim1 == nrows2,
            "Gemv: dimensions mismatch."
        );

        if ncols2 == 0 {
            // NOTE: we can't just always multiply by beta
            // because we documented the guaranty that `self` is
            // never read if `beta` is zero.
            if beta.is_zero() {
                self.fill(N::zero());
            } else {
                *self *= beta;
            }
            return;
        }

        // FIXME: avoid bound checks.
        let col2 = a.column(0);
        let val = unsafe { *x.vget_unchecked(0) };
        self.axpy(alpha * val, &col2, beta);

        for j in 1..ncols2 {
            let col2 = a.column(j);
            let val = unsafe { *x.vget_unchecked(j) };

            self.axpy(alpha * val, &col2, N::one());
        }
    }


    #[inline(always)]
    fn xxgemv<D2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: N,
        a: &SquareMatrix<N, D2, SB>,
        x: &Vector<N, D3, SC>,
        beta: N,
        dot: impl Fn(&DVectorSlice<N, SB::RStride, SB::CStride>, &DVectorSlice<N, SC::RStride, SC::CStride>) -> N,
    ) where
        N: One,
        SB: Storage<N, D2, D2>,
        SC: Storage<N, D3>,
        ShapeConstraint: DimEq<D, D2> + AreMultipliable<D2, D2, D3, U1>,
    {
        let dim1 = self.nrows();
        let dim2 = a.nrows();
        let dim3 = x.nrows();

        assert!(
            a.is_square(),
            "Symmetric cgemv: the input matrix must be square."
        );
        assert!(
            dim2 == dim3 && dim1 == dim2,
            "Symmetric cgemv: dimensions mismatch."
        );

        if dim2 == 0 {
            return;
        }

        // FIXME: avoid bound checks.
        let col2 = a.column(0);
        let val = unsafe { *x.vget_unchecked(0) };
        self.axpy(alpha * val, &col2, beta);
        self[0] += alpha * dot(&a.slice_range(1.., 0), &x.rows_range(1..));

        for j in 1..dim2 {
            let col2 = a.column(j);
            let dot = dot(&col2.rows_range(j..), &x.rows_range(j..));

            let val;
            unsafe {
                val = *x.vget_unchecked(j);
                *self.vget_unchecked_mut(j) += alpha * dot;
            }
            self.rows_range_mut(j + 1..)
                .axpy(alpha * val, &col2.rows_range(j + 1..), N::one());
        }
    }

    /// Computes `self = alpha * a * x + beta * self`, where `a` is a **symmetric** matrix, `x` a
    /// vector, and `alpha, beta` two scalars. DEPRECATED: use `sygemv` instead.
    #[inline]
    #[deprecated(note = "This is renamed `sygemv` to match the original BLAS terminology.")]
    pub fn gemv_symm<D2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: N,
        a: &SquareMatrix<N, D2, SB>,
        x: &Vector<N, D3, SC>,
        beta: N,
    ) where
        N: One,
        SB: Storage<N, D2, D2>,
        SC: Storage<N, D3>,
        ShapeConstraint: DimEq<D, D2> + AreMultipliable<D2, D2, D3, U1>,
    {
        self.sygemv(alpha, a, x, beta)
    }

    /// Computes `self = alpha * a * x + beta * self`, where `a` is a **symmetric** matrix, `x` a
    /// vector, and `alpha, beta` two scalars.
    ///
    /// For hermitian matrices, use `.hegemv` instead.
    /// If `beta` is zero, `self` is never read. If `self` is read, only its lower-triangular part
    /// (including the diagonal) is actually read.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::{Matrix2, Vector2};
    /// let mat = Matrix2::new(1.0, 2.0,
    ///                        2.0, 4.0);
    /// let mut vec1 = Vector2::new(1.0, 2.0);
    /// let vec2 = Vector2::new(0.1, 0.2);
    /// vec1.sygemv(10.0, &mat, &vec2, 5.0);
    /// assert_eq!(vec1, Vector2::new(10.0, 20.0));
    ///
    ///
    /// // The matrix upper-triangular elements can be garbage because it is never
    /// // read by this method. Therefore, it is not necessary for the caller to
    /// // fill the matrix struct upper-triangle.
    /// let mat = Matrix2::new(1.0, 9999999.9999999,
    ///                        2.0, 4.0);
    /// let mut vec1 = Vector2::new(1.0, 2.0);
    /// vec1.sygemv(10.0, &mat, &vec2, 5.0);
    /// assert_eq!(vec1, Vector2::new(10.0, 20.0));
    /// ```
    #[inline]
    pub fn sygemv<D2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: N,
        a: &SquareMatrix<N, D2, SB>,
        x: &Vector<N, D3, SC>,
        beta: N,
    ) where
        N: One,
        SB: Storage<N, D2, D2>,
        SC: Storage<N, D3>,
        ShapeConstraint: DimEq<D, D2> + AreMultipliable<D2, D2, D3, U1>,
    {
        self.xxgemv(alpha, a, x, beta, |a, b| a.dot(b))
    }

    /// Computes `self = alpha * a * x + beta * self`, where `a` is an **hermitian** matrix, `x` a
    /// vector, and `alpha, beta` two scalars.
    ///
    /// If `beta` is zero, `self` is never read. If `self` is read, only its lower-triangular part
    /// (including the diagonal) is actually read.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::{Matrix2, Vector2, Complex};
    /// let mat = Matrix2::new(Complex::new(1.0, 0.0), Complex::new(2.0, -0.1),
    ///                        Complex::new(2.0, 1.0), Complex::new(4.0, 0.0));
    /// let mut vec1 = Vector2::new(Complex::new(1.0, 2.0), Complex::new(3.0, 4.0));
    /// let vec2 = Vector2::new(Complex::new(0.1, 0.2), Complex::new(0.3, 0.4));
    /// vec1.sygemv(Complex::new(10.0, 20.0), &mat, &vec2, Complex::new(5.0, 15.0));
    /// assert_eq!(vec1, Vector2::new(Complex::new(-48.0, 44.0), Complex::new(-75.0, 110.0)));
    ///
    ///
    /// // The matrix upper-triangular elements can be garbage because it is never
    /// // read by this method. Therefore, it is not necessary for the caller to
    /// // fill the matrix struct upper-triangle.
    ///
    /// let mat = Matrix2::new(Complex::new(1.0, 0.0), Complex::new(99999999.9, 999999999.9),
    ///                        Complex::new(2.0, 1.0), Complex::new(4.0, 0.0));
    /// let mut vec1 = Vector2::new(Complex::new(1.0, 2.0), Complex::new(3.0, 4.0));
    /// let vec2 = Vector2::new(Complex::new(0.1, 0.2), Complex::new(0.3, 0.4));
    /// vec1.sygemv(Complex::new(10.0, 20.0), &mat, &vec2, Complex::new(5.0, 15.0));
    /// assert_eq!(vec1, Vector2::new(Complex::new(-48.0, 44.0), Complex::new(-75.0, 110.0)));
    /// ```
    #[inline]
    pub fn hegemv<D2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: N,
        a: &SquareMatrix<N, D2, SB>,
        x: &Vector<N, D3, SC>,
        beta: N,
    ) where
        N: ComplexField,
        SB: Storage<N, D2, D2>,
        SC: Storage<N, D3>,
        ShapeConstraint: DimEq<D, D2> + AreMultipliable<D2, D2, D3, U1>,
    {
        self.xxgemv(alpha, a, x, beta, |a, b| a.dotc(b))
    }


    #[inline(always)]
    fn gemv_xx<R2: Dim, C2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: N,
        a: &Matrix<N, R2, C2, SB>,
        x: &Vector<N, D3, SC>,
        beta: N,
        dot: impl Fn(&VectorSliceN<N, R2, SB::RStride, SB::CStride>, &Vector<N, D3, SC>) -> N,
    ) where
        N: One,
        SB: Storage<N, R2, C2>,
        SC: Storage<N, D3>,
        ShapeConstraint: DimEq<D, C2> + AreMultipliable<C2, R2, D3, U1>,
    {
        let dim1 = self.nrows();
        let (nrows2, ncols2) = a.shape();
        let dim3 = x.nrows();

        assert!(
            nrows2 == dim3 && dim1 == ncols2,
            "Gemv: dimensions mismatch."
        );

        if ncols2 == 0 {
            return;
        }

        if beta.is_zero() {
            for j in 0..ncols2 {
                let val = unsafe { self.vget_unchecked_mut(j) };
                *val = alpha * dot(&a.column(j), x)
            }
        } else {
            for j in 0..ncols2 {
                let val = unsafe { self.vget_unchecked_mut(j) };
                *val = alpha * dot(&a.column(j), x) + beta * *val;
            }
        }
    }


    /// Computes `self = alpha * a.transpose() * x + beta * self`, where `a` is a matrix, `x` a vector, and
    /// `alpha, beta` two scalars.
    ///
    /// If `beta` is zero, `self` is never read.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::{Matrix2, Vector2};
    /// let mat = Matrix2::new(1.0, 3.0,
    ///                        2.0, 4.0);
    /// let mut vec1 = Vector2::new(1.0, 2.0);
    /// let vec2 = Vector2::new(0.1, 0.2);
    /// let expected = mat.transpose() * vec2 * 10.0 + vec1 * 5.0;
    ///
    /// vec1.gemv_tr(10.0, &mat, &vec2, 5.0);
    /// assert_eq!(vec1, expected);
    /// ```
    #[inline]
    pub fn gemv_tr<R2: Dim, C2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: N,
        a: &Matrix<N, R2, C2, SB>,
        x: &Vector<N, D3, SC>,
        beta: N,
    ) where
        N: One,
        SB: Storage<N, R2, C2>,
        SC: Storage<N, D3>,
        ShapeConstraint: DimEq<D, C2> + AreMultipliable<C2, R2, D3, U1>,
    {
        self.gemv_xx(alpha, a, x, beta, |a, b| a.dot(b))
    }

    /// Computes `self = alpha * a.adjoint() * x + beta * self`, where `a` is a matrix, `x` a vector, and
    /// `alpha, beta` two scalars.
    ///
    /// For real matrices, this is the same as `.gemv_tr`.
    /// If `beta` is zero, `self` is never read.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::{Matrix2, Vector2, Complex};
    /// let mat = Matrix2::new(Complex::new(1.0, 2.0), Complex::new(3.0, 4.0),
    ///                        Complex::new(5.0, 6.0), Complex::new(7.0, 8.0));
    /// let mut vec1 = Vector2::new(Complex::new(1.0, 2.0), Complex::new(3.0, 4.0));
    /// let vec2 = Vector2::new(Complex::new(0.1, 0.2), Complex::new(0.3, 0.4));
    /// let expected = mat.adjoint() * vec2 * Complex::new(10.0, 20.0) + vec1 * Complex::new(5.0, 15.0);
    ///
    /// vec1.gemv_ad(Complex::new(10.0, 20.0), &mat, &vec2, Complex::new(5.0, 15.0));
    /// assert_eq!(vec1, expected);
    /// ```
    #[inline]
    pub fn gemv_ad<R2: Dim, C2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: N,
        a: &Matrix<N, R2, C2, SB>,
        x: &Vector<N, D3, SC>,
        beta: N,
    ) where
        N: ComplexField,
        SB: Storage<N, R2, C2>,
        SC: Storage<N, D3>,
        ShapeConstraint: DimEq<D, C2> + AreMultipliable<C2, R2, D3, U1>,
    {
        self.gemv_xx(alpha, a, x, beta, |a, b| a.dotc(b))
    }
}

impl<N, R1: Dim, C1: Dim, S: StorageMut<N, R1, C1>> Matrix<N, R1, C1, S>
where N: Scalar + Zero + ClosedAdd + ClosedMul
{
    #[inline(always)]
    fn gerx<D2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: N,
        x: &Vector<N, D2, SB>,
        y: &Vector<N, D3, SC>,
        beta: N,
        conjugate: impl Fn(N) -> N,
    ) where
        N: One,
        SB: Storage<N, D2>,
        SC: Storage<N, D3>,
        ShapeConstraint: DimEq<R1, D2> + DimEq<C1, D3>,
    {
        let (nrows1, ncols1) = self.shape();
        let dim2 = x.nrows();
        let dim3 = y.nrows();

        assert!(
            nrows1 == dim2 && ncols1 == dim3,
            "ger: dimensions mismatch."
        );

        for j in 0..ncols1 {
            // FIXME: avoid bound checks.
            let val = unsafe { conjugate(*y.vget_unchecked(j)) };
            self.column_mut(j).axpy(alpha * val, x, beta);
        }
    }

    /// Computes `self = alpha * x * y.transpose() + beta * self`.
    ///
    /// If `beta` is zero, `self` is never read.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, Vector2, Vector3};
    /// let mut mat = Matrix2x3::repeat(4.0);
    /// let vec1 = Vector2::new(1.0, 2.0);
    /// let vec2 = Vector3::new(0.1, 0.2, 0.3);
    /// let expected = vec1 * vec2.transpose() * 10.0 + mat * 5.0;
    ///
    /// mat.ger(10.0, &vec1, &vec2, 5.0);
    /// assert_eq!(mat, expected);
    /// ```
    #[inline]
    pub fn ger<D2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: N,
        x: &Vector<N, D2, SB>,
        y: &Vector<N, D3, SC>,
        beta: N,
    ) where
        N: One,
        SB: Storage<N, D2>,
        SC: Storage<N, D3>,
        ShapeConstraint: DimEq<R1, D2> + DimEq<C1, D3>,
    {
        self.gerx(alpha, x, y, beta, |e| e)
    }

    /// Computes `self = alpha * x * y.adjoint() + beta * self`.
    ///
    /// If `beta` is zero, `self` is never read.
    ///
    /// # Examples:
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Matrix2x3, Vector2, Vector3, Complex};
    /// let mut mat = Matrix2x3::repeat(Complex::new(4.0, 5.0));
    /// let vec1 = Vector2::new(Complex::new(1.0, 2.0), Complex::new(3.0, 4.0));
    /// let vec2 = Vector3::new(Complex::new(0.6, 0.5), Complex::new(0.4, 0.5), Complex::new(0.2, 0.1));
    /// let expected = vec1 * vec2.adjoint() * Complex::new(10.0, 20.0) + mat * Complex::new(5.0, 15.0);
    ///
    /// mat.gerc(Complex::new(10.0, 20.0), &vec1, &vec2, Complex::new(5.0, 15.0));
    /// assert_eq!(mat, expected);
    /// ```
    #[inline]
    pub fn gerc<D2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: N,
        x: &Vector<N, D2, SB>,
        y: &Vector<N, D3, SC>,
        beta: N,
    ) where
        N: ComplexField,
        SB: Storage<N, D2>,
        SC: Storage<N, D3>,
        ShapeConstraint: DimEq<R1, D2> + DimEq<C1, D3>,
    {
        self.gerx(alpha, x, y, beta, ComplexField::conjugate)
    }

    /// Computes `self = alpha * a * b + beta * self`, where `a, b, self` are matrices.
    /// `alpha` and `beta` are scalar.
    ///
    /// If `beta` is zero, `self` is never read.
    ///
    /// # Examples:
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Matrix2x3, Matrix3x4, Matrix2x4};
    /// let mut mat1 = Matrix2x4::identity();
    /// let mat2 = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                           4.0, 5.0, 6.0);
    /// let mat3 = Matrix3x4::new(0.1, 0.2, 0.3, 0.4,
    ///                           0.5, 0.6, 0.7, 0.8,
    ///                           0.9, 1.0, 1.1, 1.2);
    /// let expected = mat2 * mat3 * 10.0 + mat1 * 5.0;
    ///
    /// mat1.gemm(10.0, &mat2, &mat3, 5.0);
    /// assert_relative_eq!(mat1, expected);
    /// ```
    #[inline]
    pub fn gemm<R2: Dim, C2: Dim, R3: Dim, C3: Dim, SB, SC>(
        &mut self,
        alpha: N,
        a: &Matrix<N, R2, C2, SB>,
        b: &Matrix<N, R3, C3, SC>,
        beta: N,
    ) where
        N: One,
        SB: Storage<N, R2, C2>,
        SC: Storage<N, R3, C3>,
        ShapeConstraint: SameNumberOfRows<R1, R2>
            + SameNumberOfColumns<C1, C3>
            + AreMultipliable<R2, C2, R3, C3>,
    {
        let ncols1 = self.ncols();

        #[cfg(feature = "std")]
        {
            // We assume large matrices will be Dynamic but small matrices static.
            // We could use matrixmultiply for large statically-sized matrices but the performance
            // threshold to activate it would be different from SMALL_DIM because our code optimizes
            // better for statically-sized matrices.
            if R1::is::<Dynamic>()
                || C1::is::<Dynamic>()
                || R2::is::<Dynamic>()
                || C2::is::<Dynamic>()
                || R3::is::<Dynamic>()
                || C3::is::<Dynamic>() {
                // matrixmultiply can be used only if the std feature is available.
                let nrows1 = self.nrows();
                let (nrows2, ncols2) = a.shape();
                let (nrows3, ncols3) = b.shape();

                // Threshold determined empirically.
                const SMALL_DIM: usize = 5;

                if nrows1 > SMALL_DIM
                    && ncols1 > SMALL_DIM
                    && nrows2 > SMALL_DIM
                    && ncols2 > SMALL_DIM
                {
                    assert_eq!(
                        ncols2, nrows3,
                        "gemm: dimensions mismatch for multiplication."
                    );
                    assert_eq!(
                        (nrows1, ncols1),
                        (nrows2, ncols3),
                        "gemm: dimensions mismatch for addition."
                    );

                    // NOTE: this case should never happen because we enter this
                    // codepath only when ncols2 > SMALL_DIM. Though we keep this
                    // here just in case if in the future we change the conditions to
                    // enter this codepath.
                    if ncols2 == 0 {
                        // NOTE: we can't just always multiply by beta
                        // because we documented the guaranty that `self` is
                        // never read if `beta` is zero.
                        if beta.is_zero() {
                            self.fill(N::zero());
                        } else {
                            *self *= beta;
                        }
                        return;
                    }

                    if N::is::<f32>() {
                        let (rsa, csa) = a.strides();
                        let (rsb, csb) = b.strides();
                        let (rsc, csc) = self.strides();

                        unsafe {
                            matrixmultiply::sgemm(
                                nrows2,
                                ncols2,
                                ncols3,
                                mem::transmute_copy(&alpha),
                                a.data.ptr() as *const f32,
                                rsa as isize,
                                csa as isize,
                                b.data.ptr() as *const f32,
                                rsb as isize,
                                csb as isize,
                                mem::transmute_copy(&beta),
                                self.data.ptr_mut() as *mut f32,
                                rsc as isize,
                                csc as isize,
                            );
                        }
                        return;
                    } else if N::is::<f64>() {
                        let (rsa, csa) = a.strides();
                        let (rsb, csb) = b.strides();
                        let (rsc, csc) = self.strides();

                        unsafe {
                            matrixmultiply::dgemm(
                                nrows2,
                                ncols2,
                                ncols3,
                                mem::transmute_copy(&alpha),
                                a.data.ptr() as *const f64,
                                rsa as isize,
                                csa as isize,
                                b.data.ptr() as *const f64,
                                rsb as isize,
                                csb as isize,
                                mem::transmute_copy(&beta),
                                self.data.ptr_mut() as *mut f64,
                                rsc as isize,
                                csc as isize,
                            );
                        }
                        return;
                    }
                }
            }
        }


        for j1 in 0..ncols1 {
            // FIXME: avoid bound checks.
            self.column_mut(j1).gemv(alpha, a, &b.column(j1), beta);
        }
    }

    /// Computes `self = alpha * a.transpose() * b + beta * self`, where `a, b, self` are matrices.
    /// `alpha` and `beta` are scalar.
    ///
    /// If `beta` is zero, `self` is never read.
    ///
    /// # Examples:
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Matrix3x2, Matrix3x4, Matrix2x4};
    /// let mut mat1 = Matrix2x4::identity();
    /// let mat2 = Matrix3x2::new(1.0, 4.0,
    ///                           2.0, 5.0,
    ///                           3.0, 6.0);
    /// let mat3 = Matrix3x4::new(0.1, 0.2, 0.3, 0.4,
    ///                           0.5, 0.6, 0.7, 0.8,
    ///                           0.9, 1.0, 1.1, 1.2);
    /// let expected = mat2.transpose() * mat3 * 10.0 + mat1 * 5.0;
    ///
    /// mat1.gemm_tr(10.0, &mat2, &mat3, 5.0);
    /// assert_eq!(mat1, expected);
    /// ```
    #[inline]
    pub fn gemm_tr<R2: Dim, C2: Dim, R3: Dim, C3: Dim, SB, SC>(
        &mut self,
        alpha: N,
        a: &Matrix<N, R2, C2, SB>,
        b: &Matrix<N, R3, C3, SC>,
        beta: N,
    ) where
        N: One,
        SB: Storage<N, R2, C2>,
        SC: Storage<N, R3, C3>,
        ShapeConstraint: SameNumberOfRows<R1, C2>
            + SameNumberOfColumns<C1, C3>
            + AreMultipliable<C2, R2, R3, C3>,
    {
        let (nrows1, ncols1) = self.shape();
        let (nrows2, ncols2) = a.shape();
        let (nrows3, ncols3) = b.shape();

        assert_eq!(
            nrows2, nrows3,
            "gemm: dimensions mismatch for multiplication."
        );
        assert_eq!(
            (nrows1, ncols1),
            (ncols2, ncols3),
            "gemm: dimensions mismatch for addition."
        );

        for j1 in 0..ncols1 {
            // FIXME: avoid bound checks.
            self.column_mut(j1).gemv_tr(alpha, a, &b.column(j1), beta);
        }
    }


    /// Computes `self = alpha * a.adjoint() * b + beta * self`, where `a, b, self` are matrices.
    /// `alpha` and `beta` are scalar.
    ///
    /// If `beta` is zero, `self` is never read.
    ///
    /// # Examples:
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Matrix3x2, Matrix3x4, Matrix2x4, Complex};
    /// let mut mat1 = Matrix2x4::identity();
    /// let mat2 = Matrix3x2::new(Complex::new(1.0, 4.0), Complex::new(7.0, 8.0),
    ///                           Complex::new(2.0, 5.0), Complex::new(9.0, 10.0),
    ///                           Complex::new(3.0, 6.0), Complex::new(11.0, 12.0));
    /// let mat3 = Matrix3x4::new(Complex::new(0.1, 1.3), Complex::new(0.2, 1.4), Complex::new(0.3, 1.5), Complex::new(0.4, 1.6),
    ///                           Complex::new(0.5, 1.7), Complex::new(0.6, 1.8), Complex::new(0.7, 1.9), Complex::new(0.8, 2.0),
    ///                           Complex::new(0.9, 2.1), Complex::new(1.0, 2.2), Complex::new(1.1, 2.3), Complex::new(1.2, 2.4));
    /// let expected = mat2.adjoint() * mat3 * Complex::new(10.0, 20.0) + mat1 * Complex::new(5.0, 15.0);
    ///
    /// mat1.gemm_ad(Complex::new(10.0, 20.0), &mat2, &mat3, Complex::new(5.0, 15.0));
    /// assert_eq!(mat1, expected);
    /// ```
    #[inline]
    pub fn gemm_ad<R2: Dim, C2: Dim, R3: Dim, C3: Dim, SB, SC>(
        &mut self,
        alpha: N,
        a: &Matrix<N, R2, C2, SB>,
        b: &Matrix<N, R3, C3, SC>,
        beta: N,
    ) where
        N: ComplexField,
        SB: Storage<N, R2, C2>,
        SC: Storage<N, R3, C3>,
        ShapeConstraint: SameNumberOfRows<R1, C2>
        + SameNumberOfColumns<C1, C3>
        + AreMultipliable<C2, R2, R3, C3>,
    {
        let (nrows1, ncols1) = self.shape();
        let (nrows2, ncols2) = a.shape();
        let (nrows3, ncols3) = b.shape();

        assert_eq!(
            nrows2, nrows3,
            "gemm: dimensions mismatch for multiplication."
        );
        assert_eq!(
            (nrows1, ncols1),
            (ncols2, ncols3),
            "gemm: dimensions mismatch for addition."
        );

        for j1 in 0..ncols1 {
            // FIXME: avoid bound checks.
            self.column_mut(j1).gemv_ad(alpha, a, &b.column(j1), beta);
        }
    }
}

impl<N, R1: Dim, C1: Dim, S: StorageMut<N, R1, C1>> Matrix<N, R1, C1, S>
where N: Scalar + Zero + ClosedAdd + ClosedMul
{
    #[inline(always)]
    fn xxgerx<D2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: N,
        x: &Vector<N, D2, SB>,
        y: &Vector<N, D3, SC>,
        beta: N,
        conjugate: impl Fn(N) -> N,
    ) where
        N: One,
        SB: Storage<N, D2>,
        SC: Storage<N, D3>,
        ShapeConstraint: DimEq<R1, D2> + DimEq<C1, D3>,
    {
        let dim1 = self.nrows();
        let dim2 = x.nrows();
        let dim3 = y.nrows();

        assert!(
            self.is_square(),
            "Symmetric ger: the input matrix must be square."
        );
        assert!(dim1 == dim2 && dim1 == dim3, "ger: dimensions mismatch.");

        for j in 0..dim1 {
            let val = unsafe { conjugate(*y.vget_unchecked(j)) };
            let subdim = Dynamic::new(dim1 - j);
            // FIXME: avoid bound checks.
            self.generic_slice_mut((j, j), (subdim, U1)).axpy(
                alpha * val,
                &x.rows_range(j..),
                beta,
            );
        }
    }

    /// Computes `self = alpha * x * y.transpose() + beta * self`, where `self` is a **symmetric**
    /// matrix.
    ///
    /// If `beta` is zero, `self` is never read. The result is symmetric. Only the lower-triangular
    /// (including the diagonal) part of `self` is read/written.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::{Matrix2, Vector2};
    /// let mut mat = Matrix2::identity();
    /// let vec1 = Vector2::new(1.0, 2.0);
    /// let vec2 = Vector2::new(0.1, 0.2);
    /// let expected = vec1 * vec2.transpose() * 10.0 + mat * 5.0;
    /// mat.m12 = 99999.99999; // This component is on the upper-triangular part and will not be read/written.
    ///
    /// mat.ger_symm(10.0, &vec1, &vec2, 5.0);
    /// assert_eq!(mat.lower_triangle(), expected.lower_triangle());
    /// assert_eq!(mat.m12, 99999.99999); // This was untouched.
    #[inline]
    #[deprecated(note = "This is renamed `syger` to match the original BLAS terminology.")]
    pub fn ger_symm<D2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: N,
        x: &Vector<N, D2, SB>,
        y: &Vector<N, D3, SC>,
        beta: N,
    ) where
        N: One,
        SB: Storage<N, D2>,
        SC: Storage<N, D3>,
        ShapeConstraint: DimEq<R1, D2> + DimEq<C1, D3>,
    {
        self.syger(alpha, x, y, beta)
    }

    /// Computes `self = alpha * x * y.transpose() + beta * self`, where `self` is a **symmetric**
    /// matrix.
    ///
    /// For hermitian complex matrices, use `.hegerc` instead.
    /// If `beta` is zero, `self` is never read. The result is symmetric. Only the lower-triangular
    /// (including the diagonal) part of `self` is read/written.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::{Matrix2, Vector2};
    /// let mut mat = Matrix2::identity();
    /// let vec1 = Vector2::new(1.0, 2.0);
    /// let vec2 = Vector2::new(0.1, 0.2);
    /// let expected = vec1 * vec2.transpose() * 10.0 + mat * 5.0;
    /// mat.m12 = 99999.99999; // This component is on the upper-triangular part and will not be read/written.
    ///
    /// mat.syger(10.0, &vec1, &vec2, 5.0);
    /// assert_eq!(mat.lower_triangle(), expected.lower_triangle());
    /// assert_eq!(mat.m12, 99999.99999); // This was untouched.
    #[inline]
    pub fn syger<D2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: N,
        x: &Vector<N, D2, SB>,
        y: &Vector<N, D3, SC>,
        beta: N,
    ) where
        N: One,
        SB: Storage<N, D2>,
        SC: Storage<N, D3>,
        ShapeConstraint: DimEq<R1, D2> + DimEq<C1, D3>,
    {
        self.xxgerx(alpha, x, y, beta, |e| e)
    }

    /// Computes `self = alpha * x * y.adjoint() + beta * self`, where `self` is an **hermitian**
    /// matrix.
    ///
    /// If `beta` is zero, `self` is never read. The result is symmetric. Only the lower-triangular
    /// (including the diagonal) part of `self` is read/written.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::{Matrix2, Vector2, Complex};
    /// let mut mat = Matrix2::identity();
    /// let vec1 = Vector2::new(Complex::new(1.0, 3.0), Complex::new(2.0, 4.0));
    /// let vec2 = Vector2::new(Complex::new(0.2, 0.4), Complex::new(0.1, 0.3));
    /// let expected = vec1 * vec2.adjoint() * Complex::new(10.0, 20.0) + mat * Complex::new(5.0, 15.0);
    /// mat.m12 = Complex::new(99999.99999, 88888.88888); // This component is on the upper-triangular part and will not be read/written.
    ///
    /// mat.hegerc(Complex::new(10.0, 20.0), &vec1, &vec2, Complex::new(5.0, 15.0));
    /// assert_eq!(mat.lower_triangle(), expected.lower_triangle());
    /// assert_eq!(mat.m12, Complex::new(99999.99999, 88888.88888)); // This was untouched.
    #[inline]
    pub fn hegerc<D2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: N,
        x: &Vector<N, D2, SB>,
        y: &Vector<N, D3, SC>,
        beta: N,
    ) where
        N: ComplexField,
        SB: Storage<N, D2>,
        SC: Storage<N, D3>,
        ShapeConstraint: DimEq<R1, D2> + DimEq<C1, D3>,
    {
        self.xxgerx(alpha, x, y, beta, ComplexField::conjugate)
    }
}

impl<N, D1: Dim, S: StorageMut<N, D1, D1>> SquareMatrix<N, D1, S>
where N: Scalar + Zero + One + ClosedAdd + ClosedMul
{
    /// Computes the quadratic form `self = alpha * lhs * mid * lhs.transpose() + beta * self`.
    ///
    /// This uses the provided workspace `work` to avoid allocations for intermediate results.
    ///
    /// # Examples:
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{DMatrix, DVector};
    /// // Note that all those would also work with statically-sized matrices.
    /// // We use DMatrix/DVector since that's the only case where pre-allocating the
    /// // workspace is actually useful (assuming the same workspace is re-used for
    /// // several computations) because it avoids repeated dynamic allocations.
    /// let mut mat = DMatrix::identity(2, 2);
    /// let lhs = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0,
    ///                                           4.0, 5.0, 6.0]);
    /// let mid = DMatrix::from_row_slice(3, 3, &[0.1, 0.2, 0.3,
    ///                                           0.5, 0.6, 0.7,
    ///                                           0.9, 1.0, 1.1]);
    /// // The random shows that values on the workspace do not
    /// // matter as they will be overwritten.
    /// let mut workspace = DVector::new_random(2);
    /// let expected = &lhs * &mid * lhs.transpose() * 10.0 + &mat * 5.0;
    ///
    /// mat.quadform_tr_with_workspace(&mut workspace, 10.0, &lhs, &mid, 5.0);
    /// assert_relative_eq!(mat, expected);
    pub fn quadform_tr_with_workspace<D2, S2, R3, C3, S3, D4, S4>(
        &mut self,
        work: &mut Vector<N, D2, S2>,
        alpha: N,
        lhs: &Matrix<N, R3, C3, S3>,
        mid: &SquareMatrix<N, D4, S4>,
        beta: N,
    ) where
        D2: Dim,
        R3: Dim,
        C3: Dim,
        D4: Dim,
        S2: StorageMut<N, D2>,
        S3: Storage<N, R3, C3>,
        S4: Storage<N, D4, D4>,
        ShapeConstraint: DimEq<D1, D2> + DimEq<D1, R3> + DimEq<D2, R3> + DimEq<C3, D4>,
    {
        work.gemv(N::one(), lhs, &mid.column(0), N::zero());
        self.ger(alpha, work, &lhs.column(0), beta);

        for j in 1..mid.ncols() {
            work.gemv(N::one(), lhs, &mid.column(j), N::zero());
            self.ger(alpha, work, &lhs.column(j), N::one());
        }
    }

    /// Computes the quadratic form `self = alpha * lhs * mid * lhs.transpose() + beta * self`.
    ///
    /// This allocates a workspace vector of dimension D1 for intermediate results.
    /// If `D1` is a type-level integer, then the allocation is performed on the stack.
    /// Use `.quadform_tr_with_workspace(...)` instead to avoid allocations.
    ///
    /// # Examples:
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Matrix2, Matrix3, Matrix2x3, Vector2};
    /// let mut mat = Matrix2::identity();
    /// let lhs = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                          4.0, 5.0, 6.0);
    /// let mid = Matrix3::new(0.1, 0.2, 0.3,
    ///                        0.5, 0.6, 0.7,
    ///                        0.9, 1.0, 1.1);
    /// let expected = lhs * mid * lhs.transpose() * 10.0 + mat * 5.0;
    ///
    /// mat.quadform_tr(10.0, &lhs, &mid, 5.0);
    /// assert_relative_eq!(mat, expected);
    pub fn quadform_tr<R3, C3, S3, D4, S4>(
        &mut self,
        alpha: N,
        lhs: &Matrix<N, R3, C3, S3>,
        mid: &SquareMatrix<N, D4, S4>,
        beta: N,
    ) where
        R3: Dim,
        C3: Dim,
        D4: Dim,
        S3: Storage<N, R3, C3>,
        S4: Storage<N, D4, D4>,
        ShapeConstraint: DimEq<D1, D1> + DimEq<D1, R3> + DimEq<C3, D4>,
        DefaultAllocator: Allocator<N, D1>,
    {
        let mut work = unsafe { Vector::new_uninitialized_generic(self.data.shape().0, U1) };
        self.quadform_tr_with_workspace(&mut work, alpha, lhs, mid, beta)
    }

    /// Computes the quadratic form `self = alpha * rhs.transpose() * mid * rhs + beta * self`.
    ///
    /// This uses the provided workspace `work` to avoid allocations for intermediate results.
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{DMatrix, DVector};
    /// // Note that all those would also work with statically-sized matrices.
    /// // We use DMatrix/DVector since that's the only case where pre-allocating the
    /// // workspace is actually useful (assuming the same workspace is re-used for
    /// // several computations) because it avoids repeated dynamic allocations.
    /// let mut mat = DMatrix::identity(2, 2);
    /// let rhs = DMatrix::from_row_slice(3, 2, &[1.0, 2.0,
    ///                                           3.0, 4.0,
    ///                                           5.0, 6.0]);
    /// let mid = DMatrix::from_row_slice(3, 3, &[0.1, 0.2, 0.3,
    ///                                           0.5, 0.6, 0.7,
    ///                                           0.9, 1.0, 1.1]);
    /// // The random shows that values on the workspace do not
    /// // matter as they will be overwritten.
    /// let mut workspace = DVector::new_random(3);
    /// let expected = rhs.transpose() * &mid * &rhs * 10.0 + &mat * 5.0;
    ///
    /// mat.quadform_with_workspace(&mut workspace, 10.0, &mid, &rhs, 5.0);
    /// assert_relative_eq!(mat, expected);
    pub fn quadform_with_workspace<D2, S2, D3, S3, R4, C4, S4>(
        &mut self,
        work: &mut Vector<N, D2, S2>,
        alpha: N,
        mid: &SquareMatrix<N, D3, S3>,
        rhs: &Matrix<N, R4, C4, S4>,
        beta: N,
    ) where
        D2: Dim,
        D3: Dim,
        R4: Dim,
        C4: Dim,
        S2: StorageMut<N, D2>,
        S3: Storage<N, D3, D3>,
        S4: Storage<N, R4, C4>,
        ShapeConstraint:
            DimEq<D3, R4> + DimEq<D1, C4> + DimEq<D2, D3> + AreMultipliable<C4, R4, D2, U1>,
    {
        work.gemv(N::one(), mid, &rhs.column(0), N::zero());
        self.column_mut(0).gemv_tr(alpha, &rhs, work, beta);

        for j in 1..rhs.ncols() {
            work.gemv(N::one(), mid, &rhs.column(j), N::zero());
            self.column_mut(j).gemv_tr(alpha, &rhs, work, beta);
        }
    }

    /// Computes the quadratic form `self = alpha * rhs.transpose() * mid * rhs + beta * self`.
    ///
    /// This allocates a workspace vector of dimension D2 for intermediate results.
    /// If `D2` is a type-level integer, then the allocation is performed on the stack.
    /// Use `.quadform_with_workspace(...)` instead to avoid allocations.
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Matrix2, Matrix3x2, Matrix3};
    /// let mut mat = Matrix2::identity();
    /// let rhs = Matrix3x2::new(1.0, 2.0,
    ///                          3.0, 4.0,
    ///                          5.0, 6.0);
    /// let mid = Matrix3::new(0.1, 0.2, 0.3,
    ///                        0.5, 0.6, 0.7,
    ///                        0.9, 1.0, 1.1);
    /// let expected = rhs.transpose() * mid * rhs * 10.0 + mat * 5.0;
    ///
    /// mat.quadform(10.0, &mid, &rhs, 5.0);
    /// assert_relative_eq!(mat, expected);
    pub fn quadform<D2, S2, R3, C3, S3>(
        &mut self,
        alpha: N,
        mid: &SquareMatrix<N, D2, S2>,
        rhs: &Matrix<N, R3, C3, S3>,
        beta: N,
    ) where
        D2: Dim,
        R3: Dim,
        C3: Dim,
        S2: Storage<N, D2, D2>,
        S3: Storage<N, R3, C3>,
        ShapeConstraint: DimEq<D2, R3> + DimEq<D1, C3> + AreMultipliable<C3, R3, D2, U1>,
        DefaultAllocator: Allocator<N, D2>,
    {
        let mut work = unsafe { Vector::new_uninitialized_generic(mid.data.shape().0, U1) };
        self.quadform_with_workspace(&mut work, alpha, mid, rhs, beta)
    }
}
