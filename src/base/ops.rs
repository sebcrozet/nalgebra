use num::{One, Signed, Zero};
use std::cmp::{PartialOrd, Ordering};
use std::iter;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use alga::general::{ComplexField, ClosedAdd, ClosedDiv, ClosedMul, ClosedNeg, ClosedSub};

use crate::base::allocator::{Allocator, SameShapeAllocator, SameShapeC, SameShapeR};
use crate::base::constraint::{
    AreMultipliable, DimEq, SameNumberOfColumns, SameNumberOfRows, ShapeConstraint,
};
use crate::base::dimension::{Dim, DimMul, DimName, DimProd, Dynamic};
use crate::base::storage::{ContiguousStorageMut, Storage, StorageMut};
use crate::base::{DefaultAllocator, Matrix, MatrixMN, MatrixN, MatrixSum, Scalar, VectorSliceN};

/*
 *
 * Indexing.
 *
 */
impl<N: Scalar, R: Dim, C: Dim, S: Storage<N, R, C>> Index<usize> for Matrix<N, R, C, S> {
    type Output = N;

    #[inline]
    fn index(&self, i: usize) -> &Self::Output {
        let ij = self.vector_to_matrix_index(i);
        &self[ij]
    }
}

impl<N, R: Dim, C: Dim, S> Index<(usize, usize)> for Matrix<N, R, C, S>
where
    N: Scalar,
    S: Storage<N, R, C>,
{
    type Output = N;

    #[inline]
    fn index(&self, ij: (usize, usize)) -> &Self::Output {
        let shape = self.shape();
        assert!(
            ij.0 < shape.0 && ij.1 < shape.1,
            "Matrix index out of bounds."
        );

        unsafe { self.get_unchecked((ij.0, ij.1)) }
    }
}

// Mutable versions.
impl<N: Scalar, R: Dim, C: Dim, S: StorageMut<N, R, C>> IndexMut<usize> for Matrix<N, R, C, S> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut N {
        let ij = self.vector_to_matrix_index(i);
        &mut self[ij]
    }
}

impl<N, R: Dim, C: Dim, S> IndexMut<(usize, usize)> for Matrix<N, R, C, S>
where
    N: Scalar,
    S: StorageMut<N, R, C>,
{
    #[inline]
    fn index_mut(&mut self, ij: (usize, usize)) -> &mut N {
        let shape = self.shape();
        assert!(
            ij.0 < shape.0 && ij.1 < shape.1,
            "Matrix index out of bounds."
        );

        unsafe { self.get_unchecked_mut((ij.0, ij.1)) }
    }
}

/*
 *
 * Neg
 *
 */
impl<N, R: Dim, C: Dim, S> Neg for Matrix<N, R, C, S>
where
    N: Scalar + ClosedNeg,
    S: Storage<N, R, C>,
    DefaultAllocator: Allocator<N, R, C>,
{
    type Output = MatrixMN<N, R, C>;

    #[inline]
    fn neg(self) -> Self::Output {
        let mut res = self.into_owned();
        res.neg_mut();
        res
    }
}

impl<'a, N, R: Dim, C: Dim, S> Neg for &'a Matrix<N, R, C, S>
where
    N: Scalar + ClosedNeg,
    S: Storage<N, R, C>,
    DefaultAllocator: Allocator<N, R, C>,
{
    type Output = MatrixMN<N, R, C>;

    #[inline]
    fn neg(self) -> Self::Output {
        -self.clone_owned()
    }
}

impl<N, R: Dim, C: Dim, S> Matrix<N, R, C, S>
where
    N: Scalar + ClosedNeg,
    S: StorageMut<N, R, C>,
{
    /// Negates `self` in-place.
    #[inline]
    pub fn neg_mut(&mut self) {
        for e in self.iter_mut() {
            *e = -e.inlined_clone()
        }
    }
}

/*
 *
 * Addition & Subtraction
 *
 */

macro_rules! componentwise_binop_impl(
    ($Trait: ident, $method: ident, $bound: ident;
     $TraitAssign: ident, $method_assign: ident, $method_assign_statically_unchecked: ident,
     $method_assign_statically_unchecked_rhs: ident;
     $method_to: ident, $method_to_statically_unchecked: ident) => {

        impl<N, R1: Dim, C1: Dim, SA: Storage<N, R1, C1>> Matrix<N, R1, C1, SA>
            where N: Scalar + $bound {

            /*
             *
             * Methods without dimension checking at compile-time.
             * This is useful for code reuse because the sum representative system does not plays
             * easily with static checks.
             *
             */
            #[inline]
            fn $method_to_statically_unchecked<R2: Dim, C2: Dim, SB,
                                               R3: Dim, C3: Dim, SC>(&self,
                                                                     rhs: &Matrix<N, R2, C2, SB>,
                                                                     out: &mut Matrix<N, R3, C3, SC>)
                where SB: Storage<N, R2, C2>,
                      SC: StorageMut<N, R3, C3> {
                assert!(self.shape() == rhs.shape(), "Matrix addition/subtraction dimensions mismatch.");
                assert!(self.shape() == out.shape(), "Matrix addition/subtraction output dimensions mismatch.");

                // This is the most common case and should be deduced at compile-time.
                // FIXME: use specialization instead?
                if self.data.is_contiguous() && rhs.data.is_contiguous() && out.data.is_contiguous() {
                    let arr1 = self.data.as_slice();
                    let arr2 = rhs.data.as_slice();
                    let out  = out.data.as_mut_slice();
                    for i in 0 .. arr1.len() {
                        unsafe {
                            *out.get_unchecked_mut(i) = arr1.get_unchecked(i).inlined_clone().$method(arr2.get_unchecked(i).inlined_clone());
                        }
                    }
                }
                else {
                    for j in 0 .. self.ncols() {
                        for i in 0 .. self.nrows() {
                            unsafe {
                                let val = self.get_unchecked((i, j)).inlined_clone().$method(rhs.get_unchecked((i, j)).inlined_clone());
                                *out.get_unchecked_mut((i, j)) = val;
                            }
                        }
                    }
                }
            }


            #[inline]
            fn $method_assign_statically_unchecked<R2, C2, SB>(&mut self, rhs: &Matrix<N, R2, C2, SB>)
                where R2: Dim,
                      C2: Dim,
                      SA: StorageMut<N, R1, C1>,
                      SB: Storage<N, R2, C2> {
                assert!(self.shape() == rhs.shape(), "Matrix addition/subtraction dimensions mismatch.");

                // This is the most common case and should be deduced at compile-time.
                // FIXME: use specialization instead?
                if self.data.is_contiguous() && rhs.data.is_contiguous() {
                    let arr1 = self.data.as_mut_slice();
                    let arr2 = rhs.data.as_slice();
                    for i in 0 .. arr2.len() {
                        unsafe {
                            arr1.get_unchecked_mut(i).$method_assign(arr2.get_unchecked(i).inlined_clone());
                        }
                    }
                }
                else {
                    for j in 0 .. rhs.ncols() {
                        for i in 0 .. rhs.nrows() {
                            unsafe {
                                self.get_unchecked_mut((i, j)).$method_assign(rhs.get_unchecked((i, j)).inlined_clone())
                            }
                        }
                    }
                }
            }


            #[inline]
            fn $method_assign_statically_unchecked_rhs<R2, C2, SB>(&self, rhs: &mut Matrix<N, R2, C2, SB>)
                where R2: Dim,
                      C2: Dim,
                      SB: StorageMut<N, R2, C2> {
                assert!(self.shape() == rhs.shape(), "Matrix addition/subtraction dimensions mismatch.");

                // This is the most common case and should be deduced at compile-time.
                // FIXME: use specialization instead?
                if self.data.is_contiguous() && rhs.data.is_contiguous() {
                    let arr1 = self.data.as_slice();
                    let arr2 = rhs.data.as_mut_slice();
                    for i in 0 .. arr1.len() {
                        unsafe {
                            let res = arr1.get_unchecked(i).inlined_clone().$method(arr2.get_unchecked(i).inlined_clone());
                            *arr2.get_unchecked_mut(i) = res;
                        }
                    }
                }
                else {
                    for j in 0 .. self.ncols() {
                        for i in 0 .. self.nrows() {
                            unsafe {
                                let r = rhs.get_unchecked_mut((i, j));
                                *r = self.get_unchecked((i, j)).inlined_clone().$method(r.inlined_clone())
                            }
                        }
                    }
                }
            }


            /*
             *
             * Methods without dimension checking at compile-time.
             * This is useful for code reuse because the sum representative system does not plays
             * easily with static checks.
             *
             */
            /// Equivalent to `self + rhs` but stores the result into `out` to avoid allocations.
            #[inline]
            pub fn $method_to<R2: Dim, C2: Dim, SB,
                              R3: Dim, C3: Dim, SC>(&self,
                                                    rhs: &Matrix<N, R2, C2, SB>,
                                                    out: &mut Matrix<N, R3, C3, SC>)
                where SB: Storage<N, R2, C2>,
                      SC: StorageMut<N, R3, C3>,
                      ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2> +
                                       SameNumberOfRows<R1, R3> + SameNumberOfColumns<C1, C3> {
                self.$method_to_statically_unchecked(rhs, out)
            }
        }

        impl<'b, N, R1, C1, R2, C2, SA, SB> $Trait<&'b Matrix<N, R2, C2, SB>> for Matrix<N, R1, C1, SA>
            where R1: Dim, C1: Dim, R2: Dim, C2: Dim,
                  N: Scalar + $bound,
                  SA: Storage<N, R1, C1>,
                  SB: Storage<N, R2, C2>,
                  DefaultAllocator: SameShapeAllocator<N, R1, C1, R2, C2>,
                  ShapeConstraint:  SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2> {
            type Output = MatrixSum<N, R1, C1, R2, C2>;

            #[inline]
            fn $method(self, rhs: &'b Matrix<N, R2, C2, SB>) -> Self::Output {
                assert!(self.shape() == rhs.shape(), "Matrix addition/subtraction dimensions mismatch.");
                let mut res = self.into_owned_sum::<R2, C2>();
                res.$method_assign_statically_unchecked(rhs);
                res
            }
        }

        impl<'a, N, R1, C1, R2, C2, SA, SB> $Trait<Matrix<N, R2, C2, SB>> for &'a Matrix<N, R1, C1, SA>
            where R1: Dim, C1: Dim, R2: Dim, C2: Dim,
                  N: Scalar + $bound,
                  SA: Storage<N, R1, C1>,
                  SB: Storage<N, R2, C2>,
                  DefaultAllocator: SameShapeAllocator<N, R2, C2, R1, C1>,
                  ShapeConstraint:  SameNumberOfRows<R2, R1> + SameNumberOfColumns<C2, C1> {
            type Output = MatrixSum<N, R2, C2, R1, C1>;

            #[inline]
            fn $method(self, rhs: Matrix<N, R2, C2, SB>) -> Self::Output {
                let mut rhs = rhs.into_owned_sum::<R1, C1>();
                assert!(self.shape() == rhs.shape(), "Matrix addition/subtraction dimensions mismatch.");
                self.$method_assign_statically_unchecked_rhs(&mut rhs);
                rhs
            }
        }

        impl<N, R1, C1, R2, C2, SA, SB> $Trait<Matrix<N, R2, C2, SB>> for Matrix<N, R1, C1, SA>
            where R1: Dim, C1: Dim, R2: Dim, C2: Dim,
                  N: Scalar + $bound,
                  SA: Storage<N, R1, C1>,
                  SB: Storage<N, R2, C2>,
                  DefaultAllocator: SameShapeAllocator<N, R1, C1, R2, C2>,
                  ShapeConstraint:  SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2> {
            type Output = MatrixSum<N, R1, C1, R2, C2>;

            #[inline]
            fn $method(self, rhs: Matrix<N, R2, C2, SB>) -> Self::Output {
                self.$method(&rhs)
            }
        }

        impl<'a, 'b, N, R1, C1, R2, C2, SA, SB> $Trait<&'b Matrix<N, R2, C2, SB>> for &'a Matrix<N, R1, C1, SA>
            where R1: Dim, C1: Dim, R2: Dim, C2: Dim,
                  N: Scalar + $bound,
                  SA: Storage<N, R1, C1>,
                  SB: Storage<N, R2, C2>,
                  DefaultAllocator: SameShapeAllocator<N, R1, C1, R2, C2>,
                  ShapeConstraint:  SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2> {
            type Output = MatrixSum<N, R1, C1, R2, C2>;

            #[inline]
            fn $method(self, rhs: &'b Matrix<N, R2, C2, SB>) -> Self::Output {
                let mut res = unsafe {
                    let (nrows, ncols) = self.shape();
                    let nrows: SameShapeR<R1, R2> = Dim::from_usize(nrows);
                    let ncols: SameShapeC<C1, C2> = Dim::from_usize(ncols);
                    Matrix::new_uninitialized_generic(nrows, ncols)
                };

                self.$method_to_statically_unchecked(rhs, &mut res);
                res
            }
        }

        impl<'b, N, R1, C1, R2, C2, SA, SB> $TraitAssign<&'b Matrix<N, R2, C2, SB>> for Matrix<N, R1, C1, SA>
            where R1: Dim, C1: Dim, R2: Dim, C2: Dim,
                  N: Scalar + $bound,
                  SA: StorageMut<N, R1, C1>,
                  SB: Storage<N, R2, C2>,
                  ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2> {

            #[inline]
            fn $method_assign(&mut self, rhs: &'b Matrix<N, R2, C2, SB>) {
                self.$method_assign_statically_unchecked(rhs)
            }
        }

        impl<N, R1, C1, R2, C2, SA, SB> $TraitAssign<Matrix<N, R2, C2, SB>> for Matrix<N, R1, C1, SA>
            where R1: Dim, C1: Dim, R2: Dim, C2: Dim,
                  N: Scalar + $bound,
                  SA: StorageMut<N, R1, C1>,
                  SB: Storage<N, R2, C2>,
                  ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2> {

            #[inline]
            fn $method_assign(&mut self, rhs: Matrix<N, R2, C2, SB>) {
                self.$method_assign(&rhs)
            }
        }
    }
);

componentwise_binop_impl!(Add, add, ClosedAdd;
                          AddAssign, add_assign, add_assign_statically_unchecked, add_assign_statically_unchecked_mut;
                          add_to, add_to_statically_unchecked);
componentwise_binop_impl!(Sub, sub, ClosedSub;
                          SubAssign, sub_assign, sub_assign_statically_unchecked, sub_assign_statically_unchecked_mut;
                          sub_to, sub_to_statically_unchecked);

impl<N, R: DimName, C: DimName> iter::Sum for MatrixMN<N, R, C>
where
    N: Scalar + ClosedAdd + Zero,
    DefaultAllocator: Allocator<N, R, C>,
{
    fn sum<I: Iterator<Item = MatrixMN<N, R, C>>>(iter: I) -> MatrixMN<N, R, C> {
        iter.fold(Matrix::zero(), |acc, x| acc + x)
    }
}

impl<N, C: Dim> iter::Sum for MatrixMN<N, Dynamic, C>
where
    N: Scalar + ClosedAdd + Zero,
    DefaultAllocator: Allocator<N, Dynamic, C>,
{
    /// # Example
    /// ```
    /// # use nalgebra::DVector;
    /// assert_eq!(vec![DVector::repeat(3, 1.0f64),
    ///                 DVector::repeat(3, 1.0f64),
    ///                 DVector::repeat(3, 1.0f64)].into_iter().sum::<DVector<f64>>(),
    ///            DVector::repeat(3, 1.0f64) + DVector::repeat(3, 1.0f64) + DVector::repeat(3, 1.0f64));
    /// ```
    ///
    /// # Panics
    /// Panics if the iterator is empty:
    /// ```should_panic
    /// # use std::iter;
    /// # use nalgebra::DMatrix;
    /// iter::empty::<DMatrix<f64>>().sum::<DMatrix<f64>>(); // panics!
    /// ```
    fn sum<I: Iterator<Item = MatrixMN<N, Dynamic, C>>>(mut iter: I) -> MatrixMN<N, Dynamic, C> {
        if let Some(first) = iter.next() {
            iter.fold(first, |acc, x| acc + x)
        } else {
            panic!("Cannot compute `sum` of empty iterator.")
        }
    }
}

impl<'a, N, R: DimName, C: DimName> iter::Sum<&'a MatrixMN<N, R, C>> for MatrixMN<N, R, C>
where
    N: Scalar + ClosedAdd + Zero,
    DefaultAllocator: Allocator<N, R, C>,
{
    fn sum<I: Iterator<Item = &'a MatrixMN<N, R, C>>>(iter: I) -> MatrixMN<N, R, C> {
        iter.fold(Matrix::zero(), |acc, x| acc + x)
    }
}

impl<'a, N, C: Dim> iter::Sum<&'a MatrixMN<N, Dynamic, C>> for MatrixMN<N, Dynamic, C>
where
    N: Scalar + ClosedAdd + Zero,
    DefaultAllocator: Allocator<N, Dynamic, C>,
{
    /// # Example
    /// ```
    /// # use nalgebra::DVector;
    /// let v = &DVector::repeat(3, 1.0f64);
    ///
    /// assert_eq!(vec![v, v, v].into_iter().sum::<DVector<f64>>(),
    ///            v + v + v);
    /// ```
    ///
    /// # Panics
    /// Panics if the iterator is empty:
    /// ```should_panic
    /// # use std::iter;
    /// # use nalgebra::DMatrix;
    /// iter::empty::<&DMatrix<f64>>().sum::<DMatrix<f64>>(); // panics!
    /// ```
    fn sum<I: Iterator<Item = &'a MatrixMN<N, Dynamic, C>>>(mut iter: I) -> MatrixMN<N, Dynamic, C> {
        if let Some(first) = iter.next() {
            iter.fold(first.clone(), |acc, x| acc + x)
        } else {
            panic!("Cannot compute `sum` of empty iterator.")
        }
    }
}

/*
 *
 * Multiplication
 *
 */

// Matrix × Scalar
// Matrix / Scalar
macro_rules! componentwise_scalarop_impl(
    ($Trait: ident, $method: ident, $bound: ident;
     $TraitAssign: ident, $method_assign: ident) => {
        impl<N, R: Dim, C: Dim, S> $Trait<N> for Matrix<N, R, C, S>
            where N: Scalar + $bound,
                  S: Storage<N, R, C>,
                  DefaultAllocator: Allocator<N, R, C> {
            type Output = MatrixMN<N, R, C>;

            #[inline]
            fn $method(self, rhs: N) -> Self::Output {
                let mut res = self.into_owned();

                // XXX: optimize our iterator!
                //
                // Using our own iterator prevents loop unrolling, which breaks some optimization
                // (like SIMD). On the other hand, using the slice iterator is 4x faster.

                // for left in res.iter_mut() {
                for left in res.as_mut_slice().iter_mut() {
                    *left = left.inlined_clone().$method(rhs.inlined_clone())
                }

                res
            }
        }

        impl<'a, N, R: Dim, C: Dim, S> $Trait<N> for &'a Matrix<N, R, C, S>
            where N: Scalar + $bound,
                  S: Storage<N, R, C>,
                  DefaultAllocator: Allocator<N, R, C> {
            type Output = MatrixMN<N, R, C>;

            #[inline]
            fn $method(self, rhs: N) -> Self::Output {
                self.clone_owned().$method(rhs)
            }
        }

        impl<N, R: Dim, C: Dim, S> $TraitAssign<N> for Matrix<N, R, C, S>
            where N: Scalar + $bound,
                  S: StorageMut<N, R, C> {
            #[inline]
            fn $method_assign(&mut self, rhs: N) {
                for j in 0 .. self.ncols() {
                    for i in 0 .. self.nrows() {
                        unsafe { self.get_unchecked_mut((i, j)).$method_assign(rhs.inlined_clone()) };
                    }
                }
            }
        }
    }
);

componentwise_scalarop_impl!(Mul, mul, ClosedMul; MulAssign, mul_assign);
componentwise_scalarop_impl!(Div, div, ClosedDiv; DivAssign, div_assign);

macro_rules! left_scalar_mul_impl(
    ($($T: ty),* $(,)*) => {$(
        impl<R: Dim, C: Dim, S: Storage<$T, R, C>> Mul<Matrix<$T, R, C, S>> for $T
            where DefaultAllocator: Allocator<$T, R, C> {
            type Output = MatrixMN<$T, R, C>;

            #[inline]
            fn mul(self, rhs: Matrix<$T, R, C, S>) -> Self::Output {
                let mut res = rhs.into_owned();

                // XXX: optimize our iterator!
                //
                // Using our own iterator prevents loop unrolling, which breaks some optimization
                // (like SIMD). On the other hand, using the slice iterator is 4x faster.

                // for rhs in res.iter_mut() {
                for rhs in res.as_mut_slice().iter_mut() {
                    *rhs = self * *rhs
                }

                res
            }
        }

        impl<'b, R: Dim, C: Dim, S: Storage<$T, R, C>> Mul<&'b Matrix<$T, R, C, S>> for $T
            where DefaultAllocator: Allocator<$T, R, C> {
            type Output = MatrixMN<$T, R, C>;

            #[inline]
            fn mul(self, rhs: &'b Matrix<$T, R, C, S>) -> Self::Output {
                self * rhs.clone_owned()
            }
        }
    )*}
);

left_scalar_mul_impl!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, f32, f64);

// Matrix × Matrix
impl<'a, 'b, N, R1: Dim, C1: Dim, R2: Dim, C2: Dim, SA, SB> Mul<&'b Matrix<N, R2, C2, SB>>
    for &'a Matrix<N, R1, C1, SA>
where
    N: Scalar + Zero + One + ClosedAdd + ClosedMul,
    SA: Storage<N, R1, C1>,
    SB: Storage<N, R2, C2>,
    DefaultAllocator: Allocator<N, R1, C2>,
    ShapeConstraint: AreMultipliable<R1, C1, R2, C2>,
{
    type Output = MatrixMN<N, R1, C2>;

    #[inline]
    fn mul(self, rhs: &'b Matrix<N, R2, C2, SB>) -> Self::Output {
        let mut res =
            unsafe { Matrix::new_uninitialized_generic(self.data.shape().0, rhs.data.shape().1) };

        self.mul_to(rhs, &mut res);
        res
    }
}

impl<'a, N, R1: Dim, C1: Dim, R2: Dim, C2: Dim, SA, SB> Mul<Matrix<N, R2, C2, SB>>
    for &'a Matrix<N, R1, C1, SA>
where
    N: Scalar + Zero + One + ClosedAdd + ClosedMul,
    SB: Storage<N, R2, C2>,
    SA: Storage<N, R1, C1>,
    DefaultAllocator: Allocator<N, R1, C2>,
    ShapeConstraint: AreMultipliable<R1, C1, R2, C2>,
{
    type Output = MatrixMN<N, R1, C2>;

    #[inline]
    fn mul(self, rhs: Matrix<N, R2, C2, SB>) -> Self::Output {
        self * &rhs
    }
}

impl<'b, N, R1: Dim, C1: Dim, R2: Dim, C2: Dim, SA, SB> Mul<&'b Matrix<N, R2, C2, SB>>
    for Matrix<N, R1, C1, SA>
where
    N: Scalar + Zero + One + ClosedAdd + ClosedMul,
    SB: Storage<N, R2, C2>,
    SA: Storage<N, R1, C1>,
    DefaultAllocator: Allocator<N, R1, C2>,
    ShapeConstraint: AreMultipliable<R1, C1, R2, C2>,
{
    type Output = MatrixMN<N, R1, C2>;

    #[inline]
    fn mul(self, rhs: &'b Matrix<N, R2, C2, SB>) -> Self::Output {
        &self * rhs
    }
}

impl<N, R1: Dim, C1: Dim, R2: Dim, C2: Dim, SA, SB> Mul<Matrix<N, R2, C2, SB>>
    for Matrix<N, R1, C1, SA>
where
    N: Scalar + Zero + One + ClosedAdd + ClosedMul,
    SB: Storage<N, R2, C2>,
    SA: Storage<N, R1, C1>,
    DefaultAllocator: Allocator<N, R1, C2>,
    ShapeConstraint: AreMultipliable<R1, C1, R2, C2>,
{
    type Output = MatrixMN<N, R1, C2>;

    #[inline]
    fn mul(self, rhs: Matrix<N, R2, C2, SB>) -> Self::Output {
        &self * &rhs
    }
}

// FIXME: this is too restrictive:
//    − we can't use `a *= b` when `a` is a mutable slice.
//    − we can't use `a *= b` when C2 is not equal to C1.
impl<N, R1, C1, R2, SA, SB> MulAssign<Matrix<N, R2, C1, SB>> for Matrix<N, R1, C1, SA>
where
    R1: Dim,
    C1: Dim,
    R2: Dim,
    N: Scalar + Zero + One + ClosedAdd + ClosedMul,
    SB: Storage<N, R2, C1>,
    SA: ContiguousStorageMut<N, R1, C1> + Clone,
    ShapeConstraint: AreMultipliable<R1, C1, R2, C1>,
    DefaultAllocator: Allocator<N, R1, C1, Buffer = SA>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: Matrix<N, R2, C1, SB>) {
        *self = &*self * rhs
    }
}

impl<'b, N, R1, C1, R2, SA, SB> MulAssign<&'b Matrix<N, R2, C1, SB>> for Matrix<N, R1, C1, SA>
where
    R1: Dim,
    C1: Dim,
    R2: Dim,
    N: Scalar + Zero + One + ClosedAdd + ClosedMul,
    SB: Storage<N, R2, C1>,
    SA: ContiguousStorageMut<N, R1, C1> + Clone,
    ShapeConstraint: AreMultipliable<R1, C1, R2, C1>,
    // FIXME: this is too restrictive. See comments for the non-ref version.
    DefaultAllocator: Allocator<N, R1, C1, Buffer = SA>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: &'b Matrix<N, R2, C1, SB>) {
        *self = &*self * rhs
    }
}

// Transpose-multiplication.
impl<N, R1: Dim, C1: Dim, SA> Matrix<N, R1, C1, SA>
where
    N: Scalar + Zero + One + ClosedAdd + ClosedMul,
    SA: Storage<N, R1, C1>,
{
    /// Equivalent to `self.transpose() * rhs`.
    #[inline]
    pub fn tr_mul<R2: Dim, C2: Dim, SB>(&self, rhs: &Matrix<N, R2, C2, SB>) -> MatrixMN<N, C1, C2>
    where
        SB: Storage<N, R2, C2>,
        DefaultAllocator: Allocator<N, C1, C2>,
        ShapeConstraint: SameNumberOfRows<R1, R2>,
    {
        let mut res =
            unsafe { Matrix::new_uninitialized_generic(self.data.shape().1, rhs.data.shape().1) };

        self.tr_mul_to(rhs, &mut res);
        res
    }

    /// Equivalent to `self.adjoint() * rhs`.
    #[inline]
    pub fn ad_mul<R2: Dim, C2: Dim, SB>(&self, rhs: &Matrix<N, R2, C2, SB>) -> MatrixMN<N, C1, C2>
        where
            N: ComplexField,
            SB: Storage<N, R2, C2>,
            DefaultAllocator: Allocator<N, C1, C2>,
            ShapeConstraint: SameNumberOfRows<R1, R2>,
    {
        let mut res =
            unsafe { Matrix::new_uninitialized_generic(self.data.shape().1, rhs.data.shape().1) };

        self.ad_mul_to(rhs, &mut res);
        res
    }

    #[inline(always)]
    fn xx_mul_to<R2: Dim, C2: Dim, SB, R3: Dim, C3: Dim, SC>(
        &self,
        rhs: &Matrix<N, R2, C2, SB>,
        out: &mut Matrix<N, R3, C3, SC>,
        dot: impl Fn(&VectorSliceN<N, R1, SA::RStride, SA::CStride>, &VectorSliceN<N, R2, SB::RStride, SB::CStride>) -> N,
    ) where
        SB: Storage<N, R2, C2>,
        SC: StorageMut<N, R3, C3>,
        ShapeConstraint: SameNumberOfRows<R1, R2> + DimEq<C1, R3> + DimEq<C2, C3>,
    {
        let (nrows1, ncols1) = self.shape();
        let (nrows2, ncols2) = rhs.shape();
        let (nrows3, ncols3) = out.shape();

        assert!(
            nrows1 == nrows2,
            "Matrix multiplication dimensions mismatch."
        );
        assert!(
            nrows3 == ncols1 && ncols3 == ncols2,
            "Matrix multiplication output dimensions mismatch."
        );

        for i in 0..ncols1 {
            for j in 0..ncols2 {
                let dot = dot(&self.column(i), &rhs.column(j));
                unsafe { *out.get_unchecked_mut((i, j)) = dot };
            }
        }
    }

    /// Equivalent to `self.transpose() * rhs` but stores the result into `out` to avoid
    /// allocations.
    #[inline]
    pub fn tr_mul_to<R2: Dim, C2: Dim, SB, R3: Dim, C3: Dim, SC>(
        &self,
        rhs: &Matrix<N, R2, C2, SB>,
        out: &mut Matrix<N, R3, C3, SC>,
    ) where
        SB: Storage<N, R2, C2>,
        SC: StorageMut<N, R3, C3>,
        ShapeConstraint: SameNumberOfRows<R1, R2> + DimEq<C1, R3> + DimEq<C2, C3>,
    {
        self.xx_mul_to(rhs, out, |a, b| a.dot(b))
    }

    /// Equivalent to `self.adjoint() * rhs` but stores the result into `out` to avoid
    /// allocations.
    #[inline]
    pub fn ad_mul_to<R2: Dim, C2: Dim, SB, R3: Dim, C3: Dim, SC>(
        &self,
        rhs: &Matrix<N, R2, C2, SB>,
        out: &mut Matrix<N, R3, C3, SC>,
    ) where
        N: ComplexField,
        SB: Storage<N, R2, C2>,
        SC: StorageMut<N, R3, C3>,
        ShapeConstraint: SameNumberOfRows<R1, R2> + DimEq<C1, R3> + DimEq<C2, C3>,
    {
        self.xx_mul_to(rhs, out, |a, b| a.dotc(b))
    }

    /// Equivalent to `self * rhs` but stores the result into `out` to avoid allocations.
    #[inline]
    pub fn mul_to<R2: Dim, C2: Dim, SB, R3: Dim, C3: Dim, SC>(
        &self,
        rhs: &Matrix<N, R2, C2, SB>,
        out: &mut Matrix<N, R3, C3, SC>,
    ) where
        SB: Storage<N, R2, C2>,
        SC: StorageMut<N, R3, C3>,
        ShapeConstraint: SameNumberOfRows<R3, R1>
            + SameNumberOfColumns<C3, C2>
            + AreMultipliable<R1, C1, R2, C2>,
    {
        out.gemm(N::one(), self, rhs, N::zero());
    }

    /// The kronecker product of two matrices (aka. tensor product of the corresponding linear
    /// maps).
    pub fn kronecker<R2: Dim, C2: Dim, SB>(
        &self,
        rhs: &Matrix<N, R2, C2, SB>,
    ) -> MatrixMN<N, DimProd<R1, R2>, DimProd<C1, C2>>
    where
        N: ClosedMul,
        R1: DimMul<R2>,
        C1: DimMul<C2>,
        SB: Storage<N, R2, C2>,
        DefaultAllocator: Allocator<N, DimProd<R1, R2>, DimProd<C1, C2>>,
    {
        let (nrows1, ncols1) = self.data.shape();
        let (nrows2, ncols2) = rhs.data.shape();

        let mut res =
            unsafe { Matrix::new_uninitialized_generic(nrows1.mul(nrows2), ncols1.mul(ncols2)) };

        {
            let mut data_res = res.data.ptr_mut();

            for j1 in 0..ncols1.value() {
                for j2 in 0..ncols2.value() {
                    for i1 in 0..nrows1.value() {
                        unsafe {
                            let coeff = self.get_unchecked((i1, j1)).inlined_clone();

                            for i2 in 0..nrows2.value() {
                                *data_res = coeff.inlined_clone() * rhs.get_unchecked((i2, j2)).inlined_clone();
                                data_res = data_res.offset(1);
                            }
                        }
                    }
                }
            }
        }

        res
    }
}

impl<N: Scalar + ClosedAdd, R: Dim, C: Dim, S: Storage<N, R, C>> Matrix<N, R, C, S> {
    /// Adds a scalar to `self`.
    #[inline]
    #[must_use = "Did you mean to use add_scalar_mut()?"]
    pub fn add_scalar(&self, rhs: N) -> MatrixMN<N, R, C>
    where DefaultAllocator: Allocator<N, R, C> {
        let mut res = self.clone_owned();
        res.add_scalar_mut(rhs);
        res
    }

    /// Adds a scalar to `self` in-place.
    #[inline]
    pub fn add_scalar_mut(&mut self, rhs: N)
    where S: StorageMut<N, R, C> {
        for e in self.iter_mut() {
            *e += rhs.inlined_clone()
        }
    }
}

impl<N, D: DimName> iter::Product for MatrixN<N, D>
where
    N: Scalar + Zero + One + ClosedMul + ClosedAdd,
    DefaultAllocator: Allocator<N, D, D>,
{
    fn product<I: Iterator<Item = MatrixN<N, D>>>(iter: I) -> MatrixN<N, D> {
        iter.fold(Matrix::one(), |acc, x| acc * x)
    }
}

impl<'a, N, D: DimName> iter::Product<&'a MatrixN<N, D>> for MatrixN<N, D>
where
    N: Scalar + Zero + One + ClosedMul + ClosedAdd,
    DefaultAllocator: Allocator<N, D, D>,
{
    fn product<I: Iterator<Item = &'a MatrixN<N, D>>>(iter: I) -> MatrixN<N, D> {
        iter.fold(Matrix::one(), |acc, x| acc * x)
    }
}

impl<N: Scalar, R: Dim, C: Dim, S: Storage<N, R, C>> Matrix<N, R, C, S> {
    #[inline(always)]
    fn xcmp<N2>(&self, abs: impl Fn(N) -> N2, ordering: Ordering) -> N2
        where N2: Scalar + PartialOrd + Zero {
        let mut iter = self.iter();
        let mut max = iter.next().cloned().map_or(N2::zero(), &abs);

        for e in iter {
            let ae = abs(e.inlined_clone());

            if ae.partial_cmp(&max) == Some(ordering) {
                    max = ae;
            }
        }

        max
    }

    /// Returns the absolute value of the component with the largest absolute value.
    /// # Example
    /// ```
    /// # use nalgebra::Vector3;
    /// assert_eq!(Vector3::new(-1.0, 2.0, 3.0).amax(), 3.0);
    /// assert_eq!(Vector3::new(-1.0, -2.0, -3.0).amax(), 3.0);
    /// ```
    #[inline]
    pub fn amax(&self) -> N
        where N: PartialOrd + Signed {
        self.xcmp(|e| e.abs(), Ordering::Greater)
    }

    /// Returns the the 1-norm of the complex component with the largest 1-norm.
    /// # Example
    /// ```
    /// # use nalgebra::{Vector3, Complex};
    /// assert_eq!(Vector3::new(
    ///     Complex::new(-3.0, -2.0),
    ///     Complex::new(1.0, 2.0),
    ///     Complex::new(1.0, 3.0)).camax(), 5.0);
    /// ```
    #[inline]
    pub fn camax(&self) -> N::RealField
        where N: ComplexField {
        self.xcmp(|e| e.norm1(), Ordering::Greater)
    }

    /// Returns the component with the largest value.
    /// # Example
    /// ```
    /// # use nalgebra::Vector3;
    /// assert_eq!(Vector3::new(-1.0, 2.0, 3.0).max(), 3.0);
    /// assert_eq!(Vector3::new(-1.0, -2.0, -3.0).max(), -1.0);
    /// assert_eq!(Vector3::new(5u32, 2, 3).max(), 5);
    /// ```
    #[inline]
    pub fn max(&self) -> N
        where N: PartialOrd + Zero {
        self.xcmp(|e| e, Ordering::Greater)
    }

    /// Returns the absolute value of the component with the smallest absolute value.
    /// # Example
    /// ```
    /// # use nalgebra::Vector3;
    /// assert_eq!(Vector3::new(-1.0, 2.0, -3.0).amin(), 1.0);
    /// assert_eq!(Vector3::new(10.0, 2.0, 30.0).amin(), 2.0);
    /// ```
    #[inline]
    pub fn amin(&self) -> N
        where N: PartialOrd + Signed {
        self.xcmp(|e| e.abs(), Ordering::Less)
    }

    /// Returns the the 1-norm of the complex component with the smallest 1-norm.
    /// # Example
    /// ```
    /// # use nalgebra::{Vector3, Complex};
    /// assert_eq!(Vector3::new(
    ///     Complex::new(-3.0, -2.0),
    ///     Complex::new(1.0, 2.0),
    ///     Complex::new(1.0, 3.0)).camin(), 3.0);
    /// ```
    #[inline]
    pub fn camin(&self) -> N::RealField
        where N: ComplexField {
        self.xcmp(|e| e.norm1(), Ordering::Less)
    }

    /// Returns the component with the smallest value.
    /// # Example
    /// ```
    /// # use nalgebra::Vector3;
    /// assert_eq!(Vector3::new(-1.0, 2.0, 3.0).min(), -1.0);
    /// assert_eq!(Vector3::new(1.0, 2.0, 3.0).min(), 1.0);
    /// assert_eq!(Vector3::new(5u32, 2, 3).min(), 2);
    /// ```
    #[inline]
    pub fn min(&self) -> N
        where N: PartialOrd + Zero {
        self.xcmp(|e| e, Ordering::Less)
    }
}
