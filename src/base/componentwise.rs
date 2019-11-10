// Non-conventional component-wise operators.

use num::{Signed, Zero};
use std::ops::{Add, Mul};

use alga::general::{ClosedDiv, ClosedMul};

use crate::base::allocator::{Allocator, SameShapeAllocator};
use crate::base::constraint::{SameNumberOfColumns, SameNumberOfRows, ShapeConstraint};
use crate::Dim;
use crate::base::storage::{Storage, StorageMut};
use crate::base::{DefaultAllocator, Matrix, MatrixMN, MatrixSum, Scalar};

/// The type of the result of a matrix component-wise operation.
pub type MatrixComponentOp<N, R1, C1, R2, C2> = MatrixSum<N, R1, C1, R2, C2>;

impl<N: Scalar, R: Dim, C: Dim, S: Storage<N, R, C>> Matrix<N, R, C, S> {
    /// Computes the component-wise absolute value.
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::Matrix2;
    /// let a = Matrix2::new(0.0, 1.0,
    ///                      -2.0, -3.0);
    /// assert_eq!(a.abs(), Matrix2::new(0.0, 1.0, 2.0, 3.0))
    /// ```
    #[inline]
    pub fn abs(&self) -> MatrixMN<N, R, C>
    where
        N: Signed,
        DefaultAllocator: Allocator<N, R, C>,
    {
        let mut res = self.clone_owned();

        for e in res.iter_mut() {
            *e = e.abs();
        }

        res
    }

    // FIXME: add other operators like component_ln, component_pow, etc. ?
}

macro_rules! component_binop_impl(
    ($($binop: ident, $binop_mut: ident, $binop_assign: ident, $cmpy: ident, $Trait: ident . $op: ident . $op_assign: ident, $desc:expr, $desc_cmpy:expr, $desc_mut:expr);* $(;)*) => {$(
        impl<N: Scalar, R1: Dim, C1: Dim, SA: Storage<N, R1, C1>> Matrix<N, R1, C1, SA> {
            #[doc = $desc]
            #[inline]
            pub fn $binop<R2, C2, SB>(&self, rhs: &Matrix<N, R2, C2, SB>) -> MatrixComponentOp<N, R1, C1, R2, C2>
                where N: $Trait,
                      R2: Dim, C2: Dim,
                      SB: Storage<N, R2, C2>,
                      DefaultAllocator: SameShapeAllocator<N, R1, C1, R2, C2>,
                      ShapeConstraint:  SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2> {

                assert_eq!(self.shape(), rhs.shape(), "Componentwise mul/div: mismatched matrix dimensions.");
                let mut res = self.clone_owned_sum();

                for j in 0 .. res.ncols() {
                    for i in 0 .. res.nrows() {
                        unsafe {
                            res.get_unchecked_mut((i, j)).$op_assign(*rhs.get_unchecked((i, j)));
                        }
                    }
                }

                res
            }
        }

        impl<N: Scalar, R1: Dim, C1: Dim, SA: StorageMut<N, R1, C1>> Matrix<N, R1, C1, SA> {
            // componentwise binop plus Y.
            #[doc = $desc_cmpy]
            #[inline]
            pub fn $cmpy<R2, C2, SB, R3, C3, SC>(&mut self, alpha: N, a: &Matrix<N, R2, C2, SB>, b: &Matrix<N, R3, C3, SC>, beta: N)
                where N: $Trait + Zero + Mul<N, Output = N> + Add<N, Output = N>,
                      R2: Dim, C2: Dim,
                      R3: Dim, C3: Dim,
                      SB: Storage<N, R2, C2>,
                      SC: Storage<N, R3, C3>,
                      ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2> +
                                       SameNumberOfRows<R1, R3> + SameNumberOfColumns<C1, C3> {
                assert_eq!(self.shape(), a.shape(), "Componentwise mul/div: mismatched matrix dimensions.");
                assert_eq!(self.shape(), b.shape(), "Componentwise mul/div: mismatched matrix dimensions.");

                if beta.is_zero() {
                    for j in 0 .. self.ncols() {
                        for i in 0 .. self.nrows() {
                            unsafe {
                                let res = alpha * a.get_unchecked((i, j)).$op(*b.get_unchecked((i, j)));
                                *self.get_unchecked_mut((i, j)) = res;
                            }
                        }
                    }
                }
                else {
                    for j in 0 .. self.ncols() {
                        for i in 0 .. self.nrows() {
                            unsafe {
                                let res = alpha * a.get_unchecked((i, j)).$op(*b.get_unchecked((i, j)));
                                *self.get_unchecked_mut((i, j)) = beta * *self.get_unchecked((i, j)) + res;
                            }
                        }
                    }
                }
            }

            #[doc = $desc_mut]
            #[inline]
            pub fn $binop_assign<R2, C2, SB>(&mut self, rhs: &Matrix<N, R2, C2, SB>)
                where N: $Trait,
                      R2: Dim,
                      C2: Dim,
                      SB: Storage<N, R2, C2>,
                      ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2> {

                assert_eq!(self.shape(), rhs.shape(), "Componentwise mul/div: mismatched matrix dimensions.");

                for j in 0 .. self.ncols() {
                    for i in 0 .. self.nrows() {
                        unsafe {
                            self.get_unchecked_mut((i, j)).$op_assign(*rhs.get_unchecked((i, j)));
                        }
                    }
                }
            }

            #[doc = $desc_mut]
            #[inline]
            #[deprecated(note = "This is renamed using the `_assign` suffix instead of the `_mut` suffix.")]
            pub fn $binop_mut<R2, C2, SB>(&mut self, rhs: &Matrix<N, R2, C2, SB>)
                where N: $Trait,
                      R2: Dim,
                      C2: Dim,
                      SB: Storage<N, R2, C2>,
                      ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2> {
                self.$binop_assign(rhs)
            }
        }
    )*}
);

component_binop_impl!(
    component_mul, component_mul_mut, component_mul_assign, cmpy, ClosedMul.mul.mul_assign,
    r"
    Componentwise matrix or vector multiplication.

    # Example

    ```
    # use nalgebra::Matrix2;
    let a = Matrix2::new(0.0, 1.0, 2.0, 3.0);
    let b = Matrix2::new(4.0, 5.0, 6.0, 7.0);
    let expected = Matrix2::new(0.0, 5.0, 12.0, 21.0);

    assert_eq!(a.component_mul(&b), expected);
    ```
    ",
    r"
    Computes componentwise `self[i] = alpha * a[i] * b[i] + beta * self[i]`.

    # Example
    ```
    # use nalgebra::Matrix2;
    let mut m = Matrix2::new(0.0, 1.0, 2.0, 3.0);
    let a = Matrix2::new(0.0, 1.0, 2.0, 3.0);
    let b = Matrix2::new(4.0, 5.0, 6.0, 7.0);
    let expected = (a.component_mul(&b) * 5.0) + m * 10.0;

    m.cmpy(5.0, &a, &b, 10.0);
    assert_eq!(m, expected);
    ```
    ",
    r"
    Inplace componentwise matrix or vector multiplication.

    # Example
    ```
    # use nalgebra::Matrix2;
    let mut a = Matrix2::new(0.0, 1.0, 2.0, 3.0);
    let b = Matrix2::new(4.0, 5.0, 6.0, 7.0);
    let expected = Matrix2::new(0.0, 5.0, 12.0, 21.0);

    a.component_mul_assign(&b);

    assert_eq!(a, expected);
    ```
    ";
    component_div, component_div_mut, component_div_assign, cdpy, ClosedDiv.div.div_assign,
    r"
    Componentwise matrix or vector division.

    # Example

    ```
    # use nalgebra::Matrix2;
    let a = Matrix2::new(0.0, 1.0, 2.0, 3.0);
    let b = Matrix2::new(4.0, 5.0, 6.0, 7.0);
    let expected = Matrix2::new(0.0, 1.0 / 5.0, 2.0 / 6.0, 3.0 / 7.0);

    assert_eq!(a.component_div(&b), expected);
    ```
    ",
    r"
    Computes componentwise `self[i] = alpha * a[i] / b[i] + beta * self[i]`.

    # Example
    ```
    # use nalgebra::Matrix2;
    let mut m = Matrix2::new(0.0, 1.0, 2.0, 3.0);
    let a = Matrix2::new(4.0, 5.0, 6.0, 7.0);
    let b = Matrix2::new(4.0, 5.0, 6.0, 7.0);
    let expected = (a.component_div(&b) * 5.0) + m * 10.0;

    m.cdpy(5.0, &a, &b, 10.0);
    assert_eq!(m, expected);
    ```
    ",
    r"
    Inplace componentwise matrix or vector division.

    # Example
    ```
    # use nalgebra::Matrix2;
    let mut a = Matrix2::new(0.0, 1.0, 2.0, 3.0);
    let b = Matrix2::new(4.0, 5.0, 6.0, 7.0);
    let expected = Matrix2::new(0.0, 1.0 / 5.0, 2.0 / 6.0, 3.0 / 7.0);

    a.component_div_assign(&b);

    assert_eq!(a, expected);
    ```
    ";
    // FIXME: add other operators like bitshift, etc. ?
);
