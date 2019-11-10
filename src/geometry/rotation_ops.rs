/*
 *
 * This provides the following operator overladings:
 *
 * Index<(usize, usize)>
 *
 * Rotation × Rotation
 * Rotation ÷ Rotation
 * Rotation × Matrix
 * Matrix   × Rotation
 * Matrix   ÷ Rotation
 * Rotation × Point
 * Rotation × Unit<Vector>
 *
 *
 * Rotation ×= Rotation
 * Matrix   ×= Rotation
 */

use num::{One, Zero};
use std::ops::{Div, DivAssign, Index, Mul, MulAssign};

use alga::general::{ClosedAdd, ClosedMul};

use crate::base::allocator::Allocator;
use crate::base::constraint::{AreMultipliable, ShapeConstraint};
use crate::{Dim, DimName, U1};
use crate::base::storage::Storage;
use crate::base::{DefaultAllocator, Matrix, MatrixMN, Scalar, Unit, Vector, VectorN};

use crate::geometry::{Point, Rotation};

impl<N: Scalar, D: DimName> Index<(usize, usize)> for Rotation<N, D>
where DefaultAllocator: Allocator<N, D, D>
{
    type Output = N;

    #[inline]
    fn index(&self, row_col: (usize, usize)) -> &N {
        self.matrix().index(row_col)
    }
}

// Rotation × Rotation
md_impl_all!(
    Mul, mul;
    (D, D), (D, D) for D: DimName;
    self: Rotation<N, D>, right: Rotation<N, D>, Output = Rotation<N, D>;
    [val val] => Rotation::from_matrix_unchecked(self.into_inner() * right.into_inner());
    [ref val] => Rotation::from_matrix_unchecked(self.matrix() * right.into_inner());
    [val ref] => Rotation::from_matrix_unchecked(self.into_inner() * right.matrix());
    [ref ref] => Rotation::from_matrix_unchecked(self.matrix() * right.matrix());
);

// Rotation ÷ Rotation
// FIXME: instead of calling inverse explicitly, could we just add a `mul_tr` or `mul_inv` method?
md_impl_all!(
    Div, div;
    (D, D), (D, D) for D: DimName;
    self: Rotation<N, D>, right: Rotation<N, D>, Output = Rotation<N, D>;
    [val val] => self * right.inverse();
    [ref val] => self * right.inverse();
    [val ref] => self * right.inverse();
    [ref ref] => self * right.inverse();
);

// Rotation × Matrix
md_impl_all!(
    Mul, mul;
    (D1, D1), (R2, C2) for D1: DimName, R2: Dim, C2: Dim, SB: Storage<N, R2, C2>
    where DefaultAllocator: Allocator<N, D1, C2>
    where ShapeConstraint: AreMultipliable<D1, D1, R2, C2>;
    self: Rotation<N, D1>, right: Matrix<N, R2, C2, SB>, Output = MatrixMN<N, D1, C2>;
    [val val] => self.into_inner() * right;
    [ref val] => self.matrix() * right;
    [val ref] => self.into_inner() * right;
    [ref ref] => self.matrix() * right;
);

// Matrix × Rotation
md_impl_all!(
    Mul, mul;
    (R1, C1), (D2, D2) for R1: Dim, C1: Dim, D2: DimName, SA: Storage<N, R1, C1>
    where DefaultAllocator: Allocator<N, R1, D2>
    where ShapeConstraint:  AreMultipliable<R1, C1, D2, D2>;
    self: Matrix<N, R1, C1, SA>, right: Rotation<N, D2>, Output = MatrixMN<N, R1, D2>;
    [val val] => self * right.into_inner();
    [ref val] => self * right.into_inner();
    [val ref] => self * right.matrix();
    [ref ref] => self * right.matrix();
);

// Matrix ÷ Rotation
md_impl_all!(
    Div, div;
    (R1, C1), (D2, D2) for R1: Dim, C1: Dim, D2: DimName, SA: Storage<N, R1, C1>
    where DefaultAllocator: Allocator<N, R1, D2>
    where ShapeConstraint: AreMultipliable<R1, C1, D2, D2>;
    self: Matrix<N, R1, C1, SA>, right: Rotation<N, D2>, Output = MatrixMN<N, R1, D2>;
    [val val] => self * right.inverse();
    [ref val] => self * right.inverse();
    [val ref] => self * right.inverse();
    [ref ref] => self * right.inverse();
);

// Rotation × Point
// FIXME: we don't handle properly non-zero origins here. Do we want this to be the intended
// behavior?
md_impl_all!(
    Mul, mul;
    (D, D), (D, U1) for D: DimName
    where DefaultAllocator: Allocator<N, D>
    where ShapeConstraint:  AreMultipliable<D, D, D, U1>;
    self: Rotation<N, D>, right: Point<N, D>, Output = Point<N, D>;
    [val val] => self.into_inner() * right;
    [ref val] => self.matrix() * right;
    [val ref] => self.into_inner() * right;
    [ref ref] => self.matrix() * right;
);

// Rotation × Unit<Vector>
md_impl_all!(
    Mul, mul;
    (D, D), (D, U1) for D: DimName, S: Storage<N, D>
    where DefaultAllocator: Allocator<N, D>
    where ShapeConstraint:  AreMultipliable<D, D, D, U1>;
    self: Rotation<N, D>, right: Unit<Vector<N, D, S>>, Output = Unit<VectorN<N, D>>;
    [val val] => Unit::new_unchecked(self.into_inner() * right.into_inner());
    [ref val] => Unit::new_unchecked(self.matrix() * right.into_inner());
    [val ref] => Unit::new_unchecked(self.into_inner() * right.as_ref());
    [ref ref] => Unit::new_unchecked(self.matrix() * right.as_ref());
);

// Rotation ×= Rotation
// FIXME: try not to call `inverse()` explicitly.

md_assign_impl_all!(
    MulAssign, mul_assign;
    (D, D), (D, D) for D: DimName;
    self: Rotation<N, D>, right: Rotation<N, D>;
    [val] => self.matrix_mut_unchecked().mul_assign(right.into_inner());
    [ref] => self.matrix_mut_unchecked().mul_assign(right.matrix());
);

md_assign_impl_all!(
    DivAssign, div_assign;
    (D, D), (D, D) for D: DimName;
    self: Rotation<N, D>, right: Rotation<N, D>;
    [val] => self.matrix_mut_unchecked().mul_assign(right.inverse().into_inner());
    [ref] => self.matrix_mut_unchecked().mul_assign(right.inverse().matrix());
);

// Matrix *= Rotation
// FIXME: try not to call `inverse()` explicitly.
// FIXME: this shares the same limitations as for the current impl. of MulAssign for matrices.
// (In particular the number of matrix column must be equal to the number of rotation columns,
// i.e., equal to the rotation dimension.

md_assign_impl_all!(
    MulAssign, mul_assign;
    (R1, C1), (C1, C1) for R1: DimName, C1: DimName;
    self: MatrixMN<N, R1, C1>, right: Rotation<N, C1>;
    [val] => self.mul_assign(right.into_inner());
    [ref] => self.mul_assign(right.matrix());
);

md_assign_impl_all!(
    DivAssign, div_assign;
    (R1, C1), (C1, C1) for R1: DimName, C1: DimName;
    self: MatrixMN<N, R1, C1>, right: Rotation<N, C1>;
    [val] => self.mul_assign(right.inverse().into_inner());
    [ref] => self.mul_assign(right.inverse().matrix());
);
