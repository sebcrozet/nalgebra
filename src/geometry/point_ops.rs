use num::{One, Zero};
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use alga::general::{ClosedAdd, ClosedDiv, ClosedMul, ClosedNeg, ClosedSub};

use crate::base::allocator::{Allocator, SameShapeAllocator};
use crate::base::constraint::{AreMultipliable, SameNumberOfColumns, SameNumberOfRows, ShapeConstraint};
use crate::{Dim, DimName, U1};
use crate::base::storage::Storage;
use crate::base::{DefaultAllocator, Matrix, Scalar, Vector, VectorSum};

use crate::geometry::Point;

/*
 *
 * Indexing.
 *
 */
impl<N: Scalar, D: DimName> Index<usize> for Point<N, D>
where DefaultAllocator: Allocator<N, D>
{
    type Output = N;

    #[inline]
    fn index(&self, i: usize) -> &Self::Output {
        &self.coords[i]
    }
}

impl<N: Scalar, D: DimName> IndexMut<usize> for Point<N, D>
where DefaultAllocator: Allocator<N, D>
{
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        &mut self.coords[i]
    }
}

/*
 *
 * Neg.
 *
 */
impl<N: Scalar + ClosedNeg, D: DimName> Neg for Point<N, D>
where DefaultAllocator: Allocator<N, D>
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::Output::from(-self.coords)
    }
}

impl<'a, N: Scalar + ClosedNeg, D: DimName> Neg for &'a Point<N, D>
where DefaultAllocator: Allocator<N, D>
{
    type Output = Point<N, D>;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::Output::from(-&self.coords)
    }
}

/*
 *
 * Subtraction & Addition.
 *
 */

// Point - Point
add_sub_impl!(Sub, sub, ClosedSub;
    (D, U1), (D, U1) for D: DimName;
    self: &'a Point<N, D>, right: &'b Point<N, D>, Output = VectorSum<N, D, D>;
    &self.coords - &right.coords; 'a, 'b);

add_sub_impl!(Sub, sub, ClosedSub;
    (D, U1), (D, U1) for D: DimName;
    self: &'a Point<N, D>, right: Point<N, D>, Output = VectorSum<N, D, D>;
    &self.coords - right.coords; 'a);

add_sub_impl!(Sub, sub, ClosedSub;
    (D, U1), (D, U1) for D: DimName;
    self: Point<N, D>, right: &'b Point<N, D>, Output = VectorSum<N, D, D>;
    self.coords - &right.coords; 'b);

add_sub_impl!(Sub, sub, ClosedSub;
    (D, U1), (D, U1) for D: DimName;
    self: Point<N, D>, right: Point<N, D>, Output = VectorSum<N, D, D>;
    self.coords - right.coords; );

// Point - Vector
add_sub_impl!(Sub, sub, ClosedSub;
    (D1, U1), (D2, U1) -> (D1) for D1: DimName, D2: Dim, SB: Storage<N, D2>;
    self: &'a Point<N, D1>, right: &'b Vector<N, D2, SB>, Output = Point<N, D1>;
    Self::Output::from(&self.coords - right); 'a, 'b);

add_sub_impl!(Sub, sub, ClosedSub;
    (D1, U1), (D2, U1) -> (D1) for D1: DimName, D2: Dim, SB: Storage<N, D2>;
    self: &'a Point<N, D1>, right: Vector<N, D2, SB>, Output = Point<N, D1>;
    Self::Output::from(&self.coords - &right); 'a); // FIXME: should not be a ref to `right`.

add_sub_impl!(Sub, sub, ClosedSub;
    (D1, U1), (D2, U1) -> (D1) for D1: DimName, D2: Dim, SB: Storage<N, D2>;
    self: Point<N, D1>, right: &'b Vector<N, D2, SB>, Output = Point<N, D1>;
    Self::Output::from(self.coords - right); 'b);

add_sub_impl!(Sub, sub, ClosedSub;
    (D1, U1), (D2, U1) -> (D1) for D1: DimName, D2: Dim, SB: Storage<N, D2>;
    self: Point<N, D1>, right: Vector<N, D2, SB>, Output = Point<N, D1>;
    Self::Output::from(self.coords - right); );

// Point + Vector
add_sub_impl!(Add, add, ClosedAdd;
    (D1, U1), (D2, U1) -> (D1) for D1: DimName, D2: Dim, SB: Storage<N, D2>;
    self: &'a Point<N, D1>, right: &'b Vector<N, D2, SB>, Output = Point<N, D1>;
    Self::Output::from(&self.coords + right); 'a, 'b);

add_sub_impl!(Add, add, ClosedAdd;
    (D1, U1), (D2, U1) -> (D1) for D1: DimName, D2: Dim, SB: Storage<N, D2>;
    self: &'a Point<N, D1>, right: Vector<N, D2, SB>, Output = Point<N, D1>;
    Self::Output::from(&self.coords + &right); 'a); // FIXME: should not be a ref to `right`.

add_sub_impl!(Add, add, ClosedAdd;
    (D1, U1), (D2, U1) -> (D1) for D1: DimName, D2: Dim, SB: Storage<N, D2>;
    self: Point<N, D1>, right: &'b Vector<N, D2, SB>, Output = Point<N, D1>;
    Self::Output::from(self.coords + right); 'b);

add_sub_impl!(Add, add, ClosedAdd;
    (D1, U1), (D2, U1) -> (D1) for D1: DimName, D2: Dim, SB: Storage<N, D2>;
    self: Point<N, D1>, right: Vector<N, D2, SB>, Output = Point<N, D1>;
    Self::Output::from(self.coords + right); );

// XXX: replace by the shared macro: add_sub_assign_impl
macro_rules! op_assign_impl(
    ($($TraitAssign: ident, $method_assign: ident, $bound: ident);* $(;)*) => {$(
        impl<'b, N, D1: DimName, D2: Dim, SB> $TraitAssign<&'b Vector<N, D2, SB>> for Point<N, D1>
            where N:  Scalar + $bound,
                  SB: Storage<N, D2>,
                  DefaultAllocator: Allocator<N, D1>,
                  ShapeConstraint: SameNumberOfRows<D1, D2> {

            #[inline]
            fn $method_assign(&mut self, right: &'b Vector<N, D2, SB>) {
                self.coords.$method_assign(right)
            }
        }

        impl<N, D1: DimName, D2: Dim, SB> $TraitAssign<Vector<N, D2, SB>> for Point<N, D1>
            where N:  Scalar + $bound,
                  SB: Storage<N, D2>,
                  DefaultAllocator: Allocator<N, D1>,
                  ShapeConstraint: SameNumberOfRows<D1, D2> {

            #[inline]
            fn $method_assign(&mut self, right: Vector<N, D2, SB>) {
                self.coords.$method_assign(right)
            }
        }
    )*}
);

op_assign_impl!(
    AddAssign, add_assign, ClosedAdd;
    SubAssign, sub_assign, ClosedSub;
);

/*
 *
 * Matrix × Point
 *
 */
md_impl_all!(
    Mul, mul;
    (R1, C1), (D2, U1) for R1: DimName, C1: Dim, D2: DimName, SA: Storage<N, R1, C1>
    where ShapeConstraint: AreMultipliable<R1, C1, D2, U1>;
    self: Matrix<N, R1, C1, SA>, right: Point<N, D2>, Output = Point<N, R1>;
    [val val] => Point::from(self * right.coords);
    [ref val] => Point::from(self * right.coords);
    [val ref] => Point::from(self * &right.coords);
    [ref ref] => Point::from(self * &right.coords);
);

/*
 *
 * Point ×/÷ Scalar
 *
 */
macro_rules! componentwise_scalarop_impl(
    ($Trait: ident, $method: ident, $bound: ident;
     $TraitAssign: ident, $method_assign: ident) => {
        impl<N: Scalar + $bound, D: DimName> $Trait<N> for Point<N, D>
            where DefaultAllocator: Allocator<N, D> {
            type Output = Point<N, D>;

            #[inline]
            fn $method(self, right: N) -> Self::Output {
                Point::from(self.coords.$method(right))
            }
        }

        impl<'a, N: Scalar + $bound, D: DimName> $Trait<N> for &'a Point<N, D>
            where DefaultAllocator: Allocator<N, D> {
            type Output = Point<N, D>;

            #[inline]
            fn $method(self, right: N) -> Self::Output {
                Point::from((&self.coords).$method(right))
            }
        }

        impl<N: Scalar + $bound, D: DimName> $TraitAssign<N> for Point<N, D>
            where DefaultAllocator: Allocator<N, D> {
            #[inline]
            fn $method_assign(&mut self, right: N) {
                self.coords.$method_assign(right)
            }
        }
    }
);

componentwise_scalarop_impl!(Mul, mul, ClosedMul; MulAssign, mul_assign);
componentwise_scalarop_impl!(Div, div, ClosedDiv; DivAssign, div_assign);

macro_rules! left_scalar_mul_impl(
    ($($T: ty),* $(,)*) => {$(
        impl<D: DimName> Mul<Point<$T, D>> for $T
            where DefaultAllocator: Allocator<$T, D> {
            type Output = Point<$T, D>;

            #[inline]
            fn mul(self, right: Point<$T, D>) -> Self::Output {
                Point::from(self * right.coords)
            }
        }

        impl<'b, D: DimName> Mul<&'b Point<$T, D>> for $T
            where DefaultAllocator: Allocator<$T, D> {
            type Output = Point<$T, D>;

            #[inline]
            fn mul(self, right: &'b Point<$T, D>) -> Self::Output {
                Point::from(self * &right.coords)
            }
        }
    )*}
);

left_scalar_mul_impl!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, f32, f64);
