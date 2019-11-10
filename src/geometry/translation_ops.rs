use std::ops::{Div, DivAssign, Mul, MulAssign};

use alga::general::{ClosedAdd, ClosedSub};

use crate::base::allocator::{Allocator, SameShapeAllocator};
use crate::base::constraint::{SameNumberOfColumns, SameNumberOfRows, ShapeConstraint};
use crate::{DimName, U1};
use crate::base::{DefaultAllocator, Scalar};

use crate::geometry::{Point, Translation};

// Translation × Translation
add_sub_impl!(Mul, mul, ClosedAdd;
    (D, U1), (D, U1) -> (D) for D: DimName;
    self: &'a Translation<N, D>, right: &'b Translation<N, D>, Output = Translation<N, D>;
    Translation::from(&self.vector + &right.vector); 'a, 'b);

add_sub_impl!(Mul, mul, ClosedAdd;
    (D, U1), (D, U1) -> (D) for D: DimName;
    self: &'a Translation<N, D>, right: Translation<N, D>, Output = Translation<N, D>;
    Translation::from(&self.vector + right.vector); 'a);

add_sub_impl!(Mul, mul, ClosedAdd;
    (D, U1), (D, U1) -> (D) for D: DimName;
    self: Translation<N, D>, right: &'b Translation<N, D>, Output = Translation<N, D>;
    Translation::from(self.vector + &right.vector); 'b);

add_sub_impl!(Mul, mul, ClosedAdd;
    (D, U1), (D, U1) -> (D) for D: DimName;
    self: Translation<N, D>, right: Translation<N, D>, Output = Translation<N, D>;
    Translation::from(self.vector + right.vector); );

// Translation ÷ Translation
// FIXME: instead of calling inverse explicitly, could we just add a `mul_tr` or `mul_inv` method?
add_sub_impl!(Div, div, ClosedSub;
    (D, U1), (D, U1) -> (D) for D: DimName;
    self: &'a Translation<N, D>, right: &'b Translation<N, D>, Output = Translation<N, D>;
    Translation::from(&self.vector - &right.vector); 'a, 'b);

add_sub_impl!(Div, div, ClosedSub;
    (D, U1), (D, U1) -> (D) for D: DimName;
    self: &'a Translation<N, D>, right: Translation<N, D>, Output = Translation<N, D>;
    Translation::from(&self.vector - right.vector); 'a);

add_sub_impl!(Div, div, ClosedSub;
    (D, U1), (D, U1) -> (D) for D: DimName;
    self: Translation<N, D>, right: &'b Translation<N, D>, Output = Translation<N, D>;
    Translation::from(self.vector - &right.vector); 'b);

add_sub_impl!(Div, div, ClosedSub;
    (D, U1), (D, U1) -> (D) for D: DimName;
    self: Translation<N, D>, right: Translation<N, D>, Output = Translation<N, D>;
    Translation::from(self.vector - right.vector); );

// Translation × Point
// FIXME: we don't handle properly non-zero origins here. Do we want this to be the intended
// behavior?
add_sub_impl!(Mul, mul, ClosedAdd;
    (D, U1), (D, U1) -> (D) for D: DimName;
    self: &'a Translation<N, D>, right: &'b Point<N, D>, Output = Point<N, D>;
    right + &self.vector; 'a, 'b);

add_sub_impl!(Mul, mul, ClosedAdd;
    (D, U1), (D, U1) -> (D) for D: DimName;
    self: &'a Translation<N, D>, right: Point<N, D>, Output = Point<N, D>;
    right + &self.vector; 'a);

add_sub_impl!(Mul, mul, ClosedAdd;
    (D, U1), (D, U1) -> (D) for D: DimName;
    self: Translation<N, D>, right: &'b Point<N, D>, Output = Point<N, D>;
    right + self.vector; 'b);

add_sub_impl!(Mul, mul, ClosedAdd;
    (D, U1), (D, U1) -> (D) for D: DimName;
    self: Translation<N, D>, right: Point<N, D>, Output = Point<N, D>;
    right + self.vector; );

// Translation *= Translation
add_sub_assign_impl!(MulAssign, mul_assign, ClosedAdd;
    (D, U1), (D, U1) for D: DimName;
    self: Translation<N, D>, right: &'b Translation<N, D>;
    self.vector += &right.vector; 'b);

add_sub_assign_impl!(MulAssign, mul_assign, ClosedAdd;
    (D, U1), (D, U1) for D: DimName;
    self: Translation<N, D>, right: Translation<N, D>;
    self.vector += right.vector; );

add_sub_assign_impl!(DivAssign, div_assign, ClosedSub;
    (D, U1), (D, U1) for D: DimName;
    self: Translation<N, D>, right: &'b Translation<N, D>;
    self.vector -= &right.vector; 'b);

add_sub_assign_impl!(DivAssign, div_assign, ClosedSub;
    (D, U1), (D, U1) for D: DimName;
    self: Translation<N, D>, right: Translation<N, D>;
    self.vector -= right.vector; );

/*
// Translation × Matrix
add_sub_impl!(Mul, mul;
         (D1, D1), (R2, C2) for D1, R2, C2;
         self: &'a Translation<N, D1>, right: &'b Matrix<N, R2, C2>, Output = MatrixMN<N, D1, C2>;
         self.vector() * right; 'a, 'b);

add_sub_impl!(Mul, mul;
         (D1, D1), (R2, C2) for D1, R2, C2;
         self: &'a Translation<N, D1>, right: Matrix<N, R2, C2>, Output = MatrixMN<N, D1, C2>;
         self.vector() * right; 'a);

add_sub_impl!(Mul, mul;
         (D1, D1), (R2, C2) for D1, R2, C2;
         self: Translation<N, D1>, right: &'b Matrix<N, R2, C2>, Output = MatrixMN<N, D1, C2>;
         self.unwrap() * right; 'b);

add_sub_impl!(Mul, mul;
         (D1, D1), (R2, C2) for D1, R2, C2;
         self: Translation<N, D1>, right: Matrix<N, R2, C2>, Output = MatrixMN<N, D1, C2>;
         self.unwrap() * right; );

// Matrix × Translation
add_sub_impl!(Mul, mul;
         (R1, C1), (D2, D2) for R1, C1, D2;
         self: &'a Matrix<N, R1, C1>, right: &'b Translation<N, D2>, Output = MatrixMN<N, R1, D2>;
         self * right.vector(); 'a, 'b);

add_sub_impl!(Mul, mul;
         (R1, C1), (D2, D2) for R1, C1, D2;
         self: &'a Matrix<N, R1, C1>, right: Translation<N, D2>, Output = MatrixMN<N, R1, D2>;
         self * right.unwrap(); 'a);

add_sub_impl!(Mul, mul;
         (R1, C1), (D2, D2) for R1, C1, D2;
         self: Matrix<N, R1, C1>, right: &'b Translation<N, D2>, Output = MatrixMN<N, R1, D2>;
         self * right.vector(); 'b);


add_sub_impl!(Mul, mul;
         (R1, C1), (D2, D2) for R1, C1, D2;
         self: Matrix<N, R1, C1>, right: Translation<N, D2>, Output = MatrixMN<N, R1, D2>;
         self * right.unwrap(); );

// Matrix *= Translation
md_assign_impl!(MulAssign, mul_assign;
                (R1, C1), (C1, C1) for R1, C1;
                self: Matrix<N, R1, C1>, right: &'b Translation<N, C1>;
                self.mul_assign(right.vector()); 'b);

md_assign_impl!(MulAssign, mul_assign;
                (R1, C1), (C1, C1) for R1, C1;
                self: Matrix<N, R1, C1>, right: Translation<N, C1>;
                self.mul_assign(right.unwrap()); );


md_assign_impl!(DivAssign, div_assign;
                (R1, C1), (C1, C1) for R1, C1;
                self: Matrix<N, R1, C1>, right: &'b Translation<N, C1>;
                self.mul_assign(right.inverse().vector()); 'b);

md_assign_impl!(DivAssign, div_assign;
                (R1, C1), (C1, C1) for R1, C1;
                self: Matrix<N, R1, C1>, right: Translation<N, C1>;
                self.mul_assign(right.inverse().unwrap()); );
*/
