use std::ops::{Div, DivAssign, Mul, MulAssign};

use alga::general::RealField;
use crate::base::allocator::Allocator;
use crate::{U1, U2};
use crate::base::storage::Storage;
use crate::base::{DefaultAllocator, Unit, Vector, Vector2};
use crate::geometry::{Isometry, Point2, Rotation, Similarity, Translation, UnitComplex};

/*
 * This file provides:
 * ===================
 *
 * UnitComplex  × UnitComplex
 * UnitComplex  × Rotation -> UnitComplex
 * Rotation × UnitComplex  -> UnitComplex
 *
 * UnitComplex  ÷ UnitComplex
 * UnitComplex  ÷ Rotation -> UnitComplex
 * Rotation ÷ UnitComplex  -> UnitComplex
 *
 *
 * UnitComplex × Point
 * UnitComplex × Vector
 * UnitComplex × Unit<T>
 *
 * UnitComplex × Isometry<UnitComplex>
 * UnitComplex × Similarity<UnitComplex>
 * UnitComplex × Translation -> Isometry<UnitComplex>
 *
 * NOTE: -UnitComplex is already provided by `Unit<T>`.
 *
 * (Assignment Operators)
 *
 * UnitComplex  ×= UnitComplex
 * UnitComplex  ×= Rotation
 *
 * UnitComplex  ÷= UnitComplex
 * UnitComplex  ÷= Rotation
 *
 * Rotation ×= UnitComplex
 * Rotation ÷= UnitComplex
 *
 */

// UnitComplex × UnitComplex
impl<N: RealField> Mul<Self> for UnitComplex<N> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Unit::new_unchecked(self.into_inner() * rhs.into_inner())
    }
}

impl<'a, N: RealField> Mul<UnitComplex<N>> for &'a UnitComplex<N> {
    type Output = UnitComplex<N>;

    #[inline]
    fn mul(self, rhs: UnitComplex<N>) -> Self::Output {
        Unit::new_unchecked(self.complex() * rhs.into_inner())
    }
}

impl<'b, N: RealField> Mul<&'b UnitComplex<N>> for UnitComplex<N> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: &'b UnitComplex<N>) -> Self::Output {
        Unit::new_unchecked(self.into_inner() * rhs.complex())
    }
}

impl<'a, 'b, N: RealField> Mul<&'b UnitComplex<N>> for &'a UnitComplex<N> {
    type Output = UnitComplex<N>;

    #[inline]
    fn mul(self, rhs: &'b UnitComplex<N>) -> Self::Output {
        Unit::new_unchecked(self.complex() * rhs.complex())
    }
}

// UnitComplex ÷ UnitComplex
impl<N: RealField> Div<Self> for UnitComplex<N> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        Unit::new_unchecked(self.into_inner() * rhs.conjugate().into_inner())
    }
}

impl<'a, N: RealField> Div<UnitComplex<N>> for &'a UnitComplex<N> {
    type Output = UnitComplex<N>;

    #[inline]
    fn div(self, rhs: UnitComplex<N>) -> Self::Output {
        Unit::new_unchecked(self.complex() * rhs.conjugate().into_inner())
    }
}

impl<'b, N: RealField> Div<&'b UnitComplex<N>> for UnitComplex<N> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: &'b UnitComplex<N>) -> Self::Output {
        Unit::new_unchecked(self.into_inner() * rhs.conjugate().into_inner())
    }
}

impl<'a, 'b, N: RealField> Div<&'b UnitComplex<N>> for &'a UnitComplex<N> {
    type Output = UnitComplex<N>;

    #[inline]
    fn div(self, rhs: &'b UnitComplex<N>) -> Self::Output {
        Unit::new_unchecked(self.complex() * rhs.conjugate().into_inner())
    }
}

macro_rules! complex_op_impl(
    ($Op: ident, $op: ident;
     ($RDim: ident, $CDim: ident) $(for $Storage: ident: $StoragesBound: ident $(<$($BoundParam: ty),*>)*),*;
     $lhs: ident: $Lhs: ty, $rhs: ident: $Rhs: ty, Output = $Result: ty;
     $action: expr; $($lives: tt),*) => {
        impl<$($lives ,)* N: RealField $(, $Storage: $StoragesBound $(<$($BoundParam),*>)*)*> $Op<$Rhs> for $Lhs
            where DefaultAllocator: Allocator<N, $RDim, $CDim> {
            type Output = $Result;

            #[inline]
            fn $op($lhs, $rhs: $Rhs) -> Self::Output {
                $action
            }
        }
    }
);

macro_rules! complex_op_impl_all(
    ($Op: ident, $op: ident;
     ($RDim: ident, $CDim: ident) $(for $Storage: ident: $StoragesBound: ident $(<$($BoundParam: ty),*>)*),*;
     $lhs: ident: $Lhs: ty, $rhs: ident: $Rhs: ty, Output = $Result: ty;
     [val val] => $action_val_val: expr;
     [ref val] => $action_ref_val: expr;
     [val ref] => $action_val_ref: expr;
     [ref ref] => $action_ref_ref: expr;) => {

    complex_op_impl!($Op, $op;
                     ($RDim, $CDim) $(for $Storage: $StoragesBound $(<$($BoundParam),*>)*),*;
                     $lhs: $Lhs, $rhs: $Rhs, Output = $Result;
                     $action_val_val; );

    complex_op_impl!($Op, $op;
                     ($RDim, $CDim) $(for $Storage: $StoragesBound $(<$($BoundParam),*>)*),*;
                     $lhs: &'a $Lhs, $rhs: $Rhs, Output = $Result;
                     $action_ref_val; 'a);

    complex_op_impl!($Op, $op;
                     ($RDim, $CDim) $(for $Storage: $StoragesBound $(<$($BoundParam),*>)*),*;
                     $lhs: $Lhs, $rhs: &'b $Rhs, Output = $Result;
                     $action_val_ref; 'b);

    complex_op_impl!($Op, $op;
                     ($RDim, $CDim) $(for $Storage: $StoragesBound $(<$($BoundParam),*>)*),*;
                     $lhs: &'a $Lhs, $rhs: &'b $Rhs, Output = $Result;
                     $action_ref_ref; 'a, 'b);


    }
);

// UnitComplex × Rotation
complex_op_impl_all!(
    Mul, mul;
    (U2, U2);
    self: UnitComplex<N>, rhs: Rotation<N, U2>, Output = UnitComplex<N>;
    [val val] => &self * &rhs;
    [ref val] =>  self * &rhs;
    [val ref] => &self *  rhs;
    [ref ref] =>  self * UnitComplex::from_rotation_matrix(rhs);
);

// UnitComplex ÷ Rotation
complex_op_impl_all!(
    Div, div;
    (U2, U2);
    self: UnitComplex<N>, rhs: Rotation<N, U2>, Output = UnitComplex<N>;
    [val val] => &self / &rhs;
    [ref val] =>  self / &rhs;
    [val ref] => &self /  rhs;
    [ref ref] =>  self * UnitComplex::from_rotation_matrix(rhs).inverse();
);

// Rotation × UnitComplex
complex_op_impl_all!(
    Mul, mul;
    (U2, U2);
    self: Rotation<N, U2>, rhs: UnitComplex<N>, Output = UnitComplex<N>;
    [val val] => &self * &rhs;
    [ref val] =>  self * &rhs;
    [val ref] => &self *  rhs;
    [ref ref] => UnitComplex::from_rotation_matrix(self) * rhs;
);

// Rotation ÷ UnitComplex
complex_op_impl_all!(
    Div, div;
    (U2, U2);
    self: Rotation<N, U2>, rhs: UnitComplex<N>, Output = UnitComplex<N>;
    [val val] => &self / &rhs;
    [ref val] =>  self / &rhs;
    [val ref] => &self /  rhs;
    [ref ref] => UnitComplex::from_rotation_matrix(self) * rhs.inverse();
);

// UnitComplex × Point
complex_op_impl_all!(
    Mul, mul;
    (U2, U1);
    self: UnitComplex<N>, rhs: Point2<N>, Output = Point2<N>;
    [val val] => &self * &rhs;
    [ref val] =>  self * &rhs;
    [val ref] => &self *  rhs;
    [ref ref] => Point2::from(self * &rhs.coords);
);

// UnitComplex × Vector
complex_op_impl_all!(
    Mul, mul;
    (U2, U1) for S: Storage<N, U2>;
    self: UnitComplex<N>, rhs: Vector<N, U2, S>, Output = Vector2<N>;
    [val val] => &self * &rhs;
    [ref val] =>  self * &rhs;
    [val ref] => &self *  rhs;
    [ref ref] => {
        let i = self.as_ref().im;
        let r = self.as_ref().re;
        Vector2::new(r * rhs[0] - i * rhs[1], i * rhs[0] + r * rhs[1])
    };
);

// UnitComplex × Unit<Vector>
complex_op_impl_all!(
    Mul, mul;
    (U2, U1) for S: Storage<N, U2>;
    self: UnitComplex<N>, rhs: Unit<Vector<N, U2, S>>, Output = Unit<Vector2<N>>;
    [val val] => &self * &rhs;
    [ref val] =>  self * &rhs;
    [val ref] => &self *  rhs;
    [ref ref] => Unit::new_unchecked(self * rhs.as_ref());
);

// UnitComplex × Isometry<UnitComplex>
complex_op_impl_all!(
    Mul, mul;
    (U2, U1);
    self: UnitComplex<N>, rhs: Isometry<N, U2, UnitComplex<N>>,
    Output = Isometry<N, U2, UnitComplex<N>>;
    [val val] => &self * &rhs;
    [ref val] =>  self * &rhs;
    [val ref] => &self *  rhs;
    [ref ref] => {
        let shift = self * &rhs.translation.vector;
        Isometry::from_parts(Translation::from(shift), self * &rhs.rotation)
    };
);

// UnitComplex × Similarity<UnitComplex>
complex_op_impl_all!(
    Mul, mul;
    (U2, U1);
    self: UnitComplex<N>, rhs: Similarity<N, U2, UnitComplex<N>>,
    Output = Similarity<N, U2, UnitComplex<N>>;
    [val val] => &self * &rhs;
    [ref val] =>  self * &rhs;
    [val ref] => &self *  rhs;
    [ref ref] => Similarity::from_isometry(self * &rhs.isometry, rhs.scaling());
);

// UnitComplex × Translation
complex_op_impl_all!(
    Mul, mul;
    (U2, U1);
    self: UnitComplex<N>, rhs: Translation<N, U2>,
    Output = Isometry<N, U2, UnitComplex<N>>;
    [val val] => Isometry::from_parts(Translation::from(&self *  rhs.vector), self);
    [ref val] => Isometry::from_parts(Translation::from( self *  rhs.vector), self.clone());
    [val ref] => Isometry::from_parts(Translation::from(&self * &rhs.vector), self);
    [ref ref] => Isometry::from_parts(Translation::from( self * &rhs.vector), self.clone());
);

// Translation × UnitComplex
complex_op_impl_all!(
    Mul, mul;
    (U2, U1);
    self: Translation<N, U2>, right: UnitComplex<N>,
    Output = Isometry<N, U2, UnitComplex<N>>;
    [val val] => Isometry::from_parts(self, right);
    [ref val] => Isometry::from_parts(self.clone(), right);
    [val ref] => Isometry::from_parts(self, right.clone());
    [ref ref] => Isometry::from_parts(self.clone(), right.clone());
);

// UnitComplex ×= UnitComplex
impl<N: RealField> MulAssign<UnitComplex<N>> for UnitComplex<N> {
    #[inline]
    fn mul_assign(&mut self, rhs: UnitComplex<N>) {
        *self = &*self * rhs
    }
}

impl<'b, N: RealField> MulAssign<&'b UnitComplex<N>> for UnitComplex<N> {
    #[inline]
    fn mul_assign(&mut self, rhs: &'b UnitComplex<N>) {
        *self = &*self * rhs
    }
}

// UnitComplex /= UnitComplex
impl<N: RealField> DivAssign<UnitComplex<N>> for UnitComplex<N> {
    #[inline]
    fn div_assign(&mut self, rhs: UnitComplex<N>) {
        *self = &*self / rhs
    }
}

impl<'b, N: RealField> DivAssign<&'b UnitComplex<N>> for UnitComplex<N> {
    #[inline]
    fn div_assign(&mut self, rhs: &'b UnitComplex<N>) {
        *self = &*self / rhs
    }
}

// UnitComplex ×= Rotation
impl<N: RealField> MulAssign<Rotation<N, U2>> for UnitComplex<N>
where DefaultAllocator: Allocator<N, U2, U2>
{
    #[inline]
    fn mul_assign(&mut self, rhs: Rotation<N, U2>) {
        *self = &*self * rhs
    }
}

impl<'b, N: RealField> MulAssign<&'b Rotation<N, U2>> for UnitComplex<N>
where DefaultAllocator: Allocator<N, U2, U2>
{
    #[inline]
    fn mul_assign(&mut self, rhs: &'b Rotation<N, U2>) {
        *self = &*self * rhs
    }
}

// UnitComplex ÷= Rotation
impl<N: RealField> DivAssign<Rotation<N, U2>> for UnitComplex<N>
where DefaultAllocator: Allocator<N, U2, U2>
{
    #[inline]
    fn div_assign(&mut self, rhs: Rotation<N, U2>) {
        *self = &*self / rhs
    }
}

impl<'b, N: RealField> DivAssign<&'b Rotation<N, U2>> for UnitComplex<N>
where DefaultAllocator: Allocator<N, U2, U2>
{
    #[inline]
    fn div_assign(&mut self, rhs: &'b Rotation<N, U2>) {
        *self = &*self / rhs
    }
}

// Rotation ×= UnitComplex
impl<N: RealField> MulAssign<UnitComplex<N>> for Rotation<N, U2>
where DefaultAllocator: Allocator<N, U2, U2>
{
    #[inline]
    fn mul_assign(&mut self, rhs: UnitComplex<N>) {
        self.mul_assign(rhs.to_rotation_matrix())
    }
}

impl<'b, N: RealField> MulAssign<&'b UnitComplex<N>> for Rotation<N, U2>
where DefaultAllocator: Allocator<N, U2, U2>
{
    #[inline]
    fn mul_assign(&mut self, rhs: &'b UnitComplex<N>) {
        self.mul_assign(rhs.to_rotation_matrix())
    }
}

// Rotation ÷= UnitComplex
impl<N: RealField> DivAssign<UnitComplex<N>> for Rotation<N, U2>
where DefaultAllocator: Allocator<N, U2, U2>
{
    #[inline]
    fn div_assign(&mut self, rhs: UnitComplex<N>) {
        self.div_assign(rhs.to_rotation_matrix())
    }
}

impl<'b, N: RealField> DivAssign<&'b UnitComplex<N>> for Rotation<N, U2>
where DefaultAllocator: Allocator<N, U2, U2>
{
    #[inline]
    fn div_assign(&mut self, rhs: &'b UnitComplex<N>) {
        self.div_assign(rhs.to_rotation_matrix())
    }
}