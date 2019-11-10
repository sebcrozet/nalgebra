/*
 * This file provides:
 * ===================
 *
 *
 * (Quaternion)
 *
 * Index<usize>
 * IndexMut<usize>
 * Quaternion × Quaternion
 * Quaternion + Quaternion
 * Quaternion - Quaternion
 * -Quaternion
 * Quaternion × Scalar
 * Quaternion ÷ Scalar
 * Scalar × Quaternion
 *
 * (Unit Quaternion)
 * UnitQuaternion × UnitQuaternion
 * UnitQuaternion × Rotation       -> UnitQuaternion
 * Rotation       × UnitQuaternion -> UnitQuaternion
 *
 * UnitQuaternion ÷ UnitQuaternion
 * UnitQuaternion ÷ Rotation       -> UnitQuaternion
 * Rotation       ÷ UnitQuaternion -> UnitQuaternion
 *
 *
 * UnitQuaternion × Point
 * UnitQuaternion × Vector
 * UnitQuaternion × Unit<Vector>
 *
 * NOTE: -UnitQuaternion is already provided by `Unit<T>`.
 *
 *
 * (Assignment Operators)
 *
 * Quaternion ×= Scalar
 * Quaternion ×= Quaternion
 * Quaternion += Quaternion
 * Quaternion -= Quaternion
 *
 * UnitQuaternion ×= UnitQuaternion
 * UnitQuaternion ×= Rotation
 *
 * UnitQuaternion ÷= UnitQuaternion
 * UnitQuaternion ÷= Rotation
 *
 * FIXME: Rotation ×= UnitQuaternion
 * FIXME: Rotation ÷= UnitQuaternion
 *
 */

use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use alga::general::RealField;

use crate::base::allocator::Allocator;
use crate::{U1, U3, U4};
use crate::base::storage::Storage;
use crate::base::{DefaultAllocator, Unit, Vector, Vector3};

use crate::geometry::{Point3, Quaternion, Rotation, UnitQuaternion};

impl<N: RealField> Index<usize> for Quaternion<N> {
    type Output = N;

    #[inline]
    fn index(&self, i: usize) -> &Self::Output {
        &self.coords[i]
    }
}

impl<N: RealField> IndexMut<usize> for Quaternion<N> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut N {
        &mut self.coords[i]
    }
}

macro_rules! quaternion_op_impl(
    ($Op: ident, $op: ident;
     ($LhsRDim: ident, $LhsCDim: ident), ($RhsRDim: ident, $RhsCDim: ident)
     $(for $Storage: ident: $StoragesBound: ident $(<$($BoundParam: ty),*>)*),*;
     $lhs: ident: $Lhs: ty, $rhs: ident: $Rhs: ty, Output = $Result: ty $(=> $VDimA: ty, $VDimB: ty)*;
     $action: expr; $($lives: tt),*) => {
        impl<$($lives ,)* N: RealField $(, $Storage: $StoragesBound $(<$($BoundParam),*>)*)*> $Op<$Rhs> for $Lhs
            where DefaultAllocator: Allocator<N, $LhsRDim, $LhsCDim> +
                                    Allocator<N, $RhsRDim, $RhsCDim> {
            type Output = $Result;

            #[inline]
            fn $op($lhs, $rhs: $Rhs) -> Self::Output {
                $action
            }
        }
    }
);

// Quaternion + Quaternion
quaternion_op_impl!(
    Add, add;
    (U4, U1), (U4, U1);
    self: &'a Quaternion<N>, rhs: &'b Quaternion<N>, Output = Quaternion<N>;
    Quaternion::from(&self.coords + &rhs.coords);
    'a, 'b);

quaternion_op_impl!(
    Add, add;
    (U4, U1), (U4, U1);
    self: &'a Quaternion<N>, rhs: Quaternion<N>, Output = Quaternion<N>;
    Quaternion::from(&self.coords + rhs.coords);
    'a);

quaternion_op_impl!(
    Add, add;
    (U4, U1), (U4, U1);
    self: Quaternion<N>, rhs: &'b Quaternion<N>, Output = Quaternion<N>;
    Quaternion::from(self.coords + &rhs.coords);
    'b);

quaternion_op_impl!(
    Add, add;
    (U4, U1), (U4, U1);
    self: Quaternion<N>, rhs: Quaternion<N>, Output = Quaternion<N>;
    Quaternion::from(self.coords + rhs.coords);
    );

// Quaternion - Quaternion
quaternion_op_impl!(
    Sub, sub;
    (U4, U1), (U4, U1);
    self: &'a Quaternion<N>, rhs: &'b Quaternion<N>, Output = Quaternion<N>;
    Quaternion::from(&self.coords - &rhs.coords);
    'a, 'b);

quaternion_op_impl!(
    Sub, sub;
    (U4, U1), (U4, U1);
    self: &'a Quaternion<N>, rhs: Quaternion<N>, Output = Quaternion<N>;
    Quaternion::from(&self.coords - rhs.coords);
    'a);

quaternion_op_impl!(
    Sub, sub;
    (U4, U1), (U4, U1);
    self: Quaternion<N>, rhs: &'b Quaternion<N>, Output = Quaternion<N>;
    Quaternion::from(self.coords - &rhs.coords);
    'b);

quaternion_op_impl!(
    Sub, sub;
    (U4, U1), (U4, U1);
    self: Quaternion<N>, rhs: Quaternion<N>, Output = Quaternion<N>;
    Quaternion::from(self.coords - rhs.coords);
    );

// Quaternion × Quaternion
quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U4, U1);
    self: &'a Quaternion<N>, rhs: &'b Quaternion<N>, Output = Quaternion<N>;
    Quaternion::new(
        self[3] * rhs[3] - self[0] * rhs[0] - self[1] * rhs[1] - self[2] * rhs[2],
        self[3] * rhs[0] + self[0] * rhs[3] + self[1] * rhs[2] - self[2] * rhs[1],
        self[3] * rhs[1] - self[0] * rhs[2] + self[1] * rhs[3] + self[2] * rhs[0],
        self[3] * rhs[2] + self[0] * rhs[1] - self[1] * rhs[0] + self[2] * rhs[3]);
    'a, 'b);

quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U4, U1);
    self: &'a Quaternion<N>, rhs: Quaternion<N>, Output = Quaternion<N>;
    self * &rhs;
    'a);

quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U4, U1);
    self: Quaternion<N>, rhs: &'b Quaternion<N>, Output = Quaternion<N>;
    &self * rhs;
    'b);

quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U4, U1);
    self: Quaternion<N>, rhs: Quaternion<N>, Output = Quaternion<N>;
    &self * &rhs;
    );

// UnitQuaternion × UnitQuaternion
quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U4, U1);
    self: &'a UnitQuaternion<N>, rhs: &'b UnitQuaternion<N>, Output = UnitQuaternion<N>;
    UnitQuaternion::new_unchecked(self.quaternion() * rhs.quaternion());
    'a, 'b);

quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U4, U1);
    self: &'a UnitQuaternion<N>, rhs: UnitQuaternion<N>, Output = UnitQuaternion<N>;
    self * &rhs;
    'a);

quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U4, U1);
    self: UnitQuaternion<N>, rhs: &'b UnitQuaternion<N>, Output = UnitQuaternion<N>;
    &self * rhs;
    'b);

quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U4, U1);
    self: UnitQuaternion<N>, rhs: UnitQuaternion<N>, Output = UnitQuaternion<N>;
    &self * &rhs;
    );

// UnitQuaternion ÷ UnitQuaternion
quaternion_op_impl!(
    Div, div;
    (U4, U1), (U4, U1);
    self: &'a UnitQuaternion<N>, rhs: &'b UnitQuaternion<N>, Output = UnitQuaternion<N>;
    self * rhs.inverse();
    'a, 'b);

quaternion_op_impl!(
    Div, div;
    (U4, U1), (U4, U1);
    self: &'a UnitQuaternion<N>, rhs: UnitQuaternion<N>, Output = UnitQuaternion<N>;
    self / &rhs;
    'a);

quaternion_op_impl!(
    Div, div;
    (U4, U1), (U4, U1);
    self: UnitQuaternion<N>, rhs: &'b UnitQuaternion<N>, Output = UnitQuaternion<N>;
    &self / rhs;
    'b);

quaternion_op_impl!(
    Div, div;
    (U4, U1), (U4, U1);
    self: UnitQuaternion<N>, rhs: UnitQuaternion<N>, Output = UnitQuaternion<N>;
    &self / &rhs;
    );

// UnitQuaternion × Rotation
quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U3);
    self: &'a UnitQuaternion<N>, rhs: &'b Rotation<N, U3>,
    Output = UnitQuaternion<N> => U3, U3;
    // FIXME: can we avoid the conversion from a rotation matrix?
    self * UnitQuaternion::<N>::from_rotation_matrix(rhs);
    'a, 'b);

quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U3);
    self: &'a UnitQuaternion<N>, rhs: Rotation<N, U3>,
    Output = UnitQuaternion<N> => U3, U3;
    self * UnitQuaternion::<N>::from_rotation_matrix(&rhs);
    'a);

quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U3);
    self: UnitQuaternion<N>, rhs: &'b Rotation<N, U3>,
    Output = UnitQuaternion<N> => U3, U3;
    self * UnitQuaternion::<N>::from_rotation_matrix(rhs);
    'b);

quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U3);
    self: UnitQuaternion<N>, rhs: Rotation<N, U3>,
    Output = UnitQuaternion<N> => U3, U3;
    self * UnitQuaternion::<N>::from_rotation_matrix(&rhs);
    );

// UnitQuaternion ÷ Rotation
quaternion_op_impl!(
    Div, div;
    (U4, U1), (U3, U3);
    self: &'a UnitQuaternion<N>, rhs: &'b Rotation<N, U3>,
    Output = UnitQuaternion<N> => U3, U3;
    // FIXME: can we avoid the conversion to a rotation matrix?
    self / UnitQuaternion::<N>::from_rotation_matrix(rhs);
    'a, 'b);

quaternion_op_impl!(
    Div, div;
    (U4, U1), (U3, U3);
    self: &'a UnitQuaternion<N>, rhs: Rotation<N, U3>,
    Output = UnitQuaternion<N> => U3, U3;
    self / UnitQuaternion::<N>::from_rotation_matrix(&rhs);
    'a);

quaternion_op_impl!(
    Div, div;
    (U4, U1), (U3, U3);
    self: UnitQuaternion<N>, rhs: &'b Rotation<N, U3>,
    Output = UnitQuaternion<N> => U3, U3;
    self / UnitQuaternion::<N>::from_rotation_matrix(rhs);
    'b);

quaternion_op_impl!(
    Div, div;
    (U4, U1), (U3, U3);
    self: UnitQuaternion<N>, rhs: Rotation<N, U3>,
    Output = UnitQuaternion<N> => U3, U3;
    self / UnitQuaternion::<N>::from_rotation_matrix(&rhs);
    );

// Rotation × UnitQuaternion
quaternion_op_impl!(
    Mul, mul;
    (U3, U3), (U4, U1);
    self: &'a Rotation<N, U3>, rhs: &'b UnitQuaternion<N>,
    Output = UnitQuaternion<N> => U3, U3;
    // FIXME: can we avoid the conversion from a rotation matrix?
    UnitQuaternion::<N>::from_rotation_matrix(self) * rhs;
    'a, 'b);

quaternion_op_impl!(
    Mul, mul;
    (U3, U3), (U4, U1);
    self: &'a Rotation<N, U3>, rhs: UnitQuaternion<N>,
    Output = UnitQuaternion<N> => U3, U3;
    UnitQuaternion::<N>::from_rotation_matrix(self) * rhs;
    'a);

quaternion_op_impl!(
    Mul, mul;
    (U3, U3), (U4, U1);
    self: Rotation<N, U3>, rhs: &'b UnitQuaternion<N>,
    Output = UnitQuaternion<N> => U3, U3;
    UnitQuaternion::<N>::from_rotation_matrix(&self) * rhs;
    'b);

quaternion_op_impl!(
    Mul, mul;
    (U3, U3), (U4, U1);
    self: Rotation<N, U3>, rhs: UnitQuaternion<N>,
    Output = UnitQuaternion<N> => U3, U3;
    UnitQuaternion::<N>::from_rotation_matrix(&self) * rhs;
    );

// Rotation ÷ UnitQuaternion
quaternion_op_impl!(
    Div, div;
    (U3, U3), (U4, U1);
    self: &'a Rotation<N, U3>, rhs: &'b UnitQuaternion<N>,
    Output = UnitQuaternion<N> => U3, U3;
    // FIXME: can we avoid the conversion from a rotation matrix?
    UnitQuaternion::<N>::from_rotation_matrix(self) / rhs;
    'a, 'b);

quaternion_op_impl!(
    Div, div;
    (U3, U3), (U4, U1);
    self: &'a Rotation<N, U3>, rhs: UnitQuaternion<N>,
    Output = UnitQuaternion<N> => U3, U3;
    UnitQuaternion::<N>::from_rotation_matrix(self) / rhs;
    'a);

quaternion_op_impl!(
    Div, div;
    (U3, U3), (U4, U1);
    self: Rotation<N, U3>, rhs: &'b UnitQuaternion<N>,
    Output = UnitQuaternion<N> => U3, U3;
    UnitQuaternion::<N>::from_rotation_matrix(&self) / rhs;
    'b);

quaternion_op_impl!(
    Div, div;
    (U3, U3), (U4, U1);
    self: Rotation<N, U3>, rhs: UnitQuaternion<N>,
    Output = UnitQuaternion<N> => U3, U3;
    UnitQuaternion::<N>::from_rotation_matrix(&self) / rhs;
    );

// UnitQuaternion × Vector
quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U1) for SB: Storage<N, U3> ;
    self: &'a UnitQuaternion<N>, rhs: &'b Vector<N, U3, SB>,
    Output = Vector3<N> => U3, U4;
    {
        let two: N = crate::convert(2.0f64);
        let t = self.as_ref().vector().cross(rhs) * two;
        let cross = self.as_ref().vector().cross(&t);

        t * self.as_ref().scalar() + cross + rhs
    };
    'a, 'b);

quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U1) for SB: Storage<N, U3> ;
    self: &'a UnitQuaternion<N>, rhs: Vector<N, U3, SB>,
    Output = Vector3<N> => U3, U4;
    self * &rhs;
    'a);

quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U1) for SB: Storage<N, U3> ;
    self: UnitQuaternion<N>, rhs: &'b Vector<N, U3, SB>,
    Output = Vector3<N> => U3, U4;
    &self * rhs;
    'b);

quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U1) for SB: Storage<N, U3> ;
    self: UnitQuaternion<N>, rhs: Vector<N, U3, SB>,
    Output = Vector3<N> => U3, U4;
    &self * &rhs;
    );

// UnitQuaternion × Point
quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U1);
    self: &'a UnitQuaternion<N>, rhs: &'b Point3<N>,
    Output = Point3<N> => U3, U4;
    Point3::from(self * &rhs.coords);
    'a, 'b);

quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U1);
    self: &'a UnitQuaternion<N>, rhs: Point3<N>,
    Output = Point3<N> => U3, U4;
    Point3::from(self * rhs.coords);
    'a);

quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U1);
    self: UnitQuaternion<N>, rhs: &'b Point3<N>,
    Output = Point3<N> => U3, U4;
    Point3::from(self * &rhs.coords);
    'b);

quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U1);
    self: UnitQuaternion<N>, rhs: Point3<N>,
    Output = Point3<N> => U3, U4;
    Point3::from(self * rhs.coords);
    );

// UnitQuaternion × Unit<Vector>
quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U1) for SB: Storage<N, U3> ;
    self: &'a UnitQuaternion<N>, rhs: &'b Unit<Vector<N, U3, SB>>,
    Output = Unit<Vector3<N>> => U3, U4;
    Unit::new_unchecked(self * rhs.as_ref());
    'a, 'b);

quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U1) for SB: Storage<N, U3> ;
    self: &'a UnitQuaternion<N>, rhs: Unit<Vector<N, U3, SB>>,
    Output = Unit<Vector3<N>> => U3, U4;
    Unit::new_unchecked(self * rhs.into_inner());
    'a);

quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U1) for SB: Storage<N, U3> ;
    self: UnitQuaternion<N>, rhs: &'b Unit<Vector<N, U3, SB>>,
    Output = Unit<Vector3<N>> => U3, U4;
    Unit::new_unchecked(self * rhs.as_ref());
    'b);

quaternion_op_impl!(
    Mul, mul;
    (U4, U1), (U3, U1) for SB: Storage<N, U3> ;
    self: UnitQuaternion<N>, rhs: Unit<Vector<N, U3, SB>>,
    Output = Unit<Vector3<N>> => U3, U4;
    Unit::new_unchecked(self * rhs.into_inner());
    );

macro_rules! scalar_op_impl(
    ($($Op: ident, $op: ident, $OpAssign: ident, $op_assign: ident);* $(;)*) => {$(
        impl<N: RealField> $Op<N> for Quaternion<N> {
            type Output = Quaternion<N>;

            #[inline]
            fn $op(self, n: N) -> Self::Output {
                Quaternion::from(self.coords.$op(n))
            }
        }

        impl<'a, N: RealField> $Op<N> for &'a Quaternion<N> {
            type Output = Quaternion<N>;

            #[inline]
            fn $op(self, n: N) -> Self::Output {
                Quaternion::from((&self.coords).$op(n))
            }
        }

        impl<N: RealField> $OpAssign<N> for Quaternion<N> {

            #[inline]
            fn $op_assign(&mut self, n: N) {
                self.coords.$op_assign(n)
            }
        }
    )*}
);

scalar_op_impl!(
    Mul, mul, MulAssign, mul_assign;
    Div, div, DivAssign, div_assign;
);

macro_rules! left_scalar_mul_impl(
    ($($T: ty),* $(,)*) => {$(
        impl Mul<Quaternion<$T>> for $T {
            type Output = Quaternion<$T>;

            #[inline]
            fn mul(self, right: Quaternion<$T>) -> Self::Output {
                Quaternion::from(self * right.coords)
            }
        }

        impl<'b> Mul<&'b Quaternion<$T>> for $T {
            type Output = Quaternion<$T>;

            #[inline]
            fn mul(self, right: &'b Quaternion<$T>) -> Self::Output {
                Quaternion::from(self * &right.coords)
            }
        }
    )*}
);

left_scalar_mul_impl!(f32, f64);

impl<N: RealField> Neg for Quaternion<N> {
    type Output = Quaternion<N>;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::Output::from(-self.coords)
    }
}

impl<'a, N: RealField> Neg for &'a Quaternion<N> {
    type Output = Quaternion<N>;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::Output::from(-&self.coords)
    }
}

macro_rules! quaternion_op_impl(
    ($OpAssign: ident, $op_assign: ident;
     ($LhsRDim: ident, $LhsCDim: ident), ($RhsRDim: ident, $RhsCDim: ident);
     $lhs: ident: $Lhs: ty, $rhs: ident: $Rhs: ty $(=> $VDimA: ty, $VDimB: ty)*;
     $action: expr; $($lives: tt),*) => {
        impl<$($lives ,)* N: RealField> $OpAssign<$Rhs> for $Lhs
            where DefaultAllocator: Allocator<N, $LhsRDim, $LhsCDim> +
                                    Allocator<N, $RhsRDim, $RhsCDim> {

            #[inline]
            fn $op_assign(&mut $lhs, $rhs: $Rhs) {
                $action
            }
        }
    }
);

// Quaternion += Quaternion
quaternion_op_impl!(
    AddAssign, add_assign;
    (U4, U1), (U4, U1);
    self: Quaternion<N>, rhs: &'b Quaternion<N>;
    self.coords += &rhs.coords;
    'b);

quaternion_op_impl!(
    AddAssign, add_assign;
    (U4, U1), (U4, U1);
    self: Quaternion<N>, rhs: Quaternion<N>;
    self.coords += rhs.coords; );

// Quaternion -= Quaternion
quaternion_op_impl!(
    SubAssign, sub_assign;
    (U4, U1), (U4, U1);
    self: Quaternion<N>, rhs: &'b Quaternion<N>;
    self.coords -= &rhs.coords;
    'b);

quaternion_op_impl!(
    SubAssign, sub_assign;
    (U4, U1), (U4, U1);
    self: Quaternion<N>, rhs: Quaternion<N>;
    self.coords -= rhs.coords; );

// Quaternion ×= Quaternion
quaternion_op_impl!(
    MulAssign, mul_assign;
    (U4, U1), (U4, U1);
    self: Quaternion<N>, rhs: &'b Quaternion<N>;
    {
        let res = &*self * rhs;
        // FIXME: will this be optimized away?
        self.coords.copy_from(&res.coords);
    };
    'b);

quaternion_op_impl!(
    MulAssign, mul_assign;
    (U4, U1), (U4, U1);
    self: Quaternion<N>, rhs: Quaternion<N>;
    *self *= &rhs; );

// UnitQuaternion ×= UnitQuaternion
quaternion_op_impl!(
    MulAssign, mul_assign;
    (U4, U1), (U4, U1);
    self: UnitQuaternion<N>, rhs: &'b UnitQuaternion<N>;
    {
        let res = &*self * rhs;
        self.as_mut_unchecked().coords.copy_from(&res.as_ref().coords);
    };
    'b);

quaternion_op_impl!(
    MulAssign, mul_assign;
    (U4, U1), (U4, U1);
    self: UnitQuaternion<N>, rhs: UnitQuaternion<N>;
    *self *= &rhs; );

// UnitQuaternion ÷= UnitQuaternion
quaternion_op_impl!(
    DivAssign, div_assign;
    (U4, U1), (U4, U1);
    self: UnitQuaternion<N>, rhs: &'b UnitQuaternion<N>;
    {
        let res = &*self / rhs;
        self.as_mut_unchecked().coords.copy_from(&res.as_ref().coords);
    };
    'b);

quaternion_op_impl!(
    DivAssign, div_assign;
    (U4, U1), (U4, U1);
    self: UnitQuaternion<N>, rhs: UnitQuaternion<N>;
    *self /= &rhs; );

// UnitQuaternion ×= Rotation
quaternion_op_impl!(
    MulAssign, mul_assign;
    (U4, U1), (U3, U3);
    self: UnitQuaternion<N>, rhs: &'b Rotation<N, U3> => U3, U3;
    {
        let res = &*self * rhs;
        self.as_mut_unchecked().coords.copy_from(&res.as_ref().coords);
    };
    'b);

quaternion_op_impl!(
    MulAssign, mul_assign;
    (U4, U1), (U3, U3);
    self: UnitQuaternion<N>, rhs: Rotation<N, U3> => U3, U3;
    *self *= &rhs; );

// UnitQuaternion ÷= Rotation
quaternion_op_impl!(
    DivAssign, div_assign;
    (U4, U1), (U3, U3);
    self: UnitQuaternion<N>, rhs: &'b Rotation<N, U3> => U3, U3;
    {
        let res = &*self / rhs;
        self.as_mut_unchecked().coords.copy_from(&res.as_ref().coords);
    };
    'b);

quaternion_op_impl!(
    DivAssign, div_assign;
    (U4, U1), (U3, U3);
    self: UnitQuaternion<N>, rhs: Rotation<N, U3> => U3, U3;
    *self /= &rhs; );
