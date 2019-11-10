use std::ops::{Div, DivAssign, Mul, MulAssign};

use alga::general::RealField;
use alga::linear::Rotation as AlgaRotation;

use crate::base::allocator::Allocator;
use crate::{DimName, U1, U3, U4};
use crate::base::{DefaultAllocator, VectorN};

use crate::geometry::{Isometry, Point, Rotation, Similarity, Translation, UnitQuaternion};

// FIXME: there are several cloning of rotations that we could probably get rid of (but we didn't
// yet because that would require to add a bound like `where for<'a, 'b> &'a R: Mul<&'b R, Output = R>`
// which is quite ugly.

/*
 *
 * In this file, we provide:
 * =========================
 *
 *
 * (Operators)
 *
 * Similarity × Similarity
 * Similarity × R
 * Similarity × Isometry
 *
 * Isometry × Similarity
 * Isometry ÷ Similarity
 *
 *
 * Similarity ÷ Similarity
 * Similarity ÷ R
 * Similarity ÷ Isometry
 *
 * Similarity × Point
 * Similarity × Vector
 *
 *
 * Similarity  × Translation
 * Translation × Similarity
 *
 * NOTE: The following are provided explicitly because we can't have R × Similarity.
 * Rotation   × Similarity<Rotation>
 * UnitQuaternion × Similarity<UnitQuaternion>
 *
 * Rotation   ÷ Similarity<Rotation>
 * UnitQuaternion ÷ Similarity<UnitQuaternion>
 *
 * (Assignment Operators)
 *
 * Similarity ×= Translation
 *
 * Similarity ×= Similarity
 * Similarity ×= Isometry
 * Similarity ×= R
 *
 * Similarity ÷= Similarity
 * Similarity ÷= Isometry
 * Similarity ÷= R
 *
 */

// XXX: code duplication: those macros are the same as for the isometry.
macro_rules! similarity_binop_impl(
    ($Op: ident, $op: ident;
     $lhs: ident: $Lhs: ty, $rhs: ident: $Rhs: ty, Output = $Output: ty;
     $action: expr; $($lives: tt),*) => {
        impl<$($lives ,)* N: RealField, D: DimName, R> $Op<$Rhs> for $Lhs
            where R: AlgaRotation<Point<N, D>>,
                  DefaultAllocator: Allocator<N, D> {
            type Output = $Output;

            #[inline]
            fn $op($lhs, $rhs: $Rhs) -> Self::Output {
                $action
            }
        }
    }
);

macro_rules! similarity_binop_impl_all(
    ($Op: ident, $op: ident;
     $lhs: ident: $Lhs: ty, $rhs: ident: $Rhs: ty, Output = $Output: ty;
     [val val] => $action_val_val: expr;
     [ref val] => $action_ref_val: expr;
     [val ref] => $action_val_ref: expr;
     [ref ref] => $action_ref_ref: expr;) => {
        similarity_binop_impl!(
            $Op, $op;
            $lhs: $Lhs, $rhs: $Rhs, Output = $Output;
            $action_val_val; );

        similarity_binop_impl!(
            $Op, $op;
            $lhs: &'a $Lhs, $rhs: $Rhs, Output = $Output;
            $action_ref_val; 'a);

        similarity_binop_impl!(
            $Op, $op;
            $lhs: $Lhs, $rhs: &'b $Rhs, Output = $Output;
            $action_val_ref; 'b);

        similarity_binop_impl!(
            $Op, $op;
            $lhs: &'a $Lhs, $rhs: &'b $Rhs, Output = $Output;
            $action_ref_ref; 'a, 'b);
    }
);

macro_rules! similarity_binop_assign_impl_all(
    ($OpAssign: ident, $op_assign: ident;
     $lhs: ident: $Lhs: ty, $rhs: ident: $Rhs: ty;
     [val] => $action_val: expr;
     [ref] => $action_ref: expr;) => {
        impl<N: RealField, D: DimName, R> $OpAssign<$Rhs> for $Lhs
            where R: AlgaRotation<Point<N, D>>,
                  DefaultAllocator: Allocator<N, D> {
            #[inline]
            fn $op_assign(&mut $lhs, $rhs: $Rhs) {
                $action_val
            }
        }

        impl<'b, N: RealField, D: DimName, R> $OpAssign<&'b $Rhs> for $Lhs
            where R: AlgaRotation<Point<N, D>>,
                  DefaultAllocator: Allocator<N, D> {
            #[inline]
            fn $op_assign(&mut $lhs, $rhs: &'b $Rhs) {
                $action_ref
            }
        }
    }
);

// Similarity × Similarity
// Similarity ÷ Similarity
similarity_binop_impl_all!(
    Mul, mul;
    self: Similarity<N, D, R>, rhs: Similarity<N, D, R>, Output = Similarity<N, D, R>;
    [val val] => &self * &rhs;
    [ref val] =>  self * &rhs;
    [val ref] => &self *  rhs;
    [ref ref] => {
        let mut res = self * &rhs.isometry;
        res.prepend_scaling_mut(rhs.scaling());
        res
    };
);

similarity_binop_impl_all!(
    Div, div;
    self: Similarity<N, D, R>, rhs: Similarity<N, D, R>, Output = Similarity<N, D, R>;
    [val val] => self * rhs.inverse();
    [ref val] => self * rhs.inverse();
    [val ref] => self * rhs.inverse();
    [ref ref] => self * rhs.inverse();
);

// Similarity ×= Translation
similarity_binop_assign_impl_all!(
    MulAssign, mul_assign;
    self: Similarity<N, D, R>, rhs: Translation<N, D>;
    [val] => *self *= &rhs;
    [ref] => {
        let shift = self.isometry.rotation.transform_vector(&rhs.vector) * self.scaling();
        self.isometry.translation.vector += shift;
    };
);

// Similarity ×= Similarity
// Similarity ÷= Similarity
similarity_binop_assign_impl_all!(
    MulAssign, mul_assign;
    self: Similarity<N, D, R>, rhs: Similarity<N, D, R>;
    [val] => *self *= &rhs;
    [ref] => {
        *self *= &rhs.isometry;
        self.prepend_scaling_mut(rhs.scaling());
    };
);

similarity_binop_assign_impl_all!(
    DivAssign, div_assign;
    self: Similarity<N, D, R>, rhs: Similarity<N, D, R>;
    [val] => *self /= &rhs;
    // FIXME: don't invert explicitly.
    [ref] => *self *= rhs.inverse();
);

// Similarity ×= Isometry
// Similarity ÷= Isometry
similarity_binop_assign_impl_all!(
    MulAssign, mul_assign;
    self: Similarity<N, D, R>, rhs: Isometry<N, D, R>;
    [val] => *self *= &rhs;
    [ref] => {
        let shift = self.isometry.rotation.transform_vector(&rhs.translation.vector) * self.scaling();
        self.isometry.translation.vector += shift;
        self.isometry.rotation *= rhs.rotation.clone();
    };
);

similarity_binop_assign_impl_all!(
    DivAssign, div_assign;
    self: Similarity<N, D, R>, rhs: Isometry<N, D, R>;
    [val] => *self /= &rhs;
    // FIXME: don't invert explicitly.
    [ref] => *self *= rhs.inverse();
);

// Similarity ×= R
// Similarity ÷= R
similarity_binop_assign_impl_all!(
    MulAssign, mul_assign;
    self: Similarity<N, D, R>, rhs: R;
    [val] => self.isometry.rotation *= rhs;
    [ref] => self.isometry.rotation *= rhs.clone();
);

similarity_binop_assign_impl_all!(
    DivAssign, div_assign;
    self: Similarity<N, D, R>, rhs: R;
    // FIXME: don't invert explicitly?
    [val] => *self *= rhs.two_sided_inverse();
    [ref] => *self *= rhs.two_sided_inverse();
);

// Similarity × R
// Similarity ÷ R
similarity_binop_impl_all!(
    Mul, mul;
    self: Similarity<N, D, R>, rhs: R, Output = Similarity<N, D, R>;
    [val val] => {
        let scaling = self.scaling();
        Similarity::from_isometry(self.isometry * rhs, scaling)
    };
    [ref val] => Similarity::from_isometry(&self.isometry * rhs, self.scaling());
    [val ref] => {
        let scaling = self.scaling();
        Similarity::from_isometry(self.isometry * rhs, scaling)
    };
    [ref ref] => Similarity::from_isometry(&self.isometry * rhs, self.scaling());
);

similarity_binop_impl_all!(
    Div, div;
    self: Similarity<N, D, R>, rhs: R, Output = Similarity<N, D, R>;
    [val val] => {
        let scaling = self.scaling();
        Similarity::from_isometry(self.isometry / rhs, scaling)
    };
    [ref val] => Similarity::from_isometry(&self.isometry / rhs, self.scaling());
    [val ref] => {
        let scaling = self.scaling();
        Similarity::from_isometry(self.isometry / rhs, scaling)
    };
    [ref ref] => Similarity::from_isometry(&self.isometry / rhs, self.scaling());
);

// Similarity × Isometry
// Similarity ÷ Isometry
similarity_binop_impl_all!(
    Mul, mul;
    self: Similarity<N, D, R>, rhs: Isometry<N, D, R>, Output = Similarity<N, D, R>;
    [val val] => &self * &rhs;
    [ref val] => self * &rhs;
    [val ref] => &self * rhs;
    [ref ref] => {
        let shift = self.isometry.rotation.transform_vector(&rhs.translation.vector) * self.scaling();
        Similarity::from_parts(
            Translation::from(&self.isometry.translation.vector + shift),
            self.isometry.rotation.clone() * rhs.rotation.clone(),
            self.scaling())
    };
);

similarity_binop_impl_all!(
    Div, div;
    self: Similarity<N, D, R>, rhs: Isometry<N, D, R>, Output = Similarity<N, D, R>;
    [val val] => self * rhs.inverse();
    [ref val] => self * rhs.inverse();
    [val ref] => self * rhs.inverse();
    [ref ref] => self * rhs.inverse();
);

// Isometry × Similarity
// Isometry ÷ Similarity
similarity_binop_impl_all!(
    Mul, mul;
    self: Isometry<N, D, R>, rhs: Similarity<N, D, R>, Output = Similarity<N, D, R>;
    [val val] => {
        let scaling = rhs.scaling();
        Similarity::from_isometry(self * rhs.isometry, scaling)
    };
    [ref val] => {
        let scaling = rhs.scaling();
        Similarity::from_isometry(self * rhs.isometry, scaling)
    };
    [val ref] => {
        let scaling = rhs.scaling();
        Similarity::from_isometry(self * &rhs.isometry, scaling)
    };
    [ref ref] => {
        let scaling = rhs.scaling();
        Similarity::from_isometry(self * &rhs.isometry, scaling)
    };
);

similarity_binop_impl_all!(
    Div, div;
    self: Isometry<N, D, R>, rhs: Similarity<N, D, R>, Output = Similarity<N, D, R>;
    [val val] => self * rhs.inverse();
    [ref val] => self * rhs.inverse();
    [val ref] => self * rhs.inverse();
    [ref ref] => self * rhs.inverse();
);

// Similarity × Point
similarity_binop_impl_all!(
    Mul, mul;
    self: Similarity<N, D, R>, right: Point<N, D>, Output = Point<N, D>;
    [val val] => {
        let scaling = self.scaling();
        self.isometry.translation * (self.isometry.rotation.transform_point(&right) * scaling)
    };
    [ref val] => &self.isometry.translation * (self.isometry.rotation.transform_point(&right) * self.scaling());
    [val ref] => {
        let scaling = self.scaling();
        self.isometry.translation * (self.isometry.rotation.transform_point(right) * scaling)
    };
    [ref ref] => &self.isometry.translation * (self.isometry.rotation.transform_point(right) * self.scaling());
);

// Similarity × Vector
similarity_binop_impl_all!(
    Mul, mul;
    self: Similarity<N, D, R>, right: VectorN<N, D>, Output = VectorN<N, D>;
    [val val] => self.isometry.rotation.transform_vector(&right) * self.scaling();
    [ref val] => self.isometry.rotation.transform_vector(&right) * self.scaling();
    [val ref] => self.isometry.rotation.transform_vector(right) * self.scaling();
    [ref ref] => self.isometry.rotation.transform_vector(right) * self.scaling();
);

// Similarity × Translation
similarity_binop_impl_all!(
    Mul, mul;
    self: Similarity<N, D, R>, right: Translation<N, D>, Output = Similarity<N, D, R>;
    [val val] => &self * &right;
    [ref val] => self * &right;
    [val ref] => &self * right;
    [ref ref] => {
        let shift = self.isometry.rotation.transform_vector(&right.vector) * self.scaling();
        Similarity::from_parts(
            Translation::from(&self.isometry.translation.vector + shift),
            self.isometry.rotation.clone(),
            self.scaling())
    };
);

// Translation × Similarity
similarity_binop_impl_all!(
    Mul, mul;
    self: Translation<N, D>, right: Similarity<N, D, R>, Output = Similarity<N, D, R>;
    [val val] => {
        let scaling = right.scaling();
        Similarity::from_isometry(self * right.isometry, scaling)
    };
    [ref val] => {
        let scaling = right.scaling();
        Similarity::from_isometry(self * right.isometry, scaling)
    };
    [val ref] => Similarity::from_isometry(self * &right.isometry, right.scaling());
    [ref ref] => Similarity::from_isometry(self * &right.isometry, right.scaling());
);

macro_rules! similarity_from_composition_impl(
    ($Op: ident, $op: ident;
     ($R1: ty, $C1: ty),($R2: ty, $C2: ty) $(for $Dims: ident: $DimsBound: ident),*;
     $lhs: ident: $Lhs: ty, $rhs: ident: $Rhs: ty, Output = $Output: ty;
     $action: expr; $($lives: tt),*) => {
        impl<$($lives ,)* N: RealField $(, $Dims: $DimsBound)*> $Op<$Rhs> for $Lhs
            where DefaultAllocator: Allocator<N, $R1, $C1> +
                                    Allocator<N, $R2, $C2> {
            type Output = $Output;

            #[inline]
            fn $op($lhs, $rhs: $Rhs) -> Self::Output {
                $action
            }
        }
    }
);

macro_rules! similarity_from_composition_impl_all(
    ($Op: ident, $op: ident;
     ($R1: ty, $C1: ty),($R2: ty, $C2: ty) $(for $Dims: ident: $DimsBound: ident),*;
     $lhs: ident: $Lhs: ty, $rhs: ident: $Rhs: ty, Output = $Output: ty;
     [val val] => $action_val_val: expr;
     [ref val] => $action_ref_val: expr;
     [val ref] => $action_val_ref: expr;
     [ref ref] => $action_ref_ref: expr;) => {

        similarity_from_composition_impl!(
            $Op, $op;
            ($R1, $C1),($R2, $C2) $(for $Dims: $DimsBound),*;
            $lhs: $Lhs, $rhs: $Rhs, Output = $Output;
            $action_val_val; );

        similarity_from_composition_impl!(
            $Op, $op;
            ($R1, $C1),($R2, $C2) $(for $Dims: $DimsBound),*;
            $lhs: &'a $Lhs, $rhs: $Rhs, Output = $Output;
            $action_ref_val; 'a);

        similarity_from_composition_impl!(
            $Op, $op;
            ($R1, $C1),($R2, $C2) $(for $Dims: $DimsBound),*;
            $lhs: $Lhs, $rhs: &'b $Rhs, Output = $Output;
            $action_val_ref; 'b);

        similarity_from_composition_impl!(
            $Op, $op;
            ($R1, $C1),($R2, $C2) $(for $Dims: $DimsBound),*;
            $lhs: &'a $Lhs, $rhs: &'b $Rhs, Output = $Output;
            $action_ref_ref; 'a, 'b);
    }
);

// Rotation × Similarity
similarity_from_composition_impl_all!(
    Mul, mul;
    (D, D), (D, U1) for D: DimName;
    self: Rotation<N, D>, right: Similarity<N, D, Rotation<N, D>>,
    Output = Similarity<N, D, Rotation<N, D>>;
    [val val] => &self * &right;
    [ref val] =>  self * &right;
    [val ref] => &self *  right;
    [ref ref] => Similarity::from_isometry(self * &right.isometry, right.scaling());
);

// Rotation ÷ Similarity
similarity_from_composition_impl_all!(
    Div, div;
    (D, D), (D, U1) for D: DimName;
    self: Rotation<N, D>, right: Similarity<N, D, Rotation<N, D>>,
    Output = Similarity<N, D, Rotation<N, D>>;
    // FIXME: don't call inverse explicitly?
    [val val] => self * right.inverse();
    [ref val] => self * right.inverse();
    [val ref] => self * right.inverse();
    [ref ref] => self * right.inverse();
);

// UnitQuaternion × Similarity
similarity_from_composition_impl_all!(
    Mul, mul;
    (U4, U1), (U3, U1);
    self: UnitQuaternion<N>, right: Similarity<N, U3, UnitQuaternion<N>>,
    Output = Similarity<N, U3, UnitQuaternion<N>>;
    [val val] => &self * &right;
    [ref val] =>  self * &right;
    [val ref] => &self *  right;
    [ref ref] => Similarity::from_isometry(self * &right.isometry, right.scaling());
);

// UnitQuaternion ÷ Similarity
similarity_from_composition_impl_all!(
    Div, div;
    (U4, U1), (U3, U1);
    self: UnitQuaternion<N>, right: Similarity<N, U3, UnitQuaternion<N>>,
    Output = Similarity<N, U3, UnitQuaternion<N>>;
    // FIXME: don't call inverse explicitly?
    [val val] => self * right.inverse();
    [ref val] => self * right.inverse();
    [val ref] => self * right.inverse();
    [ref ref] => self * right.inverse();
);
