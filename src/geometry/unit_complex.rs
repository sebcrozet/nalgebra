use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use num_complex::Complex;
use std::fmt;

use alga::general::RealField;
use crate::base::{Matrix2, Matrix3, Unit, Vector1, Vector2};
use crate::geometry::{Rotation2, Point2};

/// A complex number with a norm equal to 1.
pub type UnitComplex<N> = Unit<Complex<N>>;

impl<N: RealField> UnitComplex<N> {
    /// The rotation angle in `]-pi; pi]` of this unit complex number.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::UnitComplex;
    /// let rot = UnitComplex::new(1.78);
    /// assert_eq!(rot.angle(), 1.78);
    /// ```
    #[inline]
    pub fn angle(&self) -> N {
        self.im.atan2(self.re)
    }

    /// The sine of the rotation angle.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::UnitComplex;
    /// let angle = 1.78f32;
    /// let rot = UnitComplex::new(angle);
    /// assert_eq!(rot.sin_angle(), angle.sin());
    /// ```
    #[inline]
    pub fn sin_angle(&self) -> N {
        self.im
    }

    /// The cosine of the rotation angle.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::UnitComplex;
    /// let angle = 1.78f32;
    /// let rot = UnitComplex::new(angle);
    /// assert_eq!(rot.cos_angle(),angle.cos());
    /// ```
    #[inline]
    pub fn cos_angle(&self) -> N {
        self.re
    }

    /// The rotation angle returned as a 1-dimensional vector.
    ///
    /// This is generally used in the context of generic programming. Using
    /// the `.angle()` method instead is more common.
    #[inline]
    pub fn scaled_axis(&self) -> Vector1<N> {
        Vector1::new(self.angle())
    }

    /// The rotation axis and angle in ]0, pi] of this complex number.
    ///
    /// This is generally used in the context of generic programming. Using
    /// the `.angle()` method instead is more common.
    /// Returns `None` if the angle is zero.
    #[inline]
    pub fn axis_angle(&self) -> Option<(Unit<Vector1<N>>, N)> {
        let ang = self.angle();

        if ang.is_zero() {
            None
        } else if ang.is_sign_negative() {
            Some((Unit::new_unchecked(Vector1::x()), -ang))
        } else {
            Some((Unit::new_unchecked(-Vector1::<N>::x()), ang))
        }
    }

    /// The underlying complex number.
    ///
    /// Same as `self.as_ref()`.
    ///
    /// # Example
    /// ```
    /// # extern crate num_complex;
    /// # use num_complex::Complex;
    /// # use nalgebra::UnitComplex;
    /// let angle = 1.78f32;
    /// let rot = UnitComplex::new(angle);
    /// assert_eq!(*rot.complex(), Complex::new(angle.cos(), angle.sin()));
    /// ```
    #[inline]
    pub fn complex(&self) -> &Complex<N> {
        self.as_ref()
    }

    /// Compute the conjugate of this unit complex number.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::UnitComplex;
    /// let rot = UnitComplex::new(1.78);
    /// let conj = rot.conjugate();
    /// assert_eq!(rot.complex().im, -conj.complex().im);
    /// assert_eq!(rot.complex().re, conj.complex().re);
    /// ```
    #[inline]
    #[must_use = "Did you mean to use conjugate_mut()?"]
    pub fn conjugate(&self) -> Self {
        Self::new_unchecked(self.conj())
    }

    /// Inverts this complex number if it is not zero.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::UnitComplex;
    /// let rot = UnitComplex::new(1.2);
    /// let inv = rot.inverse();
    /// assert_relative_eq!(rot * inv, UnitComplex::identity(), epsilon = 1.0e-6);
    /// assert_relative_eq!(inv * rot, UnitComplex::identity(), epsilon = 1.0e-6);
    /// ```
    #[inline]
    #[must_use = "Did you mean to use inverse_mut()?"]
    pub fn inverse(&self) -> Self {
        self.conjugate()
    }

    /// The rotation angle needed to make `self` and `other` coincide.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::UnitComplex;
    /// let rot1 = UnitComplex::new(0.1);
    /// let rot2 = UnitComplex::new(1.7);
    /// assert_relative_eq!(rot1.angle_to(&rot2), 1.6);
    /// ```
    #[inline]
    pub fn angle_to(&self, other: &Self) -> N {
        let delta = self.rotation_to(other);
        delta.angle()
    }

    /// The unit complex number needed to make `self` and `other` coincide.
    ///
    /// The result is such that: `self.rotation_to(other) * self == other`.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::UnitComplex;
    /// let rot1 = UnitComplex::new(0.1);
    /// let rot2 = UnitComplex::new(1.7);
    /// let rot_to = rot1.rotation_to(&rot2);
    ///
    /// assert_relative_eq!(rot_to * rot1, rot2);
    /// assert_relative_eq!(rot_to.inverse() * rot2, rot1);
    /// ```
    #[inline]
    pub fn rotation_to(&self, other: &Self) -> Self {
        other / self
    }

    /// Compute in-place the conjugate of this unit complex number.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::UnitComplex;
    /// let angle = 1.7;
    /// let rot = UnitComplex::new(angle);
    /// let mut conj = UnitComplex::new(angle);
    /// conj.conjugate_mut();
    /// assert_eq!(rot.complex().im, -conj.complex().im);
    /// assert_eq!(rot.complex().re, conj.complex().re);
    /// ```
    #[inline]
    pub fn conjugate_mut(&mut self) {
        let me = self.as_mut_unchecked();
        me.im = -me.im;
    }

    /// Inverts in-place this unit complex number.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::UnitComplex;
    /// let angle = 1.7;
    /// let mut rot = UnitComplex::new(angle);
    /// rot.inverse_mut();
    /// assert_relative_eq!(rot * UnitComplex::new(angle), UnitComplex::identity());
    /// assert_relative_eq!(UnitComplex::new(angle) * rot, UnitComplex::identity());
    /// ```
    #[inline]
    pub fn inverse_mut(&mut self) {
        self.conjugate_mut()
    }

    /// Raise this unit complex number to a given floating power.
    ///
    /// This returns the unit complex number that identifies a rotation angle equal to
    /// `self.angle() × n`.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::UnitComplex;
    /// let rot = UnitComplex::new(0.78);
    /// let pow = rot.powf(2.0);
    /// assert_relative_eq!(pow.angle(), 2.0 * 0.78);
    /// ```
    #[inline]
    pub fn powf(&self, n: N) -> Self {
        Self::from_angle(self.angle() * n)
    }

    /// Builds the rotation matrix corresponding to this unit complex number.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{UnitComplex, Rotation2};
    /// # use std::f32;
    /// let rot = UnitComplex::new(f32::consts::FRAC_PI_6);
    /// let expected = Rotation2::new(f32::consts::FRAC_PI_6);
    /// assert_eq!(rot.to_rotation_matrix(), expected);
    /// ```
    #[inline]
    pub fn to_rotation_matrix(&self) -> Rotation2<N> {
        let r = self.re;
        let i = self.im;

        Rotation2::from_matrix_unchecked(Matrix2::new(r, -i, i, r))
    }

    /// Converts this unit complex number into its equivalent homogeneous transformation matrix.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{UnitComplex, Matrix3};
    /// # use std::f32;
    /// let rot = UnitComplex::new(f32::consts::FRAC_PI_6);
    /// let expected = Matrix3::new(0.8660254, -0.5,      0.0,
    ///                             0.5,       0.8660254, 0.0,
    ///                             0.0,       0.0,       1.0);
    /// assert_eq!(rot.to_homogeneous(), expected);
    /// ```
    #[inline]
    pub fn to_homogeneous(&self) -> Matrix3<N> {
        self.to_rotation_matrix().to_homogeneous()
    }

    /// Rotate the given point by this unit complex number.
    ///
    /// This is the same as the multiplication `self * pt`.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitComplex, Point2};
    /// # use std::f32;
    /// let rot = UnitComplex::new(f32::consts::FRAC_PI_2);
    /// let transformed_point = rot.transform_point(&Point2::new(1.0, 2.0));
    /// assert_relative_eq!(transformed_point, Point2::new(-2.0, 1.0), epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn transform_point(&self, pt: &Point2<N>) -> Point2<N> {
        self * pt
    }

    /// Rotate the given vector by this unit complex number.
    ///
    /// This is the same as the multiplication `self * v`.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitComplex, Vector2};
    /// # use std::f32;
    /// let rot = UnitComplex::new(f32::consts::FRAC_PI_2);
    /// let transformed_vector = rot.transform_vector(&Vector2::new(1.0, 2.0));
    /// assert_relative_eq!(transformed_vector, Vector2::new(-2.0, 1.0), epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn transform_vector(&self, v: &Vector2<N>) -> Vector2<N> {
        self * v
    }

    /// Rotate the given point by the inverse of this unit complex number.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitComplex, Point2};
    /// # use std::f32;
    /// let rot = UnitComplex::new(f32::consts::FRAC_PI_2);
    /// let transformed_point = rot.inverse_transform_point(&Point2::new(1.0, 2.0));
    /// assert_relative_eq!(transformed_point, Point2::new(2.0, -1.0), epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn inverse_transform_point(&self, pt: &Point2<N>) -> Point2<N> {
        // FIXME: would it be useful performancewise not to call inverse explicitly (i-e. implement
        // the inverse transformation explicitly here) ?
        self.inverse() * pt
    }

    /// Rotate the given vector by the inverse of this unit complex number.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{UnitComplex, Vector2};
    /// # use std::f32;
    /// let rot = UnitComplex::new(f32::consts::FRAC_PI_2);
    /// let transformed_vector = rot.inverse_transform_vector(&Vector2::new(1.0, 2.0));
    /// assert_relative_eq!(transformed_vector, Vector2::new(2.0, -1.0), epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn inverse_transform_vector(&self, v: &Vector2<N>) -> Vector2<N> {
        self.inverse() * v
    }
}

impl<N: RealField + fmt::Display> fmt::Display for UnitComplex<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "UnitComplex angle: {}", self.angle())
    }
}

impl<N: RealField> AbsDiffEq for UnitComplex<N> {
    type Epsilon = N;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        N::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.re.abs_diff_eq(&other.re, epsilon) && self.im.abs_diff_eq(&other.im, epsilon)
    }
}

impl<N: RealField> RelativeEq for UnitComplex<N> {
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        N::default_max_relative()
    }

    #[inline]
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool
    {
        self.re.relative_eq(&other.re, epsilon, max_relative)
            && self.im.relative_eq(&other.im, epsilon, max_relative)
    }
}

impl<N: RealField> UlpsEq for UnitComplex<N> {
    #[inline]
    fn default_max_ulps() -> u32 {
        N::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.re.ulps_eq(&other.re, epsilon, max_ulps)
            && self.im.ulps_eq(&other.im, epsilon, max_ulps)
    }
}