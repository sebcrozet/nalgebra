use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use std::fmt;
use std::hash;
#[cfg(feature = "abomonation-serialize")]
use std::io::{Result as IOResult, Write};
use std::marker::PhantomData;

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "abomonation-serialize")]
use abomonation::Abomonation;

use alga::general::{RealField, SubsetOf};
use alga::linear::Rotation;

use crate::base::allocator::Allocator;
use crate::base::dimension::{DimName, DimNameAdd, DimNameSum, U1};
use crate::base::storage::Owned;
use crate::base::{DefaultAllocator, MatrixN, VectorN};
use crate::geometry::{Point, Translation};

/// A direct isometry, i.e., a rotation followed by a translation, aka. a rigid-body motion, aka. an element of a Special Euclidean (SE) group.
#[repr(C)]
#[derive(Debug)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(serialize = "R: Serialize,
                     DefaultAllocator: Allocator<N, D>,
                     Owned<N, D>: Serialize"))
)]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(deserialize = "R: Deserialize<'de>,
                       DefaultAllocator: Allocator<N, D>,
                       Owned<N, D>: Deserialize<'de>"))
)]
pub struct Isometry<N: RealField, D: DimName, R>
where DefaultAllocator: Allocator<N, D>
{
    /// The pure rotational part of this isometry.
    pub rotation: R,
    /// The pure translational part of this isometry.
    pub translation: Translation<N, D>,

    // One dummy private field just to prevent explicit construction.
    #[cfg_attr(
        feature = "serde-serialize",
        serde(skip_serializing, skip_deserializing)
    )]
    _noconstruct: PhantomData<N>,
}

#[cfg(feature = "abomonation-serialize")]
impl<N, D, R> Abomonation for Isometry<N, D, R>
where
    N: RealField,
    D: DimName,
    R: Abomonation,
    Translation<N, D>: Abomonation,
    DefaultAllocator: Allocator<N, D>,
{
    unsafe fn entomb<W: Write>(&self, writer: &mut W) -> IOResult<()> {
        self.rotation.entomb(writer)?;
        self.translation.entomb(writer)
    }

    fn extent(&self) -> usize {
        self.rotation.extent() + self.translation.extent()
    }

    unsafe fn exhume<'a, 'b>(&'a mut self, bytes: &'b mut [u8]) -> Option<&'b mut [u8]> {
        self.rotation
            .exhume(bytes)
            .and_then(|bytes| self.translation.exhume(bytes))
    }
}

impl<N: RealField + hash::Hash, D: DimName + hash::Hash, R: hash::Hash> hash::Hash for Isometry<N, D, R>
where
    DefaultAllocator: Allocator<N, D>,
    Owned<N, D>: hash::Hash,
{
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.translation.hash(state);
        self.rotation.hash(state);
    }
}

impl<N: RealField, D: DimName + Copy, R: Rotation<Point<N, D>> + Copy> Copy for Isometry<N, D, R>
where
    DefaultAllocator: Allocator<N, D>,
    Owned<N, D>: Copy,
{
}

impl<N: RealField, D: DimName, R: Rotation<Point<N, D>> + Clone> Clone for Isometry<N, D, R>
where DefaultAllocator: Allocator<N, D>
{
    #[inline]
    fn clone(&self) -> Self {
        Self::from_parts(self.translation.clone(), self.rotation.clone())
    }
}

impl<N: RealField, D: DimName, R: Rotation<Point<N, D>>> Isometry<N, D, R>
where DefaultAllocator: Allocator<N, D>
{
    /// Creates a new isometry from its rotational and translational parts.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3, Point3};
    /// let tra = Translation3::new(0.0, 0.0, 3.0);
    /// let rot = UnitQuaternion::from_scaled_axis(Vector3::y() * f32::consts::PI);
    /// let iso = Isometry3::from_parts(tra, rot);
    ///
    /// assert_relative_eq!(iso * Point3::new(1.0, 2.0, 3.0), Point3::new(-1.0, 2.0, 0.0), epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn from_parts(translation: Translation<N, D>, rotation: R) -> Self {
        Self {
            rotation: rotation,
            translation: translation,
            _noconstruct: PhantomData,
        }
    }

    /// Inverts `self`.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{Isometry2, Point2, Vector2};
    /// let iso = Isometry2::new(Vector2::new(1.0, 2.0), f32::consts::FRAC_PI_2);
    /// let inv = iso.inverse();
    /// let pt = Point2::new(1.0, 2.0);
    ///
    /// assert_eq!(inv * (iso * pt), pt);
    /// ```
    #[inline]
    #[must_use = "Did you mean to use inverse_mut()?"]
    pub fn inverse(&self) -> Self {
        let mut res = self.clone();
        res.inverse_mut();
        res
    }

    /// Inverts `self` in-place.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{Isometry2, Point2, Vector2};
    /// let mut iso = Isometry2::new(Vector2::new(1.0, 2.0), f32::consts::FRAC_PI_2);
    /// let pt = Point2::new(1.0, 2.0);
    /// let transformed_pt = iso * pt;
    /// iso.inverse_mut();
    ///
    /// assert_eq!(iso * transformed_pt, pt);
    /// ```
    #[inline]
    pub fn inverse_mut(&mut self) {
        self.rotation.two_sided_inverse_mut();
        self.translation.inverse_mut();
        self.translation.vector = self.rotation.transform_vector(&self.translation.vector);
    }

    /// Appends to `self` the given translation in-place.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{Isometry2, Translation2, Vector2};
    /// let mut iso = Isometry2::new(Vector2::new(1.0, 2.0), f32::consts::FRAC_PI_2);
    /// let tra = Translation2::new(3.0, 4.0);
    /// // Same as `iso = tra * iso`.
    /// iso.append_translation_mut(&tra);
    ///
    /// assert_eq!(iso.translation, Translation2::new(4.0, 6.0));
    /// ```
    #[inline]
    pub fn append_translation_mut(&mut self, t: &Translation<N, D>) {
        self.translation.vector += &t.vector
    }

    /// Appends to `self` the given rotation in-place.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Isometry2, Translation2, UnitComplex, Vector2};
    /// let mut iso = Isometry2::new(Vector2::new(1.0, 2.0), f32::consts::PI / 6.0);
    /// let rot = UnitComplex::new(f32::consts::PI / 2.0);
    /// // Same as `iso = rot * iso`.
    /// iso.append_rotation_mut(&rot);
    ///
    /// assert_relative_eq!(iso, Isometry2::new(Vector2::new(-2.0, 1.0), f32::consts::PI * 2.0 / 3.0), epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn append_rotation_mut(&mut self, r: &R) {
        self.rotation = self.rotation.append_rotation(&r);
        self.translation.vector = r.transform_vector(&self.translation.vector);
    }

    /// Appends in-place to `self` a rotation centered at the point `p`, i.e., the rotation that
    /// lets `p` invariant.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Isometry2, Translation2, UnitComplex, Vector2, Point2};
    /// let mut iso = Isometry2::new(Vector2::new(1.0, 2.0), f32::consts::FRAC_PI_2);
    /// let rot = UnitComplex::new(f32::consts::FRAC_PI_2);
    /// let pt = Point2::new(1.0, 0.0);
    /// iso.append_rotation_wrt_point_mut(&rot, &pt);
    ///
    /// assert_relative_eq!(iso * pt, Point2::new(-2.0, 0.0), epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn append_rotation_wrt_point_mut(&mut self, r: &R, p: &Point<N, D>) {
        self.translation.vector -= &p.coords;
        self.append_rotation_mut(r);
        self.translation.vector += &p.coords;
    }

    /// Appends in-place to `self` a rotation centered at the point with coordinates
    /// `self.translation`.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{Isometry2, Translation2, UnitComplex, Vector2, Point2};
    /// let mut iso = Isometry2::new(Vector2::new(1.0, 2.0), f32::consts::FRAC_PI_2);
    /// let rot = UnitComplex::new(f32::consts::FRAC_PI_2);
    /// iso.append_rotation_wrt_center_mut(&rot);
    ///
    /// // The translation part should not have changed.
    /// assert_eq!(iso.translation.vector, Vector2::new(1.0, 2.0));
    /// assert_eq!(iso.rotation, UnitComplex::new(f32::consts::PI));
    /// ```
    #[inline]
    pub fn append_rotation_wrt_center_mut(&mut self, r: &R) {
        self.rotation = self.rotation.append_rotation(r);
    }

    /// Transform the given point by this isometry.
    ///
    /// This is the same as the multiplication `self * pt`.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3, Point3};
    /// let tra = Translation3::new(0.0, 0.0, 3.0);
    /// let rot = UnitQuaternion::from_scaled_axis(Vector3::y() * f32::consts::FRAC_PI_2);
    /// let iso = Isometry3::from_parts(tra, rot);
    ///
    /// let transformed_point = iso.transform_point(&Point3::new(1.0, 2.0, 3.0));
    /// assert_relative_eq!(transformed_point, Point3::new(3.0, 2.0, 2.0), epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn transform_point(&self, pt: &Point<N, D>) -> Point<N, D> {
        self * pt
    }

    /// Transform the given vector by this isometry, ignoring the translation
    /// component of the isometry.
    ///
    /// This is the same as the multiplication `self * v`.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};
    /// let tra = Translation3::new(0.0, 0.0, 3.0);
    /// let rot = UnitQuaternion::from_scaled_axis(Vector3::y() * f32::consts::FRAC_PI_2);
    /// let iso = Isometry3::from_parts(tra, rot);
    ///
    /// let transformed_point = iso.transform_vector(&Vector3::new(1.0, 2.0, 3.0));
    /// assert_relative_eq!(transformed_point, Vector3::new(3.0, 2.0, -1.0), epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn transform_vector(&self, v: &VectorN<N, D>) -> VectorN<N, D> {
        self * v
    }

    /// Transform the given point by the inverse of this isometry. This may be
    /// less expensive than computing the entire isometry inverse and then
    /// transforming the point.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3, Point3};
    /// let tra = Translation3::new(0.0, 0.0, 3.0);
    /// let rot = UnitQuaternion::from_scaled_axis(Vector3::y() * f32::consts::FRAC_PI_2);
    /// let iso = Isometry3::from_parts(tra, rot);
    ///
    /// let transformed_point = iso.inverse_transform_point(&Point3::new(1.0, 2.0, 3.0));
    /// assert_relative_eq!(transformed_point, Point3::new(0.0, 2.0, 1.0), epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn inverse_transform_point(&self, pt: &Point<N, D>) -> Point<N, D> {
        self.rotation
            .inverse_transform_point(&(pt - &self.translation.vector))
    }

    /// Transform the given vector by the inverse of this isometry, ignoring the
    /// translation component of the isometry. This may be
    /// less expensive than computing the entire isometry inverse and then
    /// transforming the point.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};
    /// let tra = Translation3::new(0.0, 0.0, 3.0);
    /// let rot = UnitQuaternion::from_scaled_axis(Vector3::y() * f32::consts::FRAC_PI_2);
    /// let iso = Isometry3::from_parts(tra, rot);
    ///
    /// let transformed_point = iso.inverse_transform_vector(&Vector3::new(1.0, 2.0, 3.0));
    /// assert_relative_eq!(transformed_point, Vector3::new(-3.0, 2.0, 1.0), epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn inverse_transform_vector(&self, v: &VectorN<N, D>) -> VectorN<N, D> {
        self.rotation.inverse_transform_vector(v)
    }
}

// NOTE: we don't require `R: Rotation<...>` here because this is not useful for the implementation
// and makes it hard to use it, e.g., for Transform × Isometry implementation.
// This is OK since all constructors of the isometry enforce the Rotation bound already (and
// explicit struct construction is prevented by the dummy ZST field).
impl<N: RealField, D: DimName, R> Isometry<N, D, R>
where DefaultAllocator: Allocator<N, D>
{
    /// Converts this isometry into its equivalent homogeneous transformation matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Isometry2, Vector2, Matrix3};
    /// let iso = Isometry2::new(Vector2::new(10.0, 20.0), f32::consts::FRAC_PI_6);
    /// let expected = Matrix3::new(0.8660254, -0.5,      10.0,
    ///                             0.5,       0.8660254, 20.0,
    ///                             0.0,       0.0,       1.0);
    ///
    /// assert_relative_eq!(iso.to_homogeneous(), expected, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn to_homogeneous(&self) -> MatrixN<N, DimNameSum<D, U1>>
    where
        D: DimNameAdd<U1>,
        R: SubsetOf<MatrixN<N, DimNameSum<D, U1>>>,
        DefaultAllocator: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>>,
    {
        let mut res: MatrixN<N, _> = crate::convert_ref(&self.rotation);
        res.fixed_slice_mut::<D, U1>(0, D::dim())
            .copy_from(&self.translation.vector);

        res
    }
}

impl<N: RealField, D: DimName, R> Eq for Isometry<N, D, R>
where
    R: Rotation<Point<N, D>> + Eq,
    DefaultAllocator: Allocator<N, D>,
{
}

impl<N: RealField, D: DimName, R> PartialEq for Isometry<N, D, R>
where
    R: Rotation<Point<N, D>> + PartialEq,
    DefaultAllocator: Allocator<N, D>,
{
    #[inline]
    fn eq(&self, right: &Self) -> bool {
        self.translation == right.translation && self.rotation == right.rotation
    }
}

impl<N: RealField, D: DimName, R> AbsDiffEq for Isometry<N, D, R>
where
    R: Rotation<Point<N, D>> + AbsDiffEq<Epsilon = N::Epsilon>,
    DefaultAllocator: Allocator<N, D>,
    N::Epsilon: Copy,
{
    type Epsilon = N::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        N::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.translation.abs_diff_eq(&other.translation, epsilon)
            && self.rotation.abs_diff_eq(&other.rotation, epsilon)
    }
}

impl<N: RealField, D: DimName, R> RelativeEq for Isometry<N, D, R>
where
    R: Rotation<Point<N, D>> + RelativeEq<Epsilon = N::Epsilon>,
    DefaultAllocator: Allocator<N, D>,
    N::Epsilon: Copy,
{
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
        self.translation
            .relative_eq(&other.translation, epsilon, max_relative)
            && self
                .rotation
                .relative_eq(&other.rotation, epsilon, max_relative)
    }
}

impl<N: RealField, D: DimName, R> UlpsEq for Isometry<N, D, R>
where
    R: Rotation<Point<N, D>> + UlpsEq<Epsilon = N::Epsilon>,
    DefaultAllocator: Allocator<N, D>,
    N::Epsilon: Copy,
{
    #[inline]
    fn default_max_ulps() -> u32 {
        N::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.translation
            .ulps_eq(&other.translation, epsilon, max_ulps)
            && self.rotation.ulps_eq(&other.rotation, epsilon, max_ulps)
    }
}

/*
 *
 * Display
 *
 */
impl<N: RealField + fmt::Display, D: DimName, R> fmt::Display for Isometry<N, D, R>
where
    R: fmt::Display,
    DefaultAllocator: Allocator<N, D> + Allocator<usize, D>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let precision = f.precision().unwrap_or(3);

        writeln!(f, "Isometry {{")?;
        write!(f, "{:.*}", precision, self.translation)?;
        write!(f, "{:.*}", precision, self.rotation)?;
        writeln!(f, "}}")
    }
}
