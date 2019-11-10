#[cfg(feature = "arbitrary")]
use crate::base::storage::Owned;
#[cfg(feature = "arbitrary")]
use quickcheck::{Arbitrary, Gen};

use num::One;
use rand::distributions::{Distribution, Standard};
use rand::Rng;

use alga::general::RealField;
use alga::linear::Rotation as AlgaRotation;

use crate::base::allocator::Allocator;
use crate::{DimName, U2, U3};
use crate::base::{DefaultAllocator, Vector2, Vector3};

use crate::geometry::{
    Isometry, Point, Point3, Rotation, Rotation2, Rotation3, Translation, UnitComplex,
    UnitQuaternion, Translation2, Translation3
};

impl<N: RealField, D: DimName, R: AlgaRotation<Point<N, D>>> Isometry<N, D, R>
where DefaultAllocator: Allocator<N, D>
{
    /// Creates a new identity isometry.
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::{Isometry2, Point2, Isometry3, Point3};
    ///
    /// let iso = Isometry2::identity();
    /// let pt = Point2::new(1.0, 2.0);
    /// assert_eq!(iso * pt, pt);
    ///
    /// let iso = Isometry3::identity();
    /// let pt = Point3::new(1.0, 2.0, 3.0);
    /// assert_eq!(iso * pt, pt);
    /// ```
    #[inline]
    pub fn identity() -> Self {
        Self::from_parts(Translation::identity(), R::identity())
    }

    /// The isometry that applies the rotation `r` with its axis passing through the point `p`.
    /// This effectively lets `p` invariant.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Isometry2, Point2, UnitComplex};
    /// let rot = UnitComplex::new(f32::consts::PI);
    /// let pt = Point2::new(1.0, 0.0);
    /// let iso = Isometry2::rotation_wrt_point(rot, pt);
    ///
    /// assert_eq!(iso * pt, pt); // The rotation center is not affected.
    /// assert_relative_eq!(iso * Point2::new(1.0, 2.0), Point2::new(1.0, -2.0), epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn rotation_wrt_point(r: R, p: Point<N, D>) -> Self {
        let shift = r.transform_vector(&-&p.coords);
        Self::from_parts(Translation::from(shift + p.coords), r)
    }
}

impl<N: RealField, D: DimName, R: AlgaRotation<Point<N, D>>> One for Isometry<N, D, R>
where DefaultAllocator: Allocator<N, D>
{
    /// Creates a new identity isometry.
    #[inline]
    fn one() -> Self {
        Self::identity()
    }
}

impl<N: RealField, D: DimName, R> Distribution<Isometry<N, D, R>> for Standard
where
    R: AlgaRotation<Point<N, D>>,
    Standard: Distribution<N> + Distribution<R>,
    DefaultAllocator: Allocator<N, D>,
{
    #[inline]
    fn sample<'a, G: Rng + ?Sized>(&self, rng: &'a mut G) -> Isometry<N, D, R> {
        Isometry::from_parts(rng.gen(), rng.gen())
    }
}

#[cfg(feature = "arbitrary")]
impl<N, D: DimName, R> Arbitrary for Isometry<N, D, R>
where
    N: RealField + Arbitrary + Send,
    R: AlgaRotation<Point<N, D>> + Arbitrary + Send,
    Owned<N, D>: Send,
    DefaultAllocator: Allocator<N, D>,
{
    #[inline]
    fn arbitrary<G: Gen>(rng: &mut G) -> Self {
        Self::from_parts(Arbitrary::arbitrary(rng), Arbitrary::arbitrary(rng))
    }
}

/*
 *
 * Constructors for various static dimensions.
 *
 */

// 2D rotation.
impl<N: RealField> Isometry<N, U2, Rotation2<N>> {
    /// Creates a new 2D isometry from a translation and a rotation angle.
    ///
    /// Its rotational part is represented as a 2x2 rotation matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{Isometry2, Vector2, Point2};
    /// let iso = Isometry2::new(Vector2::new(1.0, 2.0), f32::consts::FRAC_PI_2);
    ///
    /// assert_eq!(iso * Point2::new(3.0, 4.0), Point2::new(-3.0, 5.0));
    /// ```
    #[inline]
    pub fn new(translation: Vector2<N>, angle: N) -> Self {
        Self::from_parts(
            Translation::from(translation),
            Rotation::<N, U2>::new(angle),
        )
    }

    /// Creates a new isometry from the given translation coordinates.
    #[inline]
    pub fn translation(x: N, y: N) -> Self {
        Self::new(Vector2::new(x, y), N::zero())
    }

    /// Creates a new isometry from the given rotation angle.
    #[inline]
    pub fn rotation(angle: N) -> Self {
        Self::new(Vector2::zeros(), angle)
    }
}

impl<N: RealField> Isometry<N, U2, UnitComplex<N>> {
    /// Creates a new 2D isometry from a translation and a rotation angle.
    ///
    /// Its rotational part is represented as an unit complex number.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{IsometryMatrix2, Point2, Vector2};
    /// let iso = IsometryMatrix2::new(Vector2::new(1.0, 2.0), f32::consts::FRAC_PI_2);
    ///
    /// assert_eq!(iso * Point2::new(3.0, 4.0), Point2::new(-3.0, 5.0));
    /// ```
    #[inline]
    pub fn new(translation: Vector2<N>, angle: N) -> Self {
        Self::from_parts(
            Translation::from(translation),
            UnitComplex::from_angle(angle),
        )
    }

    /// Creates a new isometry from the given translation coordinates.
    #[inline]
    pub fn translation(x: N, y: N) -> Self {
        Self::from_parts(Translation2::new(x, y), UnitComplex::identity())
    }

    /// Creates a new isometry from the given rotation angle.
    #[inline]
    pub fn rotation(angle: N) -> Self {
        Self::new(Vector2::zeros(), angle)
    }
}

// 3D rotation.
macro_rules! isometry_construction_impl(
    ($RotId: ident < $($RotParams: ident),*>, $RRDim: ty, $RCDim: ty) => {
        impl<N: RealField> Isometry<N, U3, $RotId<$($RotParams),*>> {
            /// Creates a new isometry from a translation and a rotation axis-angle.
            ///
            /// # Example
            ///
            /// ```
            /// # #[macro_use] extern crate approx;
            /// # use std::f32;
            /// # use nalgebra::{Isometry3, IsometryMatrix3, Point3, Vector3};
            /// let axisangle = Vector3::y() * f32::consts::FRAC_PI_2;
            /// let translation = Vector3::new(1.0, 2.0, 3.0);
            /// // Point and vector being transformed in the tests.
            /// let pt = Point3::new(4.0, 5.0, 6.0);
            /// let vec = Vector3::new(4.0, 5.0, 6.0);
            ///
            /// // Isometry with its rotation part represented as a UnitQuaternion
            /// let iso = Isometry3::new(translation, axisangle);
            /// assert_relative_eq!(iso * pt, Point3::new(7.0, 7.0, -1.0), epsilon = 1.0e-6);
            /// assert_relative_eq!(iso * vec, Vector3::new(6.0, 5.0, -4.0), epsilon = 1.0e-6);
            ///
            /// // Isometry with its rotation part represented as a Rotation3 (a 3x3 rotation matrix).
            /// let iso = IsometryMatrix3::new(translation, axisangle);
            /// assert_relative_eq!(iso * pt, Point3::new(7.0, 7.0, -1.0), epsilon = 1.0e-6);
            /// assert_relative_eq!(iso * vec, Vector3::new(6.0, 5.0, -4.0), epsilon = 1.0e-6);
            /// ```
            #[inline]
            pub fn new(translation: Vector3<N>, axisangle: Vector3<N>) -> Self {
                Self::from_parts(
                    Translation::from(translation),
                    $RotId::<$($RotParams),*>::from_scaled_axis(axisangle))
            }

            /// Creates a new isometry from the given translation coordinates.
            #[inline]
            pub fn translation(x: N, y: N, z: N) -> Self {
                Self::from_parts(Translation3::new(x, y, z), $RotId::identity())
            }

            /// Creates a new isometry from the given rotation angle.
            #[inline]
            pub fn rotation(axisangle: Vector3<N>) -> Self {
                Self::new(Vector3::zeros(), axisangle)
            }

            /// Creates an isometry that corresponds to the local frame of an observer standing at the
            /// point `eye` and looking toward `target`.
            ///
            /// It maps the `z` axis to the view direction `target - eye`and the origin to the `eye`.
            ///
            /// # Arguments
            ///   * eye - The observer position.
            ///   * target - The target position.
            ///   * up - Vertical direction. The only requirement of this parameter is to not be collinear
            ///   to `eye - at`. Non-collinearity is not checked.
            ///
            /// # Example
            ///
            /// ```
            /// # #[macro_use] extern crate approx;
            /// # use std::f32;
            /// # use nalgebra::{Isometry3, IsometryMatrix3, Point3, Vector3};
            /// let eye = Point3::new(1.0, 2.0, 3.0);
            /// let target = Point3::new(2.0, 2.0, 3.0);
            /// let up = Vector3::y();
            ///
            /// // Isometry with its rotation part represented as a UnitQuaternion
            /// let iso = Isometry3::face_towards(&eye, &target, &up);
            /// assert_eq!(iso * Point3::origin(), eye);
            /// assert_relative_eq!(iso * Vector3::z(), Vector3::x());
            ///
            /// // Isometry with its rotation part represented as Rotation3 (a 3x3 rotation matrix).
            /// let iso = IsometryMatrix3::face_towards(&eye, &target, &up);
            /// assert_eq!(iso * Point3::origin(), eye);
            /// assert_relative_eq!(iso * Vector3::z(), Vector3::x());
            /// ```
            #[inline]
            pub fn face_towards(eye:    &Point3<N>,
                                      target: &Point3<N>,
                                      up:     &Vector3<N>)
                                      -> Self {
                Self::from_parts(
                    Translation::from(eye.coords.clone()),
                    $RotId::face_towards(&(target - eye), up))
            }

            /// Deprecated: Use [Isometry::face_towards] instead.
            #[deprecated(note="renamed to `face_towards`")]
            pub fn new_observer_frame(eye:    &Point3<N>,
                                      target: &Point3<N>,
                                      up:     &Vector3<N>)
                                      -> Self {
                Self::face_towards(eye, target, up)
            }

            /// Builds a right-handed look-at view matrix.
            ///
            /// It maps the view direction `target - eye` to the **negative** `z` axis to and the `eye` to the origin.
            /// This conforms to the common notion of right handed camera look-at **view matrix** from
            /// the computer graphics community, i.e. the camera is assumed to look toward its local `-z` axis.
            ///
            /// # Arguments
            ///   * eye - The eye position.
            ///   * target - The target position.
            ///   * up - A vector approximately aligned with required the vertical axis. The only
            ///   requirement of this parameter is to not be collinear to `target - eye`.
            ///
            /// # Example
            ///
            /// ```
            /// # #[macro_use] extern crate approx;
            /// # use std::f32;
            /// # use nalgebra::{Isometry3, IsometryMatrix3, Point3, Vector3};
            /// let eye = Point3::new(1.0, 2.0, 3.0);
            /// let target = Point3::new(2.0, 2.0, 3.0);
            /// let up = Vector3::y();
            ///
            /// // Isometry with its rotation part represented as a UnitQuaternion
            /// let iso = Isometry3::look_at_rh(&eye, &target, &up);
            /// assert_eq!(iso * eye, Point3::origin());
            /// assert_relative_eq!(iso * Vector3::x(), -Vector3::z());
            ///
            /// // Isometry with its rotation part represented as Rotation3 (a 3x3 rotation matrix).
            /// let iso = IsometryMatrix3::look_at_rh(&eye, &target, &up);
            /// assert_eq!(iso * eye, Point3::origin());
            /// assert_relative_eq!(iso * Vector3::x(), -Vector3::z());
            /// ```
            #[inline]
            pub fn look_at_rh(eye:    &Point3<N>,
                              target: &Point3<N>,
                              up:     &Vector3<N>)
                              -> Self {
                let rotation = $RotId::look_at_rh(&(target - eye), up);
                let trans    = &rotation * (-eye);

                Self::from_parts(Translation::from(trans.coords), rotation)
            }

            /// Builds a left-handed look-at view matrix.
            ///
            /// It maps the view direction `target - eye` to the **positive** `z` axis and the `eye` to the origin.
            /// This conforms to the common notion of right handed camera look-at **view matrix** from
            /// the computer graphics community, i.e. the camera is assumed to look toward its local `z` axis.
            ///
            /// # Arguments
            ///   * eye - The eye position.
            ///   * target - The target position.
            ///   * up - A vector approximately aligned with required the vertical axis. The only
            ///   requirement of this parameter is to not be collinear to `target - eye`.
            ///
            /// # Example
            ///
            /// ```
            /// # #[macro_use] extern crate approx;
            /// # use std::f32;
            /// # use nalgebra::{Isometry3, IsometryMatrix3, Point3, Vector3};
            /// let eye = Point3::new(1.0, 2.0, 3.0);
            /// let target = Point3::new(2.0, 2.0, 3.0);
            /// let up = Vector3::y();
            ///
            /// // Isometry with its rotation part represented as a UnitQuaternion
            /// let iso = Isometry3::look_at_lh(&eye, &target, &up);
            /// assert_eq!(iso * eye, Point3::origin());
            /// assert_relative_eq!(iso * Vector3::x(), Vector3::z());
            ///
            /// // Isometry with its rotation part represented as Rotation3 (a 3x3 rotation matrix).
            /// let iso = IsometryMatrix3::look_at_lh(&eye, &target, &up);
            /// assert_eq!(iso * eye, Point3::origin());
            /// assert_relative_eq!(iso * Vector3::x(), Vector3::z());
            /// ```
            #[inline]
            pub fn look_at_lh(eye:    &Point3<N>,
                              target: &Point3<N>,
                              up:     &Vector3<N>)
                              -> Self {
                let rotation = $RotId::look_at_lh(&(target - eye), up);
                let trans    = &rotation * (-eye);

                Self::from_parts(Translation::from(trans.coords), rotation)
            }
        }
    }
);

isometry_construction_impl!(Rotation3<N>, U3, U3);
isometry_construction_impl!(UnitQuaternion<N>, U4, U1);
