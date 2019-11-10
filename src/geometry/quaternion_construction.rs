#[cfg(feature = "arbitrary")]
use crate::U4;
#[cfg(feature = "arbitrary")]
use crate::base::storage::Owned;
#[cfg(feature = "arbitrary")]
use quickcheck::{Arbitrary, Gen};

use num::{One, Zero};
use rand::distributions::{Distribution, OpenClosed01, Standard};
use rand::Rng;

use alga::general::RealField;

use crate::U3;
use crate::base::storage::Storage;
use crate::base::{Unit, Vector, Vector3, Vector4, Matrix3, Matrix4};

use crate::geometry::{Quaternion, Rotation3, UnitQuaternion};

impl<N: RealField> Quaternion<N> {
    /// Creates a quaternion from a 4D vector. The quaternion scalar part corresponds to the `w`
    /// vector component.
    #[inline]
    #[deprecated(note = "Use `::from` instead.")]
    pub fn from_vector(vector: Vector4<N>) -> Self {
        Self { coords: vector }
    }

    /// Creates a new quaternion from its individual components. Note that the arguments order does
    /// **not** follow the storage order.
    ///
    /// The storage order is `[ i, j, k, w ]` while the arguments for this functions are in the
    /// order `(w, i, j, k)`.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{Quaternion, Vector4};
    /// let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// assert!(q.i == 2.0 && q.j == 3.0 && q.k == 4.0 && q.w == 1.0);
    /// assert_eq!(*q.as_vector(), Vector4::new(2.0, 3.0, 4.0, 1.0));
    /// ```
    #[inline]
    pub fn new(w: N, i: N, j: N, k: N) -> Self {
        Self::from(Vector4::new(i, j, k, w))
    }

    /// Constructs a pure quaternion.
    #[inline]
    pub fn from_imag(vector: Vector3<N>) -> Self {
        Self::from_parts(N::zero(), vector)
    }

    /// Creates a new quaternion from its scalar and vector parts. Note that the arguments order does
    /// **not** follow the storage order.
    ///
    /// The storage order is [ vector, scalar ].
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{Quaternion, Vector3, Vector4};
    /// let w = 1.0;
    /// let ijk = Vector3::new(2.0, 3.0, 4.0);
    /// let q = Quaternion::from_parts(w, ijk);
    /// assert!(q.i == 2.0 && q.j == 3.0 && q.k == 4.0 && q.w == 1.0);
    /// assert_eq!(*q.as_vector(), Vector4::new(2.0, 3.0, 4.0, 1.0));
    /// ```
    #[inline]
    // FIXME: take a reference to `vector`?
    pub fn from_parts<SB>(scalar: N, vector: Vector<N, U3, SB>) -> Self
    where SB: Storage<N, U3> {
        Self::new(scalar, vector[0], vector[1], vector[2])
    }

    /// Constructs a real quaternion.
    #[inline]
    pub fn from_real(r: N) -> Self {
        Self::from_parts(r, Vector3::zero())
    }

    /// Creates a new quaternion from its polar decomposition.
    ///
    /// Note that `axis` is assumed to be a unit vector.
    // FIXME: take a reference to `axis`?
    pub fn from_polar_decomposition<SB>(scale: N, theta: N, axis: Unit<Vector<N, U3, SB>>) -> Self
    where SB: Storage<N, U3> {
        let rot = UnitQuaternion::<N>::from_axis_angle(&axis, theta * crate::convert(2.0f64));

        rot.into_inner() * scale
    }

    /// The quaternion multiplicative identity.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::Quaternion;
    /// let q = Quaternion::identity();
    /// let q2 = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    ///
    /// assert_eq!(q * q2, q2);
    /// assert_eq!(q2 * q, q2);
    /// ```
    #[inline]
    pub fn identity() -> Self {
        Self::from_real(N::one())
    }
}

impl<N: RealField> One for Quaternion<N> {
    #[inline]
    fn one() -> Self {
        Self::identity()
    }
}

impl<N: RealField> Zero for Quaternion<N> {
    #[inline]
    fn zero() -> Self {
        Self::from(Vector4::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.coords.is_zero()
    }
}

impl<N: RealField> Distribution<Quaternion<N>> for Standard
where Standard: Distribution<N>
{
    #[inline]
    fn sample<'a, R: Rng + ?Sized>(&self, rng: &'a mut R) -> Quaternion<N> {
        Quaternion::new(rng.gen(), rng.gen(), rng.gen(), rng.gen())
    }
}

#[cfg(feature = "arbitrary")]
impl<N: RealField + Arbitrary> Arbitrary for Quaternion<N>
where Owned<N, U4>: Send
{
    #[inline]
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        Self::new(
            N::arbitrary(g),
            N::arbitrary(g),
            N::arbitrary(g),
            N::arbitrary(g),
        )
    }
}

impl<N: RealField> UnitQuaternion<N> {
    /// The rotation identity.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{UnitQuaternion, Vector3, Point3};
    /// let q = UnitQuaternion::identity();
    /// let q2 = UnitQuaternion::new(Vector3::new(1.0, 2.0, 3.0));
    /// let v = Vector3::new_random();
    /// let p = Point3::from(v);
    ///
    /// assert_eq!(q * q2, q2);
    /// assert_eq!(q2 * q, q2);
    /// assert_eq!(q * v, v);
    /// assert_eq!(q * p, p);
    /// ```
    #[inline]
    pub fn identity() -> Self {
        Self::new_unchecked(Quaternion::identity())
    }

    /// Creates a new quaternion from a unit vector (the rotation axis) and an angle
    /// (the rotation angle).
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{UnitQuaternion, Point3, Vector3};
    /// let axis = Vector3::y_axis();
    /// let angle = f32::consts::FRAC_PI_2;
    /// // Point and vector being transformed in the tests.
    /// let pt = Point3::new(4.0, 5.0, 6.0);
    /// let vec = Vector3::new(4.0, 5.0, 6.0);
    /// let q = UnitQuaternion::from_axis_angle(&axis, angle);
    ///
    /// assert_eq!(q.axis().unwrap(), axis);
    /// assert_eq!(q.angle(), angle);
    /// assert_relative_eq!(q * pt, Point3::new(6.0, 5.0, -4.0), epsilon = 1.0e-6);
    /// assert_relative_eq!(q * vec, Vector3::new(6.0, 5.0, -4.0), epsilon = 1.0e-6);
    ///
    /// // A zero vector yields an identity.
    /// assert_eq!(UnitQuaternion::from_scaled_axis(Vector3::<f32>::zeros()), UnitQuaternion::identity());
    /// ```
    #[inline]
    pub fn from_axis_angle<SB>(axis: &Unit<Vector<N, U3, SB>>, angle: N) -> Self
    where SB: Storage<N, U3> {
        let (sang, cang) = (angle / crate::convert(2.0f64)).sin_cos();

        let q = Quaternion::from_parts(cang, axis.as_ref() * sang);
        Self::new_unchecked(q)
    }

    /// Creates a new unit quaternion from a quaternion.
    ///
    /// The input quaternion will be normalized.
    #[inline]
    pub fn from_quaternion(q: Quaternion<N>) -> Self {
        Self::new_normalize(q)
    }

    /// Creates a new unit quaternion from Euler angles.
    ///
    /// The primitive rotations are applied in order: 1 roll − 2 pitch − 3 yaw.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::UnitQuaternion;
    /// let rot = UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);
    /// let euler = rot.euler_angles();
    /// assert_relative_eq!(euler.0, 0.1, epsilon = 1.0e-6);
    /// assert_relative_eq!(euler.1, 0.2, epsilon = 1.0e-6);
    /// assert_relative_eq!(euler.2, 0.3, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn from_euler_angles(roll: N, pitch: N, yaw: N) -> Self {
        let (sr, cr) = (roll * crate::convert(0.5f64)).sin_cos();
        let (sp, cp) = (pitch * crate::convert(0.5f64)).sin_cos();
        let (sy, cy) = (yaw * crate::convert(0.5f64)).sin_cos();

        let q = Quaternion::new(
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        );

        Self::new_unchecked(q)
    }

    /// Builds an unit quaternion from a rotation matrix.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Rotation3, UnitQuaternion, Vector3};
    /// let axis = Vector3::y_axis();
    /// let angle = 0.1;
    /// let rot = Rotation3::from_axis_angle(&axis, angle);
    /// let q = UnitQuaternion::from_rotation_matrix(&rot);
    /// assert_relative_eq!(q.to_rotation_matrix(), rot, epsilon = 1.0e-6);
    /// assert_relative_eq!(q.axis().unwrap(), rot.axis().unwrap(), epsilon = 1.0e-6);
    /// assert_relative_eq!(q.angle(), rot.angle(), epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn from_rotation_matrix(rotmat: &Rotation3<N>) -> Self {
        // Robust matrix to quaternion transformation.
        // See https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion
        let tr = rotmat[(0, 0)] + rotmat[(1, 1)] + rotmat[(2, 2)];
        let res;

        let _0_25: N = crate::convert(0.25);

        if tr > N::zero() {
            let denom = (tr + N::one()).sqrt() * crate::convert(2.0);
            res = Quaternion::new(
                _0_25 * denom,
                (rotmat[(2, 1)] - rotmat[(1, 2)]) / denom,
                (rotmat[(0, 2)] - rotmat[(2, 0)]) / denom,
                (rotmat[(1, 0)] - rotmat[(0, 1)]) / denom,
            );
        } else if rotmat[(0, 0)] > rotmat[(1, 1)] && rotmat[(0, 0)] > rotmat[(2, 2)] {
            let denom = (N::one() + rotmat[(0, 0)] - rotmat[(1, 1)] - rotmat[(2, 2)]).sqrt()
                * crate::convert(2.0);
            res = Quaternion::new(
                (rotmat[(2, 1)] - rotmat[(1, 2)]) / denom,
                _0_25 * denom,
                (rotmat[(0, 1)] + rotmat[(1, 0)]) / denom,
                (rotmat[(0, 2)] + rotmat[(2, 0)]) / denom,
            );
        } else if rotmat[(1, 1)] > rotmat[(2, 2)] {
            let denom = (N::one() + rotmat[(1, 1)] - rotmat[(0, 0)] - rotmat[(2, 2)]).sqrt()
                * crate::convert(2.0);
            res = Quaternion::new(
                (rotmat[(0, 2)] - rotmat[(2, 0)]) / denom,
                (rotmat[(0, 1)] + rotmat[(1, 0)]) / denom,
                _0_25 * denom,
                (rotmat[(1, 2)] + rotmat[(2, 1)]) / denom,
            );
        } else {
            let denom = (N::one() + rotmat[(2, 2)] - rotmat[(0, 0)] - rotmat[(1, 1)]).sqrt()
                * crate::convert(2.0);
            res = Quaternion::new(
                (rotmat[(1, 0)] - rotmat[(0, 1)]) / denom,
                (rotmat[(0, 2)] + rotmat[(2, 0)]) / denom,
                (rotmat[(1, 2)] + rotmat[(2, 1)]) / denom,
                _0_25 * denom,
            );
        }

        Self::new_unchecked(res)
    }

    /// Builds an unit quaternion by extracting the rotation part of the given transformation `m`.
    ///
    /// This is an iterative method. See `.from_matrix_eps` to provide mover
    /// convergence parameters and starting solution.
    /// This implements "A Robust Method to Extract the Rotational Part of Deformations" by Müller et al.
    pub fn from_matrix(m: &Matrix3<N>) -> Self {
        Rotation3::from_matrix(m).into()
    }

    /// Builds an unit quaternion by extracting the rotation part of the given transformation `m`.
    ///
    /// This implements "A Robust Method to Extract the Rotational Part of Deformations" by Müller et al.
    ///
    /// # Parameters
    ///
    /// * `m`: the matrix from which the rotational part is to be extracted.
    /// * `eps`: the angular errors tolerated between the current rotation and the optimal one.
    /// * `max_iter`: the maximum number of iterations. Loops indefinitely until convergence if set to `0`.
    /// * `guess`: an estimate of the solution. Convergence will be significantly faster if an initial solution close
    ///           to the actual solution is provided. Can be set to `UnitQuaternion::identity()` if no other
    ///           guesses come to mind.
    pub fn from_matrix_eps(m: &Matrix3<N>, eps: N, max_iter: usize, guess: Self) -> Self {
        let guess = Rotation3::from(guess);
        Rotation3::from_matrix_eps(m, eps, max_iter, guess).into()
    }

    /// The unit quaternion needed to make `a` and `b` be collinear and point toward the same
    /// direction.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Vector3, UnitQuaternion};
    /// let a = Vector3::new(1.0, 2.0, 3.0);
    /// let b = Vector3::new(3.0, 1.0, 2.0);
    /// let q = UnitQuaternion::rotation_between(&a, &b).unwrap();
    /// assert_relative_eq!(q * a, b);
    /// assert_relative_eq!(q.inverse() * b, a);
    /// ```
    #[inline]
    pub fn rotation_between<SB, SC>(a: &Vector<N, U3, SB>, b: &Vector<N, U3, SC>) -> Option<Self>
    where
        SB: Storage<N, U3>,
        SC: Storage<N, U3>,
    {
        Self::scaled_rotation_between(a, b, N::one())
    }

    /// The smallest rotation needed to make `a` and `b` collinear and point toward the same
    /// direction, raised to the power `s`.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Vector3, UnitQuaternion};
    /// let a = Vector3::new(1.0, 2.0, 3.0);
    /// let b = Vector3::new(3.0, 1.0, 2.0);
    /// let q2 = UnitQuaternion::scaled_rotation_between(&a, &b, 0.2).unwrap();
    /// let q5 = UnitQuaternion::scaled_rotation_between(&a, &b, 0.5).unwrap();
    /// assert_relative_eq!(q2 * q2 * q2 * q2 * q2 * a, b, epsilon = 1.0e-6);
    /// assert_relative_eq!(q5 * q5 * a, b, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn scaled_rotation_between<SB, SC>(
        a: &Vector<N, U3, SB>,
        b: &Vector<N, U3, SC>,
        s: N,
    ) -> Option<Self>
    where
        SB: Storage<N, U3>,
        SC: Storage<N, U3>,
    {
        // FIXME: code duplication with Rotation.
        if let (Some(na), Some(nb)) = (
            Unit::try_new(a.clone_owned(), N::zero()),
            Unit::try_new(b.clone_owned(), N::zero()),
        ) {
            Self::scaled_rotation_between_axis(&na, &nb, s)
        } else {
            Some(Self::identity())
        }
    }

    /// The unit quaternion needed to make `a` and `b` be collinear and point toward the same
    /// direction.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Unit, Vector3, UnitQuaternion};
    /// let a = Unit::new_normalize(Vector3::new(1.0, 2.0, 3.0));
    /// let b = Unit::new_normalize(Vector3::new(3.0, 1.0, 2.0));
    /// let q = UnitQuaternion::rotation_between(&a, &b).unwrap();
    /// assert_relative_eq!(q * a, b);
    /// assert_relative_eq!(q.inverse() * b, a);
    /// ```
    #[inline]
    pub fn rotation_between_axis<SB, SC>(
        a: &Unit<Vector<N, U3, SB>>,
        b: &Unit<Vector<N, U3, SC>>,
    ) -> Option<Self>
    where
        SB: Storage<N, U3>,
        SC: Storage<N, U3>,
    {
        Self::scaled_rotation_between_axis(a, b, N::one())
    }

    /// The smallest rotation needed to make `a` and `b` collinear and point toward the same
    /// direction, raised to the power `s`.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Unit, Vector3, UnitQuaternion};
    /// let a = Unit::new_normalize(Vector3::new(1.0, 2.0, 3.0));
    /// let b = Unit::new_normalize(Vector3::new(3.0, 1.0, 2.0));
    /// let q2 = UnitQuaternion::scaled_rotation_between(&a, &b, 0.2).unwrap();
    /// let q5 = UnitQuaternion::scaled_rotation_between(&a, &b, 0.5).unwrap();
    /// assert_relative_eq!(q2 * q2 * q2 * q2 * q2 * a, b, epsilon = 1.0e-6);
    /// assert_relative_eq!(q5 * q5 * a, b, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn scaled_rotation_between_axis<SB, SC>(
        na: &Unit<Vector<N, U3, SB>>,
        nb: &Unit<Vector<N, U3, SC>>,
        s: N,
    ) -> Option<Self>
    where
        SB: Storage<N, U3>,
        SC: Storage<N, U3>,
    {
        // FIXME: code duplication with Rotation.
        let c = na.cross(&nb);

        if let Some(axis) = Unit::try_new(c, N::default_epsilon()) {
            let cos = na.dot(&nb);

            // The cosinus may be out of [-1, 1] because of inaccuracies.
            if cos <= -N::one() {
                return None;
            } else if cos >= N::one() {
                return Some(Self::identity());
            } else {
                return Some(Self::from_axis_angle(&axis, cos.acos() * s));
            }
        } else if na.dot(&nb) < N::zero() {
            // PI
            //
            // The rotation axis is undefined but the angle not zero. This is not a
            // simple rotation.
            return None;
        } else {
            // Zero
            Some(Self::identity())
        }
    }

    /// Creates an unit quaternion that corresponds to the local frame of an observer standing at the
    /// origin and looking toward `dir`.
    ///
    /// It maps the `z` axis to the direction `dir`.
    ///
    /// # Arguments
    ///   * dir - The look direction. It does not need to be normalized.
    ///   * up - The vertical direction. It does not need to be normalized.
    ///   The only requirement of this parameter is to not be collinear to `dir`. Non-collinearity
    ///   is not checked.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{UnitQuaternion, Vector3};
    /// let dir = Vector3::new(1.0, 2.0, 3.0);
    /// let up = Vector3::y();
    ///
    /// let q = UnitQuaternion::face_towards(&dir, &up);
    /// assert_relative_eq!(q * Vector3::z(), dir.normalize());
    /// ```
    #[inline]
    pub fn face_towards<SB, SC>(dir: &Vector<N, U3, SB>, up: &Vector<N, U3, SC>) -> Self
    where
        SB: Storage<N, U3>,
        SC: Storage<N, U3>,
    {
        Self::from_rotation_matrix(&Rotation3::face_towards(dir, up))
    }

    /// Deprecated: Use [UnitQuaternion::face_towards] instead.
    #[deprecated(note="renamed to `face_towards`")]
    pub fn new_observer_frames<SB, SC>(dir: &Vector<N, U3, SB>, up: &Vector<N, U3, SC>) -> Self
    where
        SB: Storage<N, U3>,
        SC: Storage<N, U3>,
    {
        Self::face_towards(dir, up)
    }

    /// Builds a right-handed look-at view matrix without translation.
    ///
    /// It maps the view direction `dir` to the **negative** `z` axis.
    /// This conforms to the common notion of right handed look-at matrix from the computer
    /// graphics community.
    ///
    /// # Arguments
    ///   * dir − The view direction. It does not need to be normalized.
    ///   * up - A vector approximately aligned with required the vertical axis. It does not need
    ///   to be normalized. The only requirement of this parameter is to not be collinear to `dir`.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{UnitQuaternion, Vector3};
    /// let dir = Vector3::new(1.0, 2.0, 3.0);
    /// let up = Vector3::y();
    ///
    /// let q = UnitQuaternion::look_at_rh(&dir, &up);
    /// assert_relative_eq!(q * dir.normalize(), -Vector3::z());
    /// ```
    #[inline]
    pub fn look_at_rh<SB, SC>(dir: &Vector<N, U3, SB>, up: &Vector<N, U3, SC>) -> Self
    where
        SB: Storage<N, U3>,
        SC: Storage<N, U3>,
    {
        Self::face_towards(&-dir, up).inverse()
    }

    /// Builds a left-handed look-at view matrix without translation.
    ///
    /// It maps the view direction `dir` to the **positive** `z` axis.
    /// This conforms to the common notion of left handed look-at matrix from the computer
    /// graphics community.
    ///
    /// # Arguments
    ///   * dir − The view direction. It does not need to be normalized.
    ///   * up - A vector approximately aligned with required the vertical axis. The only
    ///   requirement of this parameter is to not be collinear to `dir`.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{UnitQuaternion, Vector3};
    /// let dir = Vector3::new(1.0, 2.0, 3.0);
    /// let up = Vector3::y();
    ///
    /// let q = UnitQuaternion::look_at_lh(&dir, &up);
    /// assert_relative_eq!(q * dir.normalize(), Vector3::z());
    /// ```
    #[inline]
    pub fn look_at_lh<SB, SC>(dir: &Vector<N, U3, SB>, up: &Vector<N, U3, SC>) -> Self
    where
        SB: Storage<N, U3>,
        SC: Storage<N, U3>,
    {
        Self::face_towards(dir, up).inverse()
    }

    /// Creates a new unit quaternion rotation from a rotation axis scaled by the rotation angle.
    ///
    /// If `axisangle` has a magnitude smaller than `N::default_epsilon()`, this returns the identity rotation.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{UnitQuaternion, Point3, Vector3};
    /// let axisangle = Vector3::y() * f32::consts::FRAC_PI_2;
    /// // Point and vector being transformed in the tests.
    /// let pt = Point3::new(4.0, 5.0, 6.0);
    /// let vec = Vector3::new(4.0, 5.0, 6.0);
    /// let q = UnitQuaternion::new(axisangle);
    ///
    /// assert_relative_eq!(q * pt, Point3::new(6.0, 5.0, -4.0), epsilon = 1.0e-6);
    /// assert_relative_eq!(q * vec, Vector3::new(6.0, 5.0, -4.0), epsilon = 1.0e-6);
    ///
    /// // A zero vector yields an identity.
    /// assert_eq!(UnitQuaternion::new(Vector3::<f32>::zeros()), UnitQuaternion::identity());
    /// ```
    #[inline]
    pub fn new<SB>(axisangle: Vector<N, U3, SB>) -> Self
    where SB: Storage<N, U3> {
        let two: N = crate::convert(2.0f64);
        let q = Quaternion::<N>::from_imag(axisangle / two).exp();
        Self::new_unchecked(q)
    }

    /// Creates a new unit quaternion rotation from a rotation axis scaled by the rotation angle.
    ///
    /// If `axisangle` has a magnitude smaller than `eps`, this returns the identity rotation.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{UnitQuaternion, Point3, Vector3};
    /// let axisangle = Vector3::y() * f32::consts::FRAC_PI_2;
    /// // Point and vector being transformed in the tests.
    /// let pt = Point3::new(4.0, 5.0, 6.0);
    /// let vec = Vector3::new(4.0, 5.0, 6.0);
    /// let q = UnitQuaternion::new_eps(axisangle, 1.0e-6);
    ///
    /// assert_relative_eq!(q * pt, Point3::new(6.0, 5.0, -4.0), epsilon = 1.0e-6);
    /// assert_relative_eq!(q * vec, Vector3::new(6.0, 5.0, -4.0), epsilon = 1.0e-6);
    ///
    /// // An almost zero vector yields an identity.
    /// assert_eq!(UnitQuaternion::new_eps(Vector3::new(1.0e-8, 1.0e-9, 1.0e-7), 1.0e-6), UnitQuaternion::identity());
    /// ```
    #[inline]
    pub fn new_eps<SB>(axisangle: Vector<N, U3, SB>, eps: N) -> Self
    where SB: Storage<N, U3> {
        let two: N = crate::convert(2.0f64);
        let q = Quaternion::<N>::from_imag(axisangle / two).exp_eps(eps);
        Self::new_unchecked(q)
    }

    /// Creates a new unit quaternion rotation from a rotation axis scaled by the rotation angle.
    ///
    /// If `axisangle` has a magnitude smaller than `N::default_epsilon()`, this returns the identity rotation.
    /// Same as `Self::new(axisangle)`.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{UnitQuaternion, Point3, Vector3};
    /// let axisangle = Vector3::y() * f32::consts::FRAC_PI_2;
    /// // Point and vector being transformed in the tests.
    /// let pt = Point3::new(4.0, 5.0, 6.0);
    /// let vec = Vector3::new(4.0, 5.0, 6.0);
    /// let q = UnitQuaternion::from_scaled_axis(axisangle);
    ///
    /// assert_relative_eq!(q * pt, Point3::new(6.0, 5.0, -4.0), epsilon = 1.0e-6);
    /// assert_relative_eq!(q * vec, Vector3::new(6.0, 5.0, -4.0), epsilon = 1.0e-6);
    ///
    /// // A zero vector yields an identity.
    /// assert_eq!(UnitQuaternion::from_scaled_axis(Vector3::<f32>::zeros()), UnitQuaternion::identity());
    /// ```
    #[inline]
    pub fn from_scaled_axis<SB>(axisangle: Vector<N, U3, SB>) -> Self
    where SB: Storage<N, U3> {
        Self::new(axisangle)
    }

    /// Creates a new unit quaternion rotation from a rotation axis scaled by the rotation angle.
    ///
    /// If `axisangle` has a magnitude smaller than `eps`, this returns the identity rotation.
    /// Same as `Self::new_eps(axisangle, eps)`.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{UnitQuaternion, Point3, Vector3};
    /// let axisangle = Vector3::y() * f32::consts::FRAC_PI_2;
    /// // Point and vector being transformed in the tests.
    /// let pt = Point3::new(4.0, 5.0, 6.0);
    /// let vec = Vector3::new(4.0, 5.0, 6.0);
    /// let q = UnitQuaternion::from_scaled_axis_eps(axisangle, 1.0e-6);
    ///
    /// assert_relative_eq!(q * pt, Point3::new(6.0, 5.0, -4.0), epsilon = 1.0e-6);
    /// assert_relative_eq!(q * vec, Vector3::new(6.0, 5.0, -4.0), epsilon = 1.0e-6);
    ///
    /// // An almost zero vector yields an identity.
    /// assert_eq!(UnitQuaternion::from_scaled_axis_eps(Vector3::new(1.0e-8, 1.0e-9, 1.0e-7), 1.0e-6), UnitQuaternion::identity());
    /// ```
    #[inline]
    pub fn from_scaled_axis_eps<SB>(axisangle: Vector<N, U3, SB>, eps: N) -> Self
    where SB: Storage<N, U3> {
        Self::new_eps(axisangle, eps)
    }

    /// Create the mean unit quaternion from a data structure implementing IntoIterator
    /// returning unit quaternions.
    ///
    /// The method will panic if the iterator does not return any quaternions.
    ///
    /// Algorithm from: Oshman, Yaakov, and Avishy Carmi. "Attitude estimation from vector
    /// observations using a genetic-algorithm-embedded quaternion particle filter." Journal of
    /// Guidance, Control, and Dynamics 29.4 (2006): 879-891.
    /// 
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{UnitQuaternion};
    /// let q1 = UnitQuaternion::from_euler_angles(0.0, 0.0, 0.0);
    /// let q2 = UnitQuaternion::from_euler_angles(-0.1, 0.0, 0.0);
    /// let q3 = UnitQuaternion::from_euler_angles(0.1, 0.0, 0.0);
    ///
    /// let quat_vec = vec![q1, q2, q3];
    /// let q_mean = UnitQuaternion::mean_of(quat_vec);
    ///
    /// let euler_angles_mean = q_mean.euler_angles();
    /// assert_relative_eq!(euler_angles_mean.0, 0.0, epsilon = 1.0e-7)
    /// ```
    #[inline]
    pub fn mean_of(unit_quaternions: impl IntoIterator<Item = Self>) -> Self {
        let quaternions_matrix: Matrix4<N> = unit_quaternions
            .into_iter()
            .map(|q| q.as_vector() * q.as_vector().transpose())
            .sum();

        assert!(!quaternions_matrix.is_zero());
        
        let eigen_matrix = quaternions_matrix
            .try_symmetric_eigen(N::RealField::default_epsilon(), 10)
            .expect("Quaternions matrix could not be diagonalized. This behavior should not be possible.");

        let max_eigenvalue_index = eigen_matrix
            .eigenvalues
            .iter()
            .position(|v| *v == eigen_matrix.eigenvalues.max())
            .unwrap();

        let max_eigenvector = eigen_matrix.eigenvectors.column(max_eigenvalue_index);
        UnitQuaternion::from_quaternion(Quaternion::new(
            max_eigenvector[0],
            max_eigenvector[1],
            max_eigenvector[2],
            max_eigenvector[3],
        ))
    }
}

impl<N: RealField> One for UnitQuaternion<N> {
    #[inline]
    fn one() -> Self {
        Self::identity()
    }
}

impl<N: RealField> Distribution<UnitQuaternion<N>> for Standard
where OpenClosed01: Distribution<N>
{
    /// Generate a uniformly distributed random rotation quaternion.
    #[inline]
    fn sample<'a, R: Rng + ?Sized>(&self, rng: &'a mut R) -> UnitQuaternion<N> {
        // Ken Shoemake's Subgroup Algorithm
        // Uniform random rotations.
        // In D. Kirk, editor, Graphics Gems III, pages 124-132. Academic, New York, 1992.
        let x0 = rng.sample(OpenClosed01);
        let x1 = rng.sample(OpenClosed01);
        let x2 = rng.sample(OpenClosed01);
        let theta1 = N::two_pi() * x1;
        let theta2 = N::two_pi() * x2;
        let s1 = theta1.sin();
        let c1 = theta1.cos();
        let s2 = theta2.sin();
        let c2 = theta2.cos();
        let r1 = (N::one() - x0).sqrt();
        let r2 = x0.sqrt();
        Unit::new_unchecked(Quaternion::new(s1 * r1, c1 * r1, s2 * r2, c2 * r2))
    }
}

#[cfg(feature = "arbitrary")]
impl<N: RealField + Arbitrary> Arbitrary for UnitQuaternion<N>
where
    Owned<N, U4>: Send,
    Owned<N, U3>: Send,
{
    #[inline]
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        let axisangle = Vector3::arbitrary(g);
        Self::from_scaled_axis(axisangle)
    }
}

#[cfg(test)]
mod tests {
    extern crate rand_xorshift;
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn random_unit_quats_are_unit() {
        let mut rng = rand_xorshift::XorShiftRng::from_seed([0xAB; 16]);
        for _ in 0..1000 {
            let x = rng.gen::<UnitQuaternion<f32>>();
            assert!(relative_eq!(x.into_inner().norm(), 1.0))
        }
    }
}
