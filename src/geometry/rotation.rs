use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use num::{One, Zero};
use std::fmt;
use std::hash;
#[cfg(feature = "abomonation-serialize")]
use std::io::{Result as IOResult, Write};

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[cfg(feature = "serde-serialize")]
use crate::base::storage::Owned;

#[cfg(feature = "abomonation-serialize")]
use abomonation::Abomonation;

use alga::general::RealField;

use crate::base::allocator::Allocator;
use crate::base::dimension::{DimName, DimNameAdd, DimNameSum, U1};
use crate::base::{DefaultAllocator, MatrixN, Scalar, VectorN};
use crate::geometry::Point;

/// A rotation matrix.
#[repr(C)]
#[derive(Debug)]
pub struct Rotation<N: Scalar, D: DimName>
where DefaultAllocator: Allocator<N, D, D>
{
    matrix: MatrixN<N, D>,
}

impl<N: Scalar + hash::Hash, D: DimName + hash::Hash> hash::Hash for Rotation<N, D>
where
    DefaultAllocator: Allocator<N, D, D>,
    <DefaultAllocator as Allocator<N, D, D>>::Buffer: hash::Hash,
{
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.matrix.hash(state)
    }
}

impl<N: Scalar + Copy, D: DimName> Copy for Rotation<N, D>
where
    DefaultAllocator: Allocator<N, D, D>,
    <DefaultAllocator as Allocator<N, D, D>>::Buffer: Copy,
{
}

impl<N: Scalar, D: DimName> Clone for Rotation<N, D>
where
    DefaultAllocator: Allocator<N, D, D>,
    <DefaultAllocator as Allocator<N, D, D>>::Buffer: Clone,
{
    #[inline]
    fn clone(&self) -> Self {
        Self::from_matrix_unchecked(self.matrix.clone())
    }
}

#[cfg(feature = "abomonation-serialize")]
impl<N, D> Abomonation for Rotation<N, D>
where
    N: Scalar,
    D: DimName,
    MatrixN<N, D>: Abomonation,
    DefaultAllocator: Allocator<N, D, D>,
{
    unsafe fn entomb<W: Write>(&self, writer: &mut W) -> IOResult<()> {
        self.matrix.entomb(writer)
    }

    fn extent(&self) -> usize {
        self.matrix.extent()
    }

    unsafe fn exhume<'a, 'b>(&'a mut self, bytes: &'b mut [u8]) -> Option<&'b mut [u8]> {
        self.matrix.exhume(bytes)
    }
}

#[cfg(feature = "serde-serialize")]
impl<N: Scalar, D: DimName> Serialize for Rotation<N, D>
where
    DefaultAllocator: Allocator<N, D, D>,
    Owned<N, D, D>: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: Serializer {
        self.matrix.serialize(serializer)
    }
}

#[cfg(feature = "serde-serialize")]
impl<'a, N: Scalar, D: DimName> Deserialize<'a> for Rotation<N, D>
where
    DefaultAllocator: Allocator<N, D, D>,
    Owned<N, D, D>: Deserialize<'a>,
{
    fn deserialize<Des>(deserializer: Des) -> Result<Self, Des::Error>
    where Des: Deserializer<'a> {
        let matrix = MatrixN::<N, D>::deserialize(deserializer)?;

        Ok(Self::from_matrix_unchecked(matrix))
    }
}

impl<N: Scalar, D: DimName> Rotation<N, D>
where DefaultAllocator: Allocator<N, D, D>
{
    /// A reference to the underlying matrix representation of this rotation.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{Rotation2, Rotation3, Vector3, Matrix2, Matrix3};
    /// # use std::f32;
    /// let rot = Rotation3::from_axis_angle(&Vector3::z_axis(), f32::consts::FRAC_PI_6);
    /// let expected = Matrix3::new(0.8660254, -0.5,      0.0,
    ///                             0.5,       0.8660254, 0.0,
    ///                             0.0,       0.0,       1.0);
    /// assert_eq!(*rot.matrix(), expected);
    ///
    ///
    /// let rot = Rotation2::new(f32::consts::FRAC_PI_6);
    /// let expected = Matrix2::new(0.8660254, -0.5,
    ///                             0.5,       0.8660254);
    /// assert_eq!(*rot.matrix(), expected);
    /// ```
    #[inline]
    pub fn matrix(&self) -> &MatrixN<N, D> {
        &self.matrix
    }

    /// A mutable reference to the underlying matrix representation of this rotation.
    #[inline]
    #[deprecated(note = "Use `.matrix_mut_unchecked()` instead.")]
    pub unsafe fn matrix_mut(&mut self) -> &mut MatrixN<N, D> {
        &mut self.matrix
    }

    /// A mutable reference to the underlying matrix representation of this rotation.
    ///
    /// This is suffixed by "_unchecked" because this allows the user to replace the matrix by another one that is
    /// non-square, non-inversible, or non-orthonormal. If one of those properties is broken,
    /// subsequent method calls may be UB.
    #[inline]
    pub fn matrix_mut_unchecked(&mut self) -> &mut MatrixN<N, D> {
        &mut self.matrix
    }

    /// Unwraps the underlying matrix.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{Rotation2, Rotation3, Vector3, Matrix2, Matrix3};
    /// # use std::f32;
    /// let rot = Rotation3::from_axis_angle(&Vector3::z_axis(), f32::consts::FRAC_PI_6);
    /// let mat = rot.into_inner();
    /// let expected = Matrix3::new(0.8660254, -0.5,      0.0,
    ///                             0.5,       0.8660254, 0.0,
    ///                             0.0,       0.0,       1.0);
    /// assert_eq!(mat, expected);
    ///
    ///
    /// let rot = Rotation2::new(f32::consts::FRAC_PI_6);
    /// let mat = rot.into_inner();
    /// let expected = Matrix2::new(0.8660254, -0.5,
    ///                             0.5,       0.8660254);
    /// assert_eq!(mat, expected);
    /// ```
    #[inline]
    pub fn into_inner(self) -> MatrixN<N, D> {
        self.matrix
    }

    /// Unwraps the underlying matrix.
    /// Deprecated: Use [Rotation::into_inner] instead.
    #[deprecated(note="use `.into_inner()` instead")]
    #[inline]
    pub fn unwrap(self) -> MatrixN<N, D> {
        self.matrix
    }

    /// Converts this rotation into its equivalent homogeneous transformation matrix.
    ///
    /// This is the same as `self.into()`.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{Rotation2, Rotation3, Vector3, Matrix3, Matrix4};
    /// # use std::f32;
    /// let rot = Rotation3::from_axis_angle(&Vector3::z_axis(), f32::consts::FRAC_PI_6);
    /// let expected = Matrix4::new(0.8660254, -0.5,      0.0, 0.0,
    ///                             0.5,       0.8660254, 0.0, 0.0,
    ///                             0.0,       0.0,       1.0, 0.0,
    ///                             0.0,       0.0,       0.0, 1.0);
    /// assert_eq!(rot.to_homogeneous(), expected);
    ///
    ///
    /// let rot = Rotation2::new(f32::consts::FRAC_PI_6);
    /// let expected = Matrix3::new(0.8660254, -0.5,      0.0,
    ///                             0.5,       0.8660254, 0.0,
    ///                             0.0,       0.0,       1.0);
    /// assert_eq!(rot.to_homogeneous(), expected);
    /// ```
    #[inline]
    pub fn to_homogeneous(&self) -> MatrixN<N, DimNameSum<D, U1>>
    where
        N: Zero + One,
        D: DimNameAdd<U1>,
        DefaultAllocator: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>>,
    {
        // We could use `MatrixN::to_homogeneous()` here, but that would imply
        // adding the additional traits `DimAdd` and `IsNotStaticOne`. Maybe
        // these things will get nicer once specialization lands in Rust.
        let mut res = MatrixN::<N, DimNameSum<D, U1>>::identity();
        res.fixed_slice_mut::<D, D>(0, 0).copy_from(&self.matrix);

        res
    }

    /// Creates a new rotation from the given square matrix.
    ///
    /// The matrix squareness is checked but not its orthonormality.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{Rotation2, Rotation3, Matrix2, Matrix3};
    /// # use std::f32;
    /// let mat = Matrix3::new(0.8660254, -0.5,      0.0,
    ///                        0.5,       0.8660254, 0.0,
    ///                        0.0,       0.0,       1.0);
    /// let rot = Rotation3::from_matrix_unchecked(mat);
    ///
    /// assert_eq!(*rot.matrix(), mat);
    ///
    ///
    /// let mat = Matrix2::new(0.8660254, -0.5,
    ///                        0.5,       0.8660254);
    /// let rot = Rotation2::from_matrix_unchecked(mat);
    ///
    /// assert_eq!(*rot.matrix(), mat);
    /// ```
    #[inline]
    pub fn from_matrix_unchecked(matrix: MatrixN<N, D>) -> Self {
        assert!(
            matrix.is_square(),
            "Unable to create a rotation from a non-square matrix."
        );

        Self { matrix: matrix }
    }

    /// Transposes `self`.
    ///
    /// Same as `.inverse()` because the inverse of a rotation matrix is its transform.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Rotation2, Rotation3, Vector3};
    /// let rot = Rotation3::new(Vector3::new(1.0, 2.0, 3.0));
    /// let tr_rot = rot.transpose();
    /// assert_relative_eq!(rot * tr_rot, Rotation3::identity(), epsilon = 1.0e-6);
    /// assert_relative_eq!(tr_rot * rot, Rotation3::identity(), epsilon = 1.0e-6);
    ///
    /// let rot = Rotation2::new(1.2);
    /// let tr_rot = rot.transpose();
    /// assert_relative_eq!(rot * tr_rot, Rotation2::identity(), epsilon = 1.0e-6);
    /// assert_relative_eq!(tr_rot * rot, Rotation2::identity(), epsilon = 1.0e-6);
    /// ```
    #[inline]
    #[must_use = "Did you mean to use transpose_mut()?"]
    pub fn transpose(&self) -> Self {
        Self::from_matrix_unchecked(self.matrix.transpose())
    }

    /// Inverts `self`.
    ///
    /// Same as `.transpose()` because the inverse of a rotation matrix is its transform.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Rotation2, Rotation3, Vector3};
    /// let rot = Rotation3::new(Vector3::new(1.0, 2.0, 3.0));
    /// let inv = rot.inverse();
    /// assert_relative_eq!(rot * inv, Rotation3::identity(), epsilon = 1.0e-6);
    /// assert_relative_eq!(inv * rot, Rotation3::identity(), epsilon = 1.0e-6);
    ///
    /// let rot = Rotation2::new(1.2);
    /// let inv = rot.inverse();
    /// assert_relative_eq!(rot * inv, Rotation2::identity(), epsilon = 1.0e-6);
    /// assert_relative_eq!(inv * rot, Rotation2::identity(), epsilon = 1.0e-6);
    /// ```
    #[inline]
    #[must_use = "Did you mean to use inverse_mut()?"]
    pub fn inverse(&self) -> Self {
        self.transpose()
    }

    /// Transposes `self` in-place.
    ///
    /// Same as `.inverse_mut()` because the inverse of a rotation matrix is its transform.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Rotation2, Rotation3, Vector3};
    /// let rot = Rotation3::new(Vector3::new(1.0, 2.0, 3.0));
    /// let mut tr_rot = Rotation3::new(Vector3::new(1.0, 2.0, 3.0));
    /// tr_rot.transpose_mut();
    ///
    /// assert_relative_eq!(rot * tr_rot, Rotation3::identity(), epsilon = 1.0e-6);
    /// assert_relative_eq!(tr_rot * rot, Rotation3::identity(), epsilon = 1.0e-6);
    ///
    /// let rot = Rotation2::new(1.2);
    /// let mut tr_rot = Rotation2::new(1.2);
    /// tr_rot.transpose_mut();
    ///
    /// assert_relative_eq!(rot * tr_rot, Rotation2::identity(), epsilon = 1.0e-6);
    /// assert_relative_eq!(tr_rot * rot, Rotation2::identity(), epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn transpose_mut(&mut self) {
        self.matrix.transpose_mut()
    }

    /// Inverts `self` in-place.
    ///
    /// Same as `.transpose_mut()` because the inverse of a rotation matrix is its transform.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Rotation2, Rotation3, Vector3};
    /// let rot = Rotation3::new(Vector3::new(1.0, 2.0, 3.0));
    /// let mut inv = Rotation3::new(Vector3::new(1.0, 2.0, 3.0));
    /// inv.inverse_mut();
    ///
    /// assert_relative_eq!(rot * inv, Rotation3::identity(), epsilon = 1.0e-6);
    /// assert_relative_eq!(inv * rot, Rotation3::identity(), epsilon = 1.0e-6);
    ///
    /// let rot = Rotation2::new(1.2);
    /// let mut inv = Rotation2::new(1.2);
    /// inv.inverse_mut();
    ///
    /// assert_relative_eq!(rot * inv, Rotation2::identity(), epsilon = 1.0e-6);
    /// assert_relative_eq!(inv * rot, Rotation2::identity(), epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn inverse_mut(&mut self) {
        self.transpose_mut()
    }
}

impl<N: RealField, D: DimName> Rotation<N, D>
where DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
{
    /// Rotate the given point.
    ///
    /// This is the same as the multiplication `self * pt`.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Point3, Rotation2, Rotation3, UnitQuaternion, Vector3};
    /// let rot = Rotation3::new(Vector3::y() * f32::consts::FRAC_PI_2);
    /// let transformed_point = rot.transform_point(&Point3::new(1.0, 2.0, 3.0));
    ///
    /// assert_relative_eq!(transformed_point, Point3::new(3.0, 2.0, -1.0), epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn transform_point(&self, pt: &Point<N, D>) -> Point<N, D> {
        self * pt
    }

    /// Rotate the given vector.
    ///
    /// This is the same as the multiplication `self * v`.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Rotation2, Rotation3, UnitQuaternion, Vector3};
    /// let rot = Rotation3::new(Vector3::y() * f32::consts::FRAC_PI_2);
    /// let transformed_vector = rot.transform_vector(&Vector3::new(1.0, 2.0, 3.0));
    ///
    /// assert_relative_eq!(transformed_vector, Vector3::new(3.0, 2.0, -1.0), epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn transform_vector(&self, v: &VectorN<N, D>) -> VectorN<N, D> {
        self * v
    }

    /// Rotate the given point by the inverse of this rotation. This may be
    /// cheaper than inverting the rotation and then transforming the given
    /// point.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Point3, Rotation2, Rotation3, UnitQuaternion, Vector3};
    /// let rot = Rotation3::new(Vector3::y() * f32::consts::FRAC_PI_2);
    /// let transformed_point = rot.inverse_transform_point(&Point3::new(1.0, 2.0, 3.0));
    ///
    /// assert_relative_eq!(transformed_point, Point3::new(-3.0, 2.0, 1.0), epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn inverse_transform_point(&self, pt: &Point<N, D>) -> Point<N, D> {
        Point::from(self.inverse_transform_vector(&pt.coords))
    }

    /// Rotate the given vector by the inverse of this rotation. This may be
    /// cheaper than inverting the rotation and then transforming the given
    /// vector.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Rotation2, Rotation3, UnitQuaternion, Vector3};
    /// let rot = Rotation3::new(Vector3::y() * f32::consts::FRAC_PI_2);
    /// let transformed_vector = rot.inverse_transform_vector(&Vector3::new(1.0, 2.0, 3.0));
    ///
    /// assert_relative_eq!(transformed_vector, Vector3::new(-3.0, 2.0, 1.0), epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn inverse_transform_vector(&self, v: &VectorN<N, D>) -> VectorN<N, D> {
        self.matrix().tr_mul(v)
    }
}

impl<N: Scalar + Eq, D: DimName> Eq for Rotation<N, D> where DefaultAllocator: Allocator<N, D, D> {}

impl<N: Scalar + PartialEq, D: DimName> PartialEq for Rotation<N, D>
where DefaultAllocator: Allocator<N, D, D>
{
    #[inline]
    fn eq(&self, right: &Self) -> bool {
        self.matrix == right.matrix
    }
}

impl<N, D: DimName> AbsDiffEq for Rotation<N, D>
where
    N: Scalar + AbsDiffEq,
    DefaultAllocator: Allocator<N, D, D>,
    N::Epsilon: Copy,
{
    type Epsilon = N::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        N::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.matrix.abs_diff_eq(&other.matrix, epsilon)
    }
}

impl<N, D: DimName> RelativeEq for Rotation<N, D>
where
    N: Scalar + RelativeEq,
    DefaultAllocator: Allocator<N, D, D>,
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
        self.matrix
            .relative_eq(&other.matrix, epsilon, max_relative)
    }
}

impl<N, D: DimName> UlpsEq for Rotation<N, D>
where
    N: Scalar + UlpsEq,
    DefaultAllocator: Allocator<N, D, D>,
    N::Epsilon: Copy,
{
    #[inline]
    fn default_max_ulps() -> u32 {
        N::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.matrix.ulps_eq(&other.matrix, epsilon, max_ulps)
    }
}

/*
 *
 * Display
 *
 */
impl<N, D: DimName> fmt::Display for Rotation<N, D>
where
    N: RealField + fmt::Display,
    DefaultAllocator: Allocator<N, D, D> + Allocator<usize, D, D>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let precision = f.precision().unwrap_or(3);

        writeln!(f, "Rotation matrix {{")?;
        write!(f, "{:.*}", precision, self.matrix)?;
        writeln!(f, "}}")
    }
}

//          //         /*
//          //          *
//          //          * Absolute
//          //          *
//          //          */
//          //         impl<N: Absolute> Absolute for $t<N> {
//          //             type AbsoluteValue = $submatrix<N::AbsoluteValue>;
//          //
//          //             #[inline]
//          //             fn abs(m: &$t<N>) -> $submatrix<N::AbsoluteValue> {
//          //                 Absolute::abs(&m.submatrix)
//          //             }
//          //         }
