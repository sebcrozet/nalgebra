#[cfg(feature = "arbitrary")]
use quickcheck::{Arbitrary, Gen};
use rand::distributions::{Distribution, Standard};
use rand::Rng;
#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::mem;

use alga::general::RealField;

use crate::base::dimension::U3;
use crate::base::helper;
use crate::base::storage::Storage;
use crate::base::{Matrix4, Vector, Vector3};

use crate::geometry::{Point3, Projective3};

/// A 3D orthographic projection stored as a homogeneous 4x4 matrix.
pub struct Orthographic3<N: RealField> {
    matrix: Matrix4<N>,
}

impl<N: RealField> Copy for Orthographic3<N> {}

impl<N: RealField> Clone for Orthographic3<N> {
    #[inline]
    fn clone(&self) -> Self {
        Self::from_matrix_unchecked(self.matrix.clone())
    }
}

impl<N: RealField> fmt::Debug for Orthographic3<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        self.matrix.fmt(f)
    }
}

impl<N: RealField> PartialEq for Orthographic3<N> {
    #[inline]
    fn eq(&self, right: &Self) -> bool {
        self.matrix == right.matrix
    }
}

#[cfg(feature = "serde-serialize")]
impl<N: RealField + Serialize> Serialize for Orthographic3<N> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: Serializer {
        self.matrix.serialize(serializer)
    }
}

#[cfg(feature = "serde-serialize")]
impl<'a, N: RealField + Deserialize<'a>> Deserialize<'a> for Orthographic3<N> {
    fn deserialize<Des>(deserializer: Des) -> Result<Self, Des::Error>
    where Des: Deserializer<'a> {
        let matrix = Matrix4::<N>::deserialize(deserializer)?;

        Ok(Self::from_matrix_unchecked(matrix))
    }
}

impl<N: RealField> Orthographic3<N> {
    /// Creates a new orthographic projection matrix.
    ///
    /// This follows the OpenGL convention, so this will flip the `z` axis.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Orthographic3, Point3};
    /// let proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// // Check this projection actually transforms the view cuboid into the double-unit cube.
    /// // See https://www.nalgebra.org/projections/#orthographic-projection for more details.
    /// let p1 = Point3::new(1.0, 2.0, -0.1);
    /// let p2 = Point3::new(1.0, 2.0, -1000.0);
    /// let p3 = Point3::new(1.0, 20.0, -0.1);
    /// let p4 = Point3::new(1.0, 20.0, -1000.0);
    /// let p5 = Point3::new(10.0, 2.0, -0.1);
    /// let p6 = Point3::new(10.0, 2.0, -1000.0);
    /// let p7 = Point3::new(10.0, 20.0, -0.1);
    /// let p8 = Point3::new(10.0, 20.0, -1000.0);
    ///
    /// assert_relative_eq!(proj.project_point(&p1), Point3::new(-1.0, -1.0, -1.0));
    /// assert_relative_eq!(proj.project_point(&p2), Point3::new(-1.0, -1.0,  1.0));
    /// assert_relative_eq!(proj.project_point(&p3), Point3::new(-1.0,  1.0, -1.0));
    /// assert_relative_eq!(proj.project_point(&p4), Point3::new(-1.0,  1.0,  1.0));
    /// assert_relative_eq!(proj.project_point(&p5), Point3::new( 1.0, -1.0, -1.0));
    /// assert_relative_eq!(proj.project_point(&p6), Point3::new( 1.0, -1.0,  1.0));
    /// assert_relative_eq!(proj.project_point(&p7), Point3::new( 1.0,  1.0, -1.0));
    /// assert_relative_eq!(proj.project_point(&p8), Point3::new( 1.0,  1.0,  1.0));
    ///
    /// // This also works with flipped axis. In other words, we allow that
    /// // `left > right`, `bottom > top`, and/or `znear > zfar`.
    /// let proj = Orthographic3::new(10.0, 1.0, 20.0, 2.0, 1000.0, 0.1);
    ///
    /// assert_relative_eq!(proj.project_point(&p1), Point3::new( 1.0,  1.0,  1.0));
    /// assert_relative_eq!(proj.project_point(&p2), Point3::new( 1.0,  1.0, -1.0));
    /// assert_relative_eq!(proj.project_point(&p3), Point3::new( 1.0, -1.0,  1.0));
    /// assert_relative_eq!(proj.project_point(&p4), Point3::new( 1.0, -1.0, -1.0));
    /// assert_relative_eq!(proj.project_point(&p5), Point3::new(-1.0,  1.0,  1.0));
    /// assert_relative_eq!(proj.project_point(&p6), Point3::new(-1.0,  1.0, -1.0));
    /// assert_relative_eq!(proj.project_point(&p7), Point3::new(-1.0, -1.0,  1.0));
    /// assert_relative_eq!(proj.project_point(&p8), Point3::new(-1.0, -1.0, -1.0));
    /// ```
    #[inline]
    pub fn new(left: N, right: N, bottom: N, top: N, znear: N, zfar: N) -> Self {
        let matrix = Matrix4::<N>::identity();
        let mut res = Self::from_matrix_unchecked(matrix);

        res.set_left_and_right(left, right);
        res.set_bottom_and_top(bottom, top);
        res.set_znear_and_zfar(znear, zfar);

        res
    }

    /// Wraps the given matrix to interpret it as a 3D orthographic matrix.
    ///
    /// It is not checked whether or not the given matrix actually represents an orthographic
    /// projection.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{Orthographic3, Point3, Matrix4};
    /// let mat = Matrix4::new(
    ///     2.0 / 9.0, 0.0,        0.0,         -11.0 / 9.0,
    ///     0.0,       2.0 / 18.0, 0.0,         -22.0 / 18.0,
    ///     0.0,       0.0,       -2.0 / 999.9, -1000.1 / 999.9,
    ///     0.0,       0.0,        0.0,         1.0
    /// );
    /// let proj = Orthographic3::from_matrix_unchecked(mat);
    /// assert_eq!(proj, Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0));
    /// ```
    #[inline]
    pub fn from_matrix_unchecked(matrix: Matrix4<N>) -> Self {
        Self { matrix: matrix }
    }

    /// Creates a new orthographic projection matrix from an aspect ratio and the vertical field of view.
    #[inline]
    pub fn from_fov(aspect: N, vfov: N, znear: N, zfar: N) -> Self {
        assert!(
            znear != zfar,
            "The far plane must not be equal to the near plane."
        );
        assert!(
            !relative_eq!(aspect, N::zero()),
            "The apsect ratio must not be zero."
        );

        let half: N = crate::convert(0.5);
        let width = zfar * (vfov * half).tan();
        let height = width / aspect;

        Self::new(
            -width * half,
            width * half,
            -height * half,
            height * half,
            znear,
            zfar,
        )
    }

    /// Retrieves the inverse of the underlying homogeneous matrix.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Orthographic3, Point3, Matrix4};
    /// let proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// let inv = proj.inverse();
    ///
    /// assert_relative_eq!(inv * proj.as_matrix(), Matrix4::identity());
    /// assert_relative_eq!(proj.as_matrix() * inv, Matrix4::identity());
    ///
    /// let proj = Orthographic3::new(10.0, 1.0, 20.0, 2.0, 1000.0, 0.1);
    /// let inv = proj.inverse();
    /// assert_relative_eq!(inv * proj.as_matrix(), Matrix4::identity());
    /// assert_relative_eq!(proj.as_matrix() * inv, Matrix4::identity());
    /// ```
    #[inline]
    pub fn inverse(&self) -> Matrix4<N> {
        let mut res = self.to_homogeneous();

        let inv_m11 = N::one() / self.matrix[(0, 0)];
        let inv_m22 = N::one() / self.matrix[(1, 1)];
        let inv_m33 = N::one() / self.matrix[(2, 2)];

        res[(0, 0)] = inv_m11;
        res[(1, 1)] = inv_m22;
        res[(2, 2)] = inv_m33;

        res[(0, 3)] = -self.matrix[(0, 3)] * inv_m11;
        res[(1, 3)] = -self.matrix[(1, 3)] * inv_m22;
        res[(2, 3)] = -self.matrix[(2, 3)] * inv_m33;

        res
    }

    /// Computes the corresponding homogeneous matrix.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{Orthographic3, Point3, Matrix4};
    /// let proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// let expected = Matrix4::new(
    ///     2.0 / 9.0, 0.0,        0.0,         -11.0 / 9.0,
    ///     0.0,       2.0 / 18.0, 0.0,         -22.0 / 18.0,
    ///     0.0,       0.0,       -2.0 / 999.9, -1000.1 / 999.9,
    ///     0.0,       0.0,        0.0,         1.0
    /// );
    /// assert_eq!(proj.to_homogeneous(), expected);
    /// ```
    #[inline]
    pub fn to_homogeneous(&self) -> Matrix4<N> {
        self.matrix
    }

    /// A reference to the underlying homogeneous transformation matrix.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{Orthographic3, Point3, Matrix4};
    /// let proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// let expected = Matrix4::new(
    ///     2.0 / 9.0, 0.0,        0.0,         -11.0 / 9.0,
    ///     0.0,       2.0 / 18.0, 0.0,         -22.0 / 18.0,
    ///     0.0,       0.0,       -2.0 / 999.9, -1000.1 / 999.9,
    ///     0.0,       0.0,        0.0,         1.0
    /// );
    /// assert_eq!(*proj.as_matrix(), expected);
    /// ```
    #[inline]
    pub fn as_matrix(&self) -> &Matrix4<N> {
        &self.matrix
    }

    /// A reference to this transformation seen as a `Projective3`.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::Orthographic3;
    /// let proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// assert_eq!(proj.as_projective().to_homogeneous(), proj.to_homogeneous());
    /// ```
    #[inline]
    pub fn as_projective(&self) -> &Projective3<N> {
        unsafe { mem::transmute(self) }
    }

    /// This transformation seen as a `Projective3`.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::Orthographic3;
    /// let proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// assert_eq!(proj.to_projective().to_homogeneous(), proj.to_homogeneous());
    /// ```
    #[inline]
    pub fn to_projective(&self) -> Projective3<N> {
        Projective3::from_matrix_unchecked(self.matrix)
    }

    /// Retrieves the underlying homogeneous matrix.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Orthographic3, Point3, Matrix4};
    /// let proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// let expected = Matrix4::new(
    ///     2.0 / 9.0, 0.0,        0.0,         -11.0 / 9.0,
    ///     0.0,       2.0 / 18.0, 0.0,         -22.0 / 18.0,
    ///     0.0,       0.0,       -2.0 / 999.9, -1000.1 / 999.9,
    ///     0.0,       0.0,        0.0,         1.0
    /// );
    /// assert_eq!(proj.into_inner(), expected);
    /// ```
    #[inline]
    pub fn into_inner(self) -> Matrix4<N> {
        self.matrix
    }

    /// Retrieves the underlying homogeneous matrix.
    /// Deprecated: Use [Orthographic3::into_inner] instead.
    #[deprecated(note="use `.into_inner()` instead")]
    #[inline]
    pub fn unwrap(self) -> Matrix4<N> {
        self.matrix
    }

    /// The left offset of the view cuboid.
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Orthographic3;
    /// let proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// assert_relative_eq!(proj.left(), 1.0, epsilon = 1.0e-6);
    ///
    /// let proj = Orthographic3::new(10.0, 1.0, 20.0, 2.0, 1000.0, 0.1);
    /// assert_relative_eq!(proj.left(), 10.0, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn left(&self) -> N {
        (-N::one() - self.matrix[(0, 3)]) / self.matrix[(0, 0)]
    }

    /// The right offset of the view cuboid.
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Orthographic3;
    /// let proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// assert_relative_eq!(proj.right(), 10.0, epsilon = 1.0e-6);
    ///
    /// let proj = Orthographic3::new(10.0, 1.0, 20.0, 2.0, 1000.0, 0.1);
    /// assert_relative_eq!(proj.right(), 1.0, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn right(&self) -> N {
        (N::one() - self.matrix[(0, 3)]) / self.matrix[(0, 0)]
    }

    /// The bottom offset of the view cuboid.
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Orthographic3;
    /// let proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// assert_relative_eq!(proj.bottom(), 2.0, epsilon = 1.0e-6);
    ///
    /// let proj = Orthographic3::new(10.0, 1.0, 20.0, 2.0, 1000.0, 0.1);
    /// assert_relative_eq!(proj.bottom(), 20.0, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn bottom(&self) -> N {
        (-N::one() - self.matrix[(1, 3)]) / self.matrix[(1, 1)]
    }

    /// The top offset of the view cuboid.
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Orthographic3;
    /// let proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// assert_relative_eq!(proj.top(), 20.0, epsilon = 1.0e-6);
    ///
    /// let proj = Orthographic3::new(10.0, 1.0, 20.0, 2.0, 1000.0, 0.1);
    /// assert_relative_eq!(proj.top(), 2.0, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn top(&self) -> N {
        (N::one() - self.matrix[(1, 3)]) / self.matrix[(1, 1)]
    }

    /// The near plane offset of the view cuboid.
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Orthographic3;
    /// let proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// assert_relative_eq!(proj.znear(), 0.1, epsilon = 1.0e-6);
    ///
    /// let proj = Orthographic3::new(10.0, 1.0, 20.0, 2.0, 1000.0, 0.1);
    /// assert_relative_eq!(proj.znear(), 1000.0, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn znear(&self) -> N {
        (N::one() + self.matrix[(2, 3)]) / self.matrix[(2, 2)]
    }

    /// The far plane offset of the view cuboid.
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Orthographic3;
    /// let proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// assert_relative_eq!(proj.zfar(), 1000.0, epsilon = 1.0e-6);
    ///
    /// let proj = Orthographic3::new(10.0, 1.0, 20.0, 2.0, 1000.0, 0.1);
    /// assert_relative_eq!(proj.zfar(), 0.1, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn zfar(&self) -> N {
        (-N::one() + self.matrix[(2, 3)]) / self.matrix[(2, 2)]
    }

    // FIXME: when we get specialization, specialize the Mul impl instead.
    /// Projects a point. Faster than matrix multiplication.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Orthographic3, Point3};
    /// let proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    ///
    /// let p1 = Point3::new(1.0, 2.0, -0.1);
    /// let p2 = Point3::new(1.0, 2.0, -1000.0);
    /// let p3 = Point3::new(1.0, 20.0, -0.1);
    /// let p4 = Point3::new(1.0, 20.0, -1000.0);
    /// let p5 = Point3::new(10.0, 2.0, -0.1);
    /// let p6 = Point3::new(10.0, 2.0, -1000.0);
    /// let p7 = Point3::new(10.0, 20.0, -0.1);
    /// let p8 = Point3::new(10.0, 20.0, -1000.0);
    ///
    /// assert_relative_eq!(proj.project_point(&p1), Point3::new(-1.0, -1.0, -1.0));
    /// assert_relative_eq!(proj.project_point(&p2), Point3::new(-1.0, -1.0,  1.0));
    /// assert_relative_eq!(proj.project_point(&p3), Point3::new(-1.0,  1.0, -1.0));
    /// assert_relative_eq!(proj.project_point(&p4), Point3::new(-1.0,  1.0,  1.0));
    /// assert_relative_eq!(proj.project_point(&p5), Point3::new( 1.0, -1.0, -1.0));
    /// assert_relative_eq!(proj.project_point(&p6), Point3::new( 1.0, -1.0,  1.0));
    /// assert_relative_eq!(proj.project_point(&p7), Point3::new( 1.0,  1.0, -1.0));
    /// assert_relative_eq!(proj.project_point(&p8), Point3::new( 1.0,  1.0,  1.0));
    /// ```
    #[inline]
    pub fn project_point(&self, p: &Point3<N>) -> Point3<N> {
        Point3::new(
            self.matrix[(0, 0)] * p[0] + self.matrix[(0, 3)],
            self.matrix[(1, 1)] * p[1] + self.matrix[(1, 3)],
            self.matrix[(2, 2)] * p[2] + self.matrix[(2, 3)],
        )
    }

    /// Un-projects a point. Faster than multiplication by the underlying matrix inverse.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Orthographic3, Point3};
    /// let proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    ///
    /// let p1 = Point3::new(-1.0, -1.0, -1.0);
    /// let p2 = Point3::new(-1.0, -1.0,  1.0);
    /// let p3 = Point3::new(-1.0,  1.0, -1.0);
    /// let p4 = Point3::new(-1.0,  1.0,  1.0);
    /// let p5 = Point3::new( 1.0, -1.0, -1.0);
    /// let p6 = Point3::new( 1.0, -1.0,  1.0);
    /// let p7 = Point3::new( 1.0,  1.0, -1.0);
    /// let p8 = Point3::new( 1.0,  1.0,  1.0);
    ///
    /// assert_relative_eq!(proj.unproject_point(&p1), Point3::new(1.0, 2.0, -0.1), epsilon = 1.0e-6);
    /// assert_relative_eq!(proj.unproject_point(&p2), Point3::new(1.0, 2.0, -1000.0), epsilon = 1.0e-6);
    /// assert_relative_eq!(proj.unproject_point(&p3), Point3::new(1.0, 20.0, -0.1), epsilon = 1.0e-6);
    /// assert_relative_eq!(proj.unproject_point(&p4), Point3::new(1.0, 20.0, -1000.0), epsilon = 1.0e-6);
    /// assert_relative_eq!(proj.unproject_point(&p5), Point3::new(10.0, 2.0, -0.1), epsilon = 1.0e-6);
    /// assert_relative_eq!(proj.unproject_point(&p6), Point3::new(10.0, 2.0, -1000.0), epsilon = 1.0e-6);
    /// assert_relative_eq!(proj.unproject_point(&p7), Point3::new(10.0, 20.0, -0.1), epsilon = 1.0e-6);
    /// assert_relative_eq!(proj.unproject_point(&p8), Point3::new(10.0, 20.0, -1000.0), epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn unproject_point(&self, p: &Point3<N>) -> Point3<N> {
        Point3::new(
            (p[0] - self.matrix[(0, 3)]) / self.matrix[(0, 0)],
            (p[1] - self.matrix[(1, 3)]) / self.matrix[(1, 1)],
            (p[2] - self.matrix[(2, 3)]) / self.matrix[(2, 2)],
        )
    }

    // FIXME: when we get specialization, specialize the Mul impl instead.
    /// Projects a vector. Faster than matrix multiplication.
    ///
    /// Vectors are not affected by the translation part of the projection.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Orthographic3, Vector3};
    /// let proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    ///
    /// let v1 = Vector3::x();
    /// let v2 = Vector3::y();
    /// let v3 = Vector3::z();
    ///
    /// assert_relative_eq!(proj.project_vector(&v1), Vector3::x() * 2.0 / 9.0);
    /// assert_relative_eq!(proj.project_vector(&v2), Vector3::y() * 2.0 / 18.0);
    /// assert_relative_eq!(proj.project_vector(&v3), Vector3::z() * -2.0 / 999.9);
    /// ```
    #[inline]
    pub fn project_vector<SB>(&self, p: &Vector<N, U3, SB>) -> Vector3<N>
    where SB: Storage<N, U3> {
        Vector3::new(
            self.matrix[(0, 0)] * p[0],
            self.matrix[(1, 1)] * p[1],
            self.matrix[(2, 2)] * p[2],
        )
    }

    /// Sets the left offset of the view cuboid.
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Orthographic3;
    /// let mut proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// proj.set_left(2.0);
    /// assert_relative_eq!(proj.left(), 2.0, epsilon = 1.0e-6);
    ///
    /// // It is OK to set a left offset greater than the current right offset.
    /// proj.set_left(20.0);
    /// assert_relative_eq!(proj.left(), 20.0, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn set_left(&mut self, left: N) {
        let right = self.right();
        self.set_left_and_right(left, right);
    }

    /// Sets the right offset of the view cuboid.
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Orthographic3;
    /// let mut proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// proj.set_right(15.0);
    /// assert_relative_eq!(proj.right(), 15.0, epsilon = 1.0e-6);
    ///
    /// // It is OK to set a right offset smaller than the current left offset.
    /// proj.set_right(-3.0);
    /// assert_relative_eq!(proj.right(), -3.0, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn set_right(&mut self, right: N) {
        let left = self.left();
        self.set_left_and_right(left, right);
    }

    /// Sets the bottom offset of the view cuboid.
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Orthographic3;
    /// let mut proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// proj.set_bottom(8.0);
    /// assert_relative_eq!(proj.bottom(), 8.0, epsilon = 1.0e-6);
    ///
    /// // It is OK to set a bottom offset greater than the current top offset.
    /// proj.set_bottom(50.0);
    /// assert_relative_eq!(proj.bottom(), 50.0, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn set_bottom(&mut self, bottom: N) {
        let top = self.top();
        self.set_bottom_and_top(bottom, top);
    }

    /// Sets the top offset of the view cuboid.
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Orthographic3;
    /// let mut proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// proj.set_top(15.0);
    /// assert_relative_eq!(proj.top(), 15.0, epsilon = 1.0e-6);
    ///
    /// // It is OK to set a top offset smaller than the current bottom offset.
    /// proj.set_top(-3.0);
    /// assert_relative_eq!(proj.top(), -3.0, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn set_top(&mut self, top: N) {
        let bottom = self.bottom();
        self.set_bottom_and_top(bottom, top);
    }

    /// Sets the near plane offset of the view cuboid.
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Orthographic3;
    /// let mut proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// proj.set_znear(8.0);
    /// assert_relative_eq!(proj.znear(), 8.0, epsilon = 1.0e-6);
    ///
    /// // It is OK to set a znear greater than the current zfar.
    /// proj.set_znear(5000.0);
    /// assert_relative_eq!(proj.znear(), 5000.0, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn set_znear(&mut self, znear: N) {
        let zfar = self.zfar();
        self.set_znear_and_zfar(znear, zfar);
    }

    /// Sets the far plane offset of the view cuboid.
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Orthographic3;
    /// let mut proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// proj.set_zfar(15.0);
    /// assert_relative_eq!(proj.zfar(), 15.0, epsilon = 1.0e-6);
    ///
    /// // It is OK to set a zfar smaller than the current znear.
    /// proj.set_zfar(-3.0);
    /// assert_relative_eq!(proj.zfar(), -3.0, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn set_zfar(&mut self, zfar: N) {
        let znear = self.znear();
        self.set_znear_and_zfar(znear, zfar);
    }

    /// Sets the view cuboid offsets along the `x` axis.
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Orthographic3;
    /// let mut proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// proj.set_left_and_right(7.0, 70.0);
    /// assert_relative_eq!(proj.left(), 7.0, epsilon = 1.0e-6);
    /// assert_relative_eq!(proj.right(), 70.0, epsilon = 1.0e-6);
    ///
    /// // It is also OK to have `left > right`.
    /// proj.set_left_and_right(70.0, 7.0);
    /// assert_relative_eq!(proj.left(), 70.0, epsilon = 1.0e-6);
    /// assert_relative_eq!(proj.right(), 7.0, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn set_left_and_right(&mut self, left: N, right: N) {
        assert!(
            left != right,
            "The left corner must not be equal to the right corner."
        );
        self.matrix[(0, 0)] = crate::convert::<_, N>(2.0) / (right - left);
        self.matrix[(0, 3)] = -(right + left) / (right - left);
    }

    /// Sets the view cuboid offsets along the `y` axis.
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Orthographic3;
    /// let mut proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// proj.set_bottom_and_top(7.0, 70.0);
    /// assert_relative_eq!(proj.bottom(), 7.0, epsilon = 1.0e-6);
    /// assert_relative_eq!(proj.top(), 70.0, epsilon = 1.0e-6);
    ///
    /// // It is also OK to have `bottom > top`.
    /// proj.set_bottom_and_top(70.0, 7.0);
    /// assert_relative_eq!(proj.bottom(), 70.0, epsilon = 1.0e-6);
    /// assert_relative_eq!(proj.top(), 7.0, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn set_bottom_and_top(&mut self, bottom: N, top: N) {
        assert!(
            bottom != top,
            "The top corner must not be equal to the bottom corner."
        );
        self.matrix[(1, 1)] = crate::convert::<_, N>(2.0) / (top - bottom);
        self.matrix[(1, 3)] = -(top + bottom) / (top - bottom);
    }

    /// Sets the near and far plane offsets of the view cuboid.
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Orthographic3;
    /// let mut proj = Orthographic3::new(1.0, 10.0, 2.0, 20.0, 0.1, 1000.0);
    /// proj.set_znear_and_zfar(50.0, 5000.0);
    /// assert_relative_eq!(proj.znear(), 50.0, epsilon = 1.0e-6);
    /// assert_relative_eq!(proj.zfar(), 5000.0, epsilon = 1.0e-6);
    ///
    /// // It is also OK to have `znear > zfar`.
    /// proj.set_znear_and_zfar(5000.0, 0.5);
    /// assert_relative_eq!(proj.znear(), 5000.0, epsilon = 1.0e-6);
    /// assert_relative_eq!(proj.zfar(), 0.5, epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn set_znear_and_zfar(&mut self, znear: N, zfar: N) {
        assert!(
            zfar != znear,
            "The near-plane and far-plane must not be superimposed."
        );
        self.matrix[(2, 2)] = -crate::convert::<_, N>(2.0) / (zfar - znear);
        self.matrix[(2, 3)] = -(zfar + znear) / (zfar - znear);
    }
}

impl<N: RealField> Distribution<Orthographic3<N>> for Standard
where Standard: Distribution<N>
{
    fn sample<R: Rng + ?Sized>(&self, r: &mut R) -> Orthographic3<N> {
        let left = r.gen();
        let right = helper::reject_rand(r, |x: &N| *x > left);
        let bottom = r.gen();
        let top = helper::reject_rand(r, |x: &N| *x > bottom);
        let znear = r.gen();
        let zfar = helper::reject_rand(r, |x: &N| *x > znear);

        Orthographic3::new(left, right, bottom, top, znear, zfar)
    }
}

#[cfg(feature = "arbitrary")]
impl<N: RealField + Arbitrary> Arbitrary for Orthographic3<N>
where Matrix4<N>: Send
{
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        let left = Arbitrary::arbitrary(g);
        let right = helper::reject(g, |x: &N| *x > left);
        let bottom = Arbitrary::arbitrary(g);
        let top = helper::reject(g, |x: &N| *x > bottom);
        let znear = Arbitrary::arbitrary(g);
        let zfar = helper::reject(g, |x: &N| *x > znear);

        Self::new(left, right, bottom, top, znear, zfar)
    }
}

impl<N: RealField> From<Orthographic3<N>> for Matrix4<N> {
    #[inline]
    fn from(orth: Orthographic3<N>) -> Self {
        orth.into_inner()
    }
}
