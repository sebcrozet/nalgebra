/*
 *
 * Computer-graphics specific implementations.
 * Currently, it is mostly implemented for homogeneous matrices in 2- and 3-space.
 *
 */

use num::One;

use crate::base::allocator::Allocator;
use crate::base::dimension::{DimName, DimNameDiff, DimNameSub, U1};
use crate::base::storage::{Storage, StorageMut};
use crate::base::{
    DefaultAllocator, Matrix3, Matrix4, MatrixN, Scalar, SquareMatrix, Unit, Vector, Vector3,
    VectorN,
};
use crate::geometry::{
    Isometry, IsometryMatrix3, Orthographic3, Perspective3, Point, Point3, Rotation2, Rotation3,
};

use alga::general::{RealField, Ring};
use alga::linear::Transformation;

impl<N, D: DimName> MatrixN<N, D>
where
    N: Scalar + Ring,
    DefaultAllocator: Allocator<N, D, D>,
{
    /// Creates a new homogeneous matrix that applies the same scaling factor on each dimension.
    #[inline]
    pub fn new_scaling(scaling: N) -> Self {
        let mut res = Self::from_diagonal_element(scaling);
        res[(D::dim() - 1, D::dim() - 1)] = N::one();

        res
    }

    /// Creates a new homogeneous matrix that applies a distinct scaling factor for each dimension.
    #[inline]
    pub fn new_nonuniform_scaling<SB>(scaling: &Vector<N, DimNameDiff<D, U1>, SB>) -> Self
    where
        D: DimNameSub<U1>,
        SB: Storage<N, DimNameDiff<D, U1>>,
    {
        let mut res = Self::one();
        for i in 0..scaling.len() {
            res[(i, i)] = scaling[i].inlined_clone();
        }

        res
    }

    /// Creates a new homogeneous matrix that applies a pure translation.
    #[inline]
    pub fn new_translation<SB>(translation: &Vector<N, DimNameDiff<D, U1>, SB>) -> Self
    where
        D: DimNameSub<U1>,
        SB: Storage<N, DimNameDiff<D, U1>>,
    {
        let mut res = Self::one();
        res.fixed_slice_mut::<DimNameDiff<D, U1>, U1>(0, D::dim() - 1)
            .copy_from(translation);

        res
    }
}

impl<N: RealField> Matrix3<N> {
    /// Builds a 2 dimensional homogeneous rotation matrix from an angle in radian.
    #[inline]
    pub fn new_rotation(angle: N) -> Self {
        Rotation2::new(angle).to_homogeneous()
    }
}

impl<N: RealField> Matrix4<N> {
    /// Builds a 3D homogeneous rotation matrix from an axis and an angle (multiplied together).
    ///
    /// Returns the identity matrix if the given argument is zero.
    #[inline]
    pub fn new_rotation(axisangle: Vector3<N>) -> Self {
        Rotation3::new(axisangle).to_homogeneous()
    }

    /// Builds a 3D homogeneous rotation matrix from an axis and an angle (multiplied together).
    ///
    /// Returns the identity matrix if the given argument is zero.
    #[inline]
    pub fn new_rotation_wrt_point(axisangle: Vector3<N>, pt: Point3<N>) -> Self {
        let rot = Rotation3::from_scaled_axis(axisangle);
        Isometry::rotation_wrt_point(rot, pt).to_homogeneous()
    }

    /// Builds a 3D homogeneous rotation matrix from an axis and an angle (multiplied together).
    ///
    /// Returns the identity matrix if the given argument is zero.
    /// This is identical to `Self::new_rotation`.
    #[inline]
    pub fn from_scaled_axis(axisangle: Vector3<N>) -> Self {
        Rotation3::from_scaled_axis(axisangle).to_homogeneous()
    }

    /// Creates a new rotation from Euler angles.
    ///
    /// The primitive rotations are applied in order: 1 roll − 2 pitch − 3 yaw.
    pub fn from_euler_angles(roll: N, pitch: N, yaw: N) -> Self {
        Rotation3::from_euler_angles(roll, pitch, yaw).to_homogeneous()
    }

    /// Builds a 3D homogeneous rotation matrix from an axis and a rotation angle.
    pub fn from_axis_angle(axis: &Unit<Vector3<N>>, angle: N) -> Self {
        Rotation3::from_axis_angle(axis, angle).to_homogeneous()
    }

    /// Creates a new homogeneous matrix for an orthographic projection.
    #[inline]
    pub fn new_orthographic(left: N, right: N, bottom: N, top: N, znear: N, zfar: N) -> Self {
        Orthographic3::new(left, right, bottom, top, znear, zfar).into_inner()
    }

    /// Creates a new homogeneous matrix for a perspective projection.
    #[inline]
    pub fn new_perspective(aspect: N, fovy: N, znear: N, zfar: N) -> Self {
        Perspective3::new(aspect, fovy, znear, zfar).into_inner()
    }

    /// Creates an isometry that corresponds to the local frame of an observer standing at the
    /// point `eye` and looking toward `target`.
    ///
    /// It maps the view direction `target - eye` to the positive `z` axis and the origin to the
    /// `eye`.
    #[inline]
    pub fn face_towards(eye: &Point3<N>, target: &Point3<N>, up: &Vector3<N>) -> Self {
        IsometryMatrix3::face_towards(eye, target, up).to_homogeneous()
    }

    /// Deprecated: Use [Matrix4::face_towards] instead.
    #[deprecated(note="renamed to `face_towards`")]
    pub fn new_observer_frame(eye: &Point3<N>, target: &Point3<N>, up: &Vector3<N>) -> Self {
        Matrix4::face_towards(eye, target, up)
    }

    /// Builds a right-handed look-at view matrix.
    #[inline]
    pub fn look_at_rh(eye: &Point3<N>, target: &Point3<N>, up: &Vector3<N>) -> Self {
        IsometryMatrix3::look_at_rh(eye, target, up).to_homogeneous()
    }

    /// Builds a left-handed look-at view matrix.
    #[inline]
    pub fn look_at_lh(eye: &Point3<N>, target: &Point3<N>, up: &Vector3<N>) -> Self {
        IsometryMatrix3::look_at_lh(eye, target, up).to_homogeneous()
    }
}

impl<N: Scalar + Ring, D: DimName, S: Storage<N, D, D>> SquareMatrix<N, D, S> {
    /// Computes the transformation equal to `self` followed by an uniform scaling factor.
    #[inline]
    #[must_use = "Did you mean to use append_scaling_mut()?"]
    pub fn append_scaling(&self, scaling: N) -> MatrixN<N, D>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<N, D, D>,
    {
        let mut res = self.clone_owned();
        res.append_scaling_mut(scaling);
        res
    }

    /// Computes the transformation equal to an uniform scaling factor followed by `self`.
    #[inline]
    #[must_use = "Did you mean to use prepend_scaling_mut()?"]
    pub fn prepend_scaling(&self, scaling: N) -> MatrixN<N, D>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<N, D, D>,
    {
        let mut res = self.clone_owned();
        res.prepend_scaling_mut(scaling);
        res
    }

    /// Computes the transformation equal to `self` followed by a non-uniform scaling factor.
    #[inline]
    #[must_use = "Did you mean to use append_nonuniform_scaling_mut()?"]
    pub fn append_nonuniform_scaling<SB>(
        &self,
        scaling: &Vector<N, DimNameDiff<D, U1>, SB>,
    ) -> MatrixN<N, D>
    where
        D: DimNameSub<U1>,
        SB: Storage<N, DimNameDiff<D, U1>>,
        DefaultAllocator: Allocator<N, D, D>,
    {
        let mut res = self.clone_owned();
        res.append_nonuniform_scaling_mut(scaling);
        res
    }

    /// Computes the transformation equal to a non-uniform scaling factor followed by `self`.
    #[inline]
    #[must_use = "Did you mean to use prepend_nonuniform_scaling_mut()?"]
    pub fn prepend_nonuniform_scaling<SB>(
        &self,
        scaling: &Vector<N, DimNameDiff<D, U1>, SB>,
    ) -> MatrixN<N, D>
    where
        D: DimNameSub<U1>,
        SB: Storage<N, DimNameDiff<D, U1>>,
        DefaultAllocator: Allocator<N, D, D>,
    {
        let mut res = self.clone_owned();
        res.prepend_nonuniform_scaling_mut(scaling);
        res
    }

    /// Computes the transformation equal to `self` followed by a translation.
    #[inline]
    #[must_use = "Did you mean to use append_translation_mut()?"]
    pub fn append_translation<SB>(&self, shift: &Vector<N, DimNameDiff<D, U1>, SB>) -> MatrixN<N, D>
    where
        D: DimNameSub<U1>,
        SB: Storage<N, DimNameDiff<D, U1>>,
        DefaultAllocator: Allocator<N, D, D>,
    {
        let mut res = self.clone_owned();
        res.append_translation_mut(shift);
        res
    }

    /// Computes the transformation equal to a translation followed by `self`.
    #[inline]
    #[must_use = "Did you mean to use prepend_translation_mut()?"]
    pub fn prepend_translation<SB>(
        &self,
        shift: &Vector<N, DimNameDiff<D, U1>, SB>,
    ) -> MatrixN<N, D>
    where
        D: DimNameSub<U1>,
        SB: Storage<N, DimNameDiff<D, U1>>,
        DefaultAllocator: Allocator<N, D, D> + Allocator<N, DimNameDiff<D, U1>>,
    {
        let mut res = self.clone_owned();
        res.prepend_translation_mut(shift);
        res
    }
}

impl<N: Scalar + Ring, D: DimName, S: StorageMut<N, D, D>> SquareMatrix<N, D, S> {
    /// Computes in-place the transformation equal to `self` followed by an uniform scaling factor.
    #[inline]
    pub fn append_scaling_mut(&mut self, scaling: N)
    where D: DimNameSub<U1> {
        let mut to_scale = self.fixed_rows_mut::<DimNameDiff<D, U1>>(0);
        to_scale *= scaling;
    }

    /// Computes in-place the transformation equal to an uniform scaling factor followed by `self`.
    #[inline]
    pub fn prepend_scaling_mut(&mut self, scaling: N)
    where D: DimNameSub<U1> {
        let mut to_scale = self.fixed_columns_mut::<DimNameDiff<D, U1>>(0);
        to_scale *= scaling;
    }

    /// Computes in-place the transformation equal to `self` followed by a non-uniform scaling factor.
    #[inline]
    pub fn append_nonuniform_scaling_mut<SB>(&mut self, scaling: &Vector<N, DimNameDiff<D, U1>, SB>)
    where
        D: DimNameSub<U1>,
        SB: Storage<N, DimNameDiff<D, U1>>,
    {
        for i in 0..scaling.len() {
            let mut to_scale = self.fixed_rows_mut::<U1>(i);
            to_scale *= scaling[i].inlined_clone();
        }
    }

    /// Computes in-place the transformation equal to a non-uniform scaling factor followed by `self`.
    #[inline]
    pub fn prepend_nonuniform_scaling_mut<SB>(
        &mut self,
        scaling: &Vector<N, DimNameDiff<D, U1>, SB>,
    ) where
        D: DimNameSub<U1>,
        SB: Storage<N, DimNameDiff<D, U1>>,
    {
        for i in 0..scaling.len() {
            let mut to_scale = self.fixed_columns_mut::<U1>(i);
            to_scale *= scaling[i].inlined_clone();
        }
    }

    /// Computes the transformation equal to `self` followed by a translation.
    #[inline]
    pub fn append_translation_mut<SB>(&mut self, shift: &Vector<N, DimNameDiff<D, U1>, SB>)
    where
        D: DimNameSub<U1>,
        SB: Storage<N, DimNameDiff<D, U1>>,
    {
        for i in 0..D::dim() {
            for j in 0..D::dim() - 1 {
                let add = shift[j].inlined_clone() * self[(D::dim() - 1, i)].inlined_clone();
                self[(j, i)] += add;
            }
        }
    }

    /// Computes the transformation equal to a translation followed by `self`.
    #[inline]
    pub fn prepend_translation_mut<SB>(&mut self, shift: &Vector<N, DimNameDiff<D, U1>, SB>)
    where
        D: DimNameSub<U1>,
        SB: Storage<N, DimNameDiff<D, U1>>,
        DefaultAllocator: Allocator<N, DimNameDiff<D, U1>>,
    {
        let scale = self
            .fixed_slice::<U1, DimNameDiff<D, U1>>(D::dim() - 1, 0)
            .tr_dot(&shift);
        let post_translation =
            self.fixed_slice::<DimNameDiff<D, U1>, DimNameDiff<D, U1>>(0, 0) * shift;

        self[(D::dim() - 1, D::dim() - 1)] += scale;

        let mut translation = self.fixed_slice_mut::<DimNameDiff<D, U1>, U1>(0, D::dim() - 1);
        translation += post_translation;
    }
}

impl<N: RealField, D: DimNameSub<U1>, S: Storage<N, D, D>> SquareMatrix<N, D, S>
where DefaultAllocator: Allocator<N, D, D>
        + Allocator<N, DimNameDiff<D, U1>>
        + Allocator<N, DimNameDiff<D, U1>, DimNameDiff<D, U1>>
{
    /// Transforms the given vector, assuming the matrix `self` uses homogeneous coordinates.
    #[inline]
    pub fn transform_vector(
        &self,
        v: &VectorN<N, DimNameDiff<D, U1>>,
    ) -> VectorN<N, DimNameDiff<D, U1>>
    {
        let transform = self.fixed_slice::<DimNameDiff<D, U1>, DimNameDiff<D, U1>>(0, 0);
        let normalizer = self.fixed_slice::<U1, DimNameDiff<D, U1>>(D::dim() - 1, 0);
        let n = normalizer.tr_dot(&v);

        if !n.is_zero() {
            return transform * (v / n);
        }

        transform * v
    }

    /// Transforms the given point, assuming the matrix `self` uses homogeneous coordinates.
    #[inline]
    pub fn transform_point(
        &self,
        pt: &Point<N, DimNameDiff<D, U1>>,
    ) -> Point<N, DimNameDiff<D, U1>>
    {
        let transform = self.fixed_slice::<DimNameDiff<D, U1>, DimNameDiff<D, U1>>(0, 0);
        let translation = self.fixed_slice::<DimNameDiff<D, U1>, U1>(0, D::dim() - 1);
        let normalizer = self.fixed_slice::<U1, DimNameDiff<D, U1>>(D::dim() - 1, 0);
        let n = normalizer.tr_dot(&pt.coords)
            + unsafe { *self.get_unchecked((D::dim() - 1, D::dim() - 1)) };

        if !n.is_zero() {
            (transform * pt + translation) / n
        } else {
            transform * pt + translation
        }
    }
}

impl<N: RealField, D: DimNameSub<U1>> Transformation<Point<N, DimNameDiff<D, U1>>> for MatrixN<N, D>
where DefaultAllocator: Allocator<N, D, D>
        + Allocator<N, DimNameDiff<D, U1>>
        + Allocator<N, DimNameDiff<D, U1>, DimNameDiff<D, U1>>
{
    #[inline]
    fn transform_vector(
        &self,
        v: &VectorN<N, DimNameDiff<D, U1>>,
    ) -> VectorN<N, DimNameDiff<D, U1>>
    {
        self.transform_vector(v)
    }

    #[inline]
    fn transform_point(&self, pt: &Point<N, DimNameDiff<D, U1>>) -> Point<N, DimNameDiff<D, U1>> {
        self.transform_point(pt)
    }
}
