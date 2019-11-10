use num::Zero;

use alga::general::{RealField, SubsetOf, SupersetOf};
use alga::linear::Rotation as AlgaRotation;

#[cfg(feature = "mint")]
use mint;

use crate::U3;
use crate::base::{Matrix3, Matrix4, Vector4};
use crate::geometry::{
    Isometry, Point3, Quaternion, Rotation, Rotation3, Similarity, SuperTCategoryOf, TAffine,
    Transform, Translation, UnitQuaternion,
};

/*
 * This file provides the following conversions:
 * =============================================
 *
 * Quaternion     -> Quaternion
 * UnitQuaternion -> UnitQuaternion
 * UnitQuaternion -> Rotation<U3>
 * UnitQuaternion -> Isometry<U3>
 * UnitQuaternion -> Similarity<U3>
 * UnitQuaternion -> Transform<U3>
 * UnitQuaternion -> Matrix<U4> (homogeneous)
 *
 * mint::Quaternion <-> Quaternion
 * UnitQuaternion -> mint::Quaternion
 *
 * NOTE:
 * UnitQuaternion -> Quaternion is already provided by: Unit<T> -> T
 */

impl<N1, N2> SubsetOf<Quaternion<N2>> for Quaternion<N1>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
{
    #[inline]
    fn to_superset(&self) -> Quaternion<N2> {
        Quaternion::from(self.coords.to_superset())
    }

    #[inline]
    fn is_in_subset(q: &Quaternion<N2>) -> bool {
        crate::is_convertible::<_, Vector4<N1>>(&q.coords)
    }

    #[inline]
    unsafe fn from_superset_unchecked(q: &Quaternion<N2>) -> Self {
        Self {
            coords: q.coords.to_subset_unchecked(),
        }
    }
}

impl<N1, N2> SubsetOf<UnitQuaternion<N2>> for UnitQuaternion<N1>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
{
    #[inline]
    fn to_superset(&self) -> UnitQuaternion<N2> {
        UnitQuaternion::new_unchecked(self.as_ref().to_superset())
    }

    #[inline]
    fn is_in_subset(uq: &UnitQuaternion<N2>) -> bool {
        crate::is_convertible::<_, Quaternion<N1>>(uq.as_ref())
    }

    #[inline]
    unsafe fn from_superset_unchecked(uq: &UnitQuaternion<N2>) -> Self {
        Self::new_unchecked(crate::convert_ref_unchecked(uq.as_ref()))
    }
}

impl<N1, N2> SubsetOf<Rotation<N2, U3>> for UnitQuaternion<N1>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
{
    #[inline]
    fn to_superset(&self) -> Rotation3<N2> {
        let q: UnitQuaternion<N2> = self.to_superset();
        q.to_rotation_matrix()
    }

    #[inline]
    fn is_in_subset(rot: &Rotation3<N2>) -> bool {
        crate::is_convertible::<_, Rotation3<N1>>(rot)
    }

    #[inline]
    unsafe fn from_superset_unchecked(rot: &Rotation3<N2>) -> Self {
        let q = UnitQuaternion::<N2>::from_rotation_matrix(rot);
        crate::convert_unchecked(q)
    }
}

impl<N1, N2, R> SubsetOf<Isometry<N2, U3, R>> for UnitQuaternion<N1>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
    R: AlgaRotation<Point3<N2>> + SupersetOf<Self>,
{
    #[inline]
    fn to_superset(&self) -> Isometry<N2, U3, R> {
        Isometry::from_parts(Translation::identity(), crate::convert_ref(self))
    }

    #[inline]
    fn is_in_subset(iso: &Isometry<N2, U3, R>) -> bool {
        iso.translation.vector.is_zero()
    }

    #[inline]
    unsafe fn from_superset_unchecked(iso: &Isometry<N2, U3, R>) -> Self {
        crate::convert_ref_unchecked(&iso.rotation)
    }
}

impl<N1, N2, R> SubsetOf<Similarity<N2, U3, R>> for UnitQuaternion<N1>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
    R: AlgaRotation<Point3<N2>> + SupersetOf<Self>,
{
    #[inline]
    fn to_superset(&self) -> Similarity<N2, U3, R> {
        Similarity::from_isometry(crate::convert_ref(self), N2::one())
    }

    #[inline]
    fn is_in_subset(sim: &Similarity<N2, U3, R>) -> bool {
        sim.isometry.translation.vector.is_zero() && sim.scaling() == N2::one()
    }

    #[inline]
    unsafe fn from_superset_unchecked(sim: &Similarity<N2, U3, R>) -> Self {
        crate::convert_ref_unchecked(&sim.isometry)
    }
}

impl<N1, N2, C> SubsetOf<Transform<N2, U3, C>> for UnitQuaternion<N1>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
    C: SuperTCategoryOf<TAffine>,
{
    #[inline]
    fn to_superset(&self) -> Transform<N2, U3, C> {
        Transform::from_matrix_unchecked(self.to_homogeneous().to_superset())
    }

    #[inline]
    fn is_in_subset(t: &Transform<N2, U3, C>) -> bool {
        <Self as SubsetOf<_>>::is_in_subset(t.matrix())
    }

    #[inline]
    unsafe fn from_superset_unchecked(t: &Transform<N2, U3, C>) -> Self {
        Self::from_superset_unchecked(t.matrix())
    }
}

impl<N1: RealField, N2: RealField + SupersetOf<N1>> SubsetOf<Matrix4<N2>> for UnitQuaternion<N1> {
    #[inline]
    fn to_superset(&self) -> Matrix4<N2> {
        self.to_homogeneous().to_superset()
    }

    #[inline]
    fn is_in_subset(m: &Matrix4<N2>) -> bool {
        crate::is_convertible::<_, Rotation3<N1>>(m)
    }

    #[inline]
    unsafe fn from_superset_unchecked(m: &Matrix4<N2>) -> Self {
        let rot: Rotation3<N1> = crate::convert_ref_unchecked(m);
        Self::from_rotation_matrix(&rot)
    }
}

#[cfg(feature = "mint")]
impl<N: RealField> From<mint::Quaternion<N>> for Quaternion<N> {
    fn from(q: mint::Quaternion<N>) -> Self {
        Self::new(q.s, q.v.x, q.v.y, q.v.z)
    }
}

#[cfg(feature = "mint")]
impl<N: RealField> Into<mint::Quaternion<N>> for Quaternion<N> {
    fn into(self) -> mint::Quaternion<N> {
        mint::Quaternion {
            v: mint::Vector3 {
                x: self[0],
                y: self[1],
                z: self[2],
            },
            s: self[3],
        }
    }
}

#[cfg(feature = "mint")]
impl<N: RealField> Into<mint::Quaternion<N>> for UnitQuaternion<N> {
    fn into(self) -> mint::Quaternion<N> {
        mint::Quaternion {
            v: mint::Vector3 {
                x: self[0],
                y: self[1],
                z: self[2],
            },
            s: self[3],
        }
    }
}

impl<N: RealField> From<UnitQuaternion<N>> for Matrix4<N> {
    #[inline]
    fn from(q: UnitQuaternion<N>) -> Self {
        q.to_homogeneous()
    }
}

impl<N: RealField> From<UnitQuaternion<N>> for Rotation3<N> {
    #[inline]
    fn from(q: UnitQuaternion<N>) -> Self {
        q.to_rotation_matrix()
    }
}

impl<N: RealField> From<Rotation3<N>> for UnitQuaternion<N> {
    #[inline]
    fn from(q: Rotation3<N>) -> Self {
        Self::from_rotation_matrix(&q)
    }
}

impl<N: RealField> From<UnitQuaternion<N>> for Matrix3<N> {
    #[inline]
    fn from(q: UnitQuaternion<N>) -> Self {
        q.to_rotation_matrix().into_inner()
    }
}

impl<N: RealField> From<Vector4<N>> for Quaternion<N> {
    #[inline]
    fn from(coords: Vector4<N>) -> Self {
        Self { coords }
    }
}
