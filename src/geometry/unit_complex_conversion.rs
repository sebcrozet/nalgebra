use num::Zero;
use num_complex::Complex;

use alga::general::{RealField, SubsetOf, SupersetOf};
use alga::linear::Rotation as AlgaRotation;

use crate::U2;
use crate::base::{Matrix2, Matrix3};
use crate::geometry::{
    Isometry, Point2, Rotation2, Similarity, SuperTCategoryOf, TAffine, Transform, Translation,
    UnitComplex
};

/*
 * This file provides the following conversions:
 * =============================================
 *
 * UnitComplex -> UnitComplex
 * UnitComplex -> Rotation<U1>
 * UnitComplex -> Isometry<U2>
 * UnitComplex -> Similarity<U2>
 * UnitComplex -> Transform<U2>
 * UnitComplex -> Matrix<U3> (homogeneous)
 *
 * NOTE:
 * UnitComplex -> Complex is already provided by: Unit<T> -> T
 */

impl<N1, N2> SubsetOf<UnitComplex<N2>> for UnitComplex<N1>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
{
    #[inline]
    fn to_superset(&self) -> UnitComplex<N2> {
        UnitComplex::new_unchecked(self.as_ref().to_superset())
    }

    #[inline]
    fn is_in_subset(uq: &UnitComplex<N2>) -> bool {
        crate::is_convertible::<_, Complex<N1>>(uq.as_ref())
    }

    #[inline]
    unsafe fn from_superset_unchecked(uq: &UnitComplex<N2>) -> Self {
        Self::new_unchecked(crate::convert_ref_unchecked(uq.as_ref()))
    }
}

impl<N1, N2> SubsetOf<Rotation2<N2>> for UnitComplex<N1>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
{
    #[inline]
    fn to_superset(&self) -> Rotation2<N2> {
        let q: UnitComplex<N2> = self.to_superset();
        q.to_rotation_matrix().to_superset()
    }

    #[inline]
    fn is_in_subset(rot: &Rotation2<N2>) -> bool {
        crate::is_convertible::<_, Rotation2<N1>>(rot)
    }

    #[inline]
    unsafe fn from_superset_unchecked(rot: &Rotation2<N2>) -> Self {
        let q = UnitComplex::<N2>::from_rotation_matrix(rot);
        crate::convert_unchecked(q)
    }
}

impl<N1, N2, R> SubsetOf<Isometry<N2, U2, R>> for UnitComplex<N1>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
    R: AlgaRotation<Point2<N2>> + SupersetOf<Self>,
{
    #[inline]
    fn to_superset(&self) -> Isometry<N2, U2, R> {
        Isometry::from_parts(Translation::identity(), crate::convert_ref(self))
    }

    #[inline]
    fn is_in_subset(iso: &Isometry<N2, U2, R>) -> bool {
        iso.translation.vector.is_zero()
    }

    #[inline]
    unsafe fn from_superset_unchecked(iso: &Isometry<N2, U2, R>) -> Self {
        crate::convert_ref_unchecked(&iso.rotation)
    }
}

impl<N1, N2, R> SubsetOf<Similarity<N2, U2, R>> for UnitComplex<N1>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
    R: AlgaRotation<Point2<N2>> + SupersetOf<Self>,
{
    #[inline]
    fn to_superset(&self) -> Similarity<N2, U2, R> {
        Similarity::from_isometry(crate::convert_ref(self), N2::one())
    }

    #[inline]
    fn is_in_subset(sim: &Similarity<N2, U2, R>) -> bool {
        sim.isometry.translation.vector.is_zero() && sim.scaling() == N2::one()
    }

    #[inline]
    unsafe fn from_superset_unchecked(sim: &Similarity<N2, U2, R>) -> Self {
        crate::convert_ref_unchecked(&sim.isometry)
    }
}

impl<N1, N2, C> SubsetOf<Transform<N2, U2, C>> for UnitComplex<N1>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
    C: SuperTCategoryOf<TAffine>,
{
    #[inline]
    fn to_superset(&self) -> Transform<N2, U2, C> {
        Transform::from_matrix_unchecked(self.to_homogeneous().to_superset())
    }

    #[inline]
    fn is_in_subset(t: &Transform<N2, U2, C>) -> bool {
        <Self as SubsetOf<_>>::is_in_subset(t.matrix())
    }

    #[inline]
    unsafe fn from_superset_unchecked(t: &Transform<N2, U2, C>) -> Self {
        Self::from_superset_unchecked(t.matrix())
    }
}

impl<N1: RealField, N2: RealField + SupersetOf<N1>> SubsetOf<Matrix3<N2>> for UnitComplex<N1> {
    #[inline]
    fn to_superset(&self) -> Matrix3<N2> {
        self.to_homogeneous().to_superset()
    }

    #[inline]
    fn is_in_subset(m: &Matrix3<N2>) -> bool {
        crate::is_convertible::<_, Rotation2<N1>>(m)
    }

    #[inline]
    unsafe fn from_superset_unchecked(m: &Matrix3<N2>) -> Self {
        let rot: Rotation2<N1> = crate::convert_ref_unchecked(m);
        Self::from_rotation_matrix(&rot)
    }
}


impl<N: RealField> From<UnitComplex<N>> for Rotation2<N> {
    #[inline]
    fn from(q: UnitComplex<N>) -> Self {
        q.to_rotation_matrix()
    }
}

impl<N: RealField> From<Rotation2<N>> for UnitComplex<N> {
    #[inline]
    fn from(q: Rotation2<N>) -> Self {
        Self::from_rotation_matrix(&q)
    }
}

impl<N: RealField> From<UnitComplex<N>> for Matrix3<N> {
    #[inline]
    fn from(q: UnitComplex<N>) -> Matrix3<N> {
        q.to_homogeneous()
    }
}

impl<N: RealField> From<UnitComplex<N>> for Matrix2<N> {
    #[inline]
    fn from(q: UnitComplex<N>) -> Self {
        q.to_rotation_matrix().into_inner()
    }
}
