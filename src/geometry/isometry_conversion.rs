use alga::general::{RealField, SubsetOf, SupersetOf};
use alga::linear::Rotation;

use crate::base::allocator::Allocator;
use crate::{DimMin, DimName, DimNameAdd, DimNameSum, U1};
use crate::base::{DefaultAllocator, MatrixN};

use crate::geometry::{Isometry, Point, Similarity, SuperTCategoryOf, TAffine, Transform, Translation};

/*
 * This file provides the following conversions:
 * =============================================
 *
 * Isometry -> Isometry
 * Isometry -> Similarity
 * Isometry -> Transform
 * Isometry -> Matrix (homogeneous)
 */

impl<N1, N2, D: DimName, R1, R2> SubsetOf<Isometry<N2, D, R2>> for Isometry<N1, D, R1>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
    R1: Rotation<Point<N1, D>> + SubsetOf<R2>,
    R2: Rotation<Point<N2, D>>,
    DefaultAllocator: Allocator<N1, D> + Allocator<N2, D>,
{
    #[inline]
    fn to_superset(&self) -> Isometry<N2, D, R2> {
        Isometry::from_parts(self.translation.to_superset(), self.rotation.to_superset())
    }

    #[inline]
    fn is_in_subset(iso: &Isometry<N2, D, R2>) -> bool {
        crate::is_convertible::<_, Translation<N1, D>>(&iso.translation)
            && crate::is_convertible::<_, R1>(&iso.rotation)
    }

    #[inline]
    unsafe fn from_superset_unchecked(iso: &Isometry<N2, D, R2>) -> Self {
        Isometry::from_parts(
            iso.translation.to_subset_unchecked(),
            iso.rotation.to_subset_unchecked(),
        )
    }
}

impl<N1, N2, D: DimName, R1, R2> SubsetOf<Similarity<N2, D, R2>> for Isometry<N1, D, R1>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
    R1: Rotation<Point<N1, D>> + SubsetOf<R2>,
    R2: Rotation<Point<N2, D>>,
    DefaultAllocator: Allocator<N1, D> + Allocator<N2, D>,
{
    #[inline]
    fn to_superset(&self) -> Similarity<N2, D, R2> {
        Similarity::from_isometry(self.to_superset(), N2::one())
    }

    #[inline]
    fn is_in_subset(sim: &Similarity<N2, D, R2>) -> bool {
        crate::is_convertible::<_, Isometry<N1, D, R1>>(&sim.isometry) && sim.scaling() == N2::one()
    }

    #[inline]
    unsafe fn from_superset_unchecked(sim: &Similarity<N2, D, R2>) -> Self {
        crate::convert_ref_unchecked(&sim.isometry)
    }
}

impl<N1, N2, D, R, C> SubsetOf<Transform<N2, D, C>> for Isometry<N1, D, R>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
    C: SuperTCategoryOf<TAffine>,
    R: Rotation<Point<N1, D>>
        + SubsetOf<MatrixN<N1, DimNameSum<D, U1>>>
        + SubsetOf<MatrixN<N2, DimNameSum<D, U1>>>,
    D: DimNameAdd<U1> + DimMin<D, Output = D>, // needed by .is_special_orthogonal()
    DefaultAllocator: Allocator<N1, D>
        + Allocator<N1, D, D>
        + Allocator<N1, DimNameSum<D, U1>, DimNameSum<D, U1>>
        + Allocator<N2, DimNameSum<D, U1>, DimNameSum<D, U1>>
        + Allocator<N2, DimNameSum<D, U1>, DimNameSum<D, U1>>
        + Allocator<(usize, usize), D>
        + Allocator<N2, D, D>
        + Allocator<N2, D>,
{
    #[inline]
    fn to_superset(&self) -> Transform<N2, D, C> {
        Transform::from_matrix_unchecked(self.to_homogeneous().to_superset())
    }

    #[inline]
    fn is_in_subset(t: &Transform<N2, D, C>) -> bool {
        <Self as SubsetOf<_>>::is_in_subset(t.matrix())
    }

    #[inline]
    unsafe fn from_superset_unchecked(t: &Transform<N2, D, C>) -> Self {
        Self::from_superset_unchecked(t.matrix())
    }
}

impl<N1, N2, D, R> SubsetOf<MatrixN<N2, DimNameSum<D, U1>>> for Isometry<N1, D, R>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
    R: Rotation<Point<N1, D>>
        + SubsetOf<MatrixN<N1, DimNameSum<D, U1>>>
        + SubsetOf<MatrixN<N2, DimNameSum<D, U1>>>,
    D: DimNameAdd<U1> + DimMin<D, Output = D>, // needed by .is_special_orthogonal()
    DefaultAllocator: Allocator<N1, D>
        + Allocator<N1, D, D>
        + Allocator<N1, DimNameSum<D, U1>, DimNameSum<D, U1>>
        + Allocator<N2, DimNameSum<D, U1>, DimNameSum<D, U1>>
        + Allocator<N2, DimNameSum<D, U1>, DimNameSum<D, U1>>
        + Allocator<(usize, usize), D>
        + Allocator<N2, D, D>
        + Allocator<N2, D>,
{
    #[inline]
    fn to_superset(&self) -> MatrixN<N2, DimNameSum<D, U1>> {
        self.to_homogeneous().to_superset()
    }

    #[inline]
    fn is_in_subset(m: &MatrixN<N2, DimNameSum<D, U1>>) -> bool {
        let rot = m.fixed_slice::<D, D>(0, 0);
        let bottom = m.fixed_slice::<U1, D>(D::dim(), 0);

        // Scalar types agree.
        m.iter().all(|e| SupersetOf::<N1>::is_in_subset(e)) &&
        // The block part is a rotation.
        rot.is_special_orthogonal(N2::default_epsilon() * crate::convert(100.0)) &&
        // The bottom row is (0, 0, ..., 1)
        bottom.iter().all(|e| e.is_zero()) && m[(D::dim(), D::dim())] == N2::one()
    }

    #[inline]
    unsafe fn from_superset_unchecked(m: &MatrixN<N2, DimNameSum<D, U1>>) -> Self {
        let t = m.fixed_slice::<D, U1>(0, D::dim()).into_owned();
        let t = Translation {
            vector: crate::convert_unchecked(t),
        };

        Self::from_parts(t, crate::convert_unchecked(m.clone_owned()))
    }
}

impl<N: RealField, D: DimName, R> From<Isometry<N, D, R>> for MatrixN<N, DimNameSum<D, U1>>
where
    D: DimNameAdd<U1>,
    R: SubsetOf<MatrixN<N, DimNameSum<D, U1>>>,
    DefaultAllocator: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>> + Allocator<N, D>,
{
    #[inline]
    fn from(iso: Isometry<N, D, R>) -> Self {
        iso.to_homogeneous()
    }
}
