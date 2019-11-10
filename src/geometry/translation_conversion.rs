use num::{One, Zero};

use alga::general::{RealField, SubsetOf, SupersetOf};
use alga::linear::Rotation;

use crate::base::allocator::Allocator;
use crate::{DimName, DimNameAdd, DimNameSum, U1};
use crate::base::{DefaultAllocator, MatrixN, Scalar, VectorN};

use crate::geometry::{Isometry, Point, Similarity, SuperTCategoryOf, TAffine, Transform, Translation};

/*
 * This file provides the following conversions:
 * =============================================
 *
 * Translation -> Translation
 * Translation -> Isometry
 * Translation -> Similarity
 * Translation -> Transform
 * Translation -> Matrix (homogeneous)
 */

impl<N1, N2, D: DimName> SubsetOf<Translation<N2, D>> for Translation<N1, D>
where
    N1: Scalar,
    N2: Scalar + SupersetOf<N1>,
    DefaultAllocator: Allocator<N1, D> + Allocator<N2, D>,
{
    #[inline]
    fn to_superset(&self) -> Translation<N2, D> {
        Translation::from(self.vector.to_superset())
    }

    #[inline]
    fn is_in_subset(rot: &Translation<N2, D>) -> bool {
        crate::is_convertible::<_, VectorN<N1, D>>(&rot.vector)
    }

    #[inline]
    unsafe fn from_superset_unchecked(rot: &Translation<N2, D>) -> Self {
        Translation {
            vector: rot.vector.to_subset_unchecked(),
        }
    }
}

impl<N1, N2, D: DimName, R> SubsetOf<Isometry<N2, D, R>> for Translation<N1, D>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
    R: Rotation<Point<N2, D>>,
    DefaultAllocator: Allocator<N1, D> + Allocator<N2, D>,
{
    #[inline]
    fn to_superset(&self) -> Isometry<N2, D, R> {
        Isometry::from_parts(self.to_superset(), R::identity())
    }

    #[inline]
    fn is_in_subset(iso: &Isometry<N2, D, R>) -> bool {
        iso.rotation == R::identity()
    }

    #[inline]
    unsafe fn from_superset_unchecked(iso: &Isometry<N2, D, R>) -> Self {
        Self::from_superset_unchecked(&iso.translation)
    }
}

impl<N1, N2, D: DimName, R> SubsetOf<Similarity<N2, D, R>> for Translation<N1, D>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
    R: Rotation<Point<N2, D>>,
    DefaultAllocator: Allocator<N1, D> + Allocator<N2, D>,
{
    #[inline]
    fn to_superset(&self) -> Similarity<N2, D, R> {
        Similarity::from_parts(self.to_superset(), R::identity(), N2::one())
    }

    #[inline]
    fn is_in_subset(sim: &Similarity<N2, D, R>) -> bool {
        sim.isometry.rotation == R::identity() && sim.scaling() == N2::one()
    }

    #[inline]
    unsafe fn from_superset_unchecked(sim: &Similarity<N2, D, R>) -> Self {
        Self::from_superset_unchecked(&sim.isometry.translation)
    }
}

impl<N1, N2, D, C> SubsetOf<Transform<N2, D, C>> for Translation<N1, D>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
    C: SuperTCategoryOf<TAffine>,
    D: DimNameAdd<U1>,
    DefaultAllocator: Allocator<N1, D>
        + Allocator<N2, D>
        + Allocator<N1, DimNameSum<D, U1>, DimNameSum<D, U1>>
        + Allocator<N2, DimNameSum<D, U1>, DimNameSum<D, U1>>,
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

impl<N1, N2, D> SubsetOf<MatrixN<N2, DimNameSum<D, U1>>> for Translation<N1, D>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
    D: DimNameAdd<U1>,
    DefaultAllocator: Allocator<N1, D>
        + Allocator<N2, D>
        + Allocator<N1, DimNameSum<D, U1>, DimNameSum<D, U1>>
        + Allocator<N2, DimNameSum<D, U1>, DimNameSum<D, U1>>,
{
    #[inline]
    fn to_superset(&self) -> MatrixN<N2, DimNameSum<D, U1>> {
        self.to_homogeneous().to_superset()
    }

    #[inline]
    fn is_in_subset(m: &MatrixN<N2, DimNameSum<D, U1>>) -> bool {
        let id = m.fixed_slice::<DimNameSum<D, U1>, D>(0, 0);

        // Scalar types agree.
        m.iter().all(|e| SupersetOf::<N1>::is_in_subset(e)) &&
        // The block part does nothing.
        id.is_identity(N2::zero()) &&
        // The normalization factor is one.
        m[(D::dim(), D::dim())] == N2::one()
    }

    #[inline]
    unsafe fn from_superset_unchecked(m: &MatrixN<N2, DimNameSum<D, U1>>) -> Self {
        let t = m.fixed_slice::<D, U1>(0, D::dim());
        Self {
            vector: crate::convert_unchecked(t.into_owned()),
        }
    }
}

impl<N: Scalar + Zero + One, D: DimName> From<Translation<N, D>> for MatrixN<N, DimNameSum<D, U1>>
where
    D: DimNameAdd<U1>,
    DefaultAllocator: Allocator<N, D> + Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>>,
{
    #[inline]
    fn from(t: Translation<N, D>) -> Self {
        t.to_homogeneous()
    }
}

impl<N: Scalar, D: DimName> From<VectorN<N, D>> for Translation<N, D>
where DefaultAllocator: Allocator<N, D>
{
    #[inline]
    fn from(vector: VectorN<N, D>) -> Self {
        Translation { vector }
    }
}
