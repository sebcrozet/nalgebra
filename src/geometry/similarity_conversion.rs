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
 * Similarity -> Similarity
 * Similarity -> Transform
 * Similarity -> Matrix (homogeneous)
 */

impl<N1, N2, D: DimName, R1, R2> SubsetOf<Similarity<N2, D, R2>> for Similarity<N1, D, R1>
where
    N1: RealField + SubsetOf<N2>,
    N2: RealField + SupersetOf<N1>,
    R1: Rotation<Point<N1, D>> + SubsetOf<R2>,
    R2: Rotation<Point<N2, D>>,
    DefaultAllocator: Allocator<N1, D> + Allocator<N2, D>,
{
    #[inline]
    fn to_superset(&self) -> Similarity<N2, D, R2> {
        Similarity::from_isometry(self.isometry.to_superset(), self.scaling().to_superset())
    }

    #[inline]
    fn is_in_subset(sim: &Similarity<N2, D, R2>) -> bool {
        crate::is_convertible::<_, Isometry<N1, D, R1>>(&sim.isometry)
            && crate::is_convertible::<_, N1>(&sim.scaling())
    }

    #[inline]
    unsafe fn from_superset_unchecked(sim: &Similarity<N2, D, R2>) -> Self {
        Similarity::from_isometry(
            sim.isometry.to_subset_unchecked(),
            sim.scaling().to_subset_unchecked(),
        )
    }
}

impl<N1, N2, D, R, C> SubsetOf<Transform<N2, D, C>> for Similarity<N1, D, R>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
    C: SuperTCategoryOf<TAffine>,
    R: Rotation<Point<N1, D>>
        + SubsetOf<MatrixN<N1, DimNameSum<D, U1>>>
        + SubsetOf<MatrixN<N2, DimNameSum<D, U1>>>,
    D: DimNameAdd<U1> + DimMin<D, Output = D>, // needed by .determinant()
    DefaultAllocator: Allocator<N1, D>
        + Allocator<N1, D, D>
        + Allocator<N1, DimNameSum<D, U1>, DimNameSum<D, U1>>
        + Allocator<N2, DimNameSum<D, U1>, DimNameSum<D, U1>>
        + Allocator<(usize, usize), D>
        + Allocator<N2, DimNameSum<D, U1>, DimNameSum<D, U1>>
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

impl<N1, N2, D, R> SubsetOf<MatrixN<N2, DimNameSum<D, U1>>> for Similarity<N1, D, R>
where
    N1: RealField,
    N2: RealField + SupersetOf<N1>,
    R: Rotation<Point<N1, D>>
        + SubsetOf<MatrixN<N1, DimNameSum<D, U1>>>
        + SubsetOf<MatrixN<N2, DimNameSum<D, U1>>>,
    D: DimNameAdd<U1> + DimMin<D, Output = D>, // needed by .determinant()
    DefaultAllocator: Allocator<N1, D>
        + Allocator<N1, D, D>
        + Allocator<N1, DimNameSum<D, U1>, DimNameSum<D, U1>>
        + Allocator<N2, DimNameSum<D, U1>, DimNameSum<D, U1>>
        + Allocator<(usize, usize), D>
        + Allocator<N2, DimNameSum<D, U1>, DimNameSum<D, U1>>
        + Allocator<N2, D, D>
        + Allocator<N2, D>,
{
    #[inline]
    fn to_superset(&self) -> MatrixN<N2, DimNameSum<D, U1>> {
        self.to_homogeneous().to_superset()
    }

    #[inline]
    fn is_in_subset(m: &MatrixN<N2, DimNameSum<D, U1>>) -> bool {
        let mut rot = m.fixed_slice::<D, D>(0, 0).clone_owned();
        if rot
            .fixed_columns_mut::<U1>(0)
            .try_normalize_mut(N2::zero())
            .is_some()
            && rot
                .fixed_columns_mut::<U1>(1)
                .try_normalize_mut(N2::zero())
                .is_some()
            && rot
                .fixed_columns_mut::<U1>(2)
                .try_normalize_mut(N2::zero())
                .is_some()
        {
            // FIXME: could we avoid explicit the computation of the determinant?
            // (its sign is needed to see if the scaling factor is negative).
            if rot.determinant() < N2::zero() {
                rot.fixed_columns_mut::<U1>(0).neg_mut();
                rot.fixed_columns_mut::<U1>(1).neg_mut();
                rot.fixed_columns_mut::<U1>(2).neg_mut();
            }

            let bottom = m.fixed_slice::<U1, D>(D::dim(), 0);
            // Scalar types agree.
            m.iter().all(|e| SupersetOf::<N1>::is_in_subset(e)) &&
            // The normalized block part is a rotation.
            // rot.is_special_orthogonal(N2::default_epsilon().sqrt()) &&
            // The bottom row is (0, 0, ..., 1)
            bottom.iter().all(|e| e.is_zero()) && m[(D::dim(), D::dim())] == N2::one()
        } else {
            false
        }
    }

    #[inline]
    unsafe fn from_superset_unchecked(m: &MatrixN<N2, DimNameSum<D, U1>>) -> Self {
        let mut mm = m.clone_owned();
        let na = mm.fixed_slice_mut::<D, U1>(0, 0).normalize_mut();
        let nb = mm.fixed_slice_mut::<D, U1>(0, 1).normalize_mut();
        let nc = mm.fixed_slice_mut::<D, U1>(0, 2).normalize_mut();

        let mut scale = (na + nb + nc) / crate::convert(3.0); // We take the mean, for robustness.

        // FIXME: could we avoid the explicit computation of the determinant?
        // (its sign is needed to see if the scaling factor is negative).
        if mm.fixed_slice::<D, D>(0, 0).determinant() < N2::zero() {
            mm.fixed_slice_mut::<D, U1>(0, 0).neg_mut();
            mm.fixed_slice_mut::<D, U1>(0, 1).neg_mut();
            mm.fixed_slice_mut::<D, U1>(0, 2).neg_mut();
            scale = -scale;
        }

        let t = m.fixed_slice::<D, U1>(0, D::dim()).into_owned();
        let t = Translation {
            vector: crate::convert_unchecked(t),
        };

        Self::from_parts(t, crate::convert_unchecked(mm), crate::convert_unchecked(scale))
    }
}

impl<N: RealField, D: DimName, R> From<Similarity<N, D, R>> for MatrixN<N, DimNameSum<D, U1>>
where
    D: DimNameAdd<U1>,
    R: SubsetOf<MatrixN<N, DimNameSum<D, U1>>>,
    DefaultAllocator: Allocator<N, D> + Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>>,
{
    #[inline]
    fn from(sim: Similarity<N, D, R>) -> Self {
        sim.to_homogeneous()
    }
}
