use alga::general::{RealField, SubsetOf};

use crate::base::allocator::Allocator;
use crate::{DimName, DimNameAdd, DimNameSum, U1};
use crate::base::{DefaultAllocator, MatrixN};

use crate::geometry::{SuperTCategoryOf, TCategory, Transform};

impl<N1, N2, D: DimName, C1, C2> SubsetOf<Transform<N2, D, C2>> for Transform<N1, D, C1>
where
    N1: RealField + SubsetOf<N2>,
    N2: RealField,
    C1: TCategory,
    C2: SuperTCategoryOf<C1>,
    D: DimNameAdd<U1>,
    DefaultAllocator: Allocator<N1, DimNameSum<D, U1>, DimNameSum<D, U1>>
        + Allocator<N2, DimNameSum<D, U1>, DimNameSum<D, U1>>,
    N1::Epsilon: Copy,
    N2::Epsilon: Copy,
{
    #[inline]
    fn to_superset(&self) -> Transform<N2, D, C2> {
        Transform::from_matrix_unchecked(self.to_homogeneous().to_superset())
    }

    #[inline]
    fn is_in_subset(t: &Transform<N2, D, C2>) -> bool {
        <Self as SubsetOf<_>>::is_in_subset(t.matrix())
    }

    #[inline]
    unsafe fn from_superset_unchecked(t: &Transform<N2, D, C2>) -> Self {
        Self::from_superset_unchecked(t.matrix())
    }
}

impl<N1, N2, D: DimName, C> SubsetOf<MatrixN<N2, DimNameSum<D, U1>>> for Transform<N1, D, C>
where
    N1: RealField + SubsetOf<N2>,
    N2: RealField,
    C: TCategory,
    D: DimNameAdd<U1>,
    DefaultAllocator: Allocator<N1, DimNameSum<D, U1>, DimNameSum<D, U1>>
        + Allocator<N2, DimNameSum<D, U1>, DimNameSum<D, U1>>,
    N1::Epsilon: Copy,
    N2::Epsilon: Copy,
{
    #[inline]
    fn to_superset(&self) -> MatrixN<N2, DimNameSum<D, U1>> {
        self.matrix().to_superset()
    }

    #[inline]
    fn is_in_subset(m: &MatrixN<N2, DimNameSum<D, U1>>) -> bool {
        C::check_homogeneous_invariants(m)
    }

    #[inline]
    unsafe fn from_superset_unchecked(m: &MatrixN<N2, DimNameSum<D, U1>>) -> Self {
        Self::from_matrix_unchecked(crate::convert_ref_unchecked(m))
    }
}

impl<N: RealField, D: DimName, C> From<Transform<N, D, C>> for MatrixN<N, DimNameSum<D, U1>>
where
    D: DimNameAdd<U1>,
    C: TCategory,
    DefaultAllocator: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>>,
{
    #[inline]
    fn from(t: Transform<N, D, C>) -> Self {
        t.to_homogeneous()
    }
}
