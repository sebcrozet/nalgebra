use alga::general::{Field, JoinSemilattice, Lattice, MeetSemilattice, RealField};
use alga::linear::{AffineSpace, EuclideanSpace};

use crate::base::allocator::Allocator;
use crate::DimName;
use crate::base::{DefaultAllocator, Scalar, VectorN};

use crate::geometry::Point;

impl<N: Scalar + Field, D: DimName> AffineSpace for Point<N, D>
where
    N: Scalar + Field,
    DefaultAllocator: Allocator<N, D>,
{
    type Translation = VectorN<N, D>;
}

impl<N: RealField, D: DimName> EuclideanSpace for Point<N, D>
where DefaultAllocator: Allocator<N, D>
{
    type Coordinates = VectorN<N, D>;
    type RealField = N;

    #[inline]
    fn origin() -> Self {
        Self::origin()
    }

    #[inline]
    fn coordinates(&self) -> Self::Coordinates {
        self.coords.clone()
    }

    #[inline]
    fn from_coordinates(coords: Self::Coordinates) -> Self {
        Self::from(coords)
    }

    #[inline]
    fn scale_by(&self, n: N) -> Self {
        self * n
    }
}

/*
 *
 * Ordering
 *
 */
impl<N, D: DimName> MeetSemilattice for Point<N, D>
where
    N: Scalar + MeetSemilattice,
    DefaultAllocator: Allocator<N, D>,
{
    #[inline]
    fn meet(&self, other: &Self) -> Self {
        Self::from(self.coords.meet(&other.coords))
    }
}

impl<N, D: DimName> JoinSemilattice for Point<N, D>
where
    N: Scalar + JoinSemilattice,
    DefaultAllocator: Allocator<N, D>,
{
    #[inline]
    fn join(&self, other: &Self) -> Self {
        Self::from(self.coords.join(&other.coords))
    }
}

impl<N, D: DimName> Lattice for Point<N, D>
where
    N: Scalar + Lattice,
    DefaultAllocator: Allocator<N, D>,
{
    #[inline]
    fn meet_join(&self, other: &Self) -> (Self, Self) {
        let (meet, join) = self.coords.meet_join(&other.coords);

        (Self::from(meet), Self::from(join))
    }
}
