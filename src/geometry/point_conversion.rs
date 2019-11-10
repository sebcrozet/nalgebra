use alga::general::{ClosedDiv, SubsetOf, SupersetOf};
use num::{One, Zero};

use crate::base::allocator::Allocator;
use crate::{DimName, DimNameAdd, DimNameSum, U1};
use crate::base::{DefaultAllocator, Matrix, Scalar, VectorN};

#[cfg(feature = "mint")]
use crate::{U2, U3};
#[cfg(feature = "mint")]
use crate::base::storage::{Storage, StorageMut};
use crate::geometry::Point;
#[cfg(feature = "mint")]
use mint;
#[cfg(feature = "mint")]
use std::convert::{AsMut, AsRef, From, Into};
/*
 * This file provides the following conversions:
 * =============================================
 *
 * Point -> Point
 * Point -> Vector (homogeneous)
 *
 * mint::Point <-> Point
 */

impl<N1, N2, D> SubsetOf<Point<N2, D>> for Point<N1, D>
where
    D: DimName,
    N1: Scalar,
    N2: Scalar + SupersetOf<N1>,
    DefaultAllocator: Allocator<N2, D> + Allocator<N1, D>,
{
    #[inline]
    fn to_superset(&self) -> Point<N2, D> {
        Point::from(self.coords.to_superset())
    }

    #[inline]
    fn is_in_subset(m: &Point<N2, D>) -> bool {
        // FIXME: is there a way to reuse the `.is_in_subset` from the matrix implementation of
        // SubsetOf?
        m.iter().all(|e| e.is_in_subset())
    }

    #[inline]
    unsafe fn from_superset_unchecked(m: &Point<N2, D>) -> Self {
        Self::from(Matrix::from_superset_unchecked(&m.coords))
    }
}

impl<N1, N2, D> SubsetOf<VectorN<N2, DimNameSum<D, U1>>> for Point<N1, D>
where
    D: DimNameAdd<U1>,
    N1: Scalar,
    N2: Scalar + Zero + One + ClosedDiv + SupersetOf<N1>,
    DefaultAllocator: Allocator<N1, D>
        + Allocator<N1, DimNameSum<D, U1>>
        + Allocator<N2, DimNameSum<D, U1>>
        + Allocator<N2, D>,
{
    #[inline]
    fn to_superset(&self) -> VectorN<N2, DimNameSum<D, U1>> {
        let p: Point<N2, D> = self.to_superset();
        p.to_homogeneous()
    }

    #[inline]
    fn is_in_subset(v: &VectorN<N2, DimNameSum<D, U1>>) -> bool {
        crate::is_convertible::<_, VectorN<N1, DimNameSum<D, U1>>>(v) && !v[D::dim()].is_zero()
    }

    #[inline]
    unsafe fn from_superset_unchecked(v: &VectorN<N2, DimNameSum<D, U1>>) -> Self {
        let coords = v.fixed_slice::<D, U1>(0, 0) / v[D::dim()];
        Self {
            coords: crate::convert_unchecked(coords)
        }
    }
}

#[cfg(feature = "mint")]
macro_rules! impl_from_into_mint_1D(
    ($($NRows: ident => $PT:ident, $VT:ident [$SZ: expr]);* $(;)*) => {$(
        impl<N> From<mint::$PT<N>> for Point<N, $NRows>
        where N: Scalar {
            #[inline]
            fn from(p: mint::$PT<N>) -> Self {
                Self {
                    coords: VectorN::from(mint::$VT::from(p)),
                }
            }
        }

        impl<N> Into<mint::$PT<N>> for Point<N, $NRows>
        where N: Scalar {
            #[inline]
            fn into(self) -> mint::$PT<N> {
                let mint_vec: mint::$VT<N> = self.coords.into();
                mint::$PT::from(mint_vec)
            }
        }

        impl<N> AsRef<mint::$PT<N>> for Point<N, $NRows>
        where N: Scalar {
            #[inline]
            fn as_ref(&self) -> &mint::$PT<N> {
                unsafe {
                    &*(self.coords.data.ptr() as *const mint::$PT<N>)
                }
            }
        }

        impl<N> AsMut<mint::$PT<N>> for Point<N, $NRows>
        where N: Scalar {
            #[inline]
            fn as_mut(&mut self) -> &mut mint::$PT<N> {
                unsafe {
                    &mut *(self.coords.data.ptr_mut() as *mut mint::$PT<N>)
                }
            }
        }
    )*}
);

// Implement for points of dimension 2, 3.
#[cfg(feature = "mint")]
impl_from_into_mint_1D!(
    U2 => Point2, Vector2[2];
    U3 => Point3, Vector3[3];
);

impl<N: Scalar + Zero + One, D: DimName> From<Point<N, D>> for VectorN<N, DimNameSum<D, U1>>
where
    D: DimNameAdd<U1>,
    DefaultAllocator: Allocator<N, D> + Allocator<N, DimNameSum<D, U1>>,
{
    #[inline]
    fn from(t: Point<N, D>) -> Self {
        t.to_homogeneous()
    }
}

impl<N: Scalar, D: DimName> From<VectorN<N, D>> for Point<N, D>
    where
        DefaultAllocator: Allocator<N, D>,
{
    #[inline]
    fn from(coords: VectorN<N, D>) -> Self {
        Point {
            coords
        }
    }
}
