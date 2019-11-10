use alga::general::{SubsetOf, SupersetOf};
#[cfg(feature = "mint")]
use mint;
use std::convert::{AsMut, AsRef, From, Into};
use std::mem;
use std::ptr;

use generic_array::ArrayLength;
use std::ops::Mul;
use typenum::Prod;

use crate::base::allocator::{Allocator, SameShapeAllocator};
use crate::base::constraint::{SameNumberOfColumns, SameNumberOfRows, ShapeConstraint};
use crate::{
    Dim, DimName, U1, U10, U11, U12, U13, U14, U15, U16, U2, U3, U4, U5, U6, U7, U8, U9,
};
#[cfg(any(feature = "std", feature = "alloc"))]
use crate::Dynamic;
use crate::base::iter::{MatrixIter, MatrixIterMut};
use crate::base::storage::{ContiguousStorage, ContiguousStorageMut, Storage, StorageMut};
#[cfg(any(feature = "std", feature = "alloc"))]
use crate::base::VecStorage;
use crate::base::{SliceStorage, SliceStorageMut};
use crate::base::{DefaultAllocator, Matrix, ArrayStorage, MatrixMN, MatrixSlice, MatrixSliceMut, Scalar};
use crate::constraint::DimEq;

// FIXME: too bad this won't work allo slice conversions.
impl<N1, N2, R1, C1, R2, C2> SubsetOf<MatrixMN<N2, R2, C2>> for MatrixMN<N1, R1, C1>
where
    R1: Dim,
    C1: Dim,
    R2: Dim,
    C2: Dim,
    N1: Scalar,
    N2: Scalar + SupersetOf<N1>,
    DefaultAllocator:
        Allocator<N2, R2, C2> + Allocator<N1, R1, C1> + SameShapeAllocator<N1, R1, C1, R2, C2>,
    ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2>,
{
    #[inline]
    fn to_superset(&self) -> MatrixMN<N2, R2, C2> {
        let (nrows, ncols) = self.shape();
        let nrows2 = R2::from_usize(nrows);
        let ncols2 = C2::from_usize(ncols);

        let mut res = unsafe { MatrixMN::<N2, R2, C2>::new_uninitialized_generic(nrows2, ncols2) };
        for i in 0..nrows {
            for j in 0..ncols {
                unsafe { *res.get_unchecked_mut((i, j)) = N2::from_subset(self.get_unchecked((i, j))) }
            }
        }

        res
    }

    #[inline]
    fn is_in_subset(m: &MatrixMN<N2, R2, C2>) -> bool {
        m.iter().all(|e| e.is_in_subset())
    }

    #[inline]
    unsafe fn from_superset_unchecked(m: &MatrixMN<N2, R2, C2>) -> Self {
        let (nrows2, ncols2) = m.shape();
        let nrows = R1::from_usize(nrows2);
        let ncols = C1::from_usize(ncols2);

        let mut res = Self::new_uninitialized_generic(nrows, ncols);
        for i in 0..nrows2 {
            for j in 0..ncols2 {
                *res.get_unchecked_mut((i, j)) = m.get_unchecked((i, j)).to_subset_unchecked()
            }
        }

        res
    }
}

impl<'a, N: Scalar, R: Dim, C: Dim, S: Storage<N, R, C>> IntoIterator for &'a Matrix<N, R, C, S> {
    type Item = &'a N;
    type IntoIter = MatrixIter<'a, N, R, C, S>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, N: Scalar, R: Dim, C: Dim, S: StorageMut<N, R, C>> IntoIterator
    for &'a mut Matrix<N, R, C, S>
{
    type Item = &'a mut N;
    type IntoIter = MatrixIterMut<'a, N, R, C, S>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

macro_rules! impl_from_into_asref_1D(
    ($(($NRows: ident, $NCols: ident) => $SZ: expr);* $(;)*) => {$(
        impl<N> From<[N; $SZ]> for MatrixMN<N, $NRows, $NCols>
        where N: Scalar,
              DefaultAllocator: Allocator<N, $NRows, $NCols> {
            #[inline]
            fn from(arr: [N; $SZ]) -> Self {
                unsafe {
                    let mut res = Self::new_uninitialized();
                    ptr::copy_nonoverlapping(&arr[0], res.data.ptr_mut(), $SZ);

                    res
                }
            }
        }

        impl<N, S> Into<[N; $SZ]> for Matrix<N, $NRows, $NCols, S>
        where N: Scalar,
              S: ContiguousStorage<N, $NRows, $NCols> {
            #[inline]
            fn into(self) -> [N; $SZ] {
                unsafe {
                    let mut res: [N; $SZ] = mem::uninitialized();
                    ptr::copy_nonoverlapping(self.data.ptr(), &mut res[0], $SZ);

                    res
                }
            }
        }

        impl<N, S> AsRef<[N; $SZ]> for Matrix<N, $NRows, $NCols, S>
        where N: Scalar,
              S: ContiguousStorage<N, $NRows, $NCols> {
            #[inline]
            fn as_ref(&self) -> &[N; $SZ] {
                unsafe {
                    mem::transmute(self.data.ptr())
                }
            }
        }

        impl<N, S> AsMut<[N; $SZ]> for Matrix<N, $NRows, $NCols, S>
        where N: Scalar,
              S: ContiguousStorageMut<N, $NRows, $NCols> {
            #[inline]
            fn as_mut(&mut self) -> &mut [N; $SZ] {
                unsafe {
                    mem::transmute(self.data.ptr_mut())
                }
            }
        }
    )*}
);

// Implement for vectors of dimension 1 .. 16.
impl_from_into_asref_1D!(
    // Row vectors.
    (U1, U1 ) => 1;  (U1, U2 ) => 2;  (U1, U3 ) => 3;  (U1, U4 ) => 4;
    (U1, U5 ) => 5;  (U1, U6 ) => 6;  (U1, U7 ) => 7;  (U1, U8 ) => 8;
    (U1, U9 ) => 9;  (U1, U10) => 10; (U1, U11) => 11; (U1, U12) => 12;
    (U1, U13) => 13; (U1, U14) => 14; (U1, U15) => 15; (U1, U16) => 16;

    // Column vectors.
                     (U2 , U1) => 2;  (U3 , U1) => 3;  (U4 , U1) => 4;
    (U5 , U1) => 5;  (U6 , U1) => 6;  (U7 , U1) => 7;  (U8 , U1) => 8;
    (U9 , U1) => 9;  (U10, U1) => 10; (U11, U1) => 11; (U12, U1) => 12;
    (U13, U1) => 13; (U14, U1) => 14; (U15, U1) => 15; (U16, U1) => 16;
);

macro_rules! impl_from_into_asref_2D(
    ($(($NRows: ty, $NCols: ty) => ($SZRows: expr, $SZCols: expr));* $(;)*) => {$(
        impl<N: Scalar> From<[[N; $SZRows]; $SZCols]> for MatrixMN<N, $NRows, $NCols>
        where DefaultAllocator: Allocator<N, $NRows, $NCols> {
            #[inline]
            fn from(arr: [[N; $SZRows]; $SZCols]) -> Self {
                unsafe {
                    let mut res = Self::new_uninitialized();
                    ptr::copy_nonoverlapping(&arr[0][0], res.data.ptr_mut(), $SZRows * $SZCols);

                    res
                }
            }
        }

        impl<N: Scalar, S> Into<[[N; $SZRows]; $SZCols]> for Matrix<N, $NRows, $NCols, S>
        where S: ContiguousStorage<N, $NRows, $NCols> {
            #[inline]
            fn into(self) -> [[N; $SZRows]; $SZCols] {
                unsafe {
                    let mut res: [[N; $SZRows]; $SZCols] = mem::uninitialized();
                    ptr::copy_nonoverlapping(self.data.ptr(), &mut res[0][0], $SZRows * $SZCols);

                    res
                }
            }
        }

        impl<N: Scalar, S> AsRef<[[N; $SZRows]; $SZCols]> for Matrix<N, $NRows, $NCols, S>
        where S: ContiguousStorage<N, $NRows, $NCols> {
            #[inline]
            fn as_ref(&self) -> &[[N; $SZRows]; $SZCols] {
                unsafe {
                    mem::transmute(self.data.ptr())
                }
            }
        }

        impl<N: Scalar, S> AsMut<[[N; $SZRows]; $SZCols]> for Matrix<N, $NRows, $NCols, S>
        where S: ContiguousStorageMut<N, $NRows, $NCols> {
            #[inline]
            fn as_mut(&mut self) -> &mut [[N; $SZRows]; $SZCols] {
                unsafe {
                    mem::transmute(self.data.ptr_mut())
                }
            }
        }
    )*}
);

// Implement for matrices with shape 2x2 .. 6x6.
impl_from_into_asref_2D!(
    (U2, U2) => (2, 2); (U2, U3) => (2, 3); (U2, U4) => (2, 4); (U2, U5) => (2, 5); (U2, U6) => (2, 6);
    (U3, U2) => (3, 2); (U3, U3) => (3, 3); (U3, U4) => (3, 4); (U3, U5) => (3, 5); (U3, U6) => (3, 6);
    (U4, U2) => (4, 2); (U4, U3) => (4, 3); (U4, U4) => (4, 4); (U4, U5) => (4, 5); (U4, U6) => (4, 6);
    (U5, U2) => (5, 2); (U5, U3) => (5, 3); (U5, U4) => (5, 4); (U5, U5) => (5, 5); (U5, U6) => (5, 6);
    (U6, U2) => (6, 2); (U6, U3) => (6, 3); (U6, U4) => (6, 4); (U6, U5) => (6, 5); (U6, U6) => (6, 6);
);

#[cfg(feature = "mint")]
macro_rules! impl_from_into_mint_1D(
    ($($NRows: ident => $VT:ident [$SZ: expr]);* $(;)*) => {$(
        impl<N> From<mint::$VT<N>> for MatrixMN<N, $NRows, U1>
        where N: Scalar,
              DefaultAllocator: Allocator<N, $NRows, U1> {
            #[inline]
            fn from(v: mint::$VT<N>) -> Self {
                unsafe {
                    let mut res = Self::new_uninitialized();
                    ptr::copy_nonoverlapping(&v.x, res.data.ptr_mut(), $SZ);

                    res
                }
            }
        }

        impl<N, S> Into<mint::$VT<N>> for Matrix<N, $NRows, U1, S>
        where N: Scalar,
              S: ContiguousStorage<N, $NRows, U1> {
            #[inline]
            fn into(self) -> mint::$VT<N> {
                unsafe {
                    let mut res: mint::$VT<N> = mem::uninitialized();
                    ptr::copy_nonoverlapping(self.data.ptr(), &mut res.x, $SZ);

                    res
                }
            }
        }

        impl<N, S> AsRef<mint::$VT<N>> for Matrix<N, $NRows, U1, S>
        where N: Scalar,
              S: ContiguousStorage<N, $NRows, U1> {
            #[inline]
            fn as_ref(&self) -> &mint::$VT<N> {
                unsafe {
                    mem::transmute(self.data.ptr())
                }
            }
        }

        impl<N, S> AsMut<mint::$VT<N>> for Matrix<N, $NRows, U1, S>
        where N: Scalar,
              S: ContiguousStorageMut<N, $NRows, U1> {
            #[inline]
            fn as_mut(&mut self) -> &mut mint::$VT<N> {
                unsafe {
                    mem::transmute(self.data.ptr_mut())
                }
            }
        }
    )*}
);

// Implement for vectors of dimension 2 .. 4.
#[cfg(feature = "mint")]
impl_from_into_mint_1D!(
    U2 => Vector2[2];
    U3 => Vector3[3];
    U4 => Vector4[4];
);

#[cfg(feature = "mint")]
macro_rules! impl_from_into_mint_2D(
    ($(($NRows: ty, $NCols: ty) => $MV:ident{ $($component:ident),* }[$SZRows: expr]);* $(;)*) => {$(
        impl<N> From<mint::$MV<N>> for MatrixMN<N, $NRows, $NCols>
        where N: Scalar,
              DefaultAllocator: Allocator<N, $NRows, $NCols> {
            #[inline]
            fn from(m: mint::$MV<N>) -> Self {
                unsafe {
                    let mut res = Self::new_uninitialized();
                    let mut ptr = res.data.ptr_mut();
                    $(
                        ptr::copy_nonoverlapping(&m.$component.x, ptr, $SZRows);
                        ptr = ptr.offset($SZRows);
                    )*
                    let _ = ptr;
                    res
                }
            }
        }

        impl<N> Into<mint::$MV<N>> for MatrixMN<N, $NRows, $NCols>
        where N: Scalar,
              DefaultAllocator: Allocator<N, $NRows, $NCols> {
            #[inline]
            fn into(self) -> mint::$MV<N> {
                unsafe {
                    let mut res: mint::$MV<N> = mem::uninitialized();
                    let mut ptr = self.data.ptr();
                    $(
                        ptr::copy_nonoverlapping(ptr, &mut res.$component.x, $SZRows);
                        ptr = ptr.offset($SZRows);
                    )*
                    let _ = ptr;
                    res
                }
            }
        }
    )*}
);

// Implement for matrices with shape 2x2 .. 4x4.
#[cfg(feature = "mint")]
impl_from_into_mint_2D!(
    (U2, U2) => ColumnMatrix2{x, y}[2];
    (U2, U3) => ColumnMatrix2x3{x, y, z}[2];
    (U3, U3) => ColumnMatrix3{x, y, z}[3];
    (U3, U4) => ColumnMatrix3x4{x, y, z, w}[3];
    (U4, U4) => ColumnMatrix4{x, y, z, w}[4];
);

impl<'a, N, R, C, RStride, CStride> From<MatrixSlice<'a, N, R, C, RStride, CStride>>
    for Matrix<N, R, C, ArrayStorage<N, R, C>>
where
    N: Scalar,
    R: DimName,
    C: DimName,
    RStride: Dim,
    CStride: Dim,
    R::Value: Mul<C::Value>,
    Prod<R::Value, C::Value>: ArrayLength<N>,
{
    fn from(matrix_slice: MatrixSlice<'a, N, R, C, RStride, CStride>) -> Self {
        matrix_slice.into_owned()
    }
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl<'a, N, C, RStride, CStride> From<MatrixSlice<'a, N, Dynamic, C, RStride, CStride>>
    for Matrix<N, Dynamic, C, VecStorage<N, Dynamic, C>>
where
    N: Scalar,
    C: Dim,
    RStride: Dim,
    CStride: Dim,
{
    fn from(matrix_slice: MatrixSlice<'a, N, Dynamic, C, RStride, CStride>) -> Self {
        matrix_slice.into_owned()
    }
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl<'a, N, R, RStride, CStride> From<MatrixSlice<'a, N, R, Dynamic, RStride, CStride>>
    for Matrix<N, R, Dynamic, VecStorage<N, R, Dynamic>>
where
    N: Scalar,
    R: DimName,
    RStride: Dim,
    CStride: Dim,
{
    fn from(matrix_slice: MatrixSlice<'a, N, R, Dynamic, RStride, CStride>) -> Self {
        matrix_slice.into_owned()
    }
}

impl<'a, N, R, C, RStride, CStride> From<MatrixSliceMut<'a, N, R, C, RStride, CStride>>
    for Matrix<N, R, C, ArrayStorage<N, R, C>>
where
    N: Scalar,
    R: DimName,
    C: DimName,
    RStride: Dim,
    CStride: Dim,
    R::Value: Mul<C::Value>,
    Prod<R::Value, C::Value>: ArrayLength<N>,
{
    fn from(matrix_slice: MatrixSliceMut<'a, N, R, C, RStride, CStride>) -> Self {
        matrix_slice.into_owned()
    }
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl<'a, N, C, RStride, CStride> From<MatrixSliceMut<'a, N, Dynamic, C, RStride, CStride>>
    for Matrix<N, Dynamic, C, VecStorage<N, Dynamic, C>>
where
    N: Scalar,
    C: Dim,
    RStride: Dim,
    CStride: Dim,
{
    fn from(matrix_slice: MatrixSliceMut<'a, N, Dynamic, C, RStride, CStride>) -> Self {
        matrix_slice.into_owned()
    }
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl<'a, N, R, RStride, CStride> From<MatrixSliceMut<'a, N, R, Dynamic, RStride, CStride>>
    for Matrix<N, R, Dynamic, VecStorage<N, R, Dynamic>>
where
    N: Scalar,
    R: DimName,
    RStride: Dim,
    CStride: Dim,
{
    fn from(matrix_slice: MatrixSliceMut<'a, N, R, Dynamic, RStride, CStride>) -> Self {
        matrix_slice.into_owned()
    }
}

impl<'a, N, R, C, RSlice, CSlice, RStride, CStride, S> From<&'a Matrix<N, R, C, S>>
for MatrixSlice<'a, N, RSlice, CSlice, RStride, CStride>
    where
        N: Scalar,
        R: Dim,
        C: Dim,
        RSlice: Dim,
        CSlice: Dim,
        RStride: Dim,
        CStride: Dim,
        S: Storage<N, R, C>,
        ShapeConstraint: DimEq<R, RSlice> + DimEq<C, CSlice>
            + DimEq<RStride, S::RStride> + DimEq<CStride, S::CStride>
{
    fn from(m: &'a Matrix<N, R, C, S>) -> Self {
        let (row, col) = m.data.shape();
        let row_slice = RSlice::from_usize(row.value());
        let col_slice = CSlice::from_usize(col.value());

        let (rstride, cstride) = m.strides();

        let rstride_slice = RStride::from_usize(rstride);
        let cstride_slice = CStride::from_usize(cstride);

        unsafe {
            let data = SliceStorage::from_raw_parts(m.data.ptr(),
                                                    (row_slice, col_slice),
                                                    (rstride_slice, cstride_slice));
            Matrix::from_data_statically_unchecked(data)
        }
    }
}

impl<'a, N, R, C, RSlice, CSlice, RStride, CStride, S> From<&'a mut Matrix<N, R, C, S>>
for MatrixSlice<'a, N, RSlice, CSlice, RStride, CStride>
    where
        N: Scalar,
        R: Dim,
        C: Dim,
        RSlice: Dim,
        CSlice: Dim,
        RStride: Dim,
        CStride: Dim,
        S: Storage<N, R, C>,
        ShapeConstraint: DimEq<R, RSlice> + DimEq<C, CSlice>
            + DimEq<RStride, S::RStride> + DimEq<CStride, S::CStride>
{
    fn from(m: &'a mut Matrix<N, R, C, S>) -> Self {
        let (row, col) = m.data.shape();
        let row_slice = RSlice::from_usize(row.value());
        let col_slice = CSlice::from_usize(col.value());

        let (rstride, cstride) = m.strides();

        let rstride_slice = RStride::from_usize(rstride);
        let cstride_slice = CStride::from_usize(cstride);

        unsafe {
            let data = SliceStorage::from_raw_parts(m.data.ptr(),
                                                    (row_slice, col_slice),
                                                    (rstride_slice, cstride_slice));
            Matrix::from_data_statically_unchecked(data)
        }
    }
}

impl<'a, N, R, C, RSlice, CSlice, RStride, CStride, S> From<&'a mut Matrix<N, R, C, S>>
for MatrixSliceMut<'a, N, RSlice, CSlice, RStride, CStride>
    where
        N: Scalar,
        R: Dim,
        C: Dim,
        RSlice: Dim,
        CSlice: Dim,
        RStride: Dim,
        CStride: Dim,
        S: StorageMut<N, R, C>,
        ShapeConstraint: DimEq<R, RSlice> + DimEq<C, CSlice>
            + DimEq<RStride, S::RStride> + DimEq<CStride, S::CStride>
{
    fn from(m: &'a mut Matrix<N, R, C, S>) -> Self {
        let (row, col) = m.data.shape();
        let row_slice = RSlice::from_usize(row.value());
        let col_slice = CSlice::from_usize(col.value());

        let (rstride, cstride) = m.strides();

        let rstride_slice = RStride::from_usize(rstride);
        let cstride_slice = CStride::from_usize(cstride);

        unsafe {
            let data = SliceStorageMut::from_raw_parts(m.data.ptr_mut(),
                                                    (row_slice, col_slice),
                                                    (rstride_slice, cstride_slice));
            Matrix::from_data_statically_unchecked(data)
        }
    }
}