#[cfg(feature = "abomonation-serialize")]
use std::io::{Result as IOResult, Write};

#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::vec::Vec;

use crate::base::allocator::Allocator;
use crate::base::default_allocator::DefaultAllocator;
use crate::{Dim, DimName, Dynamic, U1};
use crate::base::storage::{ContiguousStorage, ContiguousStorageMut, Owned, Storage, StorageMut};
use crate::base::{Scalar, Vector};
use crate::base::constraint::{SameNumberOfRows, ShapeConstraint};

#[cfg(feature = "abomonation-serialize")]
use abomonation::Abomonation;

/*
 *
 * Storage.
 *
 */
/// A Vec-based matrix data storage. It may be dynamically-sized.
#[repr(C)]
#[derive(Eq, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct VecStorage<N, R: Dim, C: Dim> {
    data: Vec<N>,
    nrows: R,
    ncols: C,
}

#[deprecated(note="renamed to `VecStorage`")]
/// Renamed to [VecStorage].
pub type MatrixVec<N, R, C> = VecStorage<N, R, C>;

impl<N, R: Dim, C: Dim> VecStorage<N, R, C> {
    /// Creates a new dynamic matrix data storage from the given vector and shape.
    #[inline]
    pub fn new(nrows: R, ncols: C, data: Vec<N>) -> Self {
        assert!(
            nrows.value() * ncols.value() == data.len(),
            "Data storage buffer dimension mismatch."
        );
        Self {
            data: data,
            nrows: nrows,
            ncols: ncols,
        }
    }

    /// The underlying data storage.
    #[inline]
    pub fn as_vec(&self) -> &Vec<N> {
        &self.data
    }

    /// The underlying mutable data storage.
    ///
    /// This is unsafe because this may cause UB if the size of the vector is changed
    /// by the user.
    #[inline]
    pub unsafe fn as_vec_mut(&mut self) -> &mut Vec<N> {
        &mut self.data
    }

    /// Resizes the underlying mutable data storage and unwraps it.
    ///
    /// If `sz` is larger than the current size, additional elements are uninitialized.
    /// If `sz` is smaller than the current size, additional elements are truncated.
    #[inline]
    pub unsafe fn resize(mut self, sz: usize) -> Vec<N> {
        let len = self.len();

        if sz < len {
            self.data.set_len(sz);
            self.data.shrink_to_fit();
        } else {
            self.data.reserve_exact(sz - len);
            self.data.set_len(sz);
        }

        self.data
    }

    /// The number of elements on the underlying vector.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }
}

impl<N, R: Dim, C: Dim> Into<Vec<N>> for VecStorage<N, R, C>
{
    fn into(self) -> Vec<N> {
        self.data
    }
}

/*
 *
 * Dynamic − Static
 * Dynamic − Dynamic
 *
 */
unsafe impl<N: Scalar, C: Dim> Storage<N, Dynamic, C> for VecStorage<N, Dynamic, C>
where DefaultAllocator: Allocator<N, Dynamic, C, Buffer = Self>
{
    type RStride = U1;
    type CStride = Dynamic;

    #[inline]
    fn ptr(&self) -> *const N {
        self.data.as_ptr()
    }

    #[inline]
    fn shape(&self) -> (Dynamic, C) {
        (self.nrows, self.ncols)
    }

    #[inline]
    fn strides(&self) -> (Self::RStride, Self::CStride) {
        (Self::RStride::name(), self.nrows)
    }

    #[inline]
    fn is_contiguous(&self) -> bool {
        true
    }

    #[inline]
    fn into_owned(self) -> Owned<N, Dynamic, C>
    where DefaultAllocator: Allocator<N, Dynamic, C> {
        self
    }

    #[inline]
    fn clone_owned(&self) -> Owned<N, Dynamic, C>
    where DefaultAllocator: Allocator<N, Dynamic, C> {
        self.clone()
    }

    #[inline]
    fn as_slice(&self) -> &[N] {
        &self.data
    }
}

unsafe impl<N: Scalar, R: DimName> Storage<N, R, Dynamic> for VecStorage<N, R, Dynamic>
where DefaultAllocator: Allocator<N, R, Dynamic, Buffer = Self>
{
    type RStride = U1;
    type CStride = R;

    #[inline]
    fn ptr(&self) -> *const N {
        self.data.as_ptr()
    }

    #[inline]
    fn shape(&self) -> (R, Dynamic) {
        (self.nrows, self.ncols)
    }

    #[inline]
    fn strides(&self) -> (Self::RStride, Self::CStride) {
        (Self::RStride::name(), self.nrows)
    }

    #[inline]
    fn is_contiguous(&self) -> bool {
        true
    }

    #[inline]
    fn into_owned(self) -> Owned<N, R, Dynamic>
    where DefaultAllocator: Allocator<N, R, Dynamic> {
        self
    }

    #[inline]
    fn clone_owned(&self) -> Owned<N, R, Dynamic>
    where DefaultAllocator: Allocator<N, R, Dynamic> {
        self.clone()
    }

    #[inline]
    fn as_slice(&self) -> &[N] {
        &self.data
    }
}

/*
 *
 * StorageMut, ContiguousStorage.
 *
 */
unsafe impl<N: Scalar, C: Dim> StorageMut<N, Dynamic, C> for VecStorage<N, Dynamic, C>
where DefaultAllocator: Allocator<N, Dynamic, C, Buffer = Self>
{
    #[inline]
    fn ptr_mut(&mut self) -> *mut N {
        self.data.as_mut_ptr()
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [N] {
        &mut self.data[..]
    }
}

unsafe impl<N: Scalar, C: Dim> ContiguousStorage<N, Dynamic, C> for VecStorage<N, Dynamic, C> where DefaultAllocator: Allocator<N, Dynamic, C, Buffer = Self>
{}

unsafe impl<N: Scalar, C: Dim> ContiguousStorageMut<N, Dynamic, C> for VecStorage<N, Dynamic, C> where DefaultAllocator: Allocator<N, Dynamic, C, Buffer = Self>
{}

unsafe impl<N: Scalar, R: DimName> StorageMut<N, R, Dynamic> for VecStorage<N, R, Dynamic>
where DefaultAllocator: Allocator<N, R, Dynamic, Buffer = Self>
{
    #[inline]
    fn ptr_mut(&mut self) -> *mut N {
        self.data.as_mut_ptr()
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [N] {
        &mut self.data[..]
    }
}

#[cfg(feature = "abomonation-serialize")]
impl<N: Abomonation, R: Dim, C: Dim> Abomonation for VecStorage<N, R, C> {
    unsafe fn entomb<W: Write>(&self, writer: &mut W) -> IOResult<()> {
        self.data.entomb(writer)
    }

    unsafe fn exhume<'a, 'b>(&'a mut self, bytes: &'b mut [u8]) -> Option<&'b mut [u8]> {
        self.data.exhume(bytes)
    }

    fn extent(&self) -> usize {
        self.data.extent()
    }
}

unsafe impl<N: Scalar, R: DimName> ContiguousStorage<N, R, Dynamic> for VecStorage<N, R, Dynamic> where DefaultAllocator: Allocator<N, R, Dynamic, Buffer = Self>
{}

unsafe impl<N: Scalar, R: DimName> ContiguousStorageMut<N, R, Dynamic> for VecStorage<N, R, Dynamic> where DefaultAllocator: Allocator<N, R, Dynamic, Buffer = Self>
{}

impl<N, R: Dim> Extend<N> for VecStorage<N, R, Dynamic>
{
    /// Extends the number of columns of the `VecStorage` with elements
    /// from the given iterator.
    ///
    /// # Panics
    /// This function panics if the number of elements yielded by the
    /// given iterator is not a multiple of the number of rows of the
    /// `VecStorage`.
    fn extend<I: IntoIterator<Item=N>>(&mut self, iter: I)
    {
        self.data.extend(iter);
        self.ncols = Dynamic::new(self.data.len() / self.nrows.value());
        assert!(self.data.len() % self.nrows.value() == 0,
          "The number of elements produced by the given iterator was not a multiple of the number of rows.");
    }
}

impl<N, R, RV, SV> Extend<Vector<N, RV, SV>> for VecStorage<N, R, Dynamic>
where
    N: Scalar,
    R: Dim,
    RV: Dim,
    SV: Storage<N, RV>,
    ShapeConstraint: SameNumberOfRows<R, RV>,
{
    /// Extends the number of columns of the `VecStorage` with vectors
    /// from the given iterator.
    ///
    /// # Panics
    /// This function panics if the number of rows of each `Vector`
    /// yielded by the iterator is not equal to the number of rows
    /// of this `VecStorage`.
    fn extend<I: IntoIterator<Item=Vector<N, RV, SV>>>(&mut self, iter: I)
    {
        let nrows = self.nrows.value();
        let iter = iter.into_iter();
        let (lower, _upper) = iter.size_hint();
        self.data.reserve(nrows * lower);
        for vector in iter {
            assert_eq!(nrows, vector.shape().0);
            self.data.extend(vector.iter());
        }
        self.ncols = Dynamic::new(self.data.len() / nrows);
    }
}

impl<N> Extend<N> for VecStorage<N, Dynamic, U1>
{
    /// Extends the number of rows of the `VecStorage` with elements
    /// from the given iterator.
    fn extend<I: IntoIterator<Item=N>>(&mut self, iter: I)
    {
        self.data.extend(iter);
        self.nrows = Dynamic::new(self.data.len());
    }
}
