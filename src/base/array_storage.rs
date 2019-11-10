use std::fmt::{self, Debug, Formatter};
use std::hash::{Hash, Hasher};
#[cfg(feature = "abomonation-serialize")]
use std::io::{Result as IOResult, Write};
use std::ops::{Deref, DerefMut, Mul};

#[cfg(feature = "serde-serialize")]
use serde::de::{Error, SeqAccess, Visitor};
#[cfg(feature = "serde-serialize")]
use serde::ser::SerializeSeq;
#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};
#[cfg(feature = "serde-serialize")]
use std::marker::PhantomData;
#[cfg(feature = "serde-serialize")]
use std::mem;

#[cfg(feature = "abomonation-serialize")]
use abomonation::Abomonation;

use generic_array::{ArrayLength, GenericArray};
use typenum::Prod;

use crate::base::allocator::Allocator;
use crate::base::default_allocator::DefaultAllocator;
use crate::{DimName, U1};
use crate::base::storage::{ContiguousStorage, ContiguousStorageMut, Owned, Storage, StorageMut};
use crate::base::Scalar;

/*
 *
 * Static Storage.
 *
 */
/// A array-based statically sized matrix data storage.
#[repr(C)]
pub struct ArrayStorage<N, R, C>
where
    R: DimName,
    C: DimName,
    R::Value: Mul<C::Value>,
    Prod<R::Value, C::Value>: ArrayLength<N>,
{
    data: GenericArray<N, Prod<R::Value, C::Value>>,
}

#[deprecated(note="renamed to `ArrayStorage`")]
/// Renamed to [ArrayStorage].
pub type MatrixArray<N, R, C> = ArrayStorage<N, R, C>;

impl<N, R, C> Hash for ArrayStorage<N, R, C>
where
    N: Hash,
    R: DimName,
    C: DimName,
    R::Value: Mul<C::Value>,
    Prod<R::Value, C::Value>: ArrayLength<N>,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data[..].hash(state)
    }
}

impl<N, R, C> Deref for ArrayStorage<N, R, C>
where
    R: DimName,
    C: DimName,
    R::Value: Mul<C::Value>,
    Prod<R::Value, C::Value>: ArrayLength<N>,
{
    type Target = GenericArray<N, Prod<R::Value, C::Value>>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<N, R, C> DerefMut for ArrayStorage<N, R, C>
where
    R: DimName,
    C: DimName,
    R::Value: Mul<C::Value>,
    Prod<R::Value, C::Value>: ArrayLength<N>,
{
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<N, R, C> Debug for ArrayStorage<N, R, C>
where
    N: Debug,
    R: DimName,
    C: DimName,
    R::Value: Mul<C::Value>,
    Prod<R::Value, C::Value>: ArrayLength<N>,
{
    #[inline]
    fn fmt(&self, fmt: &mut Formatter) -> fmt::Result {
        self.data.fmt(fmt)
    }
}

impl<N, R, C> Copy for ArrayStorage<N, R, C>
where
    N: Copy,
    R: DimName,
    C: DimName,
    R::Value: Mul<C::Value>,
    Prod<R::Value, C::Value>: ArrayLength<N>,
    GenericArray<N, Prod<R::Value, C::Value>>: Copy,
{}

impl<N, R, C> Clone for ArrayStorage<N, R, C>
where
    N: Clone,
    R: DimName,
    C: DimName,
    R::Value: Mul<C::Value>,
    Prod<R::Value, C::Value>: ArrayLength<N>,
{
    #[inline]
    fn clone(&self) -> Self {
        ArrayStorage {
            data: self.data.clone(),
        }
    }
}

impl<N, R, C> Eq for ArrayStorage<N, R, C>
where
    N: Eq,
    R: DimName,
    C: DimName,
    R::Value: Mul<C::Value>,
    Prod<R::Value, C::Value>: ArrayLength<N>,
{}

impl<N, R, C> PartialEq for ArrayStorage<N, R, C>
where
    N: PartialEq,
    R: DimName,
    C: DimName,
    R::Value: Mul<C::Value>,
    Prod<R::Value, C::Value>: ArrayLength<N>,
{
    #[inline]
    fn eq(&self, right: &Self) -> bool {
        self.data == right.data
    }
}

unsafe impl<N, R, C> Storage<N, R, C> for ArrayStorage<N, R, C>
where
    N: Scalar,
    R: DimName,
    C: DimName,
    R::Value: Mul<C::Value>,
    Prod<R::Value, C::Value>: ArrayLength<N>,
    DefaultAllocator: Allocator<N, R, C, Buffer = Self>,
{
    type RStride = U1;
    type CStride = R;

    #[inline]
    fn ptr(&self) -> *const N {
        self[..].as_ptr()
    }

    #[inline]
    fn shape(&self) -> (R, C) {
        (R::name(), C::name())
    }

    #[inline]
    fn strides(&self) -> (Self::RStride, Self::CStride) {
        (Self::RStride::name(), Self::CStride::name())
    }

    #[inline]
    fn is_contiguous(&self) -> bool {
        true
    }

    #[inline]
    fn into_owned(self) -> Owned<N, R, C>
    where DefaultAllocator: Allocator<N, R, C> {
        self
    }

    #[inline]
    fn clone_owned(&self) -> Owned<N, R, C>
    where DefaultAllocator: Allocator<N, R, C> {
        let it = self.iter().cloned();

        DefaultAllocator::allocate_from_iterator(self.shape().0, self.shape().1, it)
    }

    #[inline]
    fn as_slice(&self) -> &[N] {
        &self[..]
    }
}

unsafe impl<N, R, C> StorageMut<N, R, C> for ArrayStorage<N, R, C>
where
    N: Scalar,
    R: DimName,
    C: DimName,
    R::Value: Mul<C::Value>,
    Prod<R::Value, C::Value>: ArrayLength<N>,
    DefaultAllocator: Allocator<N, R, C, Buffer = Self>,
{
    #[inline]
    fn ptr_mut(&mut self) -> *mut N {
        self[..].as_mut_ptr()
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [N] {
        &mut self[..]
    }
}

unsafe impl<N, R, C> ContiguousStorage<N, R, C> for ArrayStorage<N, R, C>
where
    N: Scalar,
    R: DimName,
    C: DimName,
    R::Value: Mul<C::Value>,
    Prod<R::Value, C::Value>: ArrayLength<N>,
    DefaultAllocator: Allocator<N, R, C, Buffer = Self>,
{}

unsafe impl<N, R, C> ContiguousStorageMut<N, R, C> for ArrayStorage<N, R, C>
where
    N: Scalar,
    R: DimName,
    C: DimName,
    R::Value: Mul<C::Value>,
    Prod<R::Value, C::Value>: ArrayLength<N>,
    DefaultAllocator: Allocator<N, R, C, Buffer = Self>,
{}

/*
 *
 * Allocation-less serde impls.
 *
 */
// XXX: open an issue for GenericArray so that it implements serde traits?
#[cfg(feature = "serde-serialize")]
impl<N, R, C> Serialize for ArrayStorage<N, R, C>
where
    N: Scalar + Serialize,
    R: DimName,
    C: DimName,
    R::Value: Mul<C::Value>,
    Prod<R::Value, C::Value>: ArrayLength<N>,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: Serializer {
        let mut serializer = serializer.serialize_seq(Some(R::dim() * C::dim()))?;

        for e in self.iter() {
            serializer.serialize_element(e)?;
        }

        serializer.end()
    }
}

#[cfg(feature = "serde-serialize")]
impl<'a, N, R, C> Deserialize<'a> for ArrayStorage<N, R, C>
where
    N: Scalar + Deserialize<'a>,
    R: DimName,
    C: DimName,
    R::Value: Mul<C::Value>,
    Prod<R::Value, C::Value>: ArrayLength<N>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: Deserializer<'a> {
        deserializer.deserialize_seq(ArrayStorageVisitor::new())
    }
}

#[cfg(feature = "serde-serialize")]
/// A visitor that produces a matrix array.
struct ArrayStorageVisitor<N, R, C> {
    marker: PhantomData<(N, R, C)>,
}

#[cfg(feature = "serde-serialize")]
impl<N, R, C> ArrayStorageVisitor<N, R, C>
where
    N: Scalar,
    R: DimName,
    C: DimName,
    R::Value: Mul<C::Value>,
    Prod<R::Value, C::Value>: ArrayLength<N>,
{
    /// Construct a new sequence visitor.
    pub fn new() -> Self {
        ArrayStorageVisitor {
            marker: PhantomData,
        }
    }
}

#[cfg(feature = "serde-serialize")]
impl<'a, N, R, C> Visitor<'a> for ArrayStorageVisitor<N, R, C>
where
    N: Scalar + Deserialize<'a>,
    R: DimName,
    C: DimName,
    R::Value: Mul<C::Value>,
    Prod<R::Value, C::Value>: ArrayLength<N>,
{
    type Value = ArrayStorage<N, R, C>;

    fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
        formatter.write_str("a matrix array")
    }

    #[inline]
    fn visit_seq<V>(self, mut visitor: V) -> Result<ArrayStorage<N, R, C>, V::Error>
    where V: SeqAccess<'a> {
        let mut out: Self::Value = unsafe { mem::uninitialized() };
        let mut curr = 0;

        while let Some(value) = visitor.next_element()? {
            *out.get_mut(curr).ok_or_else(|| V::Error::invalid_length(curr, &self))? = value;
            curr += 1;
        }

        if curr == R::dim() * C::dim() {
            Ok(out)
        } else {
            Err(V::Error::invalid_length(curr, &self))
        }
    }
}

#[cfg(feature = "abomonation-serialize")]
impl<N, R, C> Abomonation for ArrayStorage<N, R, C>
where
    R: DimName,
    C: DimName,
    R::Value: Mul<C::Value>,
    Prod<R::Value, C::Value>: ArrayLength<N>,
    N: Abomonation,
{
    unsafe fn entomb<W: Write>(&self, writer: &mut W) -> IOResult<()> {
        for element in self.data.as_slice() {
            element.entomb(writer)?;
        }

        Ok(())
    }

    unsafe fn exhume<'a, 'b>(&'a mut self, mut bytes: &'b mut [u8]) -> Option<&'b mut [u8]> {
        for element in self.data.as_mut_slice() {
            let temp = bytes;
            bytes = if let Some(remainder) = element.exhume(temp) {
                remainder
            } else {
                return None;
            }
        }
        Some(bytes)
    }

    fn extent(&self) -> usize {
        self.data
            .as_slice()
            .iter()
            .fold(0, |acc, e| acc + e.extent())
    }
}
