//! Abstract definition of a matrix data storage allocator.

use std::any::Any;
use std::mem;

use crate::base::constraint::{SameNumberOfColumns, SameNumberOfRows, ShapeConstraint};
use crate::base::dimension::{Dim, U1};
use crate::base::storage::ContiguousStorageMut;
use crate::base::{DefaultAllocator, Scalar};

/// A matrix allocator of a memory buffer that may contain `R::to_usize() * C::to_usize()`
/// elements of type `T`.
///
/// An allocator is said to be:
///   − static:  if `R` and `C` both implement `DimName`.
///   − dynamic: if either one (or both) of `R` or `C` is equal to `Dynamic`.
///
/// Every allocator must be both static and dynamic. Though not all implementations may share the
/// same `Buffer` type.
pub trait Allocator<T: Scalar, R: Dim, C: Dim = U1>: Any + Sized {
    /// The type of buffer this allocator can instanciate.
    type Buffer: ContiguousStorageMut<T, R, C> + Clone;

    /// Allocates a buffer with the given number of rows and columns without initializing its content.
    unsafe fn allocate_uninitialized(nrows: R, ncols: C) -> mem::MaybeUninit<Self::Buffer>;

    /// Allocates a buffer initialized with the content of the given iterator.
    fn allocate_from_iterator<I: IntoIterator<Item = T>>(
        nrows: R,
        ncols: C,
        iter: I,
    ) -> Self::Buffer;

    /// Allocates a buffer initialized with the content of the given row-major order iterator.
    fn allocate_from_row_iterator<I: IntoIterator<Item = T>>(
        nrows: R,
        ncols: C,
        iter: I,
    ) -> Self::Buffer;
}

/// A matrix reallocator. Changes the size of the memory buffer that initially contains (RFrom ×
/// CFrom) elements to a smaller or larger size (RTo, CTo).
pub trait Reallocator<T: Scalar, RFrom: Dim, CFrom: Dim, RTo: Dim, CTo: Dim>:
    Allocator<T, RFrom, CFrom> + Allocator<T, RTo, CTo>
{
    /// Reallocates a buffer of shape `(RTo, CTo)`, possibly reusing a previously allocated buffer
    /// `buf`. Data stored by `buf` are linearly copied to the output:
    ///
    /// # Safety
    /// * The copy is performed as if both were just arrays (without a matrix structure).
    /// * If `buf` is larger than the output size, then extra elements of `buf` are truncated.
    /// * If `buf` is smaller than the output size, then extra elements of the output are left
    /// uninitialized.
    unsafe fn reallocate_copy(
        nrows: RTo,
        ncols: CTo,
        buf: <Self as Allocator<T, RFrom, CFrom>>::Buffer,
    ) -> <Self as Allocator<T, RTo, CTo>>::Buffer;
}

/// The number of rows of the result of a componentwise operation on two matrices.
pub type SameShapeR<R1, R2> = <ShapeConstraint as SameNumberOfRows<R1, R2>>::Representative;

/// The number of columns of the result of a componentwise operation on two matrices.
pub type SameShapeC<C1, C2> = <ShapeConstraint as SameNumberOfColumns<C1, C2>>::Representative;

// TODO: Bad name.
/// Restricts the given number of rows and columns to be respectively the same.
pub trait SameShapeAllocator<T, R1, C1, R2, C2>:
    Allocator<T, R1, C1> + Allocator<T, SameShapeR<R1, R2>, SameShapeC<C1, C2>>
where
    R1: Dim,
    R2: Dim,
    C1: Dim,
    C2: Dim,
    T: Scalar,
    ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2>,
{
}

impl<T, R1, R2, C1, C2> SameShapeAllocator<T, R1, C1, R2, C2> for DefaultAllocator
where
    R1: Dim,
    R2: Dim,
    C1: Dim,
    C2: Dim,
    T: Scalar,
    DefaultAllocator: Allocator<T, R1, C1> + Allocator<T, SameShapeR<R1, R2>, SameShapeC<C1, C2>>,
    ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2>,
{
}

// XXX: Bad name.
/// Restricts the given number of rows to be equal.
pub trait SameShapeVectorAllocator<T, R1, R2>:
    Allocator<T, R1> + Allocator<T, SameShapeR<R1, R2>> + SameShapeAllocator<T, R1, U1, R2, U1>
where
    R1: Dim,
    R2: Dim,
    T: Scalar,
    ShapeConstraint: SameNumberOfRows<R1, R2>,
{
}

impl<T, R1, R2> SameShapeVectorAllocator<T, R1, R2> for DefaultAllocator
where
    R1: Dim,
    R2: Dim,
    T: Scalar,
    DefaultAllocator: Allocator<T, R1, U1> + Allocator<T, SameShapeR<R1, R2>>,
    ShapeConstraint: SameNumberOfRows<R1, R2>,
{
}
