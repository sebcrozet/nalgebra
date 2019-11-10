//! [Reexported at the root of this crate.] Data structures for vector and matrix computations.

pub mod allocator;
mod blas;
pub mod constraint;
pub mod coordinates;
mod default_allocator;
pub mod iter;
mod ops;
pub mod storage;

mod alias;
mod alias_slice;
mod cg;
mod componentwise;
mod construction;
mod construction_slice;
mod conversion;
mod edition;
pub mod indexing;
mod matrix;
mod matrix_alga;
mod array_storage;
mod matrix_slice;
#[cfg(any(feature = "std", feature = "alloc"))]
mod vec_storage;
mod properties;
mod scalar;
mod swizzle;
mod unit;
mod statistics;
mod norm;

#[doc(hidden)]
pub mod helper;

pub use self::matrix::*;
pub use self::scalar::*;
pub use self::unit::*;
pub use self::norm::*;

pub use self::default_allocator::*;

pub use self::alias::*;
pub use self::alias_slice::*;
pub use self::array_storage::*;
pub use self::matrix_slice::*;
#[cfg(any(feature = "std", feature = "alloc"))]
pub use self::vec_storage::*;
