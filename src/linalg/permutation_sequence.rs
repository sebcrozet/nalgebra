#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use alga::general::ClosedNeg;
use num::One;

use crate::allocator::Allocator;
use crate::base::{DefaultAllocator, Matrix, Scalar, VectorN};
#[cfg(any(feature = "std", feature = "alloc"))]
use crate::Dynamic;
use crate::{Dim, DimName, U1};
use crate::storage::StorageMut;

/// A sequence of row or column permutations.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(
        serialize = "DefaultAllocator: Allocator<(usize, usize), D>,
         VectorN<(usize, usize), D>: Serialize"
    ))
)]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(
        deserialize = "DefaultAllocator: Allocator<(usize, usize), D>,
         VectorN<(usize, usize), D>: Deserialize<'de>"
    ))
)]
#[derive(Clone, Debug)]
pub struct PermutationSequence<D: Dim>
where DefaultAllocator: Allocator<(usize, usize), D>
{
    len: usize,
    ipiv: VectorN<(usize, usize), D>,
}

impl<D: Dim> Copy for PermutationSequence<D>
where
    DefaultAllocator: Allocator<(usize, usize), D>,
    VectorN<(usize, usize), D>: Copy,
{}

impl<D: DimName> PermutationSequence<D>
where DefaultAllocator: Allocator<(usize, usize), D>
{
    /// Creates a new statically-allocated sequence of `D` identity permutations.
    #[inline]
    pub fn identity() -> Self {
        Self::identity_generic(D::name())
    }
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl PermutationSequence<Dynamic>
where DefaultAllocator: Allocator<(usize, usize), Dynamic>
{
    /// Creates a new dynamically-allocated sequence of `n` identity permutations.
    #[inline]
    pub fn identity(n: usize) -> Self {
        Self::identity_generic(Dynamic::new(n))
    }
}

impl<D: Dim> PermutationSequence<D>
where DefaultAllocator: Allocator<(usize, usize), D>
{
    /// Creates a new sequence of D identity permutations.
    #[inline]
    pub fn identity_generic(dim: D) -> Self {
        unsafe {
            Self {
                len: 0,
                ipiv: VectorN::new_uninitialized_generic(dim, U1),
            }
        }
    }

    /// Adds the interchange of the row (or column) `i` with the row (or column) `i2` to this
    /// sequence of permutations.
    #[inline]
    pub fn append_permutation(&mut self, i: usize, i2: usize) {
        if i != i2 {
            assert!(
                self.len < self.ipiv.len(),
                "Maximum number of permutations exceeded."
            );
            self.ipiv[self.len] = (i, i2);
            self.len += 1;
        }
    }

    /// Applies this sequence of permutations to the rows of `rhs`.
    #[inline]
    pub fn permute_rows<N: Scalar, R2: Dim, C2: Dim, S2>(&self, rhs: &mut Matrix<N, R2, C2, S2>)
    where S2: StorageMut<N, R2, C2> {
        for i in self.ipiv.rows_range(..self.len).iter() {
            rhs.swap_rows(i.0, i.1)
        }
    }

    /// Applies this sequence of permutations in reverse to the rows of `rhs`.
    #[inline]
    pub fn inv_permute_rows<N: Scalar, R2: Dim, C2: Dim, S2>(
        &self,
        rhs: &mut Matrix<N, R2, C2, S2>,
    ) where
        S2: StorageMut<N, R2, C2>,
    {
        for i in 0..self.len {
            let (i1, i2) = self.ipiv[self.len - i - 1];
            rhs.swap_rows(i1, i2)
        }
    }

    /// Applies this sequence of permutations to the columns of `rhs`.
    #[inline]
    pub fn permute_columns<N: Scalar, R2: Dim, C2: Dim, S2>(
        &self,
        rhs: &mut Matrix<N, R2, C2, S2>,
    ) where
        S2: StorageMut<N, R2, C2>,
    {
        for i in self.ipiv.rows_range(..self.len).iter() {
            rhs.swap_columns(i.0, i.1)
        }
    }

    /// Applies this sequence of permutations in reverse to the columns of `rhs`.
    #[inline]
    pub fn inv_permute_columns<N: Scalar, R2: Dim, C2: Dim, S2>(
        &self,
        rhs: &mut Matrix<N, R2, C2, S2>,
    ) where
        S2: StorageMut<N, R2, C2>,
    {
        for i in 0..self.len {
            let (i1, i2) = self.ipiv[self.len - i - 1];
            rhs.swap_columns(i1, i2)
        }
    }

    /// The number of non-identity permutations applied by this sequence.
    pub fn len(&self) -> usize {
        self.len
    }

    /// The determinant of the matrix corresponding to this permutation.
    #[inline]
    pub fn determinant<N: One + ClosedNeg>(&self) -> N {
        if self.len % 2 == 0 {
            N::one()
        } else {
            -N::one()
        }
    }
}
