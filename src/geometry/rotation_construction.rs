use num::{One, Zero};

use alga::general::{ClosedAdd, ClosedMul};

use crate::base::allocator::Allocator;
use crate::DimName;
use crate::base::{DefaultAllocator, MatrixN, Scalar};

use crate::geometry::Rotation;

impl<N, D: DimName> Rotation<N, D>
where
    N: Scalar + Zero + One,
    DefaultAllocator: Allocator<N, D, D>,
{
    /// Creates a new square identity rotation of the given `dimension`.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::Quaternion;
    /// let rot1 = Quaternion::identity();
    /// let rot2 = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    ///
    /// assert_eq!(rot1 * rot2, rot2);
    /// assert_eq!(rot2 * rot1, rot2);
    /// ```
    #[inline]
    pub fn identity() -> Rotation<N, D> {
        Self::from_matrix_unchecked(MatrixN::<N, D>::identity())
    }
}

impl<N, D: DimName> One for Rotation<N, D>
where
    N: Scalar + Zero + One + ClosedAdd + ClosedMul,
    DefaultAllocator: Allocator<N, D, D>,
{
    #[inline]
    fn one() -> Self {
        Self::identity()
    }
}
