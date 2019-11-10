// Matrix properties checks.
use approx::RelativeEq;
use num::{One, Zero};

use alga::general::{ClosedAdd, ClosedMul, RealField, ComplexField};

use crate::base::allocator::Allocator;
use crate::{Dim, DimMin};
use crate::base::storage::Storage;
use crate::base::{DefaultAllocator, Matrix, Scalar, SquareMatrix};

impl<N: Scalar, R: Dim, C: Dim, S: Storage<N, R, C>> Matrix<N, R, C, S> {
    /// Indicates if this is an empty matrix.
    #[inline]
    pub fn is_empty(&self) -> bool {
        let (nrows, ncols) = self.shape();
        nrows == 0 || ncols == 0
    }

    /// Indicates if this is a square matrix.
    #[inline]
    pub fn is_square(&self) -> bool {
        let (nrows, ncols) = self.shape();
        nrows == ncols
    }

    // FIXME: RelativeEq prevents us from using those methods on integer matrices…
    /// Indicated if this is the identity matrix within a relative error of `eps`.
    ///
    /// If the matrix is diagonal, this checks that diagonal elements (i.e. at coordinates `(i, i)`
    /// for i from `0` to `min(R, C)`) are equal one; and that all other elements are zero.
    #[inline]
    pub fn is_identity(&self, eps: N::Epsilon) -> bool
    where
        N: Zero + One + RelativeEq,
        N::Epsilon: Copy,
    {
        let (nrows, ncols) = self.shape();
        let d;

        if nrows > ncols {
            d = ncols;

            for i in d..nrows {
                for j in 0..ncols {
                    if !relative_eq!(self[(i, j)], N::zero(), epsilon = eps) {
                        return false;
                    }
                }
            }
        } else {
            // nrows <= ncols
            d = nrows;

            for i in 0..nrows {
                for j in d..ncols {
                    if !relative_eq!(self[(i, j)], N::zero(), epsilon = eps) {
                        return false;
                    }
                }
            }
        }

        // Off-diagonal elements of the sub-square matrix.
        for i in 1..d {
            for j in 0..i {
                // FIXME: use unsafe indexing.
                if !relative_eq!(self[(i, j)], N::zero(), epsilon = eps)
                    || !relative_eq!(self[(j, i)], N::zero(), epsilon = eps)
                {
                    return false;
                }
            }
        }

        // Diagonal elements of the sub-square matrix.
        for i in 0..d {
            if !relative_eq!(self[(i, i)], N::one(), epsilon = eps) {
                return false;
            }
        }

        true
    }
}

impl<N: ComplexField, R: Dim, C: Dim, S: Storage<N, R, C>> Matrix<N, R, C, S> {
    /// Checks that `Mᵀ × M = Id`.
    ///
    /// In this definition `Id` is approximately equal to the identity matrix with a relative error
    /// equal to `eps`.
    #[inline]
    pub fn is_orthogonal(&self, eps: N::Epsilon) -> bool
        where
            N: Zero + One + ClosedAdd + ClosedMul + RelativeEq,
            S: Storage<N, R, C>,
            N::Epsilon: Copy,
            DefaultAllocator: Allocator<N, R, C> + Allocator<N, C, C>,
    {
        (self.ad_mul(self)).is_identity(eps)
    }
}

impl<N: RealField, D: Dim, S: Storage<N, D, D>> SquareMatrix<N, D, S>
where DefaultAllocator: Allocator<N, D, D>
{
    /// Checks that this matrix is orthogonal and has a determinant equal to 1.
    #[inline]
    pub fn is_special_orthogonal(&self, eps: N) -> bool
    where
        D: DimMin<D, Output = D>,
        DefaultAllocator: Allocator<(usize, usize), D>,
    {
        self.is_square() && self.is_orthogonal(eps) && self.determinant() > N::zero()
    }

    /// Returns `true` if this matrix is invertible.
    #[inline]
    pub fn is_invertible(&self) -> bool {
        // FIXME: improve this?
        self.clone_owned().try_inverse().is_some()
    }
}
