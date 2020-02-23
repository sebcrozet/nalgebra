use alga::general::ComplexField;

use crate::base::allocator::Allocator;
use crate::base::dimension::Dim;
use crate::base::storage::{Storage, StorageMut};
use crate::base::{DefaultAllocator, MatrixN, SquareMatrix};

use crate::linalg::lu;

impl<N: ComplexField, D: Dim, S: Storage<N, D, D>> SquareMatrix<N, D, S> {
    /// Attempts to invert this matrix.
    #[inline]
    #[must_use = "Did you mean to use try_inverse_mut()?"]
    pub fn try_inverse(self) -> Option<MatrixN<N, D>>
    where DefaultAllocator: Allocator<N, D, D> {
        let mut me = self.into_owned();
        if me.try_inverse_mut() {
            Some(me)
        } else {
            None
        }
    }
}

impl<N: ComplexField, D: Dim, S: StorageMut<N, D, D>> SquareMatrix<N, D, S> {
    /// Attempts to invert this matrix in-place. Returns `false` and leaves `self` untouched if
    /// inversion fails.
    #[inline]
    pub fn try_inverse_mut(&mut self) -> bool
    where DefaultAllocator: Allocator<N, D, D> {
        assert!(self.is_square(), "Unable to invert a non-square matrix.");

        let dim = self.shape().0;

        unsafe {
            match dim {
                0 => true,
                1 => {
                    let determinant = self.get_unchecked((0, 0)).clone();
                    if determinant.is_zero() {
                        false
                    } else {
                        *self.get_unchecked_mut((0, 0)) = N::one() / determinant;
                        true
                    }
                }
                2 => {
                    let m11 = *self.get_unchecked((0, 0));
                    let m12 = *self.get_unchecked((0, 1));
                    let m21 = *self.get_unchecked((1, 0));
                    let m22 = *self.get_unchecked((1, 1));

                    let determinant = m11 * m22 - m21 * m12;

                    if determinant.is_zero() {
                        false
                    } else {
                        *self.get_unchecked_mut((0, 0)) = m22 / determinant;
                        *self.get_unchecked_mut((0, 1)) = -m12 / determinant;

                        *self.get_unchecked_mut((1, 0)) = -m21 / determinant;
                        *self.get_unchecked_mut((1, 1)) = m11 / determinant;

                        true
                    }
                }
                3 => {
                    let m11 = *self.get_unchecked((0, 0));
                    let m12 = *self.get_unchecked((0, 1));
                    let m13 = *self.get_unchecked((0, 2));

                    let m21 = *self.get_unchecked((1, 0));
                    let m22 = *self.get_unchecked((1, 1));
                    let m23 = *self.get_unchecked((1, 2));

                    let m31 = *self.get_unchecked((2, 0));
                    let m32 = *self.get_unchecked((2, 1));
                    let m33 = *self.get_unchecked((2, 2));

                    let minor_m12_m23 = m22 * m33 - m32 * m23;
                    let minor_m11_m23 = m21 * m33 - m31 * m23;
                    let minor_m11_m22 = m21 * m32 - m31 * m22;

                    let determinant =
                        m11 * minor_m12_m23 - m12 * minor_m11_m23 + m13 * minor_m11_m22;

                    if determinant.is_zero() {
                        false
                    } else {
                        *self.get_unchecked_mut((0, 0)) = minor_m12_m23 / determinant;
                        *self.get_unchecked_mut((0, 1)) = (m13 * m32 - m33 * m12) / determinant;
                        *self.get_unchecked_mut((0, 2)) = (m12 * m23 - m22 * m13) / determinant;

                        *self.get_unchecked_mut((1, 0)) = -minor_m11_m23 / determinant;
                        *self.get_unchecked_mut((1, 1)) = (m11 * m33 - m31 * m13) / determinant;
                        *self.get_unchecked_mut((1, 2)) = (m13 * m21 - m23 * m11) / determinant;

                        *self.get_unchecked_mut((2, 0)) = minor_m11_m22 / determinant;
                        *self.get_unchecked_mut((2, 1)) = (m12 * m31 - m32 * m11) / determinant;
                        *self.get_unchecked_mut((2, 2)) = (m11 * m22 - m21 * m12) / determinant;

                        true
                    }
                }
                4 => {
                    let oself = self.clone_owned();
                    do_inverse4(&oself, self)
                }
                _ => {
                    let oself = self.clone_owned();
                    lu::try_invert_to(oself, self)
                }
            }
        }
    }
}

// NOTE: this is an extremely efficient, loop-unrolled matrix inverse from MESA (MIT licensed).
fn do_inverse4<N: ComplexField, D: Dim, S: StorageMut<N, D, D>>(
    m: &MatrixN<N, D>,
    out: &mut SquareMatrix<N, D, S>,
) -> bool
where
    DefaultAllocator: Allocator<N, D, D>,
{
    let m = m.data.as_slice();

    out[(0, 0)] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15]
        + m[9] * m[7] * m[14]
        + m[13] * m[6] * m[11]
        - m[13] * m[7] * m[10];

    out[(1, 0)] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15]
        - m[9] * m[3] * m[14]
        - m[13] * m[2] * m[11]
        + m[13] * m[3] * m[10];

    out[(2, 0)] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15]
        + m[5] * m[3] * m[14]
        + m[13] * m[2] * m[7]
        - m[13] * m[3] * m[6];

    out[(3, 0)] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11]
        - m[5] * m[3] * m[10]
        - m[9] * m[2] * m[7]
        + m[9] * m[3] * m[6];

    out[(0, 1)] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15]
        - m[8] * m[7] * m[14]
        - m[12] * m[6] * m[11]
        + m[12] * m[7] * m[10];

    out[(1, 1)] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15]
        + m[8] * m[3] * m[14]
        + m[12] * m[2] * m[11]
        - m[12] * m[3] * m[10];

    out[(2, 1)] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15]
        - m[4] * m[3] * m[14]
        - m[12] * m[2] * m[7]
        + m[12] * m[3] * m[6];

    out[(3, 1)] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11]
        + m[4] * m[3] * m[10]
        + m[8] * m[2] * m[7]
        - m[8] * m[3] * m[6];

    out[(0, 2)] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15]
        + m[8] * m[7] * m[13]
        + m[12] * m[5] * m[11]
        - m[12] * m[7] * m[9];

    out[(1, 2)] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15]
        - m[8] * m[3] * m[13]
        - m[12] * m[1] * m[11]
        + m[12] * m[3] * m[9];

    out[(2, 2)] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15]
        + m[4] * m[3] * m[13]
        + m[12] * m[1] * m[7]
        - m[12] * m[3] * m[5];

    out[(0, 3)] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14]
        - m[8] * m[6] * m[13]
        - m[12] * m[5] * m[10]
        + m[12] * m[6] * m[9];

    out[(3, 2)] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11]
        - m[4] * m[3] * m[9]
        - m[8] * m[1] * m[7]
        + m[8] * m[3] * m[5];

    out[(1, 3)] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14]
        + m[8] * m[2] * m[13]
        + m[12] * m[1] * m[10]
        - m[12] * m[2] * m[9];

    out[(2, 3)] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14]
        - m[4] * m[2] * m[13]
        - m[12] * m[1] * m[6]
        + m[12] * m[2] * m[5];

    out[(3, 3)] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10]
        + m[4] * m[2] * m[9]
        + m[8] * m[1] * m[6]
        - m[8] * m[2] * m[5];

    let det = m[0] * out[(0, 0)] + m[1] * out[(0, 1)] + m[2] * out[(0, 2)] + m[3] * out[(0, 3)];

    if !det.is_zero() {
        let inv_det = N::one() / det;

        for j in 0..4 {
            for i in 0..4 {
                out[(i, j)] *= inv_det;
            }
        }
        true
    } else {
        false
    }
}
