//! Construction of householder elementary reflections.

use num::Zero;
use alga::general::ComplexField;
use crate::allocator::Allocator;
use crate::base::{DefaultAllocator, MatrixMN, MatrixN, Unit, Vector, VectorN};
use crate::Dim;
use crate::storage::{Storage, StorageMut};

use crate::geometry::Reflection;

/// Replaces `column` by the axis of the householder reflection that transforms `column` into
/// `(+/-|column|, 0, ..., 0)`.
///
/// The unit-length axis is output to `column`. Returns what would be the first component of
/// `column` after reflection and `false` if no reflection was necessary.
#[doc(hidden)]
#[inline(always)]
pub fn reflection_axis_mut<N: ComplexField, D: Dim, S: StorageMut<N, D>>(
    column: &mut Vector<N, D, S>,
) -> (N, bool) {
    let reflection_sq_norm = column.norm_squared();
    let reflection_norm = reflection_sq_norm.sqrt();

    let factor;
    let signed_norm;

    unsafe {
        let (modulus, sign) = column.vget_unchecked(0).to_exp();
        signed_norm = sign.scale(reflection_norm);
        factor = (reflection_sq_norm + modulus * reflection_norm) * crate::convert(2.0);
        *column.vget_unchecked_mut(0) += signed_norm;
    };

    if !factor.is_zero() {
        column.unscale_mut(factor.sqrt());
        (-signed_norm, true)
    } else {
        // FIXME: not sure why we don't have a - sign here.
        (signed_norm, false)
    }
}

/// Uses an householder reflection to zero out the `icol`-th column, starting with the `shift + 1`-th
/// subdiagonal element.
#[doc(hidden)]
pub fn clear_column_unchecked<N: ComplexField, R: Dim, C: Dim>(
    matrix: &mut MatrixMN<N, R, C>,
    diag_elt: &mut N,
    icol: usize,
    shift: usize,
    bilateral: Option<&mut VectorN<N, R>>,
) where
    DefaultAllocator: Allocator<N, R, C> + Allocator<N, R>,
{
    let (mut left, mut right) = matrix.columns_range_pair_mut(icol, icol + 1..);
    let mut axis = left.rows_range_mut(icol + shift..);

    let (reflection_norm, not_zero) = reflection_axis_mut(&mut axis);
    *diag_elt = reflection_norm;

    if not_zero {
        let refl = Reflection::new(Unit::new_unchecked(axis), N::zero());
        let sign = reflection_norm.signum();
        if let Some(mut work) = bilateral {
            refl.reflect_rows_with_sign(&mut right, &mut work, sign);
        }
        refl.reflect_with_sign(&mut right.rows_range_mut(icol + shift..), sign.conjugate());
    }
}

/// Uses an householder reflection to zero out the `irow`-th row, ending before the `shift + 1`-th
/// superdiagonal element.
#[doc(hidden)]
pub fn clear_row_unchecked<N: ComplexField, R: Dim, C: Dim>(
    matrix: &mut MatrixMN<N, R, C>,
    diag_elt: &mut N,
    axis_packed: &mut VectorN<N, C>,
    work: &mut VectorN<N, R>,
    irow: usize,
    shift: usize,
) where
    DefaultAllocator: Allocator<N, R, C> + Allocator<N, R> + Allocator<N, C>,
{
    let (mut top, mut bottom) = matrix.rows_range_pair_mut(irow, irow + 1..);
    let mut axis = axis_packed.rows_range_mut(irow + shift..);
    axis.tr_copy_from(&top.columns_range(irow + shift..));

    let (reflection_norm, not_zero) = reflection_axis_mut(&mut axis);
    axis.conjugate_mut(); // So that reflect_rows actually cancels the first row.
    *diag_elt = reflection_norm;

    if not_zero {
        let refl = Reflection::new(Unit::new_unchecked(axis), N::zero());
        refl.reflect_rows_with_sign(
            &mut bottom.columns_range_mut(irow + shift..),
            &mut work.rows_range_mut(irow + 1..),
            reflection_norm.signum().conjugate(),
        );
        top.columns_range_mut(irow + shift..)
            .tr_copy_from(&refl.axis());
    } else {
        top.columns_range_mut(irow + shift..).tr_copy_from(&axis);
    }
}

/// Computes the orthogonal transformation described by the elementary reflector axii stored on
/// the lower-diagonal element of the given matrix.
/// matrices.
#[doc(hidden)]
pub fn assemble_q<N: ComplexField, D: Dim>(m: &MatrixN<N, D>, signs: &[N]) -> MatrixN<N, D>
where DefaultAllocator: Allocator<N, D, D> {
    assert!(m.is_square());
    let dim = m.data.shape().0;

    // NOTE: we could build the identity matrix and call p_mult on it.
    // Instead we don't so that we take in account the matrix sparseness.
    let mut res = MatrixN::identity_generic(dim, dim);

    for i in (0..dim.value() - 1).rev() {
        let axis = m.slice_range(i + 1.., i);
        let refl = Reflection::new(Unit::new_unchecked(axis), N::zero());

        let mut res_rows = res.slice_range_mut(i + 1.., i..);
        refl.reflect_with_sign(&mut res_rows, signs[i].signum());
    }

    res
}
