#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use num::{Zero, One};
use approx::AbsDiffEq;

use alga::general::{RealField, ComplexField};
use crate::allocator::Allocator;
use crate::base::{DefaultAllocator, Matrix, Matrix2x3, MatrixMN, Vector2, VectorN};
use crate::constraint::{SameNumberOfRows, ShapeConstraint};
use crate::{Dim, DimDiff, DimMin, DimMinimum, DimSub, U1, U2};
use crate::storage::Storage;

use crate::linalg::symmetric_eigen;
use crate::linalg::Bidiagonal;
use crate::linalg::givens::GivensRotation;

/// Singular Value Decomposition of a general matrix.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(
        serialize = "DefaultAllocator: Allocator<N::RealField, DimMinimum<R, C>>    +
                           Allocator<N, DimMinimum<R, C>, C> +
                           Allocator<N, R, DimMinimum<R, C>>,
         MatrixMN<N, R, DimMinimum<R, C>>: Serialize,
         MatrixMN<N, DimMinimum<R, C>, C>: Serialize,
         VectorN<N::RealField, DimMinimum<R, C>>: Serialize"
    ))
)]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(
        deserialize = "DefaultAllocator: Allocator<N::RealField, DimMinimum<R, C>>    +
                           Allocator<N, DimMinimum<R, C>, C> +
                           Allocator<N, R, DimMinimum<R, C>>,
         MatrixMN<N, R, DimMinimum<R, C>>: Deserialize<'de>,
         MatrixMN<N, DimMinimum<R, C>, C>: Deserialize<'de>,
         VectorN<N::RealField, DimMinimum<R, C>>: Deserialize<'de>"
    ))
)]
#[derive(Clone, Debug)]
pub struct SVD<N: ComplexField, R: DimMin<C>, C: Dim>
where DefaultAllocator: Allocator<N, DimMinimum<R, C>, C>
        + Allocator<N, R, DimMinimum<R, C>>
        + Allocator<N::RealField, DimMinimum<R, C>>
{
    /// The left-singular vectors `U` of this SVD.
    pub u: Option<MatrixMN<N, R, DimMinimum<R, C>>>,
    /// The right-singular vectors `V^t` of this SVD.
    pub v_t: Option<MatrixMN<N, DimMinimum<R, C>, C>>,
    /// The singular values of this SVD.
    pub singular_values: VectorN<N::RealField, DimMinimum<R, C>>,
}

impl<N: ComplexField, R: DimMin<C>, C: Dim> Copy for SVD<N, R, C>
where
    DefaultAllocator: Allocator<N, DimMinimum<R, C>, C>
        + Allocator<N, R, DimMinimum<R, C>>
        + Allocator<N::RealField, DimMinimum<R, C>>,
    MatrixMN<N, R, DimMinimum<R, C>>: Copy,
    MatrixMN<N, DimMinimum<R, C>, C>: Copy,
    VectorN<N::RealField, DimMinimum<R, C>>: Copy,
{}

impl<N: ComplexField, R: DimMin<C>, C: Dim> SVD<N, R, C>
where
    DimMinimum<R, C>: DimSub<U1>, // for Bidiagonal.
    DefaultAllocator: Allocator<N, R, C>
        + Allocator<N, C>
        + Allocator<N, R>
        + Allocator<N, DimDiff<DimMinimum<R, C>, U1>>
        + Allocator<N, DimMinimum<R, C>, C>
        + Allocator<N, R, DimMinimum<R, C>>
        + Allocator<N, DimMinimum<R, C>>
        + Allocator<N::RealField, DimMinimum<R, C>>
        + Allocator<N::RealField, DimDiff<DimMinimum<R, C>, U1>>,
{
    /// Computes the Singular Value Decomposition of `matrix` using implicit shift.
    pub fn new(matrix: MatrixMN<N, R, C>, compute_u: bool, compute_v: bool) -> Self {
        Self::try_new(matrix, compute_u, compute_v, N::RealField::default_epsilon(), 0).unwrap()
    }

    /// Attempts to compute the Singular Value Decomposition of `matrix` using implicit shift.
    ///
    /// # Arguments
    ///
    /// * `compute_u` − set this to `true` to enable the computation of left-singular vectors.
    /// * `compute_v` − set this to `true` to enable the computation of left-singular vectors.
    /// * `eps`       − tolerance used to determine when a value converged to 0.
    /// * `max_niter` − maximum total number of iterations performed by the algorithm. If this
    /// number of iteration is exceeded, `None` is returned. If `niter == 0`, then the algorithm
    /// continues indefinitely until convergence.
    pub fn try_new(
        mut matrix: MatrixMN<N, R, C>,
        compute_u: bool,
        compute_v: bool,
        eps: N::RealField,
        max_niter: usize,
    ) -> Option<Self>
    {
        assert!(
            matrix.len() != 0,
            "Cannot compute the SVD of an empty matrix."
        );
        let (nrows, ncols) = matrix.data.shape();
        let min_nrows_ncols = nrows.min(ncols);
        let dim = min_nrows_ncols.value();

        let m_amax = matrix.camax();

        if !m_amax.is_zero() {
            matrix.unscale_mut(m_amax);
        }

        let b = Bidiagonal::new(matrix);
        let mut u = if compute_u { Some(b.u()) } else { None };
        let mut v_t = if compute_v { Some(b.v_t()) } else { None };
        let mut diagonal = b.diagonal();
        let mut off_diagonal = b.off_diagonal();

        let mut niter = 0;
        let (mut start, mut end) = Self::delimit_subproblem(&mut diagonal, &mut off_diagonal, &mut u, &mut v_t, b.is_upper_diagonal(), dim - 1, eps);

        while end != start {
            let subdim = end - start + 1;

            // Solve the subproblem.
            if subdim > 2 {
                let m = end - 1;
                let n = end;

                let mut vec;
                {
                    let dm = diagonal[m];
                    let dn = diagonal[n];
                    let fm = off_diagonal[m];

                    let tmm = dm * dm + off_diagonal[m - 1] * off_diagonal[m - 1];
                    let tmn = dm * fm;
                    let tnn = dn * dn + fm * fm;

                    let shift = symmetric_eigen::wilkinson_shift(tmm, tnn, tmn);

                    vec = Vector2::new(
                        diagonal[start] * diagonal[start] - shift,
                        diagonal[start] * off_diagonal[start],
                    );
                }

                for k in start..n {
                    let m12 = if k == n - 1 {
                        N::RealField::zero()
                    } else {
                        off_diagonal[k + 1]
                    };

                    let mut subm = Matrix2x3::new(
                        diagonal[k],
                        off_diagonal[k],
                        N::RealField::zero(),
                        N::RealField::zero(),
                        diagonal[k + 1],
                        m12,
                    );

                    if let Some((rot1, norm1)) = GivensRotation::cancel_y(&vec) {
                        rot1.inverse().rotate_rows(&mut subm.fixed_columns_mut::<U2>(0));
                        let rot1 = GivensRotation::new_unchecked(rot1.c(), N::from_real(rot1.s()));

                        if k > start {
                            // This is not the first iteration.
                            off_diagonal[k - 1] = norm1;
                        }

                        let v = Vector2::new(subm[(0, 0)], subm[(1, 0)]);
                        // FIXME: does the case `v.y == 0` ever happen?
                        let (rot2, norm2) =
                            GivensRotation::cancel_y(&v).unwrap_or((GivensRotation::identity(), subm[(0, 0)]));

                        rot2.rotate(&mut subm.fixed_columns_mut::<U2>(1));
                        let rot2 = GivensRotation::new_unchecked(rot2.c(), N::from_real(rot2.s()));

                        subm[(0, 0)] = norm2;

                        if let Some(ref mut v_t) = v_t {
                            if b.is_upper_diagonal() {
                                rot1.rotate(&mut v_t.fixed_rows_mut::<U2>(k));
                            } else {
                                rot2.rotate(&mut v_t.fixed_rows_mut::<U2>(k));
                            }
                        }

                        if let Some(ref mut u) = u {
                            if b.is_upper_diagonal() {
                                rot2.inverse()
                                    .rotate_rows(&mut u.fixed_columns_mut::<U2>(k));
                            } else {
                                rot1.inverse()
                                    .rotate_rows(&mut u.fixed_columns_mut::<U2>(k));
                            }
                        }

                        diagonal[k + 0] = subm[(0, 0)];
                        diagonal[k + 1] = subm[(1, 1)];
                        off_diagonal[k + 0] = subm[(0, 1)];

                        if k != n - 1 {
                            off_diagonal[k + 1] = subm[(1, 2)];
                        }

                        vec.x = subm[(0, 1)];
                        vec.y = subm[(0, 2)];
                    } else {
                        break;
                    }
                }
            } else if subdim == 2 {
                // Solve the remaining 2x2 subproblem.
                let (u2, s, v2) = compute_2x2_uptrig_svd(
                    diagonal[start],
                    off_diagonal[start],
                    diagonal[start + 1],
                    compute_u && b.is_upper_diagonal() || compute_v && !b.is_upper_diagonal(),
                    compute_v && b.is_upper_diagonal() || compute_u && !b.is_upper_diagonal(),
                );
                let u2 = u2.map(|u2| GivensRotation::new_unchecked(u2.c(), N::from_real(u2.s())));
                let v2 = v2.map(|v2| GivensRotation::new_unchecked(v2.c(), N::from_real(v2.s())));

                diagonal[start + 0] = s[0];
                diagonal[start + 1] = s[1];
                off_diagonal[start] = N::RealField::zero();

                if let Some(ref mut u) = u {
                    let rot = if b.is_upper_diagonal() {
                        u2.unwrap()
                    } else {
                        v2.unwrap()
                    };
                    rot.rotate_rows(&mut u.fixed_columns_mut::<U2>(start));
                }

                if let Some(ref mut v_t) = v_t {
                    let rot = if b.is_upper_diagonal() {
                        v2.unwrap()
                    } else {
                        u2.unwrap()
                    };
                    rot.inverse().rotate(&mut v_t.fixed_rows_mut::<U2>(start));
                }

                end -= 1;
            }

            // Re-delimit the subproblem in case some decoupling occurred.
            let sub = Self::delimit_subproblem(&mut diagonal, &mut off_diagonal, &mut u, &mut v_t, b.is_upper_diagonal(), end, eps);
            start = sub.0;
            end = sub.1;

            niter += 1;
            if niter == max_niter {
                return None;
            }
        }

        diagonal *= m_amax;

        // Ensure all singular value are non-negative.
        for i in 0..dim {
            let sval = diagonal[i];

            if sval < N::RealField::zero() {
                diagonal[i] = -sval;

                if let Some(ref mut u) = u {
                    u.column_mut(i).neg_mut();
                }
            }
        }

        Some(Self {
            u,
            v_t,
            singular_values: diagonal,
        })
    }

    /*
    fn display_bidiag(b: &Bidiagonal<N, R, C>, begin: usize, end: usize) {
        for i in begin .. end {
            for k in begin .. i {
                print!("    ");
            }
            println!("{}  {}", b.diagonal[i], b.off_diagonal[i]);
        }
        for k in begin .. end {
            print!("    ");
        }
        println!("{}", b.diagonal[end]);
    }
    */

    fn delimit_subproblem(
        diagonal: &mut VectorN<N::RealField, DimMinimum<R, C>>,
        off_diagonal: &mut VectorN<N::RealField, DimDiff<DimMinimum<R, C>, U1>>,
        u: &mut Option<MatrixMN<N, R, DimMinimum<R, C>>>,
        v_t: &mut Option<MatrixMN<N, DimMinimum<R, C>, C>>,
        is_upper_diagonal: bool,
        end: usize,
        eps: N::RealField,
    ) -> (usize, usize)
    {
        let mut n = end;

        while n > 0 {
            let m = n - 1;

            if off_diagonal[m].is_zero()
                || off_diagonal[m].norm1() <= eps * (diagonal[n].norm1() + diagonal[m].norm1())
            {
                off_diagonal[m] = N::RealField::zero();
            } else if diagonal[m].norm1() <= eps {
                diagonal[m] = N::RealField::zero();
                Self::cancel_horizontal_off_diagonal_elt(diagonal, off_diagonal, u, v_t, is_upper_diagonal, m, m + 1);

                if m != 0 {
                    Self::cancel_vertical_off_diagonal_elt(diagonal, off_diagonal, u, v_t, is_upper_diagonal, m - 1);
                }
            } else if diagonal[n].norm1() <= eps {
                diagonal[n] = N::RealField::zero();
                Self::cancel_vertical_off_diagonal_elt(diagonal, off_diagonal, u, v_t, is_upper_diagonal, m);
            } else {
                break;
            }

            n -= 1;
        }

        if n == 0 {
            return (0, 0);
        }

        let mut new_start = n - 1;
        while new_start > 0 {
            let m = new_start - 1;

            if off_diagonal[m].norm1() <= eps * (diagonal[new_start].norm1() + diagonal[m].norm1())
            {
                off_diagonal[m] = N::RealField::zero();
                break;
            }
            // FIXME: write a test that enters this case.
            else if diagonal[m].norm1() <= eps {
                diagonal[m] = N::RealField::zero();
                Self::cancel_horizontal_off_diagonal_elt(diagonal, off_diagonal, u, v_t, is_upper_diagonal, m, n);

                if m != 0 {
                    Self::cancel_vertical_off_diagonal_elt(diagonal, off_diagonal, u, v_t, is_upper_diagonal, m - 1);
                }
                break;
            }

            new_start -= 1;
        }

        (new_start, n)
    }

    // Cancels the i-th off-diagonal element using givens rotations.
    fn cancel_horizontal_off_diagonal_elt(
        diagonal: &mut VectorN<N::RealField, DimMinimum<R, C>>,
        off_diagonal: &mut VectorN<N::RealField, DimDiff<DimMinimum<R, C>, U1>>,
        u: &mut Option<MatrixMN<N, R, DimMinimum<R, C>>>,
        v_t: &mut Option<MatrixMN<N, DimMinimum<R, C>, C>>,
        is_upper_diagonal: bool,
        i: usize,
        end: usize,
    )
    {
        let mut v = Vector2::new(off_diagonal[i], diagonal[i + 1]);
        off_diagonal[i] = N::RealField::zero();

        for k in i..end {
            if let Some((rot, norm)) = GivensRotation::cancel_x(&v) {
                let rot = GivensRotation::new_unchecked(rot.c(), N::from_real(rot.s()));
                diagonal[k + 1] = norm;

                if is_upper_diagonal {
                    if let Some(ref mut u) = *u {
                        rot.inverse()
                            .rotate_rows(&mut u.fixed_columns_with_step_mut::<U2>(i, k - i));
                    }
                } else if let Some(ref mut v_t) = *v_t {
                    rot.rotate(&mut v_t.fixed_rows_with_step_mut::<U2>(i, k - i));
                }

                if k + 1 != end {
                    v.x = -rot.s().real() * off_diagonal[k + 1];
                    v.y = diagonal[k + 2];
                    off_diagonal[k + 1] *= rot.c();
                }
            } else {
                break;
            }
        }
    }

    // Cancels the i-th off-diagonal element using givens rotations.
    fn cancel_vertical_off_diagonal_elt(
        diagonal: &mut VectorN<N::RealField, DimMinimum<R, C>>,
        off_diagonal: &mut VectorN<N::RealField, DimDiff<DimMinimum<R, C>, U1>>,
        u: &mut Option<MatrixMN<N, R, DimMinimum<R, C>>>,
        v_t: &mut Option<MatrixMN<N, DimMinimum<R, C>, C>>,
        is_upper_diagonal: bool,
        i: usize,
    )
    {
        let mut v = Vector2::new(diagonal[i], off_diagonal[i]);
        off_diagonal[i] = N::RealField::zero();

        for k in (0..i + 1).rev() {
            if let Some((rot, norm)) = GivensRotation::cancel_y(&v) {
                let rot = GivensRotation::new_unchecked(rot.c(), N::from_real(rot.s()));
                diagonal[k] = norm;

                if is_upper_diagonal {
                    if let Some(ref mut v_t) = *v_t {
                        rot.rotate(&mut v_t.fixed_rows_with_step_mut::<U2>(k, i - k));
                    }
                } else if let Some(ref mut u) = *u {
                    rot.inverse()
                        .rotate_rows(&mut u.fixed_columns_with_step_mut::<U2>(k, i - k));
                }

                if k > 0 {
                    v.x = diagonal[k - 1];
                    v.y = rot.s().real() * off_diagonal[k - 1];
                    off_diagonal[k - 1] *= rot.c();
                }
            } else {
                break;
            }
        }
    }

    /// Computes the rank of the decomposed matrix, i.e., the number of singular values greater
    /// than `eps`.
    pub fn rank(&self, eps: N::RealField) -> usize {
        assert!(
            eps >= N::RealField::zero(),
            "SVD rank: the epsilon must be non-negative."
        );
        self.singular_values.iter().filter(|e| **e > eps).count()
    }

    /// Rebuild the original matrix.
    ///
    /// This is useful if some of the singular values have been manually modified.
    /// Returns `Err` if the right- and left- singular vectors have not been
    /// computed at construction-time.
    pub fn recompose(self) -> Result<MatrixMN<N, R, C>, &'static str> {
        match (self.u, self.v_t) {
            (Some(mut u), Some(v_t)) => {
                for i in 0..self.singular_values.len() {
                    let val = self.singular_values[i];
                    u.column_mut(i).scale_mut(val);
                }
                Ok(u * v_t)
            }
            (None, None) => Err("SVD recomposition: U and V^t have not been computed."),
            (None, _) => Err("SVD recomposition: U has not been computed."),
            (_, None) => Err("SVD recomposition: V^t has not been computed.")
        }
    }

    /// Computes the pseudo-inverse of the decomposed matrix.
    ///
    /// Any singular value smaller than `eps` is assumed to be zero.
    /// Returns `Err` if the right- and left- singular vectors have not
    /// been computed at construction-time.
    pub fn pseudo_inverse(mut self, eps: N::RealField) -> Result<MatrixMN<N, C, R>, &'static str>
    where
        DefaultAllocator: Allocator<N, C, R>,
    {
        if eps < N::RealField::zero() {
            Err("SVD pseudo inverse: the epsilon must be non-negative.")
        }
        else {
            for i in 0..self.singular_values.len() {
                let val = self.singular_values[i];

                if val > eps {
                    self.singular_values[i] = N::RealField::one() / val;
                } else {
                    self.singular_values[i] = N::RealField::zero();
                }
            }

            self.recompose().map(|m| m.adjoint())
        }
    }

    /// Solves the system `self * x = b` where `self` is the decomposed matrix and `x` the unknown.
    ///
    /// Any singular value smaller than `eps` is assumed to be zero.
    /// Returns `Err` if the singular vectors `U` and `V` have not been computed.
    // FIXME: make this more generic wrt the storage types and the dimensions for `b`.
    pub fn solve<R2: Dim, C2: Dim, S2>(
        &self,
        b: &Matrix<N, R2, C2, S2>,
        eps: N::RealField,
    ) -> Result<MatrixMN<N, C, C2>, &'static str>
    where
        S2: Storage<N, R2, C2>,
        DefaultAllocator: Allocator<N, C, C2> + Allocator<N, DimMinimum<R, C>, C2>,
        ShapeConstraint: SameNumberOfRows<R, R2>,
    {
        if eps < N::RealField::zero() {
            Err("SVD solve: the epsilon must be non-negative.")
        }
        else {
            match (&self.u, &self.v_t) {
                (Some(u), Some(v_t)) => {
                    let mut ut_b = u.ad_mul(b);

                    for j in 0..ut_b.ncols() {
                        let mut col = ut_b.column_mut(j);

                        for i in 0..self.singular_values.len() {
                            let val = self.singular_values[i];
                            if val > eps {
                                col[i] = col[i].unscale(val);
                            } else {
                                col[i] = N::zero();
                            }
                        }
                    }

                    Ok(v_t.ad_mul(&ut_b))
                }
                (None, None) => Err("SVD solve: U and V^t have not been computed."),
                (None, _) => Err("SVD solve: U has not been computed."),
                (_, None) => Err("SVD solve: V^t has not been computed.")
            }
        }
    }
}

impl<N: ComplexField, R: DimMin<C>, C: Dim, S: Storage<N, R, C>> Matrix<N, R, C, S>
where
    DimMinimum<R, C>: DimSub<U1>, // for Bidiagonal.
    DefaultAllocator: Allocator<N, R, C>
        + Allocator<N, C>
        + Allocator<N, R>
        + Allocator<N, DimDiff<DimMinimum<R, C>, U1>>
        + Allocator<N, DimMinimum<R, C>, C>
        + Allocator<N, R, DimMinimum<R, C>>
        + Allocator<N, DimMinimum<R, C>>
        + Allocator<N::RealField, DimMinimum<R, C>>
        + Allocator<N::RealField, DimDiff<DimMinimum<R, C>, U1>>,
{
    /// Computes the Singular Value Decomposition using implicit shift.
    pub fn svd(self, compute_u: bool, compute_v: bool) -> SVD<N, R, C> {
        SVD::new(self.into_owned(), compute_u, compute_v)
    }

    /// Attempts to compute the Singular Value Decomposition of `matrix` using implicit shift.
    ///
    /// # Arguments
    ///
    /// * `compute_u` − set this to `true` to enable the computation of left-singular vectors.
    /// * `compute_v` − set this to `true` to enable the computation of left-singular vectors.
    /// * `eps`       − tolerance used to determine when a value converged to 0.
    /// * `max_niter` − maximum total number of iterations performed by the algorithm. If this
    /// number of iteration is exceeded, `None` is returned. If `niter == 0`, then the algorithm
    /// continues indefinitely until convergence.
    pub fn try_svd(
        self,
        compute_u: bool,
        compute_v: bool,
        eps: N::RealField,
        max_niter: usize,
    ) -> Option<SVD<N, R, C>>
    {
        SVD::try_new(self.into_owned(), compute_u, compute_v, eps, max_niter)
    }

    /// Computes the singular values of this matrix.
    pub fn singular_values(&self) -> VectorN<N::RealField, DimMinimum<R, C>> {
        SVD::new(self.clone_owned(), false, false).singular_values
    }

    /// Computes the rank of this matrix.
    ///
    /// All singular values below `eps` are considered equal to 0.
    pub fn rank(&self, eps: N::RealField) -> usize {
        let svd = SVD::new(self.clone_owned(), false, false);
        svd.rank(eps)
    }

    /// Computes the pseudo-inverse of this matrix.
    ///
    /// All singular values below `eps` are considered equal to 0.
    pub fn pseudo_inverse(self, eps: N::RealField) -> Result<MatrixMN<N, C, R>, &'static str>
    where
        DefaultAllocator: Allocator<N, C, R>,
    {
        SVD::new(self.clone_owned(), true, true).pseudo_inverse(eps)
    }
}


// Explicit formulae inspired from the paper "Computing the Singular Values of 2-by-2 Complex
// Matrices", Sanzheng Qiao and Xiaohong Wang.
// http://www.cas.mcmaster.ca/sqrl/papers/sqrl5.pdf
fn compute_2x2_uptrig_svd<N: RealField>(
    m11: N,
    m12: N,
    m22: N,
    compute_u: bool,
    compute_v: bool,
) -> (Option<GivensRotation<N>>, Vector2<N>, Option<GivensRotation<N>>)
{
    let two: N::RealField = crate::convert(2.0f64);
    let half: N::RealField = crate::convert(0.5f64);

    let denom = (m11 + m22).hypot(m12) + (m11 - m22).hypot(m12);

    // NOTE: v1 is the singular value that is the closest to m22.
    // This prevents cancellation issues when constructing the vector `csv` below. If we chose
    // otherwise, we would have v1 ~= m11 when m12 is small. This would cause catastrophic
    // cancellation on `v1 * v1 - m11 * m11` below.
    let mut v1 = m11 * m22 * two / denom;
    let mut v2 = half * denom;

    let mut u = None;
    let mut v_t = None;

    if compute_u || compute_v {
        let (csv, sgn_v) = GivensRotation::new(m11 * m12, v1 * v1 - m11 * m11);
        v1 *= sgn_v;
        v2 *= sgn_v;

        if compute_v {
            v_t = Some(csv);
        }

        if compute_u {
            let cu = (m11.scale(csv.c()) + m12 * csv.s()) / v1;
            let su = (m22 * csv.s()) / v1;
            let (csu, sgn_u) = GivensRotation::new(cu, su);

            v1 *= sgn_u;
            v2 *= sgn_u;
            u = Some(csu);
        }
    }

    (u, Vector2::new(v1, v2), v_t)
}