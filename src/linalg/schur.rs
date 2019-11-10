#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use approx::AbsDiffEq;
use alga::general::{ComplexField, RealField};
use num_complex::Complex as NumComplex;
use std::cmp;

use crate::allocator::Allocator;
use crate::{Dim, DimDiff, DimSub, Dynamic, U1, U2, U3};
use crate::base::storage::Storage;
use crate::base::{DefaultAllocator, MatrixN, SquareMatrix, Unit, Vector2, Vector3, VectorN};

use crate::geometry::Reflection;
use crate::linalg::householder;
use crate::linalg::Hessenberg;
use crate::linalg::givens::GivensRotation;

/// Schur decomposition of a square matrix.
///
/// If this is a real matrix, this will be a RealField Schur decomposition.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(
        serialize = "DefaultAllocator: Allocator<N, D, D>,
         MatrixN<N, D>: Serialize"
    ))
)]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(
        deserialize = "DefaultAllocator: Allocator<N, D, D>,
         MatrixN<N, D>: Deserialize<'de>"
    ))
)]
#[derive(Clone, Debug)]
pub struct Schur<N: ComplexField, D: Dim>
where DefaultAllocator: Allocator<N, D, D>
{
    q: MatrixN<N, D>,
    t: MatrixN<N, D>,
}

impl<N: ComplexField, D: Dim> Copy for Schur<N, D>
where
    DefaultAllocator: Allocator<N, D, D>,
    MatrixN<N, D>: Copy,
{}

impl<N: ComplexField, D: Dim> Schur<N, D>
where
    D: DimSub<U1>,                                   // For Hessenberg.
    DefaultAllocator: Allocator<N, D, DimDiff<D, U1>>
        + Allocator<N, DimDiff<D, U1>>
        + Allocator<N, D, D>
        + Allocator<N, D>,
{
    /// Computes the Schur decomposition of a square matrix.
    pub fn new(m: MatrixN<N, D>) -> Self {
        Self::try_new(m, N::RealField::default_epsilon(), 0).unwrap()
    }

    /// Attempts to compute the Schur decomposition of a square matrix.
    ///
    /// If only eigenvalues are needed, it is more efficient to call the matrix method
    /// `.eigenvalues()` instead.
    ///
    /// # Arguments
    ///
    /// * `eps`       − tolerance used to determine when a value converged to 0.
    /// * `max_niter` − maximum total number of iterations performed by the algorithm. If this
    /// number of iteration is exceeded, `None` is returned. If `niter == 0`, then the algorithm
    /// continues indefinitely until convergence.
    pub fn try_new(m: MatrixN<N, D>, eps: N::RealField, max_niter: usize) -> Option<Self> {
        let mut work = unsafe { VectorN::new_uninitialized_generic(m.data.shape().0, U1) };

        Self::do_decompose(m, &mut work, eps, max_niter, true).map(|(q, t)| Schur {
            q: q.unwrap(),
            t: t,
        })
    }

    fn do_decompose(
        mut m: MatrixN<N, D>,
        work: &mut VectorN<N, D>,
        eps: N::RealField,
        max_niter: usize,
        compute_q: bool,
    ) -> Option<(Option<MatrixN<N, D>>, MatrixN<N, D>)>
    {
        assert!(
            m.is_square(),
            "Unable to compute the eigenvectors and eigenvalues of a non-square matrix."
        );

        let dim = m.data.shape().0;

        // Specialization would make this easier.
        if dim.value() == 0 {
            let vecs = Some(MatrixN::from_element_generic(dim, dim, N::zero()));
            let vals = MatrixN::from_element_generic(dim, dim, N::zero());
            return Some((vecs, vals));
        } else if dim.value() == 1 {
            if compute_q {
                let q = MatrixN::from_element_generic(dim, dim, N::one());
                return Some((Some(q), m));
            } else {
                return Some((None, m));
            }
        } else if dim.value() == 2 {
            return decompose_2x2(m, compute_q);
        }

        let amax_m = m.camax();
        m.unscale_mut(amax_m);

        let hess = Hessenberg::new_with_workspace(m, work);
        let mut q;
        let mut t;

        if compute_q {
            // FIXME: could we work without unpacking? Using only the internal representation of
            // hessenberg decomposition.
            let (vecs, vals) = hess.unpack();
            q = Some(vecs);
            t = vals;
        } else {
            q = None;
            t = hess.unpack_h()
        }

        // Implicit double-shift QR method.
        let mut niter = 0;
        let (mut start, mut end) = Self::delimit_subproblem(&mut t, eps, dim.value() - 1);

        while end != start {
            let subdim = end - start + 1;

            if subdim > 2 {
                let m = end - 1;
                let n = end;

                let h11 = t[(start + 0, start + 0)];
                let h12 = t[(start + 0, start + 1)];
                let h21 = t[(start + 1, start + 0)];
                let h22 = t[(start + 1, start + 1)];
                let h32 = t[(start + 2, start + 1)];

                let hnn = t[(n, n)];
                let hmm = t[(m, m)];
                let hnm = t[(n, m)];
                let hmn = t[(m, n)];

                let tra = hnn + hmm;
                let det = hnn * hmm - hnm * hmn;

                let mut axis = Vector3::new(
                    h11 * h11 + h12 * h21 - tra * h11 + det,
                    h21 * (h11 + h22 - tra),
                    h21 * h32,
                );

                for k in start..n - 1 {
                    let (norm, not_zero) = householder::reflection_axis_mut(&mut axis);

                    if not_zero {
                        if k > start {
                            t[(k + 0, k - 1)] = norm;
                            t[(k + 1, k - 1)] = N::zero();
                            t[(k + 2, k - 1)] = N::zero();
                        }

                        let refl = Reflection::new(Unit::new_unchecked(axis), N::zero());

                        {
                            let krows = cmp::min(k + 4, end + 1);
                            let mut work = work.rows_mut(0, krows);
                            refl.reflect(
                                &mut t
                                    .generic_slice_mut((k, k), (U3, Dynamic::new(dim.value() - k))),
                            );
                            refl.reflect_rows(
                                &mut t.generic_slice_mut((0, k), (Dynamic::new(krows), U3)),
                                &mut work,
                            );
                        }

                        if let Some(ref mut q) = q {
                            refl.reflect_rows(&mut q.generic_slice_mut((0, k), (dim, U3)), work);
                        }
                    }

                    axis.x = t[(k + 1, k)];
                    axis.y = t[(k + 2, k)];

                    if k < n - 2 {
                        axis.z = t[(k + 3, k)];
                    }
                }

                let mut axis = Vector2::new(axis.x, axis.y);
                let (norm, not_zero) = householder::reflection_axis_mut(&mut axis);

                if not_zero {
                    let refl = Reflection::new(Unit::new_unchecked(axis), N::zero());

                    t[(m, m - 1)] = norm;
                    t[(n, m - 1)] = N::zero();

                    {
                        let mut work = work.rows_mut(0, end + 1);
                        refl.reflect(
                            &mut t.generic_slice_mut((m, m), (U2, Dynamic::new(dim.value() - m))),
                        );
                        refl.reflect_rows(
                            &mut t.generic_slice_mut((0, m), (Dynamic::new(end + 1), U2)),
                            &mut work,
                        );
                    }

                    if let Some(ref mut q) = q {
                        refl.reflect_rows(&mut q.generic_slice_mut((0, m), (dim, U2)), work);
                    }
                }
            } else {
                // Decouple the 2x2 block if it has real eigenvalues.
                if let Some(rot) = compute_2x2_basis(&t.fixed_slice::<U2, U2>(start, start)) {
                    let inv_rot = rot.inverse();
                    inv_rot.rotate(&mut t.generic_slice_mut(
                        (start, start),
                        (U2, Dynamic::new(dim.value() - start)),
                    ));
                    rot.rotate_rows(
                        &mut t.generic_slice_mut((0, start), (Dynamic::new(end + 1), U2)),
                    );
                    t[(end, start)] = N::zero();

                    if let Some(ref mut q) = q {
                        rot.rotate_rows(&mut q.generic_slice_mut((0, start), (dim, U2)));
                    }
                }

                // Check if we reached the beginning of the matrix.
                if end > 2 {
                    end -= 2;
                } else {
                    break;
                }
            }

            let sub = Self::delimit_subproblem(&mut t, eps, end);

            start = sub.0;
            end = sub.1;

            niter += 1;
            if niter == max_niter {
                return None;
            }
        }

        t.scale_mut(amax_m);

        Some((q, t))
    }

    /// Computes the eigenvalues of the decomposed matrix.
    fn do_eigenvalues(t: &MatrixN<N, D>, out: &mut VectorN<N, D>) -> bool {
        let dim = t.nrows();
        let mut m = 0;

        while m < dim - 1 {
            let n = m + 1;

            if t[(n, m)].is_zero() {
                out[m] = t[(m, m)];
                m += 1;
            } else {
                // Complex eigenvalue.
                return false;
            }
        }

        if m == dim - 1 {
            out[m] = t[(m, m)];
        }

        true
    }

    /// Computes the complex eigenvalues of the decomposed matrix.
    fn do_complex_eigenvalues(t: &MatrixN<N, D>, out: &mut VectorN<NumComplex<N>, D>)
    where N: RealField,
          DefaultAllocator: Allocator<NumComplex<N>, D> {
        let dim = t.nrows();
        let mut m = 0;

        while m < dim - 1 {
            let n = m + 1;

            if t[(n, m)].is_zero() {
                out[m] = NumComplex::new(t[(m, m)], N::zero());
                m += 1;
            } else {
                // Solve the 2x2 eigenvalue subproblem.
                let hmm = t[(m, m)];
                let hnm = t[(n, m)];
                let hmn = t[(m, n)];
                let hnn = t[(n, n)];

                // NOTE: use the same algorithm as in compute_2x2_eigvals.
                let val = (hmm - hnn) * crate::convert(0.5);
                let discr = hnm * hmn + val * val;

                // All 2x2 blocks have negative discriminant because we already decoupled those
                // with positive eigenvalues.
                let sqrt_discr = NumComplex::new(N::zero(), (-discr).sqrt());

                let half_tra = (hnn + hmm) * crate::convert(0.5);
                out[m] = NumComplex::new(half_tra, N::zero()) + sqrt_discr;
                out[m + 1] = NumComplex::new(half_tra, N::zero()) - sqrt_discr;

                m += 2;
            }
        }

        if m == dim - 1 {
            out[m] = NumComplex::new(t[(m, m)], N::zero());
        }
    }

    fn delimit_subproblem(t: &mut MatrixN<N, D>, eps: N::RealField, end: usize) -> (usize, usize)
    where
        D: DimSub<U1>,
        DefaultAllocator: Allocator<N, DimDiff<D, U1>>,
    {
        let mut n = end;

        while n > 0 {
            let m = n - 1;

            if t[(n, m)].norm1() <= eps * (t[(n, n)].norm1() + t[(m, m)].norm1()) {
                t[(n, m)] = N::zero();
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

            let off_diag = t[(new_start, m)];
            if off_diag.is_zero()
                || off_diag.norm1() <= eps * (t[(new_start, new_start)].norm1() + t[(m, m)].norm1())
            {
                t[(new_start, m)] = N::zero();
                break;
            }

            new_start -= 1;
        }

        (new_start, n)
    }

    /// Retrieves the unitary matrix `Q` and the upper-quasitriangular matrix `T` such that the
    /// decomposed matrix equals `Q * T * Q.transpose()`.
    pub fn unpack(self) -> (MatrixN<N, D>, MatrixN<N, D>) {
        (self.q, self.t)
    }

    /// Computes the real eigenvalues of the decomposed matrix.
    ///
    /// Return `None` if some eigenvalues are complex.
    pub fn eigenvalues(&self) -> Option<VectorN<N, D>> {
        let mut out = unsafe { VectorN::new_uninitialized_generic(self.t.data.shape().0, U1) };
        if Self::do_eigenvalues(&self.t, &mut out) {
            Some(out)
        } else {
            None
        }
    }

    /// Computes the complex eigenvalues of the decomposed matrix.
    pub fn complex_eigenvalues(&self) -> VectorN<NumComplex<N>, D>
    where N: RealField,
          DefaultAllocator: Allocator<NumComplex<N>, D> {
        let mut out = unsafe { VectorN::new_uninitialized_generic(self.t.data.shape().0, U1) };
        Self::do_complex_eigenvalues(&self.t, &mut out);
        out
    }
}

fn decompose_2x2<N: ComplexField, D: Dim>(
    mut m: MatrixN<N, D>,
    compute_q: bool,
) -> Option<(Option<MatrixN<N, D>>, MatrixN<N, D>)>
where
    DefaultAllocator: Allocator<N, D, D>,
{
    let dim = m.data.shape().0;
    let mut q = None;
    match compute_2x2_basis(&m.fixed_slice::<U2, U2>(0, 0)) {
        Some(rot) => {
            let mut m = m.fixed_slice_mut::<U2, U2>(0, 0);
            let inv_rot = rot.inverse();
            inv_rot.rotate(&mut m);
            rot.rotate_rows(&mut m);
            m[(1, 0)] = N::zero();

            if compute_q {
                // XXX: we have to build the matrix manually because
                // rot.to_rotation_matrix().unwrap() causes an ICE.
                let c = N::from_real(rot.c());
                q = Some(MatrixN::from_column_slice_generic(
                    dim,
                    dim,
                    &[c, rot.s(), -rot.s().conjugate(), c],
                ));
            }
        }
        None => {
            if compute_q {
                q = Some(MatrixN::identity_generic(dim, dim));
            }
        }
    };

    Some((q, m))
}

fn compute_2x2_eigvals<N: ComplexField, S: Storage<N, U2, U2>>(
    m: &SquareMatrix<N, U2, S>,
) -> Option<(N, N)> {
    // Solve the 2x2 eigenvalue subproblem.
    let h00 = m[(0, 0)];
    let h10 = m[(1, 0)];
    let h01 = m[(0, 1)];
    let h11 = m[(1, 1)];

    // NOTE: this discriminant computation is more stable than the
    // one based on the trace and determinant: 0.25 * tra * tra - det
    // because it ensures positiveness for symmetric matrices.
    let val = (h00 - h11) * crate::convert(0.5);
    let discr = h10 * h01 + val * val;

    discr.try_sqrt().map(|sqrt_discr| {
        let half_tra = (h00 + h11) * crate::convert(0.5);
        (half_tra + sqrt_discr, half_tra - sqrt_discr)
    })
}

// Computes the 2x2 transformation that upper-triangulates a 2x2 matrix with real eigenvalues.
/// Computes the singular vectors for a 2x2 matrix.
///
/// Returns `None` if the matrix has complex eigenvalues, or is upper-triangular. In both case,
/// the basis is the identity.
fn compute_2x2_basis<N: ComplexField, S: Storage<N, U2, U2>>(
    m: &SquareMatrix<N, U2, S>,
) -> Option<GivensRotation<N>> {
    let h10 = m[(1, 0)];

    if h10.is_zero() {
        return None;
    }

    if let Some((eigval1, eigval2)) = compute_2x2_eigvals(m) {
        let x1 = eigval1 - m[(1, 1)];
        let x2 = eigval2 - m[(1, 1)];

        // NOTE: Choose the one that yields a larger x component.
        // This is necessary for numerical stability of the normalization of the complex
        // number.
        if x1.norm1() > x2.norm1() {
            Some(GivensRotation::new(x1, h10).0)
        } else {
            Some(GivensRotation::new(x2, h10).0)
        }
    } else {
        None
    }
}

impl<N: ComplexField, D: Dim, S: Storage<N, D, D>> SquareMatrix<N, D, S>
where
    D: DimSub<U1>,                                   // For Hessenberg.
    DefaultAllocator: Allocator<N, D, DimDiff<D, U1>>
        + Allocator<N, DimDiff<D, U1>>
        + Allocator<N, D, D>
        + Allocator<N, D>,
{
    /// Computes the Schur decomposition of a square matrix.
    pub fn schur(self) -> Schur<N, D> {
        Schur::new(self.into_owned())
    }

    /// Attempts to compute the Schur decomposition of a square matrix.
    ///
    /// If only eigenvalues are needed, it is more efficient to call the matrix method
    /// `.eigenvalues()` instead.
    ///
    /// # Arguments
    ///
    /// * `eps`       − tolerance used to determine when a value converged to 0.
    /// * `max_niter` − maximum total number of iterations performed by the algorithm. If this
    /// number of iteration is exceeded, `None` is returned. If `niter == 0`, then the algorithm
    /// continues indefinitely until convergence.
    pub fn try_schur(self, eps: N::RealField, max_niter: usize) -> Option<Schur<N, D>> {
        Schur::try_new(self.into_owned(), eps, max_niter)
    }

    /// Computes the eigenvalues of this matrix.
    pub fn eigenvalues(&self) -> Option<VectorN<N, D>> {
        assert!(
            self.is_square(),
            "Unable to compute eigenvalues of a non-square matrix."
        );

        let mut work = unsafe { VectorN::new_uninitialized_generic(self.data.shape().0, U1) };

        // Special case for 2x2 matrices.
        if self.nrows() == 2 {
            // FIXME: can we avoid this slicing
            // (which is needed here just to transform D to U2)?
            let me = self.fixed_slice::<U2, U2>(0, 0);
            return match compute_2x2_eigvals(&me) {
                Some((a, b)) => {
                    work[0] = a;
                    work[1] = b;
                    Some(work)
                }
                None => None,
            };
        }

        // FIXME: add balancing?
        let schur = Schur::do_decompose(
            self.clone_owned(),
            &mut work,
            N::RealField::default_epsilon(),
            0,
            false,
        )
        .unwrap();
        if Schur::do_eigenvalues(&schur.1, &mut work) {
            Some(work)
        } else {
            None
        }
    }

    /// Computes the eigenvalues of this matrix.
    pub fn complex_eigenvalues(&self) -> VectorN<NumComplex<N>, D>
    // FIXME: add balancing?
    where N: RealField,
          DefaultAllocator: Allocator<NumComplex<N>, D> {
        let dim = self.data.shape().0;
        let mut work = unsafe { VectorN::new_uninitialized_generic(dim, U1) };

        let schur = Schur::do_decompose(
            self.clone_owned(),
            &mut work,
            N::default_epsilon(),
            0,
            false,
        )
        .unwrap();
        let mut eig = unsafe { VectorN::new_uninitialized_generic(dim, U1) };
        Schur::do_complex_eigenvalues(&schur.1, &mut eig);
        eig
    }
}
