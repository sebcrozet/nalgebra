#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use num::Zero;
use approx::AbsDiffEq;

use alga::general::ComplexField;
use crate::allocator::Allocator;
use crate::base::{DefaultAllocator, Matrix2, MatrixN, SquareMatrix, Vector2, VectorN};
use crate::{Dim, DimDiff, DimSub, U1, U2};
use crate::storage::Storage;

use crate::linalg::givens::GivensRotation;
use crate::linalg::SymmetricTridiagonal;

/// Eigendecomposition of a symmetric matrix.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(
        serialize = "DefaultAllocator: Allocator<N, D, D> +
                           Allocator<N::RealField, D>,
         VectorN<N::RealField, D>: Serialize,
         MatrixN<N, D>: Serialize"
    ))
)]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(
        deserialize = "DefaultAllocator: Allocator<N, D, D> +
                           Allocator<N::RealField, D>,
         VectorN<N::RealField, D>: Deserialize<'de>,
         MatrixN<N, D>: Deserialize<'de>"
    ))
)]
#[derive(Clone, Debug)]
pub struct SymmetricEigen<N: ComplexField, D: Dim>
where DefaultAllocator: Allocator<N, D, D> + Allocator<N::RealField, D>
{
    /// The eigenvectors of the decomposed matrix.
    pub eigenvectors: MatrixN<N, D>,

    /// The unsorted eigenvalues of the decomposed matrix.
    pub eigenvalues: VectorN<N::RealField, D>,
}

impl<N: ComplexField, D: Dim> Copy for SymmetricEigen<N, D>
where
    DefaultAllocator: Allocator<N, D, D> + Allocator<N::RealField, D>,
    MatrixN<N, D>: Copy,
    VectorN<N::RealField, D>: Copy,
{}

impl<N: ComplexField, D: Dim> SymmetricEigen<N, D>
where DefaultAllocator: Allocator<N, D, D> + Allocator<N::RealField, D>
{
    /// Computes the eigendecomposition of the given symmetric matrix.
    ///
    /// Only the lower-triangular parts (including its diagonal) of `m` is read.
    pub fn new(m: MatrixN<N, D>) -> Self
    where
        D: DimSub<U1>,
        DefaultAllocator: Allocator<N, DimDiff<D, U1>> + // For tridiagonalization
        Allocator<N::RealField, DimDiff<D, U1>>,
    {
        Self::try_new(m, N::RealField::default_epsilon(), 0).unwrap()
    }

    /// Computes the eigendecomposition of the given symmetric matrix with user-specified
    /// convergence parameters.
    ///
    /// Only the lower-triangular part (including its diagonal) of `m` is read.
    ///
    /// # Arguments
    ///
    /// * `eps`       − tolerance used to determine when a value converged to 0.
    /// * `max_niter` − maximum total number of iterations performed by the algorithm. If this
    /// number of iteration is exceeded, `None` is returned. If `niter == 0`, then the algorithm
    /// continues indefinitely until convergence.
    pub fn try_new(m: MatrixN<N, D>, eps: N::RealField, max_niter: usize) -> Option<Self>
    where
        D: DimSub<U1>,
        DefaultAllocator: Allocator<N, DimDiff<D, U1>> + // For tridiagonalization
                          Allocator<N::RealField, DimDiff<D, U1>>,
    {
        Self::do_decompose(m, true, eps, max_niter).map(|(vals, vecs)| SymmetricEigen {
            eigenvectors: vecs.unwrap(),
            eigenvalues: vals,
        })
    }

    fn do_decompose(
        mut m: MatrixN<N, D>,
        eigenvectors: bool,
        eps: N::RealField,
        max_niter: usize,
    ) -> Option<(VectorN<N::RealField, D>, Option<MatrixN<N, D>>)>
    where
        D: DimSub<U1>,
        DefaultAllocator: Allocator<N, DimDiff<D, U1>> + // For tridiagonalization
                          Allocator<N::RealField, DimDiff<D, U1>>,
    {
        assert!(
            m.is_square(),
            "Unable to compute the eigendecomposition of a non-square matrix."
        );
        let dim = m.nrows();
        let m_amax = m.camax();

        if !m_amax.is_zero() {
            m.unscale_mut(m_amax);
        }

        let (mut q, mut diag, mut off_diag);

        if eigenvectors {
            let res = SymmetricTridiagonal::new(m).unpack();
            q = Some(res.0);
            diag = res.1;
            off_diag = res.2;
        } else {
            let res = SymmetricTridiagonal::new(m).unpack_tridiagonal();
            q = None;
            diag = res.0;
            off_diag = res.1;
        }

        if dim == 1 {
            diag.scale_mut(m_amax);
            return Some((diag, q));
        }

        let mut niter = 0;
        let (mut start, mut end) = Self::delimit_subproblem(&diag, &mut off_diag, dim - 1, eps);

        while end != start {
            let subdim = end - start + 1;

            if subdim > 2 {
                let m = end - 1;
                let n = end;

                let mut v = Vector2::new(
                    diag[start] - wilkinson_shift(diag[m], diag[n], off_diag[m]),
                    off_diag[start],
                );

                for i in start..n {
                    let j = i + 1;

                    if let Some((rot, norm)) = GivensRotation::cancel_y(&v) {
                        if i > start {
                            // Not the first iteration.
                            off_diag[i - 1] = norm;
                        }


                        let mii = diag[i];
                        let mjj = diag[j];
                        let mij = off_diag[i];

                        let cc = rot.c() * rot.c();
                        let ss = rot.s() * rot.s();
                        let cs = rot.c() * rot.s();

                        let b = cs * crate::convert(2.0) * mij;

                        diag[i] = (cc * mii + ss * mjj) - b;
                        diag[j] = (ss * mii + cc * mjj) + b;
                        off_diag[i] = cs * (mii - mjj) + mij * (cc - ss);

                        if i != n - 1 {
                            v.x = off_diag[i];
                            v.y = -rot.s() * off_diag[i + 1];
                            off_diag[i + 1] *= rot.c();
                        }

                        if let Some(ref mut q) = q {
                            let rot = GivensRotation::new_unchecked(rot.c(), N::from_real(rot.s()));
                            rot.inverse().rotate_rows(&mut q.fixed_columns_mut::<U2>(i));
                        }
                    } else {
                        break;
                    }
                }

                if off_diag[m].norm1() <= eps * (diag[m].norm1() + diag[n].norm1()) {
                    end -= 1;
                }
            } else if subdim == 2 {
                let m = Matrix2::new(
                    diag[start], off_diag[start].conjugate(),
                    off_diag[start], diag[start + 1],
                );
                let eigvals = m.eigenvalues().unwrap();
                let basis = Vector2::new(eigvals.x - diag[start + 1], off_diag[start]);

                diag[start + 0] = eigvals[0];
                diag[start + 1] = eigvals[1];

                if let Some(ref mut q) = q {
                    if let Some((rot, _)) = GivensRotation::try_new(basis.x, basis.y, eps) {
                        let rot = GivensRotation::new_unchecked(rot.c(), N::from_real(rot.s()));
                        rot.rotate_rows(&mut q.fixed_columns_mut::<U2>(start));
                    }
                }

                end -= 1;
            }

            // Re-delimit the subproblem in case some decoupling occurred.
            let sub = Self::delimit_subproblem(&diag, &mut off_diag, end, eps);

            start = sub.0;
            end = sub.1;

            niter += 1;
            if niter == max_niter {
                return None;
            }
        }

        diag.scale_mut(m_amax);

        Some((diag, q))
    }

    fn delimit_subproblem(
        diag: &VectorN<N::RealField, D>,
        off_diag: &mut VectorN<N::RealField, DimDiff<D, U1>>,
        end: usize,
        eps: N::RealField,
    ) -> (usize, usize)
    where
        D: DimSub<U1>,
        DefaultAllocator: Allocator<N::RealField, DimDiff<D, U1>>,
    {
        let mut n = end;

        while n > 0 {
            let m = n - 1;

            if off_diag[m].norm1() > eps * (diag[n].norm1() + diag[m].norm1()) {
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

            if off_diag[m].is_zero()
                || off_diag[m].norm1() <= eps * (diag[new_start].norm1() + diag[m].norm1())
            {
                off_diag[m] = N::RealField::zero();
                break;
            }

            new_start -= 1;
        }

        (new_start, n)
    }

    /// Rebuild the original matrix.
    ///
    /// This is useful if some of the eigenvalues have been manually modified.
    pub fn recompose(&self) -> MatrixN<N, D> {
        let mut u_t = self.eigenvectors.clone();
        for i in 0..self.eigenvalues.len() {
            let val = self.eigenvalues[i];
            u_t.column_mut(i).scale_mut(val);
        }
        u_t.adjoint_mut();
        &self.eigenvectors * u_t
    }
}

/// Computes the wilkinson shift, i.e., the 2x2 symmetric matrix eigenvalue to its tailing
/// component `tnn`.
///
/// The inputs are interpreted as the 2x2 matrix:
///     tmm  tmn
///     tmn  tnn
pub fn wilkinson_shift<N: ComplexField>(tmm: N, tnn: N, tmn: N) -> N {
    let sq_tmn = tmn * tmn;
    if !sq_tmn.is_zero() {
        // We have the guarantee that the denominator won't be zero.
        let d = (tmm - tnn) * crate::convert(0.5);
        tnn - sq_tmn / (d + d.signum() * (d * d + sq_tmn).sqrt())
    } else {
        tnn
    }
}

/*
 *
 * Computations of eigenvalues for symmetric matrices.
 *
 */
impl<N: ComplexField, D: DimSub<U1>, S: Storage<N, D, D>> SquareMatrix<N, D, S>
where DefaultAllocator: Allocator<N, D, D> + Allocator<N, DimDiff<D, U1>> +
                        Allocator<N::RealField, D> + Allocator<N::RealField, DimDiff<D, U1>>
{
    /// Computes the eigendecomposition of this symmetric matrix.
    ///
    /// Only the lower-triangular part (including the diagonal) of `m` is read.
    pub fn symmetric_eigen(self) -> SymmetricEigen<N, D> {
        SymmetricEigen::new(self.into_owned())
    }

    /// Computes the eigendecomposition of the given symmetric matrix with user-specified
    /// convergence parameters.
    ///
    /// Only the lower-triangular part (including the diagonal) of `m` is read.
    ///
    /// # Arguments
    ///
    /// * `eps`       − tolerance used to determine when a value converged to 0.
    /// * `max_niter` − maximum total number of iterations performed by the algorithm. If this
    /// number of iteration is exceeded, `None` is returned. If `niter == 0`, then the algorithm
    /// continues indefinitely until convergence.
    pub fn try_symmetric_eigen(self, eps: N::RealField, max_niter: usize) -> Option<SymmetricEigen<N, D>> {
        SymmetricEigen::try_new(self.into_owned(), eps, max_niter)
    }

    /// Computes the eigenvalues of this symmetric matrix.
    ///
    /// Only the lower-triangular part of the matrix is read.
    pub fn symmetric_eigenvalues(&self) -> VectorN<N::RealField, D> {
        SymmetricEigen::do_decompose(self.clone_owned(), false, N::RealField::default_epsilon(), 0)
            .unwrap()
            .0
    }
}

#[cfg(test)]
mod test {
    use crate::base::Matrix2;

    fn expected_shift(m: Matrix2<f64>) -> f64 {
        let vals = m.eigenvalues().unwrap();

        if (vals.x - m.m22).abs() < (vals.y - m.m22).abs() {
            vals.x
        } else {
            vals.y
        }
    }

    #[test]
    fn wilkinson_shift_random() {
        for _ in 0..1000 {
            let m = Matrix2::new_random();
            let m = m * m.transpose();

            let expected = expected_shift(m);
            let computed = super::wilkinson_shift(m.m11, m.m22, m.m12);
            assert!(relative_eq!(expected, computed, epsilon = 1.0e-7));
        }
    }

    #[test]
    fn wilkinson_shift_zero() {
        let m = Matrix2::new(0.0, 0.0, 0.0, 0.0);
        assert!(relative_eq!(
            expected_shift(m),
            super::wilkinson_shift(m.m11, m.m22, m.m12)
        ));
    }

    #[test]
    fn wilkinson_shift_zero_diagonal() {
        let m = Matrix2::new(0.0, 42.0, 42.0, 0.0);
        assert!(relative_eq!(
            expected_shift(m),
            super::wilkinson_shift(m.m11, m.m22, m.m12)
        ));
    }

    #[test]
    fn wilkinson_shift_zero_off_diagonal() {
        let m = Matrix2::new(42.0, 0.0, 0.0, 64.0);
        assert!(relative_eq!(
            expected_shift(m),
            super::wilkinson_shift(m.m11, m.m22, m.m12)
        ));
    }

    #[test]
    fn wilkinson_shift_zero_trace() {
        let m = Matrix2::new(42.0, 20.0, 20.0, -42.0);
        assert!(relative_eq!(
            expected_shift(m),
            super::wilkinson_shift(m.m11, m.m22, m.m12)
        ));
    }

    #[test]
    fn wilkinson_shift_zero_diag_diff_and_zero_off_diagonal() {
        let m = Matrix2::new(42.0, 0.0, 0.0, 42.0);
        assert!(relative_eq!(
            expected_shift(m),
            super::wilkinson_shift(m.m11, m.m22, m.m12)
        ));
    }

    #[test]
    fn wilkinson_shift_zero_det() {
        let m = Matrix2::new(2.0, 4.0, 4.0, 8.0);
        assert!(relative_eq!(
            expected_shift(m),
            super::wilkinson_shift(m.m11, m.m22, m.m12)
        ));
    }
}
