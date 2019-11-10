//! Construction of givens rotations.

use alga::general::ComplexField;
use num::{Zero, One};

use crate::{Dim, U2};
use crate::base::constraint::{ShapeConstraint, DimEq};
use crate::base::storage::{Storage, StorageMut};
use crate::base::{Vector, Matrix};


/// A Givens rotation.
#[derive(Debug, Clone, Copy)]
pub struct GivensRotation<N: ComplexField> {
    c: N::RealField,
    s: N
}

// Matrix = UnitComplex * Matrix
impl<N: ComplexField> GivensRotation<N> {
    /// The Givents rotation that does nothing.
    pub fn identity() -> Self {
        Self {
            c: N::RealField::one(),
            s: N::zero()
        }
    }

    /// Initializes a Givens rotation from its components.
    ///
    /// The components are copies as-is. It is not checked whether they describe
    /// an actually valid Givens rotation.
    pub fn new_unchecked(c: N::RealField, s: N) -> Self {
       Self {
           c, s
       }
    }

    /// Initializes a Givens rotation from its non-normalized cosine an sine components.
    pub fn new(c: N, s: N) -> (Self, N) {
        Self::try_new(c, s, N::RealField::zero()).unwrap()
    }

    /// Initializes a Givens rotation form its non-normalized cosine an sine components.
    pub fn try_new(c: N, s: N, eps: N::RealField) -> Option<(Self, N)> {
        let (mod0, sign0) = c.to_exp();
        let denom = (mod0 * mod0 + s.modulus_squared()).sqrt();

        if denom > eps {
            let norm = sign0.scale(denom);
            let c = mod0 / denom;
            let s = s / norm;
            Some((Self { c, s }, norm))
        } else {
            None
        }
    }

    /// Computes the rotation `R` required such that the `y` component of `R * v` is zero.
    ///
    /// Returns `None` if no rotation is needed (i.e. if `v.y == 0`). Otherwise, this returns the norm
    /// of `v` and the rotation `r` such that `R * v = [ |v|, 0.0 ]^t` where `|v|` is the norm of `v`.
    pub fn cancel_y<S: Storage<N, U2>>(v: &Vector<N, U2, S>) -> Option<(Self, N)> {
        if !v[1].is_zero() {
            let (mod0, sign0) = v[0].to_exp();
            let denom = (mod0 * mod0 + v[1].modulus_squared()).sqrt();
            let c = mod0 / denom;
            let s = -v[1] / sign0.scale(denom);
            let r = sign0.scale(denom);
            Some((Self { c, s }, r))
        } else {
            None
        }
    }

    /// Computes the rotation `R` required such that the `x` component of `R * v` is zero.
    ///
    /// Returns `None` if no rotation is needed (i.e. if `v.x == 0`). Otherwise, this returns the norm
    /// of `v` and the rotation `r` such that `R * v = [ 0.0, |v| ]^t` where `|v|` is the norm of `v`.
    pub fn cancel_x<S: Storage<N, U2>>(v: &Vector<N, U2, S>) -> Option<(Self, N)> {
        if !v[0].is_zero() {
            let (mod1, sign1) = v[1].to_exp();
            let denom = (mod1 * mod1 + v[0].modulus_squared()).sqrt();
            let c = mod1 / denom;
            let s = (v[0].conjugate() * sign1).unscale(denom);
            let r = sign1.scale(denom);
            Some((Self { c, s }, r))
        } else {
            None
        }
    }

    /// The cos part of this roration.
    pub fn c(&self) -> N::RealField {
        self.c
    }

    /// The sin part of this roration.
    pub fn s(&self) -> N {
        self.s
    }

    /// The inverse of this givens rotation.
    pub fn inverse(&self) -> Self {
        Self { c: self.c, s: -self.s }
    }

    /// Performs the multiplication `rhs = self * rhs` in-place.
    pub fn rotate<R2: Dim, C2: Dim, S2: StorageMut<N, R2, C2>>(
        &self,
        rhs: &mut Matrix<N, R2, C2, S2>,
    ) where
        ShapeConstraint: DimEq<R2, U2>,
    {
        assert_eq!(
            rhs.nrows(),
            2,
            "Unit complex rotation: the input matrix must have exactly two rows."
        );
        let s = self.s;
        let c = self.c;

        for j in 0..rhs.ncols() {
            unsafe {
                let a = *rhs.get_unchecked((0, j));
                let b = *rhs.get_unchecked((1, j));

                *rhs.get_unchecked_mut((0, j)) = a.scale(c) - s.conjugate() * b;
                *rhs.get_unchecked_mut((1, j)) = s * a + b.scale(c);
            }
        }
    }

    /// Performs the multiplication `lhs = lhs * self` in-place.
    pub fn rotate_rows<R2: Dim, C2: Dim, S2: StorageMut<N, R2, C2>>(
        &self,
        lhs: &mut Matrix<N, R2, C2, S2>,
    ) where
        ShapeConstraint: DimEq<C2, U2>,
    {
        assert_eq!(
            lhs.ncols(),
            2,
            "Unit complex rotation: the input matrix must have exactly two columns."
        );
        let s = self.s;
        let c = self.c;

        // FIXME: can we optimize that to iterate on one column at a time ?
        for j in 0..lhs.nrows() {
            unsafe {
                let a = *lhs.get_unchecked((j, 0));
                let b = *lhs.get_unchecked((j, 1));

                *lhs.get_unchecked_mut((j, 0)) = a.scale(c) + s * b;
                *lhs.get_unchecked_mut((j, 1)) = -s.conjugate() * a + b.scale(c);
            }
        }
    }
}

