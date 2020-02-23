#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::vec::Vec;

use num::{One, Zero};

use alga::general::{
    AbstractGroup, AbstractGroupAbelian, AbstractLoop, AbstractMagma, AbstractModule,
    AbstractMonoid, AbstractQuasigroup, AbstractSemigroup, Additive, ClosedAdd, ClosedMul,
    ClosedNeg, Field, Identity, TwoSidedInverse, JoinSemilattice, Lattice, MeetSemilattice, Module,
    Multiplicative, RingCommutative, ComplexField
};
use alga::linear::{
    FiniteDimInnerSpace, FiniteDimVectorSpace, InnerSpace, NormedSpace, VectorSpace,
};

use crate::base::allocator::Allocator;
use crate::base::dimension::{Dim, DimName};
use crate::base::storage::{Storage, StorageMut};
use crate::base::{DefaultAllocator, MatrixMN, MatrixN, Scalar};

/*
 *
 * Additive structures.
 *
 */
impl<N, R: DimName, C: DimName> Identity<Additive> for MatrixMN<N, R, C>
where
    N: Scalar + Zero,
    DefaultAllocator: Allocator<N, R, C>,
{
    #[inline]
    fn identity() -> Self {
        Self::from_element(N::zero())
    }
}

impl<N, R: DimName, C: DimName> AbstractMagma<Additive> for MatrixMN<N, R, C>
where
    N: Scalar + ClosedAdd,
    DefaultAllocator: Allocator<N, R, C>,
{
    #[inline]
    fn operate(&self, other: &Self) -> Self {
        self + other
    }
}

impl<N, R: DimName, C: DimName> TwoSidedInverse<Additive> for MatrixMN<N, R, C>
where
    N: Scalar + ClosedNeg,
    DefaultAllocator: Allocator<N, R, C>,
{
    #[inline]
    #[must_use = "Did you mean to use two_sided_inverse_mut()?"]
    fn two_sided_inverse(&self) -> Self {
        -self
    }

    #[inline]
    fn two_sided_inverse_mut(&mut self) {
        *self = -self.clone()
    }
}

macro_rules! inherit_additive_structure(
    ($($marker: ident<$operator: ident> $(+ $bounds: ident)*),* $(,)*) => {$(
        impl<N, R: DimName, C: DimName> $marker<$operator> for MatrixMN<N, R, C>
            where N: Scalar + $marker<$operator> $(+ $bounds)*,
                  DefaultAllocator: Allocator<N, R, C> { }
    )*}
);

inherit_additive_structure!(
    AbstractSemigroup<Additive> + ClosedAdd,
    AbstractMonoid<Additive> + Zero + ClosedAdd,
    AbstractQuasigroup<Additive> + ClosedAdd + ClosedNeg,
    AbstractLoop<Additive> + Zero + ClosedAdd + ClosedNeg,
    AbstractGroup<Additive> + Zero + ClosedAdd + ClosedNeg,
    AbstractGroupAbelian<Additive> + Zero + ClosedAdd + ClosedNeg
);

impl<N, R: DimName, C: DimName> AbstractModule for MatrixMN<N, R, C>
where
    N: Scalar + RingCommutative,
    DefaultAllocator: Allocator<N, R, C>,
{
    type AbstractRing = N;

    #[inline]
    fn multiply_by(&self, n: N) -> Self {
        self * n
    }
}

impl<N, R: DimName, C: DimName> Module for MatrixMN<N, R, C>
where
    N: Scalar + RingCommutative,
    DefaultAllocator: Allocator<N, R, C>,
{
    type Ring = N;
}

impl<N, R: DimName, C: DimName> VectorSpace for MatrixMN<N, R, C>
where
    N: Scalar + Field,
    DefaultAllocator: Allocator<N, R, C>,
{
    type Field = N;
}

impl<N, R: DimName, C: DimName> FiniteDimVectorSpace for MatrixMN<N, R, C>
where
    N: Scalar + Field,
    DefaultAllocator: Allocator<N, R, C>,
{
    #[inline]
    fn dimension() -> usize {
        R::dim() * C::dim()
    }

    #[inline]
    fn canonical_basis_element(i: usize) -> Self {
        assert!(i < Self::dimension(), "Index out of bound.");

        let mut res = Self::zero();
        unsafe {
            *res.data.get_unchecked_linear_mut(i) = N::one();
        }

        res
    }

    #[inline]
    fn dot(&self, other: &Self) -> N {
        self.dot(other)
    }

    #[inline]
    unsafe fn component_unchecked(&self, i: usize) -> &N {
        self.data.get_unchecked_linear(i)
    }

    #[inline]
    unsafe fn component_unchecked_mut(&mut self, i: usize) -> &mut N {
        self.data.get_unchecked_linear_mut(i)
    }
}

impl<N: ComplexField, R: DimName, C: DimName> NormedSpace for MatrixMN<N, R, C>
where DefaultAllocator: Allocator<N, R, C>
{
    type RealField = N::RealField;
    type ComplexField = N;

    #[inline]
    fn norm_squared(&self) -> N::RealField {
        self.norm_squared()
    }

    #[inline]
    fn norm(&self) -> N::RealField {
        self.norm()
    }

    #[inline]
    #[must_use = "Did you mean to use normalize_mut()?"]
    fn normalize(&self) -> Self {
        self.normalize()
    }

    #[inline]
    fn normalize_mut(&mut self) -> N::RealField {
        self.normalize_mut()
    }

    #[inline]
    #[must_use = "Did you mean to use try_normalize_mut()?"]
    fn try_normalize(&self, min_norm: N::RealField) -> Option<Self> {
        self.try_normalize(min_norm)
    }

    #[inline]
    fn try_normalize_mut(&mut self, min_norm: N::RealField) -> Option<N::RealField> {
        self.try_normalize_mut(min_norm)
    }
}

impl<N: ComplexField, R: DimName, C: DimName> InnerSpace for MatrixMN<N, R, C>
where DefaultAllocator: Allocator<N, R, C>
{
    #[inline]
    fn angle(&self, other: &Self) -> N::RealField {
        self.angle(other)
    }

    #[inline]
    fn inner_product(&self, other: &Self) -> N {
        self.dotc(other)
    }
}

// FIXME: specialization will greatly simplify this implementation in the future.
// In particular:
//   − use `x()` instead of `::canonical_basis_element`
//   − use `::new(x, y, z)` instead of `::from_slice`
impl<N: ComplexField, R: DimName, C: DimName> FiniteDimInnerSpace for MatrixMN<N, R, C>
where DefaultAllocator: Allocator<N, R, C>
{
    #[inline]
    fn orthonormalize(vs: &mut [Self]) -> usize {
        let mut nbasis_elements = 0;

        for i in 0..vs.len() {
            {
                let (elt, basis) = vs[..i + 1].split_last_mut().unwrap();

                for basis_element in &basis[..nbasis_elements] {
                    *elt -= &*basis_element * elt.dot(basis_element)
                }
            }

            if vs[i].try_normalize_mut(N::RealField::zero()).is_some() {
                // FIXME: this will be efficient on dynamically-allocated vectors but for
                // statically-allocated ones, `.clone_from` would be better.
                vs.swap(nbasis_elements, i);
                nbasis_elements += 1;

                // All the other vectors will be dependent.
                if nbasis_elements == Self::dimension() {
                    break;
                }
            }
        }

        nbasis_elements
    }

    #[inline]
    fn orthonormal_subspace_basis<F>(vs: &[Self], mut f: F)
    where F: FnMut(&Self) -> bool {
        // FIXME: is this necessary?
        assert!(
            vs.len() <= Self::dimension(),
            "The given set of vectors has no chance of being a free family."
        );

        match Self::dimension() {
            1 => {
                if vs.len() == 0 {
                    let _ = f(&Self::canonical_basis_element(0));
                }
            }
            2 => {
                if vs.len() == 0 {
                    let _ = f(&Self::canonical_basis_element(0))
                        && f(&Self::canonical_basis_element(1));
                } else if vs.len() == 1 {
                    let v = &vs[0];
                    let res = Self::from_column_slice(&[-v[1], v[0]]);

                    let _ = f(&res.normalize());
                }

                // Otherwise, nothing.
            }
            3 => {
                if vs.len() == 0 {
                    let _ = f(&Self::canonical_basis_element(0))
                        && f(&Self::canonical_basis_element(1))
                        && f(&Self::canonical_basis_element(2));
                } else if vs.len() == 1 {
                    let v = &vs[0];
                    let mut a;

                    if v[0].norm1() > v[1].norm1() {
                        a = Self::from_column_slice(&[v[2], N::zero(), -v[0]]);
                    } else {
                        a = Self::from_column_slice(&[N::zero(), -v[2], v[1]]);
                    };

                    let _ = a.normalize_mut();

                    if f(&a.cross(v)) {
                        let _ = f(&a);
                    }
                } else if vs.len() == 2 {
                    let _ = f(&vs[0].cross(&vs[1]).normalize());
                }
            }
            _ => {
                #[cfg(any(feature = "std", feature = "alloc"))]
                {
                    // XXX: use a GenericArray instead.
                    let mut known_basis = Vec::new();

                    for v in vs.iter() {
                        known_basis.push(v.normalize())
                    }

                    for i in 0..Self::dimension() - vs.len() {
                        let mut elt = Self::canonical_basis_element(i);

                        for v in &known_basis {
                            elt -= v * elt.dot(v)
                        }

                        if let Some(subsp_elt) = elt.try_normalize(N::RealField::zero()) {
                            if !f(&subsp_elt) {
                                return;
                            };

                            known_basis.push(subsp_elt);
                        }
                    }
                }
                #[cfg(all(not(feature = "std"), not(feature = "alloc")))]
                {
                    panic!("Cannot compute the orthogonal subspace basis of a vector with a dimension greater than 3 \
                            if #![no_std] is enabled and the 'alloc' feature is not enabled.")
                }
            }
        }
    }
}

/*
 *
 *
 * Multiplicative structures.
 *
 *
 */
impl<N, D: DimName> Identity<Multiplicative> for MatrixN<N, D>
where
    N: Scalar + Zero + One,
    DefaultAllocator: Allocator<N, D, D>,
{
    #[inline]
    fn identity() -> Self {
        Self::identity()
    }
}

impl<N, D: DimName> AbstractMagma<Multiplicative> for MatrixN<N, D>
where
    N: Scalar + Zero + One + ClosedAdd + ClosedMul,
    DefaultAllocator: Allocator<N, D, D>,
{
    #[inline]
    fn operate(&self, other: &Self) -> Self {
        self * other
    }
}

macro_rules! impl_multiplicative_structure(
    ($($marker: ident<$operator: ident> $(+ $bounds: ident)*),* $(,)*) => {$(
        impl<N, D: DimName> $marker<$operator> for MatrixN<N, D>
            where N: Scalar + Zero + One + ClosedAdd + ClosedMul + $marker<$operator> $(+ $bounds)*,
                  DefaultAllocator: Allocator<N, D, D> { }
    )*}
);

impl_multiplicative_structure!(
    AbstractSemigroup<Multiplicative>,
    AbstractMonoid<Multiplicative> + One
);

/*
 *
 * Ordering
 *
 */
impl<N, R: Dim, C: Dim> MeetSemilattice for MatrixMN<N, R, C>
where
    N: Scalar + MeetSemilattice,
    DefaultAllocator: Allocator<N, R, C>,
{
    #[inline]
    fn meet(&self, other: &Self) -> Self {
        self.zip_map(other, |a, b| a.meet(&b))
    }
}

impl<N, R: Dim, C: Dim> JoinSemilattice for MatrixMN<N, R, C>
where
    N: Scalar + JoinSemilattice,
    DefaultAllocator: Allocator<N, R, C>,
{
    #[inline]
    fn join(&self, other: &Self) -> Self {
        self.zip_map(other, |a, b| a.join(&b))
    }
}

impl<N, R: Dim, C: Dim> Lattice for MatrixMN<N, R, C>
where
    N: Scalar + Lattice,
    DefaultAllocator: Allocator<N, R, C>,
{
    #[inline]
    fn meet_join(&self, other: &Self) -> (Self, Self) {
        let shape = self.data.shape();
        assert!(
            shape == other.data.shape(),
            "Matrix meet/join error: mismatched dimensions."
        );

        let mut mres = unsafe { Self::new_uninitialized_generic(shape.0, shape.1) };
        let mut jres = unsafe { Self::new_uninitialized_generic(shape.0, shape.1) };

        for i in 0..shape.0.value() * shape.1.value() {
            unsafe {
                let mj = self
                    .data
                    .get_unchecked_linear(i)
                    .meet_join(other.data.get_unchecked_linear(i));
                *mres.data.get_unchecked_linear_mut(i) = mj.0;
                *jres.data.get_unchecked_linear_mut(i) = mj.1;
            }
        }

        (mres, jres)
    }
}
