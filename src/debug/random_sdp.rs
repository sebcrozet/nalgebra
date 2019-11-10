#[cfg(feature = "arbitrary")]
use crate::base::storage::Owned;
#[cfg(feature = "arbitrary")]
use quickcheck::{Arbitrary, Gen};

use alga::general::ComplexField;
use crate::base::Scalar;
use crate::base::allocator::Allocator;
use crate::{Dim, Dynamic};
use crate::base::{DefaultAllocator, MatrixN};

use crate::debug::RandomOrthogonal;

/// A random, well-conditioned, symmetric definite-positive matrix.
#[derive(Clone, Debug)]
pub struct RandomSDP<N: Scalar, D: Dim = Dynamic>
where DefaultAllocator: Allocator<N, D, D>
{
    m: MatrixN<N, D>,
}

impl<N: ComplexField, D: Dim> RandomSDP<N, D>
where DefaultAllocator: Allocator<N, D, D>
{
    /// Retrieve the generated matrix.
    pub fn unwrap(self) -> MatrixN<N, D> {
        self.m
    }

    /// Creates a new well conditioned symmetric definite-positive matrix from its dimension and a
    /// random reals generators.
    pub fn new<Rand: FnMut() -> N>(dim: D, mut rand: Rand) -> Self {
        let mut m = RandomOrthogonal::new(dim, || rand()).unwrap();
        let mt = m.adjoint();

        for i in 0..dim.value() {
            let mut col = m.column_mut(i);
            let eigenval = N::one() + N::from_real(rand().modulus());
            col *= eigenval;
        }

        RandomSDP { m: m * mt }
    }
}

#[cfg(feature = "arbitrary")]
impl<N: ComplexField + Arbitrary + Send, D: Dim> Arbitrary for RandomSDP<N, D>
where
    DefaultAllocator: Allocator<N, D, D>,
    Owned<N, D, D>: Clone + Send,
{
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        use rand::Rng;
        let dim = D::try_to_usize().unwrap_or(g.gen_range(1, 50));
        Self::new(D::from_usize(dim), || N::arbitrary(g))
    }
}
