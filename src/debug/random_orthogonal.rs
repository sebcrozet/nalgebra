#[cfg(feature = "arbitrary")]
use crate::base::storage::Owned;
#[cfg(feature = "arbitrary")]
use quickcheck::{Arbitrary, Gen};

use alga::general::ComplexField;
use crate::base::Scalar;
use crate::base::allocator::Allocator;
use crate::{Dim, Dynamic, U2};
use crate::base::{DefaultAllocator, MatrixN};
use crate::linalg::givens::GivensRotation;

/// A random orthogonal matrix.
#[derive(Clone, Debug)]
pub struct RandomOrthogonal<N: Scalar, D: Dim = Dynamic>
where DefaultAllocator: Allocator<N, D, D>
{
    m: MatrixN<N, D>,
}

impl<N: ComplexField, D: Dim> RandomOrthogonal<N, D>
where DefaultAllocator: Allocator<N, D, D>
{
    /// Retrieve the generated matrix.
    pub fn unwrap(self) -> MatrixN<N, D> {
        self.m
    }

    /// Creates a new random orthogonal matrix from its dimension and a random reals generators.
    pub fn new<Rand: FnMut() -> N>(dim: D, mut rand: Rand) -> Self {
        let mut res = MatrixN::identity_generic(dim, dim);

        // Create an orthogonal matrix by composing random Givens rotations rotations.
        for i in 0..dim.value() - 1 {
            let rot = GivensRotation::new(rand(), rand()).0;
            rot.rotate(&mut res.fixed_rows_mut::<U2>(i));
        }

        RandomOrthogonal { m: res }
    }
}

#[cfg(feature = "arbitrary")]
impl<N: ComplexField + Arbitrary + Send, D: Dim> Arbitrary for RandomOrthogonal<N, D>
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
