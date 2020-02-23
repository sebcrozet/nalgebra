use alga::general::{
    AbstractGroup, AbstractLoop, AbstractMagma, AbstractMonoid, AbstractQuasigroup,
    AbstractSemigroup, Id, Identity, TwoSidedInverse, Multiplicative, RealField,
};
use alga::linear::{
    AffineTransformation, DirectIsometry, Isometry, OrthogonalTransformation,
    ProjectiveTransformation, Rotation, Similarity, Transformation,
};

use crate::base::allocator::Allocator;
use crate::base::dimension::U2;
use crate::base::{DefaultAllocator, Vector2};
use crate::geometry::{Point2, UnitComplex};

/*
 *
 * Implementations for UnitComplex.
 *
 */
impl<N: RealField> Identity<Multiplicative> for UnitComplex<N> {
    #[inline]
    fn identity() -> Self {
        Self::identity()
    }
}

impl<N: RealField> AbstractMagma<Multiplicative> for UnitComplex<N> {
    #[inline]
    fn operate(&self, rhs: &Self) -> Self {
        self * rhs
    }
}

impl<N: RealField> TwoSidedInverse<Multiplicative> for UnitComplex<N> {
    #[inline]
    #[must_use = "Did you mean to use two_sided_inverse_mut()?"]
    fn two_sided_inverse(&self) -> Self {
        self.inverse()
    }

    #[inline]
    fn two_sided_inverse_mut(&mut self) {
        self.inverse_mut()
    }
}

macro_rules! impl_structures(
    ($($marker: ident<$operator: ident>),* $(,)*) => {$(
        impl<N: RealField> $marker<$operator> for UnitComplex<N> {
        }
    )*}
);

impl_structures!(
    AbstractSemigroup<Multiplicative>,
    AbstractQuasigroup<Multiplicative>,
    AbstractMonoid<Multiplicative>,
    AbstractLoop<Multiplicative>,
    AbstractGroup<Multiplicative>
);

impl<N: RealField> Transformation<Point2<N>> for UnitComplex<N>
where DefaultAllocator: Allocator<N, U2>
{
    #[inline]
    fn transform_point(&self, pt: &Point2<N>) -> Point2<N> {
        self.transform_point(pt)
    }

    #[inline]
    fn transform_vector(&self, v: &Vector2<N>) -> Vector2<N> {
        self.transform_vector(v)
    }
}

impl<N: RealField> ProjectiveTransformation<Point2<N>> for UnitComplex<N>
where DefaultAllocator: Allocator<N, U2>
{
    #[inline]
    fn inverse_transform_point(&self, pt: &Point2<N>) -> Point2<N> {
        self.inverse_transform_point(pt)
    }

    #[inline]
    fn inverse_transform_vector(&self, v: &Vector2<N>) -> Vector2<N> {
        self.inverse_transform_vector(v)
    }
}

impl<N: RealField> AffineTransformation<Point2<N>> for UnitComplex<N>
where DefaultAllocator: Allocator<N, U2>
{
    type Rotation = Self;
    type NonUniformScaling = Id;
    type Translation = Id;

    #[inline]
    fn decompose(&self) -> (Id, Self, Id, Self) {
        (Id::new(), self.clone(), Id::new(), Self::identity())
    }

    #[inline]
    fn append_translation(&self, _: &Self::Translation) -> Self {
        self.clone()
    }

    #[inline]
    fn prepend_translation(&self, _: &Self::Translation) -> Self {
        self.clone()
    }

    #[inline]
    fn append_rotation(&self, r: &Self::Rotation) -> Self {
        r * self
    }

    #[inline]
    fn prepend_rotation(&self, r: &Self::Rotation) -> Self {
        self * r
    }

    #[inline]
    fn append_scaling(&self, _: &Self::NonUniformScaling) -> Self {
        self.clone()
    }

    #[inline]
    fn prepend_scaling(&self, _: &Self::NonUniformScaling) -> Self {
        self.clone()
    }
}

impl<N: RealField> Similarity<Point2<N>> for UnitComplex<N>
where DefaultAllocator: Allocator<N, U2>
{
    type Scaling = Id;

    #[inline]
    fn translation(&self) -> Id {
        Id::new()
    }

    #[inline]
    fn rotation(&self) -> Self {
        self.clone()
    }

    #[inline]
    fn scaling(&self) -> Id {
        Id::new()
    }
}

macro_rules! marker_impl(
    ($($Trait: ident),*) => {$(
        impl<N: RealField> $Trait<Point2<N>> for UnitComplex<N>
        where DefaultAllocator: Allocator<N, U2> { }
    )*}
);

marker_impl!(Isometry, DirectIsometry, OrthogonalTransformation);

impl<N: RealField> Rotation<Point2<N>> for UnitComplex<N>
where DefaultAllocator: Allocator<N, U2>
{
    #[inline]
    fn powf(&self, n: N) -> Option<Self> {
        Some(self.powf(n))
    }

    #[inline]
    fn rotation_between(a: &Vector2<N>, b: &Vector2<N>) -> Option<Self> {
        Some(Self::rotation_between(a, b))
    }

    #[inline]
    fn scaled_rotation_between(a: &Vector2<N>, b: &Vector2<N>, s: N) -> Option<Self> {
        Some(Self::scaled_rotation_between(a, b, s))
    }
}
