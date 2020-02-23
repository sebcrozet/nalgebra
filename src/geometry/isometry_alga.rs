use alga::general::{
    AbstractGroup, AbstractLoop, AbstractMagma, AbstractMonoid, AbstractQuasigroup,
    AbstractSemigroup, Id, Identity, TwoSidedInverse, Multiplicative, RealField,
};
use alga::linear::Isometry as AlgaIsometry;
use alga::linear::{
    AffineTransformation, DirectIsometry, ProjectiveTransformation, Rotation, Similarity,
    Transformation,
};

use crate::base::allocator::Allocator;
use crate::base::dimension::DimName;
use crate::base::{DefaultAllocator, VectorN};

use crate::geometry::{Isometry, Point, Translation};

/*
 *
 * Algebraic structures.
 *
 */
impl<N: RealField, D: DimName, R> Identity<Multiplicative> for Isometry<N, D, R>
where
    R: Rotation<Point<N, D>>,
    DefaultAllocator: Allocator<N, D>,
{
    #[inline]
    fn identity() -> Self {
        Self::identity()
    }
}

impl<N: RealField, D: DimName, R> TwoSidedInverse<Multiplicative> for Isometry<N, D, R>
where
    R: Rotation<Point<N, D>>,
    DefaultAllocator: Allocator<N, D>,
{
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

impl<N: RealField, D: DimName, R> AbstractMagma<Multiplicative> for Isometry<N, D, R>
where
    R: Rotation<Point<N, D>>,
    DefaultAllocator: Allocator<N, D>,
{
    #[inline]
    fn operate(&self, rhs: &Self) -> Self {
        self * rhs
    }
}

macro_rules! impl_multiplicative_structures(
    ($($marker: ident<$operator: ident>),* $(,)*) => {$(
        impl<N: RealField, D: DimName, R> $marker<$operator> for Isometry<N, D, R>
            where R: Rotation<Point<N, D>>,
                  DefaultAllocator: Allocator<N, D> { }
    )*}
);

impl_multiplicative_structures!(
    AbstractSemigroup<Multiplicative>,
    AbstractMonoid<Multiplicative>,
    AbstractQuasigroup<Multiplicative>,
    AbstractLoop<Multiplicative>,
    AbstractGroup<Multiplicative>
);

/*
 *
 * Transformation groups.
 *
 */
impl<N: RealField, D: DimName, R> Transformation<Point<N, D>> for Isometry<N, D, R>
where
    R: Rotation<Point<N, D>>,
    DefaultAllocator: Allocator<N, D>,
{
    #[inline]
    fn transform_point(&self, pt: &Point<N, D>) -> Point<N, D> {
        self.transform_point(pt)
    }

    #[inline]
    fn transform_vector(&self, v: &VectorN<N, D>) -> VectorN<N, D> {
        self.transform_vector(v)
    }
}

impl<N: RealField, D: DimName, R> ProjectiveTransformation<Point<N, D>> for Isometry<N, D, R>
where
    R: Rotation<Point<N, D>>,
    DefaultAllocator: Allocator<N, D>,
{
    #[inline]
    fn inverse_transform_point(&self, pt: &Point<N, D>) -> Point<N, D> {
        self.inverse_transform_point(pt)
    }

    #[inline]
    fn inverse_transform_vector(&self, v: &VectorN<N, D>) -> VectorN<N, D> {
        self.inverse_transform_vector(v)
    }
}

impl<N: RealField, D: DimName, R> AffineTransformation<Point<N, D>> for Isometry<N, D, R>
where
    R: Rotation<Point<N, D>>,
    DefaultAllocator: Allocator<N, D>,
{
    type Rotation = R;
    type NonUniformScaling = Id;
    type Translation = Translation<N, D>;

    #[inline]
    fn decompose(&self) -> (Self::Translation, R, Id, R) {
        (
            self.translation.clone(),
            self.rotation.clone(),
            Id::new(),
            R::identity(),
        )
    }

    #[inline]
    fn append_translation(&self, t: &Self::Translation) -> Self {
        t * self
    }

    #[inline]
    fn prepend_translation(&self, t: &Self::Translation) -> Self {
        self * t
    }

    #[inline]
    fn append_rotation(&self, r: &Self::Rotation) -> Self {
        let shift = r.transform_vector(&self.translation.vector);
        Isometry::from_parts(Translation::from(shift), r.clone() * self.rotation.clone())
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

    #[inline]
    fn append_rotation_wrt_point(&self, r: &Self::Rotation, p: &Point<N, D>) -> Option<Self> {
        let mut res = self.clone();
        res.append_rotation_wrt_point_mut(r, p);
        Some(res)
    }
}

impl<N: RealField, D: DimName, R> Similarity<Point<N, D>> for Isometry<N, D, R>
where
    R: Rotation<Point<N, D>>,
    DefaultAllocator: Allocator<N, D>,
{
    type Scaling = Id;

    #[inline]
    fn translation(&self) -> Translation<N, D> {
        self.translation.clone()
    }

    #[inline]
    fn rotation(&self) -> R {
        self.rotation.clone()
    }

    #[inline]
    fn scaling(&self) -> Id {
        Id::new()
    }
}

macro_rules! marker_impl(
    ($($Trait: ident),*) => {$(
        impl<N: RealField, D: DimName, R> $Trait<Point<N, D>> for Isometry<N, D, R>
        where R: Rotation<Point<N, D>>,
              DefaultAllocator: Allocator<N, D> { }
    )*}
);

marker_impl!(AlgaIsometry, DirectIsometry);
