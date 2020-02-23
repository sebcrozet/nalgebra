use alga::general::{
    AbstractGroup, AbstractLoop, AbstractMagma, AbstractMonoid, AbstractQuasigroup,
    AbstractSemigroup, Id, Identity, TwoSidedInverse, Multiplicative, RealField,
};
use alga::linear::{
    self, AffineTransformation, DirectIsometry, Isometry, OrthogonalTransformation,
    ProjectiveTransformation, Similarity, Transformation,
};

use crate::base::allocator::Allocator;
use crate::base::dimension::DimName;
use crate::base::{DefaultAllocator, VectorN};

use crate::geometry::{Point, Rotation};

/*
 *
 * Algebraic structures.
 *
 */
impl<N: RealField, D: DimName> Identity<Multiplicative> for Rotation<N, D>
where DefaultAllocator: Allocator<N, D, D>
{
    #[inline]
    fn identity() -> Self {
        Self::identity()
    }
}

impl<N: RealField, D: DimName> TwoSidedInverse<Multiplicative> for Rotation<N, D>
where DefaultAllocator: Allocator<N, D, D>
{
    #[inline]
    #[must_use = "Did you mean to use two_sided_inverse_mut()?"]
    fn two_sided_inverse(&self) -> Self {
        self.transpose()
    }

    #[inline]
    fn two_sided_inverse_mut(&mut self) {
        self.transpose_mut()
    }
}

impl<N: RealField, D: DimName> AbstractMagma<Multiplicative> for Rotation<N, D>
where DefaultAllocator: Allocator<N, D, D>
{
    #[inline]
    fn operate(&self, rhs: &Self) -> Self {
        self * rhs
    }
}

macro_rules! impl_multiplicative_structures(
    ($($marker: ident<$operator: ident>),* $(,)*) => {$(
        impl<N: RealField, D: DimName> $marker<$operator> for Rotation<N, D>
            where DefaultAllocator: Allocator<N, D, D> { }
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
impl<N: RealField, D: DimName> Transformation<Point<N, D>> for Rotation<N, D>
where DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
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

impl<N: RealField, D: DimName> ProjectiveTransformation<Point<N, D>> for Rotation<N, D>
where DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
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

impl<N: RealField, D: DimName> AffineTransformation<Point<N, D>> for Rotation<N, D>
where DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
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

impl<N: RealField, D: DimName> Similarity<Point<N, D>> for Rotation<N, D>
where DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
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
        impl<N: RealField, D: DimName> $Trait<Point<N, D>> for Rotation<N, D>
        where DefaultAllocator: Allocator<N, D, D> +
                                Allocator<N, D> { }
    )*}
);

marker_impl!(Isometry, DirectIsometry, OrthogonalTransformation);

/// Subgroups of the n-dimensional rotation group `SO(n)`.
impl<N: RealField, D: DimName> linear::Rotation<Point<N, D>> for Rotation<N, D>
where DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>
{
    #[inline]
    fn powf(&self, _: N) -> Option<Self> {
        // XXX: Add the general case.
        // XXX: Use specialization for 2D and 3D.
        unimplemented!()
    }

    #[inline]
    fn rotation_between(_: &VectorN<N, D>, _: &VectorN<N, D>) -> Option<Self> {
        // XXX: Add the general case.
        // XXX: Use specialization for 2D and 3D.
        unimplemented!()
    }

    #[inline]
    fn scaled_rotation_between(_: &VectorN<N, D>, _: &VectorN<N, D>, _: N) -> Option<Self> {
        // XXX: Add the general case.
        // XXX: Use specialization for 2D and 3D.
        unimplemented!()
    }
}

/*
impl<N: RealField> Matrix for Rotation<N> {
    type Field     = N;
    type Row       = Matrix<N>;
    type Column    = Matrix<N>;
    type Transpose = Self;

    #[inline]
    fn nrows(&self) -> usize {
        self.submatrix.nrows()
    }

    #[inline]
    fn ncolumns(&self) -> usize {
        self.submatrix.ncolumns()
    }

    #[inline]
    fn row(&self, i: usize) -> Self::Row {
        self.submatrix.row(i)
    }

    #[inline]
    fn column(&self, i: usize) -> Self::Column {
        self.submatrix.column(i)
    }

    #[inline]
    fn get(&self, i: usize, j: usize) -> Self::Field {
        self.submatrix[(i, j)]
    }

    #[inline]
    unsafe fn get_unchecked(&self, i: usize, j: usize) -> Self::Field {
        self.submatrix.at_fast(i, j)
    }

    #[inline]
    fn transpose(&self) -> Self::Transpose {
        Rotation::from_matrix_unchecked(self.submatrix.transpose())
    }
}

impl<N: RealField> SquareMatrix for Rotation<N> {
    type Vector = Matrix<N>;

    #[inline]
    fn diagonal(&self) -> Self::Coordinates {
        self.submatrix.diagonal()
    }

    #[inline]
    fn determinant(&self) -> Self::Field {
        crate::one()
    }

    #[inline]
    fn try_inverse(&self) -> Option<Self> {
        Some(::transpose(self))
    }

    #[inline]
    fn try_inverse_mut(&mut self) -> bool {
        self.transpose_mut();
        true
    }

    #[inline]
    fn transpose_mut(&mut self) {
        self.submatrix.transpose_mut()
    }
}

impl<N: RealField> InversibleSquareMatrix for Rotation<N> { }
*/
