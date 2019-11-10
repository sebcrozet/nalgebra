use num::{One, Zero};
use std::ops::{Div, DivAssign, Index, IndexMut, Mul, MulAssign};

use alga::general::{ClosedAdd, ClosedMul, RealField, SubsetOf};

use crate::base::allocator::Allocator;
use crate::{DimName, DimNameAdd, DimNameSum, U1, U3, U4};
use crate::base::{DefaultAllocator, MatrixN, Scalar, VectorN};

use crate::geometry::{
    Isometry, Point, Rotation, Similarity, SubTCategoryOf, SuperTCategoryOf, TAffine, TCategory,
    TCategoryMul, TGeneral, TProjective, Transform, Translation, UnitQuaternion,
};

/*
 *
 * In the following, we provide:
 * =========================
 *
 * Index<(usize, usize)>
 * IndexMut<(usize, usize)> (where TCategory == TGeneral)
 *
 * (Operators)
 *
 * Transform × Isometry
 * Transform × Rotation
 * Transform × Similarity
 * Transform × Transform
 * Transform × UnitQuaternion
 * FIXME: Transform × UnitComplex
 * Transform × Translation
 * Transform × Vector
 * Transform × Point
 *
 * Isometry       × Transform
 * Rotation       × Transform
 * Similarity     × Transform
 * Translation    × Transform
 * UnitQuaternion × Transform
 * FIXME: UnitComplex × Transform
 *
 * FIXME: Transform ÷ Isometry
 * Transform ÷ Rotation
 * FIXME: Transform ÷ Similarity
 * Transform ÷ Transform
 * Transform ÷ UnitQuaternion
 * Transform ÷ Translation
 *
 * FIXME: Isometry       ÷ Transform
 * Rotation       ÷ Transform
 * FIXME: Similarity     ÷ Transform
 * Translation    ÷ Transform
 * UnitQuaternion ÷ Transform
 * FIXME: UnitComplex ÷ Transform
 *
 *
 * (Assignment Operators)
 *
 *
 * Transform ×= Transform
 * Transform ×= Similarity
 * Transform ×= Isometry
 * Transform ×= Rotation
 * Transform ×= UnitQuaternion
 * FIXME: Transform ×= UnitComplex
 * Transform ×= Translation
 *
 * Transform ÷= Transform
 * FIXME: Transform ÷= Similarity
 * FIXME: Transform ÷= Isometry
 * Transform ÷= Rotation
 * Transform ÷= UnitQuaternion
 * FIXME: Transform ÷= UnitComplex
 *
 */

/*
 *
 * Indexing.
 *
 */
impl<N: RealField, D, C: TCategory> Index<(usize, usize)> for Transform<N, D, C>
where
    D: DimName + DimNameAdd<U1>,
    DefaultAllocator: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>>,
{
    type Output = N;

    #[inline]
    fn index(&self, ij: (usize, usize)) -> &N {
        self.matrix().index(ij)
    }
}

// Only general transformations are mutably indexable.
impl<N: RealField, D> IndexMut<(usize, usize)> for Transform<N, D, TGeneral>
where
    D: DimName + DimNameAdd<U1>,
    DefaultAllocator: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>>,
{
    #[inline]
    fn index_mut(&mut self, ij: (usize, usize)) -> &mut N {
        self.matrix_mut().index_mut(ij)
    }
}

// Transform × Vector
md_impl_all!(
    Mul, mul where N: RealField;
    (DimNameSum<D, U1>, DimNameSum<D, U1>), (D, U1) for D: DimNameAdd<U1>, C: TCategory;
    self: Transform<N, D, C>, rhs: VectorN<N, D>, Output = VectorN<N, D>;
    [val val] => &self * &rhs;
    [ref val] =>  self * &rhs;
    [val ref] => &self *  rhs;
    [ref ref] => {
        let transform = self.matrix().fixed_slice::<D, D>(0, 0);

        if C::has_normalizer() {
            let normalizer = self.matrix().fixed_slice::<U1, D>(D::dim(), 0);
            let n = normalizer.tr_dot(&rhs);

            if !n.is_zero() {
                return transform * (rhs / n);
            }
        }

        transform * rhs
    };
);

// Transform × Point
md_impl_all!(
    Mul, mul where N: RealField;
    (DimNameSum<D, U1>, DimNameSum<D, U1>), (D, U1) for D: DimNameAdd<U1>, C: TCategory
    where DefaultAllocator: Allocator<N, D, D>;
    self: Transform<N, D, C>, rhs: Point<N, D>, Output = Point<N, D>;
    [val val] => &self * &rhs;
    [ref val] =>  self * &rhs;
    [val ref] => &self *  rhs;
    [ref ref] => {
        let transform   = self.matrix().fixed_slice::<D, D>(0, 0);
        let translation = self.matrix().fixed_slice::<D, U1>(0, D::dim());

        if C::has_normalizer() {
            let normalizer = self.matrix().fixed_slice::<U1, D>(D::dim(), 0);
            let n = normalizer.tr_dot(&rhs.coords) + unsafe { *self.matrix().get_unchecked((D::dim(), D::dim())) };

            if !n.is_zero() {
                return (transform * rhs + translation) / n;
            }
        }

        transform * rhs + translation
    };
);

// Transform × Transform
md_impl_all!(
    Mul, mul where N: RealField;
    (DimNameSum<D, U1>, DimNameSum<D, U1>), (DimNameSum<D, U1>, DimNameSum<D, U1>) for D: DimNameAdd<U1>, CA: TCategoryMul<CB>, CB: TCategory;
    self: Transform<N, D, CA>, rhs: Transform<N, D, CB>, Output = Transform<N, D, CA::Representative>;
    [val val] => Self::Output::from_matrix_unchecked(self.into_inner() * rhs.into_inner());
    [ref val] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.into_inner());
    [val ref] => Self::Output::from_matrix_unchecked(self.into_inner() * rhs.matrix());
    [ref ref] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.matrix());
);

// Transform × Rotation
md_impl_all!(
    Mul, mul where N: RealField;
    (DimNameSum<D, U1>, DimNameSum<D, U1>), (D, D) for D: DimNameAdd<U1>, C: TCategoryMul<TAffine>;
    self: Transform<N, D, C>, rhs: Rotation<N, D>, Output = Transform<N, D, C::Representative>;
    [val val] => Self::Output::from_matrix_unchecked(self.into_inner() * rhs.to_homogeneous());
    [ref val] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.to_homogeneous());
    [val ref] => Self::Output::from_matrix_unchecked(self.into_inner() * rhs.to_homogeneous());
    [ref ref] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.to_homogeneous());
);

// Rotation × Transform
md_impl_all!(
    Mul, mul where N: RealField;
    (D, D), (DimNameSum<D, U1>, DimNameSum<D, U1>) for D: DimNameAdd<U1>, C: TCategoryMul<TAffine>;
    self: Rotation<N, D>, rhs: Transform<N, D, C>, Output = Transform<N, D, C::Representative>;
    [val val] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.into_inner());
    [ref val] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.into_inner());
    [val ref] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.matrix());
    [ref ref] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.matrix());
);

// Transform × UnitQuaternion
md_impl_all!(
    Mul, mul where N: RealField;
    (U4, U4), (U4, U1) for C: TCategoryMul<TAffine>;
    self: Transform<N, U3, C>, rhs: UnitQuaternion<N>, Output = Transform<N, U3, C::Representative>;
    [val val] => Self::Output::from_matrix_unchecked(self.into_inner() * rhs.to_homogeneous());
    [ref val] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.to_homogeneous());
    [val ref] => Self::Output::from_matrix_unchecked(self.into_inner() * rhs.to_homogeneous());
    [ref ref] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.to_homogeneous());
);

// UnitQuaternion × Transform
md_impl_all!(
    Mul, mul where N: RealField;
    (U4, U1), (U4, U4) for C: TCategoryMul<TAffine>;
    self: UnitQuaternion<N>, rhs: Transform<N, U3, C>, Output = Transform<N, U3, C::Representative>;
    [val val] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.into_inner());
    [ref val] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.into_inner());
    [val ref] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.matrix());
    [ref ref] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.matrix());
);

// Transform × Isometry
md_impl_all!(
    Mul, mul where N: RealField;
    (DimNameSum<D, U1>, DimNameSum<D, U1>), (D, U1)
    for D: DimNameAdd<U1>, C: TCategoryMul<TAffine>, R: SubsetOf<MatrixN<N, DimNameSum<D, U1>> >;
    self: Transform<N, D, C>, rhs: Isometry<N, D, R>, Output = Transform<N, D, C::Representative>;
    [val val] => Self::Output::from_matrix_unchecked(self.into_inner() * rhs.to_homogeneous());
    [ref val] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.to_homogeneous());
    [val ref] => Self::Output::from_matrix_unchecked(self.into_inner() * rhs.to_homogeneous());
    [ref ref] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.to_homogeneous());
);

// Isometry × Transform
md_impl_all!(
    Mul, mul where N: RealField;
    (D, U1), (DimNameSum<D, U1>, DimNameSum<D, U1>)
    for D: DimNameAdd<U1>, C: TCategoryMul<TAffine>, R: SubsetOf<MatrixN<N, DimNameSum<D, U1>> >;
    self: Isometry<N, D, R>, rhs: Transform<N, D, C>, Output = Transform<N, D, C::Representative>;
    [val val] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.into_inner());
    [ref val] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.into_inner());
    [val ref] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.matrix());
    [ref ref] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.matrix());
);

// Transform × Similarity
md_impl_all!(
    Mul, mul where N: RealField;
    (DimNameSum<D, U1>, DimNameSum<D, U1>), (D, U1)
    for D: DimNameAdd<U1>, C: TCategoryMul<TAffine>, R: SubsetOf<MatrixN<N, DimNameSum<D, U1>> >;
    self: Transform<N, D, C>, rhs: Similarity<N, D, R>, Output = Transform<N, D, C::Representative>;
    [val val] => Self::Output::from_matrix_unchecked(self.into_inner() * rhs.to_homogeneous());
    [ref val] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.to_homogeneous());
    [val ref] => Self::Output::from_matrix_unchecked(self.into_inner() * rhs.to_homogeneous());
    [ref ref] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.to_homogeneous());
);

// Similarity × Transform
md_impl_all!(
    Mul, mul where N: RealField;
    (D, U1), (DimNameSum<D, U1>, DimNameSum<D, U1>)
    for D: DimNameAdd<U1>, C: TCategoryMul<TAffine>, R: SubsetOf<MatrixN<N, DimNameSum<D, U1>> >;
    self: Similarity<N, D, R>, rhs: Transform<N, D, C>, Output = Transform<N, D, C::Representative>;
    [val val] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.into_inner());
    [ref val] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.into_inner());
    [val ref] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.matrix());
    [ref ref] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.matrix());
);

/*
 *
 * FIXME: don't explicitly build the homogeneous translation matrix.
 * Directly apply the translation, just as in `Matrix::{append,prepend}_translation`. This has not
 * been done yet because of the `DimNameDiff` requirement (which is not automatically deduced from
 * `DimNameAdd` requirement).
 *
 */
// Transform × Translation
md_impl_all!(
    Mul, mul where N: RealField;
    (DimNameSum<D, U1>, DimNameSum<D, U1>), (D, U1) for D: DimNameAdd<U1>, C: TCategoryMul<TAffine>;
    self: Transform<N, D, C>, rhs: Translation<N, D>, Output = Transform<N, D, C::Representative>;
    [val val] => Self::Output::from_matrix_unchecked(self.into_inner() * rhs.to_homogeneous());
    [ref val] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.to_homogeneous());
    [val ref] => Self::Output::from_matrix_unchecked(self.into_inner() * rhs.to_homogeneous());
    [ref ref] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.to_homogeneous());
);

// Translation × Transform
md_impl_all!(
    Mul, mul where N: RealField;
    (D, U1), (DimNameSum<D, U1>, DimNameSum<D, U1>)
    for D: DimNameAdd<U1>, C: TCategoryMul<TAffine>;
    self: Translation<N, D>, rhs: Transform<N, D, C>, Output = Transform<N, D, C::Representative>;
    [val val] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.into_inner());
    [ref val] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.into_inner());
    [val ref] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.matrix());
    [ref ref] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.matrix());
);

// Transform ÷ Transform
md_impl_all!(
    Div, div where N: RealField;
    (DimNameSum<D, U1>, DimNameSum<D, U1>), (DimNameSum<D, U1>, DimNameSum<D, U1>) for D: DimNameAdd<U1>, CA: TCategoryMul<CB>, CB: SubTCategoryOf<TProjective>;
    self: Transform<N, D, CA>, rhs: Transform<N, D, CB>, Output = Transform<N, D, CA::Representative>;
    [val val] => self * rhs.inverse();
    [ref val] => self * rhs.inverse();
    [val ref] => self * rhs.clone().inverse();
    [ref ref] => self * rhs.clone().inverse();
);

// Transform ÷ Rotation
md_impl_all!(
    Div, div where N: RealField;
    (DimNameSum<D, U1>, DimNameSum<D, U1>), (D, D) for D: DimNameAdd<U1>, C: TCategoryMul<TAffine>;
    self: Transform<N, D, C>, rhs: Rotation<N, D>, Output = Transform<N, D, C::Representative>;
    [val val] => self * rhs.inverse();
    [ref val] => self * rhs.inverse();
    [val ref] => self * rhs.inverse();
    [ref ref] => self * rhs.inverse();
);

// Rotation ÷ Transform
md_impl_all!(
    Div, div where N: RealField;
    (D, D), (DimNameSum<D, U1>, DimNameSum<D, U1>) for D: DimNameAdd<U1>, C: TCategoryMul<TAffine>;
    self: Rotation<N, D>, rhs: Transform<N, D, C>, Output = Transform<N, D, C::Representative>;
    [val val] => self.inverse() * rhs;
    [ref val] => self.inverse() * rhs;
    [val ref] => self.inverse() * rhs;
    [ref ref] => self.inverse() * rhs;
);

// Transform ÷ UnitQuaternion
md_impl_all!(
    Div, div where N: RealField;
    (U4, U4), (U4, U1) for C: TCategoryMul<TAffine>;
    self: Transform<N, U3, C>, rhs: UnitQuaternion<N>, Output = Transform<N, U3, C::Representative>;
    [val val] => self * rhs.inverse();
    [ref val] => self * rhs.inverse();
    [val ref] => self * rhs.inverse();
    [ref ref] => self * rhs.inverse();
);

// UnitQuaternion ÷ Transform
md_impl_all!(
    Div, div where N: RealField;
    (U4, U1), (U4, U4) for C: TCategoryMul<TAffine>;
    self: UnitQuaternion<N>, rhs: Transform<N, U3, C>, Output = Transform<N, U3, C::Representative>;
    [val val] => self.inverse() * rhs;
    [ref val] => self.inverse() * rhs;
    [val ref] => self.inverse() * rhs;
    [ref ref] => self.inverse() * rhs;
);

//      // Transform ÷ Isometry
//      md_impl_all!(
//          Div, div where N: RealField;
//          (DimNameSum<D, U1>, DimNameSum<D, U1>), (D, U1)
//          for D: DimNameAdd<U1>, C: TCategoryMul<TAffine>, R: SubsetOf<MatrixN<N, DimNameSum<D, U1>> >
//          where SB::Alloc: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1> >;
//          self: Transform<N, D, C>, rhs: Isometry<N, D, R>, Output = Transform<N, D, C::Representative>;
//          [val val] => Self::Output::from_matrix_unchecked(self.into_inner() * rhs.inverse().to_homogeneous());
//          [ref val] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.inverse().to_homogeneous());
//          [val ref] => Self::Output::from_matrix_unchecked(self.into_inner() * rhs.inverse().to_homogeneous());
//          [ref ref] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.inverse().to_homogeneous());
//      );

//      // Isometry ÷ Transform
//      md_impl_all!(
//          Div, div where N: RealField;
//          (D, U1), (DimNameSum<D, U1>, DimNameSum<D, U1>)
//          for D: DimNameAdd<U1>, C: TCategoryMul<TAffine>, R: SubsetOf<MatrixN<N, DimNameSum<D, U1>> >
//          where SA::Alloc: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1> >;
//          self: Isometry<N, D, R>, rhs: Transform<N, D, C>, Output = Transform<N, D, C::Representative>;
//          [val val] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.into_inner());
//          [ref val] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.into_inner());
//          [val ref] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.matrix());
//          [ref ref] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.matrix());
//      );

//      // Transform ÷ Similarity
//      md_impl_all!(
//          Div, div where N: RealField;
//          (DimNameSum<D, U1>, DimNameSum<D, U1>), (D, U1)
//          for D: DimNameAdd<U1>, C: TCategoryMul<TAffine>, R: SubsetOf<MatrixN<N, DimNameSum<D, U1>> >
//          where SB::Alloc: Allocator<N, D, D >
//          where SB::Alloc: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1> >;
//          self: Transform<N, D, C>, rhs: Similarity<N, D, R>, Output = Transform<N, D, C::Representative>;
//          [val val] => Self::Output::from_matrix_unchecked(self.into_inner() * rhs.to_homogeneous());
//          [ref val] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.to_homogeneous());
//          [val ref] => Self::Output::from_matrix_unchecked(self.into_inner() * rhs.to_homogeneous());
//          [ref ref] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.to_homogeneous());
//      );

//      // Similarity ÷ Transform
//      md_impl_all!(
//          Div, div where N: RealField;
//          (D, U1), (DimNameSum<D, U1>, DimNameSum<D, U1>)
//          for D: DimNameAdd<U1>, C: TCategoryMul<TAffine>, R: SubsetOf<MatrixN<N, DimNameSum<D, U1>> >
//          where SA::Alloc: Allocator<N, D, D >
//          where SA::Alloc: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1> >;
//          self: Similarity<N, D, R>, rhs: Transform<N, D, C>, Output = Transform<N, D, C::Representative>;
//          [val val] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.into_inner());
//          [ref val] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.into_inner());
//          [val ref] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.matrix());
//          [ref ref] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.matrix());
//      );

// Transform ÷ Translation
md_impl_all!(
    Div, div where N: RealField;
    (DimNameSum<D, U1>, DimNameSum<D, U1>), (D, U1) for D: DimNameAdd<U1>, C: TCategoryMul<TAffine>;
    self: Transform<N, D, C>, rhs: Translation<N, D>, Output = Transform<N, D, C::Representative>;
    [val val] => self * rhs.inverse();
    [ref val] => self * rhs.inverse();
    [val ref] => self * rhs.inverse();
    [ref ref] => self * rhs.inverse();
);

// Translation ÷ Transform
md_impl_all!(
    Div, div where N: RealField;
    (D, U1), (DimNameSum<D, U1>, DimNameSum<D, U1>)
    for D: DimNameAdd<U1>, C: TCategoryMul<TAffine>;
    self: Translation<N, D>, rhs: Transform<N, D, C>, Output = Transform<N, D, C::Representative>;
    [val val] => self.inverse() * rhs;
    [ref val] => self.inverse() * rhs;
    [val ref] => self.inverse() * rhs;
    [ref ref] => self.inverse() * rhs;
);

// Transform ×= Transform
md_assign_impl_all!(
    MulAssign, mul_assign where N: RealField;
    (DimNameSum<D, U1>, DimNameSum<D, U1>), (DimNameSum<D, U1>, DimNameSum<D, U1>) for D: DimNameAdd<U1>, CA: TCategory, CB: SubTCategoryOf<CA>;
    self: Transform<N, D, CA>, rhs: Transform<N, D, CB>;
    [val] => *self.matrix_mut_unchecked() *= rhs.into_inner();
    [ref] => *self.matrix_mut_unchecked() *= rhs.matrix();
);

// Transform ×= Similarity
md_assign_impl_all!(
    MulAssign, mul_assign where N: RealField;
    (DimNameSum<D, U1>, DimNameSum<D, U1>), (D, U1)
    for D: DimNameAdd<U1>, C: TCategory, R: SubsetOf<MatrixN<N, DimNameSum<D, U1>> >;
    self: Transform<N, D, C>, rhs: Similarity<N, D, R>;
    [val] => *self.matrix_mut_unchecked() *= rhs.to_homogeneous();
    [ref] => *self.matrix_mut_unchecked() *= rhs.to_homogeneous();
);

// Transform ×= Isometry
md_assign_impl_all!(
    MulAssign, mul_assign where N: RealField;
    (DimNameSum<D, U1>, DimNameSum<D, U1>), (D, U1)
    for D: DimNameAdd<U1>, C: TCategory, R: SubsetOf<MatrixN<N, DimNameSum<D, U1>> >;
    self: Transform<N, D, C>, rhs: Isometry<N, D, R>;
    [val] => *self.matrix_mut_unchecked() *= rhs.to_homogeneous();
    [ref] => *self.matrix_mut_unchecked() *= rhs.to_homogeneous();
);

/*
 *
 * FIXME: don't explicitly build the homogeneous translation matrix.
 * Directly apply the translation, just as in `Matrix::{append,prepend}_translation`. This has not
 * been done yet because of the `DimNameDiff` requirement (which is not automatically deduced from
 * `DimNameAdd` requirement).
 *
 */
// Transform ×= Translation
md_assign_impl_all!(
    MulAssign, mul_assign where N: RealField;
    (DimNameSum<D, U1>, DimNameSum<D, U1>), (D, U1) for D: DimNameAdd<U1>, C: TCategory;
    self: Transform<N, D, C>, rhs: Translation<N, D>;
    [val] => *self.matrix_mut_unchecked() *= rhs.to_homogeneous();
    [ref] => *self.matrix_mut_unchecked() *= rhs.to_homogeneous();
);

// Transform ×= Rotation
md_assign_impl_all!(
    MulAssign, mul_assign where N: RealField;
    (DimNameSum<D, U1>, DimNameSum<D, U1>), (D, D) for D: DimNameAdd<U1>, C: TCategory;
    self: Transform<N, D, C>, rhs: Rotation<N, D>;
    [val] => *self.matrix_mut_unchecked() *= rhs.to_homogeneous();
    [ref] => *self.matrix_mut_unchecked() *= rhs.to_homogeneous();
);

// Transform ×= UnitQuaternion
md_assign_impl_all!(
    MulAssign, mul_assign where N: RealField;
    (U4, U4), (U4, U1) for C: TCategory;
    self: Transform<N, U3, C>, rhs: UnitQuaternion<N>;
    [val] => *self.matrix_mut_unchecked() *= rhs.to_homogeneous();
    [ref] => *self.matrix_mut_unchecked() *= rhs.to_homogeneous();
);

// Transform ÷= Transform
md_assign_impl_all!(
    DivAssign, div_assign where N: RealField;
    (DimNameSum<D, U1>, DimNameSum<D, U1>), (DimNameSum<D, U1>, DimNameSum<D, U1>)
    for D: DimNameAdd<U1>, CA: SuperTCategoryOf<CB>, CB: SubTCategoryOf<TProjective>;
    self: Transform<N, D, CA>, rhs: Transform<N, D, CB>;
    [val] => *self *= rhs.inverse();
    [ref] => *self *= rhs.clone().inverse();
);

//      // Transform ÷= Similarity
//      md_assign_impl_all!(
//          DivAssign, div_assign;
//          (DimNameSum<D, U1>, DimNameSum<D, U1>), (D, U1)
//          for D: DimNameAdd<U1>, C: TCategory, R: SubsetOf<MatrixN<N, DimNameSum<D, U1>> >;
//          self: Transform<N, D, C>, rhs: Similarity<N, D, R>;
//          [val] => *self *= rhs.inverse();
//          [ref] => *self *= rhs.inverse();
//      );
//
//
//      // Transform ÷= Isometry
//      md_assign_impl_all!(
//          DivAssign, div_assign;
//          (DimNameSum<D, U1>, DimNameSum<D, U1>), (D, U1)
//          for D: DimNameAdd<U1>, C: TCategory, R: SubsetOf<MatrixN<N, DimNameSum<D, U1>> >;
//          self: Transform<N, D, C>, rhs: Isometry<N, D, R>;
//          [val] => *self *= rhs.inverse();
//          [ref] => *self *= rhs.inverse();
//      );

// Transform ÷= Translation
md_assign_impl_all!(
    DivAssign, div_assign where N: RealField;
    (DimNameSum<D, U1>, DimNameSum<D, U1>), (D, U1) for D: DimNameAdd<U1>, C: TCategory;
    self: Transform<N, D, C>, rhs: Translation<N, D>;
    [val] => *self *= rhs.inverse();
    [ref] => *self *= rhs.inverse();
);

// Transform ÷= Rotation
md_assign_impl_all!(
    DivAssign, div_assign where N: RealField;
    (DimNameSum<D, U1>, DimNameSum<D, U1>), (D, D) for D: DimNameAdd<U1>, C: TCategory;
    self: Transform<N, D, C>, rhs: Rotation<N, D>;
    [val] => *self *= rhs.inverse();
    [ref] => *self *= rhs.inverse();
);

// Transform ÷= UnitQuaternion
md_assign_impl_all!(
    DivAssign, div_assign where N: RealField;
    (U4, U4), (U4, U1) for C: TCategory;
    self: Transform<N, U3, C>, rhs: UnitQuaternion<N>;
    [val] => *self *= rhs.inverse();
    [ref] => *self *= rhs.inverse();
);
