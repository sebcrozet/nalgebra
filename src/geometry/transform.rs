use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use std::any::Any;
use std::fmt::Debug;
use std::marker::PhantomData;

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use alga::general::{RealField, TwoSidedInverse};

use crate::base::allocator::Allocator;
use crate::base::dimension::{DimName, DimNameAdd, DimNameSum, U1};
use crate::base::storage::Owned;
use crate::base::{DefaultAllocator, MatrixN, VectorN};

use crate::geometry::Point;

/// Trait implemented by phantom types identifying the projective transformation type.
///
/// NOTE: this trait is not intended to be implemented outside of the `nalgebra` crate.
pub trait TCategory: Any + Debug + Copy + PartialEq + Send {
    /// Indicates whether a `Transform` with the category `Self` has a bottom-row different from
    /// `0 0 .. 1`.
    #[inline]
    fn has_normalizer() -> bool {
        true
    }

    /// Checks that the given matrix is a valid homogeneous representation of an element of the
    /// category `Self`.
    fn check_homogeneous_invariants<N: RealField, D: DimName>(mat: &MatrixN<N, D>) -> bool
    where
        N::Epsilon: Copy,
        DefaultAllocator: Allocator<N, D, D>;
}

/// Traits that gives the `Transform` category that is compatible with the result of the
/// multiplication of transformations with categories `Self` and `Other`.
pub trait TCategoryMul<Other: TCategory>: TCategory {
    /// The transform category that results from the multiplication of a `Transform<Self>` to a
    /// `Transform<Other>`. This is usually equal to `Self` or `Other`, whichever is the most
    /// general category.
    type Representative: TCategory;
}

/// Indicates that `Self` is a more general `Transform` category than `Other`.
pub trait SuperTCategoryOf<Other: TCategory>: TCategory {}

/// Indicates that `Self` is a more specific `Transform` category than `Other`.
///
/// Automatically implemented based on `SuperTCategoryOf`.
pub trait SubTCategoryOf<Other: TCategory>: TCategory {}
impl<T1, T2> SubTCategoryOf<T2> for T1
where
    T1: TCategory,
    T2: SuperTCategoryOf<T1>,
{
}

/// Tag representing the most general (not necessarily inversible) `Transform` type.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum TGeneral {}

/// Tag representing the most general inversible `Transform` type.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum TProjective {}

/// Tag representing an affine `Transform`. Its bottom-row is equal to `(0, 0 ... 0, 1)`.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum TAffine {}

impl TCategory for TGeneral {
    #[inline]
    fn check_homogeneous_invariants<N: RealField, D: DimName>(_: &MatrixN<N, D>) -> bool
    where
        N::Epsilon: Copy,
        DefaultAllocator: Allocator<N, D, D>,
    {
        true
    }
}

impl TCategory for TProjective {
    #[inline]
    fn check_homogeneous_invariants<N: RealField, D: DimName>(mat: &MatrixN<N, D>) -> bool
    where
        N::Epsilon: Copy,
        DefaultAllocator: Allocator<N, D, D>,
    {
        mat.is_invertible()
    }
}

impl TCategory for TAffine {
    #[inline]
    fn has_normalizer() -> bool {
        false
    }

    #[inline]
    fn check_homogeneous_invariants<N: RealField, D: DimName>(mat: &MatrixN<N, D>) -> bool
    where
        N::Epsilon: Copy,
        DefaultAllocator: Allocator<N, D, D>,
    {
        let last = D::dim() - 1;
        mat.is_invertible()
            && mat[(last, last)] == N::one()
            && (0..last).all(|i| mat[(last, i)].is_zero())
    }
}

macro_rules! category_mul_impl(
    ($($a: ident * $b: ident => $c: ty);* $(;)*) => {$(
        impl TCategoryMul<$a> for $b {
            type Representative = $c;
        }
    )*}
);

// We require stability uppon multiplication.
impl<T: TCategory> TCategoryMul<T> for T {
    type Representative = T;
}

category_mul_impl!(
//  TGeneral * TGeneral    => TGeneral;
    TGeneral * TProjective => TGeneral;
    TGeneral * TAffine     => TGeneral;

    TProjective * TGeneral    => TGeneral;
//  TProjective * TProjective => TProjective;
    TProjective * TAffine     => TProjective;

    TAffine * TGeneral    => TGeneral;
    TAffine * TProjective => TProjective;
//  TAffine * TAffine     => TAffine;
);

macro_rules! super_tcategory_impl(
    ($($a: ident >= $b: ident);* $(;)*) => {$(
        impl SuperTCategoryOf<$b> for $a { }
    )*}
);

impl<T: TCategory> SuperTCategoryOf<T> for T {}

super_tcategory_impl!(
    TGeneral    >= TProjective;
    TGeneral    >= TAffine;
    TProjective >= TAffine;
);

/// A transformation matrix in homogeneous coordinates.
///
/// It is stored as a matrix with dimensions `(D + 1, D + 1)`, e.g., it stores a 4x4 matrix for a
/// 3D transformation.
#[repr(C)]
#[derive(Debug)]
pub struct Transform<N: RealField, D: DimNameAdd<U1>, C: TCategory>
where DefaultAllocator: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>>
{
    matrix: MatrixN<N, DimNameSum<D, U1>>,
    _phantom: PhantomData<C>,
}

// FIXME
// impl<N: RealField + hash::Hash, D: DimNameAdd<U1> + hash::Hash, C: TCategory> hash::Hash for Transform<N, D, C>
//     where DefaultAllocator: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>>,
//           Owned<N, DimNameSum<D, U1>, DimNameSum<D, U1>>: hash::Hash {
//     fn hash<H: hash::Hasher>(&self, state: &mut H) {
//         self.matrix.hash(state);
//     }
// }

impl<N: RealField, D: DimNameAdd<U1> + Copy, C: TCategory> Copy for Transform<N, D, C>
where
    DefaultAllocator: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>>,
    Owned<N, DimNameSum<D, U1>, DimNameSum<D, U1>>: Copy,
{
}

impl<N: RealField, D: DimNameAdd<U1>, C: TCategory> Clone for Transform<N, D, C>
where DefaultAllocator: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>>
{
    #[inline]
    fn clone(&self) -> Self {
        Transform::from_matrix_unchecked(self.matrix.clone())
    }
}

#[cfg(feature = "serde-serialize")]
impl<N: RealField, D: DimNameAdd<U1>, C: TCategory> Serialize for Transform<N, D, C>
where
    DefaultAllocator: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>>,
    Owned<N, DimNameSum<D, U1>, DimNameSum<D, U1>>: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: Serializer {
        self.matrix.serialize(serializer)
    }
}

#[cfg(feature = "serde-serialize")]
impl<'a, N: RealField, D: DimNameAdd<U1>, C: TCategory> Deserialize<'a> for Transform<N, D, C>
where
    DefaultAllocator: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>>,
    Owned<N, DimNameSum<D, U1>, DimNameSum<D, U1>>: Deserialize<'a>,
{
    fn deserialize<Des>(deserializer: Des) -> Result<Self, Des::Error>
    where Des: Deserializer<'a> {
        let matrix = MatrixN::<N, DimNameSum<D, U1>>::deserialize(deserializer)?;

        Ok(Transform::from_matrix_unchecked(matrix))
    }
}

impl<N: RealField + Eq, D: DimNameAdd<U1>, C: TCategory> Eq for Transform<N, D, C> where DefaultAllocator: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>>
{}

impl<N: RealField, D: DimNameAdd<U1>, C: TCategory> PartialEq for Transform<N, D, C>
where DefaultAllocator: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>>
{
    #[inline]
    fn eq(&self, right: &Self) -> bool {
        self.matrix == right.matrix
    }
}

impl<N: RealField, D: DimNameAdd<U1>, C: TCategory> Transform<N, D, C>
where DefaultAllocator: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>>
{
    /// Creates a new transformation from the given homogeneous matrix. The transformation category
    /// of `Self` is not checked to be verified by the given matrix.
    #[inline]
    pub fn from_matrix_unchecked(matrix: MatrixN<N, DimNameSum<D, U1>>) -> Self {
        Transform {
            matrix: matrix,
            _phantom: PhantomData,
        }
    }

    /// Retrieves the underlying matrix.
    ///
    /// # Examples
    /// ```
    /// # use nalgebra::{Matrix3, Transform2};
    ///
    /// let m = Matrix3::new(1.0, 2.0, 0.0,
    ///                      3.0, 4.0, 0.0,
    ///                      0.0, 0.0, 1.0);
    /// let t = Transform2::from_matrix_unchecked(m);
    /// assert_eq!(t.into_inner(), m);
    /// ```
    #[inline]
    pub fn into_inner(self) -> MatrixN<N, DimNameSum<D, U1>> {
        self.matrix
    }

    /// Retrieves the underlying matrix.
    /// Deprecated: Use [Transform::into_inner] instead.
    #[deprecated(note="use `.into_inner()` instead")]
    #[inline]
    pub fn unwrap(self) -> MatrixN<N, DimNameSum<D, U1>> {
        self.matrix
    }

    /// A reference to the underlying matrix.
    ///
    /// # Examples
    /// ```
    /// # use nalgebra::{Matrix3, Transform2};
    ///
    /// let m = Matrix3::new(1.0, 2.0, 0.0,
    ///                      3.0, 4.0, 0.0,
    ///                      0.0, 0.0, 1.0);
    /// let t = Transform2::from_matrix_unchecked(m);
    /// assert_eq!(*t.matrix(), m);
    /// ```
    #[inline]
    pub fn matrix(&self) -> &MatrixN<N, DimNameSum<D, U1>> {
        &self.matrix
    }

    /// A mutable reference to the underlying matrix.
    ///
    /// It is `_unchecked` because direct modifications of this matrix may break invariants
    /// identified by this transformation category.
    ///
    /// # Examples
    /// ```
    /// # use nalgebra::{Matrix3, Transform2};
    ///
    /// let m = Matrix3::new(1.0, 2.0, 0.0,
    ///                      3.0, 4.0, 0.0,
    ///                      0.0, 0.0, 1.0);
    /// let mut t = Transform2::from_matrix_unchecked(m);
    /// t.matrix_mut_unchecked().m12 = 42.0;
    /// t.matrix_mut_unchecked().m23 = 90.0;
    ///
    ///
    /// let expected = Matrix3::new(1.0, 42.0, 0.0,
    ///                             3.0, 4.0,  90.0,
    ///                             0.0, 0.0,  1.0);
    /// assert_eq!(*t.matrix(), expected);
    /// ```
    #[inline]
    pub fn matrix_mut_unchecked(&mut self) -> &mut MatrixN<N, DimNameSum<D, U1>> {
        &mut self.matrix
    }

    /// Sets the category of this transform.
    ///
    /// This can be done only if the new category is more general than the current one, e.g., a
    /// transform with category `TProjective` cannot be converted to a transform with category
    /// `TAffine` because not all projective transformations are affine (the other way-round is
    /// valid though).
    #[inline]
    pub fn set_category<CNew: SuperTCategoryOf<C>>(self) -> Transform<N, D, CNew> {
        Transform::from_matrix_unchecked(self.matrix)
    }

    /// Clones this transform into one that owns its data.
    #[inline]
    #[deprecated(
        note = "This method is redundant with automatic `Copy` and the `.clone()` method and will be removed in a future release."
    )]
    pub fn clone_owned(&self) -> Transform<N, D, C> {
        Transform::from_matrix_unchecked(self.matrix.clone_owned())
    }

    /// Converts this transform into its equivalent homogeneous transformation matrix.
    ///
    /// # Examples
    /// ```
    /// # use nalgebra::{Matrix3, Transform2};
    ///
    /// let m = Matrix3::new(1.0, 2.0, 0.0,
    ///                      3.0, 4.0, 0.0,
    ///                      0.0, 0.0, 1.0);
    /// let t = Transform2::from_matrix_unchecked(m);
    /// assert_eq!(t.into_inner(), m);
    /// ```
    #[inline]
    pub fn to_homogeneous(&self) -> MatrixN<N, DimNameSum<D, U1>> {
        self.matrix().clone_owned()
    }

    /// Attempts to invert this transformation. You may use `.inverse` instead of this
    /// transformation has a subcategory of `TProjective` (i.e. if it is a `Projective{2,3}` or `Affine{2,3}`).
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Matrix3, Transform2};
    ///
    /// let m = Matrix3::new(2.0, 2.0, -0.3,
    ///                      3.0, 4.0, 0.1,
    ///                      0.0, 0.0, 1.0);
    /// let t = Transform2::from_matrix_unchecked(m);
    /// let inv_t = t.try_inverse().unwrap();
    /// assert_relative_eq!(t * inv_t, Transform2::identity());
    /// assert_relative_eq!(inv_t * t, Transform2::identity());
    ///
    /// // Non-invertible case.
    /// let m = Matrix3::new(0.0, 2.0, 1.0,
    ///                      3.0, 0.0, 5.0,
    ///                      0.0, 0.0, 0.0);
    /// let t = Transform2::from_matrix_unchecked(m);
    /// assert!(t.try_inverse().is_none());
    /// ```
    #[inline]
    #[must_use = "Did you mean to use try_inverse_mut()?"]
    pub fn try_inverse(self) -> Option<Transform<N, D, C>> {
        if let Some(m) = self.matrix.try_inverse() {
            Some(Transform::from_matrix_unchecked(m))
        } else {
            None
        }
    }

    /// Inverts this transformation. Use `.try_inverse` if this transform has the `TGeneral`
    /// category (i.e., a `Transform{2,3}` may not be invertible).
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Matrix3, Projective2};
    ///
    /// let m = Matrix3::new(2.0, 2.0, -0.3,
    ///                      3.0, 4.0, 0.1,
    ///                      0.0, 0.0, 1.0);
    /// let proj = Projective2::from_matrix_unchecked(m);
    /// let inv_t = proj.inverse();
    /// assert_relative_eq!(proj * inv_t, Projective2::identity());
    /// assert_relative_eq!(inv_t * proj, Projective2::identity());
    /// ```
    #[inline]
    #[must_use = "Did you mean to use inverse_mut()?"]
    pub fn inverse(self) -> Transform<N, D, C>
    where C: SubTCategoryOf<TProjective> {
        // FIXME: specialize for TAffine?
        Transform::from_matrix_unchecked(self.matrix.try_inverse().unwrap())
    }

    /// Attempts to invert this transformation in-place. You may use `.inverse_mut` instead of this
    /// transformation has a subcategory of `TProjective`.
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Matrix3, Transform2};
    ///
    /// let m = Matrix3::new(2.0, 2.0, -0.3,
    ///                      3.0, 4.0, 0.1,
    ///                      0.0, 0.0, 1.0);
    /// let t = Transform2::from_matrix_unchecked(m);
    /// let mut inv_t = t;
    /// assert!(inv_t.try_inverse_mut());
    /// assert_relative_eq!(t * inv_t, Transform2::identity());
    /// assert_relative_eq!(inv_t * t, Transform2::identity());
    ///
    /// // Non-invertible case.
    /// let m = Matrix3::new(0.0, 2.0, 1.0,
    ///                      3.0, 0.0, 5.0,
    ///                      0.0, 0.0, 0.0);
    /// let mut t = Transform2::from_matrix_unchecked(m);
    /// assert!(!t.try_inverse_mut());
    /// ```
    #[inline]
    pub fn try_inverse_mut(&mut self) -> bool {
        self.matrix.try_inverse_mut()
    }

    /// Inverts this transformation in-place. Use `.try_inverse_mut` if this transform has the
    /// `TGeneral` category (it may not be invertible).
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Matrix3, Projective2};
    ///
    /// let m = Matrix3::new(2.0, 2.0, -0.3,
    ///                      3.0, 4.0, 0.1,
    ///                      0.0, 0.0, 1.0);
    /// let proj = Projective2::from_matrix_unchecked(m);
    /// let mut inv_t = proj;
    /// inv_t.inverse_mut();
    /// assert_relative_eq!(proj * inv_t, Projective2::identity());
    /// assert_relative_eq!(inv_t * proj, Projective2::identity());
    /// ```
    #[inline]
    pub fn inverse_mut(&mut self)
    where C: SubTCategoryOf<TProjective> {
        let _ = self.matrix.try_inverse_mut();
    }
}

impl<N, D: DimNameAdd<U1>, C> Transform<N, D, C>
where
    N: RealField,
    C: TCategory,
    DefaultAllocator: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>>
        + Allocator<N, DimNameSum<D, U1>>
        + Allocator<N, D, D>
        + Allocator<N, D>,
{
    /// Transform the given point by this transformation.
    ///
    /// This is the same as the multiplication `self * pt`.
    #[inline]
    pub fn transform_point(&self, pt: &Point<N, D>) -> Point<N, D> {
        self * pt
    }

    /// Transform the given vector by this transformation, ignoring the
    /// translational component of the transformation.
    ///
    /// This is the same as the multiplication `self * v`.
    #[inline]
    pub fn transform_vector(&self, v: &VectorN<N, D>) -> VectorN<N, D> {
        self * v
    }
}

impl<N: RealField, D: DimNameAdd<U1>, C: TCategory> Transform<N, D, C>
where C: SubTCategoryOf<TProjective>,
      DefaultAllocator: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>>
        + Allocator<N, DimNameSum<D, U1>>
        + Allocator<N, D, D>
        + Allocator<N, D>,
{
    /// Transform the given point by the inverse of this transformation.
    /// This may be cheaper than inverting the transformation and transforming
    /// the point.
    #[inline]
    pub fn inverse_transform_point(&self, pt: &Point<N, D>) -> Point<N, D> {
        self.two_sided_inverse() * pt
    }

    /// Transform the given vector by the inverse of this transformation.
    /// This may be cheaper than inverting the transformation and transforming
    /// the vector.
    #[inline]
    pub fn inverse_transform_vector(&self, v: &VectorN<N, D>) -> VectorN<N, D> {
        self.two_sided_inverse() * v
    }
}

impl<N: RealField, D: DimNameAdd<U1>> Transform<N, D, TGeneral>
where DefaultAllocator: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>>
{
    /// A mutable reference to underlying matrix. Use `.matrix_mut_unchecked` instead if this
    /// transformation category is not `TGeneral`.
    #[inline]
    pub fn matrix_mut(&mut self) -> &mut MatrixN<N, DimNameSum<D, U1>> {
        self.matrix_mut_unchecked()
    }
}

impl<N: RealField, D: DimNameAdd<U1>, C: TCategory> AbsDiffEq for Transform<N, D, C>
where
    N::Epsilon: Copy,
    DefaultAllocator: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>>,
{
    type Epsilon = N::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        N::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.matrix.abs_diff_eq(&other.matrix, epsilon)
    }
}

impl<N: RealField, D: DimNameAdd<U1>, C: TCategory> RelativeEq for Transform<N, D, C>
where
    N::Epsilon: Copy,
    DefaultAllocator: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>>,
{
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        N::default_max_relative()
    }

    #[inline]
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool
    {
        self.matrix
            .relative_eq(&other.matrix, epsilon, max_relative)
    }
}

impl<N: RealField, D: DimNameAdd<U1>, C: TCategory> UlpsEq for Transform<N, D, C>
where
    N::Epsilon: Copy,
    DefaultAllocator: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>>,
{
    #[inline]
    fn default_max_ulps() -> u32 {
        N::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.matrix.ulps_eq(&other.matrix, epsilon, max_ulps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::Matrix4;

    #[test]
    fn checks_homogeneous_invariants_of_square_identity_matrix() {
        assert!(TAffine::check_homogeneous_invariants(
            &Matrix4::<f32>::identity()
        ));
    }
}
