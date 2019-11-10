use alga::general::ComplexField;
use crate::base::allocator::Allocator;
use crate::base::constraint::{AreMultipliable, DimEq, SameNumberOfRows, ShapeConstraint};
use crate::base::{DefaultAllocator, Matrix, Scalar, Unit, Vector};
use crate::{Dim, DimName, U1};
use crate::storage::{Storage, StorageMut};

use crate::geometry::Point;

/// A reflection wrt. a plane.
pub struct Reflection<N: Scalar, D: Dim, S: Storage<N, D>> {
    axis: Vector<N, D, S>,
    bias: N,
}

impl<N: ComplexField, D: Dim, S: Storage<N, D>> Reflection<N, D, S> {
    /// Creates a new reflection wrt the plane orthogonal to the given axis and bias.
    ///
    /// The bias is the position of the plane on the axis. In particular, a bias equal to zero
    /// represents a plane that passes through the origin.
    pub fn new(axis: Unit<Vector<N, D, S>>, bias: N) -> Self {
        Self {
            axis: axis.into_inner(),
            bias,
        }
    }

    /// Creates a new reflection wrt. the plane orthogonal to the given axis and that contains the
    /// point `pt`.
    pub fn new_containing_point(
        axis: Unit<Vector<N, D, S>>,
        pt: &Point<N, D>,
    ) -> Self
    where
        D: DimName,
        DefaultAllocator: Allocator<N, D>,
    {
        let bias = axis.dotc(&pt.coords);
        Self::new(axis, bias)
    }

    /// The reflexion axis.
    pub fn axis(&self) -> &Vector<N, D, S> {
        &self.axis
    }

    // FIXME: naming convention: reflect_to, reflect_assign ?
    /// Applies the reflection to the columns of `rhs`.
    pub fn reflect<R2: Dim, C2: Dim, S2>(&self, rhs: &mut Matrix<N, R2, C2, S2>)
    where
        S2: StorageMut<N, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        for i in 0..rhs.ncols() {
            // NOTE: we borrow the column twice here. First it is borrowed immutably for the
            // dot product, and then mutably. Somehow, this allows significantly
            // better optimizations of the dot product from the compiler.
            let m_two: N = crate::convert(-2.0f64);
            let factor = (self.axis.dotc(&rhs.column(i)) - self.bias) * m_two;
            rhs.column_mut(i).axpy(factor, &self.axis, N::one());
        }
    }

    // FIXME: naming convention: reflect_to, reflect_assign ?
    /// Applies the reflection to the columns of `rhs`.
    pub fn reflect_with_sign<R2: Dim, C2: Dim, S2>(&self, rhs: &mut Matrix<N, R2, C2, S2>, sign: N)
        where
            S2: StorageMut<N, R2, C2>,
            ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        for i in 0..rhs.ncols() {
            // NOTE: we borrow the column twice here. First it is borrowed immutably for the
            // dot product, and then mutably. Somehow, this allows significantly
            // better optimizations of the dot product from the compiler.
            let m_two = sign.scale(crate::convert(-2.0f64));
            let factor = (self.axis.dotc(&rhs.column(i)) - self.bias) * m_two;
            rhs.column_mut(i).axpy(factor, &self.axis, sign);
        }
    }

    /// Applies the reflection to the rows of `lhs`.
    pub fn reflect_rows<R2: Dim, C2: Dim, S2, S3>(
        &self,
        lhs: &mut Matrix<N, R2, C2, S2>,
        work: &mut Vector<N, R2, S3>,
    ) where
        S2: StorageMut<N, R2, C2>,
        S3: StorageMut<N, R2>,
        ShapeConstraint: DimEq<C2, D> + AreMultipliable<R2, C2, D, U1>,
    {
        lhs.mul_to(&self.axis, work);

        if !self.bias.is_zero() {
            work.add_scalar_mut(-self.bias);
        }

        let m_two: N = crate::convert(-2.0f64);
        lhs.gerc(m_two, &work, &self.axis, N::one());
    }

    /// Applies the reflection to the rows of `lhs`.
    pub fn reflect_rows_with_sign<R2: Dim, C2: Dim, S2, S3>(
        &self,
        lhs: &mut Matrix<N, R2, C2, S2>,
        work: &mut Vector<N, R2, S3>,
        sign: N,
    ) where
        S2: StorageMut<N, R2, C2>,
        S3: StorageMut<N, R2>,
        ShapeConstraint: DimEq<C2, D> + AreMultipliable<R2, C2, D, U1>,
    {
        lhs.mul_to(&self.axis, work);

        if !self.bias.is_zero() {
            work.add_scalar_mut(-self.bias);
        }

        let m_two = sign.scale(crate::convert(-2.0f64));
        lhs.gerc(m_two, &work, &self.axis, sign);
    }
}
