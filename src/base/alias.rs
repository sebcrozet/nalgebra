#[cfg(any(feature = "alloc", feature = "std"))]
use crate::Dynamic;
use crate::{U1, U2, U3, U4, U5, U6};
#[cfg(any(feature = "std", feature = "alloc"))]
use crate::base::vec_storage::VecStorage;
use crate::base::storage::Owned;
use crate::base::Matrix;

/*
 *
 *
 * Column-major matrices.
 *
 *
 */
/// A statically sized column-major matrix with `R` rows and `C` columns.
#[deprecated(note = "This matrix name contains a typo. Use MatrixMN instead.")]
pub type MatrixNM<N, R, C> = Matrix<N, R, C, Owned<N, R, C>>;

/// A statically sized column-major matrix with `R` rows and `C` columns.
pub type MatrixMN<N, R, C> = Matrix<N, R, C, Owned<N, R, C>>;

/// A statically sized column-major square matrix with `D` rows and columns.
pub type MatrixN<N, D> = MatrixMN<N, D, D>;

/// A dynamically sized column-major matrix.
#[cfg(any(feature = "std", feature = "alloc"))]
pub type DMatrix<N> = MatrixN<N, Dynamic>;

/// A stack-allocated, column-major, 1x1 square matrix.
pub type Matrix1<N> = MatrixN<N, U1>;
/// A stack-allocated, column-major, 2x2 square matrix.
pub type Matrix2<N> = MatrixN<N, U2>;
/// A stack-allocated, column-major, 3x3 square matrix.
pub type Matrix3<N> = MatrixN<N, U3>;
/// A stack-allocated, column-major, 4x4 square matrix.
pub type Matrix4<N> = MatrixN<N, U4>;
/// A stack-allocated, column-major, 5x5 square matrix.
pub type Matrix5<N> = MatrixN<N, U5>;
/// A stack-allocated, column-major, 6x6 square matrix.
pub type Matrix6<N> = MatrixN<N, U6>;

/// A stack-allocated, column-major, 1x2 matrix.
pub type Matrix1x2<N> = MatrixMN<N, U1, U2>;
/// A stack-allocated, column-major, 1x3 matrix.
pub type Matrix1x3<N> = MatrixMN<N, U1, U3>;
/// A stack-allocated, column-major, 1x4 matrix.
pub type Matrix1x4<N> = MatrixMN<N, U1, U4>;
/// A stack-allocated, column-major, 1x5 matrix.
pub type Matrix1x5<N> = MatrixMN<N, U1, U5>;
/// A stack-allocated, column-major, 1x6 matrix.
pub type Matrix1x6<N> = MatrixMN<N, U1, U6>;

/// A stack-allocated, column-major, 2x3 matrix.
pub type Matrix2x3<N> = MatrixMN<N, U2, U3>;
/// A stack-allocated, column-major, 2x4 matrix.
pub type Matrix2x4<N> = MatrixMN<N, U2, U4>;
/// A stack-allocated, column-major, 2x5 matrix.
pub type Matrix2x5<N> = MatrixMN<N, U2, U5>;
/// A stack-allocated, column-major, 2x6 matrix.
pub type Matrix2x6<N> = MatrixMN<N, U2, U6>;

/// A stack-allocated, column-major, 3x4 matrix.
pub type Matrix3x4<N> = MatrixMN<N, U3, U4>;
/// A stack-allocated, column-major, 3x5 matrix.
pub type Matrix3x5<N> = MatrixMN<N, U3, U5>;
/// A stack-allocated, column-major, 3x6 matrix.
pub type Matrix3x6<N> = MatrixMN<N, U3, U6>;

/// A stack-allocated, column-major, 4x5 matrix.
pub type Matrix4x5<N> = MatrixMN<N, U4, U5>;
/// A stack-allocated, column-major, 4x6 matrix.
pub type Matrix4x6<N> = MatrixMN<N, U4, U6>;

/// A stack-allocated, column-major, 5x6 matrix.
pub type Matrix5x6<N> = MatrixMN<N, U5, U6>;

/// A stack-allocated, column-major, 2x1 matrix.
pub type Matrix2x1<N> = MatrixMN<N, U2, U1>;
/// A stack-allocated, column-major, 3x1 matrix.
pub type Matrix3x1<N> = MatrixMN<N, U3, U1>;
/// A stack-allocated, column-major, 4x1 matrix.
pub type Matrix4x1<N> = MatrixMN<N, U4, U1>;
/// A stack-allocated, column-major, 5x1 matrix.
pub type Matrix5x1<N> = MatrixMN<N, U5, U1>;
/// A stack-allocated, column-major, 6x1 matrix.
pub type Matrix6x1<N> = MatrixMN<N, U6, U1>;

/// A stack-allocated, column-major, 3x2 matrix.
pub type Matrix3x2<N> = MatrixMN<N, U3, U2>;
/// A stack-allocated, column-major, 4x2 matrix.
pub type Matrix4x2<N> = MatrixMN<N, U4, U2>;
/// A stack-allocated, column-major, 5x2 matrix.
pub type Matrix5x2<N> = MatrixMN<N, U5, U2>;
/// A stack-allocated, column-major, 6x2 matrix.
pub type Matrix6x2<N> = MatrixMN<N, U6, U2>;

/// A stack-allocated, column-major, 4x3 matrix.
pub type Matrix4x3<N> = MatrixMN<N, U4, U3>;
/// A stack-allocated, column-major, 5x3 matrix.
pub type Matrix5x3<N> = MatrixMN<N, U5, U3>;
/// A stack-allocated, column-major, 6x3 matrix.
pub type Matrix6x3<N> = MatrixMN<N, U6, U3>;

/// A stack-allocated, column-major, 5x4 matrix.
pub type Matrix5x4<N> = MatrixMN<N, U5, U4>;
/// A stack-allocated, column-major, 6x4 matrix.
pub type Matrix6x4<N> = MatrixMN<N, U6, U4>;

/// A stack-allocated, column-major, 6x5 matrix.
pub type Matrix6x5<N> = MatrixMN<N, U6, U5>;

/*
 *
 *
 * Column vectors.
 *
 *
 */
/// A dynamically sized column vector.
#[cfg(any(feature = "std", feature = "alloc"))]
pub type DVector<N> = Matrix<N, Dynamic, U1, VecStorage<N, Dynamic, U1>>;

/// A statically sized D-dimensional column vector.
pub type VectorN<N, D> = MatrixMN<N, D, U1>;

/// A stack-allocated, 1-dimensional column vector.
pub type Vector1<N> = VectorN<N, U1>;
/// A stack-allocated, 2-dimensional column vector.
pub type Vector2<N> = VectorN<N, U2>;
/// A stack-allocated, 3-dimensional column vector.
pub type Vector3<N> = VectorN<N, U3>;
/// A stack-allocated, 4-dimensional column vector.
pub type Vector4<N> = VectorN<N, U4>;
/// A stack-allocated, 5-dimensional column vector.
pub type Vector5<N> = VectorN<N, U5>;
/// A stack-allocated, 6-dimensional column vector.
pub type Vector6<N> = VectorN<N, U6>;

/*
 *
 *
 * Row vectors.
 *
 *
 */
/// A dynamically sized row vector.
#[cfg(any(feature = "std", feature = "alloc"))]
pub type RowDVector<N> = Matrix<N, U1, Dynamic, VecStorage<N, U1, Dynamic>>;

/// A statically sized D-dimensional row vector.
pub type RowVectorN<N, D> = MatrixMN<N, U1, D>;

/// A stack-allocated, 1-dimensional row vector.
pub type RowVector1<N> = RowVectorN<N, U1>;
/// A stack-allocated, 2-dimensional row vector.
pub type RowVector2<N> = RowVectorN<N, U2>;
/// A stack-allocated, 3-dimensional row vector.
pub type RowVector3<N> = RowVectorN<N, U3>;
/// A stack-allocated, 4-dimensional row vector.
pub type RowVector4<N> = RowVectorN<N, U4>;
/// A stack-allocated, 5-dimensional row vector.
pub type RowVector5<N> = RowVectorN<N, U5>;
/// A stack-allocated, 6-dimensional row vector.
pub type RowVector6<N> = RowVectorN<N, U6>;
