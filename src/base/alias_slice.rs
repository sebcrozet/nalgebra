use crate::{Dynamic, U1, U2, U3, U4, U5, U6};
use crate::base::matrix_slice::{SliceStorage, SliceStorageMut};
use crate::base::Matrix;

/*
 *
 *
 * Matrix slice aliases.
 *
 *
 */
/// A column-major matrix slice with `R` rows and `C` columns.
pub type MatrixSliceMN<'a, N, R, C, RStride = U1, CStride = R> =
    Matrix<N, R, C, SliceStorage<'a, N, R, C, RStride, CStride>>;

/// A column-major matrix slice with `D` rows and columns.
pub type MatrixSliceN<'a, N, D, RStride = U1, CStride = D> =
    MatrixSliceMN<'a, N, D, D, RStride, CStride>;

/// A column-major matrix slice dynamic numbers of rows and columns.
pub type DMatrixSlice<'a, N, RStride = U1, CStride = Dynamic> =
    MatrixSliceN<'a, N, Dynamic, RStride, CStride>;

/// A column-major 1x1 matrix slice.
pub type MatrixSlice1<'a, N, RStride = U1, CStride = U1> =
    MatrixSliceN<'a, N, U1, RStride, CStride>;
/// A column-major 2x2 matrix slice.
pub type MatrixSlice2<'a, N, RStride = U1, CStride = U2> =
    MatrixSliceN<'a, N, U2, RStride, CStride>;
/// A column-major 3x3 matrix slice.
pub type MatrixSlice3<'a, N, RStride = U1, CStride = U3> =
    MatrixSliceN<'a, N, U3, RStride, CStride>;
/// A column-major 4x4 matrix slice.
pub type MatrixSlice4<'a, N, RStride = U1, CStride = U4> =
    MatrixSliceN<'a, N, U4, RStride, CStride>;
/// A column-major 5x5 matrix slice.
pub type MatrixSlice5<'a, N, RStride = U1, CStride = U5> =
    MatrixSliceN<'a, N, U5, RStride, CStride>;
/// A column-major 6x6 matrix slice.
pub type MatrixSlice6<'a, N, RStride = U1, CStride = U6> =
    MatrixSliceN<'a, N, U6, RStride, CStride>;

/// A column-major 1x2 matrix slice.
pub type MatrixSlice1x2<'a, N, RStride = U1, CStride = U1> =
    MatrixSliceMN<'a, N, U1, U2, RStride, CStride>;
/// A column-major 1x3 matrix slice.
pub type MatrixSlice1x3<'a, N, RStride = U1, CStride = U1> =
    MatrixSliceMN<'a, N, U1, U3, RStride, CStride>;
/// A column-major 1x4 matrix slice.
pub type MatrixSlice1x4<'a, N, RStride = U1, CStride = U1> =
    MatrixSliceMN<'a, N, U1, U4, RStride, CStride>;
/// A column-major 1x5 matrix slice.
pub type MatrixSlice1x5<'a, N, RStride = U1, CStride = U1> =
    MatrixSliceMN<'a, N, U1, U5, RStride, CStride>;
/// A column-major 1x6 matrix slice.
pub type MatrixSlice1x6<'a, N, RStride = U1, CStride = U1> =
    MatrixSliceMN<'a, N, U1, U6, RStride, CStride>;

/// A column-major 2x1 matrix slice.
pub type MatrixSlice2x1<'a, N, RStride = U1, CStride = U2> =
    MatrixSliceMN<'a, N, U2, U1, RStride, CStride>;
/// A column-major 2x3 matrix slice.
pub type MatrixSlice2x3<'a, N, RStride = U1, CStride = U2> =
    MatrixSliceMN<'a, N, U2, U3, RStride, CStride>;
/// A column-major 2x4 matrix slice.
pub type MatrixSlice2x4<'a, N, RStride = U1, CStride = U2> =
    MatrixSliceMN<'a, N, U2, U4, RStride, CStride>;
/// A column-major 2x5 matrix slice.
pub type MatrixSlice2x5<'a, N, RStride = U1, CStride = U2> =
    MatrixSliceMN<'a, N, U2, U5, RStride, CStride>;
/// A column-major 2x6 matrix slice.
pub type MatrixSlice2x6<'a, N, RStride = U1, CStride = U2> =
    MatrixSliceMN<'a, N, U2, U6, RStride, CStride>;

/// A column-major 3x1 matrix slice.
pub type MatrixSlice3x1<'a, N, RStride = U1, CStride = U3> =
    MatrixSliceMN<'a, N, U3, U1, RStride, CStride>;
/// A column-major 3x2 matrix slice.
pub type MatrixSlice3x2<'a, N, RStride = U1, CStride = U3> =
    MatrixSliceMN<'a, N, U3, U2, RStride, CStride>;
/// A column-major 3x4 matrix slice.
pub type MatrixSlice3x4<'a, N, RStride = U1, CStride = U3> =
    MatrixSliceMN<'a, N, U3, U4, RStride, CStride>;
/// A column-major 3x5 matrix slice.
pub type MatrixSlice3x5<'a, N, RStride = U1, CStride = U3> =
    MatrixSliceMN<'a, N, U3, U5, RStride, CStride>;
/// A column-major 3x6 matrix slice.
pub type MatrixSlice3x6<'a, N, RStride = U1, CStride = U3> =
    MatrixSliceMN<'a, N, U3, U6, RStride, CStride>;

/// A column-major 4x1 matrix slice.
pub type MatrixSlice4x1<'a, N, RStride = U1, CStride = U4> =
    MatrixSliceMN<'a, N, U4, U1, RStride, CStride>;
/// A column-major 4x2 matrix slice.
pub type MatrixSlice4x2<'a, N, RStride = U1, CStride = U4> =
    MatrixSliceMN<'a, N, U4, U2, RStride, CStride>;
/// A column-major 4x3 matrix slice.
pub type MatrixSlice4x3<'a, N, RStride = U1, CStride = U4> =
    MatrixSliceMN<'a, N, U4, U3, RStride, CStride>;
/// A column-major 4x5 matrix slice.
pub type MatrixSlice4x5<'a, N, RStride = U1, CStride = U4> =
    MatrixSliceMN<'a, N, U4, U5, RStride, CStride>;
/// A column-major 4x6 matrix slice.
pub type MatrixSlice4x6<'a, N, RStride = U1, CStride = U4> =
    MatrixSliceMN<'a, N, U4, U6, RStride, CStride>;

/// A column-major 5x1 matrix slice.
pub type MatrixSlice5x1<'a, N, RStride = U1, CStride = U5> =
    MatrixSliceMN<'a, N, U5, U1, RStride, CStride>;
/// A column-major 5x2 matrix slice.
pub type MatrixSlice5x2<'a, N, RStride = U1, CStride = U5> =
    MatrixSliceMN<'a, N, U5, U2, RStride, CStride>;
/// A column-major 5x3 matrix slice.
pub type MatrixSlice5x3<'a, N, RStride = U1, CStride = U5> =
    MatrixSliceMN<'a, N, U5, U3, RStride, CStride>;
/// A column-major 5x4 matrix slice.
pub type MatrixSlice5x4<'a, N, RStride = U1, CStride = U5> =
    MatrixSliceMN<'a, N, U5, U4, RStride, CStride>;
/// A column-major 5x6 matrix slice.
pub type MatrixSlice5x6<'a, N, RStride = U1, CStride = U5> =
    MatrixSliceMN<'a, N, U5, U6, RStride, CStride>;

/// A column-major 6x1 matrix slice.
pub type MatrixSlice6x1<'a, N, RStride = U1, CStride = U6> =
    MatrixSliceMN<'a, N, U6, U1, RStride, CStride>;
/// A column-major 6x2 matrix slice.
pub type MatrixSlice6x2<'a, N, RStride = U1, CStride = U6> =
    MatrixSliceMN<'a, N, U6, U2, RStride, CStride>;
/// A column-major 6x3 matrix slice.
pub type MatrixSlice6x3<'a, N, RStride = U1, CStride = U6> =
    MatrixSliceMN<'a, N, U6, U3, RStride, CStride>;
/// A column-major 6x4 matrix slice.
pub type MatrixSlice6x4<'a, N, RStride = U1, CStride = U6> =
    MatrixSliceMN<'a, N, U6, U4, RStride, CStride>;
/// A column-major 6x5 matrix slice.
pub type MatrixSlice6x5<'a, N, RStride = U1, CStride = U6> =
    MatrixSliceMN<'a, N, U6, U6, RStride, CStride>;

/// A column-major matrix slice with 1 row and a number of columns chosen at runtime.
pub type MatrixSlice1xX<'a, N, RStride = U1, CStride = U1> =
    MatrixSliceMN<'a, N, U1, Dynamic, RStride, CStride>;
/// A column-major matrix slice with 2 rows and a number of columns chosen at runtime.
pub type MatrixSlice2xX<'a, N, RStride = U1, CStride = U2> =
    MatrixSliceMN<'a, N, U2, Dynamic, RStride, CStride>;
/// A column-major matrix slice with 3 rows and a number of columns chosen at runtime.
pub type MatrixSlice3xX<'a, N, RStride = U1, CStride = U3> =
    MatrixSliceMN<'a, N, U3, Dynamic, RStride, CStride>;
/// A column-major matrix slice with 4 rows and a number of columns chosen at runtime.
pub type MatrixSlice4xX<'a, N, RStride = U1, CStride = U4> =
    MatrixSliceMN<'a, N, U4, Dynamic, RStride, CStride>;
/// A column-major matrix slice with 5 rows and a number of columns chosen at runtime.
pub type MatrixSlice5xX<'a, N, RStride = U1, CStride = U5> =
    MatrixSliceMN<'a, N, U5, Dynamic, RStride, CStride>;
/// A column-major matrix slice with 6 rows and a number of columns chosen at runtime.
pub type MatrixSlice6xX<'a, N, RStride = U1, CStride = U6> =
    MatrixSliceMN<'a, N, U6, Dynamic, RStride, CStride>;

/// A column-major matrix slice with a number of rows chosen at runtime and 1 column.
pub type MatrixSliceXx1<'a, N, RStride = U1, CStride = Dynamic> =
    MatrixSliceMN<'a, N, Dynamic, U1, RStride, CStride>;
/// A column-major matrix slice with a number of rows chosen at runtime and 2 columns.
pub type MatrixSliceXx2<'a, N, RStride = U1, CStride = Dynamic> =
    MatrixSliceMN<'a, N, Dynamic, U2, RStride, CStride>;
/// A column-major matrix slice with a number of rows chosen at runtime and 3 columns.
pub type MatrixSliceXx3<'a, N, RStride = U1, CStride = Dynamic> =
    MatrixSliceMN<'a, N, Dynamic, U3, RStride, CStride>;
/// A column-major matrix slice with a number of rows chosen at runtime and 4 columns.
pub type MatrixSliceXx4<'a, N, RStride = U1, CStride = Dynamic> =
    MatrixSliceMN<'a, N, Dynamic, U4, RStride, CStride>;
/// A column-major matrix slice with a number of rows chosen at runtime and 5 columns.
pub type MatrixSliceXx5<'a, N, RStride = U1, CStride = Dynamic> =
    MatrixSliceMN<'a, N, Dynamic, U5, RStride, CStride>;
/// A column-major matrix slice with a number of rows chosen at runtime and 6 columns.
pub type MatrixSliceXx6<'a, N, RStride = U1, CStride = Dynamic> =
    MatrixSliceMN<'a, N, Dynamic, U6, RStride, CStride>;

/// A column vector slice with `D` rows.
pub type VectorSliceN<'a, N, D, RStride = U1, CStride = D> =
    Matrix<N, D, U1, SliceStorage<'a, N, D, U1, RStride, CStride>>;

/// A column vector slice dynamic numbers of rows and columns.
pub type DVectorSlice<'a, N, RStride = U1, CStride = Dynamic> = VectorSliceN<'a, N, Dynamic, RStride, CStride>;

/// A 1D column vector slice.
pub type VectorSlice1<'a, N, RStride = U1, CStride = U1> = VectorSliceN<'a, N, U1, RStride, CStride>;
/// A 2D column vector slice.
pub type VectorSlice2<'a, N, RStride = U1, CStride = U2> = VectorSliceN<'a, N, U2, RStride, CStride>;
/// A 3D column vector slice.
pub type VectorSlice3<'a, N, RStride = U1, CStride = U3> = VectorSliceN<'a, N, U3, RStride, CStride>;
/// A 4D column vector slice.
pub type VectorSlice4<'a, N, RStride = U1, CStride = U4> = VectorSliceN<'a, N, U4, RStride, CStride>;
/// A 5D column vector slice.
pub type VectorSlice5<'a, N, RStride = U1, CStride = U5> = VectorSliceN<'a, N, U5, RStride, CStride>;
/// A 6D column vector slice.
pub type VectorSlice6<'a, N, RStride = U1, CStride = U6> = VectorSliceN<'a, N, U6, RStride, CStride>;

/*
 *
 *
 * Same thing, but for mutable slices.
 *
 *
 */
/// A column-major mutable matrix slice with `R` rows and `C` columns.
pub type MatrixSliceMutMN<'a, N, R, C, RStride = U1, CStride = R> =
    Matrix<N, R, C, SliceStorageMut<'a, N, R, C, RStride, CStride>>;

/// A column-major mutable matrix slice with `D` rows and columns.
pub type MatrixSliceMutN<'a, N, D, RStride = U1, CStride = D> =
    MatrixSliceMutMN<'a, N, D, D, RStride, CStride>;

/// A column-major mutable matrix slice dynamic numbers of rows and columns.
pub type DMatrixSliceMut<'a, N, RStride = U1, CStride = Dynamic> =
    MatrixSliceMutN<'a, N, Dynamic, RStride, CStride>;

/// A column-major 1x1 mutable matrix slice.
pub type MatrixSliceMut1<'a, N, RStride = U1, CStride = U1> =
    MatrixSliceMutN<'a, N, U1, RStride, CStride>;
/// A column-major 2x2 mutable matrix slice.
pub type MatrixSliceMut2<'a, N, RStride = U1, CStride = U2> =
    MatrixSliceMutN<'a, N, U2, RStride, CStride>;
/// A column-major 3x3 mutable matrix slice.
pub type MatrixSliceMut3<'a, N, RStride = U1, CStride = U3> =
    MatrixSliceMutN<'a, N, U3, RStride, CStride>;
/// A column-major 4x4 mutable matrix slice.
pub type MatrixSliceMut4<'a, N, RStride = U1, CStride = U4> =
    MatrixSliceMutN<'a, N, U4, RStride, CStride>;
/// A column-major 5x5 mutable matrix slice.
pub type MatrixSliceMut5<'a, N, RStride = U1, CStride = U5> =
    MatrixSliceMutN<'a, N, U5, RStride, CStride>;
/// A column-major 6x6 mutable matrix slice.
pub type MatrixSliceMut6<'a, N, RStride = U1, CStride = U6> =
    MatrixSliceMutN<'a, N, U6, RStride, CStride>;

/// A column-major 1x2 mutable matrix slice.
pub type MatrixSliceMut1x2<'a, N, RStride = U1, CStride = U1> =
    MatrixSliceMutMN<'a, N, U1, U2, RStride, CStride>;
/// A column-major 1x3 mutable matrix slice.
pub type MatrixSliceMut1x3<'a, N, RStride = U1, CStride = U1> =
    MatrixSliceMutMN<'a, N, U1, U3, RStride, CStride>;
/// A column-major 1x4 mutable matrix slice.
pub type MatrixSliceMut1x4<'a, N, RStride = U1, CStride = U1> =
    MatrixSliceMutMN<'a, N, U1, U4, RStride, CStride>;
/// A column-major 1x5 mutable matrix slice.
pub type MatrixSliceMut1x5<'a, N, RStride = U1, CStride = U1> =
    MatrixSliceMutMN<'a, N, U1, U5, RStride, CStride>;
/// A column-major 1x6 mutable matrix slice.
pub type MatrixSliceMut1x6<'a, N, RStride = U1, CStride = U1> =
    MatrixSliceMutMN<'a, N, U1, U6, RStride, CStride>;

/// A column-major 2x1 mutable matrix slice.
pub type MatrixSliceMut2x1<'a, N, RStride = U1, CStride = U2> =
    MatrixSliceMutMN<'a, N, U2, U1, RStride, CStride>;
/// A column-major 2x3 mutable matrix slice.
pub type MatrixSliceMut2x3<'a, N, RStride = U1, CStride = U2> =
    MatrixSliceMutMN<'a, N, U2, U3, RStride, CStride>;
/// A column-major 2x4 mutable matrix slice.
pub type MatrixSliceMut2x4<'a, N, RStride = U1, CStride = U2> =
    MatrixSliceMutMN<'a, N, U2, U4, RStride, CStride>;
/// A column-major 2x5 mutable matrix slice.
pub type MatrixSliceMut2x5<'a, N, RStride = U1, CStride = U2> =
    MatrixSliceMutMN<'a, N, U2, U5, RStride, CStride>;
/// A column-major 2x6 mutable matrix slice.
pub type MatrixSliceMut2x6<'a, N, RStride = U1, CStride = U2> =
    MatrixSliceMutMN<'a, N, U2, U6, RStride, CStride>;

/// A column-major 3x1 mutable matrix slice.
pub type MatrixSliceMut3x1<'a, N, RStride = U1, CStride = U3> =
    MatrixSliceMutMN<'a, N, U3, U1, RStride, CStride>;
/// A column-major 3x2 mutable matrix slice.
pub type MatrixSliceMut3x2<'a, N, RStride = U1, CStride = U3> =
    MatrixSliceMutMN<'a, N, U3, U2, RStride, CStride>;
/// A column-major 3x4 mutable matrix slice.
pub type MatrixSliceMut3x4<'a, N, RStride = U1, CStride = U3> =
    MatrixSliceMutMN<'a, N, U3, U4, RStride, CStride>;
/// A column-major 3x5 mutable matrix slice.
pub type MatrixSliceMut3x5<'a, N, RStride = U1, CStride = U3> =
    MatrixSliceMutMN<'a, N, U3, U5, RStride, CStride>;
/// A column-major 3x6 mutable matrix slice.
pub type MatrixSliceMut3x6<'a, N, RStride = U1, CStride = U3> =
    MatrixSliceMutMN<'a, N, U3, U6, RStride, CStride>;

/// A column-major 4x1 mutable matrix slice.
pub type MatrixSliceMut4x1<'a, N, RStride = U1, CStride = U4> =
    MatrixSliceMutMN<'a, N, U4, U1, RStride, CStride>;
/// A column-major 4x2 mutable matrix slice.
pub type MatrixSliceMut4x2<'a, N, RStride = U1, CStride = U4> =
    MatrixSliceMutMN<'a, N, U4, U2, RStride, CStride>;
/// A column-major 4x3 mutable matrix slice.
pub type MatrixSliceMut4x3<'a, N, RStride = U1, CStride = U4> =
    MatrixSliceMutMN<'a, N, U4, U3, RStride, CStride>;
/// A column-major 4x5 mutable matrix slice.
pub type MatrixSliceMut4x5<'a, N, RStride = U1, CStride = U4> =
    MatrixSliceMutMN<'a, N, U4, U5, RStride, CStride>;
/// A column-major 4x6 mutable matrix slice.
pub type MatrixSliceMut4x6<'a, N, RStride = U1, CStride = U4> =
    MatrixSliceMutMN<'a, N, U4, U6, RStride, CStride>;

/// A column-major 5x1 mutable matrix slice.
pub type MatrixSliceMut5x1<'a, N, RStride = U1, CStride = U5> =
    MatrixSliceMutMN<'a, N, U5, U1, RStride, CStride>;
/// A column-major 5x2 mutable matrix slice.
pub type MatrixSliceMut5x2<'a, N, RStride = U1, CStride = U5> =
    MatrixSliceMutMN<'a, N, U5, U2, RStride, CStride>;
/// A column-major 5x3 mutable matrix slice.
pub type MatrixSliceMut5x3<'a, N, RStride = U1, CStride = U5> =
    MatrixSliceMutMN<'a, N, U5, U3, RStride, CStride>;
/// A column-major 5x4 mutable matrix slice.
pub type MatrixSliceMut5x4<'a, N, RStride = U1, CStride = U5> =
    MatrixSliceMutMN<'a, N, U5, U4, RStride, CStride>;
/// A column-major 5x6 mutable matrix slice.
pub type MatrixSliceMut5x6<'a, N, RStride = U1, CStride = U5> =
    MatrixSliceMutMN<'a, N, U5, U6, RStride, CStride>;

/// A column-major 6x1 mutable matrix slice.
pub type MatrixSliceMut6x1<'a, N, RStride = U1, CStride = U6> =
    MatrixSliceMutMN<'a, N, U6, U1, RStride, CStride>;
/// A column-major 6x2 mutable matrix slice.
pub type MatrixSliceMut6x2<'a, N, RStride = U1, CStride = U6> =
    MatrixSliceMutMN<'a, N, U6, U2, RStride, CStride>;
/// A column-major 6x3 mutable matrix slice.
pub type MatrixSliceMut6x3<'a, N, RStride = U1, CStride = U6> =
    MatrixSliceMutMN<'a, N, U6, U3, RStride, CStride>;
/// A column-major 6x4 mutable matrix slice.
pub type MatrixSliceMut6x4<'a, N, RStride = U1, CStride = U6> =
    MatrixSliceMutMN<'a, N, U6, U4, RStride, CStride>;
/// A column-major 6x5 mutable matrix slice.
pub type MatrixSliceMut6x5<'a, N, RStride = U1, CStride = U6> =
    MatrixSliceMutMN<'a, N, U6, U5, RStride, CStride>;

/// A column-major mutable matrix slice with 1 row and a number of columns chosen at runtime.
pub type MatrixSliceMut1xX<'a, N, RStride = U1, CStride = U1> =
    MatrixSliceMutMN<'a, N, U1, Dynamic, RStride, CStride>;
/// A column-major mutable matrix slice with 2 rows and a number of columns chosen at runtime.
pub type MatrixSliceMut2xX<'a, N, RStride = U1, CStride = U2> =
    MatrixSliceMutMN<'a, N, U2, Dynamic, RStride, CStride>;
/// A column-major mutable matrix slice with 3 rows and a number of columns chosen at runtime.
pub type MatrixSliceMut3xX<'a, N, RStride = U1, CStride = U3> =
    MatrixSliceMutMN<'a, N, U3, Dynamic, RStride, CStride>;
/// A column-major mutable matrix slice with 4 rows and a number of columns chosen at runtime.
pub type MatrixSliceMut4xX<'a, N, RStride = U1, CStride = U4> =
    MatrixSliceMutMN<'a, N, U4, Dynamic, RStride, CStride>;
/// A column-major mutable matrix slice with 5 rows and a number of columns chosen at runtime.
pub type MatrixSliceMut5xX<'a, N, RStride = U1, CStride = U5> =
    MatrixSliceMutMN<'a, N, U5, Dynamic, RStride, CStride>;
/// A column-major mutable matrix slice with 6 rows and a number of columns chosen at runtime.
pub type MatrixSliceMut6xX<'a, N, RStride = U1, CStride = U6> =
    MatrixSliceMutMN<'a, N, U6, Dynamic, RStride, CStride>;

/// A column-major mutable matrix slice with a number of rows chosen at runtime and 1 column.
pub type MatrixSliceMutXx1<'a, N, RStride = U1, CStride = Dynamic> =
    MatrixSliceMutMN<'a, N, Dynamic, U1, RStride, CStride>;
/// A column-major mutable matrix slice with a number of rows chosen at runtime and 2 columns.
pub type MatrixSliceMutXx2<'a, N, RStride = U1, CStride = Dynamic> =
    MatrixSliceMutMN<'a, N, Dynamic, U2, RStride, CStride>;
/// A column-major mutable matrix slice with a number of rows chosen at runtime and 3 columns.
pub type MatrixSliceMutXx3<'a, N, RStride = U1, CStride = Dynamic> =
    MatrixSliceMutMN<'a, N, Dynamic, U3, RStride, CStride>;
/// A column-major mutable matrix slice with a number of rows chosen at runtime and 4 columns.
pub type MatrixSliceMutXx4<'a, N, RStride = U1, CStride = Dynamic> =
    MatrixSliceMutMN<'a, N, Dynamic, U4, RStride, CStride>;
/// A column-major mutable matrix slice with a number of rows chosen at runtime and 5 columns.
pub type MatrixSliceMutXx5<'a, N, RStride = U1, CStride = Dynamic> =
    MatrixSliceMutMN<'a, N, Dynamic, U5, RStride, CStride>;
/// A column-major mutable matrix slice with a number of rows chosen at runtime and 6 columns.
pub type MatrixSliceMutXx6<'a, N, RStride = U1, CStride = Dynamic> =
    MatrixSliceMutMN<'a, N, Dynamic, U6, RStride, CStride>;

/// A mutable column vector slice with `D` rows.
pub type VectorSliceMutN<'a, N, D, RStride = U1, CStride = D> =
    Matrix<N, D, U1, SliceStorageMut<'a, N, D, U1, RStride, CStride>>;

/// A mutable column vector slice dynamic numbers of rows and columns.
pub type DVectorSliceMut<'a, N, RStride = U1, CStride = Dynamic> = VectorSliceMutN<'a, N, Dynamic, RStride, CStride>;

/// A 1D mutable column vector slice.
pub type VectorSliceMut1<'a, N, RStride = U1, CStride = U1> = VectorSliceMutN<'a, N, U1, RStride, CStride>;
/// A 2D mutable column vector slice.
pub type VectorSliceMut2<'a, N, RStride = U1, CStride = U2> = VectorSliceMutN<'a, N, U2, RStride, CStride>;
/// A 3D mutable column vector slice.
pub type VectorSliceMut3<'a, N, RStride = U1, CStride = U3> = VectorSliceMutN<'a, N, U3, RStride, CStride>;
/// A 4D mutable column vector slice.
pub type VectorSliceMut4<'a, N, RStride = U1, CStride = U4> = VectorSliceMutN<'a, N, U4, RStride, CStride>;
/// A 5D mutable column vector slice.
pub type VectorSliceMut5<'a, N, RStride = U1, CStride = U5> = VectorSliceMutN<'a, N, U5, RStride, CStride>;
/// A 6D mutable column vector slice.
pub type VectorSliceMut6<'a, N, RStride = U1, CStride = U6> = VectorSliceMutN<'a, N, U6, RStride, CStride>;
