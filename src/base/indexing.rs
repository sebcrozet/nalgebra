//! Indexing

use crate::{Dim, DimName, DimDiff, DimSub, Dynamic, Matrix, MatrixSlice, MatrixSliceMut, Scalar, U1};
use crate::base::storage::{Storage, StorageMut};

use std::ops;

// N.B.: Not a public trait!
trait DimRange<D: Dim>
{
    /// The number of elements indexed by this range.
    type Length: Dim;

    /// The lower bound of the range, inclusive.
    fn lower(&self, dimension: D) -> usize;

    /// The number of elements included in the range.
    fn length(&self, dimension: D) -> Self::Length;

    /// Produces true if `Self` is contained within `dimension`.
    fn contained_by(&self, dimension: D) -> bool;
}

impl<D: Dim> DimRange<D> for usize {
    type Length = U1;

    #[inline(always)]
    fn lower(&self, _: D) -> usize {
        *self
    }

    #[inline(always)]
    fn length(&self, _: D) -> Self::Length {
        U1
    }

    #[inline(always)]
    fn contained_by(&self, dimension: D) -> bool {
        *self < dimension.value()
    }
}

#[test]
fn dimrange_usize() {
    use crate::U0;
    assert_eq!(DimRange::contained_by(&0, U0), false);
    assert_eq!(DimRange::contained_by(&0, U1), true);
}

impl<D: Dim> DimRange<D> for ops::Range<usize> {
    type Length = Dynamic;

    #[inline(always)]
    fn lower(&self, _: D) -> usize {
        self.start
    }

    #[inline(always)]
    fn length(&self, _: D) -> Self::Length {
        Dynamic::new(self.end.saturating_sub(self.start))
    }

    #[inline(always)]
    fn contained_by(&self, dimension: D) -> bool {
        (self.start < dimension.value()) && (self.end <= dimension.value())
    }
}

#[test]
fn dimrange_range_usize() {
    use std::usize::MAX;
    use crate::U0;
    assert_eq!(DimRange::contained_by(&(0..0), U0), false);
    assert_eq!(DimRange::contained_by(&(0..1), U0), false);
    assert_eq!(DimRange::contained_by(&(0..1), U1), true);
    assert_eq!(DimRange::contained_by(&((MAX - 1)..MAX), Dynamic::new(MAX)), true);
    assert_eq!(DimRange::length(&((MAX - 1)..MAX), Dynamic::new(MAX)), Dynamic::new(1));
    assert_eq!(DimRange::length(&(MAX..(MAX - 1)), Dynamic::new(MAX)), Dynamic::new(0));
    assert_eq!(DimRange::length(&(MAX..MAX), Dynamic::new(MAX)), Dynamic::new(0));
}

impl<D: Dim> DimRange<D> for ops::RangeFrom<usize> {
    type Length = Dynamic;

    #[inline(always)]
    fn lower(&self, _: D) -> usize {
        self.start
    }

    #[inline(always)]
    fn length(&self, dimension: D) -> Self::Length {
        (self.start..dimension.value()).length(dimension)
    }

    #[inline(always)]
    fn contained_by(&self, dimension: D) -> bool {
        self.start < dimension.value()
    }
}

#[test]
fn dimrange_rangefrom_usize() {
    use std::usize::MAX;
    use crate::U0;
    assert_eq!(DimRange::contained_by(&(0..), U0), false);
    assert_eq!(DimRange::contained_by(&(0..), U0), false);
    assert_eq!(DimRange::contained_by(&(0..), U1), true);
    assert_eq!(DimRange::contained_by(&((MAX - 1)..), Dynamic::new(MAX)), true);
    assert_eq!(DimRange::length(&((MAX - 1)..), Dynamic::new(MAX)), Dynamic::new(1));
    assert_eq!(DimRange::length(&(MAX..), Dynamic::new(MAX)), Dynamic::new(0));
}

impl<D: Dim, T: Dim> DimRange<D> for ops::RangeFrom<T>
where D: DimSub<T>
{
    type Length = DimDiff<D, T>;

    #[inline(always)]
    fn lower(&self, _: D) -> usize {
        self.start.value()
    }

    #[inline(always)]
    fn length(&self, dimension: D) -> Self::Length {
        dimension.sub(self.start)
    }

    #[inline(always)]
    fn contained_by(&self, _: D) -> bool {
        true
    }
}

#[test]
fn dimrange_rangefrom_dimname() {
    use crate::{U5, U4};
    assert_eq!(DimRange::length(&(U1..), U5), U4);
}

impl<D: Dim> DimRange<D> for ops::RangeFull {
    type Length = D;

    #[inline(always)]
    fn lower(&self, _: D) -> usize {
        0
    }

    #[inline(always)]
    fn length(&self, dimension: D) -> Self::Length {
        dimension
    }

    #[inline(always)]
    fn contained_by(&self, _: D) -> bool {
        true
    }
}

#[test]
fn dimrange_rangefull() {
    use crate::U0;
    assert_eq!(DimRange::contained_by(&(..), U0), true);
    assert_eq!(DimRange::length(&(..), U1), U1);
}

impl<D: Dim> DimRange<D> for ops::RangeInclusive<usize> {
    type Length = Dynamic;

    #[inline(always)]
    fn lower(&self, _: D) -> usize {
        *self.start()
    }

    #[inline(always)]
    fn length(&self, _: D) -> Self::Length {
        Dynamic::new(
            if self.end() < self.start() {
                0
            } else {
                self.end().wrapping_sub(self.start().wrapping_sub(1))
            })
    }

    #[inline(always)]
    fn contained_by(&self, dimension: D) -> bool {
        (*self.start() < dimension.value()) && (*self.end() < dimension.value())
    }
}

#[test]
fn dimrange_rangeinclusive_usize() {
    use std::usize::MAX;
    use crate::U0;
    assert_eq!(DimRange::contained_by(&(0..=0), U0), false);
    assert_eq!(DimRange::contained_by(&(0..=0), U1), true);
    assert_eq!(DimRange::contained_by(&(MAX..=MAX), Dynamic::new(MAX)), false);
    assert_eq!(DimRange::contained_by(&((MAX-1)..=MAX), Dynamic::new(MAX)), false);
    assert_eq!(DimRange::contained_by(&((MAX-1)..=(MAX-1)), Dynamic::new(MAX)), true);
    assert_eq!(DimRange::length(&(0..=0), U1), Dynamic::new(1));
    assert_eq!(DimRange::length(&((MAX - 1)..=MAX), Dynamic::new(MAX)), Dynamic::new(2));
    assert_eq!(DimRange::length(&(MAX..=(MAX - 1)), Dynamic::new(MAX)), Dynamic::new(0));
    assert_eq!(DimRange::length(&(MAX..=MAX), Dynamic::new(MAX)), Dynamic::new(1));
}

impl<D: Dim> DimRange<D> for ops::RangeTo<usize>
{
    type Length = Dynamic;

    #[inline(always)]
    fn lower(&self, _: D) -> usize {
        0
    }

    #[inline(always)]
    fn length(&self, _: D) -> Self::Length {
        Dynamic::new(self.end)
    }

    #[inline(always)]
    fn contained_by(&self, dimension: D) -> bool {
        self.end <= dimension.value()
    }
}

#[test]
fn dimrange_rangeto_usize() {
    use std::usize::MAX;
    use crate::U0;
    assert_eq!(DimRange::contained_by(&(..0), U0), true);
    assert_eq!(DimRange::contained_by(&(..1), U0), false);
    assert_eq!(DimRange::contained_by(&(..0), U1), true);
    assert_eq!(DimRange::contained_by(&(..(MAX - 1)), Dynamic::new(MAX)), true);
    assert_eq!(DimRange::length(&(..(MAX - 1)), Dynamic::new(MAX)), Dynamic::new(MAX - 1));
    assert_eq!(DimRange::length(&(..MAX), Dynamic::new(MAX)), Dynamic::new(MAX));
}

impl<D: Dim> DimRange<D> for ops::RangeToInclusive<usize>
{
    type Length = Dynamic;

    #[inline(always)]
    fn lower(&self, _: D) -> usize {
        0
    }

    #[inline(always)]
    fn length(&self, _: D) -> Self::Length {
        Dynamic::new(self.end + 1)
    }

    #[inline(always)]
    fn contained_by(&self, dimension: D) -> bool {
        self.end < dimension.value()
    }
}

#[test]
fn dimrange_rangetoinclusive_usize() {
    use std::usize::MAX;
    use crate::U0;
    assert_eq!(DimRange::contained_by(&(..=0), U0), false);
    assert_eq!(DimRange::contained_by(&(..=1), U0), false);
    assert_eq!(DimRange::contained_by(&(..=0), U1), true);
    assert_eq!(DimRange::contained_by(&(..=(MAX)), Dynamic::new(MAX)), false);
    assert_eq!(DimRange::contained_by(&(..=(MAX - 1)), Dynamic::new(MAX)), true);
    assert_eq!(DimRange::length(&(..=(MAX - 1)), Dynamic::new(MAX)), Dynamic::new(MAX));
}

/// A helper trait used for indexing operations.
pub trait MatrixIndex<'a, N: Scalar, R: Dim, C: Dim, S: Storage<N, R, C>>: Sized {

    /// The output type returned by methods.
    type Output : 'a;

    /// Produces true if the given matrix is contained by this index.
    #[doc(hidden)]
    fn contained_by(&self, matrix: &Matrix<N, R, C, S>) -> bool;

    /// Produces a shared view of the data at this location if in bounds,
    /// or `None`, otherwise.
    #[doc(hidden)]
    #[inline(always)]
    fn get(self, matrix: &'a Matrix<N, R, C, S>) -> Option<Self::Output> {
        if self.contained_by(matrix) {
            Some(unsafe{self.get_unchecked(matrix)})
        } else {
            None
        }
    }

    /// Produces a shared view of the data at this location if in bounds
    /// without any bounds checking.
    #[doc(hidden)]
    unsafe fn get_unchecked(self, matrix: &'a Matrix<N, R, C, S>) -> Self::Output;

    /// Produces a shared view to the data at this location, or panics
    /// if out of bounds.
    #[doc(hidden)]
    #[inline(always)]
    fn index(self, matrix: &'a Matrix<N, R, C, S>) -> Self::Output {
        self.get(matrix).expect("Index out of bounds.")
    }
}

/// A helper trait used for indexing operations.
pub trait MatrixIndexMut<'a, N: Scalar, R: Dim, C: Dim, S: StorageMut<N, R, C>>: MatrixIndex<'a, N, R, C, S> {
    /// The output type returned by methods.
    type OutputMut : 'a;

    /// Produces a mutable view of the data at this location, without
    /// performing any bounds checking.
    #[doc(hidden)]
    unsafe fn get_unchecked_mut(self, matrix: &'a mut Matrix<N, R, C, S>) -> Self::OutputMut;

    /// Produces a mutable view of the data at this location, if in
    /// bounds.
    #[doc(hidden)]
    #[inline(always)]
    fn get_mut(self, matrix: &'a mut Matrix<N, R, C, S>) -> Option<Self::OutputMut> {
        if self.contained_by(matrix) {
            Some(unsafe{self.get_unchecked_mut(matrix)})
        } else {
            None
        }
    }

    /// Produces a mutable view of the data at this location, or panics
    /// if out of bounds.
    #[doc(hidden)]
    #[inline(always)]
    fn index_mut(self, matrix: &'a mut Matrix<N, R, C, S>) -> Self::OutputMut {
        self.get_mut(matrix).expect("Index out of bounds.")
    }
}

/// # Indexing Operations
/// ## Indices to Individual Elements
/// ### Two-Dimensional Indices
/// ```
/// # use nalgebra::*;
/// let matrix = Matrix2::new(0, 2,
///                           1, 3);
///
/// assert_eq!(matrix.index((0, 0)), &0);
/// assert_eq!(matrix.index((1, 0)), &1);
/// assert_eq!(matrix.index((0, 1)), &2);
/// assert_eq!(matrix.index((1, 1)), &3);
/// ```
///
/// ### Linear Address Indexing
/// ```
/// # use nalgebra::*;
/// let matrix = Matrix2::new(0, 2,
///                           1, 3);
///
/// assert_eq!(matrix.get(0), Some(&0));
/// assert_eq!(matrix.get(1), Some(&1));
/// assert_eq!(matrix.get(2), Some(&2));
/// assert_eq!(matrix.get(3), Some(&3));
/// ```
///
/// ## Indices to Individual Rows and Columns
/// ### Index to a Row
/// ```
/// # use nalgebra::*;
/// let matrix = Matrix2::new(0, 2,
///                           1, 3);
///
/// assert!(matrix.index((0, ..))
///     .eq(&Matrix1x2::new(0, 2)));
/// ```
///
/// ### Index to a Column
/// ```
/// # use nalgebra::*;
/// let matrix = Matrix2::new(0, 2,
///                           1, 3);
///
/// assert!(matrix.index((.., 0))
///     .eq(&Matrix2x1::new(0,
///                         1)));
/// ```
///
/// ## Indices to Parts of Individual Rows and Columns
/// ### Index to a Partial Row
/// ```
/// # use nalgebra::*;
/// let matrix = Matrix3::new(0, 3, 6,
///                           1, 4, 7,
///                           2, 5, 8);
///
/// assert!(matrix.index((0, ..2))
///     .eq(&Matrix1x2::new(0, 3)));
/// ```
///
/// ### Index to a Partial Column
/// ```
/// # use nalgebra::*;
/// let matrix = Matrix3::new(0, 3, 6,
///                           1, 4, 7,
///                           2, 5, 8);
///
/// assert!(matrix.index((..2, 0))
///     .eq(&Matrix2x1::new(0,
///                         1)));
///
/// assert!(matrix.index((U1.., 0))
///     .eq(&Matrix2x1::new(1,
///                         2)));
/// ```
/// ## Indices to Ranges of Rows and Columns
/// ### Index to a Range of Rows
/// ```
/// # use nalgebra::*;
/// let matrix = Matrix3::new(0, 3, 6,
///                           1, 4, 7,
///                           2, 5, 8);
///
/// assert!(matrix.index((1..3, ..))
///     .eq(&Matrix2x3::new(1, 4, 7,
///                         2, 5, 8)));
/// ```
/// ### Index to a Range of Columns
/// ```
/// # use nalgebra::*;
/// let matrix = Matrix3::new(0, 3, 6,
///                           1, 4, 7,
///                           2, 5, 8);
///
/// assert!(matrix.index((.., 1..3))
///     .eq(&Matrix3x2::new(3, 6,
///                         4, 7,
///                         5, 8)));
/// ```
impl<N: Scalar, R: Dim, C: Dim, S: Storage<N, R, C>> Matrix<N, R, C, S>
{
    /// Produces a view of the data at the given index, or
    /// `None` if the index is out of bounds.
    #[inline]
    pub fn get<'a, I>(&'a self, index: I) -> Option<I::Output>
    where
        I: MatrixIndex<'a, N, R, C, S>
    {
        index.get(self)
    }

    /// Produces a mutable view of the data at the given index, or
    /// `None` if the index is out of bounds.
    #[inline]
    pub fn get_mut<'a, I>(&'a mut self, index: I) -> Option<I::OutputMut>
    where
        S: StorageMut<N, R, C>,
        I: MatrixIndexMut<'a, N, R, C, S>
    {
        index.get_mut(self)
    }

    /// Produces a view of the data at the given index, or
    /// panics if the index is out of bounds.
    #[inline]
    pub fn index<'a, I>(&'a self, index: I) -> I::Output
    where
        I: MatrixIndex<'a, N, R, C, S>
    {
        index.index(self)
    }

    /// Produces a mutable view of the data at the given index, or
    /// panics if the index is out of bounds.
    #[inline]
    pub fn index_mut<'a, I>(&'a mut self, index: I) -> I::OutputMut
    where
        S: StorageMut<N, R, C>,
        I: MatrixIndexMut<'a, N, R, C, S>
    {
        index.index_mut(self)
    }

    /// Produces a view of the data at the given index, without doing
    /// any bounds checking.
    #[inline]
    pub unsafe fn get_unchecked<'a, I>(&'a self, index: I) -> I::Output
    where
        I: MatrixIndex<'a, N, R, C, S>
    {
        index.get_unchecked(self)
    }

    /// Returns a mutable view of the data at the given index, without doing
    /// any bounds checking.
    #[inline]
    pub unsafe fn get_unchecked_mut<'a, I>(&'a mut self, index: I) -> I::OutputMut
    where
        S: StorageMut<N, R, C>,
        I: MatrixIndexMut<'a, N, R, C, S>
    {
        index.get_unchecked_mut(self)
    }
}

// EXTRACT A SINGLE ELEMENT BY 1D LINEAR ADDRESS

impl<'a, N, R, C, S> MatrixIndex<'a, N, R, C, S> for usize
where
    N: Scalar,
    R: Dim,
    C: Dim,
    S: Storage<N, R, C>
{
    type Output = &'a N;

    #[doc(hidden)]
    #[inline(always)]
    fn contained_by(&self, matrix: &Matrix<N, R, C, S>) -> bool {
        *self < matrix.len()
    }

    #[doc(hidden)]
    #[inline(always)]
    unsafe fn get_unchecked(self, matrix: &'a Matrix<N, R, C, S>) -> Self::Output {
        matrix.data.get_unchecked_linear(self)
    }
}

impl<'a, N, R, C, S> MatrixIndexMut<'a, N, R, C, S> for usize
where
    N: Scalar,
    R: Dim,
    C: Dim,
    S: StorageMut<N, R, C>
{
    type OutputMut = &'a mut N;

    #[doc(hidden)]
    #[inline(always)]
    unsafe fn get_unchecked_mut(self, matrix: &'a mut Matrix<N, R, C, S>) -> Self::OutputMut
    where S: StorageMut<N, R, C>,
    {
        matrix.data.get_unchecked_linear_mut(self)
    }
}

// EXTRACT A SINGLE ELEMENT BY 2D COORDINATES

impl<'a, N, R, C, S> MatrixIndex<'a, N, R, C, S> for (usize, usize)
where
    N: Scalar,
    R: Dim,
    C: Dim,
    S: Storage<N, R, C>
{
    type Output = &'a N;

    #[doc(hidden)]
    #[inline(always)]
    fn contained_by(&self, matrix: &Matrix<N, R, C, S>) -> bool {
        let (rows, cols) = self;
        let (nrows, ncols) = matrix.data.shape();
        DimRange::contained_by(rows, nrows) && DimRange::contained_by(cols, ncols)
    }

    #[doc(hidden)]
    #[inline(always)]
    unsafe fn get_unchecked(self, matrix: &'a Matrix<N, R, C, S>) -> Self::Output {
        let (row, col) = self;
        matrix.data.get_unchecked(row, col)
    }
}

impl<'a, N, R, C, S> MatrixIndexMut<'a, N, R, C, S> for (usize, usize)
where
    N: Scalar,
    R: Dim,
    C: Dim,
    S: StorageMut<N, R, C>
{
    type OutputMut = &'a mut N;

    #[doc(hidden)]
    #[inline(always)]
    unsafe fn get_unchecked_mut(self, matrix: &'a mut Matrix<N, R, C, S>) -> Self::OutputMut
    where S: StorageMut<N, R, C>,
    {
        let (row, col) = self;
        matrix.data.get_unchecked_mut(row, col)
    }
}

macro_rules! impl_index_pair {
    (
      $R: ident,
      $C: ident,
      [<$($RTyP: ident : $RTyPB: ty,)*> usize => $ROut: ty
        $(where $RConstraintType: ty: $RConstraintBound: ident<$($RConstraintBoundParams: ty $( = $REqBound: ty )*),*>)*],
      [<$($CTyP: ident : $CTyPB: ty,)*> usize => $COut: ty
        $(where $CConstraintType: ty: $CConstraintBound: ident<$($CConstraintBoundParams: ty $( = $CEqBound: ty )*),*>)*]
    ) => {};

    (
      $R: ident,
      $C: ident,
      [<$($RTyP: ident: $RTyPB: tt),*> $RIdx: ty => $ROut: ty
        $(where $RConstraintType: ty: $RConstraintBound: ident $(<$($RConstraintBoundParams: ty $( = $REqBound: ty )*),*>)* )*],
      [<$($CTyP: ident: $CTyPB: tt),*> $CIdx: ty => $COut: ty
        $(where $CConstraintType: ty: $CConstraintBound: ident $(<$($CConstraintBoundParams: ty $( = $CEqBound: ty )*),*>)* )*]
    ) =>
    {
        impl<'a, N, $R, $C, S, $($RTyP : $RTyPB,)* $($CTyP : $CTyPB),*> MatrixIndex<'a, N, $R, $C, S> for ($RIdx, $CIdx)
        where
            N: Scalar,
            $R: Dim,
            $C: Dim,
            S: Storage<N, R, C>,
            $( $RConstraintType: $RConstraintBound $(<$( $RConstraintBoundParams $( = $REqBound )*),*>)* ,)*
            $( $CConstraintType: $CConstraintBound $(<$( $CConstraintBoundParams $( = $CEqBound )*),*>)* ),*
        {
            type Output = MatrixSlice<'a, N, $ROut, $COut, S::RStride, S::CStride>;

            #[doc(hidden)]
            #[inline(always)]
            fn contained_by(&self, matrix: &Matrix<N, $R, $C, S>) -> bool {
                let (rows, cols) = self;
                let (nrows, ncols) = matrix.data.shape();
                DimRange::contained_by(rows, nrows) && DimRange::contained_by(cols, ncols)
            }

            #[doc(hidden)]
            #[inline(always)]
            unsafe fn get_unchecked(self, matrix: &'a Matrix<N, $R, $C, S>) -> Self::Output {
                use crate::base::SliceStorage;

                let (rows, cols) = self;
                let (nrows, ncols) = matrix.data.shape();

                let data =
                    SliceStorage::new_unchecked(&matrix.data,
                        (rows.lower(nrows),  cols.lower(ncols)),
                        (rows.length(nrows), cols.length(ncols)));

                Matrix::from_data_statically_unchecked(data)
            }
        }

        impl<'a, N, $R, $C, S, $($RTyP : $RTyPB,)* $($CTyP : $CTyPB),*> MatrixIndexMut<'a, N, $R, $C, S> for ($RIdx, $CIdx)
        where
            N: Scalar,
            $R: Dim,
            $C: Dim,
            S: StorageMut<N, R, C>,
            $( $RConstraintType: $RConstraintBound $(<$( $RConstraintBoundParams $( = $REqBound )*),*>)* ,)*
            $( $CConstraintType: $CConstraintBound $(<$( $CConstraintBoundParams $( = $CEqBound )*),*>)* ),*
        {
            type OutputMut = MatrixSliceMut<'a, N, $ROut, $COut, S::RStride, S::CStride>;

            #[doc(hidden)]
            #[inline(always)]
            unsafe fn get_unchecked_mut(self, matrix: &'a mut Matrix<N, $R, $C, S>) -> Self::OutputMut {
                use crate::base::SliceStorageMut;

                let (rows, cols) = self;
                let (nrows, ncols) = matrix.data.shape();

                let data =
                    SliceStorageMut::new_unchecked(&mut matrix.data,
                        (rows.lower(nrows),  cols.lower(ncols)),
                        (rows.length(nrows), cols.length(ncols)));

                Matrix::from_data_statically_unchecked(data)
            }
        }
    }
}

macro_rules! impl_index_pairs {
    (index $R: ident with {} index $C: ident with {$($r: tt,)* }) => {};

    (index $R: ident with {$lh : tt, $($lt : tt,)*}
     index $C: ident with { $($r: tt,)* }) =>
    {
        $(
            impl_index_pair!{$R, $C, $lh, $r}
        )*
        impl_index_pairs!{index $R with {$($lt,)*} index $C with {$($r,)*}}
    }
}

impl_index_pairs!{
    index R with {
        [<> usize                         =>  U1],
        [<> ops::Range<usize>             =>  Dynamic],
        [<> ops::RangeFrom<usize>         =>  Dynamic],
        [<> ops::RangeFull                =>  R],
        [<> ops::RangeInclusive<usize>    =>  Dynamic],
        [<> ops::RangeTo<usize>           =>  Dynamic],
        [<> ops::RangeToInclusive<usize>  =>  Dynamic],

        [<I: Dim> ops::RangeFrom<I>
          =>  DimDiff<R, I>
          where R: DimSub<I>],
    }
    index C with {
        [<> usize                         =>  U1],
        [<> ops::Range<usize>             =>  Dynamic],
        [<> ops::RangeFrom<usize>         =>  Dynamic],
        [<> ops::RangeFull                =>  C],
        [<> ops::RangeInclusive<usize>    =>  Dynamic],
        [<> ops::RangeTo<usize>           =>  Dynamic],
        [<> ops::RangeToInclusive<usize>  =>  Dynamic],

        [<J: DimName> ops::RangeFrom<J>
          =>  DimDiff<C, J>
          where C: DimSub<J>],
    }
}
