//! Matrix iterators.

use std::marker::PhantomData;
use std::mem;

use crate::{Dim, U1};
use crate::base::storage::{Storage, StorageMut};
use crate::base::{Scalar, Matrix, MatrixSlice, MatrixSliceMut};

macro_rules! iterator {
    (struct $Name:ident for $Storage:ident.$ptr: ident -> $Ptr:ty, $Ref:ty, $SRef: ty) => {
        /// An iterator through a dense matrix with arbitrary strides matrix.
        pub struct $Name<'a, N: Scalar, R: Dim, C: Dim, S: 'a + $Storage<N, R, C>> {
            ptr: $Ptr,
            inner_ptr: $Ptr,
            inner_end: $Ptr,
            size: usize, // We can't use an end pointer here because a stride might be zero.
            strides: (S::RStride, S::CStride),
            _phantoms: PhantomData<($Ref, R, C, S)>,
        }

        // FIXME: we need to specialize for the case where the matrix storage is owned (in which
        // case the iterator is trivial because it does not have any stride).
        impl<'a, N: Scalar, R: Dim, C: Dim, S: 'a + $Storage<N, R, C>> $Name<'a, N, R, C, S> {
            /// Creates a new iterator for the given matrix storage.
            pub fn new(storage: $SRef) -> $Name<'a, N, R, C, S> {
                let shape = storage.shape();
                let strides = storage.strides();
                let inner_offset = shape.0.value() * strides.0.value();
                let ptr = storage.$ptr();

                $Name {
                    ptr: ptr,
                    inner_ptr: ptr,
                    inner_end: unsafe { ptr.offset(inner_offset as isize) },
                    size: shape.0.value() * shape.1.value(),
                    strides: strides,
                    _phantoms: PhantomData,
                }
            }
        }

        impl<'a, N: Scalar, R: Dim, C: Dim, S: 'a + $Storage<N, R, C>> Iterator
            for $Name<'a, N, R, C, S>
        {
            type Item = $Ref;

            #[inline]
            fn next(&mut self) -> Option<$Ref> {
                unsafe {
                    if self.size == 0 {
                        None
                    } else {
                        self.size -= 1;

                        // Jump to the next outer dimension if needed.
                        if self.ptr == self.inner_end {
                            let stride = self.strides.1.value() as isize;
                            self.inner_end = self.ptr.offset(stride);
                            self.ptr = self.inner_ptr.offset(stride);
                            self.inner_ptr = self.ptr;
                        }

                        // Go to the next element.
                        let old = self.ptr;

                        let stride = self.strides.0.value() as isize;
                        self.ptr = self.ptr.offset(stride);

                        Some(mem::transmute(old))
                    }
                }
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                (self.size, Some(self.size))
            }

            #[inline]
            fn count(self) -> usize {
                self.size_hint().0
            }
        }

        impl<'a, N: Scalar, R: Dim, C: Dim, S: 'a + $Storage<N, R, C>> ExactSizeIterator
            for $Name<'a, N, R, C, S>
        {
            #[inline]
            fn len(&self) -> usize {
                self.size
            }
        }
    };
}

iterator!(struct MatrixIter for Storage.ptr -> *const N, &'a N, &'a S);
iterator!(struct MatrixIterMut for StorageMut.ptr_mut -> *mut N, &'a mut N, &'a mut S);


/*
 *
 * Row iterators.
 *
 */
#[derive(Clone)]
/// An iterator through the rows of a matrix.
pub struct RowIter<'a, N: Scalar, R: Dim, C: Dim, S: Storage<N, R, C>> {
    mat: &'a Matrix<N, R, C, S>,
    curr: usize
}

impl<'a, N: Scalar, R: Dim, C: Dim, S: 'a + Storage<N, R, C>> RowIter<'a, N, R, C, S> {
    pub(crate) fn new(mat: &'a Matrix<N, R, C, S>) -> Self {
        RowIter {
            mat, curr: 0
        }
    }
}


impl<'a, N: Scalar, R: Dim, C: Dim, S: 'a + Storage<N, R, C>> Iterator for RowIter<'a, N, R, C, S> {
    type Item = MatrixSlice<'a, N, U1, C, S::RStride, S::CStride>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.curr < self.mat.nrows() {
            let res = self.mat.row(self.curr);
            self.curr += 1;
            Some(res)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.mat.nrows() - self.curr, Some(self.mat.nrows() - self.curr))
    }

    #[inline]
    fn count(self) -> usize {
        self.mat.nrows() - self.curr
    }
}

impl<'a, N: Scalar, R: Dim, C: Dim, S: 'a + Storage<N, R, C>> ExactSizeIterator for RowIter<'a, N, R, C, S> {
    #[inline]
    fn len(&self) -> usize {
        self.mat.nrows() - self.curr
    }
}


/// An iterator through the mutable rows of a matrix.
pub struct RowIterMut<'a, N: Scalar, R: Dim, C: Dim, S: StorageMut<N, R, C>> {
    mat: *mut Matrix<N, R, C, S>,
    curr: usize,
    phantom: PhantomData<&'a mut Matrix<N, R, C, S>>
}

impl<'a, N: Scalar, R: Dim, C: Dim, S: 'a + StorageMut<N, R, C>> RowIterMut<'a, N, R, C, S> {
    pub(crate) fn new(mat: &'a mut Matrix<N, R, C, S>) -> Self {
        RowIterMut {
            mat,
            curr: 0,
            phantom: PhantomData
        }
    }

    fn nrows(&self) -> usize {
        unsafe {
            (*self.mat).nrows()
        }
    }
}


impl<'a, N: Scalar, R: Dim, C: Dim, S: 'a + StorageMut<N, R, C>> Iterator for RowIterMut<'a, N, R, C, S> {
    type Item = MatrixSliceMut<'a, N, U1, C, S::RStride, S::CStride>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.curr < self.nrows() {
            let res = unsafe { (*self.mat).row_mut(self.curr) };
            self.curr += 1;
            Some(res)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.nrows() - self.curr, Some(self.nrows() - self.curr))
    }

    #[inline]
    fn count(self) -> usize {
        self.nrows() - self.curr
    }
}

impl<'a, N: Scalar, R: Dim, C: Dim, S: 'a + StorageMut<N, R, C>> ExactSizeIterator for RowIterMut<'a, N, R, C, S> {
    #[inline]
    fn len(&self) -> usize {
        self.nrows() - self.curr
    }
}


/*
 *
 * Column iterators.
 *
 */
#[derive(Clone)]
/// An iterator through the columns of a matrix.
pub struct ColumnIter<'a, N: Scalar, R: Dim, C: Dim, S: Storage<N, R, C>> {
    mat: &'a Matrix<N, R, C, S>,
    curr: usize
}

impl<'a, N: Scalar, R: Dim, C: Dim, S: 'a + Storage<N, R, C>> ColumnIter<'a, N, R, C, S> {
    pub(crate) fn new(mat: &'a Matrix<N, R, C, S>) -> Self {
        ColumnIter {
            mat, curr: 0
        }
    }
}


impl<'a, N: Scalar, R: Dim, C: Dim, S: 'a + Storage<N, R, C>> Iterator for ColumnIter<'a, N, R, C, S> {
    type Item = MatrixSlice<'a, N, R, U1, S::RStride, S::CStride>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.curr < self.mat.ncols() {
            let res = self.mat.column(self.curr);
            self.curr += 1;
            Some(res)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.mat.ncols() - self.curr, Some(self.mat.ncols() - self.curr))
    }

    #[inline]
    fn count(self) -> usize {
        self.mat.ncols() - self.curr
    }
}

impl<'a, N: Scalar, R: Dim, C: Dim, S: 'a + Storage<N, R, C>> ExactSizeIterator for ColumnIter<'a, N, R, C, S> {
    #[inline]
    fn len(&self) -> usize {
        self.mat.ncols() - self.curr
    }
}


/// An iterator through the mutable columns of a matrix.
pub struct ColumnIterMut<'a, N: Scalar, R: Dim, C: Dim, S: StorageMut<N, R, C>> {
    mat: *mut Matrix<N, R, C, S>,
    curr: usize,
    phantom: PhantomData<&'a mut Matrix<N, R, C, S>>
}

impl<'a, N: Scalar, R: Dim, C: Dim, S: 'a + StorageMut<N, R, C>> ColumnIterMut<'a, N, R, C, S> {
    pub(crate) fn new(mat: &'a mut Matrix<N, R, C, S>) -> Self {
        ColumnIterMut {
            mat,
            curr: 0,
            phantom: PhantomData
        }
    }

    fn ncols(&self) -> usize {
        unsafe {
            (*self.mat).ncols()
        }
    }
}


impl<'a, N: Scalar, R: Dim, C: Dim, S: 'a + StorageMut<N, R, C>> Iterator for ColumnIterMut<'a, N, R, C, S> {
    type Item = MatrixSliceMut<'a, N, R, U1, S::RStride, S::CStride>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.curr < self.ncols() {
            let res = unsafe { (*self.mat).column_mut(self.curr) };
            self.curr += 1;
            Some(res)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.ncols() - self.curr, Some(self.ncols() - self.curr))
    }

    #[inline]
    fn count(self) -> usize {
        self.ncols() - self.curr
    }
}

impl<'a, N: Scalar, R: Dim, C: Dim, S: 'a + StorageMut<N, R, C>> ExactSizeIterator for ColumnIterMut<'a, N, R, C, S> {
    #[inline]
    fn len(&self) -> usize {
        self.ncols() - self.curr
    }
}

