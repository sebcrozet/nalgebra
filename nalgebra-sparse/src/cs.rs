use crate::pattern::SparsityPattern;
use crate::{SparseEntry, SparseEntryMut};

use std::sync::Arc;
use std::ops::Range;
use std::mem::replace;
use num_traits::One;
use nalgebra::Scalar;

/// An abstract compressed matrix.
///
/// For the time being, this is only used internally to share implementation between
/// CSR and CSC matrices.
///
/// A CSR matrix is obtained by associating rows with the major dimension, while a CSC matrix
/// is obtained by associating columns with the major dimension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsMatrix<T> {
    sparsity_pattern: Arc<SparsityPattern>,
    values: Vec<T>
}

impl<T> CsMatrix<T> {
    /// Create a zero matrix with no explicitly stored entries.
    #[inline]
    pub fn new(major_dim: usize, minor_dim: usize) -> Self {
        Self {
            sparsity_pattern: Arc::new(SparsityPattern::new(major_dim, minor_dim)),
            values: vec![],
        }
    }

    #[inline]
    pub fn pattern(&self) -> &Arc<SparsityPattern> {
        &self.sparsity_pattern
    }

    #[inline]
    pub fn values(&self) -> &[T] {
        &self.values
    }

    #[inline]
    pub fn values_mut(&mut self) -> &mut [T] {
        &mut self.values
    }

    /// Returns the raw data represented as a tuple `(major_offsets, minor_indices, values)`.
    #[inline]
    pub fn cs_data(&self) -> (&[usize], &[usize], &[T]) {
        let pattern = self.pattern().as_ref();
        (pattern.major_offsets(), pattern.minor_indices(), &self.values)
    }

    /// Returns the raw data represented as a tuple `(major_offsets, minor_indices, values)`.
    #[inline]
    pub fn cs_data_mut(&mut self) -> (&[usize], &[usize], &mut [T]) {
        let pattern = self.sparsity_pattern.as_ref();
        (pattern.major_offsets(), pattern.minor_indices(), &mut self.values)
    }

    #[inline]
    pub fn pattern_and_values_mut(&mut self) -> (&Arc<SparsityPattern>, &mut [T]) {
        (&self.sparsity_pattern, &mut self.values)
    }

    #[inline]
    pub fn from_pattern_and_values(pattern: Arc<SparsityPattern>, values: Vec<T>)
                                   -> Self {
        assert_eq!(pattern.nnz(), values.len(), "Internal error: consumers should verify shape compatibility.");
        Self {
            sparsity_pattern: pattern,
            values,
        }
    }

    /// Internal method for simplifying access to a lane's data
    #[inline]
    pub fn get_index_range(&self, row_index: usize) -> Option<Range<usize>> {
        let row_begin = *self.sparsity_pattern.major_offsets().get(row_index)?;
        let row_end = *self.sparsity_pattern.major_offsets().get(row_index + 1)?;
        Some(row_begin .. row_end)
    }

    pub fn take_pattern_and_values(self) -> (Arc<SparsityPattern>, Vec<T>) {
        (self.sparsity_pattern, self.values)
    }

    #[inline]
    pub fn disassemble(self) -> (Vec<usize>, Vec<usize>, Vec<T>) {
        // Take an Arc to the pattern, which might be the sole reference to the data after
        // taking the values. This is important, because it might let us avoid cloning the data
        // further below.
        let pattern = self.sparsity_pattern;
        let values = self.values;

        // Try to take the pattern out of the `Arc` if possible,
        // otherwise clone the pattern.
        let owned_pattern = Arc::try_unwrap(pattern)
            .unwrap_or_else(|arc| SparsityPattern::clone(&*arc));
        let (offsets, indices) = owned_pattern.disassemble();

        (offsets, indices, values)
    }

    /// Returns an entry for the given major/minor indices, or `None` if the indices are out
    /// of bounds.
    pub fn get_entry(&self, major_index: usize, minor_index: usize) -> Option<SparseEntry<T>> {
        let row_range = self.get_index_range(major_index)?;
        let (_, minor_indices, values) = self.cs_data();
        let minor_indices = &minor_indices[row_range.clone()];
        let values = &values[row_range];
        get_entry_from_slices(self.pattern().minor_dim(), minor_indices, values, minor_index)
    }

    /// Returns a mutable entry for the given major/minor indices, or `None` if the indices are out
    /// of bounds.
    pub fn get_entry_mut(&mut self, major_index: usize, minor_index: usize)
                         -> Option<SparseEntryMut<T>> {
        let row_range = self.get_index_range(major_index)?;
        let minor_dim = self.pattern().minor_dim();
        let (_, minor_indices, values) = self.cs_data_mut();
        let minor_indices = &minor_indices[row_range.clone()];
        let values = &mut values[row_range];
        get_mut_entry_from_slices(minor_dim, minor_indices, values, minor_index)
    }

    pub fn get_lane(&self, index: usize) -> Option<CsLane<T>> {
        let range = self.get_index_range(index)?;
        let (_, minor_indices, values) = self.cs_data();
        Some(CsLane {
            minor_indices: &minor_indices[range.clone()],
            values: &values[range],
            minor_dim: self.pattern().minor_dim()
        })
    }

    #[inline]
    pub fn get_lane_mut(&mut self, index: usize) -> Option<CsLaneMut<T>> {
        let range = self.get_index_range(index)?;
        let minor_dim = self.pattern().minor_dim();
        let (_, minor_indices, values) = self.cs_data_mut();
        Some(CsLaneMut {
            minor_dim,
            minor_indices: &minor_indices[range.clone()],
            values: &mut values[range]
        })
    }

    #[inline]
    pub fn lane_iter(&self) -> CsLaneIter<T> {
        CsLaneIter::new(self.pattern().as_ref(), self.values())
    }

    #[inline]
    pub fn lane_iter_mut(&mut self) -> CsLaneIterMut<T> {
        CsLaneIterMut::new(self.sparsity_pattern.as_ref(), &mut self.values)
    }

    #[inline]
    pub fn filter<P>(&self, predicate: P) -> Self
    where
        T: Clone,
        P: Fn(usize, usize, &T) -> bool
    {
        let (major_dim, minor_dim) = (self.pattern().major_dim(), self.pattern().minor_dim());
        let mut new_offsets = Vec::with_capacity(self.pattern().major_dim() + 1);
        let mut new_indices = Vec::new();
        let mut new_values = Vec::new();

        new_offsets.push(0);
        for (i, lane) in self.lane_iter().enumerate() {
            for (&j, value) in lane.minor_indices().iter().zip(lane.values) {
                if predicate(i, j, value) {
                    new_indices.push(j);
                    new_values.push(value.clone());
                }
            }

            new_offsets.push(new_indices.len());
        }

        // TODO: Avoid checks here
        let new_pattern = SparsityPattern::try_from_offsets_and_indices(
            major_dim,
            minor_dim,
            new_offsets,
            new_indices)
            .expect("Internal error: Sparsity pattern must always be valid.");

        Self::from_pattern_and_values(Arc::new(new_pattern), new_values)
    }
}

impl<T: Scalar + One> CsMatrix<T> {
    /// TODO
    #[inline]
    pub fn identity(n: usize) -> Self {
        let offsets: Vec<_> = (0 ..= n).collect();
        let indices: Vec<_> = (0 .. n).collect();
        let values = vec![T::one(); n];

        // TODO: We should skip checks here
        let pattern = SparsityPattern::try_from_offsets_and_indices(n, n, offsets, indices)
            .unwrap();
        Self::from_pattern_and_values(Arc::new(pattern), values)
    }
}

fn get_entry_from_slices<'a, T>(
    minor_dim: usize,
    minor_indices: &'a [usize],
    values: &'a [T],
    global_minor_index: usize) -> Option<SparseEntry<'a, T>> {
    let local_index = minor_indices.binary_search(&global_minor_index);
    if let Ok(local_index) = local_index {
        Some(SparseEntry::NonZero(&values[local_index]))
    } else if global_minor_index < minor_dim {
        Some(SparseEntry::Zero)
    } else {
        None
    }
}

fn get_mut_entry_from_slices<'a, T>(
    minor_dim: usize,
    minor_indices: &'a [usize],
    values: &'a mut [T],
    global_minor_indices: usize) -> Option<SparseEntryMut<'a, T>> {
    let local_index = minor_indices.binary_search(&global_minor_indices);
    if let Ok(local_index) = local_index {
        Some(SparseEntryMut::NonZero(&mut values[local_index]))
    } else if global_minor_indices < minor_dim {
        Some(SparseEntryMut::Zero)
    } else {
        None
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsLane<'a, T> {
    minor_dim: usize,
    minor_indices: &'a [usize],
    values: &'a [T]
}

#[derive(Debug, PartialEq, Eq)]
pub struct CsLaneMut<'a, T> {
    minor_dim: usize,
    minor_indices: &'a [usize],
    values: &'a mut [T]
}

pub struct CsLaneIter<'a, T> {
    // The index of the lane that will be returned on the next iteration
    current_lane_idx: usize,
    pattern: &'a SparsityPattern,
    remaining_values: &'a [T],
}

impl<'a, T> CsLaneIter<'a, T> {
    pub fn new(pattern: &'a SparsityPattern, values: &'a [T]) -> Self {
        Self {
            current_lane_idx: 0,
            pattern,
            remaining_values: values
        }
    }
}

impl<'a, T> Iterator for CsLaneIter<'a, T>
    where
        T: 'a
{
    type Item = CsLane<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        let lane = self.pattern.get_lane(self.current_lane_idx);
        let minor_dim = self.pattern.minor_dim();

        if let Some(minor_indices) = lane {
            let count = minor_indices.len();
            let values_in_lane = &self.remaining_values[..count];
            self.remaining_values = &self.remaining_values[count ..];
            self.current_lane_idx += 1;

            Some(CsLane {
                minor_dim,
                minor_indices,
                values: values_in_lane
            })
        } else {
            None
        }
    }
}

pub struct CsLaneIterMut<'a, T> {
    // The index of the lane that will be returned on the next iteration
    current_lane_idx: usize,
    pattern: &'a SparsityPattern,
    remaining_values: &'a mut [T],
}

impl<'a, T> CsLaneIterMut<'a, T> {
    pub fn new(pattern: &'a SparsityPattern, values: &'a mut [T]) -> Self {
        Self {
            current_lane_idx: 0,
            pattern,
            remaining_values: values
        }
    }
}

impl<'a, T> Iterator for CsLaneIterMut<'a, T>
    where
        T: 'a
{
    type Item = CsLaneMut<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        let lane = self.pattern.get_lane(self.current_lane_idx);
        let minor_dim = self.pattern.minor_dim();

        if let Some(minor_indices) = lane {
            let count = minor_indices.len();

            let remaining = replace(&mut self.remaining_values, &mut []);
            let (values_in_lane, remaining) = remaining.split_at_mut(count);
            self.remaining_values = remaining;
            self.current_lane_idx += 1;

            Some(CsLaneMut {
                minor_dim,
                minor_indices,
                values: values_in_lane
            })
        } else {
            None
        }
    }
}

/// Implement the methods common to both CsLane and CsLaneMut. See the documentation for the
/// methods delegated here by CsrMatrix and CscMatrix members for more information.
macro_rules! impl_cs_lane_common_methods {
    ($name:ty) => {
        impl<'a, T> $name {
            #[inline]
            pub fn minor_dim(&self) -> usize {
                self.minor_dim
            }

            #[inline]
            pub fn nnz(&self) -> usize {
                self.minor_indices.len()
            }

            #[inline]
            pub fn minor_indices(&self) -> &[usize] {
                self.minor_indices
            }

            #[inline]
            pub fn values(&self) -> &[T] {
                self.values
            }

            #[inline]
            pub fn get_entry(&self, global_col_index: usize) -> Option<SparseEntry<T>> {
                get_entry_from_slices(
                    self.minor_dim,
                    self.minor_indices,
                    self.values,
                    global_col_index)
            }
        }
    }
}

impl_cs_lane_common_methods!(CsLane<'a, T>);
impl_cs_lane_common_methods!(CsLaneMut<'a, T>);

impl<'a, T> CsLaneMut<'a, T> {
    pub fn values_mut(&mut self) -> &mut [T] {
        self.values
    }

    pub fn indices_and_values_mut(&mut self) -> (&[usize], &mut [T]) {
        (self.minor_indices, self.values)
    }

    pub fn get_entry_mut(&mut self, global_minor_index: usize) -> Option<SparseEntryMut<T>> {
        get_mut_entry_from_slices(self.minor_dim,
                                  self.minor_indices,
                                  self.values,
                                  global_minor_index)
    }
}