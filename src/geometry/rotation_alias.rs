use crate::{U2, U3};

use crate::geometry::Rotation;

/// A 2-dimensional rotation matrix.
pub type Rotation2<N> = Rotation<N, U2>;

/// A 3-dimensional rotation matrix.
pub type Rotation3<N> = Rotation<N, U3>;
