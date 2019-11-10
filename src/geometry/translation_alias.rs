use crate::{U1, U2, U3, U4, U5, U6};

use crate::geometry::Translation;

/// A 1-dimensional translation.
pub type Translation1<N> = Translation<N, U1>;

/// A 2-dimensional translation.
pub type Translation2<N> = Translation<N, U2>;

/// A 3-dimensional translation.
pub type Translation3<N> = Translation<N, U3>;

/// A 4-dimensional translation.
pub type Translation4<N> = Translation<N, U4>;

/// A 5-dimensional translation.
pub type Translation5<N> = Translation<N, U5>;

/// A 6-dimensional translation.
pub type Translation6<N> = Translation<N, U6>;
