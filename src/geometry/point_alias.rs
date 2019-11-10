use crate::{U1, U2, U3, U4, U5, U6};

use crate::geometry::Point;

/// A statically sized 1-dimensional column point.
pub type Point1<N> = Point<N, U1>;
/// A statically sized 2-dimensional column point.
pub type Point2<N> = Point<N, U2>;
/// A statically sized 3-dimensional column point.
pub type Point3<N> = Point<N, U3>;
/// A statically sized 4-dimensional column point.
pub type Point4<N> = Point<N, U4>;
/// A statically sized 5-dimensional column point.
pub type Point5<N> = Point<N, U5>;
/// A statically sized 6-dimensional column point.
pub type Point6<N> = Point<N, U6>;
