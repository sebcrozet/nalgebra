use crate::base::dimension::{U2, U3};

use crate::geometry::{TAffine, TGeneral, TProjective, Transform};

/// A 2D general transformation that may not be invertible. Stored as a homogeneous 3x3 matrix.
pub type Transform2<N> = Transform<N, U2, TGeneral>;
/// An invertible 2D general transformation. Stored as a homogeneous 3x3 matrix.
pub type Projective2<N> = Transform<N, U2, TProjective>;
/// A 2D affine transformation. Stored as a homogeneous 3x3 matrix.
pub type Affine2<N> = Transform<N, U2, TAffine>;

/// A 3D general transformation that may not be inversible. Stored as a homogeneous 4x4 matrix.
pub type Transform3<N> = Transform<N, U3, TGeneral>;
/// An invertible 3D general transformation. Stored as a homogeneous 4x4 matrix.
pub type Projective3<N> = Transform<N, U3, TProjective>;
/// A 3D affine transformation. Stored as a homogeneous 4x4 matrix.
pub type Affine3<N> = Transform<N, U3, TAffine>;
