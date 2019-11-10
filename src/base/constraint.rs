//! Compatibility constraints between matrix shapes, e.g., for addition or multiplication.

use crate::{Dim, DimName, Dynamic};

/// A type used in `where` clauses for enforcing constraints.
pub struct ShapeConstraint;

/// Constraints `C1` and `R2` to be equivalent.
pub trait AreMultipliable<R1: Dim, C1: Dim, R2: Dim, C2: Dim>: DimEq<C1, R2> {}

impl<R1: Dim, C1: Dim, R2: Dim, C2: Dim> AreMultipliable<R1, C1, R2, C2> for ShapeConstraint where ShapeConstraint: DimEq<C1, R2>
{}

/// Constraints `D1` and `D2` to be equivalent.
pub trait DimEq<D1: Dim, D2: Dim> {
    /// This is either equal to `D1` or `D2`, always choosing the one (if any) which is a type-level
    /// constant.
    type Representative: Dim;
}

impl<D: Dim> DimEq<D, D> for ShapeConstraint {
    type Representative = D;
}

impl<D: DimName> DimEq<D, Dynamic> for ShapeConstraint {
    type Representative = D;
}

impl<D: DimName> DimEq<Dynamic, D> for ShapeConstraint {
    type Representative = D;
}

macro_rules! equality_trait_decl(
    ($($doc: expr, $Trait: ident),* $(,)*) => {$(
        // XXX: we can't do something like `DimEq<D1> for D2` because we would require a blancket impl…
        #[doc = $doc]
        pub trait $Trait<D1: Dim, D2: Dim>: DimEq<D1, D2> + DimEq<D2, D1> {
            /// This is either equal to `D1` or `D2`, always choosing the one (if any) which is a type-level
            /// constant.
            type Representative: Dim;
        }

        impl<D: Dim> $Trait<D, D> for ShapeConstraint {
            type Representative = D;
        }

        impl<D: DimName> $Trait<D, Dynamic> for ShapeConstraint {
            type Representative = D;
        }

        impl<D: DimName> $Trait<Dynamic, D> for ShapeConstraint {
            type Representative = D;
        }
    )*}
);

equality_trait_decl!(
    "Constraints `D1` and `D2` to be equivalent. \
     They are both assumed to be the number of \
     rows of a matrix.",
    SameNumberOfRows,
    "Constraints `D1` and `D2` to be equivalent. \
     They are both assumed to be the number of \
     columns of a matrix.",
    SameNumberOfColumns
);

/// Constraints D1 and D2 to be equivalent, where they both designate dimensions of algebraic
/// entities (e.g. square matrices).
pub trait SameDimension<D1: Dim, D2: Dim>:
    SameNumberOfRows<D1, D2> + SameNumberOfColumns<D1, D2>
{
    /// This is either equal to `D1` or `D2`, always choosing the one (if any) which is a type-level
    /// constant.
    type Representative: Dim;
}

impl<D: Dim> SameDimension<D, D> for ShapeConstraint {
    type Representative = D;
}

impl<D: DimName> SameDimension<D, Dynamic> for ShapeConstraint {
    type Representative = D;
}

impl<D: DimName> SameDimension<Dynamic, D> for ShapeConstraint {
    type Representative = D;
}
