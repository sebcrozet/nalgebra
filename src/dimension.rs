#![allow(missing_docs)]

//! Traits and tags for identifying the dimension of all algebraic entities.

use std::any::{Any, TypeId};
use std::cmp;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};
use typenum::{
    self, Bit, Diff, Max, Maximum, Min, Minimum, Prod, Quot, Sum, UInt, UTerm, Unsigned, B1,
};

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Dim of dynamically-sized algebraic entities.
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub struct Dynamic {
    value: usize,
}

impl Dynamic {
    /// A dynamic size equal to `value`.
    #[inline]
    pub fn new(value: usize) -> Self {
        Self { value: value }
    }
}

#[cfg(feature = "serde-serialize")]
impl Serialize for Dynamic {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: Serializer {
        self.value.serialize(serializer)
    }
}

#[cfg(feature = "serde-serialize")]
impl<'de> Deserialize<'de> for Dynamic {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: Deserializer<'de> {
        usize::deserialize(deserializer).map(|x| Dynamic { value: x })
    }
}

/// Trait implemented by `Dynamic`.
pub trait IsDynamic {}
/// Trait implemented by `Dynamic` and type-level integers different from `U1`.
pub trait IsNotStaticOne {}

impl IsDynamic for Dynamic {}
impl IsNotStaticOne for Dynamic {}

/// Trait implemented by any type that can be used as a dimension. This includes type-level
/// integers and `Dynamic` (for dimensions not known at compile-time).
pub trait Dim: Any + Debug + Copy + PartialEq + Send + Sync {
    #[inline(always)]
    fn is<D: Dim>() -> bool {
        TypeId::of::<Self>() == TypeId::of::<D>()
    }

    /// Gets the compile-time value of `Self`. Returns `None` if it is not known, i.e., if `Self =
    /// Dynamic`.
    fn try_to_usize() -> Option<usize>;

    /// Gets the run-time value of `self`. For type-level integers, this is the same as
    /// `Self::try_to_usize().unwrap()`.
    fn value(&self) -> usize;

    /// Builds an instance of `Self` from a run-time value. Panics if `Self` is a type-level
    /// integer and `dim != Self::try_to_usize().unwrap()`.
    fn from_usize(dim: usize) -> Self;
}

impl Dim for Dynamic {
    #[inline]
    fn try_to_usize() -> Option<usize> {
        None
    }

    #[inline]
    fn from_usize(dim: usize) -> Self {
        Self::new(dim)
    }

    #[inline]
    fn value(&self) -> usize {
        self.value
    }
}

impl Add<usize> for Dynamic {
    type Output = Dynamic;

    #[inline]
    fn add(self, rhs: usize) -> Self {
        Self::new(self.value + rhs)
    }
}

impl Sub<usize> for Dynamic {
    type Output = Dynamic;

    #[inline]
    fn sub(self, rhs: usize) -> Self {
        Self::new(self.value - rhs)
    }
}

/*
 *
 * Operations.
 *
 */

macro_rules! dim_ops(
    ($($DimOp:    ident, $DimNameOp: ident,
       $Op:       ident, $op: ident, $op_path: path,
       $DimResOp: ident, $DimNameResOp: ident,
       $ResOp: ident);* $(;)*) => {$(
        pub type $DimResOp<D1, D2> = <D1 as $DimOp<D2>>::Output;

        pub trait $DimOp<D: Dim>: Dim {
            type Output: Dim;

            fn $op(self, other: D) -> Self::Output;
        }

        impl<D1: DimName, D2: DimName> $DimOp<D2> for D1
            where D1::Value: $Op<D2::Value>,
                  $ResOp<D1::Value, D2::Value>: NamedDim {
            type Output = <$ResOp<D1::Value, D2::Value> as NamedDim>::Name;

            #[inline]
            fn $op(self, _: D2) -> Self::Output {
                Self::Output::name()
            }
        }

        impl<D: Dim> $DimOp<D> for Dynamic {
            type Output = Dynamic;

            #[inline]
            fn $op(self, other: D) -> Dynamic {
                Dynamic::new($op_path(self.value, other.value()))
            }
        }

        impl<D: DimName> $DimOp<Dynamic> for D {
            type Output = Dynamic;

            #[inline]
            fn $op(self, other: Dynamic) -> Dynamic {
                Dynamic::new($op_path(self.value(), other.value))
            }
        }

        pub type $DimNameResOp<D1, D2> = <D1 as $DimNameOp<D2>>::Output;

        pub trait $DimNameOp<D: DimName>: DimName {
            type Output: DimName;

            fn $op(self, other: D) -> Self::Output;
        }

        impl<D1: DimName, D2: DimName> $DimNameOp<D2> for D1
            where D1::Value: $Op<D2::Value>,
                  $ResOp<D1::Value, D2::Value>: NamedDim {
            type Output = <$ResOp<D1::Value, D2::Value> as NamedDim>::Name;

            #[inline]
            fn $op(self, _: D2) -> Self::Output {
                Self::Output::name()
            }
        }
   )*}
);

dim_ops!(
    DimAdd, DimNameAdd, Add, add, Add::add, DimSum,     DimNameSum,     Sum;
    DimMul, DimNameMul, Mul, mul, Mul::mul, DimProd,    DimNameProd,    Prod;
    DimSub, DimNameSub, Sub, sub, Sub::sub, DimDiff,    DimNameDiff,    Diff;
    DimDiv, DimNameDiv, Div, div, Div::div, DimQuot,    DimNameQuot,    Quot;
    DimMin, DimNameMin, Min, min, cmp::min, DimMinimum, DimNameMinimum, Minimum;
    DimMax, DimNameMax, Max, max, cmp::max, DimMaximum, DimNameMaximum, Maximum;
);

/// Trait implemented exclusively by type-level integers.
pub trait DimName: Dim {
    type Value: NamedDim<Name = Self>;

    /// The name of this dimension, i.e., the singleton `Self`.
    #[inline]
    fn name() -> Self;

    // FIXME: this is not a very idiomatic name.
    /// The value of this dimension.
    #[inline]
    fn dim() -> usize {
        Self::Value::to_usize()
    }
}

pub trait NamedDim: Sized + Any + Unsigned {
    type Name: DimName<Value = Self>;
}

/// A type level dimension with a value of `1`.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct U1;

impl Dim for U1 {
    #[inline]
    fn try_to_usize() -> Option<usize> {
        Some(1)
    }

    #[inline]
    fn from_usize(dim: usize) -> Self {
        assert!(dim == 1, "Mismatched dimension.");
        U1
    }

    #[inline]
    fn value(&self) -> usize {
        1
    }
}

impl DimName for U1 {
    type Value = typenum::U1;

    #[inline]
    fn name() -> Self {
        U1
    }
}

impl NamedDim for typenum::U1 {
    type Name = U1;
}

macro_rules! named_dimension(
    ($($D: ident),* $(,)*) => {$(
        /// A type level dimension.
        #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
        #[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
        pub struct $D;

        impl Dim for $D {
            #[inline]
            fn try_to_usize() -> Option<usize> {
                Some(typenum::$D::to_usize())
            }

            #[inline]
            fn from_usize(dim: usize) -> Self {
                assert!(dim == typenum::$D::to_usize(), "Mismatched dimension.");
                $D
            }

            #[inline]
            fn value(&self) -> usize {
                typenum::$D::to_usize()
            }
        }

        impl DimName for $D {
            type Value = typenum::$D;

            #[inline]
            fn name() -> Self {
                $D
            }
        }

        impl NamedDim for typenum::$D {
            type Name = $D;
        }

        impl IsNotStaticOne for $D { }
    )*}
);

// We give explicit names to all Unsigned in [0, 128[
named_dimension!(
    U0, /*U1,*/ U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18,
    U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37,
    U38, U39, U40, U41, U42, U43, U44, U45, U46, U47, U48, U49, U50, U51, U52, U53, U54, U55, U56,
    U57, U58, U59, U60, U61, U62, U63, U64, U65, U66, U67, U68, U69, U70, U71, U72, U73, U74, U75,
    U76, U77, U78, U79, U80, U81, U82, U83, U84, U85, U86, U87, U88, U89, U90, U91, U92, U93, U94,
    U95, U96, U97, U98, U99, U100, U101, U102, U103, U104, U105, U106, U107, U108, U109, U110,
    U111, U112, U113, U114, U115, U116, U117, U118, U119, U120, U121, U122, U123, U124, U125, U126,
    U127
);

// For values greater than U1023, just use the typenum binary representation directly.
impl<
        A: Bit + Any + Debug + Copy + PartialEq + Send + Sync,
        B: Bit + Any + Debug + Copy + PartialEq + Send + Sync,
        C: Bit + Any + Debug + Copy + PartialEq + Send + Sync,
        D: Bit + Any + Debug + Copy + PartialEq + Send + Sync,
        E: Bit + Any + Debug + Copy + PartialEq + Send + Sync,
        F: Bit + Any + Debug + Copy + PartialEq + Send + Sync,
        G: Bit + Any + Debug + Copy + PartialEq + Send + Sync,
    > NamedDim for UInt<UInt<UInt<UInt<UInt<UInt<UInt<UInt<UTerm, B1>, A>, B>, C>, D>, E>, F>, G>
{
    type Name = Self;
}

impl<
        A: Bit + Any + Debug + Copy + PartialEq + Send + Sync,
        B: Bit + Any + Debug + Copy + PartialEq + Send + Sync,
        C: Bit + Any + Debug + Copy + PartialEq + Send + Sync,
        D: Bit + Any + Debug + Copy + PartialEq + Send + Sync,
        E: Bit + Any + Debug + Copy + PartialEq + Send + Sync,
        F: Bit + Any + Debug + Copy + PartialEq + Send + Sync,
        G: Bit + Any + Debug + Copy + PartialEq + Send + Sync,
    > Dim for UInt<UInt<UInt<UInt<UInt<UInt<UInt<UInt<UTerm, B1>, A>, B>, C>, D>, E>, F>, G>
{
    #[inline]
    fn try_to_usize() -> Option<usize> {
        Some(Self::to_usize())
    }

    #[inline]
    fn from_usize(dim: usize) -> Self {
        assert!(dim == Self::to_usize(), "Mismatched dimension.");
        Self::new()
    }

    #[inline]
    fn value(&self) -> usize {
        Self::to_usize()
    }
}

impl<
        A: Bit + Any + Debug + Copy + PartialEq + Send + Sync,
        B: Bit + Any + Debug + Copy + PartialEq + Send + Sync,
        C: Bit + Any + Debug + Copy + PartialEq + Send + Sync,
        D: Bit + Any + Debug + Copy + PartialEq + Send + Sync,
        E: Bit + Any + Debug + Copy + PartialEq + Send + Sync,
        F: Bit + Any + Debug + Copy + PartialEq + Send + Sync,
        G: Bit + Any + Debug + Copy + PartialEq + Send + Sync,
    > DimName for UInt<UInt<UInt<UInt<UInt<UInt<UInt<UInt<UTerm, B1>, A>, B>, C>, D>, E>, F>, G>
{
    type Value = Self;

    #[inline]
    fn name() -> Self {
        Self::new()
    }
}

impl<
        A: Bit + Any + Debug + Copy + PartialEq + Send + Sync,
        B: Bit + Any + Debug + Copy + PartialEq + Send + Sync,
        C: Bit + Any + Debug + Copy + PartialEq + Send + Sync,
        D: Bit + Any + Debug + Copy + PartialEq + Send + Sync,
        E: Bit + Any + Debug + Copy + PartialEq + Send + Sync,
        F: Bit + Any + Debug + Copy + PartialEq + Send + Sync,
        G: Bit + Any + Debug + Copy + PartialEq + Send + Sync,
    > IsNotStaticOne
    for UInt<UInt<UInt<UInt<UInt<UInt<UInt<UInt<UTerm, B1>, A>, B>, C>, D>, E>, F>, G>
{
}

impl<U: Unsigned + DimName, B: Bit + Any + Debug + Copy + PartialEq + Send + Sync> NamedDim
    for UInt<U, B>
{
    type Name = UInt<U, B>;
}

impl<U: Unsigned + DimName, B: Bit + Any + Debug + Copy + PartialEq + Send + Sync> Dim
    for UInt<U, B>
{
    #[inline]
    fn try_to_usize() -> Option<usize> {
        Some(Self::to_usize())
    }

    #[inline]
    fn from_usize(dim: usize) -> Self {
        assert!(dim == Self::to_usize(), "Mismatched dimension.");
        Self::new()
    }

    #[inline]
    fn value(&self) -> usize {
        Self::to_usize()
    }
}

impl<U: Unsigned + DimName, B: Bit + Any + Debug + Copy + PartialEq + Send + Sync> DimName
    for UInt<U, B>
{
    type Value = UInt<U, B>;

    #[inline]
    fn name() -> Self {
        Self::new()
    }
}

impl<U: Unsigned + DimName, B: Bit + Any + Debug + Copy + PartialEq + Send + Sync> IsNotStaticOne
    for UInt<U, B>
{
}
