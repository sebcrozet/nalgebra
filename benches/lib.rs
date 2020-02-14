#![feature(test)]
#![allow(unused_macros)]

extern crate nalgebra as na;
extern crate rand;
extern crate rand_isaac;
extern crate test;
extern crate typenum;

#[macro_use]
extern crate criterion;

use na::DMatrix;
use rand::Rng;
use rand_isaac::IsaacRng;

pub mod core;
pub mod geometry;
pub mod linalg;

fn reproductible_dmatrix(nrows: usize, ncols: usize) -> DMatrix<f64> {
    use rand::SeedableRng;
    let mut rng = IsaacRng::seed_from_u64(0);
    DMatrix::<f64>::from_fn(nrows, ncols, |_, _| rng.gen())
}

criterion_main!(
    core::matrix,
    core::vector,
    geometry::quaternion,
    linalg::bidiagonal,
    linalg::cholesky,
    linalg::full_piv_lu,
    linalg::hessenberg,
    linalg::lu,
    linalg::qr,
    linalg::schur,
    linalg::solve,
    linalg::svd,
    linalg::polar,
    linalg::symmetric_eigen,
);
