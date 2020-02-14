#![cfg_attr(rustfmt, rustfmt_skip)]

#[cfg(feature = "arbitrary")]
mod quickcheck_tests {
    macro_rules! gen_tests(
        ($module: ident, $scalar: ty) => {
            mod $module {
                use na::{
                    DMatrix, DVector, ComplexField
                };
                use std::cmp;
                #[allow(unused_imports)]
                use crate::core::helper::{RandScalar, RandComplex};

                quickcheck! {
                    fn polar_recompose(m: DMatrix<$scalar>) -> bool {
                        let m = m.map(|e| e.0);
                        if m.len() > 0 {
                            let polar = m.clone().polar();
                            let recomp_m_l = polar.clone().recompose_left().unwrap();
                            let recomp_m_r = polar.clone().recompose_right().unwrap();

                            println!("Hello");
                            println!("Size of R: {:?}", polar.clone().r.unwrap().shape());
                            println!("Size of p_R: {:?}", polar.clone().p_r.unwrap().shape());
                            println!("Size of p_L: {:?}", polar.clone().p_l.unwrap().shape());

                            relative_eq!(m, recomp_m_l, epsilon = 1.0e-5) &&
                            relative_eq!(m, recomp_m_r, epsilon = 1.0e-5) 
                        }
                        else {
                            true 
                        }
                    }

                    fn polar_properties(m: DMatrix<$scalar>) -> bool {
                        let m = m.map(|e| e.0);
                        if m.len() > 0 {
                            let polar = m.clone().polar();
                            let p_r= polar.p_r.unwrap();
                            let p_l= polar.p_l.unwrap();

                            p_r.is_square() && p_l.is_square()
                        }
                        else {
                            true 
                        }
                    }
                }
            }
        }
    );

    gen_tests!(complex, RandComplex<f64>);
    gen_tests!(f64, RandScalar<f64>);
}
