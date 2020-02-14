use na::{DMatrix, Polar};

fn polar_decompose_4x4(bh: &mut criterion::Criterion) {
    let m = DMatrix::<f64>::new_random(4,4);
    bh.bench_function("polar_decompose_4x4", move |bh| bh.iter(|| test::black_box(Polar::new(m.clone()))));
}

fn polar_decompose_10x10(bh: &mut criterion::Criterion) {
    let m = crate::reproductible_dmatrix(10, 10);
    bh.bench_function("polar_decompose_10x10", move |bh| bh.iter(|| test::black_box(Polar::new(m.clone()))));
}

fn polar_decompose_100x100(bh: &mut criterion::Criterion) {
    let m = crate::reproductible_dmatrix(100, 100);
    bh.bench_function("polar_decompose_100x100", move |bh| bh.iter(|| test::black_box(Polar::new(m.clone()))));
}

fn polar_decompose_200x200(bh: &mut criterion::Criterion) {
    let m = crate::reproductible_dmatrix(200, 200);
    bh.bench_function("polar_decompose_200x200", move |bh| bh.iter(|| test::black_box(Polar::new(m.clone()))));
}

criterion_group!(polar,
    polar_decompose_4x4,
    polar_decompose_10x10,
    polar_decompose_100x100,
    polar_decompose_200x200,
);
