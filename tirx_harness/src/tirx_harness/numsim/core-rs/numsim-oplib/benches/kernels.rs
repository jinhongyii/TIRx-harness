//! numsim-oplib kernel microbenches (W4): dtype conversion, the pinned host
//! NaN rule, the MMA increasing-K chain, `mma.sync`, warp reductions and the
//! v2 math builtins. `cargo bench -p numsim-oplib --bench kernels`;
//! baselines in `docs/development/oplib-benchmarks.md`.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use numsim_oplib::{cvt, fpenv, mma, scalar, warp};
use numsim_types::{WarpMask, WarpValue};

const N: usize = 1024;

fn inputs() -> Vec<f32> {
    (0..N).map(|i| (i as f32 - 512.0) * 0.731 + 0.01).collect()
}

/// One conversion over 1024 values (per-element cost = time / 1024).
fn conversions(c: &mut Criterion) {
    let x = inputs();
    let h: Vec<u16> = x.iter().map(|&v| cvt::f32_to_fp16_bits(v)).collect();
    let mut g = c.benchmark_group("cvt_x1024");
    g.bench_function("f32_to_fp16_rn", |b| b.iter(|| black_box(&x).iter().map(|&v| cvt::f32_to_fp16_bits(v) as u32).sum::<u32>()));
    g.bench_function("f32_to_bf16_rn", |b| b.iter(|| black_box(&x).iter().map(|&v| cvt::f32_to_bf16_bits(v) as u32).sum::<u32>()));
    g.bench_function("fp16_to_f32", |b| b.iter(|| black_box(&h).iter().map(|&v| cvt::fp16_bits_to_f32(v)).sum::<f32>()));
    g.bench_function("f32_to_tf32", |b| b.iter(|| black_box(&x).iter().map(|&v| cvt::f32_to_tf32(v)).sum::<f32>()));
    g.bench_function("f32_to_e4m3_satfinite", |b| b.iter(|| black_box(&x).iter().map(|&v| cvt::f32_to_float8_e4m3fn_bits(v) as u32).sum::<u32>()));
    g.bench_function("f32_to_e2m1_satfinite", |b| {
        b.iter(|| black_box(&x).iter().map(|&v| cvt::f32_to_narrow_float_bits_rn_satfinite(v, cvt::FLOAT4_E2M1) as u32).sum::<u32>())
    });
    g.finish();
}

/// The pinned host NaN rule (delta D8): finite inputs vs. all-NaN inputs.
fn host_nan_rule(c: &mut Criterion) {
    let x = inputs();
    let nan: Vec<f32> = (0..N).map(|i| f32::from_bits(0x7fc0_0000 | i as u32)).collect();
    let mut g = c.benchmark_group("host_nan_x1024");
    for (name, a) in [("finite", &x), ("all_nan", &nan)] {
        g.bench_function(format!("host_fma_f32_{name}"), |b| {
            b.iter(|| black_box(a).iter().zip(&x).map(|(&p, &q)| scalar::host_fma_f32(p, q, 1.0).to_bits()).fold(0, u32::wrapping_add))
        });
        g.bench_function(format!("pinned_add_f32_{name}"), |b| {
            b.iter(|| black_box(a).iter().zip(&x).map(|(&p, &q)| scalar::pin_nan2_f32(p, q, p + q).to_bits()).fold(0, u32::wrapping_add))
        });
    }
    g.finish();
}

/// The MMA increasing-K chain (`fpenv::fma_*_abt_increasing_k`) at the
/// corpus shapes; `with_nan` exercises the NaN re-pin post-pass.
fn mma_chain(c: &mut Criterion) {
    let mut g = c.benchmark_group("mma_chain");
    for (name, m, n, k, nan) in [
        ("f32_m128_n256_k16", 128, 256, 16, false),
        ("f32_m128_n256_k32", 128, 256, 32, false),
        ("f32_m64_n256_k16", 64, 256, 16, false),
        ("f32_m16_n8_k16_mma_sync", 16, 8, 16, false),
        ("f32_m128_n256_k16_with_nan", 128, 256, 16, true),
    ] {
        let a: Vec<f32> = (0..m * k).map(|i| if nan && i % 97 == 0 { f32::NAN } else { (i % 13) as f32 * 0.25 }).collect();
        let bt: Vec<f32> = (0..k * n).map(|i| (i % 7) as f32 - 3.0).collect();
        let init = vec![0.5_f32; m * n];
        g.bench_function(name, |bench| {
            bench.iter(|| {
                let mut out = init.clone();
                fpenv::fma_f32_abt_increasing_k(m, n, k, black_box(&a), &bt, &mut out).unwrap();
                out
            })
        });
    }
    let a: Vec<f64> = (0..16 * 4).map(|i| i as f64 * 0.5).collect();
    let bt: Vec<f64> = (0..4 * 8).map(|i| i as f64 - 3.0).collect();
    g.bench_function("f64_m16_n8_k4", |bench| {
        bench.iter(|| {
            let mut out = vec![0.0_f64; 16 * 8];
            fpenv::fma_f64_abt_increasing_k(16, 8, 4, black_box(&a), &bt, &mut out).unwrap();
            out
        })
    });
    g.finish();
}

/// `mma.sync.m16n8k16.f32.bf16` end to end (fragment gather, chain, scatter).
fn mma_sync(c: &mut Criterion) {
    let a: Vec<WarpValue<u32>> = (0..4).map(|r| std::array::from_fn(|l| 0x3f80_3f80 ^ ((l + r) as u32) << 4)).collect();
    let b: Vec<WarpValue<u32>> = (0..2).map(|r| std::array::from_fn(|l| 0x4000_3f80 ^ ((l * 3 + r) as u32) << 4)).collect();
    let mut g = c.benchmark_group("mma_sync");
    g.bench_function("m16n8k16_f32_bf16", |bench| {
        bench.iter(|| mma::sync::mma_sync_f32_b16(black_box(&a), &b, None, 16, mma::MatrixB16Type::Bf16).unwrap())
    });
    g.finish();
}

/// Warp butterfly reductions (`cuda_warp_reduce`, the `cta_reduce` lanes).
fn reductions(c: &mut Criterion) {
    let v: WarpValue<f32> = std::array::from_fn(|l| l as f32 * 1.25 - 7.0);
    let d: WarpValue<f64> = std::array::from_fn(|l| l as f64 * 1.25 - 7.0);
    let mut g = c.benchmark_group("warp_reduce");
    g.bench_function("sum_f32_w32", |b| b.iter(|| warp::warp_reduce_sum(WarpMask::ALL, black_box(&v), 32).unwrap()));
    g.bench_function("max_f32_w32", |b| b.iter(|| warp::warp_reduce_max(WarpMask::ALL, black_box(&v), 32).unwrap()));
    g.bench_function("sum_f64_w8", |b| b.iter(|| warp::warp_reduce_sum(WarpMask::ALL, black_box(&d), 8).unwrap()));
    g.finish();
}

/// v2 math builtins (deltas D9/D10) over 1024 values.
fn math(c: &mut Criterion) {
    let x: Vec<f32> = inputs().iter().map(|v| v.abs() * 0.01).collect();
    let mut g = c.benchmark_group("math_x1024");
    for (name, f) in [
        ("log1p_f32", scalar::log1p_f32 as fn(f32) -> f32),
        ("sigmoid_f32", scalar::sigmoid_f32),
        ("erf_f32", scalar::erf_f32),
        ("exp10_f32", scalar::exp10_f32),
        ("log10_f32", scalar::log10_f32),
    ] {
        g.bench_function(name, |b| b.iter(|| black_box(&x).iter().map(|&v| f(v)).sum::<f32>()));
    }
    g.finish();
}

criterion_group!(benches, conversions, host_nan_rule, mma_chain, mma_sync, reductions, math);
criterion_main!(benches);
