//! OpLib ALU microbenches (W4): cost of one 32-lane call of the contract
//! entry points the interpreter dispatches to. `cargo bench -p numsim-core
//! --bench oplib`. Targets (W2-6): <= 40 ns per f32 add, <= 60 ns per compare.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use numsim_core::dtype::{Dtype, Ty};
use numsim_core::oplib::{self, resolve_ptx, PtxIo};
use numsim_core::program::{BinOp, CmpOp, OpKey, Rounding, TerOp, UnOp};
use numsim_core::value::{WarpMask, WarpValue};

fn lanes(f: impl Fn(usize) -> u64) -> Vec<WarpValue<u64>> {
    vec![std::array::from_fn(f)]
}

fn f32s(seed: f32) -> Vec<WarpValue<u64>> {
    lanes(|l| u64::from((seed + l as f32 * 0.37).to_bits()))
}

fn tir(c: &mut Criterion) {
    let mut g = c.benchmark_group("tir_32_lanes");
    let mask = WarpMask(0xffff_fff7);
    let (au, bu) = (lanes(|l| l as u64 * 7 + 3), lanes(|l| 1000 - l as u64));
    let (af, bf, cf) = (f32s(1.5), f32s(-2.25), f32s(0.5));
    let ah = lanes(|l| 0x3c00 + l as u64);
    let mut out = vec![[0u64; 32]];
    let ops: Vec<(&str, Box<dyn Fn(&mut Vec<WarpValue<u64>>)>)> = vec![
        ("binary_add_u32", Box::new(|o| oplib::binary(BinOp::Add, Ty::U32, &au, &bu, o, mask).unwrap())),
        ("binary_add_f32", Box::new(|o| oplib::binary(BinOp::Add, Ty::F32, &af, &bf, o, mask).unwrap())),
        ("binary_mul_f32", Box::new(|o| oplib::binary(BinOp::Mul, Ty::F32, &af, &bf, o, mask).unwrap())),
        ("binary_add_f16", Box::new(|o| oplib::binary(BinOp::Add, Ty::F16, &ah, &ah, o, mask).unwrap())),
        ("binary_floordiv_s32", Box::new(|o| oplib::binary(BinOp::FloorDiv, Ty::S32, &au, &bu, o, mask).unwrap())),
        ("ternary_fma_f32", Box::new(|o| oplib::ternary(TerOp::Fma, Ty::F32, &af, &bf, &cf, o, mask).unwrap())),
        ("unary_neg_f32", Box::new(|o| oplib::unary(UnOp::Neg, Ty::F32, &af, o, mask).unwrap())),
        ("cast_f32_s32", Box::new(|o| oplib::cast(Ty::F32, Ty::S32, Rounding::Default, false, &af, o, mask).unwrap())),
        ("cast_f32_f16", Box::new(|o| oplib::cast(Ty::F32, Ty::F16, Rounding::Default, false, &af, o, mask).unwrap())),
        ("cast_s32_f32", Box::new(|o| oplib::cast(Ty::S32, Ty::F32, Rounding::Default, false, &au, o, mask).unwrap())),
    ];
    for (name, f) in &ops {
        g.bench_function(*name, |b| b.iter(|| f(black_box(&mut out))));
    }
    g.bench_function("compare_lt_u32", |b| {
        b.iter(|| oplib::compare(CmpOp::Lt, Ty::U32, black_box(&au), black_box(&bu), mask).unwrap())
    });
    g.bench_function("compare_lt_f32", |b| {
        b.iter(|| oplib::compare(CmpOp::Lt, Ty::F32, black_box(&af), black_box(&bf), mask).unwrap())
    });
    g.finish();
}

fn ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("ptx_32_lanes");
    let mask = WarpMask::ALL;
    let key = |name: &str, mods: &[&str]| OpKey { name: name.into(), mods: mods.iter().map(|m| m.to_string()).collect() };
    let f = |seed: f32| -> WarpValue<u64> { std::array::from_fn(|l| u64::from((seed + l as f32 * 0.37).to_bits())) };
    let cases: Vec<(&str, OpKey, Vec<Ty>, Vec<Ty>, Vec<WarpValue<u64>>)> = vec![
        ("mov_pack_b32x2", key("tirx.ptx.mov_pack_b32x2", &["b64"]), vec![Ty::U64], vec![Ty::U32, Ty::U32], vec![f(1.0), f(2.0)]),
        ("mov_unpack_b32x2", key("tirx.ptx.mov_unpack_b32x2", &["b64"]), vec![Ty::U32, Ty::U32], vec![Ty::U64], vec![f(1.0)]),
        ("fma_rn_f32", key("tirx.ptx.fma", &["rn", "f32"]), vec![Ty::F32], vec![Ty::F32; 3], vec![f(1.0), f(2.0), f(3.0)]),
        ("ex2_approx_ftz_f32", key("tirx.ptx.ex2", &["approx", "ftz", "f32"]), vec![Ty::F32], vec![Ty::F32], vec![f(0.5)]),
        ("cvt_rn_f16x2_f32", key("tirx.ptx.cvt_f16x2_f32", &["rn", "f16x2", "f32"]), vec![Ty::F16X2], vec![Ty::F32, Ty::F32], vec![f(1.0), f(2.0)]),
        ("cvt_rn_bf16x2_f32", key("tirx.ptx.cvt_bf16x2_f32", &["rn", "bf16x2", "f32"]), vec![Ty::BF16X2], vec![Ty::F32, Ty::F32], vec![f(1.0), f(2.0)]),
        ("cvt_rzi_s32_f32", key("tirx.ptx.cvt", &["rzi", "s32", "f32"]), vec![Ty::S32], vec![Ty::F32], vec![f(-3.0)]),
        ("cvt_rn_f32_f16", key("tirx.ptx.cvt", &["f32", "f16"]), vec![Ty::F32], vec![Ty::F16], vec![[0x3c01; 32]]),
    ];
    for (name, k, dt, st, srcs) in cases {
        let op = resolve_ptx(&k, &dt, &st).unwrap_or_else(|e| panic!("{name}: {e}"));
        let mut dsts = vec![[0u64; 32]; dt.iter().map(|t| t.slots() as usize).sum()];
        g.bench_function(name, |b| {
            b.iter(|| {
                let mut io = PtxIo { dsts: &mut dsts, dst_tys: &dt, srcs: black_box(&srcs), src_tys: &st, mask };
                op.call(&mut io).unwrap();
            })
        });
    }
    g.finish();
}

criterion_group!(benches, tir, ptx);
criterion_main!(benches);
