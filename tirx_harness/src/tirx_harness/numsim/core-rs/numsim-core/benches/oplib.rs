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
    type Case<'a> = (&'a str, Box<dyn Fn(&mut Vec<WarpValue<u64>>) + 'a>);
    let ops: Vec<Case> = vec![
        ("binary_add_u32", Box::new(|o| oplib::binary(BinOp::Add, Ty::U32, &au, &bu, o, mask).unwrap())),
        ("binary_add_f32", Box::new(|o| oplib::binary(BinOp::Add, Ty::F32, &af, &bf, o, mask).unwrap())),
        ("binary_mul_f32", Box::new(|o| oplib::binary(BinOp::Mul, Ty::F32, &af, &bf, o, mask).unwrap())),
        ("binary_add_f16", Box::new(|o| oplib::binary(BinOp::Add, Ty::F16, &ah, &ah, o, mask).unwrap())),
        ("binary_floordiv_s32", Box::new(|o| oplib::binary(BinOp::FloorDiv, Ty::S32, &au, &bu, o, mask).unwrap())),
        ("ternary_fma_f32", Box::new(|o| oplib::ternary(TerOp::Fma, Ty::F32, &af, &bf, &cf, o, mask).unwrap())),
        ("unary_neg_f32", Box::new(|o| oplib::unary(UnOp::Neg, Ty::F32, &af, o, mask).unwrap())),
        // Transcendentals (pure-Rust `libm` since delta D14; were the host C library).
        ("unary_exp_f32", Box::new(|o| oplib::unary(UnOp::Exp, Ty::F32, &af, o, mask).unwrap())),
        ("unary_log_f32", Box::new(|o| oplib::unary(UnOp::Log, Ty::F32, &cf, o, mask).unwrap())),
        ("unary_log2_f32", Box::new(|o| oplib::unary(UnOp::Log2, Ty::F32, &cf, o, mask).unwrap())),
        ("unary_tanh_f32", Box::new(|o| oplib::unary(UnOp::Tanh, Ty::F32, &bf, o, mask).unwrap())),
        ("cast_f32_s32", Box::new(|o| oplib::cast(Ty::F32, Ty::S32, Rounding::Default, false, &af, o, mask).unwrap())),
        ("cast_f32_f16", Box::new(|o| oplib::cast(Ty::F32, Ty::F16, Rounding::Default, false, &af, o, mask).unwrap())),
        ("cast_s32_f32", Box::new(|o| oplib::cast(Ty::S32, Ty::F32, Rounding::Default, false, &au, o, mask).unwrap())),
    ];
    for (name, f) in &ops {
        g.bench_function(*name, |b| b.iter(|| f(black_box(&mut out))));
    }
    // The Mega MoE medium mix (W4 profile: S32 add/mul ~9.5 M calls, int
    // casts ~2.7 M, predicate casts/logic ~0.5 M), at a partial and a full mask.
    let pa = lanes(|l| (l as u64 * 5) & 1);
    let pb = lanes(|l| (l as u64 / 3) & 1);
    let neg = lanes(|l| (l as u64).wrapping_mul(0x9e37_79b9) & 0xffff_ffff);
    for (suffix, m) in [("", mask), ("_full", WarpMask::ALL)] {
        let medium: Vec<Case> = vec![
            ("binary_add_s32", Box::new(|o| oplib::binary(BinOp::Add, Ty::S32, &au, &bu, o, m).unwrap())),
            ("binary_mul_s32", Box::new(|o| oplib::binary(BinOp::Mul, Ty::S32, &au, &bu, o, m).unwrap())),
            ("binary_floordiv_s32_b", Box::new(|o| oplib::binary(BinOp::FloorDiv, Ty::S32, &au, &bu, o, m).unwrap())),
            ("binary_or_pred", Box::new(|o| oplib::binary(BinOp::Or, Ty::PRED, &pa, &pb, o, m).unwrap())),
            ("binary_and_pred", Box::new(|o| oplib::binary(BinOp::And, Ty::PRED, &pa, &pb, o, m).unwrap())),
            ("cast_s32_u32", Box::new(|o| oplib::cast(Ty::S32, Ty::U32, Rounding::Default, false, &neg, o, m).unwrap())),
            ("cast_u32_s32", Box::new(|o| oplib::cast(Ty::U32, Ty::S32, Rounding::Default, false, &neg, o, m).unwrap())),
            ("cast_u32_u64", Box::new(|o| oplib::cast(Ty::U32, Ty::U64, Rounding::Default, false, &neg, o, m).unwrap())),
            ("cast_pred_u32", Box::new(|o| oplib::cast(Ty::PRED, Ty::U32, Rounding::Default, false, &pa, o, m).unwrap())),
            ("cast_u32_pred", Box::new(|o| oplib::cast(Ty::U32, Ty::PRED, Rounding::Default, false, &neg, o, m).unwrap())),
        ];
        for (name, f) in &medium {
            g.bench_function(format!("{name}{suffix}"), |b| b.iter(|| f(black_box(&mut out))));
        }
        g.bench_function(format!("compare_lt_s32{suffix}"), |b| {
            b.iter(|| oplib::compare(CmpOp::Lt, Ty::S32, black_box(&au), black_box(&bu), m).unwrap())
        });
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
    // (name, op, destination types, source types, source values)
    type Case = (&'static str, OpKey, Vec<Ty>, Vec<Ty>, Vec<WarpValue<u64>>);
    let cases: Vec<Case> = vec![
        ("mov_pack_b32x2", key("tirx.ptx.mov_pack_b32x2", &["b64"]), vec![Ty::U64], vec![Ty::U32, Ty::U32], vec![f(1.0), f(2.0)]),
        ("mov_unpack_b32x2", key("tirx.ptx.mov_unpack_b32x2", &["b64"]), vec![Ty::U32, Ty::U32], vec![Ty::U64], vec![f(1.0)]),
        ("fma_rn_f32", key("tirx.ptx.fma", &["rn", "f32"]), vec![Ty::F32], vec![Ty::F32; 3], vec![f(1.0), f(2.0), f(3.0)]),
        // Host NaN rule path (D8): every lane NaN, recomputed by the pinned rule.
        ("fma_rn_f32_all_nan", key("tirx.ptx.fma", &["rn", "f32"]), vec![Ty::F32], vec![Ty::F32; 3], vec![[0x7fc0_1234; 32], [0xffa0_0001; 32], f(3.0)]),
        ("add_rn_f32_all_nan", key("tirx.ptx.add", &["rn", "f32"]), vec![Ty::F32], vec![Ty::F32; 2], vec![[0x7fc0_1234; 32], [0xffa0_0001; 32]]),
        ("ex2_approx_ftz_f32", key("tirx.ptx.ex2", &["approx", "ftz", "f32"]), vec![Ty::F32], vec![Ty::F32], vec![f(0.5)]),
        ("cvt_rn_f16x2_f32", key("tirx.ptx.cvt_f16x2_f32", &["rn", "f16x2", "f32"]), vec![Ty::F16X2], vec![Ty::F32, Ty::F32], vec![f(1.0), f(2.0)]),
        ("cvt_rn_bf16x2_f32", key("tirx.ptx.cvt_bf16x2_f32", &["rn", "bf16x2", "f32"]), vec![Ty::BF16X2], vec![Ty::F32, Ty::F32], vec![f(1.0), f(2.0)]),
        ("cvt_rzi_s32_f32", key("tirx.ptx.cvt", &["rzi", "s32", "f32"]), vec![Ty::S32], vec![Ty::F32], vec![f(-3.0)]),
        ("cvt_rn_f32_f16", key("tirx.ptx.cvt", &["f32", "f16"]), vec![Ty::F32], vec![Ty::F16], vec![[0x3c01; 32]]),
        // fp16_bf16_gemm's other Ptx ops (W2-21).
        ("encode_instr_descriptor", key("tirx.cuda.tcgen05_encode_instr_descriptor", &["arg1=float32", "arg2=float16", "arg3=float16"]), vec![Ty::U32],
            vec![Ty::S32; 10], vec![[128; 32], [256; 32], [16; 32], [0; 32], [0; 32], [1; 32], [0; 32], [0; 32], [0; 32], [0; 32]]),
        ("encode_matrix_descriptor", key("tirx.cuda.tcgen05_encode_matrix_descriptor", &[]), vec![Ty::U64],
            vec![Ty::U64, Ty::S32, Ty::S32, Ty::S32], vec![[0x1000; 32], [1; 32], [64; 32], [2; 32]]),
        ("pack_u32x4", key("numsim.pack", &["ty=U32x4"]), vec![Ty::vector(Dtype::U32, 4)], vec![Ty::U32; 4], vec![[1; 32], [2; 32], [3; 32], [4; 32]]),
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

/// Engine-like tc_mma closures: arena-style byte stores with a validity
/// bitmap check and a merged span log per access (what `sched::run_mma` does).
struct Mem {
    smem: Vec<u8>,
    tmem: std::cell::RefCell<Vec<u8>>,
    valid: Vec<bool>,
    log: std::cell::RefCell<Vec<(u64, u64)>>,
}

impl Mem {
    fn note(&self, start: u64, len: u64) {
        let mut log = self.log.borrow_mut();
        if let Some(last) = log.last_mut() {
            if last.0 + last.1 == start {
                last.1 += len;
                return;
            }
        }
        log.push((start, len));
    }
}

/// One tcgen05.mma bench case: `(name, kind, a, b, d types, m, n, k, block_scale)`.
struct MmaCase {
    name: &'static str,
    kind: numsim_core::program::TcMmaKind,
    types: (&'static str, &'static str, &'static str),
    shape: (usize, usize, usize),
    /// Element bits of A/B in shared memory.
    bits: usize,
    /// Block-scaled: (scale dtype, block size).
    scaled: Option<(&'static str, u8)>,
    accumulate: bool,
    /// `cta_group` (2: the CTA-pair accumulator window, `cta2_runs`).
    cta_group: u8,
}

/// The tcgen05.mma kinds and shapes the canonical corpus issues (dense
/// f16/bf16 incl. M=64 Layout F, tf32, f8f6f4, i8, block-scaled mxf8f6f4 and
/// mxf4), through `tc_mma_ctas` with engine-like closures.
fn tc(c: &mut Criterion) {
    use numsim_core::arena::AllocId;
    use numsim_core::program::{CollectorOp, ConstId, Operand, TcA, TcMmaKind as K, TcgenMmaArgs};
    use numsim_core::sync::completion::TcgenMmaPayload;
    use numsim_oplib::tcgen05::encode::{
        encode_block_scaled_instr_descriptor_fields, encode_dense_instr_descriptor_fields, encode_matrix_descriptor,
    };
    let cases = [
        MmaCase { name: "f16_ss_m128_n256_k16_accumulate", kind: K::F16, types: ("float32", "bfloat16", "bfloat16"), shape: (128, 256, 16), bits: 16, scaled: None, accumulate: true, cta_group: 1 },
        MmaCase { name: "bf16_ss_m64_n256_k16_layout_f", kind: K::F16, types: ("float32", "bfloat16", "bfloat16"), shape: (64, 256, 16), bits: 16, scaled: None, accumulate: false, cta_group: 1 },
        MmaCase { name: "tf32_ss_m128_n256_k8", kind: K::Tf32, types: ("float32", "tf32", "tf32"), shape: (128, 256, 8), bits: 32, scaled: None, accumulate: true, cta_group: 1 },
        MmaCase { name: "f8f6f4_e4m3_ss_m128_n256_k32", kind: K::F8f6f4, types: ("float32", "float8_e4m3fn", "float8_e4m3fn"), shape: (128, 256, 32), bits: 8, scaled: None, accumulate: true, cta_group: 1 },
        MmaCase { name: "i8_ss_m128_n256_k32", kind: K::I8, types: ("int32", "int8", "int8"), shape: (128, 256, 32), bits: 8, scaled: None, accumulate: true, cta_group: 1 },
        MmaCase { name: "mxf8f6f4_e4m3_ss_m128_n256_k32", kind: K::MxF8f6f4, types: ("float32", "float8_e4m3fn", "float8_e4m3fn"), shape: (128, 256, 32), bits: 8, scaled: Some(("float8_e8m0fnu", 32)), accumulate: true, cta_group: 1 },
        MmaCase { name: "mxf4_e2m1_ss_m128_n256_k64", kind: K::MxF4, types: ("float32", "float4_e2m1fn", "float4_e2m1fn"), shape: (128, 256, 64), bits: 4, scaled: Some(("float8_e8m0fnu", 32)), accumulate: true, cta_group: 1 },
        MmaCase { name: "bf16_ss_cta2_m256_n256_k16_accumulate", kind: K::F16, types: ("float32", "bfloat16", "bfloat16"), shape: (256, 256, 16), bits: 16, scaled: None, accumulate: true, cta_group: 2 },
    ];
    let mut g = c.benchmark_group("tc_mma");
    for case in cases {
        let (m, n, k) = case.shape;
        let row_bytes = k * case.bits / 8;
        let mut smem = vec![0u8; 1 << 17];
        // K-major no-swizzle canonical layout (8x16B core matrices); small
        // finite codes in every format.
        let mut place = |start: usize, rows: usize| -> u64 {
            let chunks = row_bytes / 16;
            let (lbo, sbo) = (128usize, 128 * chunks);
            for row in 0..rows {
                for b in 0..row_bytes {
                    smem[start + (row % 8) * 16 + (row / 8) * sbo + (b / 16) * lbo + b % 16] = ((row * 7 + b) % 5) as u8;
                }
            }
            encode_matrix_descriptor(start as u32, (lbo >> 4) as i64, (sbo >> 4) as i64, 0)
        };
        let a_desc = place(0x1000, m);
        let b_desc = place(0x9000, n);
        let (d, a, b) = case.types;
        let idesc = match case.scaled {
            None => encode_dense_instr_descriptor_fields(d, a, b, m as i64, n as i64, k as i64, false, false, i64::from(case.cta_group), false, false, false, false),
            Some((s, _)) => encode_block_scaled_instr_descriptor_fields(d, a, b, s, s, m as i64, n as i64, k as i64, false, false, 1, false, false, false),
        }
        .unwrap() as u32;
        let op = Operand::Const(ConstId(0));
        let payload = TcgenMmaPayload {
            args: TcgenMmaArgs {
                kind: case.kind, cta_group: case.cta_group, d: op, a: TcA::Smem(op), b_desc: op, idesc: op, enable_input_d: op,
                ws: false, ws_b_buffer: 0, block_scale: case.scaled.map(|(_, block)| (op, op, block)), scale_input_d: None, sparse_meta: None,
                disable_output_lane: Vec::new(), collector_a: CollectorOp::None, collector_b: CollectorOp::None,
                ashift: false, lut_b: false, lut_b_addr: None, declared: None,
            },
            d_taddr: 0, a: a_desc, b_desc, idesc, enable_input_d: case.accumulate,
            // Scale tables after the 256 D columns (all-zero UE8M0 = 2^-127).
            scale_taddrs: case.scaled.map(|_| (320, 352)), scale_input_d: None, sparse_meta: None, disable_output_lane: Vec::new(),
            smem: vec![AllocId(0); usize::from(case.cta_group)], tmem: vec![AllocId(1); usize::from(case.cta_group)],
        };
        let tmem_bytes = 128 * 512 * 4;
        let mem = Mem { smem, tmem: std::cell::RefCell::new(vec![0u8; tmem_bytes]), valid: vec![true; tmem_bytes], log: Default::default() };
        let smem_read = |_cta: u32, a: u32, out: &mut [u8]| -> oplib::OpResult {
            let s = a as usize;
            out.copy_from_slice(&mem.smem[s..s + out.len()]);
            mem.note(u64::from(a), out.len() as u64);
            Ok(())
        };
        let tmem_read = |_cta: u32, lane: u32, col: u32, out: &mut [u8]| -> oplib::OpResult {
            let off = (lane as usize * 512 + col as usize) * 4;
            if mem.valid[off..off + out.len()].iter().any(|v| !v) {
                return Err(oplib::OpError::invalid("uninit"));
            }
            out.copy_from_slice(&mem.tmem.borrow()[off..off + out.len()]);
            mem.note(off as u64 | 1 << 40, out.len() as u64);
            Ok(())
        };
        let options = oplib::TcMmaOptions::default();
        g.bench_function(case.name, |bench| {
            bench.iter(|| {
                mem.log.borrow_mut().clear();
                let mut tmem_write = |_cta: u32, lane: u32, col: u32, data: &[u8]| -> oplib::OpResult {
                    let off = (lane as usize * 512 + col as usize) * 4;
                    mem.tmem.borrow_mut()[off..off + data.len()].copy_from_slice(data);
                    mem.note(off as u64 | 1 << 41, data.len() as u64);
                    Ok(())
                };
                oplib::tc_mma_ctas(black_box(&payload), &options, &smem_read, &tmem_read, &mut tmem_write, None)
                    .unwrap_or_else(|e| panic!("{}: {e}", case.name));
            })
        });
    }
    g.finish();
}

/// The Mega MoE medium MMA through the engine's window path (W4): block-scaled
/// `kind::mxf8f6f4`, A = e2m1 (padded 16-byte atoms), B = e4m3, cta_group::2,
/// M=256 N=16 K=32, accumulate. Operands are read through `TcMmaIo` windows
/// over all-valid allocations (as `sched::run_mma` builds them per MMA, no
/// observer), the accumulator is written through the callback.
fn tc_window(c: &mut Criterion) {
    use numsim_core::arena::{AllocId, BitSet};
    use numsim_core::program::{CollectorOp, ConstId, Operand, TcA, TcMmaKind as K, TcgenMmaArgs};
    use numsim_core::sync::completion::TcgenMmaPayload;
    use numsim_oplib::tcgen05::encode::{encode_block_scaled_instr_descriptor_fields, encode_matrix_descriptor};
    let (m, n, k) = (256usize, 16usize, 32usize);
    let row_bytes = 32;
    let mut smem = vec![0u8; 1 << 16];
    let mut place = |start: usize, rows: usize, fp4: bool| -> u64 {
        let (lbo, sbo) = (128usize, 256usize);
        for row in 0..rows {
            for b in 0..row_bytes {
                let v = ((row * 7 + b * 3) % 5) as u8;
                smem[start + (row % 8) * 16 + (row / 8) * sbo + (b / 16) * lbo + b % 16] = if fp4 { v | (v << 4) } else { v };
            }
        }
        encode_matrix_descriptor(start as u32, (lbo >> 4) as i64, (sbo >> 4) as i64, 0)
    };
    let a_desc = place(0x1000, m / 2, true);
    let b_desc = place(0x9000, n / 2, false);
    let idesc = encode_block_scaled_instr_descriptor_fields(
        "float32", "float4_e2m1fn", "float8_e4m3fn", "float8_e8m0fnu", "float8_e8m0fnu", m as i64, n as i64, k as i64, false, false, 2, false, false, false,
    )
    .unwrap() as u32;
    let op = Operand::Const(ConstId(0));
    let payload = TcgenMmaPayload {
        args: TcgenMmaArgs {
            kind: K::MxF8f6f4, cta_group: 2, d: op, a: TcA::Smem(op), b_desc: op, idesc: op, enable_input_d: op,
            ws: false, ws_b_buffer: 0, block_scale: Some((op, op, 32)), scale_input_d: None, sparse_meta: None,
            disable_output_lane: Vec::new(), collector_a: CollectorOp::None, collector_b: CollectorOp::None,
            ashift: false, lut_b: false, lut_b_addr: None, declared: None,
        },
        d_taddr: 0, a: a_desc, b_desc, idesc, enable_input_d: true,
        scale_taddrs: Some((320, 352)), scale_input_d: None, sparse_meta: None, disable_output_lane: Vec::new(),
        smem: vec![AllocId(0), AllocId(1)], tmem: vec![AllocId(2), AllocId(3)],
    };
    let tmem_bytes = 128 * 512 * 4;
    let smems = [smem.clone(), smem];
    let smem_valid = BitSet::new(smems[0].len() as u64, true);
    // Scale tables (columns 320.. and 352..) hold UE8M0 127 = 2^0 in every
    // byte, so the accumulator stays in the normal range like the real run's.
    let mut tmem0 = vec![0u8; tmem_bytes];
    for lane in 0..128 {
        for col in 320..384 {
            let off = (lane * 512 + col) * 4;
            tmem0[off..off + 4].fill(127);
        }
    }
    let tmems = [std::cell::RefCell::new(tmem0.clone()), std::cell::RefCell::new(tmem0)];
    let tmem_valid = BitSet::new(tmem_bytes as u64, true);
    let smem_cells: [std::cell::RefCell<Vec<u8>>; 2] = [std::cell::RefCell::new(smems[0].clone()), std::cell::RefCell::new(smems[1].clone())];
    let valid_cells = [std::cell::RefCell::new(smem_valid), std::cell::RefCell::new(tmem_valid)];
    let smem_read = |cta: u32, a: u32, out: &mut [u8]| -> oplib::OpResult {
        let s = a as usize;
        out.copy_from_slice(&smems[cta as usize][s..s + out.len()]);
        Ok(())
    };
    let tmem_read = |cta: u32, lane: u32, col: u32, out: &mut [u8]| -> oplib::OpResult {
        let off = (lane as usize * 512 + col as usize) * 4;
        out.copy_from_slice(&tmems[cta as usize].borrow()[off..off + out.len()]);
        Ok(())
    };
    let options = oplib::TcMmaOptions::default();
    let mut g = c.benchmark_group("tc_mma_window");
    g.bench_function("mxf8f6f4_e2m1_e4m3_cta2_m256_n16_k32_mega_moe", |bench| {
        bench.iter(|| {
            fn window<'b>(bytes: &'b std::cell::RefCell<Vec<u8>>, valid: &'b std::cell::RefCell<BitSet>) -> oplib::TcWindow<'b> {
                oplib::TcWindow { bytes: std::cell::Ref::map(bytes.borrow(), |v| &v[..]), valid: valid.borrow() }
            }
            let io = oplib::TcMmaIo {
                windows: std::cell::RefCell::new(Some([
                    [Some(window(&smem_cells[0], &valid_cells[0])), Some(window(&smem_cells[1], &valid_cells[0]))],
                    [Some(window(&tmems[0], &valid_cells[1])), Some(window(&tmems[1], &valid_cells[1]))],
                ])),
                reads: None,
            };
            let mut tmem_write = |cta: u32, lane: u32, col: u32, data: &[u8]| -> oplib::OpResult {
                let off = (lane as usize * 512 + col as usize) * 4;
                tmems[cta as usize].borrow_mut()[off..off + data.len()].copy_from_slice(data);
                Ok(())
            };
            oplib::tc_mma_ctas(black_box(&payload), &options, &smem_read, &tmem_read, &mut tmem_write, Some(&io))
                .unwrap_or_else(|e| panic!("mega moe window mma: {e}"));
        })
    });
    g.finish();
}

/// `resolve_ptx` itself (load-time cost per op), warp collectives the
/// dispatched tile reductions use (`shfl.sync`, `redux`).
fn dispatch(c: &mut Criterion) {
    use numsim_core::program::{ReduxOp, ShflMode};
    let mut g = c.benchmark_group("dispatch");
    let key = |name: &str, mods: &[&str]| OpKey { name: name.into(), mods: mods.iter().map(|m| m.to_string()).collect() };
    let direct = key("tirx.cuda.make_float2", &[]);
    g.bench_function("resolve_ptx_direct_form", |b| {
        b.iter(|| resolve_ptx(black_box(&direct), &[Ty::U64], &[Ty::F32, Ty::F32]).unwrap())
    });
    let boxed = key("tirx.ptx.cvt", &["rnd=rn", "dtype=f16", "atype=f32"]);
    g.bench_function("resolve_ptx_boxed_form_interned", |b| {
        b.iter(|| resolve_ptx(black_box(&boxed), &[Ty::F16], &[Ty::F32]).unwrap())
    });
    g.finish();
    let mut g = c.benchmark_group("warp_32_lanes");
    let src: WarpValue<u64> = std::array::from_fn(|l| u64::from((l as f32 * 0.37).to_bits()));
    let lane: WarpValue<u64> = std::array::from_fn(|l| (l ^ 1) as u64);
    let clamp = [0x1f_u64; 32];
    let members = [0xffff_ffff_u64; 32];
    for (name, mode) in [("shfl_sync_bfly", ShflMode::Bfly), ("shfl_sync_idx", ShflMode::Idx)] {
        g.bench_function(name, |b| {
            b.iter(|| oplib::shfl_sync(mode, black_box(&src), &lane, &clamp, &members, WarpMask::ALL).unwrap())
        });
    }
    let ints: WarpValue<u64> = std::array::from_fn(|l| l as u64 * 7);
    g.bench_function("redux_add_u32", |b| b.iter(|| oplib::redux(ReduxOp::Add, Ty::U32, black_box(&ints), WarpMask::ALL).unwrap()));
    g.bench_function("redux_max_f32", |b| b.iter(|| oplib::redux(ReduxOp::Max, Ty::F32, black_box(&src), WarpMask::ALL).unwrap()));
    g.finish();
}

fn tma(c: &mut Criterion) {
    use numsim_core::program::TmaMode;
    let mut g = c.benchmark_group("tma_plan");
    // A 2-D bf16 tile (64 x 128 box, 128B swizzle) of a 4096 x 4096 tensor.
    let map = oplib::TensorMapDesc {
        global_address: 0x1_0000_0000,
        rank: 2,
        elem: Some(Dtype::BF16),
        global_dim: [4096, 4096, 1, 1, 1],
        global_stride: [8192, 0, 0, 0, 0],
        box_dim: [64, 128, 1, 1, 1],
        element_stride: [1, 1, 1, 1, 1],
        swizzle: 3,
        ..Default::default()
    };
    let mut kblock = 0i64;
    g.bench_function("load_bf16_64x128_sw128", |b| {
        b.iter(|| {
            kblock = (kblock + 64) % 4096;
            black_box(oplib::tma_plan_dir(&map, oplib::TmaPlanDir::Load, TmaMode::Tile, &[kblock, 256], &[], 0x400).unwrap())
        })
    });
    // An OOB box is never cached: the planner's own cost.
    g.bench_function("load_bf16_64x128_sw128_oob_uncached", |b| {
        b.iter(|| black_box(oplib::tma_plan_dir(&map, oplib::TmaPlanDir::Load, TmaMode::Tile, &[4064, 256], &[], 0x400).unwrap()))
    });
    // FP4 (e2m1) interior box at 128-aligned inner origins: served by the
    // plan cache (sub-byte loads, W4).
    let fp4 = oplib::TensorMapDesc { elem: Some(Dtype::E2M1), box_dim: [128, 64, 1, 1, 1], global_stride: [2048, 0, 0, 0, 0], ..map.clone() };
    let mut fp4_k = 0i64;
    g.bench_function("load_e2m1_128x64_sw128", |b| {
        b.iter(|| {
            fp4_k = (fp4_k + 128) % 4096;
            black_box(oplib::tma_plan_dir(&fp4, oplib::TmaPlanDir::Load, TmaMode::Tile, &[fp4_k, 256], &[], 0x400).unwrap())
        })
    });
    // The Mega MoE medium FP4 weight load (128 x 128 box of a 2048 x 294912
    // packed e2m1 tensor, 128B swizzle), and the same box at an inner origin
    // that is not 128-aligned (planned directly).
    let moe = oplib::TensorMapDesc {
        elem: Some(Dtype::E2M1),
        global_dim: [2048, 294_912, 1, 1, 1],
        global_stride: [1024, 0, 0, 0, 0],
        box_dim: [128, 128, 1, 1, 1],
        ..map.clone()
    };
    let mut moe_row = 0i64;
    g.bench_function("load_e2m1_128x128_sw128_mega_moe", |b| {
        b.iter(|| {
            moe_row = (moe_row + 128) % 294_784;
            black_box(oplib::tma_plan_dir(&moe, oplib::TmaPlanDir::Load, TmaMode::Tile, &[(moe_row / 128 % 16) * 128, moe_row], &[], 0x400).unwrap())
        })
    });
    g.bench_function("load_e2m1_128x128_sw128_unaligned_uncached", |b| {
        b.iter(|| black_box(oplib::tma_plan_dir(&moe, oplib::TmaPlanDir::Load, TmaMode::Tile, &[64, 256], &[], 0x400).unwrap()))
    });
    let bytes = map.encode();
    g.bench_function("descriptor_decode", |b| b.iter(|| oplib::TensorMapDesc::decode(black_box(&bytes)).unwrap()));
    g.bench_function("descriptor_encode", |b| b.iter(|| black_box(&map).try_encode().unwrap()));
    g.finish();
    // `tcgen05.cp.32x128b.warpx4` (the Mega MoE scale copy): plan + pairs
    // over 8 rotating stage buffers (a memo hit per issue after warm-up).
    let mut g = c.benchmark_group("tcgen_cp");
    let mut stage = 0u64;
    g.bench_function("plan_pairs_32x128b_warpx4", |b| {
        b.iter(|| {
            stage = (stage + 1) % 8;
            let sdesc = numsim_oplib::tcgen05::encode::encode_matrix_descriptor((0x2000 + stage * 0x800) as u32, 8, 64, 0);
            let plan = oplib::tcgen_cp_plan(32, 128, 3, 0, sdesc, 0x40 + 4 * stage as u32, 1, oplib::TcArch::Sm100).unwrap();
            black_box(plan.pairs())
        })
    });
    // The issue form W13's handler builds today (plan + pairs + per-pair
    // shared-window / TMEM offsets) against the memoized `tcgen_cp_spans`.
    g.bench_function("issue_spans_via_plan_pairs_32x128b_warpx4", |b| {
        b.iter(|| {
            stage = (stage + 1) % 8;
            let sdesc = numsim_oplib::tcgen05::encode::encode_matrix_descriptor((0x2000 + stage * 0x800) as u32, 8, 64, 0);
            let plan = oplib::tcgen_cp_plan(32, 128, 3, 0, sdesc, 0x40 + 4 * stage as u32, 1, oplib::TcArch::Sm100).unwrap();
            let (sources, cells) = plan.pairs();
            let spans: Vec<(u64, u64, u32)> = sources
                .iter()
                .zip(&cells)
                .map(|(s, &(lane, col))| {
                    (u64::from(numsim_core::arena::addr::decode_shared(s.start as u32).1), s.len, numsim_core::arena::addr::tmem_byte_offset(lane, col) as u32)
                })
                .collect();
            black_box(spans)
        })
    });
    g.bench_function("issue_spans_memoized_32x128b_warpx4", |b| {
        b.iter(|| {
            stage = (stage + 1) % 8;
            let sdesc = numsim_oplib::tcgen05::encode::encode_matrix_descriptor((0x2000 + stage * 0x800) as u32, 8, 64, 0);
            black_box(oplib::tcgen_cp_spans(32, 128, 3, 0, sdesc, 0x40 + 4 * stage as u32, 1, oplib::TcArch::Sm100).unwrap())
        })
    });
    g.finish();
    let mut g = c.benchmark_group("tcgen_ldst");
    g.bench_function("map_32x32b_x64", |b| {
        b.iter(|| black_box(oplib::tcgen_ldst_map(numsim_core::program::TcShape::S32x32b, black_box(64), false, 1, 0x20_0040).unwrap()))
    });
    g.finish();
}

criterion_group!(benches, tir, ptx, tc, tc_window, tma, dispatch);
criterion_main!(benches);
