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

fn tc(c: &mut Criterion) {
    use numsim_core::arena::AllocId;
    use numsim_core::program::{CollectorOp, ConstId, Operand, TcA, TcMmaKind, TcgenMmaArgs};
    use numsim_core::sync::completion::TcgenMmaPayload;
    use numsim_oplib::tcgen05::encode::{encode_dense_instr_descriptor_fields, encode_matrix_descriptor};
    let mut g = c.benchmark_group("tc_mma");
    let (m, n, k) = (128usize, 256usize, 16usize);
    let mut smem = vec![0u8; 1 << 17];
    // K-major no-swizzle canonical layout (8x16B core matrices).
    let mut place = |start: usize, rows: usize| -> u64 {
        let row_bytes = k * 2;
        let chunks = row_bytes / 16;
        let (lbo, sbo) = (128usize, 128 * chunks);
        for row in 0..rows {
            for b in 0..row_bytes {
                smem[start + (row % 8) * 16 + (row / 8) * sbo + (b / 16) * lbo + b % 16] = ((row * 7 + b) % 13) as u8;
            }
        }
        encode_matrix_descriptor(start as u32, (lbo >> 4) as i64, (sbo >> 4) as i64, 0)
    };
    let a_desc = place(0x1000, m);
    let b_desc = place(0x8000, n);
    let idesc = encode_dense_instr_descriptor_fields("float32", "bfloat16", "bfloat16", m as i64, n as i64, k as i64, false, false, 1, false, false, false, false).unwrap() as u32;
    let op = Operand::Const(ConstId(0));
    let payload = TcgenMmaPayload {
        args: TcgenMmaArgs {
            kind: TcMmaKind::F16, cta_group: 1, d: op, a: TcA::Smem(op), b_desc: op, idesc: op, enable_input_d: op,
            ws: false, ws_b_buffer: 0, block_scale: None, scale_input_d: None, sparse_meta: None,
            disable_output_lane: Vec::new(), collector_a: CollectorOp::None, collector_b: CollectorOp::None,
            ashift: false, lut_b: false, lut_b_addr: None,
        },
        d_taddr: 0, a: a_desc, b_desc, idesc, enable_input_d: true,
        scale_taddrs: None, scale_input_d: None, sparse_meta: None, disable_output_lane: Vec::new(),
        smem: vec![AllocId(0)], tmem: vec![AllocId(1)],
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
    g.bench_function("f16_ss_m128_n256_k16_accumulate", |b| {
        b.iter(|| {
            mem.log.borrow_mut().clear();
            let mut tmem_write = |_cta: u32, lane: u32, col: u32, data: &[u8]| -> oplib::OpResult {
                let off = (lane as usize * 512 + col as usize) * 4;
                mem.tmem.borrow_mut()[off..off + data.len()].copy_from_slice(data);
                mem.note(off as u64 | 1 << 41, data.len() as u64);
                Ok(())
            };
            oplib::tc_mma_ctas(black_box(&payload), &options, &smem_read, &tmem_read, &mut tmem_write).unwrap();
        })
    });
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
    g.finish();
    let mut g = c.benchmark_group("tcgen_ldst");
    g.bench_function("map_32x32b_x64", |b| {
        b.iter(|| black_box(oplib::tcgen_ldst_map(numsim_core::program::TcShape::S32x32b, black_box(64), false, 1, 0x20_0040).unwrap()))
    });
    g.finish();
}

criterion_group!(benches, tir, ptx, tc, tma);
criterion_main!(benches);
