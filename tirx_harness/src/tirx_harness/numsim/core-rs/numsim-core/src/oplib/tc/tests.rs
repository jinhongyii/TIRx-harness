use std::cell::RefCell;
use std::collections::HashMap;

use super::super::{decode_instr_desc, decode_smem_desc, tc_mma, OpError, OpErrorKind, OpResult};
use crate::arena::AllocId;
use crate::dtype::Dtype;
use crate::program::{CollectorOp, ConstId, Operand, TcA, TcMmaKind, TcgenMmaArgs};
use crate::sync::completion::TcgenMmaPayload;
use numsim_oplib::cvt::{f32_to_bf16_bits, f32_to_float8_e4m3fn_bits, f32_to_fp16_bits};
use numsim_oplib::tcgen05::encode::{
    encode_block_scaled_instr_descriptor_fields, encode_dense_instr_descriptor_fields,
    encode_matrix_descriptor,
};
use numsim_oplib::tcgen05::layouts::layout_f_lane;

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

const D_COL: u32 = 64;
const A_COL: u32 = 0;
const SFA_COL: u32 = 32;
const SFB_COL: u32 = 40;

#[derive(Default)]
struct Machine {
    smem: Vec<u8>,
    tmem: HashMap<(u32, u32), [u8; 4]>,
}

impl Machine {
    fn new() -> Machine {
        Machine {
            smem: vec![0; 1 << 16],
            tmem: HashMap::new(),
        }
    }

    /// K-major no-swizzle canonical layout: 8x16B core matrices, LBO = 128
    /// (next 16B K chunk), SBO = 128 * chunks (next 8-row group). Returns the
    /// matrix descriptor.
    fn place(
        &mut self,
        start: u32,
        rows: usize,
        row_bytes: usize,
        byte: impl Fn(usize, usize) -> u8,
    ) -> u64 {
        let chunks = row_bytes / 16;
        let (lbo, sbo) = (128_usize, 128 * chunks);
        for row in 0..rows {
            for b in 0..row_bytes {
                let address =
                    start as usize + (row % 8) * 16 + (row / 8) * sbo + (b / 16) * lbo + b % 16;
                self.smem[address] = byte(row, b);
            }
        }
        encode_matrix_descriptor(start, (lbo >> 4) as i64, (sbo >> 4) as i64, 0)
    }

    fn set(&mut self, lane: u32, col: u32, bytes: [u8; 4]) {
        self.tmem.insert((lane, col), bytes);
    }

    fn get(&self, lane: u32, col: u32) -> Option<[u8; 4]> {
        self.tmem.get(&(lane, col)).copied()
    }

    fn run(&mut self, payload: &TcgenMmaPayload) -> OpResult {
        let smem = &self.smem;
        let tmem = RefCell::new(std::mem::take(&mut self.tmem));
        let result = {
            let read_smem = |address: u32, buf: &mut [u8]| -> OpResult {
                let start = address as usize;
                let bytes = smem
                    .get(start..start + buf.len())
                    .ok_or_else(|| OpError::invalid(format!("smem {start} out of range")))?;
                buf.copy_from_slice(bytes);
                Ok(())
            };
            let read_tmem = |lane: u32, col: u32, buf: &mut [u8]| -> OpResult {
                let cell = tmem.borrow().get(&(lane, col)).copied().unwrap_or([0; 4]);
                buf.copy_from_slice(&cell[..buf.len()]);
                Ok(())
            };
            let mut write_tmem = |lane: u32, col: u32, bytes: &[u8]| -> OpResult {
                tmem.borrow_mut()
                    .insert((lane, col), bytes.try_into().unwrap());
                Ok(())
            };
            tc_mma(payload, &read_smem, &read_tmem, &mut write_tmem)
        };
        self.tmem = tmem.into_inner();
        result
    }
}

fn op() -> Operand {
    Operand::Const(ConstId(0))
}

fn payload(
    kind: TcMmaKind,
    a: TcA,
    a_bits: u64,
    b_desc: u64,
    idesc: u32,
    enable_input_d: bool,
) -> TcgenMmaPayload {
    TcgenMmaPayload {
        args: TcgenMmaArgs {
            kind,
            cta_group: 1,
            d: op(),
            a,
            b_desc: op(),
            idesc: op(),
            enable_input_d: op(),
            ws: false,
            ws_b_buffer: 0,
            block_scale: None,
            scale_input_d: None,
            sparse_meta: None,
            disable_output_lane: Vec::new(),
            collector_a: CollectorOp::None,
            collector_b: CollectorOp::None,
            ashift: false,
            variant: None,
        },
        d_taddr: D_COL,
        a: a_bits,
        b_desc,
        idesc,
        enable_input_d,
        scale_taddrs: None,
        scale_input_d: None,
        sparse_meta: None,
        disable_output_lane: Vec::new(),
        smem: vec![AllocId(0)],
        tmem: vec![AllocId(1)],
    }
}

fn dense_idesc(d: &str, a: &str, b: &str, m: i64, n: i64, k: i64) -> u32 {
    encode_dense_instr_descriptor_fields(
        d, a, b, m, n, k, false, false, 1, false, false, false, false,
    )
    .unwrap() as u32
}

/// Small integers so every product and partial sum is exact in f32.
fn a_val(i: usize, k: usize) -> f32 {
    ((i * 3 + k * 5) % 7) as f32 - 3.0
}
fn b_val(j: usize, k: usize) -> f32 {
    ((j * 2 + k * 7) % 5) as f32 - 2.0
}

fn reference(m: usize, n: usize, k: usize, d: impl Fn(usize, usize) -> f32) -> Vec<f32> {
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            out[i * n + j] = d(i, j) + (0..k).map(|kk| a_val(i, kk) * b_val(j, kk)).sum::<f32>();
        }
    }
    out
}

/// D lane of row `i` (Layout D for M=128, Layout F for M=64).
fn d_lane(m: usize, i: usize) -> u32 {
    if m == 128 {
        i as u32
    } else {
        layout_f_lane(i).unwrap() as u32
    }
}

fn read_f32_tile(machine: &Machine, m: usize, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            out.push(f32::from_le_bytes(
                machine
                    .get(d_lane(m, i), D_COL + j as u32)
                    .expect("D cell written"),
            ));
        }
    }
    out
}

fn place_b16(
    machine: &mut Machine,
    start: u32,
    rows: usize,
    k: usize,
    value: impl Fn(usize, usize) -> f32,
    bf16: bool,
) -> u64 {
    machine.place(start, rows, k * 2, |row, byte| {
        let v = value(row, byte / 2);
        let bits = if bf16 {
            f32_to_bf16_bits(v)
        } else {
            f32_to_fp16_bits(v)
        };
        bits.to_le_bytes()[byte % 2]
    })
}

// ---------------------------------------------------------------------------
// Descriptor decoders
// ---------------------------------------------------------------------------

#[test]
fn smem_desc_round_trips_the_encoder() {
    for (swizzle, code) in [(0_i64, 0_u8), (1, 1), (2, 2), (3, 3), (4, 4)] {
        let bits = encode_matrix_descriptor(0x1a40, 0x20, 0x40, swizzle);
        let desc = decode_smem_desc(bits).unwrap();
        assert_eq!((desc.start, desc.lbo, desc.sbo), (0x1a40, 0x200, 0x400));
        assert_eq!(
            (desc.swizzle, desc.version, desc.base_offset, desc.lbo_mode),
            (code, 1, 0, 0)
        );
    }
    assert!(decode_smem_desc(0).is_err(), "version 0 is invalid");
    assert!(
        decode_smem_desc(encode_matrix_descriptor(0, 1, 1, 0) | (1 << 49)).is_err(),
        "base offset"
    );
}

#[test]
fn instr_desc_round_trips_the_encoders() {
    let d = decode_instr_desc(
        dense_idesc("float32", "bfloat16", "bfloat16", 128, 64, 16),
        TcMmaKind::F16,
    )
    .unwrap();
    assert_eq!(
        (d.m, d.n, d.a, d.b, d.d),
        (
            128,
            64,
            Some(Dtype::BF16),
            Some(Dtype::BF16),
            Some(Dtype::F32)
        )
    );
    let d = decode_instr_desc(
        dense_idesc("float16", "float16", "float16", 64, 8, 16),
        TcMmaKind::F16,
    )
    .unwrap();
    assert_eq!(
        (d.m, d.n, d.a, d.d),
        (64, 8, Some(Dtype::F16), Some(Dtype::F16))
    );
    let neg = encode_dense_instr_descriptor_fields(
        "float32", "float16", "float16", 128, 32, 16, true, false, 1, true, true, false, false,
    )
    .unwrap() as u32;
    let d = decode_instr_desc(neg, TcMmaKind::F16).unwrap();
    assert!(d.a_major_mn && !d.b_major_mn && d.negate_a && d.negate_b);
    let d = decode_instr_desc(
        dense_idesc("float32", "tf32", "tf32", 128, 16, 8),
        TcMmaKind::Tf32,
    )
    .unwrap();
    assert_eq!(
        (d.m, d.n, d.a, d.d),
        (128, 16, Some(Dtype::TF32), Some(Dtype::F32))
    );
    let d = decode_instr_desc(
        dense_idesc("float32", "float8_e4m3fn", "float8_e5m2", 128, 32, 32),
        TcMmaKind::F8f6f4,
    )
    .unwrap();
    assert_eq!(
        (d.a, d.b, d.d),
        (Some(Dtype::E4M3), Some(Dtype::E5M2), Some(Dtype::F32))
    );
    let d = decode_instr_desc(
        dense_idesc("float32", "float4_e2m1fn", "float6_e3m2fn", 64, 16, 32),
        TcMmaKind::F8f6f4,
    )
    .unwrap();
    assert_eq!((d.m, d.a, d.b), (64, Some(Dtype::E2M1), Some(Dtype::E3M2)));
    let d = decode_instr_desc(
        dense_idesc("int32", "int8", "uint8", 128, 32, 32),
        TcMmaKind::I8,
    )
    .unwrap();
    assert_eq!(
        (d.a, d.b, d.d),
        (Some(Dtype::S8), Some(Dtype::U8), Some(Dtype::S32))
    );
    // CTA-pair shapes are accepted (no cta_group in the contract).
    let d = decode_instr_desc(
        encode_dense_instr_descriptor_fields(
            "float32", "bfloat16", "bfloat16", 256, 256, 16, false, false, 2, false, false, false,
            false,
        )
        .unwrap() as u32,
        TcMmaKind::F16,
    )
    .unwrap();
    assert_eq!((d.m, d.n), (256, 256));

    let mx = |a: &str, b: &str, sf: &str, m: i64, n: i64, k: i64| {
        encode_block_scaled_instr_descriptor_fields(
            "float32", a, b, sf, sf, m, n, k, false, false, 1, false, false, false,
        )
        .unwrap() as u32
    };
    let d = decode_instr_desc(
        mx(
            "float8_e4m3fn",
            "float6_e2m3fn",
            "float8_e8m0fnu",
            128,
            64,
            32,
        ),
        TcMmaKind::MxF8f6f4,
    )
    .unwrap();
    assert_eq!(
        (d.m, d.n, d.a, d.b, d.scale_type),
        (
            128,
            64,
            Some(Dtype::E4M3),
            Some(Dtype::E2M3),
            Some(Dtype::UE8M0)
        )
    );
    let d = decode_instr_desc(
        mx(
            "float4_e2m1fn",
            "float4_e2m1fn",
            "float8_e8m0fnu",
            128,
            64,
            64,
        ),
        TcMmaKind::MxF4,
    )
    .unwrap();
    assert_eq!(
        (d.m, d.a, d.scale_type),
        (128, Some(Dtype::E2M1), Some(Dtype::UE8M0))
    );
    let d = decode_instr_desc(
        mx(
            "float4_e2m1fn",
            "float4_e2m1fn",
            "float8_e4m3fn",
            128,
            64,
            64,
        ),
        TcMmaKind::MxF4Nvf4,
    )
    .unwrap();
    assert_eq!(d.scale_type, Some(Dtype::UE4M3));

    // Kind/descriptor disagreement and reserved bits are rejected.
    assert!(decode_instr_desc(
        dense_idesc("float32", "tf32", "tf32", 128, 16, 8),
        TcMmaKind::F16
    )
    .is_err());
    assert!(decode_instr_desc(
        dense_idesc("float32", "bfloat16", "bfloat16", 128, 16, 16) | (1 << 23),
        TcMmaKind::F16
    )
    .is_err());
    assert!(decode_instr_desc(0, TcMmaKind::I8).is_err());
}

// ---------------------------------------------------------------------------
// MMA numerics
// ---------------------------------------------------------------------------

#[test]
fn bf16_ss_matches_hand_reference() {
    for (m, n) in [(128_usize, 16_usize), (64, 8)] {
        let k = 16;
        let mut machine = Machine::new();
        let a_desc = place_b16(&mut machine, 0x1000, m, k, a_val, true);
        let b_desc = place_b16(&mut machine, 0x8000, n, k, b_val, true);
        let idesc = dense_idesc(
            "float32", "bfloat16", "bfloat16", m as i64, n as i64, k as i64,
        );
        machine
            .run(&payload(
                TcMmaKind::F16,
                TcA::Smem(op()),
                a_desc,
                b_desc,
                idesc,
                false,
            ))
            .unwrap();
        assert_eq!(
            read_f32_tile(&machine, m, n),
            reference(m, n, k, |_, _| 0.0),
            "M={m} N={n}"
        );
    }
}

#[test]
fn enable_input_d_accumulates_with_scale() {
    let (m, n, k) = (128, 16, 16);
    let mut machine = Machine::new();
    let a_desc = place_b16(&mut machine, 0x1000, m, k, a_val, false);
    let b_desc = place_b16(&mut machine, 0x8000, n, k, b_val, false);
    let d_in = |i: usize, j: usize| (i as f32) - (j as f32) * 4.0;
    for i in 0..m {
        for j in 0..n {
            machine.set(i as u32, D_COL + j as u32, d_in(i, j).to_le_bytes());
        }
    }
    let idesc = dense_idesc(
        "float32", "float16", "float16", m as i64, n as i64, k as i64,
    );
    let mut p = payload(TcMmaKind::F16, TcA::Smem(op()), a_desc, b_desc, idesc, true);
    machine.run(&p).unwrap();
    let once = reference(m, n, k, d_in);
    assert_eq!(read_f32_tile(&machine, m, n), once);
    // scale-input-d = 1 halves D before the FMA chain.
    p.scale_input_d = Some(1);
    p.args.scale_input_d = Some(op());
    machine.run(&p).unwrap();
    assert_eq!(
        read_f32_tile(&machine, m, n),
        reference(m, n, k, |i, j| once[i * n + j] / 2.0)
    );
}

#[test]
fn f16_ts_with_disabled_output_lane() {
    let (m, n, k) = (64, 8, 16);
    let mut machine = Machine::new();
    // TMEM A (Layout F, M=64): eight packed words per row, two halves each.
    for i in 0..m {
        for w in 0..8 {
            let lo = f32_to_fp16_bits(a_val(i, 2 * w)) as u32;
            let hi = f32_to_fp16_bits(a_val(i, 2 * w + 1)) as u32;
            machine.set(
                layout_f_lane(i).unwrap() as u32,
                A_COL + w as u32,
                (lo | hi << 16).to_le_bytes(),
            );
        }
    }
    let b_desc = place_b16(&mut machine, 0x8000, n, k, b_val, false);
    let idesc = dense_idesc(
        "float32", "float16", "float16", m as i64, n as i64, k as i64,
    );
    let mut p = payload(
        TcMmaKind::F16,
        TcA::Tmem(op()),
        u64::from(A_COL),
        b_desc,
        idesc,
        false,
    );
    // Disable D lane 1 (row 1) and lane 32 (row 16).
    p.disable_output_lane = vec![1 << 1, 1, 0, 0];
    p.args.disable_output_lane = vec![op(); 4];
    machine.run(&p).unwrap();
    let expected = reference(m, n, k, |_, _| 0.0);
    for i in 0..m {
        for j in 0..n {
            let cell = machine.get(d_lane(m, i), D_COL + j as u32);
            if i == 1 || i == 16 {
                assert_eq!(cell, None, "disabled row {i} must not be written");
            } else {
                assert_eq!(f32::from_le_bytes(cell.unwrap()), expected[i * n + j]);
            }
        }
    }
}

#[test]
fn tf32_ss_truncates_storage_bits() {
    let (m, n, k) = (128, 8, 8);
    let mut machine = Machine::new();
    // Low mantissa bits are dropped (tf32 storage), not rounded.
    let a_desc = machine.place(0x1000, m, k * 4, |row, byte| {
        (f32::from_bits(a_val(row, byte / 4).to_bits() | 0x1fff)).to_le_bytes()[byte % 4]
    });
    let b_desc = machine.place(0x8000, n, k * 4, |row, byte| {
        b_val(row, byte / 4).to_le_bytes()[byte % 4]
    });
    let idesc = dense_idesc("float32", "tf32", "tf32", m as i64, n as i64, k as i64);
    machine
        .run(&payload(
            TcMmaKind::Tf32,
            TcA::Smem(op()),
            a_desc,
            b_desc,
            idesc,
            false,
        ))
        .unwrap();
    assert_eq!(
        read_f32_tile(&machine, m, n),
        reference(m, n, k, |_, _| 0.0)
    );
}

#[test]
fn f8_e4m3_ss_and_f16_destination() {
    let (m, n, k) = (128, 16, 32);
    let mut machine = Machine::new();
    let a_desc = machine.place(0x1000, m, k, |row, kk| {
        f32_to_float8_e4m3fn_bits(a_val(row, kk))
    });
    let b_desc = machine.place(0x8000, n, k, |row, kk| {
        f32_to_float8_e4m3fn_bits(b_val(row, kk))
    });
    let idesc = dense_idesc(
        "float32",
        "float8_e4m3fn",
        "float8_e4m3fn",
        m as i64,
        n as i64,
        k as i64,
    );
    machine
        .run(&payload(
            TcMmaKind::F8f6f4,
            TcA::Smem(op()),
            a_desc,
            b_desc,
            idesc,
            false,
        ))
        .unwrap();
    let expected = reference(m, n, k, |_, _| 0.0);
    assert_eq!(read_f32_tile(&machine, m, n), expected);
    // `.f16` D: one RNE conversion on store, high half zero.
    let idesc = dense_idesc(
        "float16",
        "float8_e4m3fn",
        "float8_e4m3fn",
        m as i64,
        n as i64,
        k as i64,
    );
    machine
        .run(&payload(
            TcMmaKind::F8f6f4,
            TcA::Smem(op()),
            a_desc,
            b_desc,
            idesc,
            false,
        ))
        .unwrap();
    let cell = machine.get(5, D_COL + 3).unwrap();
    let [lo, hi] = f32_to_fp16_bits(expected[5 * n + 3]).to_le_bytes();
    assert_eq!(cell, [lo, hi, 0, 0]);
}

#[test]
fn i8_ss_and_ts_exact_with_saturation() {
    let (m, n, k) = (128, 16, 32);
    let a_int = |i: usize, kk: usize| ((i * 37 + kk * 11) % 256) as i32 - 128; // s8
    let b_int = |j: usize, kk: usize| ((j * 13 + kk * 29) % 256) as i32; // u8
    let mut machine = Machine::new();
    let a_desc = machine.place(0x1000, m, k, |row, kk| a_int(row, kk) as i8 as u8);
    let b_desc = machine.place(0x8000, n, k, |row, kk| b_int(row, kk) as u8);
    let idesc = dense_idesc("int32", "int8", "uint8", m as i64, n as i64, k as i64);
    let dot = |i: usize, j: usize| {
        (0..k)
            .map(|kk| i64::from(a_int(i, kk)) * i64::from(b_int(j, kk)))
            .sum::<i64>()
    };
    let read = |machine: &Machine, i: usize, j: usize| {
        i32::from_le_bytes(machine.get(i as u32, D_COL + j as u32).unwrap())
    };
    machine
        .run(&payload(
            TcMmaKind::I8,
            TcA::Smem(op()),
            a_desc,
            b_desc,
            idesc,
            false,
        ))
        .unwrap();
    for i in 0..m {
        for j in 0..n {
            assert_eq!(i64::from(read(&machine, i, j)), dot(i, j));
        }
    }
    // TMEM A, accumulate into D = i32::MAX - 5 with and without .satfinite.
    for i in 0..m {
        for w in 0..8 {
            let word = (0..4).fold(0_u32, |acc, b| {
                acc | u32::from(a_int(i, 4 * w + b) as i8 as u8) << (8 * b)
            });
            machine.set(i as u32, A_COL + w as u32, word.to_le_bytes());
        }
        for j in 0..n {
            machine.set(i as u32, D_COL + j as u32, (i32::MAX - 5).to_le_bytes());
        }
    }
    machine
        .run(&payload(
            TcMmaKind::I8,
            TcA::Tmem(op()),
            u64::from(A_COL),
            b_desc,
            idesc,
            true,
        ))
        .unwrap();
    for i in 0..m {
        for j in 0..n {
            let wide = i64::from(i32::MAX - 5) + dot(i, j);
            assert_eq!(
                read(&machine, i, j),
                wide as i32,
                "wraps without .satfinite"
            );
            machine.set(i as u32, D_COL + j as u32, (i32::MAX - 5).to_le_bytes());
        }
    }
    let sat = encode_dense_instr_descriptor_fields(
        "int32", "int8", "uint8", m as i64, n as i64, k as i64, false, false, 1, false, false,
        true, false,
    )
    .unwrap() as u32;
    machine
        .run(&payload(
            TcMmaKind::I8,
            TcA::Tmem(op()),
            u64::from(A_COL),
            b_desc,
            sat,
            true,
        ))
        .unwrap();
    for i in 0..m {
        for j in 0..n {
            let wide = i64::from(i32::MAX - 5) + dot(i, j);
            assert_eq!(
                i64::from(read(&machine, i, j)),
                wide.clamp(i64::from(i32::MIN), i64::from(i32::MAX))
            );
        }
    }
}

/// Write a replicated SM100 UE8M0 scale (`mxf8_scale_layout`, four 32-lane
/// partitions) for `rows` rows at column `col`, byte 0.
fn place_replicated_scales(
    machine: &mut Machine,
    col: u32,
    rows: usize,
    scale: impl Fn(usize) -> u8,
) {
    for row in 0..rows {
        for partition in 0..4 {
            let lane = (partition * 32 + row % 32) as u32;
            let column = col + (row / 32) as u32;
            let mut cell = machine.get(lane, column).unwrap_or([0; 4]);
            cell[0] = scale(row);
            machine.set(lane, column, cell);
        }
    }
}

#[test]
fn mxf8f6f4_block_scaled_e4m3() {
    let (m, n, k) = (128, 8, 32);
    let mut machine = Machine::new();
    let a_desc = machine.place(0x1000, m, k, |row, kk| {
        f32_to_float8_e4m3fn_bits(a_val(row, kk))
    });
    let b_desc = machine.place(0x8000, n, k, |row, kk| {
        f32_to_float8_e4m3fn_bits(b_val(row, kk))
    });
    // UE8M0: 127 = 1.0, 128 = 2.0, 126 = 0.5.
    let sa = |row: usize| 126 + (row % 3) as u8;
    let sb = |row: usize| 127 + (row % 2) as u8;
    place_replicated_scales(&mut machine, SFA_COL, m, sa);
    place_replicated_scales(&mut machine, SFB_COL, n, sb);
    let idesc = encode_block_scaled_instr_descriptor_fields(
        "float32",
        "float8_e4m3fn",
        "float8_e4m3fn",
        "float8_e8m0fnu",
        "float8_e8m0fnu",
        m as i64,
        n as i64,
        k as i64,
        false,
        false,
        1,
        false,
        false,
        false,
    )
    .unwrap() as u32;
    let mut p = payload(
        TcMmaKind::MxF8f6f4,
        TcA::Smem(op()),
        a_desc,
        b_desc,
        idesc,
        false,
    );
    p.args.block_scale = Some((op(), op(), 32));
    p.scale_taddrs = Some((SFA_COL, SFB_COL));
    machine.run(&p).unwrap();
    let value = |bits: u8| 2_f32.powi(i32::from(bits) - 127);
    let mut expected = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            expected[i * n + j] = (0..k)
                .map(|kk| a_val(i, kk) * value(sa(i)) * (b_val(j, kk) * value(sb(j))))
                .sum();
        }
    }
    assert_eq!(read_f32_tile(&machine, m, n), expected);
}

#[test]
fn mxf4_block_scaled_e2m1() {
    let (m, n, k) = (128, 8, 64);
    // E2M1 codes 0..7 = 0, 0.5, 1, 1.5, 2, 3, 4, 6; 8..15 negative.
    let e2m1 = [0.0_f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let code_a = |i: usize, kk: usize| ((i + 3 * kk) % 16) as u8;
    let code_b = |j: usize, kk: usize| ((5 * j + kk) % 16) as u8;
    let value = |code: u8| {
        if code < 8 {
            e2m1[code as usize]
        } else {
            -e2m1[(code - 8) as usize]
        }
    };
    let mut machine = Machine::new();
    let pack = |code: &dyn Fn(usize, usize) -> u8, row: usize, byte: usize| {
        code(row, 2 * byte) | code(row, 2 * byte + 1) << 4
    };
    let a_desc = machine.place(0x1000, m, k / 2, |row, byte| pack(&code_a, row, byte));
    let b_desc = machine.place(0x8000, n, k / 2, |row, byte| pack(&code_b, row, byte));
    // Vec2x UE8M0 (block 32): row r, vector v at lane r % 32, column base + r / 32, byte v.
    let scale_bits = |row: usize, v: usize| 126 + ((row + v) % 3) as u8;
    for (col, rows) in [(SFA_COL, m), (SFB_COL, n)] {
        for row in 0..rows {
            let lane = (row % 32) as u32;
            let column = col + (row / 32) as u32;
            machine.set(lane, column, [scale_bits(row, 0), scale_bits(row, 1), 0, 0]);
        }
    }
    let idesc = encode_block_scaled_instr_descriptor_fields(
        "float32",
        "float4_e2m1fn",
        "float4_e2m1fn",
        "float8_e8m0fnu",
        "float8_e8m0fnu",
        m as i64,
        n as i64,
        k as i64,
        false,
        false,
        1,
        false,
        false,
        false,
    )
    .unwrap() as u32;
    let mut p = payload(
        TcMmaKind::MxF4,
        TcA::Smem(op()),
        a_desc,
        b_desc,
        idesc,
        false,
    );
    p.args.block_scale = Some((op(), op(), 32));
    p.scale_taddrs = Some((SFA_COL, SFB_COL));
    machine.run(&p).unwrap();
    let scale = |bits: u8| 2_f32.powi(i32::from(bits) - 127);
    let mut expected = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for kk in 0..k {
                let a = value(code_a(i, kk)) * scale(scale_bits(i, kk / 32));
                let b = value(code_b(j, kk)) * scale(scale_bits(j, kk / 32));
                acc = a.mul_add(b, acc);
            }
            expected[i * n + j] = acc;
        }
    }
    assert_eq!(read_f32_tile(&machine, m, n), expected);
}

#[test]
fn unmodeled_forms_fail_closed() {
    let mut machine = Machine::new();
    let idesc = dense_idesc("float32", "bfloat16", "bfloat16", 128, 16, 16);
    let base = payload(
        TcMmaKind::F16,
        TcA::Smem(op()),
        encode_matrix_descriptor(0, 8, 16, 0),
        encode_matrix_descriptor(0x8000, 8, 16, 0),
        idesc,
        false,
    );
    type Mutate = Box<dyn Fn(&mut TcgenMmaPayload)>;
    let cases: Vec<(&str, Mutate)> = vec![
        ("cta_group::2", Box::new(|p| p.args.cta_group = 2)),
        (".ws", Box::new(|p| p.args.ws = true)),
        (
            ".sp",
            Box::new(|p| {
                p.args.sparse_meta = Some(op());
                p.sparse_meta = Some(0);
            }),
        ),
        (
            ".collector",
            Box::new(|p| p.args.collector_a = CollectorOp::Fill),
        ),
        (".ashift", Box::new(|p| p.args.ashift = true)),
    ];
    for (form, mutate) in cases {
        let mut p = base.clone();
        mutate(&mut p);
        let error = machine.run(&p).unwrap_err();
        assert_eq!(error.kind, OpErrorKind::Unsupported, "{form}");
        assert!(error.message.contains(form), "{form}: {}", error.message);
    }
    assert!(machine.tmem.is_empty(), "rejected forms write nothing");
    // Invalid operand values are Invalid, closure errors keep their kind.
    let mut bad = base.clone();
    bad.idesc |= 1 << 23;
    assert_eq!(machine.run(&bad).unwrap_err().kind, OpErrorKind::Invalid);
    let mut far = base.clone();
    far.b_desc = encode_matrix_descriptor(0xfff0, 8, 16, 0);
    let error = machine.run(&far).unwrap_err();
    assert_eq!(error.kind, OpErrorKind::Invalid);
    assert!(error.message.contains("out of range"), "{}", error.message);
}
