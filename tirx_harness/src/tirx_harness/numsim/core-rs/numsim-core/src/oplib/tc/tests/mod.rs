//! `oplib::tc` tests: shared two-CTA harness and hand-reference helpers;
//! the cases live in the submodules.

mod cta2;
mod decode;
mod dense;
mod forms;
mod sparse;

use std::cell::RefCell;
use std::collections::HashMap;

use super::super::{
    decode_instr_desc, decode_instr_desc_for, decode_smem_desc, tc_collector_transition, tc_mma,
    tc_mma_ctas, OpError, OpErrorKind, OpResult, TcArch, TcMmaOptions,
};
use crate::arena::AllocId;
use crate::dtype::Dtype;
use crate::program::{CollectorOp, ConstId, Operand, TcA, TcMmaKind, TcgenMmaArgs};
use crate::sync::completion::TcgenMmaPayload;
use numsim_oplib::cvt::{f32_to_bf16_bits, f32_to_float8_e4m3fn_bits, f32_to_fp16_bits};
use numsim_oplib::tcgen05::encode::{
    encode_block_scaled_instr_descriptor_fields, encode_dense_instr_descriptor_fields,
    encode_matrix_descriptor,
};
use numsim_oplib::tcgen05::layouts::{
    layout_f_lane, sparse_metadata_location, SparseMetadataLayout,
};

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

const D_COL: u32 = 64;
const A_COL: u32 = 0;
const SFA_COL: u32 = 32;
const SFB_COL: u32 = 40;

/// Two CTAs' shared windows and TMEM; `cur` selects the CTA that
/// `place`/`set`/`get` address.
struct Machine {
    smem: [Vec<u8>; 2],
    tmem: HashMap<(u32, u32, u32), [u8; 4]>,
    cur: u32,
}

impl Machine {
    fn new() -> Machine {
        Machine {
            smem: [vec![0; 1 << 16], vec![0; 1 << 16]],
            tmem: HashMap::new(),
            cur: 0,
        }
    }

    fn on(&mut self, cta: u32) -> &mut Machine {
        self.cur = cta;
        self
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
        let chunks = row_bytes.div_ceil(16);
        let (lbo, sbo) = (128_usize, 128 * chunks);
        let smem = &mut self.smem[self.cur as usize];
        for row in 0..rows {
            for b in 0..row_bytes {
                let address =
                    start as usize + (row % 8) * 16 + (row / 8) * sbo + (b / 16) * lbo + b % 16;
                smem[address] = byte(row, b);
            }
        }
        encode_matrix_descriptor(start, (lbo >> 4) as i64, (sbo >> 4) as i64, 0)
    }

    fn set(&mut self, lane: u32, col: u32, bytes: [u8; 4]) {
        self.tmem.insert((self.cur, lane, col), bytes);
    }

    fn get(&self, lane: u32, col: u32) -> Option<[u8; 4]> {
        self.tmem.get(&(self.cur, lane, col)).copied()
    }

    fn run(&mut self, payload: &TcgenMmaPayload) -> OpResult {
        self.run_with(payload, &TcMmaOptions::default())
    }

    fn run_with(&mut self, payload: &TcgenMmaPayload, options: &TcMmaOptions) -> OpResult {
        let smem = &self.smem;
        let tmem = RefCell::new(std::mem::take(&mut self.tmem));
        let result = {
            let read_smem = |cta: u32, address: u32, buf: &mut [u8]| -> OpResult {
                let start = address as usize;
                let bytes = smem[cta as usize]
                    .get(start..start + buf.len())
                    .ok_or_else(|| OpError::invalid(format!("smem {start} out of range")))?;
                buf.copy_from_slice(bytes);
                Ok(())
            };
            let read_tmem = |cta: u32, lane: u32, col: u32, buf: &mut [u8]| -> OpResult {
                let cell = tmem
                    .borrow()
                    .get(&(cta, lane, col))
                    .copied()
                    .unwrap_or([0; 4]);
                buf.copy_from_slice(&cell[..buf.len()]);
                Ok(())
            };
            let mut write_tmem = |cta: u32, lane: u32, col: u32, bytes: &[u8]| -> OpResult {
                tmem.borrow_mut()
                    .insert((cta, lane, col), bytes.try_into().unwrap());
                Ok(())
            };
            tc_mma_ctas(payload, options, &read_smem, &read_tmem, &mut write_tmem)
        };
        self.tmem = tmem.into_inner();
        result
    }

    /// Through the single-CTA `tc_mma` wrapper.
    fn run_single(&mut self, payload: &TcgenMmaPayload) -> OpResult {
        let smem = &self.smem[0];
        let tmem = RefCell::new(std::mem::take(&mut self.tmem));
        let result = {
            let read_smem = |address: u32, buf: &mut [u8]| -> OpResult {
                let start = address as usize;
                buf.copy_from_slice(&smem[start..start + buf.len()]);
                Ok(())
            };
            let read_tmem = |lane: u32, col: u32, buf: &mut [u8]| -> OpResult {
                let cell = tmem
                    .borrow()
                    .get(&(0, lane, col))
                    .copied()
                    .unwrap_or([0; 4]);
                buf.copy_from_slice(&cell[..buf.len()]);
                Ok(())
            };
            let mut write_tmem = |lane: u32, col: u32, bytes: &[u8]| -> OpResult {
                tmem.borrow_mut()
                    .insert((0, lane, col), bytes.try_into().unwrap());
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
            lut_b: false,
            lut_b_addr: None,
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

// ---------------------------------------------------------------------------
// Ported forms: cta_group::2, .sp, .ws, .ashift, .ti16, .lut_b, collectors
// ---------------------------------------------------------------------------

const META_COL: u32 = 48;
const LUT_COL: u32 = 56;
/// The legal 2:4 metadata codes and the K indices each selects.
const CODES_2OF4: [(u8, [usize; 2]); 7] = [
    (0x4, [0, 1]),
    (0x8, [0, 2]),
    (0xc, [0, 3]),
    (0x9, [1, 2]),
    (0xd, [1, 3]),
    (0x6, [2, 1]),
    (0xe, [2, 3]),
];

fn cta2(mut p: TcgenMmaPayload) -> TcgenMmaPayload {
    p.args.cta_group = 2;
    p.smem.push(AllocId(2));
    p.tmem.push(AllocId(3));
    p
}

fn idesc_cta2(d: &str, a: &str, b: &str, m: i64, n: i64, k: i64) -> u32 {
    encode_dense_instr_descriptor_fields(
        d, a, b, m, n, k, false, false, 2, false, false, false, false,
    )
    .unwrap() as u32
}

/// `(cta, lane, column)` of D element `(i, j)` of a CTA-pair MMA: M=256 is
/// Layout D per CTA, M=128 two 64-lane banks splitting N (Layout E).
fn pair_cell(m: usize, n: usize, i: usize, j: usize) -> (u32, u32, u32) {
    let rows = m / 2;
    let (cta, r) = (i / rows, i % rows);
    if m == 256 {
        (cta as u32, r as u32, D_COL + j as u32)
    } else {
        let half = n / 2;
        (
            cta as u32,
            (r + 64 * (j / half)) as u32,
            D_COL + (j % half) as u32,
        )
    }
}

fn read_pair<T>(
    machine: &mut Machine,
    m: usize,
    n: usize,
    decode: impl Fn([u8; 4]) -> T,
) -> Vec<T> {
    let mut out = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            let (cta, lane, column) = pair_cell(m, n, i, j);
            out.push(decode(
                machine.on(cta).get(lane, column).expect("D cell written"),
            ));
        }
    }
    out
}

/// Write one 4-bit metadata code at `location`.
fn set_code(machine: &mut Machine, (lane, column, nibble): (usize, usize, usize), code: u8) {
    let mut word = u32::from_le_bytes(machine.get(lane as u32, column as u32).unwrap_or([0; 4]));
    word = word & !(0xf << (4 * nibble)) | u32::from(code) << (4 * nibble);
    machine.set(lane as u32, column as u32, word.to_le_bytes());
}

fn sparse(mut p: TcgenMmaPayload) -> TcgenMmaPayload {
    p.args.sparse_meta = Some(op());
    p.sparse_meta = Some(META_COL);
    p
}

fn ti16_bits(value: i32) -> u16 {
    (if value < 0 { 0x8000 } else { 0 }) | value.unsigned_abs() as u16
}
