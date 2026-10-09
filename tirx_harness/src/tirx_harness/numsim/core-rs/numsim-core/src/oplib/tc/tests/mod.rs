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
    tc_mma_ctas, OpError, OpErrorKind, OpResult, TcArch, TcMmaOptions, TcSpace,
};
use crate::arena::{addr, BitSet};
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

    /// Every MMA the tests run goes through both paths: the copying
    /// callbacks (whose result the test checks) and the resolved windows
    /// (`TcMmaIo`) under several validity/overlay mixes, which must give the
    /// same result, the same final TMEM and the same requested pieces.
    fn run_with(&mut self, payload: &TcgenMmaPayload, options: &TcMmaOptions) -> OpResult {
        let before = self.tmem.clone();
        let (result, pieces) = run_model(&self.smem, &mut self.tmem, payload, options, None);
        let pieces = coalesced(pieces);
        let key = |r: &OpResult| r.as_ref().map(|_| ()).map_err(|e| (e.kind, e.message.clone()));
        for mix in WindowMix::ALL {
            let mut tmem = before.clone();
            let (window_result, window_pieces) = run_model(&self.smem, &mut tmem, payload, options, Some(mix));
            let window_pieces = coalesced(window_pieces);
            assert_eq!(key(&window_result), key(&result), "windows {mix:?}: result differs");
            assert!(tmem == self.tmem, "windows {mix:?}: final TMEM differs");
            assert_eq!(window_pieces, pieces, "windows {mix:?}: accessed bytes differ");
        }
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
            // Batched reads span several cells of one lane.
            let read_tmem = |lane: u32, col: u32, buf: &mut [u8]| -> OpResult {
                for (i, chunk) in buf.chunks_mut(4).enumerate() {
                    let cell = tmem
                        .borrow()
                        .get(&(0, lane, col + i as u32))
                        .copied()
                        .unwrap_or([0; 4]);
                    chunk.copy_from_slice(&cell[..chunk.len()]);
                }
                Ok(())
            };
            let mut write_tmem = |lane: u32, col: u32, bytes: &[u8]| -> OpResult {
                for (i, chunk) in bytes.chunks(4).enumerate() {
                    tmem.borrow_mut()
                        .insert((0, lane, col + i as u32), chunk.try_into().unwrap());
                }
                Ok(())
            };
            tc_mma(payload, &read_smem, &read_tmem, &mut write_tmem)
        };
        self.tmem = tmem.into_inner();
        result
    }
}

/// A requested piece: `(kind, cta, offset, len)`; kind 0 = shared read,
/// 1 = TMEM read, 2 = TMEM write (TMEM offsets as `tmem_byte_offset`).
type Piece = (u8, u32, u64, u64);

/// The bytes the pieces touch, per (kind, cta), as sorted disjoint half-open
/// ranges: what the engine's span notes reduce to after `coalesce`
/// (adjacent and overlapping pieces merge), independent of piece order and
/// of how a contiguous run was split into pieces.
fn coalesced(mut pieces: Vec<Piece>) -> Vec<Piece> {
    pieces.sort_unstable();
    let mut out: Vec<Piece> = Vec::new();
    for (kind, cta, start, len) in pieces {
        if let Some(last) = out.last_mut() {
            if last.0 == kind && last.1 == cta && start <= last.2 + last.3 {
                last.3 = last.3.max(start + len - last.2);
                continue;
            }
        }
        out.push((kind, cta, start, len));
    }
    out
}

/// Validity/overlay mixes the window path is checked under.
#[derive(Clone, Copy, Debug)]
enum WindowMix {
    /// Every shared byte valid; TMEM valid exactly where a cell exists.
    AllValid,
    /// Every 7th shared byte and every 5th TMEM byte invalid (those pieces
    /// go to the callbacks).
    SparseInvalid,
    /// CTA 0's shared window and CTA 1's TMEM window absent (overlaid).
    Overlaid,
    /// No byte valid: every piece goes to the callbacks.
    NoneValid,
}

impl WindowMix {
    const ALL: [WindowMix; 4] = [WindowMix::AllValid, WindowMix::SparseInvalid, WindowMix::Overlaid, WindowMix::NoneValid];
}

/// Window images of the test model: per CTA shared window and TMEM bytes with
/// validity, behind one `RefCell` like the engine's arena.
struct Images {
    smem: Vec<(Vec<u8>, BitSet)>,
    tmem: Vec<(Vec<u8>, BitSet)>,
}

/// Run one MMA over the test model: callbacks only (`mix = None`), or with
/// `TcMmaIo` windows built from the model under `mix`. The callbacks never look
/// at validity (absent TMEM cells read as zero), exactly as before; the write
/// callback takes the images mutably, so a window still alive at a write
/// panics. Returns the result and every requested piece (callbacks record
/// after success, windows through `reads`).
fn run_model(
    smem: &[Vec<u8>; 2],
    tmem: &mut HashMap<(u32, u32, u32), [u8; 4]>,
    payload: &TcgenMmaPayload,
    options: &TcMmaOptions,
    mix: Option<WindowMix>,
) -> (OpResult, Vec<Piece>) {
    use super::super::{TcMmaIo, TcWindow};
    let bytes = addr::TMEM_BYTES;
    let images = RefCell::new(Images {
        smem: smem
            .iter()
            .map(|window| {
                let len = window.len() as u64;
                let mut valid = BitSet::new(len, !matches!(mix, Some(WindowMix::NoneValid)));
                if matches!(mix, Some(WindowMix::SparseInvalid)) {
                    for i in (0..len).step_by(7) {
                        valid.set_range(i, 1, false);
                    }
                }
                (window.clone(), valid)
            })
            .collect(),
        tmem: (0..2_u32)
            .map(|cta| {
                let mut image = vec![0_u8; bytes as usize];
                let mut valid = BitSet::new(bytes, false);
                if !matches!(mix, Some(WindowMix::NoneValid)) {
                    for (&(c, lane, col), cell) in tmem.iter() {
                        if c == cta && lane < addr::TMEM_LANES && col < addr::TMEM_COLS {
                            let at = addr::tmem_byte_offset(lane, col);
                            image[at as usize..at as usize + 4].copy_from_slice(cell);
                            valid.set_range(at, 4, true);
                        }
                    }
                }
                if matches!(mix, Some(WindowMix::SparseInvalid)) {
                    for i in (0..bytes).step_by(5) {
                        valid.set_range(i, 1, false);
                    }
                }
                (image, valid)
            })
            .collect(),
    });
    let model = RefCell::new(std::mem::take(tmem));
    let pieces = RefCell::new(Vec::new());
    let window_reads = RefCell::new(Vec::new());
    let result = {
        let read_smem = |cta: u32, address: u32, buf: &mut [u8]| -> OpResult {
            let start = address as usize;
            let bytes = smem[cta as usize]
                .get(start..start + buf.len())
                .ok_or_else(|| OpError::invalid(format!("smem {start} out of range")))?;
            buf.copy_from_slice(bytes);
            pieces.borrow_mut().push((0, cta, u64::from(address), buf.len() as u64));
            Ok(())
        };
        let read_tmem = |cta: u32, lane: u32, col: u32, buf: &mut [u8]| -> OpResult {
            for (i, chunk) in buf.chunks_mut(4).enumerate() {
                let cell = model.borrow().get(&(cta, lane, col + i as u32)).copied().unwrap_or([0; 4]);
                chunk.copy_from_slice(&cell[..chunk.len()]);
            }
            pieces.borrow_mut().push((1, cta, addr::tmem_byte_offset(lane, col), buf.len() as u64));
            Ok(())
        };
        let mut write_tmem = |cta: u32, lane: u32, col: u32, data: &[u8]| -> OpResult {
            // The engine writes through its arena: any live window would make
            // this borrow fail.
            let _arena = images.borrow_mut();
            for (i, chunk) in data.chunks(4).enumerate() {
                model.borrow_mut().insert((cta, lane, col + i as u32), chunk.try_into().unwrap());
            }
            pieces.borrow_mut().push((2, cta, addr::tmem_byte_offset(lane, col), data.len() as u64));
            Ok(())
        };
        let io = mix.map(|mix| {
            let overlaid = matches!(mix, WindowMix::Overlaid);
            let window = |space: usize, cta: usize| {
                let images = images.borrow();
                let held = std::cell::Ref::map(images, |images| if space == 0 { &images.smem[cta] } else { &images.tmem[cta] });
                let (bytes, valid) = std::cell::Ref::map_split(held, |(bytes, valid)| (&bytes[..], valid));
                TcWindow { bytes, valid }
            };
            TcMmaIo {
                windows: RefCell::new(Some([
                    [(!overlaid).then(|| window(0, 0)), Some(window(0, 1))],
                    [Some(window(1, 0)), (!overlaid).then(|| window(1, 1))],
                ])),
                reads: Some(&window_reads),
            }
        });
        tc_mma_ctas(payload, options, &read_smem, &read_tmem, &mut write_tmem, io.as_ref())
    };
    *tmem = model.into_inner();
    let mut all = pieces.into_inner();
    all.extend(window_reads.into_inner().into_iter().map(|(space, cta, offset, len)| {
        (if space == TcSpace::Shared { 0 } else { 1 }, cta, offset, len)
    }));
    (result, all)
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
            declared: None,
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
