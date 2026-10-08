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
use crate::arena::{addr, ValidityPolicy};
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
    /// callbacks (whose result the test checks) and the borrowed views
    /// (`TcViews`) under several validity/overlay mixes, which must give the
    /// same result, the same final TMEM and the same requested pieces.
    fn run_with(&mut self, payload: &TcgenMmaPayload, options: &TcMmaOptions) -> OpResult {
        let before = self.tmem.clone();
        let (result, pieces) = self.run_callbacks(payload, options);
        let key = |r: &OpResult| r.as_ref().map(|_| ()).map_err(|e| (e.kind, e.message.clone()));
        for mix in ViewMix::ALL {
            let (view_result, view_tmem, mut view_pieces, view_uninit) =
                run_model(&self.smem, &before, payload, options, mix, true);
            let (reference_result, reference_tmem, mut reference_pieces, reference_uninit) = if mix.matches_plain_model() {
                // The plain model reads absent cells as zero and keeps every
                // written cell; compare TMEM on the cells it holds.
                let mut tmem = self.tmem.clone();
                tmem.retain(|_, _| true);
                (result.clone(), tmem, pieces.clone(), None)
            } else {
                let (r, t, p, u) = run_model(&self.smem, &before, payload, options, mix, false);
                (r, t, p, Some(u))
            };
            assert_eq!(key(&view_result), key(&reference_result), "views {mix:?}: result differs");
            if mix.matches_plain_model() {
                for (cell, value) in &reference_tmem {
                    let viewed = view_tmem.get(cell).copied().unwrap_or([0; 4]);
                    assert_eq!(viewed, *value, "views {mix:?}: TMEM cell {cell:?} differs");
                }
            } else {
                assert!(view_tmem == reference_tmem, "views {mix:?}: final TMEM differs");
            }
            reference_pieces.sort_unstable();
            view_pieces.sort_unstable();
            assert_eq!(view_pieces, reference_pieces, "views {mix:?}: requested pieces differ");
            if let Some(reference_uninit) = reference_uninit {
                assert_eq!(view_uninit, reference_uninit, "views {mix:?}: uninitialized reads differ");
            }
        }
        result
    }

    /// The copying-callback path; returns the result and every requested
    /// piece `(space, cta, offset, len)` (TMEM offsets as `tmem_byte_offset`).
    fn run_callbacks(&mut self, payload: &TcgenMmaPayload, options: &TcMmaOptions) -> (OpResult, Vec<Piece>) {
        let pieces = RefCell::new(Vec::new());
        let smem = &self.smem;
        let tmem = RefCell::new(std::mem::take(&mut self.tmem));
        let result = {
            let read_smem = |cta: u32, address: u32, buf: &mut [u8]| -> OpResult {
                model_read_smem(smem, cta, address, buf)?;
                pieces.borrow_mut().push(piece(TcSpace::Shared, cta, u64::from(address), buf.len()));
                Ok(())
            };
            let read_tmem = |cta: u32, lane: u32, col: u32, buf: &mut [u8]| -> OpResult {
                model_read_tmem(&tmem.borrow(), cta, lane, col, buf)?;
                pieces.borrow_mut().push(piece(TcSpace::Tmem, cta, addr::tmem_byte_offset(lane, col), buf.len()));
                Ok(())
            };
            let mut write_tmem = |cta: u32, lane: u32, col: u32, bytes: &[u8]| -> OpResult {
                model_write_tmem(&mut tmem.borrow_mut(), cta, lane, col, bytes)?;
                pieces.borrow_mut().push(piece_write(cta, addr::tmem_byte_offset(lane, col), bytes.len()));
                Ok(())
            };
            tc_mma_ctas(payload, options, &read_smem, &read_tmem, &mut write_tmem, None)
        };
        self.tmem = tmem.into_inner();
        (result, pieces.into_inner())
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

/// A requested piece: `(kind, cta, offset, len)`, kind 0 = shared read,
/// 1 = TMEM read, 2 = TMEM write.
type Piece = (u8, u32, u64, u64);

fn piece(space: TcSpace, cta: u32, offset: u64, len: usize) -> Piece {
    (if space == TcSpace::Shared { 0 } else { 1 }, cta, offset, len as u64)
}

fn piece_write(cta: u32, offset: u64, len: usize) -> Piece {
    (2, cta, offset, len as u64)
}

fn model_read_smem(smem: &[Vec<u8>; 2], cta: u32, address: u32, buf: &mut [u8]) -> OpResult {
    let start = address as usize;
    let bytes = smem[cta as usize]
        .get(start..start + buf.len())
        .ok_or_else(|| OpError::invalid(format!("smem {start} out of range")))?;
    buf.copy_from_slice(bytes);
    Ok(())
}

/// Batched reads span several cells of one lane; absent cells read as zero.
fn model_read_tmem(tmem: &HashMap<(u32, u32, u32), [u8; 4]>, cta: u32, lane: u32, col: u32, buf: &mut [u8]) -> OpResult {
    for (i, chunk) in buf.chunks_mut(4).enumerate() {
        let cell = tmem.get(&(cta, lane, col + i as u32)).copied().unwrap_or([0; 4]);
        chunk.copy_from_slice(&cell[..chunk.len()]);
    }
    Ok(())
}

fn model_write_tmem(tmem: &mut HashMap<(u32, u32, u32), [u8; 4]>, cta: u32, lane: u32, col: u32, bytes: &[u8]) -> OpResult {
    for (i, chunk) in bytes.chunks(4).enumerate() {
        tmem.insert((cta, lane, col + i as u32), chunk.try_into().unwrap());
    }
    Ok(())
}

/// Validity/overlay mixes the view path is checked under.
#[derive(Clone, Copy, Debug)]
enum ViewMix {
    /// Every shared byte valid; TMEM valid exactly where a cell exists.
    AllValid,
    /// Every 7th shared byte invalid (pieces over them go to the callbacks).
    SparseInvalidShared,
    /// CTA 0's shared window and CTA 1's TMEM have no view (overlays).
    Overlaid,
    /// No shared byte valid: every shared piece falls back.
    NoValidShared,
    /// Every 5th TMEM byte invalid on top of absent cells, under a policy.
    InvalidTmem(ValidityPolicy),
}

impl ViewMix {
    const ALL: [ViewMix; 7] = [
        ViewMix::AllValid,
        ViewMix::SparseInvalidShared,
        ViewMix::Overlaid,
        ViewMix::NoValidShared,
        ViewMix::InvalidTmem(ValidityPolicy::Error),
        ViewMix::InvalidTmem(ValidityPolicy::Allow),
        ViewMix::InvalidTmem(ValidityPolicy::ZeroAndReport),
    ];

    /// Mixes whose callback reference is the plain test model (absent TMEM
    /// cells read as zero, i.e. `ZeroAndReport` over zero bytes).
    fn matches_plain_model(self) -> bool {
        !matches!(self, ViewMix::InvalidTmem(_))
    }

    fn policy(self) -> ValidityPolicy {
        match self {
            ViewMix::InvalidTmem(policy) => policy,
            _ => ValidityPolicy::ZeroAndReport,
        }
    }
}

type TmemModel = HashMap<(u32, u32, u32), [u8; 4]>;

/// What one modelled run produced: result, final TMEM (valid cells), every
/// requested piece (callbacks and views), and the uninitialized TMEM reads.
type ModelRun = (OpResult, TmemModel, Vec<Piece>, Vec<(u32, u64, u64)>);

/// The engine's TMEM range check and error (`tmem_of` in `run_mma`).
fn model_tmem_range(lane: u32, col: u32, len: usize) -> OpResult {
    if lane >= addr::TMEM_LANES || col >= addr::TMEM_COLS || u64::from(col) * 4 + len as u64 > u64::from(addr::TMEM_COLS) * 4 {
        return Err(OpError::invalid(format!("tmem cells ({lane}, {col}) + {len} bytes out of range")));
    }
    Ok(())
}

fn tmem_alloc(cta: u32) -> AllocId {
    AllocId(100 + cta)
}

/// An engine-like model over whole-allocation images: shared windows with
/// validity, TMEM images with validity and `mix`'s policy. With `use_views`
/// the MMA gets `TcViews` (TMEM callbacks then only see out-of-range pieces
/// for view CTAs); without, everything goes through copying callbacks that
/// apply the same validity rules as `Arena::read`.
fn run_model(
    smem: &[Vec<u8>; 2],
    tmem: &TmemModel,
    payload: &TcgenMmaPayload,
    options: &TcMmaOptions,
    mix: ViewMix,
    use_views: bool,
) -> ModelRun {
    use super::super::{TcShared, TcTmem, TcViews};
    use crate::arena::{ArenaError, BitSet};
    let bytes = addr::TMEM_BYTES;
    let policy = mix.policy();
    let overlaid = matches!(mix, ViewMix::Overlaid);
    let mut images: Vec<(Vec<u8>, BitSet)> = (0..2_u32)
        .map(|cta| {
            let mut image = vec![0_u8; bytes as usize];
            let mut valid = BitSet::new(bytes, false);
            for (&(c, lane, col), cell) in tmem {
                if c == cta && lane < addr::TMEM_LANES && col < addr::TMEM_COLS {
                    let at = addr::tmem_byte_offset(lane, col);
                    image[at as usize..at as usize + 4].copy_from_slice(cell);
                    valid.set_range(at, 4, true);
                }
            }
            if matches!(mix, ViewMix::InvalidTmem(_)) {
                for i in (0..bytes).step_by(5) {
                    valid.set_range(i, 1, false);
                }
            }
            (image, valid)
        })
        .collect();
    let smem_valid: Vec<BitSet> = smem
        .iter()
        .map(|window| {
            let len = window.len() as u64;
            let mut valid = BitSet::new(len, !matches!(mix, ViewMix::NoValidShared));
            if matches!(mix, ViewMix::SparseInvalidShared) {
                for i in (0..len).step_by(7) {
                    valid.set_range(i, 1, false);
                }
            }
            valid
        })
        .collect();
    // TMEM served by callbacks: every CTA without views, the overlaid CTA with.
    let callback_cta = |cta: u32| !use_views || (overlaid && cta == 1);
    let (view_images, callback_images): (Vec<_>, Vec<_>) =
        images.iter_mut().enumerate().partition(|(cta, _)| !callback_cta(*cta as u32));
    let callback_images: RefCell<HashMap<u32, &mut (Vec<u8>, BitSet)>> =
        RefCell::new(callback_images.into_iter().map(|(cta, image)| (cta as u32, image)).collect());
    let pieces = RefCell::new(Vec::new());
    let callback_uninit = RefCell::new(Vec::new());
    let mut view_reads = Vec::new();
    let mut view_writes = Vec::new();
    let mut view_uninit = Vec::new();
    let result = {
        let read_smem = |cta: u32, address: u32, buf: &mut [u8]| -> OpResult {
            model_read_smem(smem, cta, address, buf)?;
            pieces.borrow_mut().push(piece(TcSpace::Shared, cta, u64::from(address), buf.len()));
            Ok(())
        };
        let read_tmem = |cta: u32, lane: u32, col: u32, buf: &mut [u8]| -> OpResult {
            model_tmem_range(lane, col, buf.len())?;
            let mut images = callback_images.borrow_mut();
            let (image, valid) = images
                .get_mut(&cta)
                .unwrap_or_else(|| panic!("in-range TMEM callback read on view CTA {cta}"));
            let start = addr::tmem_byte_offset(lane, col);
            let len = buf.len() as u64;
            if let Some(first) = valid.first_clear(start, len) {
                callback_uninit.borrow_mut().push((cta, start, len));
                if policy == ValidityPolicy::Error {
                    return Err(OpError::invalid(ArenaError::Uninit { alloc: tmem_alloc(cta), offset: first }.to_string()));
                }
            }
            buf.copy_from_slice(&image[start as usize..(start + len) as usize]);
            if policy == ValidityPolicy::ZeroAndReport {
                for (i, byte) in buf.iter_mut().enumerate() {
                    if !valid.get(start + i as u64) {
                        *byte = 0;
                    }
                }
            }
            pieces.borrow_mut().push(piece(TcSpace::Tmem, cta, start, buf.len()));
            Ok(())
        };
        let mut write_tmem = |cta: u32, lane: u32, col: u32, data: &[u8]| -> OpResult {
            model_tmem_range(lane, col, data.len())?;
            let mut images = callback_images.borrow_mut();
            let (image, valid) = images
                .get_mut(&cta)
                .unwrap_or_else(|| panic!("in-range TMEM callback write on view CTA {cta}"));
            let start = addr::tmem_byte_offset(lane, col);
            image[start as usize..start as usize + data.len()].copy_from_slice(data);
            valid.set_range(start, data.len() as u64, true);
            pieces.borrow_mut().push(piece_write(cta, start, data.len()));
            Ok(())
        };
        let views = use_views.then(|| {
            let mut tmem_views: Vec<Option<TcTmem<'_>>> = vec![None, None];
            for (cta, (image, valid)) in view_images {
                tmem_views[cta] = Some(TcTmem { alloc: tmem_alloc(cta as u32), bytes: image, valid });
            }
            TcViews {
                smem: (0..2)
                    .map(|cta| (!(overlaid && cta == 0)).then(|| TcShared { bytes: &smem[cta], valid: &smem_valid[cta] }))
                    .collect(),
                tmem: tmem_views,
                reads: Some(&mut view_reads),
                writes: &mut view_writes,
                policy,
                uninit: &mut view_uninit,
            }
        });
        tc_mma_ctas(payload, options, &read_smem, &read_tmem, &mut write_tmem, views)
    };
    let mut all = pieces.into_inner();
    all.extend(view_reads.iter().map(|&(space, cta, offset, len)| piece(space, cta, offset, len as usize)));
    all.extend(view_writes.iter().map(|&(cta, offset, len)| piece_write(cta, offset, len as usize)));
    let mut uninit = callback_uninit.into_inner();
    uninit.extend(view_uninit);
    let mut out = TmemModel::new();
    for (cta, (image, valid)) in images.iter().enumerate() {
        for lane in 0..addr::TMEM_LANES {
            for col in 0..addr::TMEM_COLS {
                let at = addr::tmem_byte_offset(lane, col);
                if (0..4).any(|i| valid.get(at + i)) {
                    let cell: [u8; 4] = image[at as usize..at as usize + 4].try_into().unwrap();
                    out.insert((cta as u32, lane, col), cell);
                }
            }
        }
    }
    (result, out, all, uninit)
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
