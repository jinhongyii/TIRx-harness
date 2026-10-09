//! `tcgen05.ld` / `tcgen05.st` / `tcgen05.cp` maps.

use super::super::{OpError, OpResult, TcArch};
use crate::arena::ByteSpan;
use crate::dtype::Dtype;
use crate::program::{ReduxOp, TcShape};
use numsim_oplib::tcgen05::layouts::{
    cp_decode_word, cp_destination_lanes, cp_rows_words, ldst_location, tmem_address, LdstShape,
};
use numsim_oplib::tcgen05::ld::{ld_destination_count, spcompress_lane, LdReduction};
use numsim_oplib::tcgen05::smem_desc::{
    cp_source_span, decode_matrix_descriptor_for_layout, MatrixDescriptorLayout, SharedWindow,
};

const WARP: usize = 32;
/// Shared-window addresses: the descriptor start is a window address, so the
/// window is one flat region based at 0 (as in `oplib::tc_mma_ctas`).
const WINDOW: SharedWindow = SharedWindow::whole(0, 1 << 32);

fn ldst_shape(shape: TcShape) -> LdstShape {
    match shape {
        TcShape::S32x32b => LdstShape::Shape32x32b,
        TcShape::S16x64b => LdstShape::Shape16x64b,
        TcShape::S16x128b => LdstShape::Shape16x128b,
        TcShape::S16x256b => LdstShape::Shape16x256b,
        TcShape::S16x32bx2 { split_off } => LdstShape::Shape16x32bx2(split_off as usize),
    }
}

fn narrow<T: TryFrom<usize>>(value: usize, what: &str) -> OpResult<T> {
    T::try_from(value).map_err(|_| OpError::invalid(format!("{what} {value} overflows")))
}

/// Register count of a `tcgen05.ld`/`st` data payload (`registers_per_num *
/// num`), validating the shape's legal `.num` values (x1..x128 by shape).
pub fn tcgen_ldst_registers(shape: TcShape, num: u16) -> OpResult<usize> {
    let shape = ldst_shape(shape);
    if !shape.valid_num(usize::from(num)) {
        return Err(OpError::invalid(format!(
            "tcgen05.ld/st {shape:?} has no .x{num} form"
        )));
    }
    Ok(shape.registers_per_num() * usize::from(num))
}

/// One contiguous piece of a `tcgen05.ld`/`st` register: `len` bytes at byte
/// `reg_byte` of the 32-bit register move to/from byte `cell_byte` of TMEM
/// cell `(tmem_lane, column)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TcgenLdstPiece {
    pub tmem_lane: u32,
    pub column: u32,
    pub cell_byte: u8,
    pub reg_byte: u8,
    pub len: u8,
}

/// Register x lane -> TMEM map of one `tcgen05.ld`/`st` (all 32 lanes; the
/// instruction is warp-collective).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TcgenLdstMap {
    /// Data registers per lane (`.red` / `.spcompress` outputs excluded).
    pub registers: usize,
    /// 1, or 2 for `.pack::16b` / `.unpack::16b` (two 16-bit halves in the
    /// low halves of two consecutive columns).
    pub pieces_per_register: usize,
    pieces: std::rc::Rc<[TcgenLdstPiece]>,
    runs: std::rc::Rc<[TcgenCellRun]>,
}

/// A run of consecutive TMEM cells of one lane that a `tcgen05.ld`/`st`
/// touches (`cells` cells from `column`); see [`TcgenLdstMap::cell_runs`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TcgenCellRun {
    pub tmem_lane: u32,
    pub column: u32,
    pub cells: u32,
}

impl TcgenLdstMap {
    /// Pieces of `register` in `lane`, in register byte order.
    pub fn pieces(&self, register: usize, lane: usize) -> &[TcgenLdstPiece] {
        let start = (register * WARP + lane) * self.pieces_per_register;
        &self.pieces[start..start + self.pieces_per_register]
    }

    /// Every piece (register-major, then lane, then piece).
    pub fn all(&self) -> &[TcgenLdstPiece] {
        &self.pieces
    }

    /// The TMEM cells the pieces touch, as maximal runs of consecutive
    /// columns per lane, sorted by (lane, column); each cell appears once.
    /// Lets the engine move whole runs instead of one piece at a time.
    pub fn cell_runs(&self) -> &[TcgenCellRun] {
        &self.runs
    }
}

/// Legacy `raw_tcgen05_ldst_location` for every register and lane of a
/// `tcgen05.ld`/`st`. `warp_in_cta` selects the warp's 32-lane TMEM
/// subpartition (`% 4`); a taddr lane below 32 is warp-relative, otherwise it
/// must lie in the warp's subpartition. `pack16` = `.pack::16b` (ld) /
/// `.unpack::16b` (st). The `.16x32bx2` half-split offset rides in
/// `TcShape::S16x32bx2 { split_off }`.
pub fn tcgen_ldst_map(
    shape: TcShape,
    num: u16,
    pack16: bool,
    warp_in_cta: u32,
    taddr: u32,
) -> OpResult<TcgenLdstMap> {
    // Maps are pure in their inputs and shared (`Rc`) once built; kernels
    // issue the same few forms every iteration (perf, W2-21).
    type Key = (TcShape, u16, bool, u32, u32);
    thread_local! {
        static MAPS: std::cell::RefCell<std::collections::HashMap<Key, TcgenLdstMap>> =
            std::cell::RefCell::new(std::collections::HashMap::new());
    }
    let key: Key = (shape, num, pack16, warp_in_cta, taddr);
    if let Some(map) = MAPS.with(|maps| maps.borrow().get(&key).cloned()) {
        return Ok(map);
    }
    let map = build_ldst_map(shape, num, pack16, warp_in_cta, taddr)?;
    MAPS.with(|maps| {
        let mut maps = maps.borrow_mut();
        if maps.len() >= 1024 {
            maps.clear();
        }
        maps.insert(key, map.clone());
    });
    Ok(map)
}

fn cell_runs(pieces: &[TcgenLdstPiece]) -> Vec<TcgenCellRun> {
    let mut cells: Vec<(u32, u32)> = pieces.iter().map(|p| (p.tmem_lane, p.column)).collect();
    cells.sort_unstable();
    cells.dedup();
    let mut runs: Vec<TcgenCellRun> = Vec::new();
    for (lane, column) in cells {
        match runs.last_mut() {
            Some(run) if run.tmem_lane == lane && run.column + run.cells == column => run.cells += 1,
            _ => runs.push(TcgenCellRun { tmem_lane: lane, column, cells: 1 }),
        }
    }
    runs
}

fn build_ldst_map(
    shape: TcShape,
    num: u16,
    pack16: bool,
    warp_in_cta: u32,
    taddr: u32,
) -> OpResult<TcgenLdstMap> {
    let registers = tcgen_ldst_registers(shape, num)?;
    let shape = ldst_shape(shape);
    let per = if pack16 { 2 } else { 1 };
    let mut pieces = Vec::with_capacity(registers * WARP * per);
    for register in 0..registers {
        for lane in 0..WARP {
            let (row, column) = ldst_location(
                warp_in_cta as usize,
                taddr,
                0,
                0,
                shape,
                pack16,
                register,
                lane,
            )?;
            let row: u32 = narrow(row, "TMEM lane")?;
            let column: u32 = narrow(column, "TMEM column")?;
            if column >= crate::arena::addr::TMEM_COLS + 1 - per as u32 {
                return Err(OpError::invalid(format!(
                    "tcgen05.ld/st column {column} is outside TMEM"
                )));
            }
            if pack16 {
                for half in 0..2_u8 {
                    pieces.push(TcgenLdstPiece {
                        tmem_lane: row,
                        column: column + u32::from(half),
                        cell_byte: 0,
                        reg_byte: 2 * half,
                        len: 2,
                    });
                }
            } else {
                pieces.push(TcgenLdstPiece {
                    tmem_lane: row,
                    column,
                    cell_byte: 0,
                    reg_byte: 0,
                    len: 4,
                });
            }
        }
    }
    let runs = cell_runs(&pieces);
    Ok(TcgenLdstMap {
        registers,
        pieces_per_register: per,
        pieces: pieces.into(),
        runs: runs.into(),
    })
}

/// Destination register count of a `tcgen05.ld` variant, including the
/// trailing `.red` register; validates the variant (`.red` needs unpacked
/// `32x32b`/`16x32bx2` with at least x2; `.spcompress` = `Some((max, abs))`
/// needs unpacked `32x32b` with at least x4).
pub fn tcgen_ld_dst_count(
    shape: TcShape,
    num: u16,
    pack16: bool,
    red: bool,
    spcompress: Option<(bool, bool)>,
) -> OpResult<usize> {
    tcgen_ldst_registers(shape, num)?;
    Ok(ld_destination_count(
        ldst_shape(shape),
        usize::from(num),
        pack16,
        red,
        spcompress,
    )?)
}

/// `tcgen05.ld.red` operator (legacy `LoadReduction` impls).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TcgenLdRed(LdReduction);

impl TcgenLdRed {
    /// `.red.{min,max}` over the reduction register's type (`F32` with
    /// optional `.abs` / `.NaN`, `U32`, `S32`).
    pub fn new(op: ReduxOp, ty: Dtype, abs: bool, nan: bool) -> OpResult<TcgenLdRed> {
        let max = match op {
            ReduxOp::Min => false,
            ReduxOp::Max => true,
            other => {
                return Err(OpError::invalid(format!(
                    "tcgen05.ld.red has no .{other:?} operator"
                )))
            }
        };
        if (abs || nan) && ty != Dtype::F32 {
            return Err(OpError::invalid("tcgen05.ld.red .abs/.NaN require .f32"));
        }
        Ok(TcgenLdRed(match ty {
            Dtype::F32 => LdReduction::F32 { max, abs, nan },
            Dtype::U32 => LdReduction::U32 { max },
            Dtype::S32 => LdReduction::I32 { max },
            other => {
                return Err(OpError::invalid(format!(
                    "tcgen05.ld.red type {other:?} is not f32/u32/s32"
                )))
            }
        }))
    }
}

/// One lane's reduction value: left fold over the loaded 32-bit words in
/// register order (legacy: the values just loaded, not a second read).
pub fn tcgen_ld_reduce(red: TcgenLdRed, values: &[u32]) -> OpResult<u32> {
    if values.len() < 2 {
        return Err(OpError::invalid("tcgen05.ld.red needs at least two values"));
    }
    Ok(red.0.reduce(values.iter().copied()).unwrap_or(0))
}

/// One lane of `tcgen05.ld.spcompress ... .sp::2:4 .f32.b2` (PTX ISA 9.4
/// 9.7.18.8.3; `num` loaded words and their byte-complete validity).
///
/// Definition: each group of 4 consecutive f32 words keeps 2, chosen by
/// `max` / `min` of the value (of `|value|` with `abs`). NaNs are chosen
/// first; ties keep the lower index (legacy `sparse_pair_indices`). The
/// output is `num.div_ceil(32)` metadata words, then the `num / 2` kept
/// values in ascending index order within each group. Metadata packs one
/// 2-bit in-group index per kept value, 16 per word (kept value `e` in bits
/// `2 * (e % 16)` of word `e / 16`). Validity is per output. The
/// `spcompress_tests` check this against an independent model, including
/// that decompressing by the metadata reproduces the kept values.
pub fn tcgen_ld_spcompress(
    values: &[u32],
    valid: &[bool],
    max: bool,
    abs: bool,
) -> OpResult<(Vec<u32>, Vec<bool>)> {
    if values.len() != valid.len() || values.len() < 4 || !values.len().is_multiple_of(4) {
        return Err(OpError::invalid(
            "tcgen05.ld.spcompress needs a multiple of four values with validity",
        ));
    }
    Ok(spcompress_lane(
        values,
        valid,
        max,
        abs,
        numsim_oplib::arith::sparse::sparse_pair_indices,
    ))
}

// ---------------------------------------------------------------------------
// tcgen05.cp
// ---------------------------------------------------------------------------

/// One copied source word: `src` (window bytes; 4, or 2/3 when b4/b6
/// compressed) lands as one decoded 4-byte cell in every `lanes[..lane_count]`
/// at `column`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TcgenCpWord {
    pub src: ByteSpan,
    pub lanes: [u32; 4],
    pub lane_count: u8,
    pub column: u32,
}

impl TcgenCpWord {
    /// The destination TMEM lanes of this word (1, 2 or 4).
    pub fn lanes(&self) -> &[u32] {
        &self.lanes[..usize::from(self.lane_count)]
    }
}

/// Plan of one `tcgen05.cp` in one target CTA (the same plan applies to the
/// peer of a `cta_group::2` copy, reading the peer's own shared window).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TcgenCpPlan {
    /// Source-row-major, then word (legacy order).
    pub words: Vec<TcgenCpWord>,
    /// Contract code: 0 none, 4 = `b4x16_p64`, 6 = `b6x16_p32`.
    pub decompress_bits: u8,
    /// One past the highest destination lane / column.
    pub lane_end: u32,
    pub column_end: u32,
}

impl TcgenCpPlan {
    /// Pairwise form for `Payload::TcgenCp`: `src[k]` (2/3/4 bytes) decodes
    /// with [`tcgen_cp_decode`] into destination cell `dst[k] = (lane,
    /// column)` (4 bytes). Multicast repeats a source span per lane.
    pub fn pairs(&self) -> (Vec<ByteSpan>, Vec<(u32, u32)>) {
        let cells = self.words.iter().map(|word| usize::from(word.lane_count)).sum();
        let mut src = Vec::with_capacity(cells);
        let mut dst = Vec::with_capacity(cells);
        for word in &self.words {
            for &lane in word.lanes() {
                src.push(word.src);
                dst.push((lane, word.column));
            }
        }
        (src, dst)
    }
}

fn cp_shape_code(rows: u16, bits: u16, multicast: u8) -> OpResult<u8> {
    use numsim_oplib::tcgen05::layouts::cp_shape::*;
    Ok(match (rows, bits, multicast) {
        (32, 128, 3) => WARPX4_32X128B,
        (64, 128, 1) => WARPX2_02_13_64X128B,
        (64, 128, 2) => WARPX2_01_23_64X128B,
        (128, 128, 0) => SHAPE_128X128B,
        (128, 256, 0) => SHAPE_128X256B,
        (4, 256, 0) => SHAPE_4X256B,
        _ => {
            return Err(OpError::invalid(format!(
                "tcgen05.cp has no .{rows}x{bits}b form with multicast code {multicast}"
            )))
        }
    })
}

fn decompress_code(bits: u8) -> OpResult<u8> {
    match bits {
        0 => Ok(0),
        4 => Ok(1),
        6 => Ok(2),
        other => Err(OpError::invalid(format!(
            "tcgen05.cp decompression b{other} is not b4x16_p64/b6x16_p32"
        ))),
    }
}

/// Legacy `raw_tcgen05_cp` / `raw_tcgen05_cp_footprints` for one target CTA.
/// `rows`/`bits`/`multicast`/`decompress_bits` are `TcgenCpArgs` (multicast
/// 0 none, 1 = `warpx2::02_13`, 2 = `warpx2::01_23`, 3 = `warpx4`);
/// `sdesc` is the shared matrix descriptor (source spans are shared-window
/// byte addresses, swizzled per the descriptor); `taddr` the destination.
///
/// Plans are a pure function of the arguments, so successful plans are
/// memoized per thread (perf, W4: on the Mega MoE medium every issue of the
/// 32x128b warpx4 scale copy replanned 128 swizzled source words, ~3 us).
/// Errors are not cached; a failing request recomputes and fails the same way.
#[allow(clippy::too_many_arguments)]
pub fn tcgen_cp_plan(
    rows: u16,
    bits: u16,
    multicast: u8,
    decompress_bits: u8,
    sdesc: u64,
    taddr: u32,
    cta_group: u8,
    arch: TcArch,
) -> OpResult<TcgenCpPlan> {
    type Key = (u16, u16, u8, u8, u64, u32, u8, TcArch);
    thread_local! {
        static PLANS: std::cell::RefCell<std::collections::HashMap<Key, std::rc::Rc<TcgenCpPlan>>> =
            std::cell::RefCell::new(std::collections::HashMap::new());
    }
    let key: Key = (rows, bits, multicast, decompress_bits, sdesc, taddr, cta_group, arch);
    if let Some(hit) = PLANS.with(|plans| plans.borrow().get(&key).cloned()) {
        return Ok((*hit).clone());
    }
    let plan = tcgen_cp_plan_uncached(rows, bits, multicast, decompress_bits, sdesc, taddr, cta_group, arch)?;
    PLANS.with(|plans| {
        let mut plans = plans.borrow_mut();
        if plans.len() >= 4096 {
            plans.clear();
        }
        plans.insert(key, std::rc::Rc::new(plan.clone()));
    });
    Ok(plan)
}

/// [`tcgen_cp_plan`] without the per-thread memo.
#[allow(clippy::too_many_arguments)]
pub(crate) fn tcgen_cp_plan_uncached(
    rows: u16,
    bits: u16,
    multicast: u8,
    decompress_bits: u8,
    sdesc: u64,
    taddr: u32,
    cta_group: u8,
    arch: TcArch,
) -> OpResult<TcgenCpPlan> {
    if !matches!(cta_group, 0..=2) {
        return Err(OpError::invalid(format!(
            "raw tcgen05.cp cta_group must be 1 or 2, got {cta_group}"
        )));
    }
    let shape = cp_shape_code(rows, bits, multicast)?;
    let decompress = decompress_code(decompress_bits)?;
    let layout = match arch {
        TcArch::Sm100 => MatrixDescriptorLayout::Sm100,
        TcArch::Sm103 => MatrixDescriptorLayout::Sm103,
        TcArch::Sm107 => MatrixDescriptorLayout::Sm107,
    };
    let descriptor = decode_matrix_descriptor_for_layout(sdesc, layout)?;
    let (base_row, base_col) = tmem_address(taddr, 0, 0)?;
    let (source_rows, words) = cp_rows_words(shape)?;
    let column_end = base_col + words;
    if column_end > crate::arena::addr::TMEM_COLS as usize {
        return Err(OpError::invalid(format!(
            "raw tcgen05.cp TMEM columns [{base_col}, {column_end}) exceed TMEM"
        )));
    }
    let mut lane_end = 0;
    let mut plan = Vec::with_capacity(source_rows * words);
    for source_row in 0..source_rows {
        let destinations = cp_destination_lanes(shape, source_row)?;
        let mut lanes = [0_u32; 4];
        for (slot, &lane) in destinations.as_slice().iter().enumerate() {
            let lane = base_row + lane;
            if lane >= 128 {
                return Err(OpError::invalid(format!(
                    "raw tcgen05.cp TMEM lane {lane} is outside 128 lanes"
                )));
            }
            lane_end = lane_end.max(lane + 1);
            lanes[slot] = lane as u32;
        }
        for word in 0..words {
            let (offset, len) = cp_source_span(WINDOW, descriptor, source_row, word, decompress)?;
            plan.push(TcgenCpWord {
                src: ByteSpan::new(offset as u64, len as u64),
                lanes,
                lane_count: destinations.as_slice().len() as u8,
                column: (base_col + word) as u32,
            });
        }
    }
    Ok(TcgenCpPlan {
        words: plan,
        decompress_bits,
        lane_end: lane_end as u32,
        column_end: column_end as u32,
    })
}

/// Decode one source word of a `tcgen05.cp` (contract `decompress_bits`:
/// 0 = 4 bytes as-is, 4 = 2 bytes of b4 -> four `bits << 2` bytes, 6 = 3 bytes
/// of b6 -> four 6-bit bytes).
pub fn tcgen_cp_decode(src: &[u8], decompress_bits: u8) -> OpResult<[u8; 4]> {
    let decompress = decompress_code(decompress_bits)?;
    if src.len() > 4 {
        return Err(OpError::invalid("tcgen05.cp source word exceeds 4 bytes"));
    }
    let mut packed = [0_u8; 4];
    packed[..src.len()].copy_from_slice(src);
    Ok(cp_decode_word(&packed, src.len(), decompress)?)
}

#[cfg(test)]
mod cp_plan_cache_tests {
    use super::*;

    /// The memoized `tcgen_cp_plan` returns exactly the direct planner's
    /// plan or error, first call and repeats alike, over every shape,
    /// multicast, decompression, arch, CTA group and random descriptors and
    /// TMEM addresses (including invalid ones), and `pairs()` is the
    /// word-major, lane-minor expansion of the plan.
    #[test]
    fn memoized_cp_plans_equal_direct_plans() {
        let mut seed = 0x7c0f_fee1_u64;
        let mut next = move |bound: u64| {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (seed >> 33) % bound
        };
        let shapes = [(32_u16, 128_u16, 3_u8), (64, 128, 1), (64, 128, 2), (128, 128, 0), (128, 256, 0), (4, 256, 0), (64, 256, 1)];
        let (mut ok, mut err) = (0, 0);
        for _ in 0..3000 {
            let (rows, bits, multicast) = shapes[next(shapes.len() as u64) as usize];
            let decompress = [0_u8, 0, 4, 6, 5][next(5) as usize];
            let sdesc = numsim_oplib::tcgen05::encode::encode_matrix_descriptor(
                (next(1 << 10) as u32) << 4,
                next(64) as i64,
                next(128) as i64,
                next(5) as i64,
            );
            let sdesc = if next(10) == 0 { sdesc ^ (1 << (49 + next(3))) } else { sdesc };
            // Bit 14: accepted by the SM107 descriptor layout only.
            let sdesc = if next(4) == 0 { sdesc | (1 << 14) } else { sdesc };
            let lane = [0_u32, 0, 32, 64, 96, 100][next(6) as usize];
            let taddr = (lane << 16) | (next(520) as u32);
            let group = [1_u8, 2, 1, 3][next(4) as usize];
            let arch = [TcArch::Sm100, TcArch::Sm103, TcArch::Sm107][next(3) as usize];
            let direct = tcgen_cp_plan_uncached(rows, bits, multicast, decompress, sdesc, taddr, group, arch).map_err(|e| e.to_string());
            for _ in 0..2 {
                let memo = tcgen_cp_plan(rows, bits, multicast, decompress, sdesc, taddr, group, arch).map_err(|e| e.to_string());
                assert_eq!(memo, direct, "{rows}x{bits} mc{multicast} dc{decompress} sdesc {sdesc:#x} taddr {taddr:#x} group {group} {arch:?}");
            }
            // Neighbours differing in one key field each (a key missing a
            // field would serve the first plan for the second request).
            let neighbours = [
                (rows, bits, multicast ^ 1, decompress, sdesc, taddr, group, arch),
                (rows, bits, multicast, decompress ^ 4, sdesc, taddr, group, arch),
                (rows, bits, multicast, decompress, sdesc + 0x10, taddr, group, arch),
                (rows, bits, multicast, decompress, sdesc, taddr ^ 4, group, arch),
                (rows, bits, multicast, decompress, sdesc, taddr ^ (32 << 16), group, arch),
                (rows, bits, multicast, decompress, sdesc, taddr, 3 - group.min(2), arch),
                (rows, bits, multicast, decompress, sdesc | (1 << 46), taddr, group, TcArch::Sm103),
                (rows, bits ^ 384, multicast, decompress, sdesc, taddr, group, arch),
                (rows ^ 96, bits, multicast, decompress, sdesc, taddr, group, arch),
                (rows, bits, multicast, decompress, sdesc, taddr, group, TcArch::Sm100),
                (rows, bits, multicast, decompress, sdesc, taddr, group, TcArch::Sm103),
                (rows, bits, multicast, decompress, sdesc, taddr, group, TcArch::Sm107),
            ];
            for (r, b, m, d, sd, t, g, a) in neighbours {
                let want = tcgen_cp_plan_uncached(r, b, m, d, sd, t, g, a).map_err(|e| e.to_string());
                let memo = tcgen_cp_plan(r, b, m, d, sd, t, g, a).map_err(|e| e.to_string());
                assert_eq!(memo, want, "{r}x{b} mc{m} dc{d} sdesc {sd:#x} taddr {t:#x} group {g} {a:?}");
            }
            match &direct {
                Ok(plan) => {
                    ok += 1;
                    let (src, dst) = plan.pairs();
                    let mut k = 0;
                    for word in &plan.words {
                        for &lane in word.lanes() {
                            assert_eq!((src[k], dst[k]), (word.src, (lane, word.column)));
                            k += 1;
                        }
                    }
                    assert_eq!((src.len(), dst.len()), (k, k));
                }
                Err(_) => err += 1,
            }
        }
        assert!(ok > 300 && err > 100, "ok {ok} err {err}");
    }
}

#[cfg(test)]
mod run_tests {
    use super::*;

    /// `cell_runs` covers exactly the cells of the pieces, merged per lane;
    /// the cached map equals a freshly built one.
    #[test]
    fn ldst_cell_runs_cover_the_pieces_once() {
        for (shape, num, pack16) in [
            (TcShape::S32x32b, 8_u16, false),
            (TcShape::S32x32b, 4, true),
            (TcShape::S16x64b, 4, false),
            (TcShape::S16x256b, 2, false),
        ] {
            let map = tcgen_ldst_map(shape, num, pack16, 1, 0x20_0010).unwrap();
            assert_eq!(map, build_ldst_map(shape, num, pack16, 1, 0x20_0010).unwrap());
            let mut from_runs: Vec<(u32, u32)> = map
                .cell_runs()
                .iter()
                .flat_map(|r| (r.column..r.column + r.cells).map(move |c| (r.tmem_lane, c)))
                .collect();
            let mut from_pieces: Vec<(u32, u32)> = map.all().iter().map(|p| (p.tmem_lane, p.column)).collect();
            from_pieces.sort_unstable();
            from_pieces.dedup();
            let n = from_runs.len();
            from_runs.dedup();
            assert_eq!(from_runs.len(), n, "{shape:?}: a cell in two runs");
            assert_eq!(from_runs, from_pieces, "{shape:?}");
            assert!(map.cell_runs().len() <= 32, "{shape:?}: one run per lane expected");
        }
    }
}

#[cfg(test)]
mod spcompress_tests {
    use super::*;

    /// Independent model of PTX `tcgen05.ld.spcompress ... .sp::2:4 .f32.b2`
    /// (9.7.18.x): every group of 4 loaded f32 values keeps 2, chosen by
    /// `.max`/`.min` of the value (or of `|value|` with `.abs`), NaNs first;
    /// the kept values are written in ascending index order; the metadata
    /// packs one 2-bit index per kept value, 16 per 32-bit word. Decompressing
    /// (scattering kept values back by index) must reproduce them.
    fn reference(values: &[f32], max: bool, abs: bool) -> (Vec<u32>, Vec<u32>) {
        let mut meta = vec![0u32; values.len().div_ceil(32)];
        let mut kept = Vec::new();
        for group in 0..values.len() / 4 {
            let v = &values[group * 4..group * 4 + 4];
            let key = |i: usize| if abs { v[i].abs() } else { v[i] };
            let mut order: Vec<usize> = (0..4).collect();
            order.sort_by(|&a, &b| {
                let (ka, kb) = (key(a), key(b));
                (!ka.is_nan()).cmp(&!kb.is_nan()).then_with(|| {
                    let ord = ka.partial_cmp(&kb).unwrap_or(std::cmp::Ordering::Equal);
                    if max { ord.reverse() } else { ord }
                })
            });
            let mut pick = [order[0], order[1]];
            pick.sort_unstable();
            for (j, &index) in pick.iter().enumerate() {
                let element = group * 2 + j;
                meta[element / 16] |= (index as u32) << ((element % 16) * 2);
                kept.push(v[index].to_bits());
            }
        }
        (meta, kept)
    }

    #[test]
    fn spcompress_matches_an_independent_selection_and_decompresses() {
        let mut seed = 0x1234_5678_u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for num in [4usize, 8, 32, 128] {
            for (max, abs) in [(true, false), (false, false), (true, true), (false, true)] {
                for _ in 0..20 {
                    // Distinct finite values (no ties) plus at most two NaNs per group.
                    let values: Vec<f32> = (0..num)
                        .map(|i| if next() % 11 == 0 && i % 4 < 2 { f32::NAN } else { (next() % 20_000) as f32 / 7.0 - 1400.0 + i as f32 * 1e-3 })
                        .collect();
                    let words: Vec<u32> = values.iter().map(|v| v.to_bits()).collect();
                    let (out, valid) = tcgen_ld_spcompress(&words, &vec![true; num], max, abs).unwrap();
                    assert!(valid.iter().all(|&v| v));
                    let (meta, kept) = reference(&values, max, abs);
                    assert_eq!(out.len(), meta.len() + kept.len(), "num {num}");
                    assert_eq!(&out[..meta.len()], &meta[..], "metadata num {num} max {max} abs {abs}");
                    assert_eq!(&out[meta.len()..], &kept[..], "values num {num} max {max} abs {abs}");
                    // Decompress: each kept value sits at its group's index.
                    for (element, &bits) in kept.iter().enumerate() {
                        let index = (meta[element / 16] >> ((element % 16) * 2)) & 3;
                        assert_eq!(words[(element / 2) * 4 + index as usize], bits);
                    }
                }
            }
        }
    }
}
