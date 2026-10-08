//! Pure value helpers W1 lowers to `Instr::Ptx` (`v2/lowering/builtins.py`
//! `HELPERS` with kind `pure`, plus `tirx.fma` / `tirx.reinterpret`).
//!
//! Semantics follow the legacy frontend (`frontend-rs/src/emit/pure.rs`,
//! `emit/cuda_helper.rs`, `emit/raw_tcgen.rs`) through the `numsim-oplib`
//! kernels its `BINDINGS` name. Operand order is the TIR call's argument
//! order: a value-returning helper's result is `dsts[0]`; out-parameters
//! (`o`) are destinations, in/out parameters (`x`) are both a source (current
//! value) and a destination; string-literal arguments arrive as modifiers
//! (bare tokens in argument order, or `arg<i>=<text>`).
//!
//! Every helper checks operand arity and exact carrier widths at resolve time
//! (fail closed), so the per-lane bodies index one 64-bit slot per operand.

use super::{Mods, Operands, Resolved};
use crate::arena::addr;
use crate::dtype::{Dtype, Ty};
use crate::oplib::{OpError, OpResult, PtxIo};
use numsim_oplib::scalar::F32RoundingMode;
use numsim_oplib::{arith, codec, cvt, scalar, tcgen05};
use numsim_types::WARP_SIZE;

#[cfg(test)]
mod tests;

pub(in crate::oplib) const NAMES: &[&str] = &[
    "tirx.cuda.make_float2",
    "tirx.cuda.float2_x",
    "tirx.cuda.float2_y",
    "tirx.cuda.uint_as_float",
    "tirx.cuda.float_as_uint",
    "tirx.cuda.ffs_u32",
    "tirx.cuda.float22bfloat162_rn",
    "tirx.cuda.float22bfloat162_rn_from_float2",
    "tirx.cuda.bfloat1622float2",
    "tirx.cuda.hmin2",
    "tirx.cuda.hmax2",
    "tirx.cuda.fmul2_rn",
    "tirx.cuda.fadd2_rn",
    "tirx.cuda.fdividef",
    "tirx.cuda.fp8x4_e4m3_from_float4",
    "tirx.cuda.half2float",
    "tirx.cuda.bfloat162float",
    "tirx.cuda.clock64",
    "tirx.cuda.get_tmem_addr",
    "tirx.cuda.runtime_instr_desc",
    "tirx.cuda.tcgen05_encode_matrix_descriptor",
    "tirx.cuda.tcgen05_encode_instr_descriptor",
    "tirx.cuda.tcgen05_encode_instr_descriptor_block_scaled",
    "tirx.cuda.sm100_2sm_leader_smem_addr",
    "tirx.cuda.float22half2",
    "tirx.cuda.float8tohalf8",
    "tirx.cuda.half8tofloat8",
    "tirx.fma",
    "tirx.reinterpret",
    "tirx.log1p",
    "tirx.sigmoid",
];

/// Run `body(lane)` for every executing lane.
macro_rules! each_lane {
    ($io:ident, |$lane:ident| $body:expr) => {
        for $lane in 0..WARP_SIZE {
            if $io.mask.contains($lane) {
                $body;
            }
        }
    };
}

#[inline]
fn f32_at(io: &PtxIo<'_>, src: usize, lane: usize) -> f32 {
    f32::from_bits(io.srcs[src][lane] as u32)
}

#[inline]
fn bits32(value: f32) -> u64 {
    u64::from(value.to_bits())
}

// --- hot direct forms --------------------------------------------------------

fn make_float2(io: &mut PtxIo<'_>) -> OpResult {
    each_lane!(io, |lane| io.dsts[0][lane] =
        cvt::make_float2(f32_at(io, 0, lane), f32_at(io, 1, lane)));
    Ok(())
}

fn float2_x(io: &mut PtxIo<'_>) -> OpResult {
    each_lane!(io, |lane| io.dsts[0][lane] =
        bits32(cvt::float2_x(io.srcs[0][lane])));
    Ok(())
}

fn float2_y(io: &mut PtxIo<'_>) -> OpResult {
    each_lane!(io, |lane| io.dsts[0][lane] =
        bits32(cvt::float2_y(io.srcs[0][lane])));
    Ok(())
}

/// `float_as_uint` / `uint_as_float`: a 32-bit payload move.
fn move32(io: &mut PtxIo<'_>) -> OpResult {
    each_lane!(io, |lane| io.dsts[0][lane] = io.srcs[0][lane] & 0xffff_ffff);
    Ok(())
}

/// `tirx.reinterpret` between equal-width carriers: copy every slot.
fn reinterpret(io: &mut PtxIo<'_>) -> OpResult {
    let slots = io.dst_tys[0].slots() as usize;
    for slot in 0..slots {
        each_lane!(io, |lane| io.dsts[slot][lane] = io.srcs[slot][lane]);
    }
    Ok(())
}

fn clock64(io: &mut PtxIo<'_>) -> OpResult {
    // Legacy `cuda_clock64` is the deterministic representative 0 (the
    // engine's `SpecialReg::Clock64` read is W1's usual lowering).
    each_lane!(io, |lane| io.dsts[0][lane] = 0);
    Ok(())
}

/// SM100 2-SM pair leader address: clear the shared::cluster CTA-rank bit 0
/// (hardware bit 24; the engine's encoding is the hardware's `rank << 24 |
/// offset`, `arena::addr::shared_addr`), i.e. the same offset in the even
/// CTA of the pair (legacy `sm100_tma_2sm_mbarrier_address`).
fn sm100_2sm_leader_smem_addr(io: &mut PtxIo<'_>) -> OpResult {
    each_lane!(io, |lane| io.dsts[0][lane] = (io.srcs[0][lane] as u32 & 0xfeff_ffff) as u64);
    Ok(())
}

// --- other fixed-signature forms -----------------------------------------------

fn ffs_u32(io: &mut PtxIo<'_>) -> OpResult {
    each_lane!(io, |lane| io.dsts[0][lane] =
        u64::from(scalar::cuda_ffs_u32(io.srcs[0][lane] as u32) as u32));
    Ok(())
}

fn float22bfloat162_rn(io: &mut PtxIo<'_>) -> OpResult {
    each_lane!(io, |lane| io.dsts[0][lane] = u64::from(cvt::pack_bf16x2(
        f32_at(io, 0, lane),
        f32_at(io, 1, lane)
    )));
    Ok(())
}

fn float22bfloat162_rn_from_float2(io: &mut PtxIo<'_>) -> OpResult {
    each_lane!(io, |lane| {
        let v = io.srcs[0][lane];
        io.dsts[0][lane] = u64::from(cvt::pack_bf16x2(cvt::float2_x(v), cvt::float2_y(v)))
    });
    Ok(())
}

fn bfloat1622float2(io: &mut PtxIo<'_>) -> OpResult {
    each_lane!(io, |lane| io.dsts[0][lane] =
        cvt::unpack_bf16x2(io.srcs[0][lane] as u32));
    Ok(())
}

fn hmin2_bf16(io: &mut PtxIo<'_>) -> OpResult {
    each_lane!(io, |lane| io.dsts[0][lane] = u64::from(cvt::hmin2_bf16(
        io.srcs[0][lane] as u32,
        io.srcs[1][lane] as u32
    )));
    Ok(())
}

fn hmax2_bf16(io: &mut PtxIo<'_>) -> OpResult {
    each_lane!(io, |lane| io.dsts[0][lane] = u64::from(cvt::hmax2_bf16(
        io.srcs[0][lane] as u32,
        io.srcs[1][lane] as u32
    )));
    Ok(())
}

fn hmin2_f16(io: &mut PtxIo<'_>) -> OpResult {
    each_lane!(io, |lane| io.dsts[0][lane] = u64::from(cvt::hmin2_f16(
        io.srcs[0][lane] as u32,
        io.srcs[1][lane] as u32
    )));
    Ok(())
}

fn hmax2_f16(io: &mut PtxIo<'_>) -> OpResult {
    each_lane!(io, |lane| io.dsts[0][lane] = u64::from(cvt::hmax2_f16(
        io.srcs[0][lane] as u32,
        io.srcs[1][lane] as u32
    )));
    Ok(())
}

fn fmul2_rn(io: &mut PtxIo<'_>) -> OpResult {
    each_lane!(io, |lane| io.dsts[0][lane] = cvt::mul_f32x2(
        io.srcs[0][lane],
        io.srcs[1][lane],
        F32RoundingMode::Nearest,
        false
    ));
    Ok(())
}

fn fadd2_rn(io: &mut PtxIo<'_>) -> OpResult {
    each_lane!(io, |lane| io.dsts[0][lane] = cvt::add_f32x2(
        io.srcs[0][lane],
        io.srcs[1][lane],
        F32RoundingMode::Nearest,
        false
    ));
    Ok(())
}

fn fdividef(io: &mut PtxIo<'_>) -> OpResult {
    // Legacy lowers `__fdividef` to an IEEE round-to-nearest `f32` division.
    each_lane!(io, |lane| io.dsts[0][lane] = bits32(scalar::div_f32_rn(
        f32_at(io, 0, lane),
        f32_at(io, 1, lane)
    )));
    Ok(())
}

fn fp8x4_e4m3_from_float4(io: &mut PtxIo<'_>) -> OpResult {
    each_lane!(io, |lane| io.dsts[0][lane] =
        u64::from(cvt::fp8x4_e4m3_from_float4(
            f32_at(io, 0, lane),
            f32_at(io, 1, lane),
            f32_at(io, 2, lane),
            f32_at(io, 3, lane),
        )));
    Ok(())
}

fn half2float(io: &mut PtxIo<'_>) -> OpResult {
    each_lane!(io, |lane| io.dsts[0][lane] =
        bits32(scalar::cuda_fp16_bits_to_f32(io.srcs[0][lane] as u16)));
    Ok(())
}

fn bfloat162float(io: &mut PtxIo<'_>) -> OpResult {
    each_lane!(io, |lane| io.dsts[0][lane] = bits32(
        scalar::cuda_canonicalize_nan_f32(cvt::bf16_bits_to_f32(io.srcs[0][lane] as u16))
    ));
    Ok(())
}

fn get_tmem_addr(io: &mut PtxIo<'_>) -> OpResult {
    each_lane!(io, |lane| io.dsts[0][lane] =
        u64::from(tcgen05::layouts::get_tmem_addr(
            io.srcs[0][lane] as u32,
            io.srcs[1][lane] as u32 as i32,
            io.srcs[2][lane] as u32,
        )));
    Ok(())
}

fn runtime_instr_desc(io: &mut PtxIo<'_>) -> OpResult {
    each_lane!(io, |lane| io.dsts[0][lane] =
        u64::from(codec::tcgen_runtime_instruction_descriptor(
            io.srcs[0][lane] as u32,
            io.srcs[1][lane] as u32,
        )));
    Ok(())
}

fn fma_f32(io: &mut PtxIo<'_>) -> OpResult {
    each_lane!(io, |lane| io.dsts[0][lane] = bits32(arith::fma_f32(
        f32_at(io, 0, lane),
        f32_at(io, 1, lane),
        f32_at(io, 2, lane),
        F32RoundingMode::Nearest,
        false,
        false,
    )));
    Ok(())
}

fn fma_f64(io: &mut PtxIo<'_>) -> OpResult {
    each_lane!(io, |lane| io.dsts[0][lane] = scalar::fma_f64(
        f64::from_bits(io.srcs[0][lane]),
        f64::from_bits(io.srcs[1][lane]),
        f64::from_bits(io.srcs[2][lane]),
        F32RoundingMode::Nearest,
    )
    .to_bits());
    Ok(())
}

fn unary32(io: &mut PtxIo<'_>, f: fn(f32) -> f32) -> OpResult {
    each_lane!(io, |lane| io.dsts[0][lane] = bits32(f(f32_at(io, 0, lane))));
    Ok(())
}

fn unary64(io: &mut PtxIo<'_>, f: fn(f64) -> f64) -> OpResult {
    each_lane!(io, |lane| io.dsts[0][lane] = f(f64::from_bits(io.srcs[0][lane])).to_bits());
    Ok(())
}

fn unary16(io: &mut PtxIo<'_>, bf16: bool, f: fn(f32) -> f32) -> OpResult {
    each_lane!(io, |lane| io.dsts[0][lane] = u64::from(scalar::half_unary(io.srcs[0][lane] as u16, bf16, f)));
    Ok(())
}

// --- resolution ---------------------------------------------------------------

/// Require exact carrier widths (`bits`) for every destination and source.
fn widths(ops: &Operands, dsts: &[u32], srcs: &[u32], name: &str) -> OpResult<()> {
    ops.arity(dsts.len(), srcs.len(), name)?;
    let got = |tys: &[Ty]| tys.iter().map(|t| t.bits()).collect::<Vec<_>>();
    if got(&ops.dst_tys) != dsts || got(&ops.src_tys) != srcs {
        return Err(OpError::unsupported(format!(
            "{name}: expected carrier widths {dsts:?} <- {srcs:?}, got {:?} <- {:?}",
            ops.dst_tys, ops.src_tys
        )));
    }
    Ok(())
}

/// Reject any modifier on a helper that takes no string argument.
fn no_mods(mods: &Mods, name: &str) -> OpResult<()> {
    if let Some((slot, token)) = mods.pairs.first() {
        return Err(OpError::unsupported(format!(
            "{name}: unexpected modifier {slot}={token}"
        )));
    }
    Ok(())
}

/// String-literal arguments, in argument order (bare tokens, or `arg<i>=`).
fn string_args(mods: &Mods, name: &str) -> OpResult<Vec<String>> {
    let mut indexed = Vec::new();
    for (position, (slot, token)) in mods.pairs.iter().enumerate() {
        let index = if slot.is_empty() {
            position
        } else if let Some(i) = slot
            .strip_prefix("arg")
            .and_then(|i| i.parse::<usize>().ok())
        {
            i
        } else {
            return Err(OpError::unsupported(format!(
                "{name}: unexpected modifier {slot}={token}"
            )));
        };
        indexed.push((index, token.clone()));
    }
    indexed.sort_by_key(|(i, _)| *i);
    Ok(indexed.into_iter().map(|(_, t)| t).collect())
}

/// An integer/bool source value as `i64` (signed carriers sign-extend).
#[inline]
fn int_value(ty: Ty, raw: u64) -> i64 {
    let bits = ty.bits();
    let signed = matches!(ty.elem, Dtype::S8 | Dtype::S16 | Dtype::S32 | Dtype::S64);
    if bits >= 64 {
        raw as i64
    } else if signed {
        let shift = 64 - bits;
        ((raw << shift) as i64) >> shift
    } else {
        (raw & ((1u64 << bits) - 1)) as i64
    }
}

fn integer_sources(ops: &Operands, name: &str) -> OpResult<()> {
    for ty in &ops.src_tys {
        let ok = ty.lanes == 1
            && matches!(
                ty.elem,
                Dtype::Pred
                    | Dtype::U8
                    | Dtype::U16
                    | Dtype::U32
                    | Dtype::U64
                    | Dtype::S8
                    | Dtype::S16
                    | Dtype::S32
                    | Dtype::S64
            );
        if !ok {
            return Err(OpError::unsupported(format!(
                "{name}: non-integer argument carrier {ty:?}"
            )));
        }
    }
    Ok(())
}

fn encode_instr_descriptor(
    name: &str,
    mods: &Mods,
    ops: &Operands,
    block_scaled: bool,
) -> OpResult<Resolved> {
    let dtypes = string_args(mods, name)?;
    let (n_dtypes, n_values) = if block_scaled { (5, 11) } else { (3, 10) };
    if dtypes.len() != n_dtypes {
        return Err(OpError::unsupported(format!(
            "{name}: expected {n_dtypes} dtype string arguments, got {dtypes:?}"
        )));
    }
    ops.arity(1, n_values, name)?;
    if ops.dst_tys[0].bits() != 32 {
        return Err(OpError::unsupported(format!(
            "{name}: descriptor carrier {:?} is not 32-bit",
            ops.dst_tys[0]
        )));
    }
    integer_sources(ops, name)?;
    let ops = ops.clone();
    let form = format!("{block_scaled}:{}", dtypes.join(","));
    Ok(Resolved::Boxed(Box::new(move |io: &mut PtxIo<'_>| {
        // Operands are usually warp-uniform: encode once per distinct tuple
        // (the string dtype arguments make one encode ~80 ns; perf, W2-21).
        let mut last: Option<([i64; 11], u32)> = None;
        let mut previous: Option<(usize, u32)> = None;
        for lane in 0..WARP_SIZE {
            if !io.mask.contains(lane) {
                continue;
            }
            if let Some((prev, encoded)) = previous {
                if (0..n_values).all(|i| ops.src(io, i, lane) == ops.src(io, i, prev)) {
                    ops.put(io, 0, lane, u64::from(encoded), 32, false);
                    continue;
                }
            }
            let v = |i: usize| int_value(ops.src_tys[i], ops.src(io, i, lane));
            let mut inputs = [0_i64; 11];
            for (i, slot) in inputs.iter_mut().enumerate().take(n_values) {
                *slot = v(i);
            }
            if let Some((seen, encoded)) = last {
                if seen == inputs {
                    previous = Some((lane, encoded));
                    ops.put(io, 0, lane, encoded as u64, 32, false);
                    continue;
                }
            }
            // Across calls: the same few descriptors are encoded every
            // iteration; remember them per thread.
            thread_local! {
                static ENCODED: std::cell::RefCell<std::collections::HashMap<String, std::collections::HashMap<[i64; 11], u32>>> =
                    std::cell::RefCell::new(std::collections::HashMap::new());
            }
            let cached = ENCODED.with(|cache| cache.borrow().get(form.as_str()).and_then(|m| m.get(&inputs)).copied());
            if let Some(encoded) = cached {
                last = Some((inputs, encoded));
                previous = Some((lane, encoded));
                ops.put(io, 0, lane, u64::from(encoded), 32, false);
                continue;
            }
            let encoded = if block_scaled {
                // srcs: sfa_tmem_addr, sfb_tmem_addr (unused: a/b_sf_id = 0),
                // M, N, K, trans_a, trans_b, n_cta_groups, neg_a, neg_b, is_sparse.
                tcgen05::encode::encode_block_scaled_instr_descriptor_fields(
                    &dtypes[0],
                    &dtypes[1],
                    &dtypes[2],
                    &dtypes[3],
                    &dtypes[4],
                    v(2),
                    v(3),
                    v(4),
                    v(5) != 0,
                    v(6) != 0,
                    v(7),
                    v(8) != 0,
                    v(9) != 0,
                    v(10) != 0,
                )?
            } else {
                // srcs: M, N, K, trans_a, trans_b, n_cta_groups, neg_a, neg_b, sat_d, is_sparse.
                tcgen05::encode::encode_dense_instr_descriptor_fields(
                    &dtypes[0],
                    &dtypes[1],
                    &dtypes[2],
                    v(0),
                    v(1),
                    v(2),
                    v(3) != 0,
                    v(4) != 0,
                    v(5),
                    v(6) != 0,
                    v(7) != 0,
                    v(8) != 0,
                    v(9) != 0,
                )?
            };
            last = Some((inputs, encoded as u32));
            previous = Some((lane, encoded as u32));
            ENCODED.with(|cache| {
                let mut cache = cache.borrow_mut();
                let forms = cache.entry(form.clone()).or_default();
                if forms.len() >= 4096 {
                    forms.clear();
                }
                forms.insert(inputs, encoded as u32);
            });
            ops.put(io, 0, lane, encoded as u64, 32, false);
        }
        Ok(())
    })))
}

fn encode_matrix_descriptor(name: &str, mods: &Mods, ops: &Operands) -> OpResult<Resolved> {
    no_mods(mods, name)?;
    ops.arity(1, 4, name)?;
    if ops.dst_tys[0].bits() != 64 {
        return Err(OpError::unsupported(format!(
            "{name}: descriptor carrier {:?} is not 64-bit",
            ops.dst_tys[0]
        )));
    }
    integer_sources(ops, name)?;
    let address_bits = ops.src_tys[0].bits();
    if address_bits != 32 && address_bits != 64 {
        return Err(OpError::unsupported(format!(
            "{name}: address carrier {:?}",
            ops.src_tys[0]
        )));
    }
    let ops = ops.clone();
    Ok(Resolved::Boxed(Box::new(move |io: &mut PtxIo<'_>| {
        for lane in 0..WARP_SIZE {
            if !io.mask.contains(lane) {
                continue;
            }
            let raw = ops.src(io, 0, lane);
            // `__cvta_generic_to_shared(addr)` then `(addr & 0x3FFFF) >> 4`
            // with no validity check at encode time (V2C-10): kernels encode
            // a template from address 0 and patch the address field later,
            // and a consumer validates the descriptor when it is used. A
            // generic shared pointer contributes its window offset; any other
            // value (0 included) contributes its own low bits.
            let shared = if address_bits == 64 {
                match addr::classify_generic(raw) {
                    addr::Generic::Shared(offset) => offset,
                    _ => raw as u32,
                }
            } else {
                raw as u32
            };
            let v = |i: usize| int_value(ops.src_tys[i], ops.src(io, i, lane));
            // TVM's C signature takes `int ldo, int sdo, int swizzle`.
            let int = |i: usize| i64::from(v(i) as i32);
            let descriptor =
                tcgen05::encode::encode_matrix_descriptor(shared, int(1), int(2), int(3));
            ops.put(io, 0, lane, descriptor, 64, false);
        }
        Ok(())
    })))
}

pub(in crate::oplib) fn resolve(
    name: &str,
    mods: &Mods,
    ops: &Operands,
) -> OpResult<Option<Resolved>> {
    use Resolved::Direct;
    let direct =
        |f: crate::oplib::ptx::DirectFn, dsts: &[u32], srcs: &[u32]| -> OpResult<Option<Resolved>> {
            no_mods(mods, name)?;
            widths(ops, dsts, srcs, name)?;
            Ok(Some(Direct(f)))
        };
    match name {
        "tirx.cuda.make_float2" => direct(make_float2, &[64], &[32, 32]),
        "tirx.cuda.float2_x" => direct(float2_x, &[32], &[64]),
        "tirx.cuda.float2_y" => direct(float2_y, &[32], &[64]),
        "tirx.cuda.uint_as_float" | "tirx.cuda.float_as_uint" => direct(move32, &[32], &[32]),
        "tirx.cuda.ffs_u32" => direct(ffs_u32, &[32], &[32]),
        "tirx.cuda.float22bfloat162_rn" => direct(float22bfloat162_rn, &[32], &[32, 32]),
        "tirx.cuda.float22bfloat162_rn_from_float2" => direct(float22bfloat162_rn_from_float2, &[32], &[64]),
        "tirx.cuda.bfloat1622float2" => direct(bfloat1622float2, &[64], &[32]),
        "tirx.cuda.hmin2" | "tirx.cuda.hmax2" => {
            // Legacy lowers both on `uint32` operands as bf16x2; a `float16x2`
            // carrier selects the f16x2 form.
            let f16 = ops.src_tys.iter().all(|t| t.elem == Dtype::F16) && !ops.src_tys.is_empty();
            let f: crate::oplib::ptx::DirectFn = match (name == "tirx.cuda.hmin2", f16) {
                (true, false) => hmin2_bf16,
                (true, true) => hmin2_f16,
                (false, false) => hmax2_bf16,
                (false, true) => hmax2_f16,
            };
            direct(f, &[32], &[32, 32])
        }
        "tirx.cuda.fmul2_rn" => direct(fmul2_rn, &[64], &[64, 64]),
        "tirx.cuda.fadd2_rn" => direct(fadd2_rn, &[64], &[64, 64]),
        "tirx.cuda.fdividef" => direct(fdividef, &[32], &[32, 32]),
        "tirx.cuda.fp8x4_e4m3_from_float4" => direct(fp8x4_e4m3_from_float4, &[32], &[32, 32, 32, 32]),
        "tirx.cuda.half2float" => direct(half2float, &[32], &[16]),
        "tirx.cuda.bfloat162float" => direct(bfloat162float, &[32], &[16]),
        "tirx.cuda.clock64" => direct(clock64, &[64], &[]),
        "tirx.cuda.get_tmem_addr" => direct(get_tmem_addr, &[32], &[32, 32, 32]),
        "tirx.cuda.runtime_instr_desc" => direct(runtime_instr_desc, &[32], &[32, 32]),
        "tirx.cuda.tcgen05_encode_matrix_descriptor" => encode_matrix_descriptor(name, mods, ops).map(Some),
        "tirx.cuda.tcgen05_encode_instr_descriptor" => encode_instr_descriptor(name, mods, ops, false).map(Some),
        "tirx.cuda.tcgen05_encode_instr_descriptor_block_scaled" => {
            encode_instr_descriptor(name, mods, ops, true).map(Some)
        }
        "tirx.cuda.sm100_2sm_leader_smem_addr" => direct(sm100_2sm_leader_smem_addr, &[32], &[64]),
        "tirx.cuda.float22half2" | "tirx.cuda.float8tohalf8" | "tirx.cuda.half8tofloat8" => {
            Err(OpError::unsupported(format!(
                "{name}: pointer-based helper reads/writes memory; not a pure value op (needs a memory lowering)"
            )))
        }
        "tirx.fma" => {
            let elem = ops.dst_tys.first().map(|t| (t.elem, t.lanes));
            match elem {
                Some((Dtype::F32, 1)) => direct(fma_f32, &[32], &[32, 32, 32]),
                Some((Dtype::F64, 1)) => direct(fma_f64, &[64], &[64, 64, 64]),
                _ => Err(OpError::unsupported(format!("tirx.fma: unmodeled carrier {:?}", ops.dst_tys))),
            }
        }
        "tirx.log1p" | "tirx.sigmoid" => {
            // `numsim_oplib::scalar::{log1p,sigmoid}_*` (host rule, pinned
            // NaNs); f16/bf16 through f32 with RNE back (legacy: f32 only).
            no_mods(mods, name)?;
            ops.arity(1, 1, name)?;
            let (d, s) = (ops.dst_tys[0], ops.src_tys[0]);
            if d != s || d.lanes != 1 {
                return Err(OpError::unsupported(format!("{name}: expected one scalar float -> same type, got {s:?} -> {d:?}")));
            }
            let log1p = name == "tirx.log1p";
            let f: crate::oplib::ptx::DirectFn = match (d.elem, log1p) {
                (Dtype::F32, true) => |io| unary32(io, scalar::log1p_f32),
                (Dtype::F32, false) => |io| unary32(io, scalar::sigmoid_f32),
                (Dtype::F64, true) => |io| unary64(io, scalar::log1p_f64),
                (Dtype::F64, false) => |io| unary64(io, scalar::sigmoid_f64),
                (Dtype::F16, true) => |io| unary16(io, false, scalar::log1p_f32),
                (Dtype::F16, false) => |io| unary16(io, false, scalar::sigmoid_f32),
                (Dtype::BF16, true) => |io| unary16(io, true, scalar::log1p_f32),
                (Dtype::BF16, false) => |io| unary16(io, true, scalar::sigmoid_f32),
                _ => return Err(OpError::unsupported(format!("{name}: unsupported type {d:?}"))),
            };
            Ok(Some(Direct(f)))
        }
        "tirx.reinterpret" => {
            no_mods(mods, name)?;
            ops.arity(1, 1, name)?;
            if ops.dst_tys[0].bits() != ops.src_tys[0].bits() || ops.dst_tys[0].slots() != ops.src_tys[0].slots() {
                return Err(OpError::unsupported(format!(
                    "tirx.reinterpret: carrier widths differ ({:?} <- {:?})",
                    ops.dst_tys[0], ops.src_tys[0]
                )));
            }
            Ok(Some(Direct(reinterpret)))
        }
        _ => Ok(None),
    }
}
