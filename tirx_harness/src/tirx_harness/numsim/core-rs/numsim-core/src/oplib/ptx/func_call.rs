//! Reviewed CUDA `func_call` helpers. W1 lowers them as
//! `Ptx "tirx.cuda.func_call.<name>"` with modifier
//! `source_sha256=<digest>`. The digest is the first 16 hex digits of the
//! sha256 of the whitespace-free helper body (`v2/lowering/builtins.py`
//! `REVIEWED_HELPERS`).
//!
//! Semantics follow the legacy frontend (`frontend-rs/src/emit/cuda_helper.rs`
//! `emit_known_cuda_func_call`). Each body here is the legacy expression for
//! that helper. A helper resolves only with its reviewed digest, so a body
//! that differs fails closed here as well as in lowering.

use super::{Mods, Operands, Resolved};
use crate::oplib::{OpError, OpResult, PtxIo};
use numsim_oplib::{cvt, scalar};
use numsim_types::WARP_SIZE;

pub(in crate::oplib) const NAMES: &[&str] = &[
    "tirx.cuda.func_call.combine_int_frac_ex2",
    "tirx.cuda.func_call.flashkda_fmaf_rn",
    "tirx.cuda.func_call.flashkda_rsqrtf",
    "tirx.cuda.func_call.flashkda_tanh_approx",
    "tirx.cuda.func_call.gdn_lg2_approx_ftz",
    "tirx.cuda.func_call.shl_u32_clamp",
    "tirx.cuda.func_call.tvm_builtin_fma_scale_sub_f32x2",
    "tirx.cuda.func_call.tvm_builtin_smem_desc_add_16B_offset",
];

/// (helper, reviewed body digest, destination widths, source widths, body).
type Body = fn(&mut PtxIo<'_>) -> OpResult;
const REVIEWED: &[(&str, &str, &[u32], &[u32], Body)] = &[
    ("combine_int_frac_ex2", "4b134dcd27e35d98", &[32], &[32, 32], combine_int_frac_ex2),
    ("flashkda_fmaf_rn", "f8e53464d3e437cc", &[32], &[32, 32, 32], flashkda_fmaf_rn),
    ("flashkda_rsqrtf", "3c5001e580bc296a", &[32], &[32], flashkda_rsqrtf),
    ("flashkda_tanh_approx", "571e03c6d4d0c042", &[32], &[32], flashkda_tanh_approx),
    ("gdn_lg2_approx_ftz", "cb5de3ec432d0b59", &[32], &[32], gdn_lg2_approx_ftz),
    ("shl_u32_clamp", "4e9357a6d632e267", &[32], &[32, 32], shl_u32_clamp),
    ("tvm_builtin_fma_scale_sub_f32x2", "ee5c4b843f718df3", &[64], &[64, 64, 64], fma_scale_sub_f32x2),
    ("tvm_builtin_smem_desc_add_16B_offset", "58c07995af0dadea", &[64], &[64, 32], smem_desc_add_16b_offset),
];

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

/// `shl.b32 x_rounded, 23` then `add.s32` with the fraction's bits.
fn combine_int_frac_ex2(io: &mut PtxIo<'_>) -> OpResult {
    each_lane!(io, |lane| io.dsts[0][lane] = u64::from(
        (io.srcs[0][lane] as u32).wrapping_shl(23).wrapping_add(io.srcs[1][lane] as u32)
    ));
    Ok(())
}

/// `__fmaf_rn`: one rounding.
fn flashkda_fmaf_rn(io: &mut PtxIo<'_>) -> OpResult {
    each_lane!(io, |lane| io.dsts[0][lane] =
        bits32(f32_at(io, 0, lane).mul_add(f32_at(io, 1, lane), f32_at(io, 2, lane))));
    Ok(())
}

/// `rsqrtf`: legacy `1.0 / x.sqrt()` (two IEEE roundings).
fn flashkda_rsqrtf(io: &mut PtxIo<'_>) -> OpResult {
    each_lane!(io, |lane| io.dsts[0][lane] = bits32(1.0_f32 / f32_at(io, 0, lane).sqrt()));
    Ok(())
}

/// `tanh.approx.f32`: the deterministic representative.
fn flashkda_tanh_approx(io: &mut PtxIo<'_>) -> OpResult {
    each_lane!(io, |lane| io.dsts[0][lane] = bits32(scalar::ptx_tanh_approx_f32(f32_at(io, 0, lane))));
    Ok(())
}

/// `lg2.approx.ftz.f32`: subnormal input and result flush to zero.
fn gdn_lg2_approx_ftz(io: &mut PtxIo<'_>) -> OpResult {
    each_lane!(io, |lane| io.dsts[0][lane] = bits32(scalar::ptx_lg2_approx_ftz_f32(f32_at(io, 0, lane))));
    Ok(())
}

/// `shl.b32`: a shift count of 32 or more gives 0 (PTX clamps).
fn shl_u32_clamp(io: &mut PtxIo<'_>) -> OpResult {
    each_lane!(io, |lane| io.dsts[0][lane] =
        u64::from((io.srcs[0][lane] as u32).checked_shl(io.srcs[1][lane] as u32).unwrap_or(0)));
    Ok(())
}

/// Per packed f32 lane: `fmaf(score, scale, -lse)` (x = low word).
fn fma_scale_sub_f32x2(io: &mut PtxIo<'_>) -> OpResult {
    each_lane!(io, |lane| {
        let (s, k, l) = (io.srcs[0][lane], io.srcs[1][lane], io.srcs[2][lane]);
        io.dsts[0][lane] = cvt::make_float2(
            cvt::float2_x(s).mul_add(cvt::float2_x(k), -cvt::float2_x(l)),
            cvt::float2_y(s).mul_add(cvt::float2_y(k), -cvt::float2_y(l)),
        );
    });
    Ok(())
}

/// `desc.lo += (uint32_t)offset`: wraps within the low 32 bits only.
fn smem_desc_add_16b_offset(io: &mut PtxIo<'_>) -> OpResult {
    each_lane!(io, |lane| {
        let desc = io.srcs[0][lane];
        let lo = (desc as u32).wrapping_add(io.srcs[1][lane] as u32);
        io.dsts[0][lane] = (desc & 0xffff_ffff_0000_0000) | u64::from(lo);
    });
    Ok(())
}

pub(in crate::oplib) fn resolve(name: &str, mods: &Mods, ops: &Operands) -> OpResult<Option<Resolved>> {
    let Some(helper) = name.strip_prefix("tirx.cuda.func_call.") else {
        return Ok(None);
    };
    let Some(&(_, digest, dsts, srcs, body)) = REVIEWED.iter().find(|r| r.0 == helper) else {
        if helper.starts_with("tvm_builtin_cast_") {
            return Err(OpError::unsupported(format!(
                "{name}: pointer-based helper reads/writes memory; not a pure value op (needs a memory lowering)"
            )));
        }
        return Err(OpError::unsupported(format!("{name}: no reviewed implementation")));
    };
    let mut seen = None;
    for (slot, token) in &mods.pairs {
        if slot != "source_sha256" || seen.is_some() {
            return Err(OpError::unsupported(format!("{name}: unexpected modifier {slot}={token}")));
        }
        seen = Some(token.as_str());
    }
    if seen != Some(digest) {
        return Err(OpError::unsupported(format!(
            "{name}: helper body digest {seen:?} is not the reviewed {digest}"
        )));
    }
    ops.arity(dsts.len(), srcs.len(), name)?;
    let got = |tys: &[crate::dtype::Ty]| tys.iter().map(|t| t.bits()).collect::<Vec<_>>();
    if got(&ops.dst_tys) != dsts || got(&ops.src_tys) != srcs {
        return Err(OpError::unsupported(format!(
            "{name}: expected carrier widths {dsts:?} <- {srcs:?}, got {:?} <- {:?}",
            ops.dst_tys, ops.src_tys
        )));
    }
    Ok(Some(Resolved::Direct(body)))
}

#[cfg(test)]
#[path = "func_call_tests.rs"]
mod tests;
