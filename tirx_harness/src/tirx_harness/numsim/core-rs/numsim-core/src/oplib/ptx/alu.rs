//! ALU family: every non-`cvt` pure-register op of the TIRx PTX table
//! (arithmetic, logic, bit manipulation, compare/select, moves and packs,
//! approximate transcendentals, mixed-precision vectors, sparse
//! (de)compression and `createpolicy`).
//!
//! Numerics delegate to `numsim_oplib::{arith, scalar, cvt}`; this module only
//! maps a modifier spelling to the kernel parameters the legacy engine variant
//! of that spelling used (frontend-rs `emit/ptx_*.rs` + engine-rs
//! `runtime/instructions/reg*.rs`). A spelling the legacy engine had no
//! variant for, or whose modifier it silently dropped, is `Unsupported`.
//!
//! Modifiers are normalised against [`table::TABLE`] first: `slot=token`
//! pairs are checked against the slot's choices, bare tokens are assigned to
//! slots positionally (the lowering emits them in table slot order), and a
//! missing required slot or an unknown token fails closed.
//!
//! Hot forms resolve to monomorphic `Resolved::Direct` fns (no per-call
//! parsing) when every operand is a one-slot register and the destination
//! carrier needs no narrowing or sign extension; everything else is a boxed
//! per-lane closure over [`Operands`].

mod bits;
mod compare;
mod float;
mod half;
mod int;
mod movs;
mod table;
#[cfg(test)]
mod tests;

use super::{Mods, Operands, Resolved};
use crate::oplib::{OpError, OpResult, PtxIo};
use numsim_oplib::scalar::{F32RoundingMode, LowPrecisionFormat};

pub(in crate::oplib) const NAMES: &[&str] = &{
    let mut names = [""; table::TABLE.len()];
    let mut index = 0;
    while index < table::TABLE.len() {
        names[index] = table::TABLE[index].0;
        index += 1;
    }
    names
};

pub(in crate::oplib) fn resolve(
    name: &str,
    mods: &Mods,
    ops: &Operands,
) -> OpResult<Option<Resolved>> {
    let Some(m) = Md::normalize(name, mods)? else {
        return Ok(None);
    };
    let short = &name["tirx.ptx.".len()..];
    let resolved = match short {
        "add" | "sub" | "mul" | "fma" | "mad_f" | "div_f" | "copysign" | "neg" | "abs_f"
        | "rcp" | "sqrt" | "rsqrt" | "sin" | "cos" | "ex2" | "lg2" | "tanh" | "max3" | "min3" => {
            float::resolve(short, &m, ops)?
        }
        "max" | "min" => match m.get("type") {
            "f32" | "f64" => float::resolve(short, &m, ops)?,
            "f16" | "f16x2" | "bf16" | "bf16x2" => half::resolve(short, &m, ops)?,
            _ => int::resolve(short, &m, ops)?,
        },
        "add_half"
        | "sub_half"
        | "mul_half"
        | "fma_half"
        | "neg_half"
        | "abs_half"
        | "ex2_half"
        | "tanh_half"
        | "add_mixed_vec_up"
        | "sub_mixed_vec_up"
        | "fma_mixed_vec"
        | "add_mixed_vec_down_f16"
        | "add_mixed_vec_down_bf16"
        | "sub_mixed_vec_down_f16"
        | "sub_mixed_vec_down_bf16"
        | "mul_mixed_vec_down_f16"
        | "mul_mixed_vec_down_bf16"
        | "mul_mixed_vec_bf16_f16"
        | "mul_mixed_vec_f16_bf16" => half::resolve(short, &m, ops)?,
        "add_int" | "sub_int" | "mul_int" | "mad_int" | "mul_wide" | "mad_wide" | "mul24"
        | "mad24" | "sad" | "div" | "rem" | "neg_int" | "abs" | "dp2a" | "dp4a" => {
            int::resolve(short, &m, ops)?
        }
        "and" | "or" | "xor" | "not" | "cnot" | "brev" | "clz" | "popc" | "bfe" | "bfi"
        | "bfind" | "bmsk" | "clmad" | "shf" | "shl" | "shr" | "szext" | "prmt" | "fns"
        | "lop3" | "lop3_bool" | "lop3_bool_sink" => bits::resolve(short, &m, ops)?,
        "setp" | "setp_bool" | "setp_pq" | "setp_bool_pq" | "setp_half" | "setp_half_bool"
        | "setp_half_pq" | "setp_half_bool_pq" | "set" | "set_bool" | "set_half"
        | "set_half_bool" | "set_packed" | "selp" | "slct" | "testp" => {
            compare::resolve(short, &m, ops)?
        }
        _ => movs::resolve(short, &m, ops)?,
    };
    Ok(Some(resolved))
}

// ---------------------------------------------------------------------------
// Modifier normalisation
// ---------------------------------------------------------------------------

/// Modifiers of one ALU op, keyed by table slot.
#[derive(Clone, Debug)]
pub(super) struct Md {
    pub op: &'static str,
    pub vals: Vec<(&'static str, String)>,
}

impl Md {
    /// `Ok(None)` for an op this family does not own.
    fn normalize(name: &str, mods: &Mods) -> OpResult<Option<Md>> {
        let Some(&(op, slots)) = table::TABLE.iter().find(|(op, _)| *op == name) else {
            return Ok(None);
        };
        let mut assigned: Vec<Option<String>> = vec![None; slots.len()];
        let mut cursor = 0usize;
        for (slot, token) in &mods.pairs {
            let index = if slot.is_empty() {
                (cursor..slots.len())
                    .find(|&i| assigned[i].is_none() && slots[i].2.contains(&token.as_str()))
                    .ok_or_else(|| {
                        OpError::unsupported(format!("{op}: unmodeled modifier `{token}`"))
                    })?
            } else {
                let i = slots
                    .iter()
                    .position(|(s, _, _)| s == slot)
                    .ok_or_else(|| {
                        OpError::unsupported(format!("{op}: unmodeled modifier {slot}={token}"))
                    })?;
                if !slots[i].2.contains(&token.as_str()) {
                    return Err(OpError::unsupported(format!(
                        "{op}: unmodeled modifier {slot}={token}"
                    )));
                }
                i
            };
            if assigned[index].is_some() {
                return Err(OpError::unsupported(format!(
                    "{op}: duplicate modifier slot `{}`",
                    slots[index].0
                )));
            }
            assigned[index] = Some(token.clone());
            cursor = index + 1;
        }
        let mut vals = Vec::new();
        for (spec, token) in slots.iter().zip(assigned) {
            match token {
                Some(token) => vals.push((spec.0, token)),
                None if !spec.1 => {
                    return Err(OpError::unsupported(format!(
                        "{op}: missing modifier slot `{}`",
                        spec.0
                    )));
                }
                None => {}
            }
        }
        Ok(Some(Md { op, vals }))
    }

    /// Token of `slot`, `""` when absent.
    pub fn get(&self, slot: &str) -> &str {
        self.vals
            .iter()
            .find(|(s, _)| *s == slot)
            .map(|(_, t)| t.as_str())
            .unwrap_or("")
    }

    /// Whether optional `slot` is present.
    pub fn has(&self, slot: &str) -> bool {
        self.vals.iter().any(|(s, _)| *s == slot)
    }

    /// Fail closed: `what` is not a modeled spelling.
    pub fn unsupported<T>(&self, what: impl std::fmt::Display) -> OpResult<T> {
        Err(OpError::unsupported(format!(
            "{}: {what} (modifiers {:?})",
            self.op, self.vals
        )))
    }
}

// ---------------------------------------------------------------------------
// Shared parameter decoding
// ---------------------------------------------------------------------------

/// `.rn/.rz/.rm/.rp`; an absent rounding modifier means RN (legacy
/// `round_marker("") = Rn`).
pub(super) fn rounding(m: &Md, token: &str) -> OpResult<F32RoundingMode> {
    Ok(match token {
        "" | "rn" => F32RoundingMode::Nearest,
        "rz" => F32RoundingMode::Zero,
        "rm" => F32RoundingMode::Down,
        "rp" => F32RoundingMode::Up,
        other => return m.unsupported(format!("rounding `{other}`")),
    })
}

/// Half format and packing of a `.f16/.bf16{x2}` token.
pub(super) fn half_format(m: &Md, token: &str) -> OpResult<(LowPrecisionFormat, bool)> {
    Ok(match token {
        "f16" => (LowPrecisionFormat::F16, false),
        "f16x2" => (LowPrecisionFormat::F16, true),
        "bf16" => (LowPrecisionFormat::Bf16, false),
        "bf16x2" => (LowPrecisionFormat::Bf16, true),
        other => return m.unsupported(format!("half type `{other}`")),
    })
}

/// Integer PTX type token -> (bits, signed).
pub(super) fn int_type(m: &Md, token: &str) -> OpResult<(u32, bool)> {
    Ok(match token {
        "u8" | "b8" => (8, false),
        "s8" => (8, true),
        "u16" | "b16" => (16, false),
        "s16" => (16, true),
        "u32" | "b32" => (32, false),
        "s32" => (32, true),
        "u64" | "b64" => (64, false),
        "s64" => (64, true),
        other => return m.unsupported(format!("integer type `{other}`")),
    })
}

#[inline(always)]
pub(super) fn mask(bits: u32) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

/// Sign-extend the low `bits` of `value` to i64.
#[inline(always)]
pub(super) fn sext(value: u64, bits: u32) -> i64 {
    if bits >= 64 {
        value as i64
    } else {
        let shift = 64 - bits;
        ((value << shift) as i64) >> shift
    }
}

#[inline(always)]
pub(super) fn f32_of(value: u64) -> f32 {
    f32::from_bits(value as u32)
}

#[inline(always)]
pub(super) fn bits_f32(value: f32) -> u64 {
    u64::from(value.to_bits())
}

// ---------------------------------------------------------------------------
// Per-lane closure builders
// ---------------------------------------------------------------------------

/// Boxed op over `N` one-value sources (low 64 bits each) writing `M`
/// destinations, destination `j` as a PTX value of `outs[j] = (bits, signed)`.
pub(super) fn lanes<const N: usize, const M: usize, F>(
    m: &Md,
    ops: &Operands,
    outs: [(u32, bool); M],
    f: F,
) -> OpResult<Resolved>
where
    F: Fn([u64; N]) -> OpResult<[u64; M]> + Send + Sync + 'static,
{
    ops.arity(M, N, m.op)?;
    let ops = ops.clone();
    Ok(Resolved::Boxed(Box::new(move |io: &mut PtxIo<'_>| {
        for lane in io.mask.lanes() {
            let args: [u64; N] = std::array::from_fn(|i| ops.src(io, i, lane));
            let out = f(args)?;
            for (j, value) in out.into_iter().enumerate() {
                ops.put(io, j, lane, value, outs[j].0, outs[j].1);
            }
        }
        Ok(())
    })))
}

/// One fallible destination.
pub(super) fn try_map<const N: usize, F>(
    m: &Md,
    ops: &Operands,
    bits: u32,
    signed: bool,
    f: F,
) -> OpResult<Resolved>
where
    F: Fn([u64; N]) -> OpResult<u64> + Send + Sync + 'static,
{
    lanes::<N, 1, _>(m, ops, [(bits, signed)], move |args| f(args).map(|v| [v]))
}

/// One infallible destination.
pub(super) fn map<const N: usize, F>(
    m: &Md,
    ops: &Operands,
    bits: u32,
    signed: bool,
    f: F,
) -> OpResult<Resolved>
where
    F: Fn([u64; N]) -> u64 + Send + Sync + 'static,
{
    lanes::<N, 1, _>(m, ops, [(bits, signed)], move |args| Ok([f(args)]))
}

/// Whether a direct fn may index `io.srcs[i]` / `io.dsts[j]` as one slot per
/// operand and store an unsigned `dst_bits` result without masking: exact
/// arity, one slot each, destination carriers of `dst_bits..=64` bits.
pub(super) fn flat(ops: &Operands, dsts: usize, srcs: usize, dst_bits: u32) -> bool {
    ops.dst_tys.len() == dsts
        && ops.src_tys.len() == srcs
        && ops.src_tys.iter().all(|t| t.slots() == 1)
        && ops
            .dst_tys
            .iter()
            .all(|t| t.slots() == 1 && t.bits() >= dst_bits && t.bits() <= 64)
}

/// Like [`flat`] for a signed `dst_bits` result: the carrier must be exactly
/// `dst_bits` wide (no sign extension needed).
pub(super) fn flat_exact(ops: &Operands, dsts: usize, srcs: usize, dst_bits: u32) -> bool {
    flat(ops, dsts, srcs, dst_bits) && ops.dst_tys.iter().all(|t| t.bits() == dst_bits)
}
