//! TIR-level ALU (`Instr::Unary/Binary/Ternary/Compare/Cast`) and single-value
//! conversions (`convert_bits`, `FloatScalar::from_f64` for f32).
//!
//! Semantics are those the legacy NumSim frontend emitted for the TIR nodes
//! (host Rust with C/TVM semantics); see `tir/alu.rs` and `tir/convert.rs` for
//! the per-dtype rules. Vector `Ty` (lanes > 1) is element-wise over densely
//! packed elements (element 0 in the low bits). Every form is resolved before
//! touching lanes, so an unmodeled (op, dtype) is `Unsupported` even under an
//! empty mask. Outputs are written only for lanes in `mask`.
//!
//! Result types: `Unary` writes a value of `ty`, except `IsNan/IsInf/IsFinite`
//! which write `Ty{Pred, ty.lanes}` (0/1 per element). `Binary`/`Ternary`
//! write `ty`. `Compare` takes scalar operands of `ty` and returns the mask.

mod alu;
mod convert;
mod elem;
#[cfg(test)]
mod tests;

use super::{OpError, OpErrorKind, OpResult};
use crate::dtype::{Dtype, Ty};
use crate::program::{BinOp, CmpOp, Rounding, TerOp, UnOp};
use crate::value::{WarpMask, WarpValue};
use elem::{check_slots, get, load, put, store, Packed};

pub(super) use convert::f32_from_f64;

/// Propagate only `Unsupported` from a probe evaluation (value-dependent
/// `Invalid` errors are reported per lane).
#[inline]
fn probe<T>(r: OpResult<T>) -> OpResult {
    match r {
        Err(e) if e.kind == OpErrorKind::Unsupported => Err(e),
        _ => Ok(()),
    }
}

/// Map `f` over every element of every active lane: inputs of `in_ty`,
/// output of `out_ty` (same lane count).
#[inline]
fn map_elems<const N: usize>(
    in_ty: Ty,
    out_ty: Ty,
    ins: [&[WarpValue<u64>]; N],
    out: &mut [WarpValue<u64>],
    mask: WarpMask,
    mut f: impl FnMut([u128; N]) -> OpResult<u128>,
) -> OpResult {
    let (ib, ob) = (in_ty.elem.bits(), out_ty.elem.bits());
    for lane in mask.lanes() {
        let vals: [Packed; N] = std::array::from_fn(|k| load(ins[k], in_ty, lane));
        let mut r: Packed = [0; 4];
        for e in 0..in_ty.lanes as usize {
            let x: [u128; N] = std::array::from_fn(|k| get(&vals[k], e, ib));
            put(&mut r, e, ob, f(x)?);
        }
        store(out, out_ty, lane, &r);
    }
    Ok(())
}

pub(super) fn unary(op: UnOp, ty: Ty, a: &[WarpValue<u64>], out: &mut [WarpValue<u64>], mask: WarpMask) -> OpResult {
    let out_ty = if alu::unary_yields_pred(op) { Ty::vector(Dtype::Pred, ty.lanes) } else { ty };
    check_slots(a.len(), ty, "unary operand")?;
    check_slots(out.len(), out_ty, "unary result")?;
    probe(alu::unary(op, ty.elem, 0))?;
    map_elems(ty, out_ty, [a], out, mask, |[x]| alu::unary(op, ty.elem, x))
}

pub(super) fn binary(
    op: BinOp,
    ty: Ty,
    a: &[WarpValue<u64>],
    b: &[WarpValue<u64>],
    out: &mut [WarpValue<u64>],
    mask: WarpMask,
) -> OpResult {
    check_slots(a.len(), ty, "binary lhs")?;
    check_slots(b.len(), ty, "binary rhs")?;
    check_slots(out.len(), ty, "binary result")?;
    probe(alu::binary(op, ty.elem, 0, 1))?;
    map_elems(ty, ty, [a, b], out, mask, |[x, y]| alu::binary(op, ty.elem, x, y))
}

pub(super) fn ternary(
    op: TerOp,
    ty: Ty,
    a: &[WarpValue<u64>],
    b: &[WarpValue<u64>],
    c: &[WarpValue<u64>],
    out: &mut [WarpValue<u64>],
    mask: WarpMask,
) -> OpResult {
    check_slots(a.len(), ty, "ternary a")?;
    check_slots(b.len(), ty, "ternary b")?;
    check_slots(c.len(), ty, "ternary c")?;
    check_slots(out.len(), ty, "ternary result")?;
    probe(alu::ternary(op, ty.elem, 0, 1, 1))?;
    map_elems(ty, ty, [a, b, c], out, mask, |[x, y, z]| alu::ternary(op, ty.elem, x, y, z))
}

pub(super) fn compare(op: CmpOp, ty: Ty, a: &[WarpValue<u64>], b: &[WarpValue<u64>], mask: WarpMask) -> OpResult<WarpMask> {
    if !ty.is_scalar() {
        return Err(OpError::unsupported(format!("compare {op:?} on vector {ty}")));
    }
    check_slots(a.len(), ty, "compare lhs")?;
    check_slots(b.len(), ty, "compare rhs")?;
    probe(alu::compare(op, ty.elem, 0, 0))?;
    let bits = ty.elem.bits();
    let mut result = 0u32;
    for lane in mask.lanes() {
        let x = get(&load(a, ty, lane), 0, bits);
        let y = get(&load(b, ty, lane), 0, bits);
        if alu::compare(op, ty.elem, x, y)? {
            result |= 1 << lane;
        }
    }
    Ok(WarpMask(result))
}

fn check_cast(from: Ty, to: Ty) -> OpResult {
    if from.lanes != to.lanes {
        return Err(OpError::invalid(format!("cast {from} -> {to}: lane counts differ")));
    }
    Ok(())
}

pub(super) fn cast(
    from: Ty,
    to: Ty,
    rnd: Rounding,
    sat: bool,
    src: &[WarpValue<u64>],
    out: &mut [WarpValue<u64>],
    mask: WarpMask,
) -> OpResult {
    check_cast(from, to)?;
    check_slots(src.len(), from, "cast source")?;
    check_slots(out.len(), to, "cast result")?;
    probe(convert::cast_elem(from.elem, to.elem, rnd, sat, 0))?;
    map_elems(from, to, [src], out, mask, |[x]| convert::cast_elem(from.elem, to.elem, rnd, sat, x))
}

pub(super) fn convert_bits(from: Ty, to: Ty, rnd: Rounding, sat: bool, src: u128) -> OpResult<u128> {
    check_cast(from, to)?;
    if from.bits() > 128 || to.bits() > 128 {
        return Err(OpError::unsupported(format!("convert {from} -> {to}: wider than 128 bits")));
    }
    let v: Packed = [src as u64, (src >> 64) as u64, 0, 0];
    let mut r: Packed = [0; 4];
    let (ib, ob) = (from.elem.bits(), to.elem.bits());
    for e in 0..from.lanes as usize {
        put(&mut r, e, ob, convert::cast_elem(from.elem, to.elem, rnd, sat, get(&v, e, ib))?);
    }
    Ok(u128::from(r[0]) | (u128::from(r[1]) << 64))
}
