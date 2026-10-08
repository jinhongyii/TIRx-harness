//! Ordering-only cache hints lowered to `Instr::Ptx` (W1 `lower_generic_
//! ordering`): `prefetch{.global,.local,.const,.param}{.L1,.L2}
//! {.L2::evict_*}{.tensormap}`, `prefetch.L1::32B.valid_addr`,
//! `prefetchu.L1`, every `applypriority*` form, and the non-tensor
//! `cp.async.bulk.prefetch.L2{.L2::cache_hint,.L2::evict_last}` (the tensor
//! prefetch is `Instr::Tma { dir: Prefetch }`).
//!
//! Legacy (`SUPPORTED_OPS.md`, family `ptx_cache_hint`, fidelity
//! `ordering_only`; `instructions/async_copy.rs` `prefetch_tensormap`) gave
//! them no numerical or synchronization effect: cache residency and eviction
//! are not modeled, and the instruction predicate gates operand evaluation.
//! OpLib therefore resolves them to a no-op after validating the modifiers
//! against the TVM table (unknown/duplicate/missing modifiers fail closed).
//! They produce no `Access`. Two legacy checks need engine state and are the
//! handler's job (CONTRACT_REQUESTS W4-9): `prefetch_valid_addr` validated
//! that the address names addressable global memory, and the
//! `applypriority.async.bulk*` forms joined the issuing thread's bulk async
//! group (`completion=bulk_group`).

use super::{Mods, Operands, Resolved};
use crate::oplib::{OpError, OpResult, PtxIo};

mod table;

use table::HINT_OPS;

pub(in crate::oplib) const NAMES: &[&str] = &{
    let mut names = [""; HINT_OPS.len()];
    let mut i = 0;
    while i < HINT_OPS.len() {
        names[i] = HINT_OPS[i].0;
        i += 1;
    }
    names
};

fn no_op(_io: &mut PtxIo<'_>) -> OpResult {
    Ok(())
}

/// Legacy `validate_global_cache_hint_address` alignment, on the address
/// value (the global aperture keeps allocation offsets' alignment).
fn aligned(io: &PtxIo<'_>, alignment: u64, label: &str) -> OpResult {
    for lane in io.mask.lanes() {
        let address = io.srcs[0][lane];
        if !address.is_multiple_of(alignment) {
            return Err(OpError::invalid(format!(
                "{label} requires a {alignment}-byte aligned global address on lane {lane}, got {address:#x}"
            )));
        }
    }
    Ok(())
}

/// `applypriority.global.L2::evict_normal [a], 128`: 128-byte aligned.
fn applypriority(io: &mut PtxIo<'_>) -> OpResult {
    aligned(io, 128, "applypriority")
}

/// `cp.async.bulk.prefetch.L2 [a], size`: legacy
/// `validate_bulk_cache_hint_range(.., 16, ..)` — size a multiple of 16 and
/// a 16-byte aligned address.
fn bulk_prefetch(io: &mut PtxIo<'_>) -> OpResult {
    for lane in io.mask.lanes() {
        let size = io.srcs[1][lane] as u32;
        if !size.is_multiple_of(16) {
            return Err(OpError::invalid(format!(
                "cp.async.bulk.prefetch size {size} must be a multiple of 16 on lane {lane}"
            )));
        }
    }
    aligned(io, 16, "cp.async.bulk.prefetch")
}

pub(in crate::oplib) fn resolve(name: &str, mods: &Mods, ops: &Operands) -> OpResult<Option<Resolved>> {
    let Some(&(_, slots)) = HINT_OPS.iter().find(|(op, _)| *op == name) else {
        return Ok(None);
    };
    mods.normalize(slots, name)?;
    if !ops.dst_tys.is_empty() {
        return Err(OpError::unsupported(format!("{name}: a cache hint has no destination")));
    }
    Ok(Some(Resolved::Direct(match name {
        "tirx.ptx.applypriority" if !ops.src_tys.is_empty() => applypriority,
        "tirx.ptx.cp_async_bulk_prefetch" | "tirx.ptx.cp_async_bulk_prefetch_evict_last" => {
            if ops.src_tys.len() < 2 {
                return Err(OpError::unsupported(format!("{name}: expected address and size operands")));
            }
            bulk_prefetch
        }
        _ => no_op,
    })))
}

#[cfg(test)]
mod tests {
    use crate::dtype::Ty;
    use crate::oplib::{resolve_ptx, OpErrorKind, PtxIo};
    use crate::program::OpKey;
    use crate::value::WarpMask;

    fn key(name: &str, mods: &[&str]) -> OpKey {
        OpKey { name: name.into(), mods: mods.iter().map(|m| m.to_string()).collect() }
    }

    #[test]
    fn prefetch_and_applypriority_are_validated_no_ops() {
        for (name, mods, srcs) in [
            ("tirx.ptx.prefetch", vec!["global", "L2"], vec![Ty::U64]),
            ("tirx.ptx.prefetch", vec!["tensormap"], vec![Ty::U64]),
            ("tirx.ptx.prefetch", vec!["space=global", "level=L2", "evict=L2::evict_last"], vec![Ty::U64]),
            ("tirx.ptx.prefetchu", vec!["L1"], vec![Ty::U64]),
            ("tirx.ptx.prefetch_valid_addr", vec!["global", "L1::32B", "valid_addr"], vec![Ty::U64]),
            ("tirx.ptx.applypriority", vec!["global", "L2::evict_normal"], vec![Ty::U64, Ty::U32]),
            ("tirx.ptx.cp_async_bulk_prefetch_evict_last", vec!["async", "bulk", "prefetch", "L2", "global", "L2::evict_last"], vec![Ty::U64, Ty::U32]),
        ] {
            let f = resolve_ptx(&key(name, &mods), &[], &srcs).unwrap_or_else(|e| panic!("{name} {mods:?}: {e}"));
            assert!(f.is_direct());
            let srcs_v = vec![[0x1000u64; 32]; srcs.len()];
            let mut dsts = vec![];
            let mut io = PtxIo { dsts: &mut dsts, dst_tys: &[], srcs: &srcs_v, src_tys: &srcs, mask: WarpMask::ALL };
            f.call(&mut io).unwrap();
        }
        // cp.async.bulk.prefetch: legacy size/alignment validation.
        let f = resolve_ptx(
            &key("tirx.ptx.cp_async_bulk_prefetch", &["async", "bulk", "prefetch", "L2", "global"]),
            &[],
            &[Ty::U64, Ty::U32],
        )
        .unwrap();
        let run = |addr: u64, size: u64| {
            let srcs = vec![[addr; 32], [size; 32]];
            let mut dsts = vec![];
            let mut io = PtxIo { dsts: &mut dsts, dst_tys: &[], srcs: &srcs, src_tys: &[Ty::U64, Ty::U32], mask: WarpMask(1) };
            f.call(&mut io)
        };
        assert!(run(0x1000, 256).is_ok());
        assert_eq!(run(0x1000, 24).unwrap_err().kind, OpErrorKind::Invalid);
        assert_eq!(run(0x1008, 32).unwrap_err().kind, OpErrorKind::Invalid);
        let bad = resolve_ptx(&key("tirx.ptx.prefetch", &["L3"]), &[], &[Ty::U64]).unwrap_err();
        assert_eq!(bad.kind, OpErrorKind::Unsupported);
        assert!(resolve_ptx(&key("tirx.ptx.prefetch_valid_addr", &["global"]), &[], &[Ty::U64]).is_err());
        assert!(resolve_ptx(&key("tirx.ptx.prefetch", &["L2"]), &[Ty::U32], &[Ty::U64]).is_err());
    }
}
