//! Bit-exact ALU cases through `resolve_ptx` (expected values from the
//! ported legacy tests in `numsim-oplib/src/{arith,scalar}/tests.rs` where
//! available), plus fail-closed checks.

pub(super) use crate::dtype::{Dtype, Ty};
pub(super) use crate::oplib::{resolve_ptx, OpErrorKind, OpResult, PtxIo};
pub(super) use crate::program::OpKey;
pub(super) use crate::value::{WarpMask, WarpValue};

pub(super) const POISON: u64 = 0xdead_beef_dead_beef;
pub(super) const S16: Ty = Ty::scalar(Dtype::S16);

pub(super) fn key(name: &str, mods: &[&str]) -> OpKey {
    OpKey {
        name: format!("tirx.ptx.{name}"),
        mods: mods.iter().map(|m| m.to_string()).collect(),
    }
}

/// Run on lanes 0 and 7 with identical source slots; return lane 0's
/// destination slots (checking lane 7 agrees and inactive lanes are untouched).
pub(super) fn run(
    name: &str,
    mods: &[&str],
    dst_tys: &[Ty],
    src_tys: &[Ty],
    srcs: &[u64],
) -> OpResult<Vec<u64>> {
    let f = resolve_ptx(&key(name, mods), dst_tys, src_tys)?;
    let dst_slots: usize = dst_tys.iter().map(|t| t.slots() as usize).sum();
    let src_slots: usize = src_tys.iter().map(|t| t.slots() as usize).sum();
    assert_eq!(src_slots, srcs.len(), "{name}: source slot count");
    let src_vals: Vec<WarpValue<u64>> = srcs.iter().map(|&v| [v; 32]).collect();
    let mut dst_vals: Vec<WarpValue<u64>> = vec![[POISON; 32]; dst_slots];
    let mut io = PtxIo {
        dsts: &mut dst_vals,
        dst_tys,
        srcs: &src_vals,
        src_tys,
        mask: WarpMask(1 | (1 << 7)),
    };
    f.call(&mut io)?;
    for slot in &dst_vals {
        assert_eq!(slot[0], slot[7], "{name}: lanes disagree");
        assert_eq!(slot[1], POISON, "{name}: inactive lane written");
    }
    Ok(dst_vals.iter().map(|slot| slot[0]).collect())
}

pub(super) fn ok(
    name: &str,
    mods: &[&str],
    dst_tys: &[Ty],
    src_tys: &[Ty],
    srcs: &[u64],
) -> Vec<u64> {
    run(name, mods, dst_tys, src_tys, srcs).unwrap_or_else(|e| panic!("{name} {mods:?}: {e}"))
}

pub(super) fn one(name: &str, mods: &[&str], dst: Ty, src_tys: &[Ty], srcs: &[u64]) -> u64 {
    ok(name, mods, &[dst], src_tys, srcs)[0]
}

pub(super) fn kind(result: OpResult<Vec<u64>>) -> OpErrorKind {
    result.expect_err("expected an error").kind
}

pub(super) fn f(v: f32) -> u64 {
    u64::from(v.to_bits())
}

pub(super) fn d(v: f64) -> u64 {
    v.to_bits()
}

pub(super) fn f2(x: f32, y: f32) -> u64 {
    f(x) | (f(y) << 32)
}

pub(super) const F32: Ty = Ty::F32;
pub(super) const U32: Ty = Ty::U32;
pub(super) const U16: Ty = Ty::U16;
pub(super) const U64: Ty = Ty::U64;
pub(super) const S32: Ty = Ty::S32;
pub(super) const PRED: Ty = Ty::PRED;

mod bits;
mod compare;
mod float;
mod half;
mod int;
mod mma_sp;
mod movs;

#[test]
fn fail_closed_on_unknown_ops_and_modifiers() {
    // Unknown op name: no family owns it.
    assert_eq!(
        kind(run("not_an_op", &[], &[U32], &[U32], &[0])),
        OpErrorKind::Unsupported
    );
    // Unknown token / unknown slot / out-of-domain token / missing required slot.
    assert_eq!(
        kind(run("add", &["rna", "f32"], &[F32], &[F32; 2], &[0, 0])),
        OpErrorKind::Unsupported
    );
    assert_eq!(
        kind(run(
            "add",
            &["rnd=rna", "type=f32"],
            &[F32],
            &[F32; 2],
            &[0, 0]
        )),
        OpErrorKind::Unsupported
    );
    assert_eq!(
        kind(run(
            "add",
            &["bogus=1", "type=f32"],
            &[F32],
            &[F32; 2],
            &[0, 0]
        )),
        OpErrorKind::Unsupported
    );
    assert_eq!(
        kind(run("fma", &["f32"], &[F32], &[F32; 3], &[0, 0, 0])),
        OpErrorKind::Unsupported
    );
    assert_eq!(
        kind(run("add", &["f32", "f32"], &[F32], &[F32; 2], &[0, 0])),
        OpErrorKind::Unsupported
    );
    // Wrong operand arity.
    assert_eq!(
        kind(run("add", &["f32"], &[F32], &[F32], &[0])),
        OpErrorKind::Unsupported
    );
    // cvt / helpers are not ALU-owned.
    assert!(super::NAMES.iter().all(|n| !n.starts_with("tirx.ptx.cvt")));
    assert_eq!(super::NAMES.len(), 110);
}

#[test]
fn hot_forms_are_direct_and_stable() {
    // Direct forms resolve to the same fn item every time, independent of
    // the interning table; boxed forms intern per (key, tys).
    let a = resolve_ptx(&key("mul", &["f32"]), &[F32], &[F32, F32]).unwrap();
    let b = resolve_ptx(&key("mul", &["rnd=rn", "type=f32"]), &[F32], &[F32, F32]).unwrap();
    assert!(a.is_direct() && a.same(&b));
    let narrow = resolve_ptx(&key("mul", &["f32"]), &[U16], &[F32, F32]).unwrap();
    assert!(!narrow.is_direct() && !a.same(&narrow));
}
