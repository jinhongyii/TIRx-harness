//! `clusterlaunchcontrol.query_cancel.{is_canceled,get_first_ctaid{::x,::y,
//! ::z,.v4}}`: pure decoding of the 16-byte `try_cancel` response.
//!
//! Response encoding (legacy `instructions/control.rs` `clc_try_cancel` +
//! `frontend-rs/src/emit/clc.rs`): bytes 0..4 hold the linear base CTA id of
//! the cancelled cluster (NumSim's logical launch domain is linear: x = that
//! id, y = z = 0); `0xFFFF_FFFF` when no cluster could be cancelled (legacy
//! and the v2 engine). Under an execution subset a non-resident cluster 0 can
//! be claimed (base CTA 0), so `is_canceled = low32 != u32::MAX` (W12-gaps
//! 6). Operands (W1
//! `lower_generic`): `is_canceled` dsts `[p]`, srcs `[response]`;
//! `get_first_ctaid::*` dsts `[d]`, srcs `[d, response]` (read-write `d`);
//! `.v4` dsts `[d0..d3]`, srcs `[d0..d3, response]`. The response is the last
//! source (one `b128` register, or a narrower carrier zero-extended).

use super::{Mods, Operands, Resolved};
use crate::oplib::{OpError, OpResult, PtxIo};

type Slot = (&'static str, &'static [&'static str], bool);

const IS_CANCELED: &[Slot] =
    &[("action", &["query_cancel"], false), ("query", &["is_canceled"], false), ("ptype", &["pred"], false), ("type", &["b128"], false)];
const FIRST_CTAID: &[Slot] = &[
    ("action", &["query_cancel"], false),
    ("query", &["get_first_ctaid::x", "get_first_ctaid::y", "get_first_ctaid::z"], false),
    ("dtype", &["b32"], false),
    ("type", &["b128"], false),
];
const FIRST_CTAID_V4: &[Slot] = &[
    ("action", &["query_cancel"], false),
    ("query", &["get_first_ctaid"], false),
    ("vec", &["v4"], false),
    ("dtype", &["b32"], false),
    ("type", &["b128"], false),
];

pub(in crate::oplib) const NAMES: &[&str] = &[
    "tirx.ptx.clusterlaunchcontrol_query_cancel_is_canceled",
    "tirx.ptx.clusterlaunchcontrol_query_cancel_get_first_ctaid",
    "tirx.ptx.clusterlaunchcontrol_query_cancel_get_first_ctaid_v4",
];

/// Low 32 bits of the response (the base CTA id word).
#[inline]
fn first_word(ops: &Operands, io: &PtxIo<'_>, lane: usize) -> u32 {
    let last = ops.src_tys.len() - 1;
    ops.src128(io, last, lane) as u32
}

#[inline]
fn canceled(word: u32) -> bool {
    word != u32::MAX
}

pub(in crate::oplib) fn resolve(name: &str, mods: &Mods, ops: &Operands) -> OpResult<Option<Resolved>> {
    let ops = ops.clone();
    Ok(Some(match name {
        "tirx.ptx.clusterlaunchcontrol_query_cancel_is_canceled" => {
            mods.normalize(IS_CANCELED, name)?;
            ops.arity(1, 1, name)?;
            Resolved::Boxed(Box::new(move |io| {
                for lane in io.mask.lanes() {
                    let p = canceled(first_word(&ops, io, lane));
                    ops.put(io, 0, lane, u64::from(p), 1, false);
                }
                Ok(())
            }))
        }
        "tirx.ptx.clusterlaunchcontrol_query_cancel_get_first_ctaid" => {
            let m = mods.normalize(FIRST_CTAID, name)?;
            ops.arity(1, 2, name)?;
            let is_x = m.get("query") == Some("get_first_ctaid::x");
            Resolved::Boxed(Box::new(move |io| {
                for lane in io.mask.lanes() {
                    let v = if is_x { first_word(&ops, io, lane) } else { 0 };
                    ops.put(io, 0, lane, u64::from(v), 32, false);
                }
                Ok(())
            }))
        }
        "tirx.ptx.clusterlaunchcontrol_query_cancel_get_first_ctaid_v4" => {
            mods.normalize(FIRST_CTAID_V4, name)?;
            if ops.dst_tys.len() != 4 || ops.src_tys.len() != 5 {
                return Err(OpError::unsupported(format!(
                    "{name}: expected 4 dst / 5 src registers, got {:?} / {:?}",
                    ops.dst_tys, ops.src_tys
                )));
            }
            Resolved::Boxed(Box::new(move |io| {
                for lane in io.mask.lanes() {
                    let x = first_word(&ops, io, lane);
                    for i in 0..4 {
                        ops.put(io, i, lane, u64::from(if i == 0 { x } else { 0 }), 32, false);
                    }
                }
                Ok(())
            }))
        }
        _ => return Ok(None),
    }))
}

#[cfg(test)]
mod tests {
    use crate::dtype::Ty;
    use crate::oplib::{resolve_ptx, PtxIo};
    use crate::program::OpKey;
    use crate::value::{WarpMask, WarpValue};

    fn key(name: &str, mods: &[&str]) -> OpKey {
        OpKey { name: name.into(), mods: mods.iter().map(|m| m.to_string()).collect() }
    }

    fn run(name: &str, mods: &[&str], dsts: &[Ty], srcs: &[Ty], vals: &[WarpValue<u64>]) -> Vec<WarpValue<u64>> {
        let f = resolve_ptx(&key(name, mods), dsts, srcs).unwrap_or_else(|e| panic!("{name}: {e}"));
        let mut out = vec![[0x5555u64; 32]; dsts.iter().map(|t| t.slots() as usize).sum()];
        let mut io = PtxIo { dsts: &mut out, dst_tys: dsts, srcs: vals, src_tys: srcs, mask: WarpMask(0b11) };
        f.call(&mut io).unwrap();
        out
    }

    #[test]
    fn query_cancel_decodes_the_no_cluster_sentinel() {
        // Lane 0: the "no cluster" sentinel 0xFFFF_FFFF; lane 1: also the
        // sentinel; others: a cancelled cluster at base CTA 6 (inactive lanes
        // untouched). Base CTA 0 is a real claim under a subset (W12-gaps 6).
        let mut lo = [6u64; 32];
        lo[0] = 0xffff_ffff;
        lo[1] = 0xffff_ffff;
        let resp = vec![lo, [0u64; 32]];
        let b128 = [Ty::B128];
        let p = run(
            "tirx.ptx.clusterlaunchcontrol_query_cancel_is_canceled",
            &["query_cancel", "is_canceled", "pred", "b128"],
            &[Ty::PRED],
            &b128,
            &resp,
        );
        assert_eq!((p[0][0], p[0][1], p[0][2]), (0, 0, 0x5555));
        let mut cancelled = resp.clone();
        cancelled[0][0] = 6;
        let p = run(
            "tirx.ptx.clusterlaunchcontrol_query_cancel_is_canceled",
            &["query_cancel", "is_canceled", "pred", "b128"],
            &[Ty::PRED],
            &b128,
            &cancelled,
        );
        assert_eq!(p[0][0], 1);
        let mut zero = resp.clone();
        zero[0][0] = 0;
        let p = run(
            "tirx.ptx.clusterlaunchcontrol_query_cancel_is_canceled",
            &["query_cancel", "is_canceled", "pred", "b128"],
            &[Ty::PRED],
            &b128,
            &zero,
        );
        assert_eq!(p[0][0], 1, "base CTA 0 is a cancelled cluster");
        let mut srcs = vec![[7u64; 32]];
        srcs.extend(cancelled.clone());
        for (axis, want) in [("get_first_ctaid::x", 6), ("get_first_ctaid::y", 0), ("get_first_ctaid::z", 0)] {
            let d = run(
                "tirx.ptx.clusterlaunchcontrol_query_cancel_get_first_ctaid",
                &["query_cancel", axis, "b32", "b128"],
                &[Ty::U32],
                &[Ty::U32, Ty::B128],
                &srcs,
            );
            assert_eq!(d[0][0], want, "{axis}");
        }
        let mut srcs4 = vec![[7u64; 32]; 4];
        srcs4.extend(cancelled);
        let d = run(
            "tirx.ptx.clusterlaunchcontrol_query_cancel_get_first_ctaid_v4",
            &["query_cancel", "get_first_ctaid", "v4", "b32", "b128"],
            &[Ty::U32; 4],
            &[Ty::U32, Ty::U32, Ty::U32, Ty::U32, Ty::B128],
            &srcs4,
        );
        assert_eq!((d[0][0], d[1][0], d[2][0], d[3][0]), (6, 0, 0, 0));
        assert!(resolve_ptx(
            &key("tirx.ptx.clusterlaunchcontrol_query_cancel_is_canceled", &["query_cancel", "is_canceled", "b32", "b128"]),
            &[Ty::PRED],
            &[Ty::B128]
        )
        .is_err());
    }
}
