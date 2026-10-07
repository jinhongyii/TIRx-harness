//! Structural fingerprint of a single-resource projection.
//!
//! Two projections with equal fingerprints have isomorphic state spaces
//! (identical per-warp programs up to warp renaming by local index, identical
//! counts and parities, identical happens-before gates), so the verdict of
//! one is the verdict of the other. Today: `mbarrier_state_search_fingerprint`
//! and `has_equivalent_mbarrier_state_search` (`sync_fixed_unified.rs:883-1041`),
//! restricted there to mbarrier projections. The full encoding is used as the
//! key, so no separate structural-equality pass is needed.

use crate::event::SyncOp;
use crate::projection::ProjectionKey;
use crate::ts::ProtocolTs;

pub fn fingerprint(ts: &ProtocolTs<'_>) -> Option<Vec<u64>> {
    if !matches!(ts.key, ProjectionKey::Resource(_)) {
        return None;
    }
    let mut out = vec![ts.programs.len() as u64];
    for program in &ts.programs {
        out.push(program.len() as u64);
        for &cmd in program {
            let local = &ts.cmds[cmd];
            match local.kind {
                SyncOp::MbarInit { expected, .. } => out.extend([0, u64::from(expected)]),
                SyncOp::MbarArrive { count, expect_tx, .. } => {
                    out.extend([1, u64::from(count), u64::from(expect_tx)])
                }
                SyncOp::MbarExpectTx { tx, .. } => out.extend([2, u64::from(tx)]),
                SyncOp::MbarTxIssue { tx, .. } => out.extend([3, u64::from(tx)]),
                SyncOp::MbarWait { parity, .. } => out.extend([4, u64::from(parity)]),
                SyncOp::NamedArrive { expected, count, .. } => {
                    out.extend([5, u64::from(expected), u64::from(count)])
                }
                SyncOp::NamedSync { expected, count, .. } => {
                    out.extend([6, u64::from(expected), u64::from(count)])
                }
                SyncOp::ClusterArrive { participants, .. } => out.extend([7, u64::from(participants)]),
                SyncOp::ClusterWait { .. } => out.push(8),
            }
            out.push(local.gate.len() as u64);
            for &(warp, count) in local.gate.iter() {
                out.extend([u64::from(warp), u64::from(count)]);
            }
        }
    }
    Some(out)
}
