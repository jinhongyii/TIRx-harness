//! Structural fingerprint of a projection: equal fingerprints mean isomorphic
//! state spaces (same per-warp programs up to warp and resource renaming by
//! projection-local index, same commands, counts, gates and resource kinds),
//! so one verdict serves both. Generalizes today's mbarrier-only
//! `mbarrier_state_search_fingerprint` (`sync_fixed_unified.rs:883-1041`).
//! The full encoding is the key, so no separate equivalence pass is needed.

use std::fmt::Write;

use super::ts::Ts;
use crate::sync::ResourceId;

fn kind(id: ResourceId) -> String {
    match id {
        ResourceId::Mbarrier { .. } => "M".into(),
        ResourceId::Named { .. } => "N".into(),
        ResourceId::Cluster { .. } => "C".into(),
        ResourceId::AsyncGroup { domain, .. } => format!("G{domain:?}"),
        ResourceId::TcgenLifecycle { .. } => "T".into(),
        ResourceId::TcgenWork { .. } => "W".into(),
        ResourceId::RegPool { .. } => "R".into(),
        other => format!("{other:?}"),
    }
}

pub fn fingerprint(ts: &Ts<'_>) -> String {
    let mut out = String::new();
    for &r in &ts.resources {
        let _ = write!(out, "{};", kind(ts.program.resources[r]));
    }
    out.push('|');
    for list in &ts.programs {
        let _ = write!(out, "[{}]", list.len());
        for &c in list {
            let lc = &ts.cmds[c];
            let _ = write!(
                out,
                "{{p{:?}c{:?}i{:?}q{}g{:?}}}",
                lc.participants,
                lc.cmds,
                lc.issued,
                u8::from(lc.conditional),
                lc.gate
            );
        }
    }
    out
}
