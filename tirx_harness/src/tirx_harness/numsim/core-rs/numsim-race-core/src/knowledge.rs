//! `Knowledge`: what one actor (a lane, or an async op) has observed, with the
//! proxy as a dimension.
//!
//! * `hb` — ordinary happens-before within one proxy (generic↔generic,
//!   async↔async).
//! * `g2a[d]` / `a2g[d]` — the proxy bridge: generic events of domain `d`
//!   visible to the async proxy, and vice versa. Set by `fence.proxy.async`
//!   (snapshot of `hb`, both directions) and, for `a2g`, by implicit async
//!   completion (mbarrier complete_tx, bulk/cp.async wait_group). Invariant:
//!   every bridge ⊑ `hb`, so transitivity through bridges is sound and a
//!   single-witness frontier can drop a prior that a later access observed.
//! * `tcgen` — TMEM order, the view a tcgen05 op is issued with. Fed only by
//!   `tcgen05.fence::after_thread_sync` (snapshot of `hb`), by the issuing
//!   warp's own `tcgen05.wait::{ld,st}` and by architected pipeline order. It
//!   is **not** propagated by acquire: a thread sync orders tcgen05 work only
//!   with the fence pair on both sides.
//! * `tcgen_rel` — the tcgen05 fence frontier in transit: what
//!   `fence::before_thread_sync` published (issued pipelined ops + waited
//!   ld/st) and what a `tcgen05.commit` completion forwards. It travels
//!   through *every* thread sync, including relaxed arrives / relaxed waits
//!   (PTX 9.7.18.6.4.4), and is moved into `tcgen` by `after_thread_sync`.
//!
//! Shared memory, TMEM and global memory use this one structure; their
//! differences are which slots are ever consulted (TMEM: `tcgen`; shared and
//! global: `hb` + bridges for their domain).

use crate::clock::{Clock, JoinMemo, Stamp};
use crate::input::{Domain, Proxy, Scope};

pub const NDOM: usize = 3;

#[derive(Clone, Debug, Default)]
pub struct Knowledge {
    pub hb: Clock,
    pub g2a: [Clock; NDOM],
    pub a2g: [Clock; NDOM],
    pub tcgen: Clock,
    pub tcgen_rel: Clock,
}

/// A release payload stored with a write (or a fence-release head).
#[derive(Clone, Debug)]
pub struct Rel {
    pub k: Knowledge,
    /// `None` for payloads that carry only `tcgen_rel` (relaxed stores).
    pub scope: Option<Scope>,
    pub warp: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum View {
    Hb,
    G2a(usize),
    A2g(usize),
    Tcgen,
}

/// Which clock judges "prior (in `prior` proxy, domain `d`) before current
/// (in `cur` proxy)". `None` domain is TMEM.
#[inline(always)]
pub fn select_view(prior: Proxy, cur: Proxy, d: Option<Domain>) -> View {
    match (prior, cur) {
        (Proxy::Tcgen, Proxy::Tcgen) => View::Tcgen,
        (Proxy::Generic, Proxy::Async) => d.map_or(View::Hb, |d| View::G2a(d as usize)),
        (Proxy::Async, Proxy::Generic) => d.map_or(View::Hb, |d| View::A2g(d as usize)),
        _ => View::Hb,
    }
}

/// Bridge slots a `fence.proxy.async[.d]` sets.
/// The bridge slot is the *prior* access's window domain. A
/// `fence.proxy.async.shared::cta` does not bridge a generic write made
/// through a `shared::cluster` (mapa) window, even to the same bytes
/// (test_native_proxy_async_fence.py same_rank_mapa / shared_cta modes).
pub fn fence_domains(d: Option<Domain>) -> &'static [usize] {
    match d {
        None => &[0, 1, 2],
        Some(Domain::Global) => &[0],
        Some(Domain::SharedCta) => &[1],
        Some(Domain::SharedCluster) => &[2],
    }
}

impl Knowledge {
    #[inline(always)]
    pub fn view(&self, v: View) -> &Clock {
        match v {
            View::Hb => &self.hb,
            View::G2a(d) => &self.g2a[d],
            View::A2g(d) => &self.a2g[d],
            View::Tcgen => &self.tcgen,
        }
    }

    pub fn view_mut(&mut self, v: View) -> &mut Clock {
        match v {
            View::Hb => &mut self.hb,
            View::G2a(d) => &mut self.g2a[d],
            View::A2g(d) => &mut self.a2g[d],
            View::Tcgen => &mut self.tcgen,
        }
    }

    #[inline(always)]
    pub fn observes(&self, v: View, stamp: Stamp, lane: u8) -> bool {
        self.view(v).observes(stamp, lane)
    }

    /// Join the parts an acquire propagates (everything except `tcgen`).
    pub fn join_propagating(&mut self, o: &Knowledge, memo: &JoinMemo) {
        self.hb.join(&o.hb, memo);
        for d in 0..NDOM {
            self.g2a[d].join(&o.g2a[d], memo);
            self.a2g[d].join(&o.a2g[d], memo);
        }
        self.tcgen_rel.join(&o.tcgen_rel, memo);
    }

    pub fn join_all(&mut self, o: &Knowledge, memo: &JoinMemo) {
        self.join_propagating(o, memo);
        self.tcgen.join(&o.tcgen, memo);
    }

    /// The parts an acquire propagates, without `tcgen`.
    pub fn propagating(&self) -> Knowledge {
        Knowledge {
            hb: self.hb.clone(),
            g2a: self.g2a.clone(),
            a2g: self.a2g.clone(),
            tcgen: Clock::default(),
            tcgen_rel: self.tcgen_rel.clone(),
        }
    }
}
