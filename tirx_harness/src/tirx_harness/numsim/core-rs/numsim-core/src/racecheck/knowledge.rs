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
//! * `tmap_rel` / `g2t` — the tensormap proxy (descriptor bytes written
//!   generically, read by TMA through the tensormap proxy). Each
//!   `fence.proxy.tensormap::generic.release.<scope>` adds a head
//!   `(releasing warp, scope, hb snapshot)`; heads propagate like a bridge.
//!   The consuming thread's `.acquire.<scope> [addr], size` keeps the heads
//!   whose *releasing fence* and the acquire mutually include each other's
//!   thread (PTX §8.9.4, §9.7.15.4), per acquired byte range. A TMA issued
//!   afterwards inherits those ranges; `g2t` is the view computed for one
//!   tensormap-proxy access.
//!
//! Shared memory, TMEM and global memory use this one structure; their
//! differences are which slots are ever consulted (TMEM: `tcgen`; shared and
//! global: `hb` + bridges for their domain).

use super::clock::{Clock, JoinMemo, Stamp};
use super::input::{Domain, Proxy, Scope, SiteId};

pub const NDOM: usize = 3;

#[derive(Clone, Debug, Default)]
pub struct Knowledge {
    pub hb: Clock,
    pub g2a: [Clock; NDOM],
    pub a2g: [Clock; NDOM],
    pub tcgen: Clock,
    pub tcgen_rel: Clock,
    pub tmap_rel: TmapHeads,
    pub g2t: Clock,
}

/// Tensormap release heads keyed by `(releasing warp, scope)`, sorted.
pub type TmapHeads = Option<std::sync::Arc<Vec<(u32, Scope, Clock)>>>;

/// `a ⊔= b` on head lists (join clocks with equal keys).
pub fn join_tmap(a: &mut TmapHeads, b: &TmapHeads, memo: &JoinMemo) {
    let Some(bv) = b else { return };
    match a {
        None => *a = Some(bv.clone()),
        Some(av) if std::sync::Arc::ptr_eq(av, bv) => {}
        Some(av) => {
            let v = std::sync::Arc::make_mut(av);
            for (w, s, c) in bv.iter() {
                match v.binary_search_by_key(&(*w, *s), |(x, y, _)| (*x, *y)) {
                    Ok(i) => {
                        v[i].2.join(c, memo);
                    }
                    Err(i) => v.insert(i, (*w, *s, c.clone())),
                }
            }
        }
    }
}

/// The release heads a write carries: its own head plus the heads it
/// inherits through an observation-order chain of morally strong atomics
/// (PTX §8.9.2). Each head is scope-checked against the acquirer on its own.
pub type Heads = std::sync::Arc<HeadList>;

/// A head list plus the join of its `.gpu`-or-wider heads.
///
/// An RMW chain on a counter (a grid-wide arrival count) inherits every
/// earlier RMW's head, so a list grows by one head per RMW, and an acquirer
/// joined each head on its own: on mega_moe e24, 2.8K waits joined ~88 heads
/// each, 59% of the checker's time. A head released at `.gpu` or wider
/// passes the scope check of every acquirer whose own scope is `.gpu` or
/// wider (`required_scope` never exceeds `.gpu` within a launch), so for
/// such an acquirer those heads act as one payload: `gpu` is their join,
/// built incrementally as the chain grows (one join per RMW). The join is a
/// lattice join, so acquiring `gpu` gives the same clock values as acquiring
/// each of those heads in turn.
#[derive(Debug)]
pub struct HeadList {
    rels: Vec<std::sync::Arc<Rel>>,
    /// Join of the acquire-relevant parts (`hb`, bridges, `tcgen_rel`,
    /// `tmap_rel`) of every head with `scope >= Gpu`; `None` if there is none.
    gpu: Option<std::sync::Arc<Knowledge>>,
}

impl std::ops::Deref for HeadList {
    type Target = [std::sync::Arc<Rel>];
    fn deref(&self) -> &Self::Target {
        &self.rels
    }
}

impl HeadList {
    /// Does `rel` belong to the joined `.gpu`-or-wider class?
    #[inline]
    pub fn wide(rel: &Rel) -> bool {
        rel.scope.is_some_and(|s| s >= Scope::Gpu)
    }

    /// `base`'s heads followed by `own`.
    pub fn extend(base: Option<&Heads>, own: &std::sync::Arc<Rel>, memo: &JoinMemo) -> Heads {
        let mut rels: Vec<std::sync::Arc<Rel>> = base.map(|b| b.rels.clone()).unwrap_or_default();
        rels.push(own.clone());
        let prior = base.and_then(|b| b.gpu.clone());
        let gpu = if Self::wide(own) {
            Some(std::sync::Arc::new(match prior {
                None => acquire_part(&own.k),
                Some(p) => {
                    let mut k = (*p).clone();
                    join_acquire_part(&mut k, &own.k, memo);
                    k
                }
            }))
        } else {
            prior
        };
        std::sync::Arc::new(HeadList { rels, gpu })
    }

    /// The joined `.gpu`-or-wider heads (see the type doc).
    pub fn gpu(&self) -> Option<&Knowledge> {
        self.gpu.as_deref()
    }
}

/// The parts of a payload an acquire reads (`Warp::acquire`).
fn acquire_part(k: &Knowledge) -> Knowledge {
    Knowledge { hb: k.hb.clone(), g2a: k.g2a.clone(), a2g: k.a2g.clone(), tcgen_rel: k.tcgen_rel.clone(), tmap_rel: k.tmap_rel.clone(), ..Default::default() }
}

fn join_acquire_part(x: &mut Knowledge, k: &Knowledge, memo: &JoinMemo) {
    x.hb.join(&k.hb, memo);
    for d in 0..NDOM {
        x.g2a[d].join(&k.g2a[d], memo);
        x.a2g[d].join(&k.a2g[d], memo);
    }
    x.tcgen_rel.join(&k.tcgen_rel, memo);
    join_tmap(&mut x.tmap_rel, &k.tmap_rel, memo);
}

/// A release payload stored with a write (or a fence-release head).
#[derive(Clone, Debug)]
pub struct Rel {
    pub k: Knowledge,
    /// `None` for payloads that carry only `tcgen_rel` (relaxed stores).
    pub scope: Option<Scope>,
    pub warp: u32,
    /// Site of the releasing operation (evidence for `ScopeMismatch`).
    pub site: SiteId,
    /// A relaxed wait's payload: the record token it carries (acquired at
    /// the fence, which then raises the delivery token; checker/tokens.rs).
    pub tok: Option<(super::clock::ActorId, super::clock::Epoch)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum View {
    Hb,
    G2a(usize),
    A2g(usize),
    Tcgen,
    G2t,
}

/// Which clock judges "prior (in `prior` proxy, domain `d`) before current
/// (in `cur` proxy)". `None` domain is TMEM.
#[inline(always)]
pub fn select_view(prior: Proxy, cur: Proxy, d: Option<Domain>) -> View {
    match (prior, cur) {
        (Proxy::Tcgen, Proxy::Tcgen) => View::Tcgen,
        (Proxy::Generic, Proxy::TensorMap) => View::G2t,
        // A descriptor read (TMA issue) before a later generic write of the
        // descriptor: the ISA defines only the generic->tensormap direction
        // (fence.proxy.tensormap::generic release/acquire); the read is
        // ordered by hb (program order, or the TMA's observed completion)
        // like legacy (deltas I9).
        (Proxy::TensorMap, Proxy::Generic) => View::Hb,
        (Proxy::Generic, Proxy::Async) => d.map_or(View::Hb, |d| View::G2a(d as usize)),
        (Proxy::Async, Proxy::Generic) => d.map_or(View::Hb, |d| View::A2g(d as usize)),
        _ => View::Hb,
    }
}

/// Bridge slots a `fence.proxy.async[.d]` sets.
/// The bridge slot is the *prior* access's window domain. A
/// `fence.proxy.async.shared::cta` does not bridge a generic write made
/// through a `shared::cluster` (mapa) window, even to the same bytes
/// (test_native_proxy_async_fence.py same_rank_mapa).
pub fn fence_domains(d: Option<Domain>) -> &'static [usize] {
    match d {
        None => &[0, 1, 2],
        Some(Domain::Global) => &[0],
        Some(Domain::SharedCta) => &[1],
        // The shared::cta window lies inside the shared::cluster window
        // (PTX §5.1.7), so a .shared::cluster fence also covers shared::cta
        // objects. The converse is ISA-silent: fail closed.
        Some(Domain::SharedCluster) => &[1, 2],
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
            View::G2t => &self.g2t,
        }
    }

    pub fn view_mut(&mut self, v: View) -> &mut Clock {
        match v {
            View::Hb => &mut self.hb,
            View::G2a(d) => &mut self.g2a[d],
            View::A2g(d) => &mut self.a2g[d],
            View::Tcgen => &mut self.tcgen,
            View::G2t => &mut self.g2t,
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
        join_tmap(&mut self.tmap_rel, &o.tmap_rel, memo);
    }

    pub fn join_all(&mut self, o: &Knowledge, memo: &JoinMemo) {
        self.join_propagating(o, memo);
        self.tcgen.join(&o.tcgen, memo);
        self.g2t.join(&o.g2t, memo);
    }

    /// The parts an acquire propagates, without `tcgen`.
    pub fn propagating(&self) -> Knowledge {
        Knowledge {
            hb: self.hb.clone(),
            g2a: self.g2a.clone(),
            a2g: self.a2g.clone(),
            tcgen: Clock::default(),
            tcgen_rel: self.tcgen_rel.clone(),
            tmap_rel: self.tmap_rel.clone(),
            g2t: Clock::default(),
        }
    }
}
