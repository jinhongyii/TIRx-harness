//! Token re-attribution of completed copies (racecheck-parallel-design.md
//! §17; soundness review W6).
//!
//! A copy's component enters clocks raw at exactly two places: its
//! completion folded into a phase record (`Phase::completion`) and a
//! warp-target delivery. Each such raise is accompanied, in the same clock
//! and the same views (hb and a2g[d] for every d), by a token:
//! - the record token `R_P` (per retained record, pooled per cluster),
//!   raised to the next value of its counter at each copy fold;
//! - a delivery token (`D_W` for full-mask acquisitions, raised in `base`;
//!   `D_{W,l}` for partial ones, raised in `extra[l]`), raised wherever a
//!   warp acquires raw token-bearing knowledge: warp-target delivery,
//!   an acquire wait on a tokened record, and the fence that consumes a
//!   relaxed wait's pending payload.
//!
//! Every other move of knowledge copies whole clocks, so "a clock knows the
//! copy's component" is equivalent to "it knows one of the copy's
//! attributions". Once a copy is complete (`done == 2`, every target
//! delivered), its witnesses are rewritten to a virtual actor whose record
//! holds the decode snapshot and the attributions, and its slot is
//! reclaimed. A record token retires when its record is pruned and no
//! pending payload holds it: its attributions are replaced by the delivery
//! tokens logged at its acquisitions, and the id returns to its pool above
//! its last value.
use super::*;
use std::sync::Weak;

// Actor ids: warps [0, nw); delivery tokens nw + 33·w + {0: full mask,
// 1 + lane}; then async slots and record tokens, both from `next_slot`
// (`slot_base() + index`), so every clock component stays dense.
/// First virtual actor id (re-attributed witnesses; never a clock
/// component).
pub(super) const VBASE: ActorId = 1 << 31;
/// Fresh record-token ids a checker partition reserves per cluster.
pub(super) const TOKEN_RESERVE: usize = 8;



/// A re-attributed copy: decode snapshot plus attributions.
pub(super) struct VRec {
    pub snap: AsyncActor,
    pub refs: Vec<(ActorId, Epoch)>,
}

#[derive(Default)]
pub(super) struct Tokens {
    /// Free record-token ids per cluster, with the last value each used.
    pub pools: HashMap<u32, Vec<(ActorId, Epoch)>>,
    /// Acquisitions of each record token: (record value held, delivery
    /// token, its value).
    pub logs: HashMap<ActorId, Vec<(Epoch, ActorId, Epoch)>>,
    /// Pending relaxed-wait payloads carrying each record token.
    pub pend: HashMap<ActorId, Vec<Weak<Rel>>>,
    /// Record tokens whose record was pruned: (id, last value, cluster).
    pub retired: Vec<(ActorId, Epoch, u32)>,
    /// Re-attributed copies, by `actor - VBASE`.
    pub vrecs: Arc<Vec<Option<Arc<VRec>>>>,
    pub vfree: Vec<usize>,
    /// Records naming each token (main checker).
    pub users: HashMap<ActorId, HashSet<usize>>,
    /// In-flight ops (slot index) whose attributions name each token.
    pub attr_users: HashMap<ActorId, HashSet<usize>>,
}

/// Knowledge raising token `t` to `v` in the views a copy completion uses.
fn tok_knowledge(t: ActorId, v: Epoch) -> Knowledge {
    let mut k = Knowledge::default();
    k.hb.raise(t, v);
    for d in 0..NDOM {
        k.a2g[d].raise(t, v);
    }
    k
}

impl Checker {
    /// Actor id of async slot / record-token index 0.
    #[inline(always)]
    pub(super) fn slot_base(&self) -> ActorId {
        34 * self.topo.num_warps()
    }

    fn dtoken(&self, w: WarpId, lane: Option<u8>) -> ActorId {
        self.topo.num_warps() + 33 * w + lane.map_or(0, |l| 1 + l as u32)
    }

    pub(super) fn tokens_on(&self) -> bool {
        super::super::tuning::on(&super::super::tuning::REATTRIBUTE_TOKENS)
    }

    pub(super) fn vrec(&self, a: ActorId) -> Option<&VRec> {
        self.tk.vrecs.get((a - VBASE) as usize).and_then(|r| r.as_deref())
    }

    /// Is this op's completion tokenised (a copy issued in the async proxy)?
    pub(super) fn tokenised(&self, i: usize) -> bool {
        let a = &self.asyncs[i];
        self.tokens_on() && a.kind == AsyncKind::Copy && a.proxy == Proxy::Async
    }

    fn token_cluster(&self, obj: SyncObjId) -> u32 {
        self.obj_cluster(obj).unwrap_or(u32::MAX)
    }

    /// Can this checker give record `obj` a token now? (A partition only
    /// from its reserved pool.)
    pub(super) fn token_available(&self, obj: SyncObjId) -> bool {
        self.part.is_none() || self.tk.pools.get(&self.token_cluster(obj)).is_some_and(|p| !p.is_empty())
    }

    /// Fold hook: raise record (obj, phase)'s token in `c` and attribute
    /// op `i` to it.
    pub(super) fn token_fold(&mut self, i: usize, obj: SyncObjId, phase: u64, c: &mut Knowledge) {
        let cluster = self.token_cluster(obj);
        let cur = self.phases.get(&obj).and_then(|m| m.get(&phase)).and_then(|p| p.tok);
        let (id, last) = match cur {
            Some(t) => t,
            None => {
                let from_pool = self.tk.pools.get_mut(&cluster).and_then(|p| p.pop());
                match from_pool {
                    Some(t) => t,
                    None => {
                        assert!(self.part.is_none(), "partition record token pool exhausted (needs_main)");
                        let i = self.next_slot;
                        self.next_slot += 1;
                        (self.slot_base() + i as u32, 0)
                    }
                }
            }
        };
        let v = last + 1;
        if let Some(p) = self.phases.get_mut(&obj).and_then(|m| m.get_mut(&phase)) {
            p.tok = Some((id, v));
        }
        c.join_propagating(&tok_knowledge(id, v), &self.memo);
        self.asyncs[i].attr.push((id, v));
        self.tk.attr_users.entry(id).or_default().insert(i);
    }

    /// Raise warp `w`'s delivery token(s) for `lanes` (full mask: `D_W` in
    /// base; otherwise `D_{W,l}` per lane). Returns (token, value) pairs.
    pub(super) fn deliver(&mut self, w: WarpId, lanes: LaneMask) -> Vec<(ActorId, Epoch)> {
        let mut out = Vec::new();
        let full = self.dtoken(w, None);
        let per: Vec<ActorId> = (0..32u8).map(|l| self.dtoken(w, Some(l))).collect();
        let memo = &self.memo;
        let warp = &mut self.warps[w as usize];
        if lanes.is_all() {
            warp.dtok[32] += 1;
            let (t, v) = (full, warp.dtok[32]);
            warp.acquire(lanes, &tok_knowledge(t, v), memo);
            debug_assert!(warp.base.hb.epochs.get(t) >= v, "a full-mask delivery token lands in base");
            out.push((t, v));
        } else {
            for l in lanes.lanes8() {
                warp.dtok[l as usize] += 1;
                let (t, v) = (per[l as usize], warp.dtok[l as usize]);
                warp.acquire(one_lane(l), &tok_knowledge(t, v), memo);
                out.push((t, v));
            }
        }
        out
    }

    /// Log that record token (id, rval) was acquired with `ds`.
    pub(super) fn token_log(&mut self, id: ActorId, rval: Epoch, ds: &[(ActorId, Epoch)]) {
        let log = self.tk.logs.entry(id).or_default();
        for &(t, v) in ds {
            log.push((rval, t, v));
        }
    }

    /// A record left the retained list.
    pub(super) fn token_retire(&mut self, obj: SyncObjId, tok: (ActorId, Epoch)) {
        let cluster = self.token_cluster(obj);
        self.tk.retired.push((tok.0, tok.1, cluster));
    }

    /// `ordered` for a virtual-actor witness: cur knows one attribution.
    pub(super) fn vordered(&self, cur: Cur, prior: &Witness, view: View) -> bool {
        let Some(rec) = self.vrec(prior.stamp.actor()) else { return false };
        debug_assert!(matches!(view, View::Hb | View::A2g(_)), "a copy's async witness is judged in Hb or A2g");
        rec.refs.iter().any(|&(t, v)| {
            let s = Stamp::new(t, v);
            match cur {
                Cur::Lane { w, lane, .. } => self.cur_warp(w).knows(lane, view, s, 0),
                Cur::Async { a } => self.asyncs[a].k.observes(view, s, 0),
            }
        })
    }

    /// GC, before the walk: copies to re-attribute (slot actor -> virtual).
    pub(super) fn reattribute_plan(&mut self) -> HashMap<ActorId, ActorId, crate::sync::FxBuild> {
        let mut remap: HashMap<ActorId, ActorId, crate::sync::FxBuild> = HashMap::default();
        if !self.tokens_on() {
            return remap;
        }
        // Copies some commit tracks keep their slot (§17.1).
        let mut in_commit: HashSet<usize> = HashSet::new();
        for a in self.asyncs.iter().filter(|a| a.in_use && a.kind == AsyncKind::TcgenCommit) {
            in_commit.extend(a.preds.iter().map(|(p, _)| *p));
        }
        let held: Vec<usize> = self.asyncs.iter_indexed().map(|(i, _)| i).collect();
        for i in held {
            let a = &self.asyncs[i];
            if !(a.in_use && a.done >= 2 && a.kind == AsyncKind::Copy && a.proxy == Proxy::Async) || in_commit.contains(&i) {
                continue;
            }
            let mut snap = a.clone();
            snap.k = Knowledge::default();
            snap.preds.clear();
            snap.footprint.clear();
            let refs = std::mem::take(&mut self.asyncs[i].attr);
            for (t, _) in &refs {
                if let Some(u) = self.tk.attr_users.get_mut(t) {
                    u.remove(&i);
                }
            }
            let actor = snap.actor;
            let gen = snap.gen_base;
            let rec = Arc::new(VRec { snap, refs: refs.clone() });
            let recs = Arc::make_mut(&mut self.tk.vrecs);
            let r = match self.tk.vfree.pop() {
                Some(r) => {
                    recs[r] = Some(rec);
                    r
                }
                None => {
                    recs.push(Some(rec));
                    recs.len() - 1
                }
            };
            for (t, _) in refs {
                self.tk.users.entry(t).or_default().insert(r);
            }
            for side in 1..=2 {
                self.op_reg.remove(&(actor, gen + side));
            }
            remap.insert(actor, VBASE + r as ActorId);
        }
        remap
    }

    /// GC, after the walk: free unused records, convert retired record
    /// tokens no pending payload holds.
    pub(super) fn tokens_after_gc(&mut self, live: &HashSet<ActorId>) {
        if !self.tokens_on() {
            return;
        }
        let n = self.tk.vrecs.len();
        let dead: Vec<usize> = (0..n).filter(|&r| self.tk.vrecs[r].is_some() && !live.contains(&(VBASE + r as ActorId))).collect();
        if !dead.is_empty() {
            let recs = Arc::make_mut(&mut self.tk.vrecs);
            for r in dead {
                if let Some(rec) = recs[r].take() {
                    for (t, _) in &rec.refs {
                        if let Some(u) = self.tk.users.get_mut(t) {
                            u.remove(&r);
                            if u.is_empty() {
                                self.tk.users.remove(t);
                            }
                        }
                    }
                }
                self.tk.vfree.push(r);
            }
        }
        // Retired record tokens.
        let retired = std::mem::take(&mut self.tk.retired);
        for (id, last, cluster) in retired {
            let held = self.tk.pend.get_mut(&id).is_some_and(|v| {
                v.retain(|w| w.strong_count() > 0);
                !v.is_empty()
            });
            if held {
                self.tk.retired.push((id, last, cluster));
                continue;
            }
            self.tk.pend.remove(&id);
            let log = self.tk.logs.remove(&id).unwrap_or_default();
            let conv = |refs: &mut Vec<(ActorId, Epoch)>| -> Vec<ActorId> {
                let mut added = Vec::new();
                let mut out: Vec<(ActorId, Epoch)> = Vec::new();
                for &(t, k) in refs.iter() {
                    if t != id {
                        out.push((t, k));
                        continue;
                    }
                    for &(rv, d, dv) in &log {
                        if rv >= k && !out.contains(&(d, dv)) {
                            out.push((d, dv));
                            added.push(d);
                        }
                    }
                }
                *refs = out;
                added
            };
            if let Some(users) = self.tk.users.remove(&id) {
                let recs = Arc::make_mut(&mut self.tk.vrecs);
                for r in users {
                    let Some(rec) = recs[r].as_mut() else { continue };
                    let rec = Arc::make_mut(rec);
                    for d in conv(&mut rec.refs) {
                        self.tk.users.entry(d).or_default().insert(r);
                    }
                }
            }
            for i in self.tk.attr_users.remove(&id).unwrap_or_default() {
                let Some(a) = self.asyncs.0.get_mut(i) else { continue };
                if a.attr.iter().any(|(t, _)| *t == id) {
                    for d in conv(&mut a.attr) {
                        self.tk.attr_users.entry(d).or_default().insert(i);
                    }
                }
            }
            self.tk.pools.entry(cluster).or_default().push((id, last));
        }
    }
}

impl Clone for VRec {
    fn clone(&self) -> Self {
        VRec { snap: self.snap.clone(), refs: self.refs.clone() }
    }
}
