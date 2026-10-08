//! Checker partitions (racecheck-parallel-design.md, milestone 1; contract
//! decision 17).
//!
//! A scheduling partition's replayed events of one phase are processed by a
//! *child* checker that owns the partition's cluster-local state: its warps,
//! its async slot pool, its shared/TMEM allocations (shadow, declared words,
//! alias tracker) and its barrier phases. The child processes events in seq
//! order up to the first event that needs state outside the cluster
//! (`needs_main`), deferring weak accesses to global memory with a snapshot
//! of the accessing actor ("HB handle"). Everything from the first such event
//! on is stashed. `Checker::absorb` (the round merge, called in partition
//! order) moves the state back, merges the child's findings / incompletes /
//! stats and applies the deferred global accesses in seq order, then
//! processes the stash in the main checker. Every event that touches global
//! state is therefore handled by the main checker in replay order, which is
//! the serial order (review H1/H2/H4/W3/W4 by construction).

use super::*;

/// What a child knows of allocations it does not hold.
#[derive(Clone, Copy)]
pub(crate) struct AllocMeta {
    pub space: Space,
    pub size: u64,
}

/// One weak global access deferred to the main checker.
pub(crate) struct Deferred {
    pub a: Access,
    pub cur: Cur,
    pub stamp: Stamp,
    pub lane: u8,
    pub lane_g2t: Option<Clock>,
    pub snap: Snap,
    pub tag: Tag,
}

/// The accessing actor's state as of the access.
pub(crate) enum Snap {
    Lane(Box<Warp>),
    Async(Box<AsyncActor>),
}

/// Position of a child-side item in the partition's replay order: the seq
/// of the access, or for a sync event the seq the next access will get
/// (so it sorts before that access); `rank` 0 = sync, 1 = access.
pub(crate) type Tag = (u64, u8);

pub(crate) struct PartCtx {
    pub meta: Arc<HashMap<AllocId, AllocMeta>>,
    pub clusters: HashSet<u32>,
    pub pool: u32,
    pub deferred: Vec<Deferred>,
    pub suspended: bool,
    /// Tag of the event being processed.
    pub tag: Tag,
    /// Tags of the child's report entries, parallel to `report.findings`
    /// and `report.incomplete`.
    pub finding_tags: Vec<Tag>,
    pub incomplete_tags: Vec<Tag>,
    /// Dedup key of each child finding (`None`: never deduplicated).
    pub finding_keys: Vec<Option<FKey>>,
}

/// Dedup key of a finding, as the report functions key them.
#[derive(Clone)]
pub(crate) enum FKey {
    Race((AllocId, RaceClass, SiteId, SiteId, bool)),
    Advisory((AdvisoryKind, AllocId, SiteId)),
    Scope((SiteId, SiteId, Scope, Scope)),
    Alias(AliasKey),
}

impl Warp {
    /// The warp's ordering state without its site table (a deferred
    /// access's HB handle; the site table stays with the live warp).
    pub(crate) fn snapshot(&self) -> Box<Warp> {
        Box::new(Warp {
            actor: self.actor,
            epoch: self.epoch,
            done: self.done,
            row: self.row.clone(),
            base: self.base.clone(),
            extra: self.extra.clone(),
            bridge_rows: self.bridge_rows.clone(),
            fence_rel: self.fence_rel.clone(),
            pending_acq: self.pending_acq.clone(),
            tcgen: self.tcgen.clone(),
            tcgen_in: self.tcgen_in.clone(),
            tcgen_issued: self.tcgen_issued.clone(),
            tcgen_waited: self.tcgen_waited.clone(),
            tcgen_pub: self.tcgen_pub.clone(),
            g2t_ranges: self.g2t_ranges.clone(),
            sites: Vec::new(),
        })
    }
}

impl Checker {
    /// Cluster owning a barrier object, if any.
    fn obj_cluster(&self, obj: SyncObjId) -> Option<u32> {
        let cpc = self.topo.ctas_per_cluster.max(1);
        match obj {
            SyncObjId::Mbarrier { cta, .. } | SyncObjId::Named { cta, .. } | SyncObjId::RegPool { cta } => Some(cta.0 / cpc),
            SyncObjId::Cluster { cluster } | SyncObjId::TcgenLifecycle { cluster, .. } => Some(cluster),
            SyncObjId::AsyncGroup { warp, .. } | SyncObjId::TcgenWork { warp, .. } => Some(self.topo.cluster_of(warp.0)),
            _ => None,
        }
    }

    fn warp_held(&self, w: WarpId) -> bool {
        self.warps.get(w as usize).is_some()
    }

    /// Does `e` need state this child does not hold (or a decision only the
    /// main checker can take)? Weak global accesses are not such events:
    /// they are deferred.
    fn needs_main(&self, e: &Event) -> bool {
        let p = self.part.as_ref().expect("child");
        match e {
            Event::Access(a) => {
                let held_actor = match a.who {
                    Who::Lane { warp, .. } => self.warp_held(warp),
                    Who::Async { op, .. } => self.op_slot(op).is_some(),
                };
                if !held_actor {
                    return true;
                }
                if self.allocs.contains_key(&a.alloc) {
                    return false;
                }
                match p.meta.get(&a.alloc) {
                    // Strong or atomic global accesses read-from / release:
                    // their effects feed back into this partition's state.
                    // Out-of-bounds goes to the main checker as well.
                    Some(m) => {
                        !(m.space == Space::Global && a.order == MemOrder::Weak && !a.atomic && a.scope.is_none() && a.range.end <= m.size && a.range.start <= a.range.end)
                    }
                    None => true,
                }
            }
            Event::Sync(s) => match s {
                SyncEvent::AllocBegin { .. } | SyncEvent::AllocEnd { .. } | SyncEvent::DeclareWord { .. } | SyncEvent::WaitVerdicts { .. } => true,
                SyncEvent::WarpSync { warp, .. } => !self.warp_held(*warp),
                SyncEvent::Arrive { warp, obj, .. } | SyncEvent::Wait { warp, obj, .. } => {
                    !self.warp_held(*warp) || !self.obj_cluster(*obj).is_some_and(|c| p.clusters.contains(&c))
                }
                SyncEvent::Fence { warp, kind, .. } => {
                    !self.warp_held(*warp)
                        || matches!(kind, FenceKind::Sc(_) | FenceKind::TensormapRelease { .. } | FenceKind::TensormapAcquire { .. })
                }
                SyncEvent::AsyncIssue { op, warp, .. } => {
                    // A fresh slot comes only from the pool's reservation.
                    !self.warp_held(*warp) || pool_key(*op) != p.pool || self.pools.get(&p.pool).is_none_or(|pl| pl.free.is_empty())
                }
                SyncEvent::AsyncComplete { op, target, .. } => {
                    self.op_slot(*op).is_none()
                        || match target {
                            CompletionTarget::Phase { obj, .. } => !self.obj_cluster(*obj).is_some_and(|c| p.clusters.contains(&c)),
                            CompletionTarget::Warp { warp, .. } => !self.warp_held(*warp),
                        }
                }
            },
        }
    }

    /// Child entry: process `e` here, or report that it (and everything
    /// after it) belongs to the main checker.
    pub(crate) fn part_event(&mut self, e: &Event) -> bool {
        let p = self.part.as_mut().expect("child");
        if p.suspended {
            return false;
        }
        if let Event::Access(a) = e {
            p.tag = (a.seq, 1);
        } else {
            p.tag = (p.tag.0 + p.tag.1 as u64, 0);
        }
        if self.needs_main(e) {
            self.part.as_mut().unwrap().suspended = true;
            return false;
        }
        match e {
            Event::Access(a) if !self.allocs.contains_key(&a.alloc) => self.defer_access(a),
            Event::Access(a) => self.access(a),
            Event::Sync(s) => self.sync(s.clone()),
        }
        true
    }

    /// The actor-side prelude of a weak global access, then a deferred
    /// record with the actor's state (the HB handle).
    pub(crate) fn defer_access(&mut self, a: &Access) {
        if let Who::Lane { warp, .. } = a.who {
            self.flush_polls(warp);
        }
        self.stats.accesses += 1;
        let (cur, stamp, lane) = match a.who {
            Who::Lane { warp, lane, epoch } => {
                if !self.tick(warp, epoch) {
                    return;
                }
                let sites = &mut self.warps[warp as usize].sites;
                if sites.last().is_none_or(|(e, _)| *e != epoch) {
                    sites.push((epoch, a.site));
                }
                (Cur::Lane { w: warp as usize, lane, epoch }, Stamp::new(warp, epoch), lane)
            }
            Who::Async { op, side } => {
                let Some(i) = self.async_idx(op) else { return };
                let act = &self.asyncs[i];
                (Cur::Async { a: i }, Stamp::new(act.actor, act.gen_base + side_index(side)), 0)
            }
        };
        if a.range.start == a.range.end {
            return;
        }
        self.lane_g2t = None;
        if a.proxy == Proxy::TensorMap {
            let ranges = match cur {
                Cur::Async { a: i } => self.asyncs[i].g2t_ranges.clone(),
                Cur::Lane { w, lane, .. } => self.warps[w].g2t_ranges[lane as usize].clone(),
            };
            let mut cuts: Vec<u64> = ranges
                .iter()
                .filter(|(al, _, _)| *al == a.alloc)
                .flat_map(|(_, r, _)| [r.start, r.end])
                .filter(|x| a.range.start < *x && *x < a.range.end)
                .collect();
            if !cuts.is_empty() {
                cuts.sort_unstable();
                cuts.dedup();
                let mut lo = a.range.start;
                for hi in cuts.into_iter().chain([a.range.end]) {
                    let mut piece = a.clone();
                    piece.range = lo..hi;
                    self.stats.accesses -= 1;
                    self.defer_access_piece(&piece);
                    lo = hi;
                }
                return;
            }
            let mut v = Clock::default();
            for (al, r, k) in ranges.iter() {
                if *al == a.alloc && r.start < a.range.end && a.range.start < r.end {
                    v.join(k, &self.memo);
                }
            }
            match cur {
                Cur::Async { a: i } => self.asyncs[i].k.g2t = v,
                Cur::Lane { .. } => self.lane_g2t = Some(v),
            }
        }
        let snap = match cur {
            Cur::Lane { w, .. } => Snap::Lane(self.warps[w].snapshot()),
            Cur::Async { a: i } => Snap::Async(Box::new(self.asyncs[i].clone())),
        };
        let tag = self.part.as_ref().unwrap().tag;
        let lane_g2t = self.lane_g2t.take();
        self.part.as_mut().unwrap().deferred.push(Deferred { a: a.clone(), cur, stamp, lane, lane_g2t, snap, tag });
    }

    /// A tensormap-split piece: the same prelude path (the split recursion
    /// of `access`, kept on the deferral side).
    fn defer_access_piece(&mut self, a: &Access) {
        self.defer_access(a);
    }

    /// Main checker: run one deferred global access with the actor's state
    /// as of the access swapped in.
    fn apply_deferred(&mut self, d: Deferred) {
        let Deferred { a, cur, stamp, lane, lane_g2t, snap, .. } = d;
        let domain = a.domain;
        match (cur, snap) {
            (Cur::Lane { w, epoch, .. }, Snap::Lane(mut s)) => {
                self.site_reg.insert((w as WarpId, epoch), a.site);
                let mut live = self.warps.0[w].take().expect("absorbed warp");
                std::mem::swap(&mut s.sites, &mut live.sites);
                self.warps.0[w] = Some(s);
                self.lane_g2t = lane_g2t;
                self.as_of_seq = Some(a.seq);
                self.access_core(&a, cur, stamp, lane, domain);
                self.as_of_seq = None;
                let mut s = self.warps.0[w].take().unwrap();
                std::mem::swap(&mut s.sites, &mut live.sites);
                self.warps.0[w] = Some(live);
            }
            (Cur::Async { a: i }, Snap::Async(s)) => {
                let reg: Arc<AsyncActor> = Arc::new((*s).clone());
                for side in 1..=2 {
                    self.op_reg.insert((s.actor, s.gen_base + side), reg.clone());
                }
                let live = self.asyncs.0[i].take().expect("absorbed slot");
                self.asyncs.0[i] = Some(s);
                self.lane_g2t = None;
                self.as_of_seq = Some(a.seq);
                self.access_core(&a, cur, stamp, lane, domain);
                self.as_of_seq = None;
                self.asyncs.0[i] = Some(live);
            }
            _ => unreachable!("snapshot kind matches the actor"),
        }
    }

    /// Split off the child for one partition (`key` = its first cluster id,
    /// `ctas` = its CTAs). The main checker keeps everything else.
    pub(crate) fn split(&mut self, key: u32, ctas: &[u32], meta: Arc<HashMap<AllocId, AllocMeta>>, reserve: usize) -> Checker {
        let pool = key + 1;
        // Deterministic slot reservation for the phase (the main checker
        // grows the pool; a child that runs out suspends at that issue).
        let nw = self.topo.num_warps();
        let free = self.pools.get(&pool).map_or(0, |p| p.free.len());
        for _ in free..reserve {
            let i = self.asyncs.0.len();
            self.asyncs.0.push(Some(Box::new(AsyncActor::placeholder(nw + i as u32))));
            let p = self.pools.entry(pool).or_default();
            p.slots.push(i);
            p.free.insert(0, i);
        }
        let cpc = self.topo.ctas_per_cluster.max(1);
        let clusters: HashSet<u32> = ctas.iter().map(|c| c / cpc).collect();
        let mut child = Checker::new_empty(self);
        let wpc = self.topo.warps_per_cta;
        for &c in ctas {
            for w in c * wpc..(c + 1) * wpc {
                child.warps.0[w as usize] = self.warps.0[w as usize].take();
                if let Some(v) = self.poll_stash.remove(&w) {
                    child.poll_stash.insert(w, v);
                }
            }
        }
        if let Some(p) = self.pools.remove(&pool) {
            for &i in &p.slots {
                if child.asyncs.0.len() <= i {
                    child.asyncs.0.resize_with(i + 1, || None);
                }
                child.asyncs.0[i] = self.asyncs.0[i].take();
            }
            child.pools.insert(pool, p);
        }
        let ids: Vec<AllocId> =
            self.allocs.iter().filter(|(_, a)| matches!(a.space, Space::Shared | Space::Tmem) && clusters.contains(&(a.cta / cpc))).map(|(id, _)| *id).collect();
        for id in ids {
            child.allocs.insert(id, self.allocs.remove(&id).unwrap());
            if let Some(w) = self.words.remove(&id) {
                child.words.insert(id, w);
            }
            if let Some(w) = self.alias_writers.remove(&id) {
                child.alias_writers.insert(id, w);
            }
        }
        let objs: Vec<SyncObjId> = self.phases.keys().copied().filter(|o| self.obj_cluster(*o).is_some_and(|c| clusters.contains(&c))).collect();
        for o in objs {
            child.phases.insert(o, self.phases.remove(&o).unwrap());
        }
        child.part = Some(Box::new(PartCtx {
            meta,
            clusters,
            pool,
            deferred: Vec::new(),
            suspended: false,
            tag: (0, 0),
            finding_tags: Vec::new(),
            incomplete_tags: Vec::new(),
            finding_keys: Vec::new(),
        }));
        child
    }

    /// A child shell: same configuration, no state.
    fn new_empty(main: &Checker) -> Checker {
        let mut c = Checker::new_shell(main.topo);
        c.site_buffer = main.site_buffer.clone();
        c.operand_buffer = main.operand_buffer.clone();
        c.site_buffer_space = main.site_buffer_space.clone();
        c.poll_sites = main.poll_sites.clone();
        c.gc_every = main.gc_every;
        c.mbarrier_scope_assumed = main.mbarrier_scope_assumed;
        c.max_findings = 0;
        c.asyncs.0.resize_with(main.asyncs.0.len(), || None);
        c
    }

    /// The round merge for one partition (in partition order): take the
    /// child's state back, merge its report entries and apply its deferred
    /// global accesses in tag order, then process `stash` here.
    pub(crate) fn absorb(&mut self, mut child: Checker, stash: Vec<Stashed>) {
        let part = *child.part.take().expect("child");
        for (w, slot) in child.warps.0.iter_mut().enumerate() {
            if let Some(x) = slot.take() {
                self.warps.0[w] = Some(x);
            }
        }
        for (w, v) in child.poll_stash.drain() {
            self.poll_stash.insert(w, v);
        }
        for (k, p) in child.pools.drain() {
            for &i in &p.slots {
                if self.asyncs.0.len() <= i {
                    self.asyncs.0.resize_with(i + 1, || None);
                }
                if let Some(x) = child.asyncs.0.get_mut(i).and_then(|s| s.take()) {
                    self.asyncs.0[i] = Some(x);
                }
            }
            self.pools.insert(k, p);
        }
        for (id, a) in child.allocs.drain() {
            self.allocs.insert(id, a);
        }
        for (id, w) in child.words.drain() {
            self.words.insert(id, w);
        }
        for (id, w) in child.alias_writers.drain() {
            self.alias_writers.insert(id, w);
        }
        for (o, ph) in child.phases.drain() {
            self.phases.insert(o, ph);
        }
        for (k, v) in child.sc.drain() {
            self.sc.insert(k, v);
        }
        self.since_gc += child.since_gc;
        self.last_seq = self.last_seq.max(child.last_seq);
        self.stats.accesses += child.stats.accesses;
        // Report entries and deferred accesses, in replay order.
        enum Item {
            Finding(usize),
            Incomplete(usize),
            Deferred(Deferred),
        }
        let mut items: Vec<(Tag, u8, usize, Item)> = Vec::new();
        for (i, t) in part.finding_tags.iter().enumerate() {
            items.push((*t, 0, i, Item::Finding(i)));
        }
        for (i, t) in part.incomplete_tags.iter().enumerate() {
            items.push((*t, 0, i, Item::Incomplete(i)));
        }
        for (i, d) in part.deferred.into_iter().enumerate() {
            items.push((d.tag, 1, i, Item::Deferred(d)));
        }
        // Same tag: a child-side entry comes from the child event itself;
        // a deferred access has no report entry of its own in the child,
        // so the order within one tag is entries first, by push order.
        items.sort_by_key(|(t, r, i, _)| (*t, *r, *i));
        let mut findings: Vec<Option<Finding>> = child.report.findings.into_iter().map(Some).collect();
        for (_, _, _, it) in items {
            match it {
                Item::Finding(i) => {
                    let f = findings[i].take().unwrap();
                    let key = part.finding_keys[i].clone();
                    self.merge_finding(f, key);
                }
                Item::Incomplete(i) => {
                    let n = child.report.incomplete_counts[i];
                    let inc = child.report.incomplete[i].clone();
                    match self.incomplete_index.get(&inc) {
                        Some(&k) => self.report.incomplete_counts[k] += n,
                        None => {
                            self.incomplete_index.insert(inc.clone(), self.report.incomplete.len());
                            self.report.incomplete.push(inc);
                            self.report.incomplete_counts.push(n);
                        }
                    }
                }
                Item::Deferred(d) => self.apply_deferred(d),
            }
        }
        for s in stash {
            match s {
                Stashed::Event(e) => self.event(e),
                Stashed::Note(i) => self.note_incomplete(i),
            }
        }
    }

    /// Fold a child's finding (all its occurrences) into the report as the
    /// report functions would have, one occurrence at a time.
    fn merge_finding(&mut self, f: Finding, key: Option<FKey>) {
        let existing = match &key {
            Some(FKey::Race(k)) => self.dedup.get(k).copied(),
            Some(FKey::Advisory(k)) => self.advisory_dedup.get(k).copied(),
            Some(FKey::Scope(k)) => self.scope_dedup.get(k).copied(),
            Some(FKey::Alias(k)) => self.alias_dedup.get(k).copied(),
            None => None,
        };
        match existing {
            Some(i) => {
                let tmem = self.is_tmem(self.report.findings[i].alloc);
                let e = &mut self.report.findings[i];
                e.occurrences += f.occurrences;
                if matches!(key, Some(FKey::Race(_))) {
                    e.bytes = e.bytes.start.min(f.bytes.start)..e.bytes.end.max(f.bytes.end);
                }
                if tmem {
                    if let Some(t) = &f.tmem {
                        let et = e.tmem.get_or_insert_with(Default::default);
                        merge_tmem(et, t);
                    }
                }
                for s in &f.spans {
                    add_span(&mut e.spans, s.clone());
                }
            }
            None => {
                if self.max_findings != 0 && self.report.findings.len() >= self.max_findings {
                    // Serial would have tried (and dropped) every occurrence.
                    self.dropped_findings += f.occurrences;
                    return;
                }
                self.report.findings.push(f);
                let i = self.report.findings.len() - 1;
                match key {
                    Some(FKey::Race(k)) => {
                        self.dedup.insert(k, i);
                    }
                    Some(FKey::Advisory(k)) => {
                        self.advisory_dedup.insert(k, i);
                    }
                    Some(FKey::Scope(k)) => {
                        self.scope_dedup.insert(k, i);
                    }
                    Some(FKey::Alias(k)) => {
                        self.alias_dedup.insert(k, i);
                    }
                    None => {}
                }
            }
        }
    }
}

/// Events and notes a child could not process (from its first
/// global-dependent event on), for the main checker in order.
pub(crate) enum Stashed {
    Event(Event),
    Note(Incomplete),
}

/// Union of a child finding's TMEM rectangles into the main one (the
/// rectangles are per-axis unions, so the merge is order-free).
fn merge_tmem(e: &mut TmemRects, t: &TmemRects) {
    for r in &t.0 {
        add_span(&mut e.0, r.clone());
    }
    for r in &t.1 {
        add_span(&mut e.1, r.clone());
    }
}
