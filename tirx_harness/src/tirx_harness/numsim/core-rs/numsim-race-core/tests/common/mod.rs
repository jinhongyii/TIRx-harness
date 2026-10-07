//! A tiny event-sequence builder for handwritten semantic cases.
#![allow(dead_code)]

use std::collections::HashMap;
use std::ops::Range;

use numsim_race_core::input::*;
use numsim_race_core::*;

pub const SMEM: AllocId = 1;
pub const GMEM: AllocId = 2;
pub const GMEM2: AllocId = 3;
pub const TMEM: AllocId = 4;
pub const SMEM1: AllocId = 5; // second CTA's shared memory

pub struct K {
    pub topo: Topology,
    pub ev: Vec<Event>,
    epoch: HashMap<WarpId, u32>,
    spaces: HashMap<AllocId, Space>,
    phase: HashMap<SyncObjId, u32>,
    next_op: AsyncId,
}

#[derive(Clone, Copy)]
pub struct Op {
    pub kind: AccessKind,
    pub order: MemOrder,
    pub scope: Option<Scope>,
    pub atomic: bool,
}

pub const PLAIN_ST: Op = Op { kind: AccessKind::Write, order: MemOrder::Weak, scope: None, atomic: false };
pub const PLAIN_LD: Op = Op { kind: AccessKind::Read, order: MemOrder::Weak, scope: None, atomic: false };
pub const fn st(order: MemOrder, s: Scope) -> Op {
    Op { kind: AccessKind::Write, order, scope: Some(s), atomic: false }
}
pub const fn ld(order: MemOrder, s: Scope) -> Op {
    Op { kind: AccessKind::Read, order, scope: Some(s), atomic: false }
}
pub const fn atom(order: MemOrder, s: Scope) -> Op {
    Op { kind: AccessKind::Rmw, order, scope: Some(s), atomic: true }
}

impl K {
    pub fn new(warps_per_cta: u32, ctas_per_cluster: u32, num_ctas: u32) -> Self {
        let mut k = K {
            topo: Topology { warps_per_cta, ctas_per_cluster, num_ctas },
            ev: vec![],
            epoch: HashMap::new(),
            spaces: HashMap::new(),
            phase: HashMap::new(),
            next_op: 100,
        };
        k.alloc(SMEM, Space::Shared, 4096);
        k.alloc(SMEM1, Space::Shared, 4096);
        k.alloc(GMEM, Space::Global, 4096);
        k.alloc(GMEM2, Space::Global, 4096);
        k.alloc(TMEM, Space::Tmem, 1 << 16);
        k
    }
    pub fn one_warp() -> Self {
        K::new(1, 1, 1)
    }

    pub fn alloc(&mut self, alloc: AllocId, space: Space, size: u64) {
        self.spaces.insert(alloc, space);
        self.ev.push(Event::Sync(SyncEvent::AllocBegin { alloc, space, size, cta: 0 }));
    }

    pub fn tick(&mut self, w: WarpId) -> u32 {
        let e = self.epoch.entry(w).or_insert(0);
        *e += 1;
        *e
    }

    fn domain(&self, alloc: AllocId) -> Option<Domain> {
        match self.spaces[&alloc] {
            Space::Shared => Some(Domain::SharedCta),
            Space::Global => Some(Domain::Global),
            Space::Tmem => None,
        }
    }

    /// One warp instruction: every lane in `lanes` accesses `f(lane)`.
    pub fn inst(&mut self, w: WarpId, lanes: &[u8], op: Op, f: impl Fn(u8) -> (AllocId, Range<u64>)) -> &mut Self {
        self.inst_in(w, lanes, op, None, f)
    }

    pub fn inst_in(&mut self, w: WarpId, lanes: &[u8], op: Op, domain: Option<Domain>, f: impl Fn(u8) -> (AllocId, Range<u64>)) -> &mut Self {
        let e = self.tick(w);
        for &lane in lanes {
            let (alloc, range) = f(lane);
            let d = domain.or(self.domain(alloc));
            self.ev.push(Event::Access(Access {
                who: Who::Lane { warp: w, lane, epoch: e },
                alloc,
                range,
                kind: op.kind,
                order: op.order,
                scope: op.scope,
                atomic: op.atomic,
                proxy: Proxy::Generic,
                domain: d,
                site: w * 1000 + e,
            }));
        }
        self
    }

    pub fn a(&mut self, w: WarpId, lane: u8, op: Op, alloc: AllocId, r: Range<u64>) -> &mut Self {
        self.inst(w, &[lane], op, |_| (alloc, r.clone()))
    }
    pub fn st(&mut self, w: WarpId, lane: u8, alloc: AllocId, r: Range<u64>) -> &mut Self {
        self.a(w, lane, PLAIN_ST, alloc, r)
    }
    pub fn ld(&mut self, w: WarpId, lane: u8, alloc: AllocId, r: Range<u64>) -> &mut Self {
        self.a(w, lane, PLAIN_LD, alloc, r)
    }

    pub fn syncwarp(&mut self, w: WarpId, mask: u32) -> &mut Self {
        let e = self.tick(w);
        self.ev.push(Event::Sync(SyncEvent::WarpSync { warp: w, mask: LaneMask(mask), epoch: e }));
        self
    }

    pub fn next_phase(&mut self, obj: SyncObjId) -> u32 {
        let p = self.phase.entry(obj).or_insert(0);
        *p += 1;
        *p - 1
    }

    /// bar.sync / cluster barrier over whole warps.
    pub fn bar(&mut self, obj: SyncObjId, warps: &[WarpId]) -> &mut Self {
        let p = self.next_phase(obj);
        for &w in warps {
            self.arrive(w, u32::MAX, obj, p, true);
        }
        for &w in warps {
            self.wait(w, u32::MAX, obj, p, true);
        }
        self
    }

    pub fn arrive(&mut self, w: WarpId, lanes: u32, obj: SyncObjId, phase: u32, release: bool) -> &mut Self {
        let e = self.tick(w);
        self.ev.push(Event::Sync(SyncEvent::Arrive { warp: w, lanes: LaneMask(lanes), obj, phase, release, epoch: e }));
        self
    }

    pub fn wait(&mut self, w: WarpId, lanes: u32, obj: SyncObjId, phase: u32, acquire: bool) -> &mut Self {
        let e = self.tick(w);
        self.ev.push(Event::Sync(SyncEvent::Wait { warp: w, lanes: LaneMask(lanes), obj, phase, acquire, epoch: e }));
        self
    }

    pub fn fence(&mut self, w: WarpId, lanes: u32, kind: FenceKind) -> &mut Self {
        let e = self.tick(w);
        self.ev.push(Event::Sync(SyncEvent::Fence { warp: w, lanes: LaneMask(lanes), kind, epoch: e }));
        self
    }

    pub fn issue(&mut self, w: WarpId, lane: u8, kind: AsyncKind, proxy: Proxy, preds: &[AsyncId], footprint: &[(AllocId, Range<u64>)]) -> AsyncId {
        let op = self.next_op;
        self.next_op += 1;
        let e = self.tick(w);
        self.ev.push(Event::Sync(SyncEvent::AsyncIssue {
            op,
            warp: w,
            lanes: LaneMask::lane(lane),
            kind,
            proxy,
            preds: preds.to_vec(),
            footprint: footprint.to_vec(),
            epoch: e,
        }));
        op
    }

    pub fn aacc(&mut self, op: AsyncId, side: Milestone, kind: AccessKind, proxy: Proxy, alloc: AllocId, r: Range<u64>) -> &mut Self {
        let d = self.domain(alloc);
        self.ev.push(Event::Access(Access {
            who: Who::Async { op, side },
            alloc,
            range: r,
            kind,
            order: MemOrder::Weak,
            scope: None,
            atomic: false,
            proxy,
            domain: d,
            site: 900_000 + op,
        }));
        self
    }

    /// Async read (milestone 1).
    pub fn ar(&mut self, op: AsyncId, proxy: Proxy, alloc: AllocId, r: Range<u64>) -> &mut Self {
        self.aacc(op, Milestone::Read, AccessKind::Read, proxy, alloc, r)
    }
    /// Async write (milestone 2).
    pub fn aw(&mut self, op: AsyncId, proxy: Proxy, alloc: AllocId, r: Range<u64>) -> &mut Self {
        self.aacc(op, Milestone::Write, AccessKind::Write, proxy, alloc, r)
    }

    pub fn done_phase(&mut self, op: AsyncId, m: Milestone, obj: SyncObjId, phase: u32) -> &mut Self {
        self.ev.push(Event::Sync(SyncEvent::AsyncComplete { op, milestone: m, target: CompletionTarget::Phase { obj, phase } }));
        self
    }
    pub fn done_warp(&mut self, op: AsyncId, m: Milestone, w: WarpId, lanes: u32) -> &mut Self {
        self.ev.push(Event::Sync(SyncEvent::AsyncComplete { op, milestone: m, target: CompletionTarget::Warp { warp: w, lanes: LaneMask(lanes) } }));
        self
    }

    pub fn declare(&mut self, alloc: AllocId, r: Range<u64>) -> &mut Self {
        self.ev.push(Event::Sync(SyncEvent::DeclareWord { alloc, range: r }));
        self
    }

    pub fn wait_until(&mut self, w: WarpId, lane: u8, alloc: AllocId, r: Range<u64>, scope: Scope, accepted: u64, observed: u32) -> &mut Self {
        let e = self.tick(w);
        self.ev.push(Event::Sync(SyncEvent::WaitVerdicts {
            warp: w,
            lanes: LaneMask::lane(lane),
            alloc,
            range: r,
            scope,
            accepted: vec![accepted],
            observed,
            pred_reads: vec![],
            epoch: e,
        }));
        self
    }

    pub fn raw(&mut self, s: SyncEvent) -> &mut Self {
        self.ev.push(Event::Sync(s));
        self
    }

    pub fn run(&self) -> Report {
        Checker::run(self.topo, self.ev.clone())
    }
}

pub fn races(r: &Report) -> Vec<&Finding> {
    r.races().collect()
}

pub fn clean(r: &Report) -> bool {
    r.is_clean()
}

pub fn has_failure(r: &Report, pred: impl Fn(OrderingFailure) -> bool) -> bool {
    r.findings.iter().any(|f| match f.kind {
        FindingKind::DataRace { failure, .. } | FindingKind::TmemLifetimeReview { failure, .. } => pred(failure),
        _ => false,
    })
}

pub fn has_class(r: &Report, class: RaceClass) -> bool {
    r.findings.iter().any(|f| matches!(f.kind, FindingKind::DataRace { class: c, .. } if c == class))
}

pub fn has_scope_mismatch(r: &Report) -> bool {
    r.findings.iter().any(|f| matches!(f.kind, FindingKind::ScopeMismatch { .. }))
}

pub fn review_only(r: &Report) -> bool {
    !r.findings.is_empty() && r.findings.iter().all(|f| f.severity == Severity::Review)
}

/// At least one error-severity data race (not merely incomplete).
pub fn has_race(r: &Report) -> bool {
    r.findings.iter().any(|f| f.severity == Severity::Error && matches!(f.kind, FindingKind::DataRace { .. }))
}
