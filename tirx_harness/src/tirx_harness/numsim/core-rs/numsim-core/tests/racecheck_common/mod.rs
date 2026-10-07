//! Event-sequence builder for racecheck scenario tests. It emits *contract*
//! events (`observe::Access` batches and `observe::SyncEvent`s) and drives
//! `racecheck::RaceObserver` through the `Observer` trait, exactly as the
//! engine will. (Switch to `testutil::ProgramBuilder` + the interpreter
//! once the W2 handlers emit these events.)
#![allow(dead_code, unused_imports)]

use std::collections::HashMap;
use std::ops::Range;

pub use numsim_core::arena::AllocId;
use numsim_core::arena::{ByteSpan, Space};
use numsim_core::observe::{
    Access as CAccess, AccessSeq, Actor, AsyncClass, CtaId, LaneSpan, Observer, PublishTarget, SyncEvent as CSync,
    SyncKind, WarpId as CWarpId, Window, ALL_LANES,
};
use numsim_core::program::Sem;
use numsim_core::site::SiteId;
pub use numsim_core::sync::completion::AsyncId;
use numsim_core::sync::completion::ResourceId;
use numsim_core::value::WarpMask as LaneMask;

pub use numsim_core::observe::FenceEvent as FenceKind;
pub use numsim_core::racecheck::input::{AccessKind, Domain, MemOrder, Milestone, Proxy, Scope, Topology};
pub use numsim_core::racecheck::{
    AdvisoryKind, Incomplete, OrderingFailure, RaceClass, RaceFinding as Finding, RaceFindingKind as FindingKind,
    RaceObserver, RaceReport as Report, RacecheckConfig, Severity,
};
pub type AsyncKind = AsyncClass;
pub type WarpId = u32;
/// mbarrier number (see [`mbar`]).
pub type SyncObjId = u32;

pub const SMEM: AllocId = AllocId(1);
pub const GMEM: AllocId = AllocId(2);
pub const GMEM2: AllocId = AllocId(3);
pub const TMEM: AllocId = AllocId(4);
pub const SMEM1: AllocId = AllocId(5); // second CTA's shared memory

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

fn sem_of(op: &Op) -> Sem {
    match (op.order, op.scope) {
        (MemOrder::Weak, _) => Sem::Weak,
        (MemOrder::Relaxed, _) => Sem::Relaxed,
        (MemOrder::Acquire, _) => Sem::Acquire,
        (MemOrder::Release, _) => Sem::Release,
        (MemOrder::AcqRel, _) => Sem::AcqRel,
    }
}

/// Owned form of one contract event.
#[derive(Clone, Debug)]
pub enum Ev {
    Access {
        actor: Actor,
        site: SiteId,
        alloc: AllocId,
        space: Space,
        kind: AccessKind,
        sem: Sem,
        scope: Scope,
        atomic: bool,
        returns_value: bool,
        proxy: Proxy,
        window: Option<Window>,
        spans: Vec<LaneSpan>,
    },
    Sync(CSync),
}

pub fn mbar(n: u32) -> ResourceId {
    ResourceId::Mbarrier { cta: CtaId(0), alloc: AllocId(99), offset: n }
}

pub struct K {
    pub topo: Topology,
    pub ev: Vec<Ev>,
    epoch: HashMap<WarpId, u32>,
    spaces: HashMap<AllocId, Space>,
    phase: HashMap<u64, u64>,
    next_op: u64,
    pub gc_every: u64,
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
            gc_every: 1 << 14,
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

    fn sync_ev(&mut self, actor: Actor, lanes: LaneMask, kind: SyncKind) {
        self.ev.push(Ev::Sync(CSync { actor, seq: 0, site: SiteId(0), frames: vec![], lanes, kind }));
    }

    pub fn alloc(&mut self, alloc: AllocId, space: Space, size: u64) {
        self.spaces.insert(alloc, space);
        self.sync_ev(Actor::Host, LaneMask::NONE, SyncKind::AllocBegin { alloc, space, size, cta: CtaId(0) });
    }

    pub fn alloc_end(&mut self, alloc: AllocId) -> &mut Self {
        self.sync_ev(Actor::Host, LaneMask::NONE, SyncKind::AllocEnd { alloc });
        self
    }

    pub fn tick(&mut self, w: WarpId) -> u32 {
        let e = self.epoch.entry(w).or_insert(0);
        *e += 1;
        *e
    }

    fn wactor(&mut self, w: WarpId) -> Actor {
        let e = self.tick(w);
        Actor::Warp { warp: CWarpId(w), epoch: e }
    }

    fn window(&self, alloc: AllocId) -> Option<Window> {
        match self.spaces[&alloc] {
            Space::Shared => Some(Window::SharedCta),
            Space::Global => Some(Window::Global),
            _ => None,
        }
    }

    /// One warp instruction: every lane in `lanes` accesses `f(lane)`.
    /// All lanes must touch one allocation.
    pub fn inst(&mut self, w: WarpId, lanes: &[u8], op: Op, f: impl Fn(u8) -> (AllocId, Range<u64>)) -> &mut Self {
        self.inst_in(w, lanes, op, None, f)
    }

    pub fn inst_in(&mut self, w: WarpId, lanes: &[u8], op: Op, window: Option<Domain>, f: impl Fn(u8) -> (AllocId, Range<u64>)) -> &mut Self {
        let actor = self.wactor(w);
        let Actor::Warp { epoch, .. } = actor else { unreachable!() };
        let mut spans = Vec::new();
        let mut alloc = None;
        for &lane in lanes {
            let (a, r) = f(lane);
            assert!(alloc.is_none_or(|x| x == a), "one allocation per instruction");
            alloc = Some(a);
            spans.push(LaneSpan { lane, span: ByteSpan::new(r.start, r.end - r.start) });
        }
        let alloc = alloc.unwrap();
        let space = self.spaces[&alloc];
        let window = window.or(self.window(alloc));
        self.ev.push(Ev::Access {
            actor,
            site: SiteId(w * 1000 + epoch),
            alloc,
            space,
            kind: op.kind,
            sem: sem_of(&op),
            scope: op.scope.unwrap_or(Scope::Gpu),
            atomic: op.atomic,
            returns_value: op.atomic,
            proxy: Proxy::Generic,
            window,
            spans,
        });
        self
    }

    /// `red` (an RMW that returns nothing).
    pub fn red(&mut self, w: WarpId, lane: u8, order: MemOrder, s: Scope, alloc: AllocId, r: Range<u64>) -> &mut Self {
        self.inst(w, &[lane], atom(order, s), |_| (alloc, r.clone()));
        if let Some(Ev::Access { returns_value, .. }) = self.ev.last_mut() {
            *returns_value = false;
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
        let a = self.wactor(w);
        self.sync_ev(a, LaneMask(mask), SyncKind::WarpSync { mask: LaneMask(mask) });
        self
    }

    pub fn next_phase(&mut self, key: u64) -> u64 {
        let p = self.phase.entry(key).or_insert(0);
        *p += 1;
        *p - 1
    }

    pub fn arrive_r(&mut self, w: WarpId, lanes: u32, obj: ResourceId, phase: u64, release: bool) -> &mut Self {
        let a = self.wactor(w);
        self.sync_ev(a, LaneMask(lanes), SyncKind::Arrive { obj, phase, release });
        self
    }
    pub fn wait_r(&mut self, w: WarpId, lanes: u32, obj: ResourceId, phase: u64, acquire: bool) -> &mut Self {
        let a = self.wactor(w);
        self.sync_ev(a, LaneMask(lanes), SyncKind::Wait { obj, phase, acquire });
        self
    }

    /// Named barrier `bar.sync` over whole warps (participants, no scope).
    pub fn bar(&mut self, id: u8, warps: &[WarpId]) -> &mut Self {
        let p = self.next_phase(1000 + id as u64);
        let cta = self.topo.cta_of(warps[0]);
        let obj = ResourceId::Named { cta: CtaId(cta), id };
        for &w in warps {
            self.arrive_r(w, u32::MAX, obj, p, true);
        }
        for &w in warps {
            self.wait_r(w, u32::MAX, obj, p, true);
        }
        self
    }
    /// `barrier.cluster.arrive` + `wait` (defaults release/acquire).
    pub fn cluster_bar(&mut self, warps: &[WarpId]) -> &mut Self {
        let p = self.next_phase(2000);
        let obj = ResourceId::Cluster { cluster: self.topo.cluster_of(warps[0]) };
        for &w in warps {
            self.arrive_r(w, u32::MAX, obj, p, true);
        }
        for &w in warps {
            self.wait_r(w, u32::MAX, obj, p, true);
        }
        self
    }

    /// mbarrier arrive / wait (`obj` names an mbarrier).
    pub fn arrive(&mut self, w: WarpId, lanes: u32, obj: u32, phase: u64, release: bool) -> &mut Self {
        self.arrive_r(w, lanes, mbar(obj), phase, release)
    }
    pub fn wait(&mut self, w: WarpId, lanes: u32, obj: u32, phase: u64, acquire: bool) -> &mut Self {
        self.wait_r(w, lanes, mbar(obj), phase, acquire)
    }

    pub fn fence(&mut self, w: WarpId, lanes: u32, kind: FenceKind) -> &mut Self {
        let a = self.wactor(w);
        self.sync_ev(a, LaneMask(lanes), SyncKind::Fence(kind));
        self
    }

    pub fn issue(&mut self, w: WarpId, lane: u8, kind: AsyncKind, proxy: Proxy, preds: &[AsyncId], footprint: &[(AllocId, Range<u64>)]) -> AsyncId {
        let op = AsyncId(self.next_op);
        self.next_op += 1;
        let a = self.wactor(w);
        let footprint = footprint.iter().map(|(a, r)| (*a, ByteSpan::new(r.start, r.end - r.start))).collect();
        self.sync_ev(
            a,
            LaneMask::lane(lane as usize),
            SyncKind::AsyncIssue { op, class: kind, proxy, preds: preds.to_vec(), footprint, targets: vec![] },
        );
        op
    }

    pub fn aacc(&mut self, op: AsyncId, side: Milestone, kind: AccessKind, proxy: Proxy, alloc: AllocId, r: Range<u64>) -> &mut Self {
        let window = self.window(alloc);
        self.ev.push(Ev::Access {
            actor: Actor::Async { op, side },
            site: SiteId(900_000 + op.0 as u32),
            alloc,
            space: self.spaces[&alloc],
            kind,
            sem: Sem::Weak,
            scope: Scope::Gpu,
            atomic: false,
            returns_value: false,
            proxy,
            window,
            spans: vec![LaneSpan { lane: ALL_LANES, span: ByteSpan::new(r.start, r.end - r.start) }],
        });
        self
    }

    /// Async read (read side).
    pub fn ar(&mut self, op: AsyncId, proxy: Proxy, alloc: AllocId, r: Range<u64>) -> &mut Self {
        self.aacc(op, Milestone::Read, AccessKind::Read, proxy, alloc, r)
    }
    /// Async write (write side).
    pub fn aw(&mut self, op: AsyncId, proxy: Proxy, alloc: AllocId, r: Range<u64>) -> &mut Self {
        self.aacc(op, Milestone::Write, AccessKind::Write, proxy, alloc, r)
    }

    pub fn done_phase(&mut self, op: AsyncId, m: Milestone, obj: u32, phase: u64) -> &mut Self {
        self.sync_ev(Actor::Async { op, side: m }, LaneMask::NONE, SyncKind::AsyncComplete { op, milestone: m, target: PublishTarget::Phase { obj: mbar(obj), phase } });
        self
    }
    pub fn done_warp(&mut self, op: AsyncId, m: Milestone, w: WarpId, lanes: u32) -> &mut Self {
        self.sync_ev(
            Actor::Async { op, side: m },
            LaneMask::NONE,
            SyncKind::AsyncComplete { op, milestone: m, target: PublishTarget::Warp { warp: CWarpId(w), lanes: LaneMask(lanes) } },
        );
        self
    }

    pub fn declare(&mut self, alloc: AllocId, r: Range<u64>) -> &mut Self {
        self.sync_ev(Actor::Host, LaneMask::NONE, SyncKind::DeclareWord { alloc, span: ByteSpan::new(r.start, r.end - r.start) });
        self
    }

    pub fn wait_until(&mut self, w: WarpId, lane: u8, alloc: AllocId, r: Range<u64>, scope: Scope, accepted: u64, observed: u32) -> &mut Self {
        self.wait_until_pred(w, lane, alloc, r, scope, accepted, observed, &[])
    }

    #[allow(clippy::too_many_arguments)]
    pub fn wait_until_pred(
        &mut self,
        w: WarpId,
        lane: u8,
        alloc: AllocId,
        r: Range<u64>,
        scope: Scope,
        accepted: u64,
        observed: u32,
        pred_reads: &[(AllocId, Range<u64>)],
    ) -> &mut Self {
        let a = self.wactor(w);
        self.sync_ev(
            a,
            LaneMask::lane(lane as usize),
            SyncKind::WaitVerdicts {
                alloc,
                span: ByteSpan::new(r.start, r.end - r.start),
                scope,
                accepted: vec![accepted],
                observed,
                pred_reads: pred_reads.iter().map(|(a, r)| (*a, ByteSpan::new(r.start, r.end - r.start))).collect(),
            },
        );
        self
    }

    /// Drive a fresh observer through every event; return it finished.
    pub fn observe(&self) -> RaceObserver {
        let mut obs = RaceObserver::new(RacecheckConfig::default());
        obs.gc_every = self.gc_every;
        obs.start_launch(self.topo, 0);
        for (i, e) in self.ev.iter().enumerate() {
            match e {
                Ev::Access { actor, site, alloc, space, kind, sem, scope, atomic, returns_value, proxy, window, spans } => {
                    let a = CAccess {
                        seq: AccessSeq(i as u64),
                        actor: *actor,
                        site: *site,
                        alloc: *alloc,
                        space: *space,
                        kind: *kind,
                        sem: *sem,
                        scope: *scope,
                        atomic: *atomic,
                        returns_value: *returns_value,
                        proxy: *proxy,
                        window: *window,
                        spans,
                        declared_word: false,
                    };
                    obs.access(&a);
                }
                Ev::Sync(s) => obs.sync(s),
            }
        }
        obs.finish_launch();
        obs
    }

    pub fn run(&self) -> Report {
        self.observe().launches.pop().unwrap().report
    }
}

pub fn races(r: &Report) -> Vec<&Finding> {
    r.races().collect()
}

/// No error and no incomplete. Review advisories are allowed; tests that
/// care assert them explicitly.
pub fn clean(r: &Report) -> bool {
    r.errors().next().is_none() && r.incomplete.is_empty() && r.findings.iter().all(|f| matches!(f.kind, FindingKind::Advisory { .. }))
}

pub fn has_advisory(r: &Report, kind: AdvisoryKind) -> bool {
    r.findings.iter().any(|f| f.kind == FindingKind::Advisory { kind })
}

/// At least one error-severity data race.
pub fn has_race(r: &Report) -> bool {
    r.findings.iter().any(|f| f.severity == Severity::Error && matches!(f.kind, FindingKind::DataRace { .. }))
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
