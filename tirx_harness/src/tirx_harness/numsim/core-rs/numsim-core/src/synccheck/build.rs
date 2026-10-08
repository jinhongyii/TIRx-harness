//! Handwritten contract `SyncEvent` logs for tests and benchmarks.

use std::collections::BTreeMap;

use crate::arena::AllocId;
use crate::observe::{
    Actor, AsyncTarget, Collective, Counts, CtaId, Observer, ProtocolCmd, ProtocolStatus, RecordingObserver, SyncEvent,
    SyncKind, WarpId,
};
use crate::site::SiteId;
use crate::sync::{async_group, cluster, mbarrier, named, setmaxnreg, tcgen, ResourceId, SyncCmd, SyncError, FULL_MASK};
use crate::value::WarpMask;

#[derive(Default)]
pub struct LogBuilder {
    /// `SyncEvent::kernel` of every event.
    pub kernel: u32,
    obs: RecordingObserver,
    seq: BTreeMap<u32, u32>,
    epoch: BTreeMap<u32, u32>,
    next_collective: u64,
}

impl LogBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// One `Protocol` event with full control over its fields; `observed`
    /// is the parity a successful test/try_wait observed (every target).
    pub fn event(
        &mut self,
        warp: u32,
        site: u32,
        cmds: Vec<(ResourceId, SyncCmd)>,
        issued: Vec<AsyncTarget>,
        observed: Option<u8>,
        collective: Option<Collective>,
        status: ProtocolStatus,
    ) -> &mut Self {
        let cmds = cmds
            .into_iter()
            .map(|(res, cmd)| ProtocolCmd { res, cmd, counts: Counts::default(), observed_parity: observed })
            .collect();
        let epoch = self.epoch.entry(warp).or_insert(0);
        *epoch += 1;
        let seq = self.seq.entry(warp).or_insert(0);
        let event = SyncEvent {
            kernel: self.kernel,
            actor: Actor::Warp { warp: WarpId(warp), epoch: u64::from(*epoch) },
            seq: *seq,
            site: SiteId(site),
            frames: Vec::new(),
            lanes: WarpMask(u32::MAX),
            kind: SyncKind::Protocol { cmds, collective, issued, status: status.clone() },
        };
        if status == ProtocolStatus::Committed {
            *seq += 1;
        }
        self.obs.sync(&event);
        self
    }

    pub fn cmd(&mut self, warp: u32, site: u32, res: ResourceId, cmd: SyncCmd) -> &mut Self {
        self.event(warp, site, vec![(res, cmd)], Vec::new(), None, None, ProtocolStatus::Committed)
    }

    pub fn cmds(&mut self, warp: u32, site: u32, cmds: Vec<(ResourceId, SyncCmd)>) -> &mut Self {
        self.event(warp, site, cmds, Vec::new(), None, None, ProtocolStatus::Committed)
    }

    /// A successful `try_wait`/`test_wait` that observed `parity`.
    pub fn test_ok(&mut self, warp: u32, site: u32, res: ResourceId, parity: u8) -> &mut Self {
        self.event(warp, site, vec![(res, test_parity(parity))], Vec::new(), Some(parity), None, ProtocolStatus::Committed)
    }

    /// An async op (TMA, commit, cp.async.mbarrier.arrive) promising `bytes`
    /// and/or `arrivals` to mbarrier `res`, as the contract delivers it: the
    /// event lists the `issued` target and only the commands the instruction
    /// itself executes (`cmds`, e.g. `TcgenWork(Commit)` or `ArriveOn`); no
    /// `Mbarrier(Issue)` is injected (the explorer derives the token).
    pub fn issue(&mut self, warp: u32, site: u32, res: ResourceId, bytes: u64, arrivals: u64, cmds: Vec<(ResourceId, SyncCmd)>) -> &mut Self {
        self.event(warp, site, cmds, vec![AsyncTarget { res, bytes, arrivals }], None, None, ProtocolStatus::Committed)
    }

    /// One collective rendezvous recorded by every participant.
    pub fn collective(&mut self, warps: &[u32], site: u32, cmds: Vec<(ResourceId, SyncCmd)>) -> &mut Self {
        let id = self.next_collective;
        self.next_collective += 1;
        let c = Collective { id, participants: warps.iter().map(|&w| WarpId(w)).collect() };
        for &w in warps {
            self.event(w, site, cmds.clone(), Vec::new(), None, Some(c.clone()), ProtocolStatus::Committed);
        }
        self
    }

    pub fn failed(&mut self, warp: u32, site: u32, res: ResourceId, cmd: SyncCmd, error: SyncError) -> &mut Self {
        self.event(warp, site, vec![(res, cmd)], Vec::new(), None, None, ProtocolStatus::Failed(error))
    }

    pub fn blocked_at_exit(&mut self, warp: u32, site: u32, res: ResourceId, cmd: SyncCmd) -> &mut Self {
        self.event(warp, site, vec![(res, cmd)], Vec::new(), None, None, ProtocolStatus::BlockedAtExit)
    }

    /// A host-side `Protocol` event (`Actor::Host`), e.g. the launch-bounds
    /// setmaxnreg `Configure` the scheduler applies before any warp runs.
    pub fn host(&mut self, cmds: Vec<(ResourceId, SyncCmd)>) -> &mut Self {
        let event = SyncEvent {
            kernel: self.kernel,
            actor: Actor::Host,
            seq: 0,
            site: SiteId::NONE,
            frames: Vec::new(),
            lanes: WarpMask(u32::MAX),
            kind: SyncKind::Protocol {
                cmds: cmds
                    .into_iter()
                    .map(|(res, cmd)| ProtocolCmd { res, cmd, counts: Counts::default(), observed_parity: None })
                    .collect(),
                collective: None,
                issued: Vec::new(),
                status: ProtocolStatus::Committed,
            },
        };
        self.obs.sync(&event);
        self
    }

    /// Start a new launch: per-warp sequences and epochs restart.
    pub fn restart_seq(&mut self) -> &mut Self {
        self.seq.clear();
        self.epoch.clear();
        self
    }

    pub fn build(&self) -> RecordingObserver {
        self.obs.clone()
    }
}

pub fn mbar(cta: u32, offset: u32) -> ResourceId {
    ResourceId::Mbarrier { cta: CtaId(cta), alloc: AllocId(cta), offset }
}
pub fn named_bar(cta: u32, id: u8) -> ResourceId {
    ResourceId::Named { cta: CtaId(cta), id }
}
pub fn cluster_bar(cluster: u32) -> ResourceId {
    ResourceId::Cluster { cluster }
}
pub fn reg_pool(cta: u32) -> ResourceId {
    ResourceId::RegPool { cta: CtaId(cta) }
}
pub fn tmem(pair: u32) -> ResourceId {
    // W3: `pair` is an opaque pair index here (cluster `pair`, pair rank 0).
    ResourceId::TcgenLifecycle { cluster: pair, pair_rank: 0 }
}
pub fn tcgen_work(warp: u32, lane: u8) -> ResourceId {
    ResourceId::TcgenWork { warp: WarpId(warp), lane }
}
pub fn async_group_res(warp: u32, lane: u8, domain: async_group::Domain) -> ResourceId {
    ResourceId::AsyncGroup { warp: WarpId(warp), lane, domain }
}

pub fn init(count: u64) -> SyncCmd {
    SyncCmd::Mbarrier(mbarrier::Cmd::Init { count, layout_v1: false })
}
pub fn inval() -> SyncCmd {
    SyncCmd::Mbarrier(mbarrier::Cmd::Inval)
}
pub fn arrive(count: u64) -> SyncCmd {
    SyncCmd::Mbarrier(mbarrier::Cmd::Arrive { count, tx: None, drop: false, no_complete: false })
}
pub fn arrive_tx(count: u64, tx: u64) -> SyncCmd {
    SyncCmd::Mbarrier(mbarrier::Cmd::Arrive { count, tx: Some(tx), drop: false, no_complete: false })
}
pub fn expect_tx(bytes: u64) -> SyncCmd {
    SyncCmd::Mbarrier(mbarrier::Cmd::ExpectTx { bytes })
}
pub fn inc_pending(count: u64) -> SyncCmd {
    SyncCmd::Mbarrier(mbarrier::Cmd::IncPending { count })
}
pub fn wait(parity: u64) -> SyncCmd {
    SyncCmd::Mbarrier(mbarrier::Cmd::WaitParity { parity })
}
pub fn test_parity(parity: u8) -> SyncCmd {
    SyncCmd::Mbarrier(mbarrier::Cmd::TestParity { parity: u64::from(parity) })
}
fn contribution(warp: u32, count: u64) -> named::Contribution {
    named::Contribution { warp, mask: FULL_MASK, live: FULL_MASK, count, aligned: true }
}
pub fn bar_sync(warp_in_cta: u32, count: u64) -> SyncCmd {
    SyncCmd::Named(named::Cmd::Sync(contribution(warp_in_cta, count)))
}
pub fn bar_arrive(warp_in_cta: u32, count: u64) -> SyncCmd {
    SyncCmd::Named(named::Cmd::Arrive(contribution(warp_in_cta, count)))
}
pub fn cl_arrive(warp: u32) -> SyncCmd {
    SyncCmd::Cluster(cluster::Cmd::Arrive { warp, mask: FULL_MASK, aligned: true })
}
pub fn cl_wait(warp: u32) -> SyncCmd {
    SyncCmd::Cluster(cluster::Cmd::Wait { warp, mask: FULL_MASK, aligned: true })
}
pub fn setmax(wg: u32, inc: bool, count: u32) -> SyncCmd {
    SyncCmd::RegPool(setmaxnreg::Cmd::Set { wg, inc, count })
}
pub fn configure(count: u32) -> SyncCmd {
    SyncCmd::RegPool(setmaxnreg::Cmd::Configure { count })
}
pub fn wg_sync(wg: u32) -> SyncCmd {
    SyncCmd::RegPool(setmaxnreg::Cmd::WarpgroupSync { wg })
}
pub fn tmem_alloc(columns: u32) -> SyncCmd {
    SyncCmd::Tcgen(tcgen::Cmd::Alloc { who: tcgen::Who::One(0), columns, exclusive: false })
}
pub fn tmem_dealloc(taddr: u32, columns: u32) -> SyncCmd {
    SyncCmd::Tcgen(tcgen::Cmd::Dealloc { who: tcgen::Who::One(0), taddr, columns, exclusive: false })
}
pub fn tmem_relinquish() -> SyncCmd {
    SyncCmd::Tcgen(tcgen::Cmd::Relinquish { who: tcgen::Who::One(0) })
}
pub fn group(cmd: async_group::Cmd) -> SyncCmd {
    SyncCmd::AsyncGroup(cmd)
}
pub fn work(cmd: tcgen::WorkCmd) -> SyncCmd {
    SyncCmd::TcgenWork(cmd)
}

/// `cta_sync` (`bar.sync 0`) by `warps` warps of CTA `cta`.
pub fn cta_sync(log: &mut LogBuilder, cta: u32, warps: &[u32], warps_per_cta: u32) {
    for &w in warps {
        log.cmd(w, 100, named_bar(cta, 0), bar_sync(w % warps_per_cta, u64::from(warps_per_cta) * 32));
    }
}

/// 1 producer + `warps - 1` consumers, `stages`-deep full/empty mbarrier
/// ring, `iterations` trips; TMA producer when `tma_bytes > 0`.
pub fn pipeline(warps: u32, stages: u32, iterations: u32, tma_bytes: u64) -> RecordingObserver {
    let consumers = u64::from(warps - 1);
    let full = |s: u32| mbar(0, 8 * s);
    let empty = |s: u32| mbar(0, 8 * (stages + s));
    let mut log = LogBuilder::new();
    for s in 0..stages {
        log.cmd(0, 1, full(s), init(1));
        log.cmd(0, 2, empty(s), init(consumers));
    }
    cta_sync(&mut log, 0, &(0..warps).collect::<Vec<_>>(), warps);
    for i in 0..iterations {
        let (s, round) = (i % stages, i / stages);
        if round >= 1 {
            log.cmd(0, 10, empty(s), wait(u64::from((round - 1) & 1)));
        }
        if tma_bytes > 0 {
            log.cmd(0, 11, full(s), arrive_tx(1, tma_bytes));
            log.issue(0, 12, full(s), tma_bytes, 0, Vec::new());
        } else {
            log.cmd(0, 11, full(s), arrive(1));
        }
    }
    for w in 1..warps {
        for i in 0..iterations {
            let (s, round) = (i % stages, i / stages);
            log.cmd(w, 20, full(s), wait(u64::from(round & 1)));
            log.cmd(w, 21, empty(s), arrive(1));
        }
    }
    log.build()
}

/// SM100 UMMA ring: warp 0 = TMA producer (`arrive.expect_tx` + TMA into
/// full[s]), warp 1 = MMA warp (waits full[s], `tcgen05.commit` arrives on
/// empty[s]; per tile waits tmem_empty and commits tmem_full), warps 2..6 =
/// epilogue (wait tmem_full, arrive tmem_empty). `kblocks` k-blocks per tile,
/// `tiles` tiles (persistent kernel), `stages`-deep smem ring.
pub fn umma_ring(stages: u32, kblocks: u32, tiles: u32) -> RecordingObserver {
    let full = |s: u32| mbar(0, 8 * s);
    let empty = |s: u32| mbar(0, 8 * (stages + s));
    let tmem_full = mbar(0, 8 * (2 * stages));
    let tmem_empty = mbar(0, 8 * (2 * stages + 1));
    let epilogue = [2u32, 3, 4, 5];
    let mut log = LogBuilder::new();
    for s in 0..stages {
        log.cmd(0, 1, full(s), init(1)).cmd(0, 1, empty(s), init(1));
    }
    log.cmd(0, 2, tmem_full, init(1)).cmd(0, 2, tmem_empty, init(epilogue.len() as u64));
    cta_sync(&mut log, 0, &(0..6).collect::<Vec<_>>(), 6);
    let bytes = 32 * 1024;
    for t in 0..tiles {
        for k in 0..kblocks {
            let i = t * kblocks + k;
            let (s, round) = (i % stages, i / stages);
            if round >= 1 {
                log.cmd(0, 10, empty(s), wait(u64::from((round - 1) & 1)));
            }
            log.cmd(0, 11, full(s), arrive_tx(1, bytes));
            log.issue(0, 12, full(s), bytes, 0, Vec::new());
        }
    }
    for t in 0..tiles {
        if t >= 1 {
            log.cmd(1, 20, tmem_empty, wait(u64::from((t - 1) & 1)));
        }
        for k in 0..kblocks {
            let i = t * kblocks + k;
            let (s, round) = (i % stages, i / stages);
            log.cmd(1, 21, full(s), wait(u64::from(round & 1)));
            log.cmd(1, 22, tcgen_work(1, 0), work(tcgen::WorkCmd::Issue));
            log.issue(1, 23, empty(s), 0, 1, vec![(tcgen_work(1, 0), work(tcgen::WorkCmd::Commit))]);
        }
        log.issue(1, 24, tmem_full, 0, 1, vec![(tcgen_work(1, 0), work(tcgen::WorkCmd::Commit))]);
    }
    for &w in &epilogue {
        for t in 0..tiles {
            log.cmd(w, 30, tmem_full, wait(u64::from(t & 1)));
            log.cmd(w, 31, tmem_empty, arrive(1));
        }
    }
    log.build()
}
