//! The fixed-program protocol transition system of one projection.
//!
//! State = per-warp cursor + per-warp pending retry (a registered blocking
//! command) + per-resource protocol state + pending completions + exit flag.
//! Witnesses and clocks are not part of the state, so equal protocol
//! situations reached by different orders merge.

use std::collections::HashMap;

use super::backend::{self, Res};
use super::explore::TransitionSystem;
use super::program::Program;
use super::projection::{ProjectionKey, ProjectionSpec};
use super::reference::ReferenceRun;
use crate::sync::{async_group, cluster, mbarrier, named, setmaxnreg, Outcome, ResourceId, ResourceInit, SyncCmd, SyncError};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Transition {
    /// Issue a head command (all participants at it).
    Issue(u32),
    /// Retry the registered blocking command of a local warp.
    Resume(u32),
    /// Land pending completion `(issuing command, ordinal)`.
    Complete(u32, u16),
    /// setmaxnreg pool grant `(resource, warpgroup)`.
    Grant(u32, u32),
    /// A blocking parity wait parks on local resource `r` (mbarrier
    /// `armed`). Keyed by resource, not warp: every parked waiter produces
    /// the same state, so per-warp arming would only break commutation.
    Arm(u32),
    Exit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PendingKind {
    Tx { gen: u64, bytes: u64 },
    /// Deferred arrive-on; `after` = async group `(resource, ordinal)` whose
    /// full completion releases it (`cp.async.mbarrier.arrive`).
    Arrive { gen: u64, count: u64, after: Option<(u32, u64)> },
    Milestone { ordinal: u64, milestone: async_group::Milestone },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Pending {
    pub cmd: u32,
    pub ord: u16,
    pub res: u32,
    pub kind: PendingKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct State {
    pub cursors: Box<[u32]>,
    pub retry: Box<[Option<(u32, SyncCmd)>]>,
    pub res: Box<[Res]>,
    pub pending: Box<[Pending]>,
    pub exited: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ErrKind {
    Protocol(SyncError),
    /// A situation the model does not cover (fail closed).
    Incomplete { reason: &'static str, detail: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TsError {
    /// Global command id.
    pub cmd: Option<usize>,
    pub kind: ErrKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Deadlock {
    pub domain: String,
    pub unfinished: Vec<usize>,
    pub blocked: Vec<usize>,
    pub pending_completions: usize,
    pub heads: Vec<String>,
}

/// Causal side effects of one step (consumed by the reference run).
#[derive(Clone, Debug, Default)]
pub struct Effects {
    /// `(global resource, gen)` the issuer's clock is released into.
    pub release: Vec<(usize, u64)>,
    /// `(global resource, gen)` the participants acquire.
    pub acquire: Vec<(usize, u64)>,
    /// Generation per protocol command (aligned with `Command::cmds`).
    pub gens: Vec<Option<u64>>,
    /// Captured generation per issued target (aligned with `Command::issued`).
    pub issued_gens: Vec<Option<u64>>,
    /// The command returned (cursor advanced) in this step.
    pub returned: bool,
}

#[derive(Clone, Debug)]
pub struct LocalCmd {
    pub global: usize,
    /// Local warps.
    pub participants: Vec<usize>,
    /// Program position of the command in each participant (aligned).
    pub positions: Vec<usize>,
    /// Protocol commands on local resources, including synthesized `Issue`s.
    pub cmds: Vec<(usize, SyncCmd)>,
    /// Index into the original `Command::cmds` (None = synthesized).
    pub origin: Vec<Option<usize>>,
    pub issued: Vec<(usize, u64, u64)>,
    pub conditional: bool,
    /// HB gate: `(local warp, count)` - that warp must have returned from its
    /// first `count` projection commands.
    pub gate: Box<[(u32, u32)]>,
}

pub struct Ts<'p> {
    pub program: &'p Program,
    pub key: ProjectionKey,
    pub warps: Vec<usize>,
    pub programs: Vec<Vec<usize>>,
    pub resources: Vec<usize>,
    pub cmds: Vec<LocalCmd>,
    initial_res: Vec<Res>,
    /// Commands per local resource (persistent-transition rule).
    resource_cmds: Vec<Vec<usize>>,
}

enum Tried {
    Disabled,
    /// A single-target blocking wait changed the resource (arming) without returning.
    Armed(State),
    Next(State, Effects),
    Error(TsError),
}

fn is_issue(cmd: &SyncCmd) -> bool {
    matches!(cmd, SyncCmd::Mbarrier(mbarrier::Cmd::Issue))
}

impl<'p> Ts<'p> {
    pub fn new(
        program: &'p Program,
        spec: &ProjectionSpec,
        init: &ResourceInit,
        reference: Option<&ReferenceRun>,
    ) -> Result<Self, String> {
        let mut in_projection = vec![false; program.commands.len()];
        for &c in &spec.commands {
            in_projection[c] = true;
        }
        let mut warp_local = HashMap::new();
        let mut warps = Vec::new();
        for (warp, list) in program.warp_programs.iter().enumerate() {
            if list.iter().any(|&c| in_projection[c]) {
                warp_local.insert(warp, warps.len());
                warps.push(warp);
            }
        }
        let mut resource_local = HashMap::new();
        let mut resources = Vec::new();
        let mut local_res = |r: usize| -> usize {
            *resource_local.entry(r).or_insert_with(|| {
                resources.push(r);
                resources.len() - 1
            })
        };
        let mut global_local = HashMap::new();
        let mut cmds = Vec::new();
        let mut sorted = spec.commands.clone();
        sorted.sort_unstable();
        for &g in &sorted {
            let c = &program.commands[g];
            let participants = c.participants.iter().map(|w| warp_local[w]).collect::<Vec<_>>();
            let mut local_cmds = c.cmds.iter().map(|&(r, cmd)| (local_res(r), cmd)).collect::<Vec<_>>();
            let mut origin = (0..c.cmds.len()).map(Some).collect::<Vec<_>>();
            let issued = c.issued.iter().map(|&(r, b, a)| (local_res(r), b, a)).collect::<Vec<_>>();
            // Each landing completion consumes one `Issue` token; synthesize
            // the ones the event did not list.
            let mut need = HashMap::<usize, usize>::new();
            for &(r, b, a) in &issued {
                *need.entry(r).or_default() += usize::from(b > 0) + usize::from(a > 0);
            }
            let mut need = need.into_iter().collect::<Vec<_>>();
            need.sort_unstable();
            for (r, n) in need {
                let have = local_cmds.iter().filter(|(lr, cmd)| *lr == r && is_issue(cmd)).count();
                for _ in have..n {
                    local_cmds.push((r, SyncCmd::Mbarrier(mbarrier::Cmd::Issue)));
                    origin.push(None);
                }
            }
            global_local.insert(g, cmds.len());
            cmds.push(LocalCmd {
                global: g,
                positions: Vec::new(),
                participants,
                cmds: local_cmds,
                origin,
                issued,
                conditional: c.conditional,
                gate: Box::new([]),
            });
        }
        let programs = warps
            .iter()
            .map(|&w| {
                program.warp_programs[w]
                    .iter()
                    .filter(|&&c| in_projection[c])
                    .map(|c| global_local[c])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for (local, cmd) in cmds.iter_mut().enumerate() {
            cmd.positions = cmd
                .participants
                .iter()
                .map(|&w| programs[w].iter().position(|&c| c == local).expect("participant holds command"))
                .collect();
        }
        let mut initial_res = Vec::new();
        for &r in &resources {
            let id: ResourceId = program.resources[r];
            initial_res.push(backend::fresh(id, init).ok_or_else(|| format!("resource {id:?} has no protocol model"))?);
        }
        let mut resource_cmds = vec![Vec::new(); resources.len()];
        for (i, c) in cmds.iter().enumerate() {
            let mut rs = c.cmds.iter().map(|(r, _)| *r).chain(c.issued.iter().map(|(r, _, _)| *r)).collect::<Vec<_>>();
            rs.sort_unstable();
            rs.dedup();
            for r in rs {
                resource_cmds[r].push(i);
            }
        }
        let mut ts = Ts { program, key: spec.key, warps, programs, resources, cmds, initial_res, resource_cmds };
        if spec.gated {
            if let Some(reference) = reference {
                ts.compute_gates(reference);
            }
        }
        Ok(ts)
    }

    fn compute_gates(&mut self, reference: &ReferenceRun) {
        let finals = self
            .programs
            .iter()
            .map(|list| list.iter().map(|&c| reference.final_[self.cmds[c].global].clone()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        for c in 0..self.cmds.len() {
            let Some(initial) = reference.initial[self.cmds[c].global].clone() else {
                continue;
            };
            let mut gate = Vec::new();
            for (w, list) in finals.iter().enumerate() {
                if self.cmds[c].participants.contains(&w) {
                    continue;
                }
                let n = list.partition_point(|f| f.as_ref().is_some_and(|f| f.leq(&initial)));
                if n > 0 {
                    gate.push((w as u32, n as u32));
                }
            }
            self.cmds[c].gate = gate.into_boxed_slice();
        }
    }

    pub fn initial(&self) -> State {
        State {
            cursors: vec![0; self.warps.len()].into_boxed_slice(),
            retry: vec![None; self.warps.len()].into_boxed_slice(),
            res: self.initial_res.clone().into_boxed_slice(),
            pending: Box::new([]),
            exited: false,
        }
    }

    pub fn head(&self, s: &State, w: usize) -> Option<usize> {
        self.programs[w].get(s.cursors[w] as usize).copied()
    }

    fn rid(&self, local: usize) -> ResourceId {
        self.program.resources[self.resources[local]]
    }

    fn err(&self, cmd: usize, e: SyncError) -> TsError {
        TsError { cmd: Some(self.cmds[cmd].global), kind: ErrKind::Protocol(e) }
    }

    fn all_finished(&self, s: &State) -> bool {
        s.cursors.iter().zip(&self.programs).all(|(c, p)| *c as usize == p.len())
    }

    fn gate_open(&self, s: &State, c: usize) -> bool {
        self.cmds[c].gate.iter().all(|&(w, n)| s.cursors[w as usize] >= n)
    }

    fn issue_ready(&self, s: &State, c: usize) -> bool {
        let lc = &self.cmds[c];
        lc.participants
            .iter()
            .all(|&w| s.retry[w].is_none() && self.head(s, w) == Some(c))
            && self.gate_open(s, c)
    }

    fn try_issue(&self, s: &State, c: usize) -> Tried {
        let lc = &self.cmds[c];
        let mut next = s.clone();
        let mut fx = Effects {
            gens: vec![None; self.program.commands[lc.global].cmds.len()],
            issued_gens: vec![None; lc.issued.len()],
            ..Effects::default()
        };
        let single = lc.cmds.len() == 1;
        let mut retry: Option<(u32, SyncCmd)> = None;
        let mut issued_gens = HashMap::<usize, Vec<u64>>::new();
        let mut arrive_on: Option<(u32, u64)> = None;
        let mut new_groups = Vec::new();
        for (i, &(r, cmd)) in lc.cmds.iter().enumerate() {
            let before = backend::async_next_ordinal(&next.res[r]);
            let mut st = next.res[r].clone();
            let out = match backend::step(&mut st, self.rid(r), cmd) {
                Ok(out) => out,
                Err(e) => return Tried::Error(self.err(c, e)),
            };
            if out.is_blocked() {
                // A single-target blocking wait may arm (mbarrier `armed`):
                // that is a state change; otherwise the command is disabled.
                if single && st != s.res[r] {
                    next.res[r] = st;
                    return Tried::Armed(next);
                }
                return Tried::Disabled;
            }
            if lc.conditional
                && matches!(out, Outcome::Mbarrier(mbarrier::Outcome::NotReady))
            {
                return Tried::Disabled;
            }
            next.res[r] = st;
            let gr = self.resources[r];
            let gen = match out {
                Outcome::Mbarrier(o) => match o {
                    mbarrier::Outcome::Arrived { gen, .. } => {
                        fx.release.push((gr, gen));
                        Some(gen)
                    }
                    mbarrier::Outcome::Updated { gen } => Some(gen),
                    mbarrier::Outcome::Issued { gen } => {
                        issued_gens.entry(r).or_default().push(gen);
                        Some(gen)
                    }
                    mbarrier::Outcome::Ready { gen: Some(gen) } => {
                        fx.acquire.push((gr, gen));
                        Some(gen)
                    }
                    _ => None,
                },
                Outcome::Named(o) => match o {
                    named::Outcome::Arrived { gen, .. } => {
                        fx.release.push((gr, gen));
                        Some(gen)
                    }
                    named::Outcome::Registered { gen } => {
                        fx.release.push((gr, gen));
                        retry = Some((r as u32, SyncCmd::Named(named::Cmd::Resume { gen })));
                        Some(gen)
                    }
                    named::Outcome::Ready { gen } => {
                        fx.release.push((gr, gen));
                        fx.acquire.push((gr, gen));
                        Some(gen)
                    }
                    named::Outcome::Blocked => None,
                },
                Outcome::Cluster(o) => match o {
                    cluster::Outcome::Arrived { gen, rearrival_without_wait, .. } => {
                        if rearrival_without_wait {
                            return Tried::Error(TsError {
                                cmd: Some(lc.global),
                                kind: ErrKind::Incomplete {
                                    reason: "cluster_barrier_rearrival_without_wait_unmodeled",
                                    detail: format!("warp re-arrives on {:?} generation {gen} without waiting", self.rid(r)),
                                },
                            });
                        }
                        fx.release.push((gr, gen));
                        Some(gen)
                    }
                    cluster::Outcome::Ready { gen } => {
                        fx.acquire.push((gr, gen));
                        Some(gen)
                    }
                    _ => None,
                },
                Outcome::RegPool(setmaxnreg::Outcome::Pending { .. }) => {
                    if let SyncCmd::RegPool(setmaxnreg::Cmd::Set { wg, .. }) = cmd {
                        retry = Some((r as u32, SyncCmd::RegPool(setmaxnreg::Cmd::Poll { wg })));
                    }
                    None
                }
                Outcome::AsyncGroup(async_group::Outcome::ArriveOn { group: Some(o) }) => {
                    arrive_on = Some((r as u32, o));
                    None
                }
                _ => None,
            };
            if let Some(o) = lc.origin[i] {
                fx.gens[o] = gen;
            }
            if let Some(from) = before {
                for ordinal in backend::async_new_groups(&next.res[r], from) {
                    new_groups.push((r, ordinal));
                }
            }
        }
        let mut pending = next.pending.to_vec();
        let mut ord = 0u16;
        for (r, ordinal) in new_groups {
            for milestone in [async_group::Milestone::ReadsDone, async_group::Milestone::FullyDone] {
                pending.push(Pending { cmd: c as u32, ord, res: r as u32, kind: PendingKind::Milestone { ordinal, milestone } });
                ord += 1;
            }
        }
        for (i, &(r, bytes, arrivals)) in lc.issued.iter().enumerate() {
            let gens = issued_gens.entry(r).or_default();
            let mut take = || if gens.is_empty() { None } else { Some(gens.remove(0)) };
            if bytes > 0 {
                let Some(gen) = take() else { return Tried::Error(self.internal(c, "issued target without token")) };
                fx.issued_gens[i] = Some(gen);
                pending.push(Pending { cmd: c as u32, ord, res: r as u32, kind: PendingKind::Tx { gen, bytes } });
                ord += 1;
            }
            if arrivals > 0 {
                let Some(gen) = take() else { return Tried::Error(self.internal(c, "issued target without token")) };
                fx.issued_gens[i] = Some(gen);
                pending.push(Pending { cmd: c as u32, ord, res: r as u32, kind: PendingKind::Arrive { gen, count: arrivals, after: arrive_on } });
                ord += 1;
            }
        }
        pending.sort_by_key(|p| (p.cmd, p.ord));
        next.pending = pending.into_boxed_slice();
        match retry {
            Some(rc) => {
                for &w in &lc.participants {
                    next.retry[w] = Some(rc);
                }
            }
            None => {
                for &w in &lc.participants {
                    next.cursors[w] += 1;
                }
                fx.returned = true;
            }
        }
        Tried::Next(next, fx)
    }

    fn internal(&self, c: usize, detail: &str) -> TsError {
        TsError {
            cmd: Some(self.cmds[c].global),
            kind: ErrKind::Incomplete { reason: "fixed_sync_program_model_incomplete", detail: detail.to_owned() },
        }
    }

    fn try_resume(&self, s: &State, w: usize) -> Tried {
        let Some((r, cmd)) = s.retry[w] else { return Tried::Disabled };
        let r = r as usize;
        let mut st = s.res[r].clone();
        let out = match backend::step(&mut st, self.rid(r), cmd) {
            Ok(out) => out,
            Err(e) => {
                let c = self.head(s, w).expect("retrying warp has a head");
                return Tried::Error(self.err(c, e));
            }
        };
        if out.is_blocked() {
            return Tried::Disabled;
        }
        let mut next = s.clone();
        next.res[r] = st;
        next.retry[w] = None;
        next.cursors[w] += 1;
        let mut fx = Effects { returned: true, ..Effects::default() };
        if let Outcome::Named(named::Outcome::Ready { gen }) = out {
            fx.acquire.push((self.resources[r], gen));
        }
        Tried::Next(next, fx)
    }

    fn pending_enabled(&self, s: &State, p: &Pending) -> bool {
        match p.kind {
            PendingKind::Arrive { after: Some((g, ordinal)), .. } => {
                !backend::async_group_pending(&s.res[g as usize], ordinal)
            }
            PendingKind::Milestone { .. } => matches!(self.try_complete(s, p), Tried::Next(..)),
            _ => true,
        }
    }

    fn try_complete(&self, s: &State, p: &Pending) -> Tried {
        let r = p.res as usize;
        let cmd = match p.kind {
            PendingKind::Tx { gen, bytes } => SyncCmd::Mbarrier(mbarrier::Cmd::CompleteTx { gen, bytes }),
            PendingKind::Arrive { gen, count, .. } => SyncCmd::Mbarrier(mbarrier::Cmd::DeferredArrive { gen, count }),
            PendingKind::Milestone { ordinal, milestone } => {
                SyncCmd::AsyncGroup(async_group::Cmd::Complete { ordinal, milestone })
            }
        };
        let mut st = s.res[r].clone();
        match backend::step(&mut st, self.rid(r), cmd) {
            Ok(_) => {
                let mut next = s.clone();
                next.res[r] = st;
                next.pending = s.pending.iter().filter(|q| (q.cmd, q.ord) != (p.cmd, p.ord)).copied().collect();
                let mut fx = Effects::default();
                if let PendingKind::Tx { gen, .. } | PendingKind::Arrive { gen, .. } = p.kind {
                    fx.release.push((self.resources[r], gen));
                }
                Tried::Next(next, fx)
            }
            Err(e) => {
                if matches!(p.kind, PendingKind::Milestone { .. }) {
                    return Tried::Disabled;
                }
                Tried::Error(self.err(p.cmd as usize, e))
            }
        }
    }

    pub fn enabled(&self, s: &State) -> Vec<Transition> {
        if s.exited {
            return Vec::new();
        }
        let mut out = Vec::new();
        for w in 0..self.warps.len() {
            if s.retry[w].is_some() {
                if !matches!(self.try_resume(s, w), Tried::Disabled) {
                    out.push(Transition::Resume(w as u32));
                }
                continue;
            }
            let Some(c) = self.head(s, w) else { continue };
            if self.cmds[c].participants.iter().min() != Some(&w) {
                continue;
            }
            if self.issue_ready(s, c) {
                match self.try_issue(s, c) {
                    Tried::Disabled => {}
                    Tried::Armed(_) => {
                        let arm = Transition::Arm(self.cmds[c].cmds[0].0 as u32);
                        if !out.contains(&arm) {
                            out.push(arm);
                        }
                    }
                    _ => out.push(Transition::Issue(c as u32)),
                }
            }
        }
        for p in s.pending.iter() {
            if self.pending_enabled(s, p) {
                out.push(Transition::Complete(p.cmd, p.ord));
            }
        }
        for (r, res) in s.res.iter().enumerate() {
            for wg in backend::enabled_grants(res) {
                out.push(Transition::Grant(r as u32, wg));
            }
        }
        if out.is_empty() && self.all_finished(s) && s.pending.is_empty() && s.retry.iter().all(Option::is_none) {
            out.push(Transition::Exit);
        }
        out
    }

    pub fn step_fx(&self, s: &State, t: &Transition) -> Result<(State, Effects), TsError> {
        let tried = match *t {
            Transition::Issue(c) => {
                if !self.issue_ready(s, c as usize) {
                    Tried::Disabled
                } else {
                    self.try_issue(s, c as usize)
                }
            }
            Transition::Resume(w) => self.try_resume(s, w as usize),
            Transition::Arm(r) => (0..self.warps.len())
                .filter(|&w| s.retry[w].is_none())
                .filter_map(|w| self.head(s, w))
                .filter(|&c| self.cmds[c].cmds.len() == 1 && self.cmds[c].cmds[0].0 == r as usize && self.issue_ready(s, c))
                .find_map(|c| match self.try_issue(s, c) {
                    Tried::Armed(next) => Some(Tried::Next(next, Effects::default())),
                    _ => None,
                })
                .unwrap_or(Tried::Disabled),
            Transition::Complete(c, o) => match s.pending.iter().find(|p| (p.cmd, p.ord) == (c, o)) {
                Some(p) if self.pending_enabled(s, p) => self.try_complete(s, p),
                _ => Tried::Disabled,
            },
            Transition::Grant(r, wg) => {
                let mut st = s.res[r as usize].clone();
                match backend::step(&mut st, self.rid(r as usize), SyncCmd::RegPool(setmaxnreg::Cmd::Grant { wg })) {
                    Ok(_) => {
                        let mut next = s.clone();
                        next.res[r as usize] = st;
                        Tried::Next(next, Effects::default())
                    }
                    Err(e) => Tried::Error(TsError { cmd: None, kind: ErrKind::Protocol(e) }),
                }
            }
            Transition::Exit => {
                for (r, res) in s.res.iter().enumerate() {
                    if let Err(e) = backend::quiescent(res) {
                        let _ = r;
                        return Err(TsError { cmd: None, kind: ErrKind::Protocol(e) });
                    }
                }
                let mut next = s.clone();
                next.exited = true;
                Tried::Next(next, Effects::default())
            }
        };
        match tried {
            Tried::Next(next, fx) => Ok((next, fx)),
            Tried::Armed(_) => Err(TsError {
                cmd: None,
                kind: ErrKind::Incomplete { reason: "internal", detail: format!("arming through {t:?}") },
            }),
            Tried::Error(e) => Err(e),
            Tried::Disabled => Err(TsError {
                cmd: None,
                kind: ErrKind::Incomplete { reason: "internal", detail: format!("disabled transition {t:?}") },
            }),
        }
    }

    pub fn describe_deadlock(&self, s: &State) -> Deadlock {
        let mut unfinished = Vec::new();
        let mut blocked = Vec::new();
        let mut heads = Vec::new();
        for w in 0..self.warps.len() {
            let Some(c) = self.head(s, w) else { continue };
            let warp = self.program.warp_ids[self.warps[w]].0 as usize;
            unfinished.push(warp);
            if s.retry[w].is_some() {
                blocked.push(warp);
            }
            let lc = &self.cmds[c];
            let g = &self.program.commands[lc.global];
            let unmet = lc
                .gate
                .iter()
                .filter(|&&(gw, n)| s.cursors[gw as usize] < n)
                .map(|&(gw, n)| format!("warp {} must pass {n}", self.program.warp_ids[self.warps[gw as usize]].0))
                .collect::<Vec<_>>();
            heads.push(format!(
                "warp {warp}: head #{} site {} {:?}, retry={:?}, unmet causal predecessors={unmet:?}",
                g.seq, g.site.0, g.cmds.iter().map(|(_, c)| c).collect::<Vec<_>>(), s.retry[w]
            ));
        }
        Deadlock { domain: format!("{:?}", self.key), unfinished, blocked, pending_completions: s.pending.len(), heads }
    }

    /// Human-readable transition at `s` (witness evidence).
    pub fn describe(&self, s: &State, t: &Transition) -> (String, Option<usize>) {
        match *t {
            Transition::Issue(c) => {
                let g = self.cmds[c as usize].global;
                let cmd = &self.program.commands[g];
                (format!("warp {} issues #{} {:?}", cmd.warp.0, cmd.seq, cmd.cmds.iter().map(|(_, c)| c).collect::<Vec<_>>()), Some(g))
            }
            Transition::Resume(w) => {
                let c = self.head(s, w as usize).map(|c| self.cmds[c].global);
                (format!("warp {} resumes {:?}", self.program.warp_ids[self.warps[w as usize]].0, s.retry[w as usize].map(|r| r.1)), c)
            }
            Transition::Complete(c, o) => {
                let g = self.cmds[c as usize].global;
                let kind = s.pending.iter().find(|p| (p.cmd, p.ord) == (c, o)).map(|p| p.kind);
                (format!("completion {kind:?} issued by warp {} #{} lands", self.program.commands[g].warp.0, self.program.commands[g].seq), Some(g))
            }
            Transition::Grant(r, wg) => (format!("setmaxnreg grant for warpgroup {wg} on {:?}", self.rid(r as usize)), None),
            Transition::Arm(r) => (format!("a blocking wait parks on {:?}", self.rid(r as usize)), None),
            Transition::Exit => ("validate exit".to_owned(), None),
        }
    }
}

impl TransitionSystem for Ts<'_> {
    type State = State;
    type Transition = Transition;
    type Error = TsError;
    type Deadlock = Deadlock;

    fn initial_state(&self) -> State {
        self.initial()
    }
    fn enabled(&self, s: &State) -> Vec<Transition> {
        Ts::enabled(self, s)
    }
    fn step(&self, s: &State, t: &Transition) -> Result<State, TsError> {
        self.step_fx(s, t).map(|(n, _)| n)
    }
    fn is_complete(&self, s: &State) -> bool {
        s.exited
    }
    fn describe_deadlock(&self, s: &State) -> Deadlock {
        Ts::describe_deadlock(self, s)
    }

    /// Today's terminal-completion rule (`sync_fixed_unified.rs:5304-5380`):
    /// a pending transaction completion may run first when no other pending
    /// completion targets its barrier on another generation and no
    /// not-yet-issued command of the projection can observe the barrier
    /// except parity waits/tests that this completion makes ready.
    fn persistent_transition(&self, s: &State, enabled: &[Transition]) -> Option<Transition> {
        enabled.iter().copied().find(|t| {
            let Transition::Complete(c, o) = *t else { return false };
            let Some(p) = s.pending.iter().find(|p| (p.cmd, p.ord) == (c, o)) else { return false };
            let PendingKind::Tx { gen, .. } = p.kind else { return false };
            if s.pending.iter().any(|q| q.res == p.res && !matches!(q.kind, PendingKind::Tx { gen: g, .. } if g == gen)) {
                return false;
            }
            let parity = gen & 1;
            let conflicts = self.resource_cmds[p.res as usize].iter().any(|&lc| {
                let cmd = &self.cmds[lc];
                let issued = cmd.participants.iter().zip(&cmd.positions).all(|(&w, &pos)| s.cursors[w] as usize > pos);
                if issued {
                    return false;
                }
                cmd.issued.iter().any(|(r, _, _)| *r == p.res as usize)
                    || cmd.cmds.iter().any(|&(r, ref k)| {
                        r == p.res as usize
                            && !matches!(k, SyncCmd::Mbarrier(mbarrier::Cmd::WaitParity { parity: q } | mbarrier::Cmd::TestParity { parity: q }) if *q == parity)
                    })
            });
            !conflicts && self.step(s, t).is_ok()
        })
    }
}
