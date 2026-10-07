//! The fixed-program protocol transition system for one projection.
//!
//! State = per-warp cursor + blocked flag + per-resource protocol state +
//! pending async completions (sorted by issuer) + exit flag. Witness history
//! and scheduler bookkeeping are *not* part of the state, so equal states
//! reached by different orders merge in the visited set (today's
//! `FixedSyncState` equality, `sync_fixed_unified.rs:300-335`).

use std::collections::HashMap;

use crate::event::{OpId, ResourceId, SyncOp, WarpId};
use crate::explore::TransitionSystem;
use crate::program::{Program, ReferenceRun};
use crate::projection::{ProjectionKey, ProjectionSpec};
use crate::protocol::{ClusterState, MbarState, NamedState, ProtoErr, ResState};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Transition {
    /// Issue the head command of a projection-local warp.
    Issue(u32),
    /// Deliver the async completion issued by a projection-local command.
    Complete(u32),
    /// Validate quiescent exit state.
    Exit,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Pending {
    pub issuer: u32,
    pub resource: u32,
    pub generation: u32,
    pub tx: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct State {
    pub cursors: Box<[u32]>,
    pub blocked: Box<[bool]>,
    pub resources: Box<[ResState]>,
    pub pending: Box<[Pending]>,
    pub exited: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TsError {
    pub op: Option<OpId>,
    pub error: ProtoErr,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Deadlock {
    pub domain: String,
    pub unfinished_warps: Vec<WarpId>,
    pub blocked_warps: Vec<WarpId>,
    pub pending_completions: usize,
    pub unready_heads: Vec<String>,
}

/// Projection-local command.
#[derive(Clone, Debug)]
pub struct LocalCmd {
    pub global: usize,
    pub op: OpId,
    pub kind: SyncOp,
    pub warp: usize,
    pub resource: usize,
    /// Happens-before gate: `(local warp, count)` means that warp's cursor must
    /// have passed its first `count` projection commands before this command
    /// may issue (today's `causal_predecessors`, compressed per warp).
    pub gate: Box<[(u32, u32)]>,
}

/// Side effects of one step, consumed by the reference run's clock tracking.
#[derive(Clone, Debug, Default)]
pub struct Effects {
    pub generation: Option<u32>,
    /// The issuing command's clock joins the release payload of `generation`.
    pub release: bool,
    /// The issuing warp acquires the release payload of this generation.
    pub acquire: Option<u32>,
    /// `(local warp, local cmd)` released from a blocking named sync.
    pub released: Vec<(usize, usize)>,
    /// The issuing warp is left blocked.
    pub blocked: bool,
}

pub struct ProtocolTs<'p> {
    pub program: &'p Program,
    pub key: ProjectionKey,
    /// Local warp -> dense global warp index.
    pub warps: Vec<usize>,
    /// Local warp -> local command ids in program order.
    pub programs: Vec<Vec<usize>>,
    /// Local resource -> dense global resource index.
    pub resources: Vec<usize>,
    pub cmds: Vec<LocalCmd>,
    /// Commands per local resource, used by the persistent-transition rule.
    resource_cmds: Vec<Vec<usize>>,
    /// Program position of each local command within its warp.
    pub positions: Vec<usize>,
}

impl<'p> ProtocolTs<'p> {
    pub fn new(program: &'p Program, spec: &ProjectionSpec, reference: Option<&ReferenceRun>) -> Self {
        let mut in_projection = vec![false; program.commands.len()];
        for &command in &spec.commands {
            in_projection[command] = true;
        }
        let mut warps = Vec::new();
        let mut programs = Vec::new();
        let mut warp_local = HashMap::new();
        let mut resource_local = HashMap::new();
        let mut resources = Vec::new();
        let mut cmds = Vec::new();
        let mut positions = Vec::new();
        for (warp, warp_program) in program.warp_programs.iter().enumerate() {
            let local_program = warp_program
                .iter()
                .copied()
                .filter(|command| in_projection[*command])
                .collect::<Vec<_>>();
            if local_program.is_empty() {
                continue;
            }
            let local_warp = warps.len();
            warp_local.insert(warp, local_warp);
            warps.push(warp);
            let mut ids = Vec::with_capacity(local_program.len());
            for (position, global) in local_program.into_iter().enumerate() {
                let command = &program.commands[global];
                let resource = *resource_local.entry(command.resource).or_insert_with(|| {
                    resources.push(command.resource);
                    resources.len() - 1
                });
                ids.push(cmds.len());
                positions.push(position);
                cmds.push(LocalCmd {
                    global,
                    op: command.op,
                    kind: command.kind,
                    warp: local_warp,
                    resource,
                    gate: Box::new([]),
                });
            }
            programs.push(ids);
        }
        if spec.gated {
            if let Some(reference) = reference {
                compute_gates(&mut cmds, &programs, reference);
            }
        }
        let mut resource_cmds = vec![Vec::new(); resources.len()];
        for (index, cmd) in cmds.iter().enumerate() {
            resource_cmds[cmd.resource].push(index);
        }
        Self {
            program,
            key: spec.key,
            warps,
            programs,
            resources,
            cmds,
            resource_cmds,
            positions,
        }
    }

    pub fn resource_id(&self, local: usize) -> ResourceId {
        self.program.resources[self.resources[local]]
    }

    pub fn initial(&self) -> State {
        let resources = self
            .resources
            .iter()
            .map(|&global| match self.program.resources[global] {
                ResourceId::Mbarrier(_) => ResState::Mbar(MbarState::default()),
                ResourceId::Named(_) => ResState::Named(NamedState::default()),
                ResourceId::Cluster(_) => ResState::Cluster(ClusterState::with_warps(self.warps.len())),
            })
            .collect();
        State {
            cursors: vec![0; self.warps.len()].into_boxed_slice(),
            blocked: vec![false; self.warps.len()].into_boxed_slice(),
            resources,
            pending: Box::new([]),
            exited: false,
        }
    }

    pub fn head(&self, state: &State, warp: usize) -> Option<usize> {
        self.programs[warp].get(state.cursors[warp] as usize).copied()
    }

    fn all_finished(&self, state: &State) -> bool {
        state
            .cursors
            .iter()
            .zip(&self.programs)
            .all(|(cursor, program)| *cursor as usize == program.len())
    }

    fn gate_open(&self, state: &State, cmd: usize) -> bool {
        self.cmds[cmd]
            .gate
            .iter()
            .all(|&(warp, count)| state.cursors[warp as usize] >= count)
    }

    fn issue_enabled(&self, state: &State, cmd: usize) -> bool {
        let local = &self.cmds[cmd];
        match (&local.kind, &state.resources[local.resource]) {
            (SyncOp::MbarWait { parity, .. }, ResState::Mbar(mbar)) => mbar.wait_enabled(*parity),
            (SyncOp::ClusterWait { .. }, ResState::Cluster(cluster)) => {
                cluster.wait_enabled(local.warp)
            }
            _ => true,
        }
    }

    pub fn enabled(&self, state: &State) -> Vec<Transition> {
        if state.exited {
            return Vec::new();
        }
        let mut enabled = Vec::new();
        for warp in 0..self.warps.len() {
            if state.blocked[warp] {
                continue;
            }
            let Some(cmd) = self.head(state, warp) else {
                continue;
            };
            if self.gate_open(state, cmd) && self.issue_enabled(state, cmd) {
                enabled.push(Transition::Issue(warp as u32));
            }
        }
        enabled.extend(
            state
                .pending
                .iter()
                .map(|pending| Transition::Complete(pending.issuer)),
        );
        if enabled.is_empty() && self.all_finished(state) && state.pending.is_empty() {
            enabled.push(Transition::Exit);
        }
        enabled
    }

    pub fn step(&self, state: &State, transition: &Transition) -> Result<State, TsError> {
        self.step_with_effects(state, transition).map(|(next, _)| next)
    }

    pub fn step_with_effects(
        &self,
        state: &State,
        transition: &Transition,
    ) -> Result<(State, Effects), TsError> {
        let mut next = state.clone();
        let mut effects = Effects::default();
        match *transition {
            Transition::Issue(warp) => {
                let warp = warp as usize;
                let cmd = self.head(state, warp).ok_or_else(|| self.internal("finished warp issued"))?;
                let local = &self.cmds[cmd];
                let fail = |error: ProtoErr| TsError {
                    op: Some(local.op),
                    error,
                };
                let resource = &mut next.resources[local.resource];
                let mut advance = true;
                match (local.kind, resource) {
                    (SyncOp::MbarInit { expected, .. }, ResState::Mbar(mbar)) => {
                        mbar.init(expected).map_err(fail)?;
                    }
                    (SyncOp::MbarArrive { count, expect_tx, .. }, ResState::Mbar(mbar)) => {
                        let (generation, _) = mbar.arrive(count, expect_tx).map_err(fail)?;
                        effects.generation = Some(generation);
                        effects.release = true;
                    }
                    (SyncOp::MbarExpectTx { tx, .. }, ResState::Mbar(mbar)) => {
                        effects.generation = Some(mbar.expect_tx(tx).map_err(fail)?);
                    }
                    (SyncOp::MbarTxIssue { tx, .. }, ResState::Mbar(mbar)) => {
                        let generation = mbar.capture().map_err(fail)?;
                        effects.generation = Some(generation);
                        let mut pending = next.pending.to_vec();
                        pending.push(Pending {
                            issuer: cmd as u32,
                            resource: local.resource as u32,
                            generation,
                            tx,
                        });
                        pending.sort_by_key(|pending| pending.issuer);
                        next.pending = pending.into_boxed_slice();
                    }
                    (SyncOp::MbarWait { parity, .. }, ResState::Mbar(mbar)) => {
                        let consumed = mbar.wait(parity).map_err(fail)?;
                        effects.generation = consumed;
                        effects.acquire = consumed;
                    }
                    (SyncOp::NamedArrive { expected, count, .. }, ResState::Named(named)) => {
                        let (generation, completed) = named.contribute(expected, count).map_err(fail)?;
                        effects.generation = Some(generation);
                        effects.release = true;
                        if completed {
                            self.release_named(&mut next, local.resource, &mut effects);
                        }
                    }
                    (SyncOp::NamedSync { expected, count, .. }, ResState::Named(named)) => {
                        let (generation, completed) = named.contribute(expected, count).map_err(fail)?;
                        effects.generation = Some(generation);
                        effects.release = true;
                        if completed {
                            effects.released.push((warp, cmd));
                            self.release_named(&mut next, local.resource, &mut effects);
                        } else {
                            next.blocked[warp] = true;
                            effects.blocked = true;
                            advance = false;
                        }
                    }
                    (SyncOp::ClusterArrive { participants, .. }, ResState::Cluster(cluster)) => {
                        let (generation, _) = cluster.arrive(warp, participants).map_err(fail)?;
                        effects.generation = Some(generation);
                        effects.release = true;
                    }
                    (SyncOp::ClusterWait { .. }, ResState::Cluster(cluster)) => {
                        let generation = cluster.wait(warp).map_err(fail)?;
                        effects.generation = Some(generation);
                        effects.acquire = Some(generation);
                    }
                    _ => return Err(self.internal("command and resource kinds disagree")),
                }
                if advance {
                    next.cursors[warp] += 1;
                }
            }
            Transition::Complete(issuer) => {
                let index = next
                    .pending
                    .iter()
                    .position(|pending| pending.issuer == issuer)
                    .ok_or_else(|| self.internal("completion is not pending"))?;
                let mut pending = next.pending.to_vec();
                let completion = pending.remove(index);
                next.pending = pending.into_boxed_slice();
                let ResState::Mbar(mbar) = &mut next.resources[completion.resource as usize] else {
                    return Err(self.internal("completion targets a non-mbarrier"));
                };
                mbar.complete_tx(completion.generation, completion.tx)
                    .map_err(|error| TsError {
                        op: Some(self.cmds[issuer as usize].op),
                        error,
                    })?;
                effects.generation = Some(completion.generation);
                effects.release = true;
            }
            Transition::Exit => {
                for resource in next.resources.iter() {
                    let result = match resource {
                        ResState::Mbar(_) => Ok(()),
                        ResState::Named(named) => named.exit_check(),
                        ResState::Cluster(cluster) => cluster.exit_check(),
                    };
                    result.map_err(|error| TsError { op: None, error })?;
                }
                next.exited = true;
            }
        }
        Ok((next, effects))
    }

    fn release_named(&self, next: &mut State, resource: usize, effects: &mut Effects) {
        for warp in 0..self.warps.len() {
            if !next.blocked[warp] {
                continue;
            }
            let Some(cmd) = self.head(next, warp) else {
                continue;
            };
            if self.cmds[cmd].resource == resource
                && matches!(self.cmds[cmd].kind, SyncOp::NamedSync { .. })
            {
                next.blocked[warp] = false;
                next.cursors[warp] += 1;
                effects.released.push((warp, cmd));
            }
        }
    }

    fn internal(&self, detail: &str) -> TsError {
        TsError {
            op: None,
            error: ProtoErr {
                kind: "internal",
                detail: detail.to_owned(),
                incomplete: true,
            },
        }
    }

    pub fn is_complete(&self, state: &State) -> bool {
        state.exited
    }

    pub fn describe_deadlock(&self, state: &State) -> Deadlock {
        let mut unfinished_warps = Vec::new();
        let mut blocked_warps = Vec::new();
        let mut unready_heads = Vec::new();
        for warp in 0..self.warps.len() {
            let warp_id = self.program.warp_ids[self.warps[warp]];
            let Some(cmd) = self.head(state, warp) else {
                continue;
            };
            unfinished_warps.push(warp_id);
            if state.blocked[warp] {
                blocked_warps.push(warp_id);
            }
            let local = &self.cmds[cmd];
            let unmet = local
                .gate
                .iter()
                .filter(|&&(gate_warp, count)| state.cursors[gate_warp as usize] < count)
                .map(|&(gate_warp, count)| {
                    format!("warp {} must pass {count}", self.program.warp_ids[self.warps[gate_warp as usize]])
                })
                .collect::<Vec<_>>();
            unready_heads.push(format!(
                "warp {warp_id}: head {} {:?} on {:?}, blocked={}, unmet causal predecessors={unmet:?}, resource={:?}",
                local.kind.name(),
                local.op,
                self.resource_id(local.resource),
                state.blocked[warp],
                state.resources[local.resource],
            ));
        }
        Deadlock {
            domain: format!("{:?}", self.key),
            unfinished_warps,
            blocked_warps,
            pending_completions: state.pending.len(),
            unready_heads,
        }
    }

    /// Human-readable description of a transition at `state` (for witnesses).
    pub fn describe(&self, state: &State, transition: &Transition) -> String {
        match *transition {
            Transition::Issue(warp) => match self.head(state, warp as usize) {
                Some(cmd) => {
                    let local = &self.cmds[cmd];
                    format!(
                        "warp {} issues {} #{} on {:?}",
                        local.op.warp,
                        local.kind.name(),
                        local.op.seq,
                        self.resource_id(local.resource)
                    )
                }
                None => format!("{transition:?}"),
            },
            Transition::Complete(issuer) => {
                let local = &self.cmds[issuer as usize];
                format!("completion of warp {} #{} lands", local.op.warp, local.op.seq)
            }
            Transition::Exit => "validate exit".to_owned(),
        }
    }
}

fn compute_gates(cmds: &mut [LocalCmd], programs: &[Vec<usize>], reference: &ReferenceRun) {
    // Per local warp: final clocks of its projection commands, in order.
    let finals = programs
        .iter()
        .map(|program| {
            program
                .iter()
                .map(|&cmd| reference.final_[cmds[cmd].global].clone())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    for cmd in cmds.iter_mut() {
        let Some(initial) = reference.initial[cmd.global].clone() else {
            continue;
        };
        let own = cmd.warp;
        let mut gate = Vec::new();
        for (warp, warp_finals) in finals.iter().enumerate() {
            if warp == own {
                continue;
            }
            // Final clocks are monotone along a warp, so `final <= initial`
            // holds on a prefix.
            let count = warp_finals.partition_point(|clock| {
                clock.as_ref().is_some_and(|clock| clock.leq(&initial))
            });
            if count > 0 {
                gate.push((warp as u32, count as u32));
            }
        }
        cmd.gate = gate.into_boxed_slice();
    }
}

impl TransitionSystem for ProtocolTs<'_> {
    type State = State;
    type Transition = Transition;
    type Error = TsError;
    type Deadlock = Deadlock;

    fn initial_state(&self) -> State {
        self.initial()
    }

    fn enabled(&self, state: &State) -> Vec<Transition> {
        ProtocolTs::enabled(self, state)
    }

    fn step(&self, state: &State, transition: &Transition) -> Result<State, TsError> {
        ProtocolTs::step(self, state, transition)
    }

    fn is_complete(&self, state: &State) -> bool {
        ProtocolTs::is_complete(self, state)
    }

    fn describe_deadlock(&self, state: &State) -> Deadlock {
        ProtocolTs::describe_deadlock(self, state)
    }

    /// Today's terminal-completion rule (`sync_fixed_unified.rs:5304-5380`):
    /// a pending transaction completion may run first when no not-yet-issued
    /// command of the projection can observe its barrier in a way that
    /// distinguishes the order (no future mutation, no wait for the other
    /// parity) and no other pending completion targets another generation.
    fn persistent_transition(&self, state: &State, enabled: &[Transition]) -> Option<Transition> {
        enabled.iter().find_map(|transition| {
            let Transition::Complete(issuer) = transition else {
                return None;
            };
            let pending = state.pending.iter().find(|p| p.issuer == *issuer)?;
            let resource = pending.resource as usize;
            if state
                .pending
                .iter()
                .any(|other| other.resource == pending.resource && other.generation != pending.generation)
            {
                return None;
            }
            let parity = (pending.generation & 1) as u8;
            let conflicts = self.resource_cmds[resource].iter().any(|&cmd| {
                let local = &self.cmds[cmd];
                let issued = state.cursors[local.warp] as usize > self.positions[cmd];
                if issued {
                    return false;
                }
                !matches!(local.kind, SyncOp::MbarWait { parity: wait_parity, .. } if wait_parity == parity)
            });
            (!conflicts && self.step(state, transition).is_ok()).then(|| transition.clone())
        })
    }
}
