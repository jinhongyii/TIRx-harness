//! Per-warp command sequences from the contract `SyncEvent` log (Phase B
//! input) and the Phase A failures the engine already surfaced.

use std::collections::{BTreeMap, HashMap};

use crate::observe::{Actor, Collective, LoopFrame, ProtocolStatus, RecordingObserver, SyncEvent, SyncKind, WarpId};
use crate::site::SiteId;
use crate::sync::{mbarrier, named, setmaxnreg, tcgen, ResourceId, SyncCmd, SyncError};

/// One committed instruction (or one collective rendezvous) in the fixed program.
#[derive(Clone, Debug)]
pub struct Command {
    /// Dense warp indices; `participants[0]` is the reporting warp.
    pub participants: Vec<usize>,
    pub warp: WarpId,
    pub seq: u32,
    pub epoch: u64,
    pub site: SiteId,
    pub frames: Vec<LoopFrame>,
    /// Protocol commands on dense resource indices, applied all-or-nothing.
    pub cmds: Vec<(usize, SyncCmd)>,
    /// Async completions promised at issue: `(resource, bytes, arrivals)`.
    pub issued: Vec<(usize, u64, u64)>,
    /// A recorded successful `test_wait`/`try_wait`: only schedules where it
    /// succeeds belong to this fixed program.
    pub conditional: bool,
    /// `.cta_group` values this instruction uses (kernel-wide rule).
    pub tcgen_groups: Vec<u8>,
    /// A `tcgen05.commit`: its deferred arrivals land in issue order with the
    /// issuing warp's other commits (tcgen05 pipeline order).
    pub commit: bool,
}

#[derive(Clone, Debug)]
pub struct Program {
    /// Kernel index within the `Module` (from `SyncEvent::kernel`).
    pub kernel: u32,
    pub warp_ids: Vec<WarpId>,
    pub warp_programs: Vec<Vec<usize>>,
    pub commands: Vec<Command>,
    pub resources: Vec<ResourceId>,
    /// Commands applied to the initial resource states before any warp runs:
    /// host-side protocol events (e.g. the launch-bounds setmaxnreg
    /// `Configure`) and every `Configure` wherever it was logged.
    pub init_cmds: Vec<(usize, SyncCmd)>,
    /// Largest `.exclusive` tcgen05.alloc (PTX Table 58: 512 columns, 576 on
    /// sm_107f). The recording has no arch, but an alloc's width is a static
    /// fact of its command that the engine already validated against the
    /// launch's arch (a rejected one is a Phase A failure). So the default is
    /// the largest width the run committed, and at least 512.
    /// `SynccheckConfig::tcgen_exclusive_max` overrides it (W2-18).
    pub tcgen_exclusive_max: u32,
}

/// A failure the engine already hit in the concrete run (Phase A).
#[derive(Clone, Debug)]
pub struct PhaseAFailure {
    pub event: SyncEvent,
    /// `None` = `BlockedAtExit` (the concrete run deadlocked).
    pub error: Option<SyncError>,
}

/// Commands the explorer re-derives itself and therefore drops from the log:
/// named `Resume` and setmaxnreg `Poll` (retries of a registered blocking
/// command), and failed parity polls (no state change).
fn keep(cmd: &SyncCmd, observed: Option<u8>) -> bool {
    !matches!(
        cmd,
        SyncCmd::Named(named::Cmd::Resume { .. })
            | SyncCmd::RegPool(setmaxnreg::Cmd::Poll { .. })
    ) && !(matches!(cmd, SyncCmd::Mbarrier(mbarrier::Cmd::TestParity { .. })) && observed.is_none())
}

pub fn build(log: &RecordingObserver) -> Result<(Program, Vec<PhaseAFailure>), String> {
    let mut failures = Vec::new();
    let mut resource_index = HashMap::<ResourceId, usize>::new();
    let mut resources = Vec::<ResourceId>::new();
    let mut intern = |id: ResourceId| -> usize {
        *resource_index.entry(id).or_insert_with(|| {
            resources.push(id);
            resources.len() - 1
        })
    };
    // Dense warps: every warp with at least one kept protocol event.
    struct Raw {
        warp: WarpId,
        seq: u32,
        epoch: u64,
        site: SiteId,
        frames: Vec<LoopFrame>,
        cmds: Vec<(usize, SyncCmd)>,
        issued: Vec<(usize, u64, u64)>,
        conditional: bool,
        collective: Option<Collective>,
        tcgen_groups: Vec<u8>,
        commit: bool,
    }
    let mut kernel = None::<u32>;
    let mut raws = Vec::<Raw>::new();
    let mut init_cmds = Vec::<(usize, SyncCmd)>::new();
    for event in &log.other {
        if let SyncKind::Protocol { cmds, status: ProtocolStatus::Committed, .. } = &event.kind {
            kernel.get_or_insert(event.kernel);
            init_cmds.extend(cmds.iter().map(|pc| (intern(pc.res), pc.cmd)));
        }
    }
    for events in &log.per_warp {
        for event in events {
            let SyncKind::Protocol { cmds, collective, issued, status } = &event.kind else {
                continue;
            };
            let Actor::Warp { warp, epoch } = event.actor else {
                continue;
            };
            match status {
                ProtocolStatus::Committed => {}
                ProtocolStatus::Failed(error) => {
                    failures.push(PhaseAFailure { event: event.clone(), error: Some(error.clone()) });
                    continue;
                }
                ProtocolStatus::BlockedAtExit => {
                    failures.push(PhaseAFailure { event: event.clone(), error: None });
                    continue;
                }
            }
            kernel.get_or_insert(event.kernel);
            // The kernel-wide `.cta_group` rule is order-independent (any
            // two distinct groups fail whichever comes first), so it is
            // checked once over the program instead of explored.
            let mut tcgen_groups = Vec::new();
            for pc in cmds {
                match pc.cmd {
                    SyncCmd::TcgenGroup(g) => tcgen_groups.push(g),
                    SyncCmd::Tcgen(tcgen::Cmd::Alloc { who, .. } | tcgen::Cmd::Dealloc { who, .. } | tcgen::Cmd::Relinquish { who }) => {
                        tcgen_groups.push(who.group())
                    }
                    _ => {}
                }
            }
            let conditional = cmds.iter().any(|pc| {
                pc.observed_parity.is_some() || matches!(pc.cmd, SyncCmd::Mbarrier(mbarrier::Cmd::TestState { .. }))
            });
            // The launch-bounds register budget precedes every warp; hoist it
            // wherever it was logged (a per-warp copy would race with `Set`).
            for pc in cmds {
                if let SyncCmd::RegPool(setmaxnreg::Cmd::Configure { .. }) = pc.cmd {
                    let r = intern(pc.res);
                    if !init_cmds.contains(&(r, pc.cmd)) {
                        init_cmds.push((r, pc.cmd));
                    }
                }
            }
            let kept = cmds
                .iter()
                .filter(|pc| !matches!(pc.cmd, SyncCmd::RegPool(setmaxnreg::Cmd::Configure { .. })))
                // `TcgenGroup` is checked statically; `TcgenWork` commands are
                // total and never block, so they carry no protocol state the
                // search needs (and would couple every commit's barriers).
                .filter(|pc| {
                    keep(&pc.cmd, pc.observed_parity) && !matches!(pc.cmd, SyncCmd::TcgenGroup(_) | SyncCmd::TcgenWork(_))
                })
                .map(|pc| (intern(pc.res), pc.cmd))
                .collect::<Vec<_>>();
            let issued = issued
                .iter()
                .map(|t| (intern(t.res), t.bytes, t.arrivals))
                .collect::<Vec<_>>();
            let commit = cmds.iter().any(|pc| matches!(pc.cmd, SyncCmd::TcgenWork(tcgen::WorkCmd::Commit)));
            if kept.is_empty() && issued.is_empty() && tcgen_groups.is_empty() {
                continue;
            }
            raws.push(Raw {
                warp,
                seq: event.seq,
                epoch,
                site: event.site,
                frames: event.frames.clone(),
                cmds: kept,
                issued,
                conditional,
                collective: collective.clone(),
                tcgen_groups,
                commit,
            });
        }
    }
    let mut warp_index = BTreeMap::<WarpId, usize>::new();
    for raw in &raws {
        let next = warp_index.len();
        warp_index.entry(raw.warp).or_insert(next);
        if let Some(c) = &raw.collective {
            for &w in &c.participants {
                let next = warp_index.len();
                warp_index.entry(w).or_insert(next);
            }
        }
    }
    // Renumber dense warps in WarpId order for determinism.
    let warp_ids = warp_index.keys().copied().collect::<Vec<_>>();
    let dense = |w: WarpId| warp_ids.binary_search(&w).expect("indexed warp");
    let mut positions = vec![Vec::<(u32, usize)>::new(); warp_ids.len()];
    let mut commands = Vec::<Command>::new();
    let mut collectives = HashMap::<u64, (usize, Vec<usize>)>::new();
    // A collective's participants are the union of what its records declare
    // and the warps that recorded it: an engine may list only the recording
    // warp in each record (W2 cta_group::2 tcgen05 pairs, V2C-4).
    let mut members = HashMap::<u64, std::collections::BTreeSet<WarpId>>::new();
    for raw in &raws {
        if let Some(c) = &raw.collective {
            let m = members.entry(c.id).or_default();
            m.insert(raw.warp);
            m.extend(c.participants.iter().copied());
        }
    }
    for raw in raws {
        let warp = dense(raw.warp);
        if let Some(c) = &raw.collective {
            if let Some((command, seen)) = collectives.get_mut(&c.id) {
                if commands[*command].cmds != raw.cmds || commands[*command].issued != raw.issued {
                    return Err(format!("collective {} records different commands across participants", c.id));
                }
                seen.push(warp);
                positions[warp].push((raw.seq, *command));
                continue;
            }
            let mut participants = members[&c.id].iter().map(|&w| dense(w)).collect::<Vec<_>>();
            participants.retain(|&p| p != warp);
            participants.insert(0, warp);
            collectives.insert(c.id, (commands.len(), vec![warp]));
            positions[warp].push((raw.seq, commands.len()));
            commands.push(Command {
                participants,
                warp: raw.warp,
                seq: raw.seq,
                epoch: raw.epoch,
                site: raw.site,
                frames: raw.frames,
                cmds: raw.cmds,
                issued: raw.issued,
                conditional: raw.conditional,
                tcgen_groups: raw.tcgen_groups,
                commit: raw.commit,
            });
            continue;
        }
        positions[warp].push((raw.seq, commands.len()));
        commands.push(Command {
            participants: vec![warp],
            warp: raw.warp,
            seq: raw.seq,
            epoch: raw.epoch,
            site: raw.site,
            frames: raw.frames,
            cmds: raw.cmds,
            issued: raw.issued,
            conditional: raw.conditional,
            tcgen_groups: raw.tcgen_groups,
            commit: raw.commit,
        });
    }
    for (id, (command, seen)) in &collectives {
        let mut seen = seen.clone();
        seen.sort_unstable();
        let mut want = commands[*command].participants.clone();
        want.sort_unstable();
        if seen != want {
            let missing = want.iter().filter(|w| !seen.contains(w)).map(|&w| warp_ids[w].0).collect::<Vec<_>>();
            return Err(format!("collective {id} is missing the records of participant warps {missing:?}"));
        }
    }
    let mut warp_programs = Vec::with_capacity(warp_ids.len());
    for (warp, mut list) in positions.into_iter().enumerate() {
        list.sort_by_key(|(seq, _)| *seq);
        if let Some(pair) = list.windows(2).find(|p| p[0].0 == p[1].0) {
            return Err(format!("warp {} records sequence {} twice", warp_ids[warp].0, pair[0].0));
        }
        warp_programs.push(list.into_iter().map(|(_, c)| c).collect());
    }
    let tcgen_exclusive_max = commands
        .iter()
        .flat_map(|c| &c.cmds)
        .filter_map(|(_, cmd)| match cmd {
            SyncCmd::Tcgen(tcgen::Cmd::Alloc { columns, exclusive: true, .. }) => Some(*columns),
            _ => None,
        })
        .fold(tcgen::TMEM_COLUMNS, u32::max);
    Ok((Program { kernel: kernel.unwrap_or(0), warp_ids, warp_programs, commands, resources, init_cmds, tcgen_exclusive_max }, failures))
}

impl Program {
    /// Position of `command` in each participant's program.
    pub fn position(&self, warp: usize, command: usize) -> usize {
        self.warp_programs[warp].iter().position(|&c| c == command).expect("participant holds command")
    }

    /// Resources a command touches (protocol commands and issued targets).
    pub fn command_resources(&self, command: usize) -> Vec<usize> {
        let c = &self.commands[command];
        let mut out = c.cmds.iter().map(|(r, _)| *r).chain(c.issued.iter().map(|(r, _, _)| *r)).collect::<Vec<_>>();
        out.sort_unstable();
        out.dedup();
        out
    }
}

impl Program {
    /// The kernel-wide tcgen05 `.cta_group` rule, through the production
    /// `tcgen::use_cta_group` over every use in program order. Order does not
    /// matter: any two distinct groups (or an invalid one) fail.
    pub fn cta_group_error(&self) -> Option<(usize, SyncError)> {
        let mut k = tcgen::KernelState::default();
        for (c, cmd) in self.commands.iter().enumerate() {
            for &g in &cmd.tcgen_groups {
                if let Err(e) = tcgen::use_cta_group(&mut k, g) {
                    return Some((c, SyncError::Tcgen(e)));
                }
            }
        }
        None
    }
}
