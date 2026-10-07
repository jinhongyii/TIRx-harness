//! Per-warp command sequences and the reference run.

use std::collections::{BTreeMap, HashMap};

use crate::clock::Clock;
use crate::event::{OpId, ResourceId, SyncEvent, SyncOp, WarpId};
use crate::projection::{ProjectionKey, ProjectionSpec};
use crate::ts::{Deadlock, ProtocolTs, Transition, TsError};

#[derive(Clone, Debug)]
pub struct Command {
    pub op: OpId,
    pub kind: SyncOp,
    /// Dense warp index into [`Program::warp_ids`].
    pub warp: usize,
    /// Dense resource index into [`Program::resources`].
    pub resource: usize,
}

/// The fixed synchronization program: every warp's executed path, in order.
#[derive(Clone, Debug)]
pub struct Program {
    pub warp_ids: Vec<WarpId>,
    pub warp_programs: Vec<Vec<usize>>,
    pub commands: Vec<Command>,
    pub resources: Vec<ResourceId>,
}

impl Program {
    /// Group events by warp and order each warp by its per-warp sequence.
    ///
    /// This is today's `protocol_projections_from_snapshot` staging step minus
    /// collectives (setmaxnreg/tcgen), which are out of scope for the prototype.
    pub fn from_events(events: &[SyncEvent]) -> Result<Self, String> {
        let mut by_warp = BTreeMap::<WarpId, Vec<&SyncEvent>>::new();
        for event in events {
            by_warp.entry(event.op.warp).or_default().push(event);
        }
        let mut resource_index = BTreeMap::<ResourceId, usize>::new();
        for event in events {
            let next = resource_index.len();
            resource_index.entry(event.kind.resource()).or_insert(next);
        }
        // Re-number resources in sorted order for deterministic projections.
        let resources = resource_index.keys().copied().collect::<Vec<_>>();
        let resource_index = resources
            .iter()
            .enumerate()
            .map(|(index, resource)| (*resource, index))
            .collect::<HashMap<_, _>>();

        let mut warp_ids = Vec::new();
        let mut warp_programs = Vec::new();
        let mut commands = Vec::new();
        for (warp_index, (warp_id, mut warp_events)) in by_warp.into_iter().enumerate() {
            warp_events.sort_by_key(|event| event.op.seq);
            if let Some(pair) = warp_events.windows(2).find(|pair| pair[0].op.seq == pair[1].op.seq) {
                return Err(format!(
                    "warp {warp_id} records sequence {} twice",
                    pair[0].op.seq
                ));
            }
            warp_ids.push(warp_id);
            let mut program = Vec::with_capacity(warp_events.len());
            for event in warp_events {
                program.push(commands.len());
                commands.push(Command {
                    op: event.op,
                    kind: event.kind,
                    warp: warp_index,
                    resource: resource_index[&event.kind.resource()],
                });
            }
            warp_programs.push(program);
        }
        Ok(Self {
            warp_ids,
            warp_programs,
            commands,
            resources,
        })
    }

    pub fn whole_spec(&self) -> ProjectionSpec {
        ProjectionSpec {
            key: ProjectionKey::Whole,
            commands: (0..self.commands.len()).collect(),
            gated: false,
        }
    }
}

#[derive(Clone, Debug)]
pub enum RunOutcome {
    Complete,
    Error(TsError),
    Deadlock(Deadlock),
}

/// One complete schedule of the whole program with causal annotations.
///
/// Stand-in for the clocks and generations today's engine records in
/// `ResolvedTransitionLog` during the online NumSim run. Any complete
/// schedule is acceptable: the certificates prove the annotations are
/// schedule-independent before relying on them.
#[derive(Clone, Debug)]
pub struct ReferenceRun {
    /// Clock at issue (registration) of each command.
    pub initial: Vec<Option<Clock>>,
    /// Clock when the command returned (after any acquire).
    pub final_: Vec<Option<Clock>>,
    /// Generation each command contributed to or consumed.
    pub generation: Vec<Option<u32>>,
    pub schedule: Vec<Transition>,
    pub outcome: RunOutcome,
}

impl ReferenceRun {
    pub fn is_complete(&self) -> bool {
        matches!(self.outcome, RunOutcome::Complete)
    }
}

/// Run the whole program once, round-robin over warps, recording clocks.
pub fn reference_run(program: &Program) -> ReferenceRun {
    let warps = program.warp_ids.len();
    let spec = program.whole_spec();
    let ts = ProtocolTs::new(program, &spec, None);
    let mut state = ts.initial();
    let mut clocks = vec![Clock::zero(warps); warps];
    let mut payload = HashMap::<(usize, u32), Clock>::new();
    let mut run = ReferenceRun {
        initial: vec![None; program.commands.len()],
        final_: vec![None; program.commands.len()],
        generation: vec![None; program.commands.len()],
        schedule: Vec::new(),
        outcome: RunOutcome::Complete,
    };
    let mut round_robin = 0usize;
    loop {
        let enabled = ts.enabled(&state);
        let Some(transition) = pick_round_robin(&enabled, round_robin, ts.warps.len()) else {
            run.outcome = if ts.is_complete(&state) {
                RunOutcome::Complete
            } else {
                RunOutcome::Deadlock(ts.describe_deadlock(&state))
            };
            return run;
        };
        if let Transition::Exit = transition {
            if let Err(error) = ts.step(&state, &transition) {
                run.schedule.push(transition);
                run.outcome = RunOutcome::Error(error);
            }
            return run;
        }
        let issued = match transition {
            Transition::Issue(warp) => Some(ts.head(&state, warp as usize).expect("enabled head")),
            _ => None,
        };
        let (next, effects) = match ts.step_with_effects(&state, &transition) {
            Ok(result) => result,
            Err(error) => {
                run.schedule.push(transition);
                run.outcome = RunOutcome::Error(error);
                return run;
            }
        };
        run.schedule.push(transition.clone());
        match transition {
            Transition::Issue(local_warp) => {
                round_robin = local_warp as usize + 1;
                let local = issued.expect("issued command");
                let command = ts.cmds[local].global;
                let warp = program.commands[command].warp;
                let resource = program.commands[command].resource;
                clocks[warp].tick(warp);
                run.initial[command] = Some(clocks[warp].clone());
                run.generation[command] = effects.generation;
                if let (true, Some(generation)) = (effects.release, effects.generation) {
                    payload
                        .entry((resource, generation))
                        .or_insert_with(|| Clock::zero(warps))
                        .join(&clocks[warp]);
                }
                if let Some(generation) = effects.acquire {
                    if let Some(released) = payload.get(&(resource, generation)) {
                        clocks[warp].join(released);
                    }
                }
                for &(released_local_warp, released_local_cmd) in &effects.released {
                    let released_warp = ts.warps[released_local_warp];
                    let released_command = ts.cmds[released_local_cmd].global;
                    let generation = run.generation[released_command]
                        .or(effects.generation)
                        .expect("released command has a generation");
                    if let Some(released) = payload.get(&(resource, generation)) {
                        clocks[released_warp].join(released);
                    }
                    run.final_[released_command] = Some(clocks[released_warp].clone());
                }
                if !effects.blocked {
                    run.final_[command] = Some(clocks[warp].clone());
                }
            }
            Transition::Complete(local) => {
                let command = ts.cmds[local as usize].global;
                let resource = program.commands[command].resource;
                if let (Some(generation), Some(issuer)) =
                    (effects.generation, run.initial[command].clone())
                {
                    payload
                        .entry((resource, generation))
                        .or_insert_with(|| Clock::zero(warps))
                        .join(&issuer);
                }
            }
            Transition::Exit => unreachable!("handled above"),
        }
        state = next;
    }
}

fn pick_round_robin(enabled: &[Transition], start: usize, warps: usize) -> Option<Transition> {
    if let Some(exit) = enabled.iter().find(|t| matches!(t, Transition::Exit)) {
        return Some(exit.clone());
    }
    let issue = enabled
        .iter()
        .filter_map(|t| match t {
            Transition::Issue(warp) => Some(*warp as usize),
            _ => None,
        })
        .min_by_key(|warp| (warp + warps - start % warps.max(1)) % warps.max(1));
    if let Some(warp) = issue {
        return Some(Transition::Issue(warp as u32));
    }
    enabled.first().cloned()
}
