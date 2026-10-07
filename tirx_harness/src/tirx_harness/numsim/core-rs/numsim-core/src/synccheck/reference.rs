//! One complete schedule of the whole program with causal annotations.
//!
//! Replaces the clocks and generations today's engine records online
//! (`ResolvedTransitionLog`): any complete schedule works, because the
//! certificates and the gated per-resource searches prove the annotations
//! schedule-independent before relying on them.

use std::collections::HashMap;

use super::clock::Clock;
use super::program::Program;
use super::projection::{ProjectionKey, ProjectionSpec};
use super::ts::{Deadlock, Transition, Ts, TsError};
use crate::sync::ResourceInit;

#[derive(Clone, Debug)]
pub enum RunOutcome {
    Complete,
    Error(TsError),
    Deadlock(Deadlock),
}

#[derive(Clone, Debug)]
pub struct ReferenceRun {
    /// Clock at issue of each command (join over participants).
    pub initial: Vec<Option<Clock>>,
    /// Clock when the command returned (after acquires).
    pub final_: Vec<Option<Clock>>,
    /// Generation per protocol command of each command.
    pub gens: Vec<Vec<Option<u64>>>,
    /// Captured generation per issued target.
    pub issued_gens: Vec<Vec<Option<u64>>>,
    pub schedule: Vec<Transition>,
    pub outcome: RunOutcome,
    /// Exit lints of the terminal state: `(global resource, kind, detail)`.
    pub lints: Vec<(usize, crate::report::FindingKind, String)>,
}

impl ReferenceRun {
    pub fn is_complete(&self) -> bool {
        matches!(self.outcome, RunOutcome::Complete)
    }
}

pub fn whole_spec(program: &Program) -> ProjectionSpec {
    ProjectionSpec {
        key: ProjectionKey::Whole,
        commands: (0..program.commands.len()).collect(),
        gated: false,
        resource_count: program.resources.len(),
    }
}

/// Round-robin over warps; completions and grants only when no warp can move.
pub fn run(program: &Program, init: &ResourceInit) -> Result<ReferenceRun, String> {
    let ts = Ts::new(program, &whole_spec(program), init, None)?;
    let warps = program.warp_ids.len();
    let n = program.commands.len();
    let mut clocks = vec![Clock::zero(warps); warps];
    let mut payload = HashMap::<(usize, u64), Clock>::new();
    let mut run = ReferenceRun {
        initial: vec![None; n],
        final_: vec![None; n],
        gens: program.commands.iter().map(|c| vec![None; c.cmds.len()]).collect(),
        issued_gens: program.commands.iter().map(|c| vec![None; c.issued.len()]).collect(),
        schedule: Vec::new(),
        outcome: RunOutcome::Complete,
        lints: Vec::new(),
    };
    let mut state = ts.initial();
    let mut rr = 0usize;
    loop {
        let enabled = ts.enabled(&state);
        let warp_of = |t: &Transition| -> Option<usize> {
            match *t {
                Transition::Issue(c) => Some(*ts.cmds[c as usize].participants.iter().min().expect("participants")),
                Transition::Resume(w) => Some(w as usize),
                _ => None,
            }
        };
        let local_warps = ts.warps.len().max(1);
        let pick = enabled
            .iter()
            .filter(|t| warp_of(t).is_some())
            .min_by_key(|t| (warp_of(t).unwrap() + local_warps - rr % local_warps) % local_warps)
            .or_else(|| enabled.first())
            .copied();
        let Some(t) = pick else {
            run.outcome = if state.exited { RunOutcome::Complete } else { RunOutcome::Deadlock(ts.describe_deadlock(&state)) };
            return Ok(run);
        };
        if t == Transition::Exit {
            if let Err(e) = ts.step_fx(&state, &t) {
                run.schedule.push(t);
                run.outcome = RunOutcome::Error(e);
            }
            for (r, res) in state.res.iter().enumerate() {
                if let Some((kind, detail)) = super::backend::exit_lint(res) {
                    run.lints.push((ts.resources[r], kind, detail));
                }
            }
            return Ok(run);
        }
        let head_of_resume = match t {
            Transition::Resume(w) => ts.head(&state, w as usize),
            _ => None,
        };
        let (next, fx) = match ts.step_fx(&state, &t) {
            Ok(x) => x,
            Err(e) => {
                run.schedule.push(t);
                run.outcome = RunOutcome::Error(e);
                return Ok(run);
            }
        };
        run.schedule.push(t);
        match t {
            Transition::Issue(c) => {
                let lc = &ts.cmds[c as usize];
                let g = lc.global;
                rr = lc.participants.iter().min().copied().unwrap_or(0) + 1;
                let dense = lc.participants.iter().map(|&w| ts.warps[w]).collect::<Vec<_>>();
                // Arming a wait re-issues the same command later: keep the first issue clock.
                if run.initial[g].is_none() {
                    let mut joined = Clock::zero(warps);
                    for &w in &dense {
                        clocks[w].tick(w);
                        joined.join(&clocks[w]);
                    }
                    if dense.len() > 1 {
                        for &w in &dense {
                            clocks[w] = joined.clone();
                        }
                    }
                    run.initial[g] = Some(joined);
                }
                let initial = run.initial[g].clone().expect("set above");
                for &(r, gen) in &fx.release {
                    payload.entry((r, gen)).or_insert_with(|| Clock::zero(warps)).join(&initial);
                }
                for &(r, gen) in &fx.acquire {
                    if let Some(p) = payload.get(&(r, gen)).cloned() {
                        for &w in &dense {
                            clocks[w].join(&p);
                        }
                    }
                }
                for (i, gen) in fx.gens.iter().enumerate() {
                    if gen.is_some() {
                        run.gens[g][i] = *gen;
                    }
                }
                for (i, gen) in fx.issued_gens.iter().enumerate() {
                    if gen.is_some() {
                        run.issued_gens[g][i] = *gen;
                    }
                }
                if fx.returned {
                    let mut f = Clock::zero(warps);
                    for &w in &dense {
                        f.join(&clocks[w]);
                    }
                    run.final_[g] = Some(f);
                }
            }
            Transition::Resume(w) => {
                rr = w as usize + 1;
                let dense = ts.warps[w as usize];
                for &(r, gen) in &fx.acquire {
                    if let Some(p) = payload.get(&(r, gen)).cloned() {
                        clocks[dense].join(&p);
                    }
                }
                if let Some(c) = head_of_resume {
                    let g = ts.cmds[c].global;
                    let mut f = run.final_[g].clone().unwrap_or_else(|| Clock::zero(warps));
                    f.join(&clocks[dense]);
                    run.final_[g] = Some(f);
                }
            }
            Transition::Complete(c, _) => {
                let g = ts.cmds[c as usize].global;
                if let Some(issuer) = run.initial[g].clone() {
                    for &(r, gen) in &fx.release {
                        payload.entry((r, gen)).or_insert_with(|| Clock::zero(warps)).join(&issuer);
                    }
                }
            }
            Transition::Grant(..) | Transition::Arm(_) | Transition::Exit => {}
        }
        state = next;
    }
}
