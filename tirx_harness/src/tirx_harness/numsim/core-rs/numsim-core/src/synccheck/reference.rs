//! One complete schedule of the whole program with causal annotations.
//!
//! Replaces the clocks and generations today's engine records online
//! (`ResolvedTransitionLog`): any complete schedule works, because the
//! certificates and the gated per-resource searches prove the annotations
//! schedule-independent before relying on them.

use std::collections::HashMap;

use super::clock::Clock;
use super::program::Program;
use super::projection::{project, ProjectionKey, ProjectionMode, ProjectionSpec};
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
    /// Landing clocks of commit-FIFO completions, by `(global command, ordinal)`.
    pub landings: HashMap<(usize, u16), Clock>,
    /// Exit lints of the terminal state: `(global resource, kind, detail)`.
    pub lints: Vec<(usize, crate::report::FindingKind, String)>,
    /// The system `schedule` indexes: the component that failed (whole
    /// program when the run completed).
    pub spec: ProjectionSpec,
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

pub enum RunError {
    Build(String),
    /// The wall-clock deadline passed (checked every 1024 steps).
    WallTime,
}

/// Round-robin over warps; completions and grants only when no warp can move.
///
/// Connected components (warps and resources linked by commands, see
/// [`ProjectionMode::Components`]) never synchronize with each other, so each
/// runs on its own with clocks over its own warps (V2C-31: a 128-CTA launch
/// is 128 small runs, not one run whose every step clones 1792 warps and
/// 2048 resources). Every clock comparison stays inside one component: each
/// gated projection and certificate lies inside one.
pub fn run(program: &Program, init: &ResourceInit, deadline: Option<std::time::Instant>) -> Result<ReferenceRun, RunError> {
    let n = program.commands.len();
    let mut run = ReferenceRun {
        initial: vec![None; n],
        final_: vec![None; n],
        gens: program.commands.iter().map(|c| vec![None; c.cmds.len()]).collect(),
        issued_gens: program.commands.iter().map(|c| vec![None; c.issued.len()]).collect(),
        schedule: Vec::new(),
        outcome: RunOutcome::Complete,
        lints: Vec::new(),
        landings: HashMap::new(),
        spec: whole_spec(program),
    };
    let mut steps = 0u64;
    for spec in project(program, ProjectionMode::Components) {
        let ts = Ts::new(program, &spec, init, None).map_err(RunError::Build)?;
        if !run_component(program, &ts, &mut run, &mut steps, deadline)? {
            run.spec = spec;
            return Ok(run);
        }
    }
    Ok(run)
}

/// Run one component to its end; `false` = it stopped with an error or a
/// deadlock (recorded in `run.outcome` / `run.schedule`).
fn run_component(
    program: &Program,
    ts: &Ts<'_>,
    run: &mut ReferenceRun,
    steps: &mut u64,
    deadline: Option<std::time::Instant>,
) -> Result<bool, RunError> {
    let warps = ts.warps.len();
    // One extra clock component per tcgen05.commit FIFO (issuing warp): a
    // commit's landing is an event of its own, ordered after the warp's
    // earlier commits, so HB through one landing implies the earlier ones.
    let mut fifo_index = HashMap::<usize, usize>::new();
    for c in ts.cmds.iter().map(|lc| &program.commands[lc.global]).filter(|c| c.commit) {
        let next = warps + fifo_index.len();
        fifo_index.entry(c.participants[0]).or_insert(next);
    }
    let dims = warps + fifo_index.len();
    let mut fifo_last = HashMap::<usize, Clock>::new();
    let mut clocks = vec![Clock::zero(dims); warps];
    let mut payload = HashMap::<(usize, u64), Clock>::new();
    run.schedule.clear();
    let mut state = ts.initial();
    let mut rr = 0usize;
    loop {
        *steps += 1;
        if *steps % 1024 == 0 && deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return Err(RunError::WallTime);
        }
        let local_warps = ts.warps.len().max(1);
        let pick = ts
            .first_warp_transition(&state, rr % local_warps)
            .or_else(|| ts.enabled(&state).first().copied());
        let Some(t) = pick else {
            if state.exited {
                return Ok(true);
            }
            run.outcome = RunOutcome::Deadlock(ts.describe_deadlock(&state));
            return Ok(false);
        };
        if t == Transition::Exit {
            if let Err(e) = ts.step_fx(&state, &t) {
                run.schedule.push(t);
                run.outcome = RunOutcome::Error(e);
                return Ok(false);
            }
            for (r, res) in state.res.iter().enumerate() {
                if let Some((kind, detail)) = super::backend::exit_lint(res) {
                    run.lints.push((ts.resources[r], kind, detail));
                }
            }
            return Ok(true);
        }
        let landing_fifo = match t {
            Transition::Complete(c, o) => state.pending.iter().find(|p| (p.cmd, p.ord) == (c, o)).and_then(|p| p.fifo),
            _ => None,
        };
        let head_of_resume = match t {
            Transition::Resume(w) => ts.head(&state, w as usize),
            _ => None,
        };
        let (next, fx) = match ts.step_fx(&state, &t) {
            Ok(x) => x,
            Err(e) => {
                run.schedule.push(t);
                run.outcome = RunOutcome::Error(e);
                return Ok(false);
            }
        };
        run.schedule.push(t);
        match t {
            Transition::Issue(c) => {
                let lc = &ts.cmds[c as usize];
                let g = lc.global;
                rr = lc.participants.iter().min().copied().unwrap_or(0) + 1;
                let dense = lc.participants.clone();
                // Arming a wait re-issues the same command later: keep the first issue clock.
                if run.initial[g].is_none() {
                    let mut joined = Clock::zero(dims);
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
                    payload.entry((r, gen)).or_insert_with(|| Clock::zero(dims)).join(&initial);
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
                    let mut f = Clock::zero(dims);
                    for &w in &dense {
                        f.join(&clocks[w]);
                    }
                    run.final_[g] = Some(f);
                }
            }
            Transition::Resume(w) => {
                rr = w as usize + 1;
                let dense = w as usize;
                for &(r, gen) in &fx.acquire {
                    if let Some(p) = payload.get(&(r, gen)).cloned() {
                        clocks[dense].join(&p);
                    }
                }
                if let Some(c) = head_of_resume {
                    let g = ts.cmds[c].global;
                    let mut f = run.final_[g].clone().unwrap_or_else(|| Clock::zero(dims));
                    f.join(&clocks[dense]);
                    run.final_[g] = Some(f);
                }
            }
            Transition::Complete(c, o) => {
                let g = ts.cmds[c as usize].global;
                if let Some(issuer) = run.initial[g].clone() {
                    let mut released = issuer;
                    if let Some(f) = landing_fifo.and_then(|w| fifo_index.get(&(w as usize)).copied()) {
                        if let Some(prev) = fifo_last.get(&f) {
                            released.join(prev);
                        }
                        released.tick(f);
                        fifo_last.insert(f, released.clone());
                        run.landings.insert((g, o), released.clone());
                    }
                    for &(r, gen) in &fx.release {
                        payload.entry((r, gen)).or_insert_with(|| Clock::zero(dims)).join(&released);
                    }
                }
            }
            Transition::Grant(..) | Transition::Arm(_) | Transition::Exit => {}
        }
        state = next;
    }
}
