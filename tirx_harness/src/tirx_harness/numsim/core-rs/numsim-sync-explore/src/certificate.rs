//! Causal certificates: O(commands) proofs that one resource's protocol is
//! schedule-independent, replacing state search (`visited_states: 1`).
//!
//! Each certificate consumes the reference run's per-command generation and
//! vector clocks. The argument has two halves:
//!
//! * **Counting.** Within one generation, contributions commute (they only add
//!   to counters). If every generation's totals are exact, every schedule that
//!   keeps the same generation assignment reaches the same protocol states.
//! * **Assignment.** A schedule can only change which generation a command
//!   lands in by overtaking. Requiring the release of generation `g` (or a
//!   consuming wait of `g`) to happen-before every contribution of `g + 1`,
//!   and every wait of `g` to happen-before some prerequisite of `g + 1`,
//!   rules that out for every HB-respecting schedule.
//!
//! `None` means "not applicable, fall back to state search".
//!
//! Ports: `verify_named_barrier_causally` (`sync_fixed_unified.rs:1043-1296`),
//! `verify_cluster_barrier_causally` (`:1298-1554`),
//! `verify_mbarrier_causally` (`:1556-2069`). Omitted for the prototype:
//! named-barrier lane masks / aligned-site checks, conditional waits, wait
//! batches, invalidate/re-init, `.noinc` pending increments.

use std::collections::BTreeMap;

use crate::clock::Clock;
use crate::event::{OpId, SyncOp};
use crate::program::ReferenceRun;
use crate::projection::ProjectionKey;
use crate::ts::ProtocolTs;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CertError {
    pub kind: &'static str,
    pub op: Option<OpId>,
    pub related: Vec<OpId>,
    pub detail: String,
    pub incomplete: bool,
}

fn error(kind: &'static str, op: OpId, related: Vec<OpId>, detail: String) -> CertError {
    CertError {
        kind,
        op: Some(op),
        related,
        detail,
        incomplete: false,
    }
}

fn incomplete(kind: &'static str, op: Option<OpId>, detail: String) -> CertError {
    CertError {
        kind,
        op,
        related: Vec::new(),
        detail,
        incomplete: true,
    }
}

pub fn certify(ts: &ProtocolTs<'_>, reference: &ReferenceRun) -> Option<Result<(), CertError>> {
    if !matches!(ts.key, ProjectionKey::Resource(_)) || !reference.is_complete() {
        return None;
    }
    match ts.cmds.first()?.kind {
        SyncOp::NamedArrive { .. } | SyncOp::NamedSync { .. } => Some(named(ts, reference)),
        SyncOp::ClusterArrive { .. } | SyncOp::ClusterWait { .. } => Some(cluster(ts, reference)),
        _ => mbarrier(ts, reference),
    }
}

struct Annotated<'a> {
    op: OpId,
    kind: SyncOp,
    warp: usize,
    position: usize,
    generation: Option<u32>,
    initial: &'a Clock,
    final_: &'a Clock,
}

fn annotate<'a>(ts: &ProtocolTs<'_>, reference: &'a ReferenceRun) -> Option<Vec<Annotated<'a>>> {
    ts.cmds
        .iter()
        .enumerate()
        .map(|(index, cmd)| {
            Some(Annotated {
                op: cmd.op,
                kind: cmd.kind,
                warp: cmd.warp,
                position: ts.positions[index],
                generation: reference.generation[cmd.global],
                initial: reference.initial[cmd.global].as_ref()?,
                final_: reference.final_[cmd.global].as_ref()?,
            })
        })
        .collect()
}

fn named(ts: &ProtocolTs<'_>, reference: &ReferenceRun) -> Result<(), CertError> {
    let Some(cmds) = annotate(ts, reference) else {
        return Err(incomplete("named_barrier_unannotated", None, "missing clocks".into()));
    };
    struct Generation {
        expected: u32,
        arrived: u32,
        release: Clock,
        first: OpId,
        members: Vec<usize>,
    }
    let mut generations = BTreeMap::<u32, Generation>::new();
    for (index, cmd) in cmds.iter().enumerate() {
        let (SyncOp::NamedArrive { expected, count, .. } | SyncOp::NamedSync { expected, count, .. }) =
            cmd.kind
        else {
            return Err(incomplete("named_barrier_mixed_projection", Some(cmd.op), "another protocol".into()));
        };
        let Some(generation) = cmd.generation else {
            return Err(incomplete("named_barrier_unannotated", Some(cmd.op), "no generation".into()));
        };
        let entry = generations.entry(generation).or_insert_with(|| Generation {
            expected,
            arrived: 0,
            release: cmd.initial.clone(),
            first: cmd.op,
            members: Vec::new(),
        });
        if entry.expected != expected {
            return Err(error(
                "named_barrier_contract_mismatch",
                cmd.op,
                vec![entry.first],
                format!("generation {generation} changes expected arrivals from {} to {expected}", entry.expected),
            ));
        }
        entry.arrived += count;
        entry.release.join(cmd.initial);
        entry.members.push(index);
    }
    let mut prior: Option<(u32, &Generation)> = None;
    for (&generation, state) in &generations {
        let expected_generation = prior.map_or(0, |(g, _)| g + 1);
        if generation != expected_generation {
            return Err(incomplete(
                "named_barrier_generation_gap",
                Some(state.first),
                format!("skips generation {expected_generation} before {generation}"),
            ));
        }
        if state.arrived != state.expected {
            return Err(error(
                "named_barrier_arrival_count",
                state.first,
                Vec::new(),
                format!("generation {generation} has {} of {} required arrivals", state.arrived, state.expected),
            ));
        }
        if let Some((prior_generation, prior_state)) = prior {
            for &member in &state.members {
                if !prior_state.release.leq(cmds[member].initial) {
                    return Err(error(
                        "named_barrier_generation_not_ordered",
                        cmds[member].op,
                        vec![prior_state.first],
                        format!(
                            "contribution to generation {generation} is not happens-before ordered after generation {prior_generation}'s release"
                        ),
                    ));
                }
            }
        }
        prior = Some((generation, state));
    }
    Ok(())
}

fn cluster(ts: &ProtocolTs<'_>, reference: &ReferenceRun) -> Result<(), CertError> {
    let Some(cmds) = annotate(ts, reference) else {
        return Err(incomplete("cluster_barrier_unannotated", None, "missing clocks".into()));
    };
    let mut participants = None;
    // generation -> (warp -> arrival index), (warp -> wait index)
    let mut arrivals = BTreeMap::<u32, BTreeMap<usize, usize>>::new();
    let mut waits = BTreeMap::<u32, BTreeMap<usize, usize>>::new();
    for (index, cmd) in cmds.iter().enumerate() {
        let Some(generation) = cmd.generation else {
            return Err(incomplete("cluster_barrier_unannotated", Some(cmd.op), "no generation".into()));
        };
        match cmd.kind {
            SyncOp::ClusterArrive { participants: count, .. } => {
                if *participants.get_or_insert(count) != count {
                    return Err(error("cluster_barrier_contract_mismatch", cmd.op, Vec::new(), "participant count changes".into()));
                }
                if arrivals.entry(generation).or_default().insert(cmd.warp, index).is_some() {
                    return Err(error("cluster_barrier_early_arrival", cmd.op, Vec::new(), format!("duplicate arrival in generation {generation}")));
                }
            }
            SyncOp::ClusterWait { .. } => {
                waits.entry(generation).or_default().insert(cmd.warp, index);
            }
            _ => return Err(incomplete("cluster_barrier_mixed_projection", Some(cmd.op), "another protocol".into())),
        }
    }
    let Some(participants) = participants else {
        return Ok(());
    };
    for (expected_generation, (&generation, members)) in arrivals.iter().enumerate() {
        if generation as usize != expected_generation {
            return Err(incomplete("cluster_barrier_generation_gap", None, format!("skips generation {expected_generation}")));
        }
        if members.len() as u32 != participants {
            return Err(incomplete(
                "cluster_barrier_warp_exit_unmodeled",
                members.values().next().map(|&i| cmds[i].op),
                format!("generation {generation} has {} of {participants} participant warps", members.len()),
            ));
        }
        for (&warp, &wait) in waits.get(&generation).into_iter().flatten() {
            let Some(&arrival) = members.get(&warp) else {
                return Err(error("cluster_barrier_wait_before_arrival", cmds[wait].op, Vec::new(), "wait without arrival".into()));
            };
            if cmds[arrival].position >= cmds[wait].position {
                return Err(error("cluster_barrier_wait_before_arrival", cmds[wait].op, vec![cmds[arrival].op], "wait not program-ordered after arrival".into()));
            }
        }
        if generation > 0 {
            let prior = waits.get(&(generation - 1));
            for (&warp, &arrival) in members {
                let Some(&prior_wait) = prior.and_then(|w| w.get(&warp)) else {
                    return Err(incomplete(
                        "cluster_barrier_rearrival_without_wait_unmodeled",
                        Some(cmds[arrival].op),
                        format!("warp re-arrives for generation {generation} without consuming {}", generation - 1),
                    ));
                };
                if cmds[prior_wait].position >= cmds[arrival].position {
                    return Err(error(
                        "cluster_barrier_early_arrival",
                        cmds[arrival].op,
                        vec![cmds[prior_wait].op],
                        format!("arrival for generation {generation} can run before the warp consumes {}", generation - 1),
                    ));
                }
            }
        }
    }
    Ok(())
}

fn mbarrier(ts: &ProtocolTs<'_>, reference: &ReferenceRun) -> Option<Result<(), CertError>> {
    let cmds = annotate(ts, reference)?;
    let inits = cmds
        .iter()
        .filter(|cmd| matches!(cmd.kind, SyncOp::MbarInit { .. }))
        .collect::<Vec<_>>();
    // Re-initialization needs the exact state machine.
    let [init] = inits.as_slice() else {
        return None;
    };
    let SyncOp::MbarInit { expected, .. } = init.kind else {
        unreachable!()
    };
    #[derive(Default)]
    struct Generation {
        arrivals: u32,
        expected_tx: u64,
        completed_tx: u64,
        mutations: Vec<usize>,
        waits: Vec<usize>,
    }
    let mut generations = BTreeMap::<u32, Generation>::new();
    for (index, cmd) in cmds.iter().enumerate() {
        if std::ptr::eq(cmd, *init) {
            continue;
        }
        if !init.final_.hb(cmd.initial) {
            return Some(Err(error(
                "mbarrier_init_not_happens_before_use",
                cmd.op,
                vec![init.op],
                format!("{} is not happens-before ordered after init", cmd.kind.name()),
            )));
        }
        let Some(generation) = cmd.generation else {
            // A parity-1 wait on a fresh barrier observes no generation.
            if matches!(cmd.kind, SyncOp::MbarWait { parity: 1, .. }) {
                continue;
            }
            return Some(Err(incomplete("mbarrier_unannotated", Some(cmd.op), "no generation".into())));
        };
        let entry = generations.entry(generation).or_default();
        match cmd.kind {
            SyncOp::MbarArrive { count, expect_tx, .. } => {
                entry.arrivals += count;
                entry.expected_tx += u64::from(expect_tx);
                entry.mutations.push(index);
            }
            SyncOp::MbarExpectTx { tx, .. } => {
                entry.expected_tx += u64::from(tx);
                entry.mutations.push(index);
            }
            SyncOp::MbarTxIssue { tx, .. } => {
                entry.completed_tx += u64::from(tx);
                entry.mutations.push(index);
            }
            SyncOp::MbarWait { parity, .. } => {
                if u32::from(parity) != generation & 1 {
                    return Some(Err(error(
                        "mbarrier_invalid_phase",
                        cmd.op,
                        Vec::new(),
                        format!("wait requests parity {parity} but consumed generation {generation}"),
                    )));
                }
                entry.waits.push(index);
            }
            _ => return Some(Err(incomplete("mbarrier_mixed_projection", Some(cmd.op), "another protocol".into()))),
        }
    }
    for (position, (&generation, state)) in generations.iter().enumerate() {
        if generation as usize != position {
            return Some(Err(incomplete("mbarrier_generation_gap", None, format!("skips generation {position}"))));
        }
        let has_next = generations.contains_key(&(generation + 1));
        let requires_completion = has_next || !state.waits.is_empty();
        let first = state.mutations.first().map_or(init.op, |&i| cmds[i].op);
        if state.arrivals > expected
            || (state.arrivals == expected && state.completed_tx > state.expected_tx)
            || (requires_completion
                && (state.arrivals != expected || state.completed_tx != state.expected_tx))
        {
            return Some(Err(error(
                "mbarrier_generation_count_mismatch",
                first,
                Vec::new(),
                format!(
                    "generation {generation} has {}/{expected} arrivals and {}/{} completed/expected transaction bytes",
                    state.arrivals, state.completed_tx, state.expected_tx
                ),
            )));
        }
        if has_next && state.waits.is_empty() {
            return Some(Err(error(
                "mbarrier_prior_generation_not_consumed",
                first,
                Vec::new(),
                format!("advances past generation {generation} without a consuming wait"),
            )));
        }
        if generation > 0 {
            let prior = &generations[&(generation - 1)];
            for &mutation in &state.mutations {
                let ordered = prior
                    .waits
                    .iter()
                    .any(|&wait| cmds[wait].final_.hb(cmds[mutation].initial));
                if !ordered {
                    return Some(Err(error(
                        "mbarrier_prior_generation_consumption_not_happens_before",
                        cmds[mutation].op,
                        prior.waits.first().map(|&w| cmds[w].op).into_iter().collect(),
                        format!(
                            "generation {generation} mutation can run before generation {} is consumed",
                            generation - 1
                        ),
                    )));
                }
            }
        }
        // A wait can only be overtaken by a next generation that completes.
        // (Today's certificate also checks terminal, never-completing
        // generations, which rejects programs the exhaustive search accepts.)
        if let Some(next) = generations
            .get(&(generation + 1))
            .filter(|next| next.arrivals == expected && next.completed_tx == next.expected_tx)
        {
            for &wait in &state.waits {
                let precedes = next
                    .mutations
                    .iter()
                    .any(|&mutation| cmds[wait].initial.hb(cmds[mutation].initial));
                if !precedes {
                    return Some(Err(error(
                        "mbarrier_wait_overtaken",
                        cmds[wait].op,
                        next.mutations.first().map(|&m| cmds[m].op).into_iter().collect(),
                        format!(
                            "wait for generation {generation} can be overtaken by generation {} completion",
                            generation + 1
                        ),
                    )));
                }
            }
        }
    }
    Some(Ok(()))
}
