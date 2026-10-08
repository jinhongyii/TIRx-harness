//! Causal certificates: O(commands) proofs that one resource's protocol is
//! schedule-independent (`visited_states = 1`). See
//! `docs/development/synccheck-explorer.md` section 2.3 for the counting +
//! vector-clock argument. `None` = not applicable; fall back to search.
//!
//! Inputs are the reference run's per-command generations and clocks.

use std::collections::{BTreeMap, BTreeSet};

use super::clock::Clock;
use super::reference::ReferenceRun;
use super::ts::Ts;
use crate::sync::{cluster, mbarrier, named, ResourceId, SyncCmd};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CertError {
    /// Stable kind string (payload `source_kind`).
    pub kind: &'static str,
    /// `"Mbarrier" | "NamedBarrier" | "ClusterBarrier"` (payload `protocol`).
    pub protocol: &'static str,
    pub cmd: Option<usize>,
    pub related: Vec<usize>,
    pub detail: String,
    pub incomplete: bool,
}

struct Item<'a> {
    global: usize,
    cmd: SyncCmd,
    gen: Option<u64>,
    /// Captured generation and `(bytes, arrivals)` of issued targets on this resource.
    issued: Vec<(Option<u64>, u64, u64)>,
    warp: usize,
    position: usize,
    initial: &'a Clock,
    final_: &'a Clock,
}

/// Single-resource projections whose commands each carry exactly one
/// protocol command on that resource (plus issued targets on it).
fn items<'a>(ts: &Ts<'_>, reference: &'a ReferenceRun) -> Option<Vec<Item<'a>>> {
    if ts.resources.len() != 1 || !reference.is_complete() {
        return None;
    }
    let mut out = Vec::new();
    // Uses the projection's commands as the contract delivers them, with
    // the explorer's synthesized `Issue` for an event that only lists
    // `issued` targets (a TMA event carries no explicit `Issue`).
    for lc in &ts.cmds {
        if lc.participants.len() != 1 || lc.cmds.len() != 1 {
            return None;
        }
        let g = &ts.program.commands[lc.global];
        let issued = g
            .issued
            .iter()
            .enumerate()
            .map(|(i, &(_, b, a))| (reference.issued_gens[lc.global][i], b, a))
            .collect();
        out.push(Item {
            global: lc.global,
            cmd: lc.cmds[0].1,
            gen: lc.origin[0].and_then(|o| reference.gens[lc.global][o]),
            issued,
            warp: lc.participants[0],
            position: lc.positions[0],
            initial: reference.initial[lc.global].as_ref()?,
            final_: reference.final_[lc.global].as_ref()?,
        });
    }
    Some(out)
}

pub fn certify(ts: &Ts<'_>, reference: &ReferenceRun, cluster_warps: u32) -> Option<Result<(), CertError>> {
    let items = items(ts, reference)?;
    match ts.program.resources[ts.resources[0]] {
        ResourceId::Named { .. } => named_cert(&items),
        ResourceId::Cluster { .. } => cluster_cert(&items, cluster_warps),
        ResourceId::Mbarrier { .. } => mbarrier_cert(&items),
        _ => None,
    }
}

fn err(kind: &'static str, protocol: &'static str, cmd: usize, related: Vec<usize>, detail: String) -> Option<Result<(), CertError>> {
    Some(Err(CertError { kind, protocol, cmd: Some(cmd), related, detail, incomplete: false }))
}

fn inc(kind: &'static str, protocol: &'static str, cmd: Option<usize>, detail: String) -> Option<Result<(), CertError>> {
    Some(Err(CertError { kind, protocol, cmd, related: Vec::new(), detail, incomplete: true }))
}

/// `verify_named_barrier_causally` (`sync_fixed_unified.rs:1043-1296`).
fn named_cert(items: &[Item<'_>]) -> Option<Result<(), CertError>> {
    const P: &str = "NamedBarrier";
    struct Gen {
        expected: u64,
        arrived: u64,
        release: Clock,
        first: usize,
        members: Vec<usize>,
        warps: BTreeSet<(u32, u8)>,
        red: Option<bool>,
    }
    let mut gens = BTreeMap::<u64, Gen>::new();
    for (i, it) in items.iter().enumerate() {
        let (c, flavor) = match it.cmd {
            SyncCmd::Named(named::Cmd::Arrive(c)) => (c, 0u8),
            SyncCmd::Named(named::Cmd::Sync(c)) => (c, 1),
            SyncCmd::Named(named::Cmd::Red(c)) => (c, 2),
            _ => return None,
        };
        let Some(gen) = it.gen else {
            return inc("named_barrier_unannotated", P, Some(it.global), "no generation".into());
        };
        let e = gens.entry(gen).or_insert_with(|| Gen {
            expected: c.count,
            arrived: 0,
            release: it.initial.clone(),
            first: it.global,
            members: Vec::new(),
            warps: BTreeSet::new(),
            red: None,
        });
        if e.expected != c.count {
            return err("named_barrier_contract_mismatch", P, it.global, vec![e.first], format!("generation {gen} expects {} threads, contribution expects {}", e.expected, c.count));
        }
        if !e.warps.insert((c.warp, flavor)) {
            return err("named_barrier_duplicate_contribution", P, it.global, vec![e.first], format!("warp {} contributes twice to generation {gen}", c.warp));
        }
        let red = flavor == 2;
        if *e.red.get_or_insert(red) != red {
            return err("named_barrier_red_mixed", P, it.global, vec![e.first], format!("generation {gen} mixes .red with sync/arrive"));
        }
        e.arrived += named::WARP_SIZE;
        e.release.join(it.initial);
        e.members.push(i);
    }
    let mut prior: Option<(u64, &Gen)> = None;
    for (&gen, st) in &gens {
        let want = prior.map_or(0, |(g, _)| g + 1);
        if gen != want {
            return inc("named_barrier_generation_gap", P, Some(st.first), format!("skips generation {want} before {gen}"));
        }
        let last = gens.keys().next_back() == Some(&gen);
        if st.arrived > st.expected || (!last && st.arrived != st.expected) {
            return err("named_barrier_arrival_count", P, st.first, Vec::new(), format!("generation {gen} has {} of {} required threads", st.arrived, st.expected));
        }
        if let Some((pg, ps)) = prior {
            for &m in &st.members {
                if !ps.release.leq(items[m].initial) {
                    return err("named_barrier_generation_not_ordered", P, items[m].global, vec![ps.first], format!("contribution to generation {gen} is not happens-before ordered after generation {pg}'s release"));
                }
            }
        }
        prior = Some((gen, st));
    }
    Some(Ok(()))
}

/// `verify_cluster_barrier_causally` (`sync_fixed_unified.rs:1298-1554`).
fn cluster_cert(items: &[Item<'_>], participants: u32) -> Option<Result<(), CertError>> {
    const P: &str = "ClusterBarrier";
    let mut arrivals = BTreeMap::<u64, BTreeMap<u32, usize>>::new();
    let mut waits = BTreeMap::<u64, BTreeMap<u32, usize>>::new();
    let mut last = BTreeMap::<usize, usize>::new();
    for it in items {
        let p = last.entry(it.warp).or_insert(it.position);
        *p = (*p).max(it.position);
    }
    for (i, it) in items.iter().enumerate() {
        // A warp's exit after its last arrival/wait (delta C1) is a no-op in
        // every schedule when every generation below has all participants:
        // the warp's generations already completed, and an exit with nothing
        // arrived completes nothing. Any other exit needs the state machine.
        if let SyncCmd::Cluster(cluster::Cmd::Exit { .. }) = it.cmd {
            if last.get(&it.warp) == Some(&it.position) {
                continue;
            }
            return None;
        }
        let gen = it.gen?;
        match it.cmd {
            SyncCmd::Cluster(cluster::Cmd::Arrive { warp, .. }) => {
                if arrivals.entry(gen).or_default().insert(warp, i).is_some() {
                    return err("cluster_barrier_early_arrival", P, it.global, Vec::new(), format!("duplicate arrival in generation {gen}"));
                }
            }
            SyncCmd::Cluster(cluster::Cmd::Wait { warp, .. }) => {
                // A duplicate wait is a protocol error the state machine
                // reports (`DuplicateWait`); do not overwrite, fall back.
                if waits.entry(gen).or_default().insert(warp, i).is_some() {
                    return None;
                }
            }
            // Exit-aware membership changes need the state machine.
            _ => return None,
        }
    }
    for (want, (&gen, members)) in arrivals.iter().enumerate() {
        if gen != want as u64 {
            return inc("cluster_barrier_generation_gap", P, None, format!("skips generation {want}"));
        }
        // Exit-aware membership (a participant that exits instead of
        // arriving) needs the state machine.
        if members.len() as u32 != participants {
            return None;
        }
        for (warp, &w) in waits.get(&gen).into_iter().flatten() {
            let Some(&a) = members.get(warp) else {
                return err("cluster_barrier_wait_before_arrival", P, items[w].global, Vec::new(), "wait without arrival".into());
            };
            if items[a].warp != items[w].warp || items[a].position >= items[w].position {
                return err("cluster_barrier_wait_before_arrival", P, items[w].global, vec![items[a].global], "wait not program-ordered after arrival".into());
            }
        }
        if gen > 0 {
            for (warp, &a) in members {
                let Some(&pw) = waits.get(&(gen - 1)).and_then(|w| w.get(warp)) else {
                    return inc("cluster_barrier_rearrival_without_wait_unmodeled", P, Some(items[a].global), format!("warp re-arrives for generation {gen} without consuming {}", gen - 1));
                };
                if items[pw].position >= items[a].position {
                    return err("cluster_barrier_early_arrival", P, items[a].global, vec![items[pw].global], format!("arrival for generation {gen} can run before the warp consumes {}", gen - 1));
                }
            }
        }
    }
    Some(Ok(()))
}

/// `verify_mbarrier_causally` (`sync_fixed_unified.rs:1556-2069`), with the
/// terminal-generation fix described in the spec.
fn mbarrier_cert(items: &[Item<'_>]) -> Option<Result<(), CertError>> {
    const P: &str = "Mbarrier";
    let inits = items
        .iter()
        .filter(|it| matches!(it.cmd, SyncCmd::Mbarrier(mbarrier::Cmd::Init { .. })))
        .collect::<Vec<_>>();
    let [init] = inits.as_slice() else { return None };
    let SyncCmd::Mbarrier(mbarrier::Cmd::Init { count: expected, .. }) = init.cmd else { unreachable!() };
    #[derive(Default)]
    struct Gen {
        arrivals: u64,
        tx_expected: u64,
        tx_completed: u64,
        /// (index, requires prior consumption)
        mutations: Vec<(usize, bool)>,
        waits: Vec<usize>,
        tx_issues: Vec<usize>,
    }
    let mut gens = BTreeMap::<u64, Gen>::new();
    let mut vacuous = Vec::<usize>::new();
    for (i, it) in items.iter().enumerate() {
        if it.global == init.global {
            continue;
        }
        let SyncCmd::Mbarrier(cmd) = it.cmd else { return None };
        // Exact state machine needed for these.
        if matches!(
            cmd,
            mbarrier::Cmd::Init { .. } | mbarrier::Cmd::Inval | mbarrier::Cmd::IncPending { .. } | mbarrier::Cmd::TestState { .. }
        ) || matches!(cmd, mbarrier::Cmd::Arrive { drop: true, .. } | mbarrier::Cmd::Arrive { no_complete: true, .. })
        {
            return None;
        }
        if !init.final_.hb(it.initial) {
            return err("mbarrier_init_not_happens_before_use", P, it.global, vec![init.global], format!("{cmd:?} is not happens-before ordered after init"));
        }
        match cmd {
            mbarrier::Cmd::Arrive { count, tx, .. } => {
                let Some(gen) = it.gen else { return inc("mbarrier_unannotated", P, Some(it.global), "no generation".into()) };
                let e = gens.entry(gen).or_default();
                e.arrivals += count;
                e.tx_expected += tx.unwrap_or(0);
                e.mutations.push((i, true));
            }
            mbarrier::Cmd::ExpectTx { bytes } => {
                let Some(gen) = it.gen else { return inc("mbarrier_unannotated", P, Some(it.global), "no generation".into()) };
                let e = gens.entry(gen).or_default();
                e.tx_expected += bytes;
                e.mutations.push((i, true));
            }
            mbarrier::Cmd::Issue => {
                for &(gen, bytes, arrivals) in &it.issued {
                    let Some(gen) = gen else { return inc("mbarrier_unannotated", P, Some(it.global), "no captured generation".into()) };
                    let e = gens.entry(gen).or_default();
                    e.tx_completed += bytes;
                    e.arrivals += arrivals;
                    // Deferred arrivals are arrive-ons: they need consumption.
                    // A transaction-only issue's captured generation is only
                    // fixed when it is ordered after the prior consumption;
                    // otherwise fall back to the state search (below).
                    e.mutations.push((i, arrivals > 0));
                    e.tx_issues.push(i);
                }
            }
            mbarrier::Cmd::WaitParity { .. } | mbarrier::Cmd::TestParity { .. } => match it.gen {
                Some(gen) => gens.entry(gen).or_default().waits.push(i),
                // Vacuous parity-1 success before generation 0 completed.
                None => vacuous.push(i),
            },
            _ => return None,
        }
    }
    for (want, (&gen, st)) in gens.iter().enumerate() {
        if gen != want as u64 {
            return inc("mbarrier_generation_gap", P, None, format!("skips generation {want}"));
        }
        let has_next = gens.contains_key(&(gen + 1));
        let requires = has_next || !st.waits.is_empty();
        let first = st.mutations.first().map_or(init.global, |&(i, _)| items[i].global);
        if st.arrivals > expected
            || (st.arrivals == expected && st.tx_completed > st.tx_expected)
            || (requires && (st.arrivals != expected || st.tx_completed != st.tx_expected))
        {
            return err(
                "mbarrier_generation_count_mismatch",
                P,
                first,
                Vec::new(),
                format!("generation {gen} has {}/{expected} arrivals and {}/{} completed/expected transaction bytes", st.arrivals, st.tx_completed, st.tx_expected),
            );
        }
        if has_next && st.waits.is_empty() {
            return err("mbarrier_prior_generation_not_consumed", P, first, Vec::new(), format!("advances past generation {gen} without a consuming wait"));
        }
        if gen > 0 {
            let prior = &gens[&(gen - 1)];
            for &m in &st.tx_issues {
                if !prior.waits.iter().any(|&w| items[w].final_.hb(items[m].initial)) {
                    return None;
                }
            }
            for &(m, needs) in &st.mutations {
                if needs && !prior.waits.iter().any(|&w| items[w].final_.hb(items[m].initial)) {
                    return err(
                        "mbarrier_prior_generation_consumption_not_happens_before",
                        P,
                        items[m].global,
                        prior.waits.first().map(|&w| items[w].global).into_iter().collect(),
                        format!("generation {gen} mutation can run before generation {} is consumed", gen - 1),
                    );
                }
            }
        }
        // Review S1(b): a wait observed generation `gen` in the reference;
        // unless it is ordered after generation `gen - 1` completed (after a
        // consuming wait of `gen - 1` or a mutation of `gen`), another
        // schedule lets it pass on an older generation of the same parity.
        // Fall back to the exhaustive search for such shapes.
        if gen > 0 {
            let prior = &gens[&(gen - 1)];
            for &w in &st.waits {
                let after_prior = prior.waits.iter().any(|&p| items[p].final_.hb(items[w].initial))
                    || st.mutations.iter().any(|&(m, _)| items[m].initial.hb(items[w].initial));
                if !after_prior {
                    return None;
                }
            }
        }
        if let Some(next) = gens.get(&(gen + 1)).filter(|n| n.arrivals == expected && n.tx_completed == n.tx_expected) {
            // An overtaken wait either blocks forever or passes on a later
            // generation of the same parity; which one is a property of the
            // whole schedule space, so let the search decide (review S1).
            for &w in &st.waits {
                if !next.mutations.iter().any(|&(m, _)| items[w].initial.hb(items[m].initial)) {
                    return None;
                }
            }
        }
    }
    // Review S1(a): a vacuous parity-1 success must be ordered before some
    // prerequisite of generation 0's completion; otherwise generation 0 can
    // complete first and the wait blocks until generation 1. Fall back.
    if let Some(first) = gens.get(&0).filter(|g| g.arrivals == expected && g.tx_completed == g.tx_expected) {
        for &w in &vacuous {
            if !first.mutations.iter().any(|&(m, _)| items[w].initial.hb(items[m].initial)) {
                return None;
            }
        }
    }
    Some(Ok(()))
}
