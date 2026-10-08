//! Reductions must not change results: every configuration is compared with
//! the exhaustive whole-program search on random contract logs.
//!
//! The oracle explores every schedule and collects *all* failure kinds. A
//! variant must reach the same verdict, and (for the search-based variants)
//! its finding kind must be one the oracle found. The only tolerated
//! difference is a fail-closed `incomplete` from a gated projection whose
//! schedule left the reference generation assignment (review S5).
//!
//! The rich generator emits the shapes review S1/S8 named: first waits at
//! parity 1, waits that skip a generation, TMA issues without an explicit
//! `Issue` command, conditional waits, inval/re-init, multi-target waits,
//! tcgen05 alloc/dealloc and commit, and cluster barriers.

mod synccheck_support;

use std::collections::BTreeSet;

use numsim_core::observe::{AsyncTarget, ProtocolStatus, RecordingObserver};
use numsim_core::report::{Report, Status, Verdict};
use numsim_core::sync::{tcgen, ResourceInit};
use numsim_core::synccheck::build::*;
use numsim_core::synccheck::explore::Options;
use numsim_core::synccheck::{check, serialize, ProjectionMode, SynccheckConfig};
use synccheck_support::*;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 32) as u32
    }
    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }
    fn chance(&mut self, num: u32, den: u32) -> bool {
        self.below(den) < num
    }
}

fn random_log(rng: &mut Rng, warps: u32) -> RecordingObserver {
    let bars = 1 + rng.below(2);
    let mut log = LogBuilder::new();
    for b in 0..bars {
        log.cmd(0, 1, mbar(0, 8 * b), init(1 + u64::from(rng.below(4) == 0)));
    }
    if rng.below(4) != 0 {
        cta_sync(&mut log, 0, &(0..warps).collect::<Vec<_>>(), warps);
    }
    for w in 0..warps {
        let mut parity = [0u64; 2];
        for _ in 0..1 + rng.below(4) {
            let b = rng.below(bars);
            match rng.below(5) {
                0 | 1 => {
                    log.cmd(w, 3, mbar(0, 8 * b), arrive(1));
                }
                2 | 3 => {
                    log.cmd(w, 4, mbar(0, 8 * b), wait(parity[b as usize]));
                    parity[b as usize] ^= 1;
                }
                _ => {
                    log.cmd(w, 5, named_bar(0, 1), bar_sync(w, 64));
                }
            }
        }
    }
    log.build()
}

/// Per barrier one producer and one consumer with an optional named-barrier
/// hand-shake, randomly interleaved per warp.
fn structured_log(rng: &mut Rng, warps: u32) -> RecordingObserver {
    structured(rng, warps, false)
}

/// `structured_log` whose producers use TMA (`arrive.expect_tx` + an event
/// with only an issued target) or `tcgen05.commit` arrivals and whose
/// consumers sometimes use a recorded successful `try_wait`.
fn rich_structured_log(rng: &mut Rng, warps: u32) -> RecordingObserver {
    structured(rng, warps, true)
}

enum Step {
    Cmd(numsim_core::sync::ResourceId, numsim_core::sync::SyncCmd),
    Tma(numsim_core::sync::ResourceId),
    Commit(numsim_core::sync::ResourceId),
    Try(numsim_core::sync::ResourceId, u8),
}

fn structured(rng: &mut Rng, warps: u32, rich: bool) -> RecordingObserver {
    let bars = 1 + rng.below(2);
    let mut per_warp: Vec<Vec<Vec<Step>>> = (0..warps).map(|_| Vec::new()).collect();
    for b in 0..bars {
        let producer = rng.below(warps);
        let consumer = (producer + 1 + rng.below(warps - 1)) % warps;
        let rounds = 1 + rng.below(3);
        let handshake = rng.below(2) == 0;
        let (mut p, mut c) = (Vec::new(), Vec::new());
        let style = if rich { rng.below(3) } else { 0 };
        for round in 0..rounds {
            match style {
                1 => {
                    p.push(Step::Cmd(mbar(0, 8 * b), arrive_tx(1, 64)));
                    p.push(Step::Tma(mbar(0, 8 * b)));
                }
                2 => p.push(Step::Commit(mbar(0, 8 * b))),
                _ => p.push(Step::Cmd(mbar(0, 8 * b), arrive(1))),
            }
            if rich && rng.chance(1, 3) {
                c.push(Step::Try(mbar(0, 8 * b), (round & 1) as u8));
            } else {
                c.push(Step::Cmd(mbar(0, 8 * b), wait(u64::from(round & 1))));
            }
            if handshake && round + 1 < rounds {
                p.push(Step::Cmd(named_bar(0, 1 + b as u8), bar_sync(producer, 64)));
                c.push(Step::Cmd(named_bar(0, 1 + b as u8), bar_sync(consumer, 64)));
            }
        }
        per_warp[producer as usize].push(p);
        per_warp[consumer as usize].push(c);
    }
    let mut log = LogBuilder::new();
    for b in 0..bars {
        log.cmd(0, 1, mbar(0, 8 * b), init(1));
    }
    if rng.below(5) != 0 {
        cta_sync(&mut log, 0, &(0..warps).collect::<Vec<_>>(), warps);
    }
    for (w, lists) in per_warp.into_iter().enumerate() {
        let mut cursors = vec![0usize; lists.len()];
        loop {
            let live = (0..lists.len()).filter(|&i| cursors[i] < lists[i].len()).collect::<Vec<_>>();
            if live.is_empty() {
                break;
            }
            let pick = live[rng.below(live.len() as u32) as usize];
            let site = 10 + pick as u32;
            match lists[pick][cursors[pick]] {
                Step::Cmd(res, cmd) => {
                    log.cmd(w as u32, site, res, cmd);
                }
                Step::Tma(res) => {
                    log.event(w as u32, site, Vec::new(), vec![AsyncTarget { res, bytes: 64, arrivals: 0 }], None, None, ProtocolStatus::Committed);
                }
                Step::Commit(res) => {
                    log.issue(w as u32, site, res, 0, 1, vec![(tcgen_work(w as u32, 0), work(tcgen::WorkCmd::Commit))]);
                }
                Step::Try(res, parity) => {
                    log.test_ok(w as u32, site, res, parity);
                }
            }
            cursors[pick] += 1;
        }
    }
    log.build()
}

/// Everything the explorer models, in small random doses.
fn rich_log(rng: &mut Rng, warps: u32) -> RecordingObserver {
    let bars = 1 + rng.below(2);
    let m = |b: u32| mbar(0, 8 * b);
    let mut log = LogBuilder::new();
    let tma = rng.chance(1, 3);
    for b in 0..bars {
        log.cmd(0, 1, m(b), init(1 + u64::from(rng.chance(1, 4))));
    }
    if rng.chance(3, 4) {
        cta_sync(&mut log, 0, &(0..warps).collect::<Vec<_>>(), warps);
    }
    for w in 0..warps {
        let mut parity = [0u64; 2];
        for _ in 0..1 + rng.below(4) {
            let b = rng.below(bars);
            match rng.below(12) {
                0 | 1 if tma => {
                    // arrive.expect_tx then a TMA that lists only its issued target.
                    log.cmd(w, 20, m(b), arrive_tx(1, 64));
                    log.event(w, 21, Vec::new(), vec![AsyncTarget { res: m(b), bytes: 64, arrivals: 0 }], None, None, ProtocolStatus::Committed);
                }
                0 | 1 | 2 => {
                    log.cmd(w, 22, m(b), arrive(1));
                }
                3 | 4 => {
                    log.cmd(w, 23, m(b), wait(parity[b as usize]));
                    parity[b as usize] ^= 1;
                }
                5 => {
                    // A first wait at parity 1 or a wait that skips a generation.
                    log.cmd(w, 24, m(b), wait(parity[b as usize] ^ 1));
                }
                6 => {
                    log.test_ok(w, 25, m(b), parity[b as usize] as u8);
                    parity[b as usize] ^= 1;
                }
                7 if bars == 2 => {
                    log.cmds(w, 26, vec![(m(0), wait(parity[0])), (m(1), wait(parity[1]))]);
                    parity = [parity[0] ^ 1, parity[1] ^ 1];
                }
                8 => {
                    log.cmd(w, 27, m(b), inval());
                    log.cmd(w, 28, m(b), init(1));
                    parity[b as usize] = 0;
                }
                9 => {
                    // tcgen05.commit arriving on the barrier (deferred arrive-on).
                    log.issue(w, 29, m(b), 0, 1, vec![(tcgen_work(w, 0), work(tcgen::WorkCmd::Commit))]);
                }
                10 => {
                    log.cmd(w, 30, tmem(0), tmem_alloc(256));
                    log.cmd(w, 31, tmem(0), tmem_dealloc(0, 256));
                }
                _ => {
                    log.cmd(w, 32, cluster_bar(0), cl_arrive(w));
                    log.cmd(w, 33, cluster_bar(0), cl_wait(w));
                }
            }
        }
    }
    log.build()
}

/// Phase laps without back-pressure on one barrier: low-numbered waiters,
/// high-numbered repeat arrivers, optional early (arming) waiters. This is
/// the shape where a one-step strong diamond picks a waiter first and misses
/// the schedule in which the arrivers complete the next phase (review S8).
fn lap_log(rng: &mut Rng, warps: u32) -> RecordingObserver {
    let warps = warps.max(3);
    let mut log = LogBuilder::new();
    let expected = 1 + u64::from(rng.chance(1, 2));
    log.cmd(0, 1, mbar(0, 0), init(expected));
    cta_sync(&mut log, 0, &(0..warps).collect::<Vec<_>>(), warps);
    let waiters = 1 + rng.below(warps - 1);
    for w in 0..warps {
        if w < waiters {
            log.cmd(w, 2, mbar(0, 0), wait(0));
            if rng.chance(1, 3) {
                log.cmd(w, 3, mbar(0, 0), wait(1));
            }
        } else {
            for _ in 0..1 + rng.below(3) {
                log.cmd(w, 4, mbar(0, 0), arrive(1));
            }
        }
    }
    log.build()
}

/// Finding kinds of a report (payload `source_kind` for protocol errors).
fn kinds(r: &Report) -> BTreeSet<String> {
    let p = serialize(r);
    let mut out = BTreeSet::new();
    for f in p["findings"].as_array().into_iter().flatten().chain(std::iter::once(&p["execution_error"])) {
        if f.is_null() {
            continue;
        }
        let k = if f["kind"] == "fixed_sync_protocol_error" { &f["source_kind"] } else { &f["kind"] };
        out.insert(k.as_str().unwrap_or("").to_owned());
    }
    out
}

fn fail_closed(r: &Report) -> bool {
    r.verdict == Verdict::Incomplete
        && r.findings.iter().any(|f| {
            f.status == Status::Incomplete
                && f.attrs.get("source").and_then(|s| s.as_str()).is_some_and(|s| s.starts_with("generation_assignment_differs"))
        })
}

fn compare(seed: u64, cases: u32, gen: fn(&mut Rng, u32) -> RecordingObserver) {
    compare_with(seed, cases, gen, 20)
}

/// `max_closed_percent`: tolerated fail-closed (`generation_assignment_differs`)
/// variant results, in percent of the cases.
fn compare_with(seed: u64, cases: u32, gen: fn(&mut Rng, u32) -> RecordingObserver, max_closed_percent: usize) {
    let mut rng = Rng(seed);
    let mut counts = [0usize; 4];
    let mut closed = 0usize;
    for case in 0..cases {
        let warps = 2 + rng.below(3);
        let init = ResourceInit { cluster_warps: warps, ..cta(warps) };
        let log = gen(&mut rng, warps);
        let oracle = SynccheckConfig {
            mode: ProjectionMode::Whole,
            certificates: false,
            fingerprints: false,
            explore: Options { stop_on_first_failure: false, ..Options::NONE },
            state_budget: 200_000,
            ..config(init)
        };
        let truth = check(&log, &oracle);
        if truth.verdict == Verdict::Incomplete {
            continue;
        }
        counts[truth.verdict as usize] += 1;
        let truth_kinds = kinds(&truth);
        let first = SynccheckConfig { explore: Options { stop_on_first_failure: true, ..Options::NONE }, ..oracle.clone() };
        let variants = [
            ("whole+all", SynccheckConfig { explore: Options::ALL, ..first.clone() }, true),
            ("components+all", SynccheckConfig { mode: ProjectionMode::Components, explore: Options::ALL, ..first.clone() }, true),
            ("per-resource+none", SynccheckConfig { mode: ProjectionMode::PerResource, ..first.clone() }, true),
            ("per-resource+all", SynccheckConfig { mode: ProjectionMode::PerResource, explore: Options::ALL, ..first.clone() }, true),
            ("default", SynccheckConfig { state_budget: 200_000, ..config(init) }, false),
        ];
        for (name, cfg, same_kinds) in variants {
            let got = check(&log, &cfg);
            if got.verdict != truth.verdict {
                if fail_closed(&got) && truth.verdict != Verdict::Clean {
                    closed += 1;
                    continue;
                }
                panic!("case {case} {name}: {:?} != {:?}\n{log:#?}\noracle {:#}\n{name} {:#}", got.verdict, truth.verdict, serialize(&truth), serialize(&got));
            }
            if same_kinds && got.verdict == Verdict::Error {
                let k = kinds(&got);
                assert!(k.is_subset(&truth_kinds), "case {case} {name}: kinds {k:?} not among oracle {truth_kinds:?}\n{log:#?}");
            }
        }
    }
    eprintln!("verdicts [clean, review, incomplete, error] = {counts:?}; fail-closed variants: {closed}");
    assert!(counts[Verdict::Clean as usize] * 100 > cases as usize, "{counts:?}");
    assert!(counts[Verdict::Error as usize] * 100 > cases as usize, "{counts:?}");
    assert!(closed * 100 <= cases as usize * max_closed_percent, "too many fail-closed results: {closed}");
}

#[test]
fn reductions_preserve_results_on_random_logs() {
    compare(0x9e37_79b9_7f4a_7c15, 400, random_log);
}

#[test]
fn reductions_preserve_results_on_structured_logs() {
    compare(0x2545_f491_4f6c_dd1d, 400, structured_log);
}

#[test]
fn reductions_preserve_results_on_rich_logs() {
    compare(0x5851_f42d_4c95_7f2d, 800, rich_log);
}

#[test]
fn reductions_preserve_results_on_rich_structured_logs() {
    compare(0x1405_7b7e_f767_814f, 600, rich_structured_log);
}

#[test]
fn reductions_preserve_results_on_lap_logs() {
    // Lap-prone programs often leave the reference generation assignment in
    // gated projections, which fails closed by design.
    compare_with(0x0bad_cafe_f00d_d00d, 600, |rng, w| lap_log(rng, w + 1), 40);
}
