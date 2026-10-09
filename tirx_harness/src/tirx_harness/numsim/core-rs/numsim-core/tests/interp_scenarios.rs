//! Interpreter + scheduler semantics on hand-built programs
//! (`testutil::scenarios`).

use numsim_core::arena::ValidityPolicy;
use numsim_core::interp::ExecErrorKind;
use numsim_core::oplib::OpErrorKind;
use numsim_core::observe::{Observer, RecordingObserver, SyncEvent, SyncKind};
use numsim_core::report::{FindingKind, Status};
use numsim_core::sched::{self, CompletionPolicy, RunConfig, RunOutcome, RunStatus};
use numsim_core::sync::ResourceId;
use numsim_core::testutil::scenarios::{self, Scenario};

fn run_cfg(s: &Scenario, config: &RunConfig) -> RunOutcome {
    let mut obs = RecordingObserver::new();
    sched::run_with_config(&s.module, &s.inputs, &mut obs, config).expect("run starts")
}

fn run(s: &Scenario) -> RunOutcome {
    run_cfg(s, &s.config)
}

fn u32s(o: &RunOutcome, name: &str) -> Vec<u32> {
    o.outputs.buffers[name].0.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn f32s(o: &RunOutcome, name: &str) -> Vec<f32> {
    o.outputs.buffers[name].0.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn completed(o: &RunOutcome) {
    assert_eq!(o.status, RunStatus::Completed, "{:?}", o.status);
}

#[test]
fn vector_add() {
    for seed in [0, 1, 7] {
        let s = scenarios::vector_add();
        let o = run_cfg(&s, &RunConfig { seed, ..s.config.clone() });
        completed(&o);
        let c = f32s(&o, "c");
        for (i, v) in c.iter().enumerate() {
            let want = if (i as u32) < scenarios::VADD_N { i as f32 * 0.5 + 2.0 } else { -1.0 };
            assert_eq!(*v, want, "c[{i}]");
        }
        assert!(o.diagnostics.is_empty());
        assert!(o.sync_leftovers.is_empty());
    }
}

#[test]
fn divergent_if_else_reconverges() {
    let o = run(&scenarios::divergent_if_else());
    completed(&o);
    let out = u32s(&o, "out");
    for t in 0..64 {
        assert_eq!(out[t as usize], scenarios::divergent_expected(t), "tid {t}");
    }
}

#[test]
fn nested_loops_break_continue() {
    let o = run(&scenarios::nested_loops());
    completed(&o);
    let out = u32s(&o, "out");
    for l in 0..32 {
        assert_eq!(out[l as usize], scenarios::nested_loops_expected(l), "lane {l}");
    }
}

#[test]
fn nested_loops_small_quantum() {
    // Resumption at arbitrary pcs must not change results.
    for q in [1, 3, 7] {
        let s = scenarios::nested_loops();
        let o = run_cfg(&s, &RunConfig { quantum: q, ..s.config.clone() });
        completed(&o);
        let out = u32s(&o, "out");
        for l in 0..32 {
            assert_eq!(out[l as usize], scenarios::nested_loops_expected(l));
        }
    }
}

#[test]
fn mbarrier_producer_consumer_expect_tx_bulk() {
    for completions in [CompletionPolicy::Eager, CompletionPolicy::Seeded] {
        for seed in 0..4 {
            let s = scenarios::mbarrier_producer_consumer();
            let o = run_cfg(&s, &RunConfig { seed, completions, ..s.config.clone() });
            completed(&o);
            let out = u32s(&o, "out");
            assert_eq!(out, (0..32).map(|x| x * 7 + 3).collect::<Vec<_>>());
            assert!(o.sync_leftovers.is_empty(), "{:?}", o.sync_leftovers);
        }
    }
}

#[test]
fn named_barrier_arrive_sync_red() {
    let o = run(&scenarios::named_barrier());
    completed(&o);
    assert_eq!(u32s(&o, "out"), (0..64).map(|t| t * 3).collect::<Vec<_>>());
    assert_eq!(u32s(&o, "cnt"), vec![50; 128]);
}

#[test]
fn elect_gated_region() {
    let o = run(&scenarios::elect_region());
    completed(&o);
    assert_eq!(u32s(&o, "counter"), vec![4]);
    assert_eq!(u32s(&o, "out"), vec![0; 128]);
}

#[test]
fn loop_budget_is_incomplete() {
    let s = scenarios::loop_budget();
    let o = run(&s);
    match &o.status {
        RunStatus::Incomplete { reason, site, .. } => {
            assert!(reason.contains("Budget"), "{reason}");
            let site = site.expect("site");
            assert_eq!(s.module.kernels[0].sites[site.0 as usize].op_name, "spin");
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(o.failed_kernel, Some(0));
}

#[test]
fn deadlock_detected() {
    let o = run(&scenarios::deadlock_wait());
    match &o.status {
        RunStatus::Deadlock { blocked } => {
            assert_eq!(blocked.len(), 1);
            assert!(matches!(blocked[0].1, ResourceId::Mbarrier { .. }));
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn spin_loop_is_parked_into_deadlock() {
    let o = run(&scenarios::deadlock_spin());
    assert!(matches!(o.status, RunStatus::Deadlock { .. }), "{:?}", o.status);
    // Parking: a handful of rounds, not 2^40 iterations.
    assert!(o.stats.rounds < 100, "rounds {}", o.stats.rounds);
}

#[test]
fn uninit_read_policies() {
    let s = scenarios::uninit_read();
    let o = run_cfg(&s, &RunConfig { validity: ValidityPolicy::Error, ..s.config.clone() });
    match &o.status {
        RunStatus::Error(e) => assert_eq!(e.kind, ExecErrorKind::Uninit),
        other => panic!("{other:?}"),
    }
    let o = run_cfg(&s, &RunConfig { validity: ValidityPolicy::ZeroAndReport, ..s.config.clone() });
    completed(&o);
    assert_eq!(u32s(&o, "out"), vec![0; 32]);
    assert_eq!(o.diagnostics.len(), 1, "{:?}", o.diagnostics);
    let f = &o.diagnostics[0];
    assert_eq!((f.kind.clone(), f.status), (FindingKind::UninitRead, Status::Review));
    assert_eq!(f.evidence[0].bytes.map(|b| (b.start, b.len)), Some((0, 128)));
    let o = run_cfg(&s, &RunConfig { validity: ValidityPolicy::Allow, ..s.config.clone() });
    completed(&o);
    assert!(o.diagnostics.is_empty());
}

#[test]
fn out_of_bounds_is_an_error() {
    let o = run(&scenarios::oob());
    match &o.status {
        RunStatus::Error(e) => {
            assert_eq!(e.kind, ExecErrorKind::OutOfBounds);
            assert_eq!(e.lanes.first(), Some(16));
            assert_eq!(e.kernel, 0);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn divergent_wait_arrive_interleaves_arms() {
    let o = run(&scenarios::divergent_wait_arrive());
    completed(&o);
    let mut want = vec![2u32; 32];
    want[0] = 1;
    assert_eq!(u32s(&o, "out"), want);
}

#[test]
fn cp_async_groups() {
    for seed in 0..3 {
        let s = scenarios::cp_async_copy();
        let o = run_cfg(&s, &RunConfig { seed, ..s.config.clone() });
        completed(&o);
        assert_eq!(u32s(&o, "out"), (0..128).map(|x| x ^ 0x55).collect::<Vec<_>>());
    }
}

/// Records sync events and asks for declared-word histories.
#[derive(Default)]
struct History(Vec<SyncEvent>);

impl Observer for History {
    fn wants_word_history(&self) -> bool {
        true
    }
    fn sync(&mut self, e: &SyncEvent) {
        self.0.push(e.clone());
    }
}

#[test]
fn wait_until_and_verdicts() {
    let s = scenarios::wait_until_flag();
    let mut h = History::default();
    let o = sched::run_with_config(&s.module, &s.inputs, &mut h, &s.config).unwrap();
    completed(&o);
    assert_eq!(u32s(&o, "out"), vec![43; 32]);
    let v: Vec<_> = h
        .0
        .iter()
        .filter_map(|e| match &e.kind {
            SyncKind::WaitVerdicts { verdicts, .. } => Some(verdicts.clone()),
            _ => None,
        })
        .collect();
    assert!(!v.is_empty());
    for vs in &v {
        for lv in vs {
            // History: [0 (launch value), 1 (publish)]; only entry 1 accepts.
            assert_eq!(lv.accepted, vec![0b10]);
            assert_eq!(lv.observed, 1);
        }
    }
    assert!(h.0.iter().any(|e| matches!(e.kind, SyncKind::DeclareWord { .. })));
}

#[test]
fn deterministic_for_fixed_seed() {
    for s in [scenarios::mbarrier_producer_consumer(), scenarios::named_barrier(), scenarios::cp_async_copy()] {
        let log = |seed| {
            let mut obs = RecordingObserver::new();
            let o = sched::run_with_config(&s.module, &s.inputs, &mut obs, &RunConfig { seed, ..s.config.clone() }).unwrap();
            (o, obs)
        };
        let (a, la) = log(5);
        let (b, lb) = log(5);
        assert_eq!(a, b);
        assert_eq!(la, lb, "{}", s.name);
    }
}

/// Every observer callback, in order, as text (Access seq included).
#[derive(Default)]
struct Trace(Vec<String>, bool);

impl numsim_core::observe::Observer for Trace {
    fn wants_word_history(&self) -> bool {
        self.1
    }
    fn begin_launch(&mut self, i: &numsim_core::observe::LaunchInfo<'_>) {
        self.0.push(format!("begin {}", i.kernel_index));
    }
    fn end_launch(&mut self, i: &numsim_core::observe::LaunchInfo<'_>) {
        self.0.push(format!("end {}", i.kernel_index));
    }
    fn access(&mut self, a: &numsim_core::observe::Access<'_>) {
        self.0.push(format!("{a:?}"));
    }
    fn sync(&mut self, e: &SyncEvent) {
        self.0.push(format!("{e:?}"));
    }
    fn warp_done(&mut self, w: numsim_core::observe::WarpId, end: numsim_core::observe::WarpEnd) {
        self.0.push(format!("done {w:?} {end:?}"));
    }
    fn round_boundary(&mut self, c: numsim_core::observe::CtaId, r: u64) {
        self.0.push(format!("drain {c:?} {r}"));
    }
}

#[test]
fn workers_do_not_change_results_or_streams() {
    for s in scenarios::all() {
        for history in [false, true] {
            let mut t1 = Trace(Vec::new(), history);
            let a = sched::run_with_config(&s.module, &s.inputs, &mut t1, &s.config).unwrap();
            for workers in [2, 8, 33] {
                let mut tn = Trace(Vec::new(), history);
                let cfg = RunConfig { workers, ..s.config.clone() };
                let b = sched::run_with_config(&s.module, &s.inputs, &mut tn, &cfg).unwrap();
                assert_eq!(a, b, "{} workers={workers}", s.name);
                assert!(t1.0 == tn.0, "{} workers={workers}: observer streams differ", s.name);
            }
        }
        // NumSim (no observer) gives the same outputs as observed runs.
        let mut o = numsim_core::observe::NoopObserver;
        let c = sched::run_with_config(&s.module, &s.inputs, &mut o, &RunConfig { workers: 8, ..s.config.clone() }).unwrap();
        let mut t = Trace(Vec::new(), true);
        let d = sched::run_with_config(&s.module, &s.inputs, &mut t, &s.config).unwrap();
        assert_eq!(c.outputs, d.outputs, "{}: observer changed outputs", s.name);
        assert_eq!(c.status, d.status, "{}", s.name);
    }
}

#[test]
fn moe_synthetic_partitions_and_serial_atomics() {
    let (ctas, iters) = (24, 4);
    let s = scenarios::moe_synthetic(ctas, iters);
    let (want_out, want_counts) = scenarios::moe_expected(ctas, iters);
    for workers in [1, 4, 16] {
        let o = run_cfg(&s, &RunConfig { workers, ..s.config.clone() });
        completed(&o);
        assert_eq!(u32s(&o, "out"), want_out, "workers={workers}");
        assert_eq!(u32s(&o, "counts"), want_counts, "workers={workers}");
    }
}

#[test]
fn tcgen_alloc_st_ld_dealloc() {
    let o = run(&scenarios::tcgen_ld_st());
    completed(&o);
    assert_eq!(u32s(&o, "out"), (0..128).collect::<Vec<_>>());
    assert!(o.sync_leftovers.is_empty(), "{:?}", o.sync_leftovers);
}

#[test]
fn tma_tile_loads_with_oob_fill() {
    for seed in 0..3 {
        let s = scenarios::tma_load();
        let o = run_cfg(&s, &RunConfig { seed, ..s.config.clone() });
        completed(&o);
        let out = f32s(&o, "out");
        let cols = scenarios::TMA_COLS;
        for (bi, row0) in [2u32, 6].into_iter().enumerate() {
            for r in 0..scenarios::TMA_BOX_ROWS {
                for c in 0..cols {
                    let row = row0 + r;
                    let want = if row < scenarios::TMA_ROWS { (row * 100 + c) as f32 } else { 0.0 };
                    let i = (bi as u32 * scenarios::TMA_BOX_ROWS * cols + r * cols + c) as usize;
                    assert_eq!(out[i], want, "box {bi} row {row} col {c}");
                }
            }
        }
    }
}

#[test]
fn copy_report_bits() {
    let o = run(&scenarios::copy_report());
    completed(&o);
    assert_eq!(u32s(&o, "out"), vec![0, 1]);
}

#[test]
fn syncwarp_across_divergent_arms() {
    let o = run(&scenarios::divergent_syncwarp());
    completed(&o);
    assert_eq!(u32s(&o, "out"), (0..32u32).map(|l| (l ^ 16) * 10).collect::<Vec<_>>());
}

fn u16s(o: &RunOutcome, name: &str) -> Vec<u16> {
    o.outputs.buffers[name].0.chunks(2).map(|c| u16::from_le_bytes(c.try_into().unwrap())).collect()
}

#[test]
fn ldmatrix_stmatrix_m8n8() {
    let o = run(&scenarios::matrix_roundtrip());
    completed(&o);
    let m: Vec<u16> = (0..256).map(|x| (x * 7 + 1) as u16).collect();
    // Element (matrix i, row r, col c) = m[64 i + 8 r + c] (PTX m8n8.b16).
    let el = |i: usize, r: usize, c: usize| m[64 * i + 8 * r + c] as u32;
    let ld = u32s(&o, "out_ld");
    let ldt = u32s(&o, "out_ldt");
    for t in 0..32 {
        for i in 0..4 {
            let (r, c) = (t / 4, 2 * (t % 4));
            assert_eq!(ld[t * 4 + i], el(i, r, c) | el(i, r, c + 1) << 16, "ld lane {t} matrix {i}");
            assert_eq!(ldt[t * 4 + i], el(i, c, r) | el(i, c + 1, r) << 16, "ld.trans lane {t} matrix {i}");
        }
    }
    assert_eq!(u16s(&o, "out_st"), m, "stmatrix round trip");
}

fn tcgen_cp_expected() -> Vec<u32> {
    let plan = numsim_core::oplib::tcgen_cp_plan(128, 256, 0, 0, scenarios::TCGEN_CP_SDESC, 0, 1, numsim_core::oplib::TcArch::Sm100).unwrap();
    let (srcs, cells) = plan.pairs();
    let input: Vec<u32> = (0..1024).map(|x| x * 3 + 7).collect();
    let mut want = vec![0u32; 1024];
    for (s, &(lane, col)) in srcs.iter().zip(&cells) {
        want[lane as usize * 8 + col as usize] = input[(s.start / 4) as usize];
    }
    want
}

/// A second commit with nothing new issued still tracks the in-flight copy
/// (found by gdn_prefill_sm100 under seeded latency).
#[test]
fn second_commit_tracks_in_flight_tcgen_ops() {
    let want = tcgen_cp_expected();
    for seed in 0..12 {
        let s = scenarios::tcgen_cp_ld_with(true);
        let o = run_cfg(&s, &RunConfig { seed, ..s.config.clone() });
        completed(&o);
        assert_eq!(u32s(&o, "out"), want, "seed {seed}");
    }
}

#[test]
fn tcgen_cp_then_ld() {
    let o = run(&scenarios::tcgen_cp_ld());
    completed(&o);
    // Expected: oplib's cp plan applied to the source block.
    let plan = numsim_core::oplib::tcgen_cp_plan(128, 256, 0, 0, scenarios::TCGEN_CP_SDESC, 0, 1, numsim_core::oplib::TcArch::Sm100).unwrap();
    let (srcs, cells) = plan.pairs();
    let input: Vec<u32> = (0..1024).map(|x| x * 3 + 7).collect();
    let mut want = vec![0u32; 1024];
    for (s, &(lane, col)) in srcs.iter().zip(&cells) {
        assert_eq!(s.len, 4);
        assert!(col < 8 && lane < 128);
        want[lane as usize * 8 + col as usize] = input[(s.start / 4) as usize];
    }
    assert_eq!(u32s(&o, "out"), want);
}

/// Records accesses (owned) and sync events.
#[derive(Default)]
struct Events {
    accesses: Vec<(numsim_core::observe::AccessKind, bool, numsim_core::arena::Space, Vec<numsim_core::observe::LaneSpan>)>,
    syncs: Vec<SyncEvent>,
}

impl Observer for Events {
    fn access(&mut self, a: &numsim_core::observe::Access<'_>) {
        self.accesses.push((a.kind, a.atomic, a.space, a.spans.to_vec()));
    }
    fn sync(&mut self, e: &SyncEvent) {
        self.syncs.push(e.clone());
    }
}

fn run_events(s: &Scenario) -> (RunOutcome, Events) {
    let mut ev = Events::default();
    let o = sched::run_with_config(&s.module, &s.inputs, &mut ev, &s.config).unwrap();
    (o, ev)
}

#[test]
fn bulk_copy_mask_and_ignore_oob_narrow_bytes_and_footprint() {
    let s = scenarios::bulk_masked_copy();
    let (o, ev) = run_events(&s);
    completed(&o);
    assert_eq!(o.outputs.buffers["out"].0, scenarios::bulk_masked_expected());
    // The async write side covers exactly the masked-in bytes: transferred
    // ones plus the zeroed `.ignore_oob` edges (legacy writes the window).
    let want: Vec<u64> = (0..64u64).filter(|i| i % 16 < 8).collect();
    let mut got: Vec<u64> = ev
        .accesses
        .iter()
        .filter(|(k, _, sp, _)| *k == numsim_core::observe::AccessKind::Write && *sp == numsim_core::arena::Space::Shared)
        .flat_map(|(_, _, _, spans)| spans.iter().flat_map(|s| s.span.start..s.span.end()))
        .filter(|&b| b >= 16) // the block follows the 8-byte mbarrier at a 16-byte boundary
        .map(|b| b - 16)
        .collect();
    got.sort();
    got.dedup();
    // Warp stores of the 0xEE prefill also write the block; keep async bytes only.
    let async_bytes: Vec<u64> = got.into_iter().filter(|b| want.contains(b)).collect();
    assert_eq!(async_bytes, want);
}

#[test]
fn bulk_reductions_are_atomic_and_serialized() {
    for workers in [1, 8] {
        let s = scenarios::bulk_reduce(6);
        let cfg = RunConfig { workers, ..s.config.clone() };
        let mut ev = Events::default();
        let o = sched::run_with_config(&s.module, &s.inputs, &mut ev, &cfg).unwrap();
        completed(&o);
        let want: Vec<u32> = (0..32u32).map(|i| i * 7 + (0..6u32).map(|c| c * 1000 + i).sum::<u32>()).collect();
        assert_eq!(u32s(&o, "acc"), want, "workers={workers}");
        let rmw: Vec<_> = ev.accesses.iter().filter(|(k, _, sp, _)| *k == numsim_core::observe::AccessKind::Rmw && *sp == numsim_core::arena::Space::Global).collect();
        assert_eq!(rmw.len(), 6);
        for (_, atomic, _, spans) in rmw {
            assert!(*atomic);
            assert_eq!(spans.len(), 32);
            assert!(spans.iter().all(|s| s.span.len == 4));
        }
    }
}

#[test]
fn broken_lanes_join_a_later_barrier() {
    let o = run(&scenarios::break_then_barrier());
    completed(&o);
    assert_eq!(u32s(&o, "out"), vec![5; 64]);
}

#[test]
fn launch_bounds_setmaxnreg_is_logged_and_completes() {
    let s = scenarios::setmaxnreg_launch_bounds();
    let (o, ev) = run_events(&s);
    completed(&o);
    assert_eq!(u32s(&o, "out"), (0..256u32).map(|t| t / 128).collect::<Vec<_>>());
    let configure = ev.syncs.iter().any(|e| matches!(&e.kind, SyncKind::Protocol { cmds, .. }
        if cmds.iter().any(|c| matches!(c.cmd, numsim_core::sync::SyncCmd::RegPool(numsim_core::sync::setmaxnreg::Cmd::Configure { count: 128 })))));
    assert!(configure, "the launch-bounds Configure is logged");
}

#[test]
fn tcgen_alloc_orders_the_warp() {
    let (o, ev) = run_events(&scenarios::tcgen_ld_st());
    completed(&o);
    let site_of = |e: &SyncEvent| e.site;
    let alloc_site = scenarios::tcgen_ld_st().module.kernels[0].sites.iter().position(|s| s.op_name == "tcgen_alloc").unwrap();
    assert!(ev.syncs.iter().any(|e| site_of(e).0 as usize == alloc_site && matches!(e.kind, SyncKind::WarpSync { .. })));
}

// ---------------------------------------------------------------------------
// Engine-review regressions (docs/development/engine-review.md)
// ---------------------------------------------------------------------------

fn all_events(log: &RecordingObserver) -> Vec<&SyncEvent> {
    log.per_warp.iter().flatten().chain(log.other.iter()).collect()
}

/// H1: exited warps leave count-less named barriers; an explicit-count
/// barrier left waiting on exited warps is `incomplete` (G8), never Deadlock.
#[test]
fn exited_warps_release_count_less_barriers() {
    for seed in 0..4 {
        let s = scenarios::exit_then_barrier(false);
        let o = run_cfg(&s, &RunConfig { seed, ..s.config.clone() });
        completed(&o);
        let out = u32s(&o, "out");
        assert_eq!(&out[..32], &[7; 32]);
        assert_eq!(&out[32..], &[0; 32]);
        let s = scenarios::exit_then_barrier(true);
        let o = run_cfg(&s, &RunConfig { seed, ..s.config.clone() });
        match &o.status {
            RunStatus::Incomplete { reason, .. } => assert!(reason.contains("G8"), "{reason}"),
            other => panic!("expected incomplete (G8), got {other:?}"),
        }
    }
}

/// H2: a bounded probe loop of failed polls is never parked.
#[test]
fn bounded_probe_loop_is_not_a_spin() {
    let o = run(&scenarios::bounded_probe_loop());
    completed(&o);
    assert_eq!(u32s(&o, "out"), vec![3; 32]);
}

/// M1: a failed poll inside an inner loop exited by `break` reaches the
/// enclosing spin, which parks: Deadlock within a few rounds.
#[test]
fn nested_spin_with_break_parks() {
    let o = run(&scenarios::nested_spin_break());
    assert!(matches!(o.status, RunStatus::Deadlock { .. }), "{:?}", o.status);
    assert!(o.stats.rounds < 10, "rounds {}", o.stats.rounds);
}

/// M2: a spin on `ld.acquire` parks (Deadlock when never released) and
/// completes when another warp publishes.
#[test]
fn load_acquire_spin_parks_and_completes() {
    let o = run(&scenarios::load_flag_spin(true));
    completed(&o);
    assert_eq!(u32s(&o, "out"), vec![5; 32]);
    let o = run(&scenarios::load_flag_spin(false));
    match &o.status {
        RunStatus::Deadlock { blocked } => assert!(blocked.iter().any(|(_, r)| matches!(r, ResourceId::Word { .. })), "{blocked:?}"),
        other => panic!("expected Deadlock, got {other:?}"),
    }
    assert!(o.stats.rounds < 10, "rounds {}", o.stats.rounds);
}

/// H6: `Seeded` completion keeps seed-dependent latency across rounds (a
/// missing wait is visible for some seeds); `Eager` lands at the end of
/// the issuing round.
#[test]
fn seeded_completion_latency_survives_rounds() {
    let s = scenarios::cp_async_no_wait();
    let mut failed = 0;
    for seed in 0..32 {
        let cfg = RunConfig { seed, validity: ValidityPolicy::Error, completions: CompletionPolicy::Seeded, ..s.config.clone() };
        if matches!(run_cfg(&s, &cfg).status, RunStatus::Error(_)) {
            failed += 1;
        }
    }
    assert!(failed > 0 && failed < 32, "{failed} of 32 seeds read before landing");
    let cfg = RunConfig { validity: ValidityPolicy::Error, completions: CompletionPolicy::Eager, ..s.config.clone() };
    completed(&run_cfg(&s, &cfg));
}

/// H3: `wait_group.read 1` publishes the read milestone of the OLDER group
/// only; `wait_group 0` then publishes both writes.
#[test]
fn wait_group_read_covers_only_the_awaited_prefix() {
    let s = scenarios::bulk_wait_read();
    let mut log = RecordingObserver::new();
    let o = sched::run_with_config(&s.module, &s.inputs, &mut log, &s.config).unwrap();
    completed(&o);
    assert_eq!(u32s(&o, "out"), (0..32).collect::<Vec<_>>());
    let mut reads = 0;
    let mut writes = 0;
    for e in all_events(&log) {
        if let SyncKind::AsyncComplete { milestone, target: numsim_core::observe::PublishTarget::Warp { .. }, .. } = e.kind {
            match milestone {
                numsim_core::observe::Side::Read => reads += 1,
                numsim_core::observe::Side::Write => writes += 1,
            }
        }
    }
    assert_eq!((reads, writes), (1, 2));
}

/// H4 / M3: per-target `Arrive` lanes; one Protocol per wait instruction.
#[test]
fn lane_split_mbarrier_events() {
    let s = scenarios::lane_split_mbarrier();
    let mut log = RecordingObserver::new();
    let o = sched::run_with_config(&s.module, &s.inputs, &mut log, &s.config).unwrap();
    completed(&o);
    let mut arrive_lanes: Vec<u32> = all_events(&log)
        .iter()
        .filter(|e| matches!(e.kind, SyncKind::Arrive { .. }))
        .map(|e| e.lanes.bits())
        .collect();
    arrive_lanes.sort();
    assert_eq!(arrive_lanes, vec![0x0000_ffff, 0xffff_0000]);
    let waits = log.per_warp[0]
        .iter()
        .filter(|e| match &e.kind {
            SyncKind::Protocol { cmds, .. } => cmds.iter().any(|c| format!("{:?}", c.cmd).contains("WaitParity")),
            _ => false,
        })
        .count();
    assert_eq!(waits, 1, "one Protocol event for the lane-varying wait");
    let wait_events = all_events(&log).iter().filter(|e| matches!(e.kind, SyncKind::Wait { .. })).count();
    assert_eq!(wait_events, 2);
}

/// M4: lanes whose target completed leave a lane-varying wait.
#[test]
fn lane_varying_mbar_wait_latches() {
    for seed in 0..4 {
        let s = scenarios::mbar_latch();
        let o = run_cfg(&s, &RunConfig { seed, ..s.config.clone() });
        completed(&o);
        assert_eq!(u32s(&o, "out"), vec![1; 32]);
    }
}

/// H5: 128-bit exch/cas atomics.
#[test]
fn b128_exch_and_cas() {
    let o = run(&scenarios::atom_b128());
    completed(&o);
    assert_eq!(u32s(&o, "g"), vec![9, 9, 9, 9, 7, 7, 7, 7]);
    assert_eq!(u32s(&o, "out"), vec![1, 2, 3, 4, 5, 6, 7, 8]);
}

/// M5 / W5-8: st.async data lands; its accesses are generic-proxy strong
/// release writes at the instruction's scope.
#[test]
fn st_async_lands_as_generic_release() {
    struct Acc(Vec<(numsim_core::program::Proxy, numsim_core::program::Sem, numsim_core::program::Scope, bool)>);
    impl Observer for Acc {
        fn access(&mut self, a: &numsim_core::observe::Access<'_>) {
            if let numsim_core::observe::Actor::Async { side: numsim_core::observe::Side::Write, .. } = a.actor {
                self.0.push((a.proxy, a.sem, a.scope, a.atomic));
            }
        }
    }
    let s = scenarios::st_async_copy();
    let mut acc = Acc(Vec::new());
    let o = sched::run_with_config(&s.module, &s.inputs, &mut acc, &s.config).unwrap();
    completed(&o);
    assert_eq!(u32s(&o, "out"), (100..132).collect::<Vec<_>>());
    assert_eq!(acc.0.len(), 32);
    use numsim_core::program::{Proxy, Scope, Sem};
    assert!(acc.0.iter().all(|&x| x == (Proxy::Generic, Sem::Release, Scope::Cluster, true)), "{:?}", acc.0[0]);
}

/// M7: direct tcgen05.st to deallocated columns is a BadAddress error.
#[test]
fn tcgen_st_after_dealloc_is_bad_address() {
    let o = run(&scenarios::tcgen_after_dealloc());
    match &o.status {
        RunStatus::Error(e) => assert_eq!(e.kind, ExecErrorKind::BadAddress, "{e:?}"),
        other => panic!("expected BadAddress, got {other:?}"),
    }
}

/// M6: a wait_until predicate's buffer reads are recorded.
#[test]
fn wait_until_predicate_reads_are_captured() {
    let s = scenarios::wait_until_pred_reads();
    let mut obs = (Trace(Vec::new(), true), RecordingObserver::new());
    let o = sched::run_with_config(&s.module, &s.inputs, &mut obs, &s.config).unwrap();
    completed(&o);
    let found = all_events(&obs.1).iter().any(|e| matches!(&e.kind, SyncKind::WaitVerdicts { pred_reads, .. } if !pred_reads.is_empty()));
    assert!(found, "no WaitVerdicts with pred_reads");
}

/// M11: a declared word written past MAX_WORD_HISTORY makes an accepting
/// wait on it incomplete, with or without an observer: the write count is
/// kept in every mode (W13-1 ruling).
#[test]
fn word_history_overflow_is_incomplete() {
    let s = scenarios::word_history_overflow(scenarios::MAX_HISTORY_PROBE);
    let mut t = Trace(Vec::new(), true);
    let watched = sched::run_with_config(&s.module, &s.inputs, &mut t, &s.config).unwrap();
    match &watched.status {
        RunStatus::Incomplete { reason, .. } => assert!(reason.contains("history"), "{reason}"),
        other => panic!("expected incomplete, got {other:?}"),
    }
    let plain = run(&s);
    assert_eq!(plain.status, watched.status);
    // Below the limit the verdict is computed.
    let s = scenarios::word_history_overflow(100);
    let mut t = Trace(Vec::new(), true);
    let o = sched::run_with_config(&s.module, &s.inputs, &mut t, &s.config).unwrap();
    completed(&o);
}

/// M9 / review scenario 8: three-way divergent hand-off in a loop with
/// `continue` in one arm.
#[test]
fn divergent_three_way_handoff() {
    for seed in 0..4 {
        let s = scenarios::divergent_nesting();
        let o = run_cfg(&s, &RunConfig { seed, ..s.config.clone() });
        completed(&o);
        let out = u32s(&o, "out");
        for (l, v) in out.iter().enumerate() {
            let want = if (8..16).contains(&l) { 11 } else { 22 };
            assert_eq!(*v, want, "lane {l}");
        }
    }
}

/// M10: two clusters; one publishes, the other spins (visible one round
/// later); and the store-buffering cycle carries an incomplete diagnostic
/// only when observed, with identical outputs.
#[test]
fn cross_cluster_visibility_and_stream_cycles() {
    let o = run(&scenarios::cross_cluster_flag(true));
    completed(&o);
    assert_eq!(u32s(&o, "out"), vec![6]);
    let s = scenarios::cross_cluster_sb();
    let observed = run(&s);
    completed(&observed);
    assert_eq!(u32s(&observed, "out"), vec![0, 0], "both read the round-start value");
    assert!(
        observed.diagnostics.iter().any(|f| f.status == Status::Incomplete && f.message.contains("cannot order")),
        "{:?}",
        observed.diagnostics
    );
    let mut noop = numsim_core::observe::NoopObserver;
    let plain = sched::run_with_config(&s.module, &s.inputs, &mut noop, &s.config).unwrap();
    assert_eq!(plain.outputs, observed.outputs);
    assert!(plain.diagnostics.is_empty());
}

/// Review test gap: the partitioned (sharded) scheduler against the
/// single-partition sequential reference on race-free multi-cluster
/// programs, for several worker counts.
#[test]
fn sharded_matches_single_partition_reference() {
    let cases = vec![
        scenarios::vector_add(),
        scenarios::moe_synthetic(12, 3),
        scenarios::bulk_reduce(6),
        scenarios::cross_cluster_flag(true),
    ];
    for s in cases {
        for seed in [0, 3] {
            let reference = run_cfg(&s, &RunConfig { seed, single_partition: true, ..s.config.clone() });
            completed(&reference);
            for workers in [1, 4] {
                let o = run_cfg(&s, &RunConfig { seed, workers, ..s.config.clone() });
                completed(&o);
                assert_eq!(o.outputs, reference.outputs, "{} seed={seed} workers={workers}", s.name);
            }
        }
    }
}

/// M8: an observer that panics during replay with `workers > 1` propagates
/// the panic instead of hanging the process.
#[test]
fn observer_panic_with_workers_does_not_hang() {
    struct Boom;
    impl Observer for Boom {
        fn sync(&mut self, _e: &SyncEvent) {
            panic!("observer boom");
        }
    }
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let s = scenarios::moe_synthetic(4, 1);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut b = Boom;
            let _ = sched::run_with_config(&s.module, &s.inputs, &mut b, &RunConfig { workers: 2, ..s.config.clone() });
        }));
        let _ = tx.send(r.is_err());
    });
    let panicked = rx.recv_timeout(std::time::Duration::from_secs(120)).expect("run hung after an observer panic");
    assert!(panicked);
}

/// L7: a loop whose lanes all leave on its budget+1-th pass is not a
/// budget error.
#[test]
fn loop_exit_on_last_pass_is_not_over_budget() {
    let s = scenarios::bounded_probe_loop();
    let o = run_cfg(&s, &RunConfig { loop_budget: 3, ..s.config.clone() });
    completed(&o);
}

// ---------------------------------------------------------------------------
// Conformance-sweep regressions
// ---------------------------------------------------------------------------

/// V2C-9: a global load of a parameter generic address reads the
/// parameter; a store there is a BadAddress finding.
#[test]
fn param_aperture_reads_params_and_rejects_stores() {
    let o = run(&scenarios::param_aperture(false));
    completed(&o);
    assert_eq!(o.outputs.buffers["out"].0, 0x1234_5678_9abcu64.to_le_bytes().to_vec());
    let o = run(&scenarios::param_aperture(true));
    match &o.status {
        RunStatus::Error(e) => assert_eq!(e.kind, ExecErrorKind::BadAddress, "{e:?}"),
        other => panic!("expected BadAddress, got {other:?}"),
    }
}

/// V2C-14: without launch-bounds registers the initial budget follows the
/// legacy caller base, logged as a host `Configure`.
#[test]
fn setmaxnreg_initial_budget_from_launch_bounds() {
    let s = scenarios::setmaxnreg_default_budget();
    let mut log = RecordingObserver::new();
    let o = sched::run_with_config(&s.module, &s.inputs, &mut log, &s.config).unwrap();
    completed(&o);
    assert!(log.other.iter().any(|e| format!("{:?}", e.kind).contains("Configure { count: 256 }")), "no Configure 256");
    // min_blocks_per_sm = 4 caps the base at 512 / 4 = 128 (< 256: still an increase).
    let mut s = scenarios::setmaxnreg_default_budget();
    s.module.kernels[0].topology.min_blocks_per_sm = Some(4);
    let mut log = RecordingObserver::new();
    let o = sched::run_with_config(&s.module, &s.inputs, &mut log, &s.config).unwrap();
    assert!(log.other.iter().any(|e| format!("{:?}", e.kind).contains("Configure { count: 128 }")), "no Configure 128");
    let _ = o;
}

/// W8-6: views of one host array share one allocation.
#[test]
fn aliased_views_share_one_allocation() {
    let o = run(&scenarios::aliased_views());
    completed(&o);
    assert_eq!(u32s(&o, "mem"), vec![10, 11, 12, 13, 104, 105]);
    assert_eq!(u32s(&o, "x"), vec![10, 11, 12, 13]);
    assert_eq!(u32s(&o, "z"), vec![12, 13, 104, 105]);
    assert_eq!(u32s(&o, "out"), vec![12, 13, 104, 105]);
}

/// W4-9: prefetch.valid_addr address check; applypriority.async.bulk in
/// the bulk group.
#[test]
fn hint_ops_engine_effects() {
    let s = scenarios::hint_ops(true);
    let mut log = RecordingObserver::new();
    let o = sched::run_with_config(&s.module, &s.inputs, &mut log, &s.config).unwrap();
    completed(&o);
    let issued = log.per_warp[0].iter().any(|e| matches!(&e.kind, SyncKind::Protocol { cmds, .. } if cmds.iter().any(|c| format!("{:?}", c.cmd).contains("Issue"))));
    assert!(issued, "applypriority.async.bulk did not join the bulk group");
    // An address no binding covers cannot be proven invalid (W12-gaps 1
    // ruling): `incomplete`, superseding W4-9's `BadAddress`.
    let o = run(&scenarios::hint_ops(false));
    match &o.status {
        RunStatus::Incomplete { reason, .. } => assert!(reason.contains("integer_address_without_binding"), "{reason}"),
        other => panic!("expected incomplete, got {other:?}"),
    }
}

/// W12-gaps 7: `mbarrier.init` pointer neither warp-uniform nor one-to-one.
#[test]
fn mbar_init_partial_lane_aliasing_is_an_error() {
    let o = run(&scenarios::mbar_init_partial_alias());
    match &o.status {
        RunStatus::Error(e) => {
            assert_eq!(e.kind, ExecErrorKind::Divergence, "{e:?}");
            assert!(e.message.contains("mbarrier.init pointer must be warp-uniform or one-to-one across active lanes"), "{}", e.message);
        }
        other => panic!("expected an error, got {other:?}"),
    }
    completed(&run(&scenarios::lane_split_mbarrier()));
}

/// W12-gaps 8: a bulk `applypriority` needs a 128-byte aligned address.
#[test]
fn bulk_applypriority_requires_128_byte_alignment() {
    let o = run(&scenarios::bulk_applypriority_misaligned());
    match &o.status {
        RunStatus::Error(e) => {
            assert_eq!(e.kind, ExecErrorKind::Op(OpErrorKind::Invalid), "{e:?}");
            assert!(e.message.contains("128-byte aligned"), "{}", e.message);
        }
        other => panic!("expected an error, got {other:?}"),
    }
}

/// `.exclusive` TMEM allocation limit follows `Program::arch`.
#[test]
fn exclusive_tmem_limit_follows_arch() {
    let o = run(&scenarios::tcgen_exclusive_576("sm_107f"));
    completed(&o);
    assert!(o.sync_leftovers.is_empty(), "{:?}", o.sync_leftovers);
    let o = run(&scenarios::tcgen_exclusive_576("sm_100a"));
    match &o.status {
        RunStatus::Error(e) => assert!(e.message.contains("InvalidColumns"), "{e:?}"),
        other => panic!("expected InvalidColumns, got {other:?}"),
    }
}

/// V2C-19/20: register-space buffers are memory-backed per lane and their
/// uninitialized reads report `space: register`.
#[test]
fn register_buffer_uninit_reads_report_register_space() {
    let o = run(&scenarios::reg_buffer_uninit());
    completed(&o);
    assert_eq!(u32s(&o, "out"), (0..32).collect::<Vec<_>>(), "uninit element reads as zero");
    let f: Vec<_> = o.diagnostics.iter().filter(|f| f.kind == FindingKind::UninitRead).collect();
    // One finding per lane's read range (lane-major layout), all register space.
    assert_eq!(f.len(), 32, "{:#?}", o.diagnostics);
    assert!(f.iter().all(|x| x.evidence[0].space == Some(numsim_core::arena::Space::Reg)));
}

/// W1 ruling: a `mapa.shared::cluster` value works as a SharedCluster and as
/// a Generic (zero-extended) mbarrier operand.
#[test]
fn mapa_value_as_cluster_and_generic_mbarrier_operand() {
    for seed in 0..3 {
        let s = scenarios::mapa_cluster_arrive();
        let o = run_cfg(&s, &RunConfig { seed, ..s.config.clone() });
        completed(&o);
        assert_eq!(u32s(&o, "out"), vec![1, 1]);
    }
}

/// V2C-24: implicit TMEM ownership without tcgen05.alloc.
#[test]
fn implicit_tmem_views_work_without_alloc() {
    let o = run(&scenarios::implicit_tmem());
    completed(&o);
    let want: Vec<u32> = (0..128u32).map(|i| (i / 4) * 10 + i % 4).collect();
    assert_eq!(u32s(&o, "out"), want);
}

/// W8-7: planned global addresses are the ones the run uses.
#[test]
fn planned_global_addresses_match_the_run() {
    use numsim_core::dtype::{Dtype, Ty};
    use numsim_core::testutil::ProgramBuilder;
    let mut b = ProgramBuilder::new("addr_probe", 32);
    let x = b.global("x", Dtype::U32);
    let z = b.global("z", Dtype::U32);
    let out = b.global("out", Dtype::U64);
    let a = b.reg(Ty::U64);
    let k0 = b.k_u32(0);
    let k1 = b.k_u32(1);
    b.addr_of(a, x, k0);
    b.st(Ty::U64, out, k0, a);
    b.addr_of(a, z, k0);
    b.st(Ty::U64, out, k1, a);
    b.exit();
    let module = b.build_module();
    let s = scenarios::aliased_views();
    let mut inputs = s.inputs.clone();
    inputs.args.insert("out".into(), sched::ArgValue::Buffer { bytes: vec![0; 16], valid: None });
    let plan = sched::plan_global_addresses(&module, &inputs).unwrap();
    let o = sched::run_with_config(&module, &inputs, &mut numsim_core::observe::NoopObserver, &RunConfig::default()).unwrap();
    completed(&o);
    let got: Vec<u64> = o.outputs.buffers["out"].0.chunks(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
    assert_eq!(got, vec![plan["x"], plan["z"]]);
    assert_eq!(plan["z"] - plan["x"], 8);
    assert_eq!(plan["x"], plan["mem"]);
}

/// W4-11: packed FP4 TMA store with masked sub-byte fragments, partly out
/// of bounds: the engine applies byte spans then fragments exactly like the
/// oplib plan says.
#[test]
fn fp4_tma_store_applies_bit_fragments() {
    use numsim_core::oplib::{tma_plan_dir, TmaPlanDir};
    use numsim_core::program::TmaMode;
    let s = scenarios::fp4_tma_store();
    let o = run(&s);
    completed(&o);
    let va = sched::plan_global_addresses(&s.module, &s.inputs).unwrap()["dst"];
    let mut d = scenarios::fp4_store_desc();
    d.global_address = va;
    let plan = tma_plan_dir(&d, TmaPlanDir::Store, TmaMode::Tile, &[32, 0], &[], 0).unwrap();
    assert!(!plan.global_bits.is_empty(), "expected sub-byte fragments");
    let shared: Vec<u8> = (0..64u32).map(|i| ((i % 16) * 16 + i / 2 + 1) as u8).collect();
    let mut want = vec![0xeeu8; 64];
    let src: Vec<u8> = plan.smem.iter().flat_map(|s| shared[s.start as usize..s.end() as usize].to_vec()).collect();
    let mut at = 0usize;
    for g in &plan.global {
        let off = (g.start - va) as usize;
        want[off..off + g.len as usize].copy_from_slice(&src[at..at + g.len as usize]);
        at += g.len as usize;
    }
    for f in &plan.global_bits {
        let mask = f.mask << f.target_shift;
        let sb = (shared[f.smem as usize] >> f.source_shift) & f.mask;
        let g = &mut want[(f.global - va) as usize];
        *g = (*g & !mask) | ((sb << f.target_shift) & mask);
    }
    assert_eq!(o.outputs.buffers["dst"].0, want);
    assert_ne!(want, vec![0xeeu8; 64]);
}

/// V2C-35 ruling: a top-level buffer's synthetic base is aligned like
/// `cudaMalloc` whatever its host pointer; a view keeps its byte offset
/// inside its region (misaligned sub-views stay visible).
#[test]
fn global_bases_are_aligned_and_views_keep_offsets() {
    let mut s = scenarios::aliased_views();
    s.inputs.host_addrs.insert("mem".into(), 0x7f00_1234_5678_9a44);
    s.inputs.host_addrs.insert("out".into(), 0x7f00_0000_0000_0010);
    let plan = sched::plan_global_addresses(&s.module, &s.inputs).unwrap();
    assert_eq!(plan["mem"] % 256, 0);
    assert_eq!(plan["out"] % 256, 0);
    assert_eq!(plan["x"], plan["mem"]);
    assert_eq!(plan["z"], plan["mem"] + 8);
    let o = run(&s);
    completed(&o);
    assert_eq!(u32s(&o, "out"), vec![12, 13, 104, 105]);
    assert_eq!(u32s(&o, "mem"), vec![10, 11, 12, 13, 104, 105]);
}

/// W8-8: a buffer argument no parameter references is still placed (its
/// address is planned, it is returned unchanged) and does not move the
/// referenced buffers.
#[test]
fn unreferenced_buffer_arguments_are_allocated() {
    let mut s = scenarios::aliased_views();
    let before = sched::plan_global_addresses(&s.module, &s.inputs).unwrap();
    s.inputs.args.insert("zz_extra".into(), sched::ArgValue::Buffer { bytes: vec![7, 8, 9], valid: None });
    let plan = sched::plan_global_addresses(&s.module, &s.inputs).unwrap();
    assert!(plan.contains_key("zz_extra"));
    assert_eq!(plan["zz_extra"] % 256, 0);
    for (k, v) in &before {
        assert_eq!(plan[k], *v, "{k} moved");
    }
    let o = run(&s);
    completed(&o);
    assert_eq!(o.outputs.buffers["zz_extra"].0, vec![7, 8, 9]);
}

/// Ruling: physical special registers read 0.
#[test]
fn physical_special_registers_read_zero() {
    let o = run(&scenarios::physical_sregs());
    completed(&o);
    assert_eq!(o.outputs.buffers["out"].0, vec![0u8; 15 * 8]);
}

/// Ruling: an out-of-cluster multicast mask is a BadAddress error.
#[test]
fn multicast_mask_outside_cluster_is_an_error() {
    let o = run(&scenarios::multicast_outside_cluster());
    match &o.status {
        RunStatus::Error(e) => assert_eq!(e.kind, ExecErrorKind::BadAddress, "{e:?}"),
        other => panic!("expected BadAddress, got {other:?}"),
    }
}

/// W9: `.ignore_oob` counts above 15 are rejected; an explicit
/// `.shared::cta` mbarrier operand naming another CTA is an error.
#[test]
fn ignore_oob_count_range_is_checked() {
    let mut s = scenarios::bulk_masked_copy();
    let k = &mut s.module.kernels[0];
    let id = numsim_core::program::ConstId(k.consts.len() as u32);
    k.consts.push(numsim_core::program::Const { ty: numsim_core::dtype::Ty::U32, bits: 16 });
    for ins in &mut k.code {
        if let numsim_core::program::Instr::BulkCopy(a) = ins {
            if let Some(io) = &mut a.ignore_oob {
                io.ignore_bytes_left = Some(numsim_core::program::Operand::Const(id));
            }
        }
    }
    let o = run(&s);
    match &o.status {
        RunStatus::Error(e) => assert!(e.message.contains("0..=15"), "{e:?}"),
        other => panic!("expected an ignore_oob range error, got {other:?}"),
    }
}

/// Contract batch 4: sub-word TMEM cells.
#[test]
fn tmem_subword_cells_pack_and_rmw() {
    let o = run(&scenarios::tmem_subword());
    completed(&o);
    let out: Vec<u16> = o.outputs.buffers["out"].0.chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let want: Vec<u16> = (0..128u32).map(|i| ((i / 4) * 100 + i % 4) as u16).collect();
    assert_eq!(out, want);
    let cells = u32s(&o, "cells");
    for l in 0..32u32 {
        assert_eq!(cells[l as usize], (0xbeefu32 << 16) | (l * 100), "lane {l}");
    }
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn fegetround() -> i32;
    fn fesetround(round: i32) -> i32;
}

/// W4-14: a run does not inherit the caller's rounding mode, and restores
/// it afterwards (workers 1 and 4).
#[cfg(target_os = "linux")]
#[test]
fn run_uses_its_own_fp_environment_and_restores_the_callers() {
    use numsim_core::dtype::{Dtype, Ty};
    use numsim_core::program::BinOp;
    use numsim_core::testutil::ProgramBuilder;
    const FE_DOWNWARD: i32 = 0x400;
    let mut b = ProgramBuilder::new("fp_env", 32);
    b.grid(4, 1, 1);
    let x = b.global("x", Dtype::F32);
    let out = b.global("out", Dtype::F32);
    let lane = b.reg(Ty::U32);
    let v = b.reg(Ty::F32);
    let w = b.reg(Ty::F32);
    b.lane_id(lane);
    b.ld_f32(v, x, lane);
    let k3 = b.k_f32(3.0);
    b.binary(BinOp::Div, Ty::F32, w, v, k3);
    b.add_f32(w, w, v);
    b.st_f32(out, lane, w);
    b.exit();
    let module = b.build_module();
    let inputs = scenarios::inputs(vec![("x", scenarios::f32_buf((0..32).map(|i| 1.0 + i as f32 * 0.1))), ("out", scenarios::f32_buf([0.0; 32]))]);
    let run_it = |workers| {
        let cfg = RunConfig { workers, ..RunConfig::default() };
        sched::run_with_config(&module, &inputs, &mut numsim_core::observe::NoopObserver, &cfg).unwrap().outputs
    };
    let reference = run_it(1);
    for workers in [1, 4] {
        // SAFETY: test-thread rounding mode, restored below.
        unsafe { fesetround(FE_DOWNWARD) };
        let o = run_it(workers);
        let mode = unsafe { fegetround() };
        unsafe { fesetround(0) };
        assert_eq!(mode, FE_DOWNWARD, "caller's rounding mode restored");
        assert_eq!(o, reference, "workers={workers}: results independent of the caller's rounding");
    }
}

/// W1 batch 4: 16-bit TMEM view rows map to the right lanes (warp 1 in
/// lanes 32..64).
#[test]
fn tmem_f16_view_lanes() {
    let o = run(&scenarios::tmem_f16_rows());
    completed(&o);
    let out: Vec<u16> = o.outputs.buffers["out"].0.chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let want: Vec<u16> = (0..512u32).map(|i| (i + i % 8) as u16).collect();
    assert_eq!(out, want);
}

/// W5-11: cp.async published through cp.async.mbarrier.arrive.
#[test]
fn cp_async_mbarrier_arrive_publishes_copies() {
    for seed in 0..4 {
        let s = scenarios::cp_async_mbar_publish();
        let mut log = RecordingObserver::new();
        let o = sched::run_with_config(&s.module, &s.inputs, &mut log, &RunConfig { seed, ..s.config.clone() }).unwrap();
        completed(&o);
        assert_eq!(u32s(&o, "out"), (0..32).map(|i| i * 11 + 1).collect::<Vec<_>>());
        let published = all_events(&log)
            .iter()
            .filter(|e| matches!(e.kind, SyncKind::AsyncComplete { target: numsim_core::observe::PublishTarget::Phase { .. }, .. }))
            .count();
        assert_eq!(published, 32, "every lane's copy is published to the phase");
    }
}

/// Contract batch 4: a host-prelude tensor map with a runtime box gives the
/// same result as the bound static map.
#[test]
fn param_dependent_tensor_map_box() {
    let a = run(&scenarios::tma_load());
    let b = run(&scenarios::tma_load_param_box());
    completed(&a);
    completed(&b);
    assert_eq!(a.outputs.buffers["out"], b.outputs.buffers["out"]);
}

/// TMEM access footprints of a run, in delivery order: (kind, {(lane, byte)}).
#[derive(Default)]
struct TmemAccesses(Vec<(numsim_core::observe::AccessKind, std::collections::BTreeSet<(u8, u64)>)>);

impl Observer for TmemAccesses {
    fn access(&mut self, a: &numsim_core::observe::Access<'_>) {
        if a.space == numsim_core::arena::Space::Tmem {
            let set = a.spans.iter().flat_map(|s| (s.span.start..s.span.end()).map(move |b| (s.lane, b))).collect();
            self.0.push((a.kind, set));
        }
    }
}

/// W5 (c06149c): `tcgen05.ld/st` observer spans cover exactly the cells
/// each lane's pieces touch, identically on the run (fast) path and the
/// per-piece (fallback) path.
#[test]
fn tcgen_ldst_spans_are_exact_on_both_paths() {
    use numsim_core::arena::addr::tmem_byte_offset;
    use numsim_core::observe::AccessKind;
    let exact: std::collections::BTreeSet<(u8, u64)> =
        (0..32u32).flat_map(|l| (0..16u64).map(move |b| (l as u8, tmem_byte_offset(l, 0) + b))).collect();
    let go = |store: bool| {
        let s = scenarios::tcgen_ld_wide(store);
        let mut obs = TmemAccesses::default();
        let cfg = RunConfig { validity: ValidityPolicy::ZeroAndReport, ..s.config.clone() };
        let o = sched::run_with_config(&s.module, &s.inputs, &mut obs, &cfg).expect("run starts");
        completed(&o);
        obs.0
    };
    // st (fallback: fresh TMEM), st (run path), ld (run path).
    let fast = go(true);
    let kinds: Vec<AccessKind> = fast.iter().map(|x| x.0).collect();
    assert_eq!(kinds, vec![AccessKind::Write, AccessKind::Write, AccessKind::Read], "{kinds:?}");
    for (k, set) in &fast {
        assert_eq!(set, &exact, "{k:?}");
    }
    // ld of never-written cells (fallback path).
    let slow = go(false);
    assert_eq!(slow.len(), 1);
    assert_eq!(slow[0], fast[2]);
}

/// Readonly-proxy contract (W5, c06149c; legacy `test_readonly_proxy`): a
/// global write overlapping bytes read through `ld.global.nc`, in either
/// order or from another partition, is an execution error; disjoint writes
/// are fine.
#[test]
fn readonly_proxy_writes_are_rejected() {
    for v in ["clean", "disjoint"] {
        let o = run(&scenarios::readonly_proxy(v));
        completed(&o);
        assert_eq!(u32s(&o, "out"), (0..32).collect::<Vec<u32>>(), "{v}");
    }
    for v in ["after", "before", "cross_cta"] {
        for workers in [1usize, 2] {
            let s = scenarios::readonly_proxy(v);
            let cfg = RunConfig { workers, ..s.config.clone() };
            let o = run_cfg(&s, &cfg);
            match &o.status {
                RunStatus::Error(e) => assert!(e.message.contains("write overlaps readonly bytes"), "{v}: {e:?}"),
                other => panic!("{v}: expected the readonly error, got {other:?}"),
            }
        }
    }
}

/// Q3 ruling: a non-`.aligned` `barrier.sync` reached by divergent lanes of
/// a warp waits for the rest of the warp (one warp arrival, any site); the
/// missing lanes reaching another barrier id, or exiting, is `PartialWarp`.
#[test]
fn divergent_named_barrier_waits_for_the_warp() {
    let o = run(&scenarios::divergent_named_barrier("same"));
    completed(&o);
    assert_eq!(u32s(&o, "out"), vec![7; 64]);
    for v in ["other_id", "exit"] {
        let o = run(&scenarios::divergent_named_barrier(v));
        match &o.status {
            RunStatus::Error(e) => assert!(
                matches!(e.kind, ExecErrorKind::Protocol(numsim_core::sync::SyncError::Named(numsim_core::sync::named::Error::PartialWarp { .. }))),
                "{v}: {e:?}"
            ),
            other => panic!("{v}: expected PartialWarp, got {other:?}"),
        }
    }
}

/// W5-12: lanes read the `tcgen05.alloc` result right after the collective.
#[test]
fn tcgen_alloc_result_is_read_by_every_lane() {
    let o = run(&scenarios::tcgen_alloc_lanes_read());
    completed(&o);
    assert_eq!(u32s(&o, "out"), vec![0; 64]);
}

/// W2-8: `.per_16bytes` reports compare only the first element of each
/// 16-byte source chunk with the pattern.
#[test]
fn copy_report_per_16bytes_samples_chunk_heads() {
    for sampled in [false, true] {
        let o = run(&scenarios::copy_report_16(sampled));
        completed(&o);
        assert_eq!(u32s(&o, "out"), vec![0, sampled as u32], "sampled={sampled}");
    }
}

/// W5-14: a `sync_words` view is declared one word per element of its
/// dtype (32 four-byte words for 128 bytes of u32), never one word.
#[test]
fn sync_words_are_declared_per_element() {
    #[derive(Default)]
    struct Words(Vec<numsim_core::arena::ByteSpan>);
    impl Observer for Words {
        fn wants_word_history(&self) -> bool {
            true
        }
        fn sync(&mut self, e: &SyncEvent) {
            if let SyncKind::DeclareWord { span, .. } = e.kind {
                self.0.push(span);
            }
        }
    }
    let s = scenarios::polled_flag_words();
    let mut obs = Words::default();
    let o = sched::run_with_config(&s.module, &s.inputs, &mut obs, &s.config).expect("run starts");
    completed(&o);
    assert_eq!(u32s(&o, "out"), (1..=32).collect::<Vec<u32>>());
    assert_eq!(obs.0.len(), 32, "{:?}", obs.0);
    assert!(obs.0.iter().all(|s| s.len == 4), "{:?}", obs.0);
}

/// W11-2: a PTX op key resolves per operand-type signature, not once per
/// name: `cvt.s8.s8` sign-extends -1 into s16, s32 and s64 carriers.
#[test]
fn ptx_op_resolves_per_operand_signature() {
    let o = run(&scenarios::ptx_op_per_signature());
    completed(&o);
    let v: Vec<i64> = o.outputs.buffers["out"].0.chunks(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect();
    assert_eq!(v, vec![-1, -1, -1]);
}

/// W8: a generic-address `tensormap.replace` on a shared-memory descriptor
/// image resolves through the generic shared window.
#[test]
fn tensormap_replace_through_a_generic_shared_address() {
    let o = run(&scenarios::tmap_replace_generic_shared());
    completed(&o);
    let bytes: [u8; 128] = o.outputs.buffers["out"].0[..128].try_into().unwrap();
    let d = numsim_core::oplib::TensorMapDesc::decode(&bytes).expect("decodes");
    assert_eq!(d.rank, 2);
}

/// README decision 15 (W5-15): `SiteInfo.buffers` JSON; a module without
/// it (legacy `buffer`) derives `[buffer]`; both fields are written.
#[test]
fn site_info_buffers_json_shapes() {
    use numsim_core::site::SiteInfo;
    let legacy = r#"{"kind":"k","spans":[],"op_name":"o","text":"t","dtype":null,"buffer":"x"}"#;
    let s: SiteInfo = serde_json::from_str(legacy).unwrap();
    assert_eq!(s.buffers, vec![Some("x".to_string())]);
    assert_eq!(s.buffer(), Some("x"));
    let new = r#"{"kind":"k","spans":[],"op_name":"o","text":"t","dtype":null,"buffers":["d",null,"s"]}"#;
    let s: SiteInfo = serde_json::from_str(new).unwrap();
    assert_eq!(s.buffer_of(0), Some("d"));
    assert_eq!(s.buffer_of(1), None);
    assert_eq!(s.buffer_of(2), Some("s"));
    let back: serde_json::Value = serde_json::to_value(&s).unwrap();
    assert_eq!(back["buffer"], "d");
    assert_eq!(back["buffers"], serde_json::json!(["d", null, "s"]));
    assert_eq!(serde_json::from_value::<SiteInfo>(back).unwrap(), s);
    let bad = r#"{"kind":"k","spans":[],"op_name":"o","text":"t","dtype":null,"buffers":[],"extra":1}"#;
    assert!(serde_json::from_str::<SiteInfo>(bad).is_err());
}

/// README decision 15: `Access.operand` of a TMA load: the shared
/// destination writes are operand 0, the global source reads operand 1,
/// and the issue-time tensor-map read is operand 1.
#[test]
fn access_operand_names_the_pointer_operand() {
    #[derive(Default)]
    struct Ops(Vec<(numsim_core::observe::Actor, numsim_core::arena::Space, numsim_core::observe::AccessKind, u8)>);
    impl Observer for Ops {
        fn access(&mut self, a: &numsim_core::observe::Access<'_>) {
            self.0.push((a.actor, a.space, a.kind, a.operand));
        }
    }
    use numsim_core::arena::Space;
    use numsim_core::observe::{AccessKind, Actor};
    let s = scenarios::tma_load();
    let mut obs = Ops::default();
    let o = sched::run_with_config(&s.module, &s.inputs, &mut obs, &s.config).expect("run starts");
    completed(&o);
    let async_writes: Vec<u8> = obs.0.iter().filter(|x| matches!(x.0, Actor::Async { .. }) && x.2 == AccessKind::Write && x.1 == Space::Shared).map(|x| x.3).collect();
    let async_reads: Vec<u8> = obs.0.iter().filter(|x| matches!(x.0, Actor::Async { .. }) && x.2 == AccessKind::Read && x.1 == Space::Global).map(|x| x.3).collect();
    assert!(!async_writes.is_empty() && async_writes.iter().all(|&o| o == 0), "{async_writes:?}");
    assert!(!async_reads.is_empty() && async_reads.iter().all(|&o| o == 1), "{async_reads:?}");
    assert!(obs.0.iter().any(|x| matches!(x.0, Actor::Warp { .. }) && x.1 == Space::Param && x.3 == 1), "{:?}", obs.0);
}

/// W11-pin-message 1: an ALU fault names the faulting lanes and carries the
/// operation and the operands as structured attrs.
#[test]
fn alu_fault_names_the_faulting_lane_and_operands() {
    let o = run(&scenarios::alu_div_by_zero_lane7());
    let RunStatus::Error(e) = &o.status else { panic!("{:?}", o.status) };
    assert_eq!(e.lanes.0, 1 << 7, "{e:?}");
    assert_eq!(e.attrs["faulting_lanes"], serde_json::json!([7]));
    assert_eq!(e.attrs["operands"], serde_json::json!([29, 0]));
    assert!(e.attrs["operation"].as_str().unwrap().starts_with("Div"), "{e:?}");
}

/// W11-pin-message 5: incomplete stops carry warp / lanes / budget attrs.
#[test]
fn incomplete_stops_carry_structured_location() {
    let s = scenarios::loop_budget();
    let o = run(&s);
    let RunStatus::Incomplete { attrs, .. } = &o.status else { panic!("{:?}", o.status) };
    assert_eq!(attrs["budget"], serde_json::json!(s.config.loop_budget));
    assert!(attrs.contains_key("warp") && attrs.contains_key("lanes") && attrs.contains_key("iteration"), "{attrs:?}");
    let o = run(&scenarios::divergent_stuck_wait());
    let RunStatus::Incomplete { reason, attrs, .. } = &o.status else { panic!("{:?}", o.status) };
    assert!(reason.starts_with("divergent_block"), "{reason}");
    assert_eq!(attrs["warp"], serde_json::json!(0));
    assert_eq!(attrs["lanes"], serde_json::json!(0xffff_fffeu32));
}

/// W9 phase 6: a vector (`u64x2`) CAS compares and replaces all 16 bytes at
/// once: a compare differing in one component changes nothing.
#[test]
fn vector_cas_is_one_128_bit_compare_and_swap() {
    let o = run(&scenarios::cas128());
    completed(&o);
    let u64s = |n: &str| {
        o.outputs.buffers[n]
            .0
            .chunks(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect::<Vec<u64>>()
    };
    assert_eq!(u64s("out"), vec![7, 9, 7, 9]);
    assert_eq!(u64s("mem"), vec![23, 29]);
}

/// W5-12: the `tcgen05.alloc` address store is one warp-collective access
/// (`ALL_LANES`), not one lane's.
#[test]
fn tcgen_alloc_address_store_is_one_warp_access() {
    #[derive(Default)]
    struct Stores(Vec<Vec<u8>>);
    impl Observer for Stores {
        fn access(&mut self, a: &numsim_core::observe::Access<'_>) {
            if a.space == numsim_core::arena::Space::Shared
                && a.kind == numsim_core::observe::AccessKind::Write
            {
                self.0.push(a.spans.iter().map(|s| s.lane).collect());
            }
        }
    }
    let s = scenarios::tcgen_alloc_lanes_read();
    let mut obs = Stores::default();
    let o = sched::run_with_config(&s.module, &s.inputs, &mut obs, &s.config)
        .expect("run starts");
    completed(&o);
    assert!(!obs.0.is_empty());
    assert!(
        obs.0
            .iter()
            .all(|lanes| lanes == &vec![numsim_core::observe::ALL_LANES]),
        "{:?}",
        obs.0
    );
}

/// Host operands of the `tcgen_mma_f16` tests: small integers (exact in
/// f16), and the reference `d0 + A x B^T`.
fn mma_operands() -> (Vec<u16>, Vec<u16>, Vec<f32>) {
    use numsim_core::testutil::scenarios::{MMA_K as K, MMA_M as M, MMA_N as N};
    let f16 = numsim_oplib::arith::half::encode_f16;
    let a: Vec<f32> = (0..M * K)
        .map(|i| ((i / K + i % K) % 5) as f32 - 2.0)
        .collect();
    let b: Vec<f32> = (0..N * K)
        .map(|i| ((3 * (i / K) + i % K) % 4) as f32 - 1.0)
        .collect();
    let mut d = vec![0f32; M * N];
    for m in 0..M {
        for n in 0..N {
            d[m * N + n] = (0..K).map(|k| a[m * K + k] * b[n * K + k]).sum();
        }
    }
    (
        a.iter().map(|&x| f16(x)).collect(),
        b.iter().map(|&x| f16(x)).collect(),
        d,
    )
}

/// The reusable MMA program computes D = A x B^T through real descriptors.
#[test]
fn tcgen_mma_f16_program_matches_the_reference() {
    let (a, b, want) = mma_operands();
    let o = run(&scenarios::tcgen_mma_f16(
        scenarios::MmaSpec {
            accumulate: false,
            init_d: None,
            two_issuers: false,
            collectors: &[],
            sparse: false,
        },
        &a,
        &b,
    ));
    completed(&o);
    assert_eq!(f32s(&o, "out"), want);
    let o = run(&scenarios::tcgen_mma_f16(
        scenarios::MmaSpec {
            accumulate: true,
            init_d: Some(1.0),
            two_issuers: false,
            collectors: &[],
            sparse: false,
        },
        &a,
        &b,
    ));
    completed(&o);
    assert_eq!(
        f32s(&o, "out"),
        want.iter().map(|x| x + 1.0).collect::<Vec<_>>()
    );
    assert!(
        !o.diagnostics
            .iter()
            .any(|f| f.kind == FindingKind::UninitRead),
        "{:?}",
        o.diagnostics
    );
}

/// W9 phase 6: an MMA accumulating into never-written TMEM D reports the
/// uninitialized D read (checked when read, before the MMA writes D); D
/// reads as zero.
#[test]
fn tcgen_mma_into_never_written_tmem_reports_uninit_read() {
    let (a, b, want) = mma_operands();
    let s = scenarios::tcgen_mma_f16(
        scenarios::MmaSpec {
            accumulate: true,
            init_d: None,
            two_issuers: false,
            collectors: &[],
            sparse: false,
        },
        &a,
        &b,
    );
    let o = run_cfg(
        &s,
        &RunConfig {
            validity: ValidityPolicy::ZeroAndReport,
            ..s.config.clone()
        },
    );
    completed(&o);
    assert_eq!(f32s(&o, "out"), want);
    let tmem = o.diagnostics.iter().filter(|f| {
        f.kind == FindingKind::UninitRead
            && f.evidence
                .iter()
                .any(|e| e.space == Some(numsim_core::arena::Space::Tmem))
    });
    assert!(tmem.count() > 0, "{:?}", o.diagnostics);
}

/// W12-tile-forms 1: `AddrOf` scales by the scalar element of a vector
/// buffer dtype, the same rule as Load/Store.
#[test]
fn addr_of_uses_the_scalar_element_size() {
    let o = run(&scenarios::addr_of_vector_buffer());
    completed(&o);
    assert_eq!(u32s(&o, "dist"), (0..32).map(|l| 4 * l).collect::<Vec<u32>>());
    assert_eq!(u32s(&o, "val"), (100..132).collect::<Vec<u32>>());
}

/// An observer that hashes the whole stream (accesses, sync events, warp
/// ends) and wants the declared-word history, so `WaitVerdicts` are
/// produced and the partitioned word merge runs.
#[derive(Default)]
struct StreamHash {
    h: u64,
    verdicts: Vec<String>,
}

impl StreamHash {
    fn mix(&mut self, s: &str) {
        for b in s.bytes() {
            self.h = (self.h ^ b as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
}

impl Observer for StreamHash {
    fn wants_word_history(&self) -> bool {
        true
    }
    fn access(&mut self, a: &numsim_core::observe::Access<'_>) {
        let s = format!("{:?}/{:?}/{:?}/{:?}/{:?}/{:?}", a.actor, a.site, a.alloc, a.kind, a.spans, a.declared_word);
        self.mix(&s);
    }
    fn sync(&mut self, e: &SyncEvent) {
        let s = format!("{e:?}");
        if matches!(e.kind, SyncKind::WaitVerdicts { .. }) {
            self.verdicts.push(s.clone());
        }
        self.mix(&s);
    }
}

/// Partitioned declared words: a wait_until chain across 8 single-CTA
/// clusters gives the same outputs and the same observer stream (incl.
/// WaitVerdicts numbering) at 1, 8 and 32 workers.
#[test]
fn partitioned_wait_until_is_deterministic_across_workers() {
    let s = scenarios::wait_until_chain(8);
    let mut seen: Option<(Vec<u32>, u64, Vec<String>)> = None;
    for workers in [1usize, 8, 32] {
        let mut obs = StreamHash::default();
        let cfg = RunConfig { workers, ..s.config.clone() };
        let o = sched::run_with_config(&s.module, &s.inputs, &mut obs, &cfg).expect("run starts");
        completed(&o);
        assert_eq!(u32s(&o, "out"), (0..8).collect::<Vec<u32>>(), "workers {workers}");
        assert_eq!(u32s(&o, "flag"), vec![8]);
        assert_eq!(obs.verdicts.len(), 8, "one verdict per CTA: {:?}", obs.verdicts);
        let now = (u32s(&o, "out"), obs.h, obs.verdicts.clone());
        if let Some(prev) = &seen {
            assert_eq!(prev, &now, "workers {workers} differ");
        }
        seen = Some(now);
    }
}

/// The partitioned word merge runs only for history-consuming observers and
/// changes nothing program-visible: same outputs with no observer and with
/// a history observer.
#[test]
fn partitioned_words_do_not_depend_on_the_observer() {
    let s = scenarios::wait_until_chain(8);
    let plain = sched::run_with_config(&s.module, &s.inputs, &mut numsim_core::observe::NoopObserver, &s.config).unwrap();
    let mut obs = StreamHash::default();
    let watched = sched::run_with_config(&s.module, &s.inputs, &mut obs, &s.config).unwrap();
    assert_eq!(plain.status, watched.status);
    assert_eq!(plain.outputs, watched.outputs);
    assert_eq!(plain.stats.rounds, watched.stats.rounds);
    assert_eq!(plain.stats.instrs, watched.stats.instrs);
}

/// W5-14 ruling: a `sync_words` buffer whose dtype is not a whole number of
/// bytes has no well-defined word: `incomplete` (history observers only).
#[test]
fn sub_byte_sync_words_fail_closed() {
    let mut s = scenarios::wait_until_flag();
    let flag = s.module.kernels[0].buffers.iter().position(|b| b.name == "flag").unwrap();
    s.module.kernels[0].buffers[flag].dtype = numsim_core::dtype::Ty::scalar(numsim_core::dtype::Dtype::E2M1);
    let mut obs = StreamHash::default();
    let o = sched::run_with_config(&s.module, &s.inputs, &mut obs, &s.config).unwrap();
    match &o.status {
        RunStatus::Incomplete { reason, .. } => assert!(reason.contains("sync_words"), "{reason}"),
        other => panic!("expected incomplete, got {other:?}"),
    }
}

/// W6 (sync §1.5 / §2.9): an mbarrier armed for more tx bytes than its TMA
/// delivers, waited on by the whole (converged) warp, is a protocol error
/// naming the byte counts, not a generic deadlock; the matching kernel
/// completes.
#[test]
fn tma_under_delivery_is_a_protocol_error() {
    completed(&run(&scenarios::tma_load()));
    let o = run(&scenarios::tma_under_delivery_full_warp());
    let RunStatus::Error(e) = &o.status else { panic!("{:?}", o.status) };
    assert!(
        matches!(
            e.kind,
            ExecErrorKind::Protocol(numsim_core::sync::SyncError::Mbarrier(numsim_core::sync::mbarrier::Error::TxUnderDelivered { gen: 0, expected: 516, completed: 512 }))
        ),
        "{e:?}"
    );
    assert!(e.message.contains("512 of 516 bytes"), "{}", e.message);
}

/// W12-gaps 1 ruling: an integer address naming no binding is
/// `incomplete` (`integer_address_without_binding`); just past a real
/// allocation (its guard gap) it is an out-of-bounds error.
#[test]
fn unbound_integer_address_is_incomplete() {
    let o = run(&scenarios::unbound_integer_load(false));
    match &o.status {
        RunStatus::Incomplete { reason, .. } => assert!(reason.contains("integer_address_without_binding"), "{reason}"),
        other => panic!("expected incomplete, got {other:?}"),
    }
    let o = run(&scenarios::unbound_integer_load(true));
    let RunStatus::Error(e) = &o.status else { panic!("{:?}", o.status) };
    assert_eq!(e.kind, ExecErrorKind::OutOfBounds, "{e:?}");
}

/// W12-gaps 3: two lanes executing one `tcgen05.mma` site is a kernel
/// error (single issuing thread).
#[test]
fn tcgen_mma_from_two_lanes_is_an_error() {
    let (a, b, _) = mma_operands();
    let o = run(&scenarios::tcgen_mma_f16(scenarios::MmaSpec { accumulate: false, init_d: None, two_issuers: true, collectors: &[], sparse: false }, &a, &b));
    let RunStatus::Error(e) = &o.status else { panic!("{:?}", o.status) };
    assert!(e.message.contains("single thread"), "{e:?}");
    assert_eq!(e.lanes.0, 0b11);
}

/// W6 S-b: every `WaitVerdicts.observed` index equals the number of
/// declared-word writes delivered before it (history numbering follows the
/// delivery order even when several partitions write the word in the serial
/// phase of one round).
#[test]
fn wait_verdict_indices_follow_the_delivery_order() {
    #[derive(Default)]
    struct Check {
        writes: u32,
        verdicts: Vec<(u32, u32)>,
    }
    impl Observer for Check {
        fn wants_word_history(&self) -> bool {
            true
        }
        fn access(&mut self, a: &numsim_core::observe::Access<'_>) {
            if a.declared_word && a.writes() {
                self.writes += a.spans.len() as u32;
            }
        }
        fn sync(&mut self, e: &SyncEvent) {
            if let SyncKind::WaitVerdicts { verdicts, .. } = &e.kind {
                for v in verdicts {
                    self.verdicts.push((v.observed, self.writes));
                }
            }
        }
    }
    for workers in [1usize, 8] {
        let s = scenarios::atomic_count_wait(4);
        let mut obs = Check::default();
        let cfg = RunConfig { workers, ..s.config.clone() };
        let o = sched::run_with_config(&s.module, &s.inputs, &mut obs, &cfg).expect("run starts");
        completed(&o);
        assert_eq!(u32s(&o, "out"), vec![4; 4]);
        assert_eq!(obs.verdicts.len(), 4, "{:?}", obs.verdicts);
        for (observed, delivered) in &obs.verdicts {
            assert_eq!(observed, delivered, "verdict index vs delivered writes: {:?}", obs.verdicts);
        }
    }
}

/// W6 (sync §4.6, Q11): a non-aligned cluster arrive from the two arms of a
/// divergent `If` is gathered into one warp arrival; a missing arm that
/// exits instead is `PartialWarp`.
#[test]
fn cluster_barrier_gathers_partial_warp_arrivals() {
    let o = run(&scenarios::cluster_partial_arrive(false));
    completed(&o);
    assert_eq!(u32s(&o, "out"), vec![2]);
    let o = run(&scenarios::cluster_partial_arrive(true));
    let RunStatus::Error(e) = &o.status else { panic!("{:?}", o.status) };
    assert!(
        matches!(e.kind, ExecErrorKind::Protocol(numsim_core::sync::SyncError::Cluster(numsim_core::sync::cluster::Error::PartialWarp { .. }))),
        "{e:?}"
    );
}

/// W12-gaps 9: collector A usage across MMAs. fill -> use -> lastuse is
/// legal and each MMA still reads its operands (3 x A·B); an intervening
/// MMA without a collector qualifier (PTX default `::discard`, as a typed
/// gemm dispatch emits) invalidates the fill, so a later `use` is an error.
#[test]
fn tcgen_collector_use_requires_a_live_fill() {
    use numsim_core::program::CollectorOp as C;
    let (a, b, want) = mma_operands();
    let spec = |collectors| scenarios::MmaSpec { accumulate: false, init_d: None, two_issuers: false, collectors, sparse: false };
    let o = run(&scenarios::tcgen_mma_f16(spec(&[C::Fill, C::Use, C::LastUse]), &a, &b));
    completed(&o);
    assert_eq!(f32s(&o, "out"), want.iter().map(|x| 3.0 * x).collect::<Vec<_>>());
    for seq in [&[C::Fill, C::None, C::Use][..], &[C::Use][..], &[C::Fill, C::LastUse, C::Use][..], &[C::Fill, C::Discard, C::Use][..]] {
        let o = run(&scenarios::tcgen_mma_f16(spec(seq), &a, &b));
        let RunStatus::Error(e) = &o.status else { panic!("{seq:?}: {:?}", o.status) };
        assert_eq!(e.kind, ExecErrorKind::Op(OpErrorKind::Invalid), "{e:?}");
        assert!(e.message.contains("requires a valid previous fill"), "{seq:?}: {}", e.message);
    }
}

/// Async-side reads of every MMA (actor `Async`, read kind): (operand,
/// space, allocation-relative spans).
#[derive(Default)]
struct MmaReads(Vec<(u8, numsim_core::arena::Space, Vec<numsim_core::arena::ByteSpan>)>);
impl Observer for MmaReads {
    fn access(&mut self, a: &numsim_core::observe::Access<'_>) {
        if matches!(a.actor, numsim_core::observe::Actor::Async { .. }) && a.kind == numsim_core::observe::AccessKind::Read {
            self.0.push((a.operand, a.space, a.spans.iter().map(|s| s.span).collect()));
        }
    }
}

/// Coordinator ruling (A-only restricted commit): every TMEM operand read
/// of an MMA (D when accumulating, sparse metadata, ...) is an async read of
/// the MMA op itself, named by its own pointer-operand index (site
/// `buffers` order: d, [a_tmem], [lut | sp_meta], [sfa, sfb]); shared A/B
/// reads (descriptors, not pointer operands) are MMA_SHARED_A/_B. Sparse
/// f16 (metadata 0x4444_4444 keeps elements 0 and 1 of each group of four)
/// also checks the numerics.
#[test]
fn tcgen_mma_operand_reads_are_named_per_operand() {
    use numsim_core::arena::{addr, Space};
    use numsim_core::sched::{MMA_SHARED_A, MMA_SHARED_B};
    let f16 = numsim_oplib::arith::half::encode_f16;
    let (m, n, k) = (scenarios::MMA_M, scenarios::MMA_N, scenarios::MMA_K);
    let av: Vec<f32> = (0..m * k).map(|i| ((i / k + 2 * (i % k)) % 5) as f32 - 2.0).collect();
    let bv: Vec<f32> = (0..n * 2 * k).map(|i| ((3 * (i / (2 * k)) + i % (2 * k)) % 4) as f32 - 1.0).collect();
    let mut want = vec![0f32; m * n];
    for r in 0..m {
        for c in 0..n {
            for g in 0..k / 2 {
                want[r * n + c] += av[r * k + 2 * g] * bv[c * 2 * k + 4 * g] + av[r * k + 2 * g + 1] * bv[c * 2 * k + 4 * g + 1];
            }
        }
    }
    let s = scenarios::tcgen_mma_f16(
        scenarios::MmaSpec { accumulate: true, init_d: Some(0.0), two_issuers: false, collectors: &[], sparse: true },
        &av.iter().map(|&x| f16(x)).collect::<Vec<_>>(),
        &bv.iter().map(|&x| f16(x)).collect::<Vec<_>>(),
    );
    let mut obs = MmaReads::default();
    let o = sched::run_with_config(&s.module, &s.inputs, &mut obs, &s.config).unwrap();
    completed(&o);
    assert_eq!(f32s(&o, "out"), want);
    let cols = |spans: &[numsim_core::arena::ByteSpan]| -> (u32, u32) {
        let row = addr::TMEM_COLS as u64 * 4;
        let lo = spans.iter().map(|s| ((s.start % row) / 4) as u32).min().unwrap();
        let hi = spans.iter().map(|s| (((s.end() - 1) % row) / 4) as u32).max().unwrap();
        (lo, hi)
    };
    let tmem: Vec<_> = obs.0.iter().filter(|r| r.1 == Space::Tmem).collect();
    let d = tmem.iter().find(|r| r.0 == 0).expect("accumulating D read (operand 0)");
    assert!(cols(&d.2).1 < 16, "D reads stay in D's columns: {:?}", cols(&d.2));
    let meta = tmem.iter().find(|r| r.0 == 1).expect("sparse metadata read (operand 1)");
    assert!(cols(&meta.2).0 >= 16, "metadata reads start at its base column: {:?}", cols(&meta.2));
    assert!(tmem.iter().all(|r| r.0 <= 1), "{:?}", tmem.iter().map(|r| r.0).collect::<Vec<_>>());
    let shared: Vec<u8> = obs.0.iter().filter(|r| r.1 == Space::Shared).map(|r| r.0).collect();
    assert!(shared.contains(&MMA_SHARED_A) && shared.contains(&MMA_SHARED_B), "{shared:?}");
}

/// W12-gaps 6: under a cluster subset the resident clusters claim every
/// non-resident cluster's task exactly once through CLC, deterministically
/// (claims are serial points in partition order): CTA 0 claims 2, CTA 1
/// claims 3, at any worker count and with or without an observer; a single
/// partition claims in its own (issue) order, also exactly once.
#[test]
fn clc_claims_non_resident_tasks_under_a_subset() {
    let s = scenarios::clc_task_steal();
    let mut first = None;
    for workers in [1usize, 4, 16] {
        for observe in [false, true] {
            let cfg = RunConfig { workers, ..s.config.clone() };
            let o = if observe {
                let mut log = RecordingObserver::new();
                sched::run_with_config(&s.module, &s.inputs, &mut log, &cfg).unwrap()
            } else {
                sched::run_with_config(&s.module, &s.inputs, &mut numsim_core::observe::NoopObserver, &cfg).unwrap()
            };
            completed(&o);
            let out = u32s(&o, "out");
            assert_eq!(out, vec![1, 1, 100, 101], "workers {workers} observe {observe}");
            match &first {
                None => first = Some(out),
                Some(f) => assert_eq!(&out, f),
            }
        }
    }
    let cfg = RunConfig { single_partition: true, ..s.config.clone() };
    let o = run_cfg(&s, &cfg);
    completed(&o);
    let out = u32s(&o, "out");
    assert_eq!(&out[..2], &[1, 1]);
    assert!(out[2..].iter().all(|&v| v == 100 || v == 101), "{out:?}");
    // Without a subset every cluster is resident: nothing to claim.
    let mut full = scenarios::clc_task_steal();
    full.config.subset = None;
    let o = run(&full);
    completed(&o);
    assert_eq!(u32s(&o, "out"), vec![1, 1, 1, 1]);
}

/// Records every callback; with `forks`, offers a child per partition
/// (decision 17) and appends the child's record at `join`.
#[derive(Default)]
struct ForkRec {
    forks: bool,
    log: Vec<String>,
}
impl Observer for ForkRec {
    fn wants_word_history(&self) -> bool {
        true
    }
    fn access(&mut self, a: &numsim_core::observe::Access<'_>) {
        self.log.push(format!("A {:?} {:?} {:?} {:?} {:?} {:?} {}", a.seq, a.actor, a.site, a.alloc, a.kind, a.spans, a.operand));
    }
    fn sync(&mut self, e: &SyncEvent) {
        self.log.push(format!("S {e:?}"));
    }
    fn warp_done(&mut self, w: numsim_core::observe::WarpId, end: numsim_core::observe::WarpEnd) {
        self.log.push(format!("D {w:?} {end:?}"));
    }
    fn round_boundary(&mut self, c: numsim_core::observe::CtaId, r: u64) {
        self.log.push(format!("R {c:?} {r}"));
    }
    fn phase_end(&mut self, r: u64) {
        self.log.push(format!("P {r}"));
    }
    fn fork(&mut self, _p: &numsim_core::observe::PartitionInfo<'_>) -> Option<Box<dyn numsim_core::observe::ForkedObserver>> {
        self.forks.then(|| Box::new(ForkRec::default()) as Box<dyn numsim_core::observe::ForkedObserver>)
    }
    fn join(&mut self, _p: &numsim_core::observe::PartitionInfo<'_>, child: Box<dyn numsim_core::observe::ForkedObserver>) {
        let child = child.into_any().downcast::<ForkRec>().expect("our child");
        self.log.extend(child.log);
    }
}
impl numsim_core::observe::ForkedObserver for ForkRec {
    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// Decision 17: a forking observer (children per partition, replayed on the
/// pool, joined in replay order) sees exactly the stream a non-forking one
/// sees, including `phase_end` points and `Access::seq`, at 1/8/32 workers.
#[test]
fn forked_replay_equals_serial_replay() {
    for s in &scenarios::all() {
        for workers in [1usize, 8, 32] {
            let cfg = RunConfig { workers, ..s.config.clone() };
            let mut serial = ForkRec::default();
            let a = sched::run_with_config(&s.module, &s.inputs, &mut serial, &cfg).unwrap();
            let mut forked = ForkRec { forks: true, log: Vec::new() };
            let b = sched::run_with_config(&s.module, &s.inputs, &mut forked, &cfg).unwrap();
            assert_eq!(format!("{:?}", a.status), format!("{:?}", b.status), "{} {workers}", s.name);
            assert_eq!(serial.log, forked.log, "{} at {workers} workers", s.name);
            assert!(serial.log.iter().any(|l| l.starts_with("P ")), "{}: no phase_end", s.name);
        }
    }
}

/// Counts, per declared word, the write accesses delivered after its
/// `DeclareWord`, and checks every `WaitVerdicts.observed` against it.
#[derive(Default)]
struct Numbering {
    words: Vec<(numsim_core::arena::AllocId, numsim_core::arena::ByteSpan, u32)>,
    verdicts: usize,
    first_use: usize,
    bad: Vec<String>,
}
impl Observer for Numbering {
    fn wants_word_history(&self) -> bool {
        true
    }
    fn access(&mut self, a: &numsim_core::observe::Access<'_>) {
        use numsim_core::observe::AccessKind;
        if !matches!(a.kind, AccessKind::Write | AccessKind::Rmw) {
            return;
        }
        for w in self.words.iter_mut().filter(|w| w.0 == a.alloc) {
            if a.spans.iter().any(|s| s.span.overlaps(w.1)) {
                w.2 += 1;
            }
        }
    }
    fn sync(&mut self, e: &SyncEvent) {
        match &e.kind {
            SyncKind::DeclareWord { alloc, span } => {
                if matches!(e.actor, numsim_core::observe::Actor::Warp { .. }) {
                    self.first_use += 1;
                }
                if !self.words.iter().any(|w| w.0 == *alloc && w.1 == *span) {
                    self.words.push((*alloc, *span, 0));
                }
            }
            SyncKind::WaitVerdicts { alloc, span, verdicts, .. } => {
                self.verdicts += 1;
                let n = self.words.iter().find(|w| w.0 == *alloc && w.1 == *span).map(|w| w.2);
                for v in verdicts {
                    if Some(v.observed) != n {
                        self.bad.push(format!("{alloc:?} {span:?}: observed {} vs {n:?} writes delivered", v.observed));
                    }
                }
            }
            _ => {}
        }
    }
}

/// W6-P2: a global word declared at its first `wait_until` (no
/// `sync_words`) while another cluster writes it in the same round: the
/// declaration is a serial point, so the history numbering equals the
/// delivered writes at 1/8/32 workers (and status/outputs are worker- and
/// observer-independent).
#[test]
fn first_use_word_numbering_follows_the_delivery() {
    let s = scenarios::first_use_wait(true);
    let mut first = None;
    for workers in [1usize, 8, 32] {
        let cfg = RunConfig { workers, ..s.config.clone() };
        let mut obs = Numbering::default();
        let o = sched::run_with_config(&s.module, &s.inputs, &mut obs, &cfg).unwrap();
        completed(&o);
        assert!(obs.first_use > 0 && obs.verdicts > 0, "first-use declaration not exercised");
        assert!(obs.bad.is_empty(), "{workers} workers: {:?}", obs.bad);
        let plain = sched::run_with_config(&s.module, &s.inputs, &mut numsim_core::observe::NoopObserver, &cfg).unwrap();
        assert_eq!(format!("{:?}{:?}", plain.status, plain.outputs), format!("{:?}{:?}", o.status, o.outputs));
        match &first {
            None => first = Some(u32s(&o, "out")),
            Some(f) => assert_eq!(&u32s(&o, "out"), f),
        }
    }
}
