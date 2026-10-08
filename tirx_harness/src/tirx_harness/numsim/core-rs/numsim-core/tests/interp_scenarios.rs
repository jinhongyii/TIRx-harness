//! Interpreter + scheduler semantics on hand-built programs
//! (`testutil::scenarios`).

use numsim_core::arena::ValidityPolicy;
use numsim_core::interp::ExecErrorKind;
use numsim_core::observe::{Observer, RecordingObserver, SyncEvent, SyncKind};
use numsim_core::report::{FindingKind, Status};
use numsim_core::sched::{self, Backend, CompletionPolicy, RunConfig, RunOutcome, RunStatus};
use numsim_core::sync::ResourceId;
use numsim_core::testutil::scenarios::{self, Scenario};

fn run_cfg(s: &Scenario, config: &RunConfig) -> RunOutcome {
    let mut obs = RecordingObserver::new();
    sched::run_with_config(&s.module, &s.inputs, &mut obs, &Backend::Interp, config).expect("run starts")
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
        RunStatus::Incomplete { reason, site } => {
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
    let o = sched::run_with_config(&s.module, &s.inputs, &mut h, &Backend::Interp, &s.config).unwrap();
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
            let o = sched::run_with_config(&s.module, &s.inputs, &mut obs, &Backend::Interp, &RunConfig { seed, ..s.config.clone() }).unwrap();
            (o, obs)
        };
        let (a, la) = log(5);
        let (b, lb) = log(5);
        assert_eq!(a, b);
        assert_eq!(la, lb, "{}", s.name);
    }
}

#[test]
fn codegen_backend_shape_uses_the_same_path() {
    // `Backend::Codegen` with the interpreter's own step function must be
    // indistinguishable from `Backend::Interp`.
    for s in scenarios::all() {
        let mut o1 = RecordingObserver::new();
        let mut o2 = RecordingObserver::new();
        let a = sched::run_with_config(&s.module, &s.inputs, &mut o1, &Backend::Interp, &s.config).unwrap();
        let fns = vec![numsim_core::interp::step_warp as numsim_core::interp::WarpStepFn; s.module.kernels.len()];
        let b = sched::run_with_config(&s.module, &s.inputs, &mut o2, &Backend::Codegen(fns), &s.config).unwrap();
        assert_eq!(a, b, "{}", s.name);
        assert_eq!(o1, o2, "{}", s.name);
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
    fn inbox_drain(&mut self, c: numsim_core::observe::CtaId, r: u64) {
        self.0.push(format!("drain {c:?} {r}"));
    }
}

#[test]
fn workers_do_not_change_results_or_streams() {
    for s in scenarios::all() {
        for history in [false, true] {
            let mut t1 = Trace(Vec::new(), history);
            let a = sched::run_with_config(&s.module, &s.inputs, &mut t1, &Backend::Interp, &s.config).unwrap();
            for workers in [2, 8, 33] {
                let mut tn = Trace(Vec::new(), history);
                let cfg = RunConfig { workers, ..s.config.clone() };
                let b = sched::run_with_config(&s.module, &s.inputs, &mut tn, &Backend::Interp, &cfg).unwrap();
                assert_eq!(a, b, "{} workers={workers}", s.name);
                assert!(t1.0 == tn.0, "{} workers={workers}: observer streams differ", s.name);
            }
        }
        // NumSim (no observer) gives the same outputs as observed runs.
        let mut o = numsim_core::observe::NoopObserver;
        let c = sched::run_with_config(&s.module, &s.inputs, &mut o, &Backend::Interp, &RunConfig { workers: 8, ..s.config.clone() }).unwrap();
        let mut t = Trace(Vec::new(), true);
        let d = sched::run_with_config(&s.module, &s.inputs, &mut t, &Backend::Interp, &s.config).unwrap();
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
    let o = sched::run_with_config(&s.module, &s.inputs, &mut ev, &Backend::Interp, &s.config).unwrap();
    (o, ev)
}

#[test]
fn bulk_copy_mask_and_ignore_oob_narrow_bytes_and_footprint() {
    let s = scenarios::bulk_masked_copy();
    let (o, ev) = run_events(&s);
    completed(&o);
    assert_eq!(o.outputs.buffers["out"].0, scenarios::bulk_masked_expected());
    // The async write side covers exactly the transferred bytes.
    let want: Vec<u64> = (0..64u64).filter(|i| i % 16 < 8 && (4..56).contains(i)).collect();
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
        let o = sched::run_with_config(&s.module, &s.inputs, &mut ev, &Backend::Interp, &cfg).unwrap();
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
