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

#[test]
fn workers_do_not_change_results() {
    for s in scenarios::all() {
        let a = run(&s);
        let b = run_cfg(&s, &RunConfig { workers: 8, ..s.config.clone() });
        assert_eq!(a, b, "{}", s.name);
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
