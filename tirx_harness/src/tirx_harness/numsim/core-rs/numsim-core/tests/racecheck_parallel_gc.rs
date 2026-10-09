//! Parallel collector (racecheck-parallel-design.md §15): the per-allocation
//! shadow walk and the declared-word scan run on `gc_threads` threads. The
//! thread count is a run-time parameter only: the payload must equal the
//! inline collector's at every engine worker count, serial and fork/join.
//! The parallel path is forced (`gc_par_min_cells = 0`) so that every
//! collection of every scenario takes it.
use numsim_core::racecheck::observer::{RaceObserver, RacecheckConfig};
use numsim_core::report::Report;
use numsim_core::sched::{self, RunConfig};
use numsim_core::testutil::scenarios::{self, Scenario};
use numsim_core::testutil::fixtures;

fn payload(s: &Scenario, workers: usize, fork_join: bool, gc_threads: usize) -> (String, String) {
    let cfg = RunConfig { workers, ..s.config.clone() };
    let mut obs = RaceObserver::new(RacecheckConfig::default());
    obs.fork_join = fork_join;
    obs.phase_gc = true;
    obs.gc_threads = gc_threads;
    if gc_threads > 1 {
        obs.gc_par_min_cells = 0;
    }
    let o = sched::run_with_config(&s.module, &s.inputs, &mut obs, &cfg).expect("run starts");
    let report: Report = obs.finish();
    (format!("{:?}", o.status), format!("{report:?}"))
}

fn parallel_gc_matches_inline(s: &Scenario, workers: &[usize], threads: &[usize]) {
    for fork_join in [false, true] {
        let inline = payload(s, 1, fork_join, 1);
        for &w in workers {
            for &t in threads {
                assert_eq!(payload(s, w, fork_join, t), inline, "{}: gc_threads {t} differs at {w} workers (fork_join {fork_join})", s.name);
            }
        }
    }
}

#[test]
fn scenarios_parallel_gc_matches_inline() {
    for s in scenarios::all() {
        parallel_gc_matches_inline(&s, &[1, 8, 32], &[2, 16]);
    }
}

/// Recorded corpus fixtures (`examples/record_race_fixtures.py`; skipped
/// when absent, as in CI). Release builds only; 16 threads at 1 and 32
/// workers: about four minutes in release, e24 dominating.
#[test]
fn corpus_parallel_gc_matches_inline() {
    if cfg!(debug_assertions) {
        return;
    }
    let dir = fixtures::dir();
    for c in ["fp16_bf16_gemm", "gdn_decode_bf16_wide_vec_mtp", "radix_topk_multi_cta", "recurrent_kda_decode_one_warp", "mega_moe_t8_h1024_i512_e24_k2_g1"] {
        if !fixtures::exists(&dir, c) {
            continue;
        }
        let (module, inputs, config) = fixtures::load(&dir, c);
        parallel_gc_matches_inline(&Scenario { name: c, module, inputs, config }, &[1, 32], &[16]);
    }
}
