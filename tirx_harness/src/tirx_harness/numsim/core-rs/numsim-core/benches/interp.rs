//! Interpreter microbenches (W2): dispatch cost per instruction on a
//! register-only loop, and per-warp load/store cost.
//! `cargo bench -p numsim-core --bench interp`.
//!
//! Throughput is reported per executed warp instruction (`RunStats::instrs`
//! of one run), so `time/elem` is the cost of one warp instruction.
//!
//! `recorded_word_history`: the recorded `sm100_fp8_fp4_mega_moe`
//! twenty_four_experts racecheck stream (148 SMs, 16 workers) under an
//! observer that only asks for declared-word history, i.e. engine plus the
//! partition word-history merge/refresh (`Scheduler::merge_words`). Needs the
//! fixture (`examples/record_race_fixtures.py OUT mega_moe:t8_h1024_i512_e24_k2_g1`
//! into `$RACE_FIXTURES` or `core-rs/target/race-fixtures`); skipped otherwise.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use numsim_core::observe::{NoopObserver, Observer, RecordingObserver};
use numsim_core::sched::{self};
use numsim_core::testutil::fixtures;
use numsim_core::testutil::scenarios::{self, Scenario};

fn group(c: &mut Criterion, name: &str, s: &Scenario) {
    let probe = sched::run_with_config(&s.module, &s.inputs, &mut NoopObserver, &s.config).unwrap();
    assert_eq!(probe.status, sched::RunStatus::Completed, "{}", s.name);
    let instrs = probe.stats.instrs;
    let mut g = c.benchmark_group(name);
    g.throughput(Throughput::Elements(instrs));
    g.sample_size(20);
    g.bench_function("noop_observer", |b| {
        b.iter(|| sched::run_with_config(&s.module, &s.inputs, &mut NoopObserver, &s.config).unwrap())
    });
    g.bench_function("recording_observer", |b| {
        b.iter(|| {
            let mut o = RecordingObserver::new();
            sched::run_with_config(&s.module, &s.inputs, &mut o, &s.config).unwrap()
        })
    });
    g.finish();
}

fn bench(c: &mut Criterion) {
    group(c, "scalar_loop_dispatch", &scenarios::scalar_loop(20_000));
    group(c, "warp_load_store", &scenarios::warp_ldst(5_000));
}

/// Events built and dropped, no word history (the observing baseline).
struct Observing;
impl Observer for Observing {}

/// Engine with declared-word history on and no checker.
struct WordsOnly;
impl Observer for WordsOnly {
    fn wants_word_history(&self) -> bool {
        true
    }
}

fn recorded_word_history(c: &mut Criterion) {
    let dir = fixtures::dir();
    let case = "mega_moe_t8_h1024_i512_e24_k2_g1";
    if !fixtures::exists(&dir, case) {
        eprintln!("recorded_word_history: fixture {dir}/{case}.* missing, skipped");
        return;
    }
    let (m, i, cfg) = fixtures::load(&dir, case);
    let mut g = c.benchmark_group("recorded_word_history");
    g.sample_size(10);
    g.bench_function("mega_moe_e24_noop", |b| b.iter(|| sched::run_with_config(&m, &i, &mut NoopObserver, &cfg).unwrap()));
    g.bench_function("mega_moe_e24_observing", |b| b.iter(|| sched::run_with_config(&m, &i, &mut Observing, &cfg).unwrap()));
    g.bench_function("mega_moe_e24_word_history", |b| b.iter(|| sched::run_with_config(&m, &i, &mut WordsOnly, &cfg).unwrap()));
    g.finish();
}

criterion_group!(benches, bench, recorded_word_history);
criterion_main!(benches);
