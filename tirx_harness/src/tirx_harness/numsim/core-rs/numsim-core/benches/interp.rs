//! Interpreter microbenches (W2): dispatch cost per instruction on a
//! register-only loop, and per-warp load/store cost.
//! `cargo bench -p numsim-core --bench interp`.
//!
//! Throughput is reported per executed warp instruction (`RunStats::instrs`
//! of one run), so `time/elem` is the cost of one warp instruction.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use numsim_core::observe::{NoopObserver, RecordingObserver};
use numsim_core::sched::{self, Backend};
use numsim_core::testutil::scenarios::{self, Scenario};

fn group(c: &mut Criterion, name: &str, s: &Scenario) {
    let probe = sched::run_with_config(&s.module, &s.inputs, &mut NoopObserver, &Backend::Interp, &s.config).unwrap();
    assert_eq!(probe.status, sched::RunStatus::Completed, "{}", s.name);
    let instrs = probe.stats.instrs;
    let mut g = c.benchmark_group(name);
    g.throughput(Throughput::Elements(instrs));
    g.sample_size(20);
    g.bench_function("noop_observer", |b| {
        b.iter(|| sched::run_with_config(&s.module, &s.inputs, &mut NoopObserver, &Backend::Interp, &s.config).unwrap())
    });
    g.bench_function("recording_observer", |b| {
        b.iter(|| {
            let mut o = RecordingObserver::new();
            sched::run_with_config(&s.module, &s.inputs, &mut o, &Backend::Interp, &s.config).unwrap()
        })
    });
    g.finish();
}

fn bench(c: &mut Criterion) {
    group(c, "scalar_loop_dispatch", &scenarios::scalar_loop(20_000));
    group(c, "warp_load_store", &scenarios::warp_ldst(5_000));
}

criterion_group!(benches, bench);
criterion_main!(benches);
