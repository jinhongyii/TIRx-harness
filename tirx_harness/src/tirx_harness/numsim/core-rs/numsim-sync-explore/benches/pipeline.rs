//! Visited states and time for a synthetic 16-warp x 4-stage x 32-iteration
//! producer/consumer pipeline under each reduction.
//!
//! `cargo bench --bench pipeline` first prints a reduction table (visited
//! states, explored transitions, verdict) and then runs criterion timings.
//! Configurations without projection exhaust the state budget: that is the
//! point of the table.

use std::time::{Duration, Instant};

use criterion::{black_box, BenchmarkId, Criterion};
use numsim_sync_explore::explore::{Limits, Options};
use numsim_sync_explore::projection::ProjectionMode;
use numsim_sync_explore::synth::{pipeline, PipelineShape};
use numsim_sync_explore::{check, CheckConfig, SyncEvent};

const SHAPE: PipelineShape = PipelineShape {
    warps: 16,
    stages: 4,
    iterations: 32,
    tma_bytes: 0,
};

const BUDGET: Limits = Limits {
    max_states: 100_000,
    max_transitions: 2_000_000,
};

fn configs() -> Vec<(&'static str, CheckConfig)> {
    let sleep = Options { sleep_sets: true, ..Options::NONE };
    let raw = CheckConfig {
        mode: ProjectionMode::Whole,
        certificates: false,
        fingerprints: false,
        explore: Options::NONE,
        limits: BUDGET,
    };
    let per_resource = CheckConfig { mode: ProjectionMode::PerResource, ..raw };
    vec![
        ("whole/plain", raw),
        ("whole/sleep", CheckConfig { explore: sleep, ..raw }),
        ("whole/sleep+diamond+persistent", CheckConfig { explore: Options::ALL, ..raw }),
        ("components/sleep+diamond+persistent", CheckConfig { mode: ProjectionMode::Components, explore: Options::ALL, ..raw }),
        ("per-resource/plain", per_resource),
        ("per-resource/sleep", CheckConfig { explore: sleep, ..per_resource }),
        ("per-resource/diamond", CheckConfig { explore: Options { strong_diamonds: true, ..Options::NONE }, ..per_resource }),
        ("per-resource/sleep+diamond+persistent", CheckConfig { explore: Options::ALL, ..per_resource }),
        ("per-resource/all+fingerprint", CheckConfig { explore: Options::ALL, fingerprints: true, ..per_resource }),
        ("per-resource/all+fingerprint+certificates", CheckConfig { explore: Options::ALL, fingerprints: true, certificates: true, ..per_resource }),
    ]
}

fn table(name: &str, events: &[SyncEvent]) {
    eprintln!("\n== {name}: {} events ==", events.len());
    eprintln!(
        "{:<44} {:>10} {:>12} {:>6} {:>6} {:>6} {:>10}  verdict",
        "config", "states", "transitions", "progs", "reused", "cert", "time"
    );
    for (label, config) in configs() {
        let started = Instant::now();
        let report = check(events, &config);
        let elapsed = started.elapsed();
        eprintln!(
            "{:<44} {:>10} {:>12} {:>6} {:>6} {:>6} {:>9.1?}  {:?} ({})",
            label,
            report.stats.visited_states,
            report.stats.explored_transitions,
            report.stats.programs,
            report.stats.reused_clean_programs,
            report.stats.certified_programs,
            elapsed,
            report.verdict,
            report.termination,
        );
    }
}

fn main() {
    let plain = pipeline(SHAPE);
    let tma = pipeline(PipelineShape { tma_bytes: 4096, ..SHAPE });
    // Small enough that the unprojected product space fits the budget.
    table(
        "4 warps x 2 stages x 8 iterations",
        &pipeline(PipelineShape { warps: 4, stages: 2, iterations: 8, tma_bytes: 0 }),
    );
    table("16 warps x 4 stages x 32 iterations", &plain);
    table("same, TMA producer (arrive.expect_tx + async completion)", &tma);

    let mut criterion = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_secs(2))
        .configure_from_args();
    let mut group = criterion.benchmark_group("pipeline_16x4x32");
    for (label, config) in configs() {
        group.bench_with_input(BenchmarkId::from_parameter(label), &plain, |b, events| {
            b.iter(|| black_box(check(events, &config)))
        });
    }
    group.finish();
    let mut group = criterion.benchmark_group("pipeline_16x4x32_tma");
    for (label, config) in configs().into_iter().skip(7) {
        group.bench_with_input(BenchmarkId::from_parameter(label), &tma, |b, events| {
            b.iter(|| black_box(check(events, &config)))
        });
    }
    group.finish();
    criterion.final_summary();
}
