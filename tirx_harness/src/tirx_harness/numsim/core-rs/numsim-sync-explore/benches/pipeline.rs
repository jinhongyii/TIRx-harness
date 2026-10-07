//! Visited states and time for a synthetic 16-warp x 4-stage x 32-iteration
//! producer/consumer pipeline (contract `SyncEvent` log) under each reduction.
//!
//! `cargo bench -p numsim-sync-explore --bench pipeline` first prints the
//! reduction tables, then runs criterion timings. Unprojected configurations
//! exhaust the state budget: that is the point of the table.

use std::time::{Duration, Instant};

use criterion::{black_box, BenchmarkId, Criterion};
use numsim_core::observe::RecordingObserver;
use numsim_core::sync::ResourceInit;
use numsim_sync_explore::build::pipeline;
use numsim_sync_explore::explore::Options;
use numsim_sync_explore::{check, ProjectionMode, SynccheckConfig};

fn configs(warps: u32) -> Vec<(&'static str, SynccheckConfig)> {
    let raw = SynccheckConfig {
        mode: ProjectionMode::Whole,
        certificates: false,
        fingerprints: false,
        explore: Options::NONE,
        state_budget: 100_000,
        transition_budget: 2_000_000,
        init: ResourceInit { warps_per_cta: warps, cluster_warps: warps, ..ResourceInit::default() },
        ..SynccheckConfig::default()
    };
    let sleep = Options { sleep_sets: true, ..Options::NONE };
    let per = SynccheckConfig { mode: ProjectionMode::PerResource, ..raw.clone() };
    vec![
        ("whole/plain", raw.clone()),
        ("whole/sleep", SynccheckConfig { explore: sleep, ..raw.clone() }),
        ("whole/sleep+diamond+persistent", SynccheckConfig { explore: Options::ALL, ..raw.clone() }),
        ("components/sleep+diamond+persistent", SynccheckConfig { mode: ProjectionMode::Components, explore: Options::ALL, ..raw }),
        ("per-resource/plain", per.clone()),
        ("per-resource/sleep", SynccheckConfig { explore: sleep, ..per.clone() }),
        ("per-resource/diamond", SynccheckConfig { explore: Options { strong_diamonds: true, ..Options::NONE }, ..per.clone() }),
        ("per-resource/sleep+diamond+persistent", SynccheckConfig { explore: Options::ALL, ..per.clone() }),
        ("per-resource/all+fingerprint", SynccheckConfig { explore: Options::ALL, fingerprints: true, ..per.clone() }),
        ("per-resource/all+fingerprint+certificates", SynccheckConfig { explore: Options::ALL, fingerprints: true, certificates: true, ..per }),
    ]
}

fn stat(r: &numsim_core::report::Report, key: &str) -> u64 {
    r.coverage.iter().find(|(k, _)| k == key).map_or(0, |(_, v)| *v)
}

fn table(name: &str, warps: u32, log: &RecordingObserver) {
    eprintln!("\n== {name}: {} events ==", log.total());
    eprintln!("{:<44} {:>10} {:>12} {:>6} {:>6} {:>6} {:>10}  verdict", "config", "states", "transitions", "progs", "reused", "cert", "time");
    for (label, cfg) in configs(warps) {
        let t = Instant::now();
        let r = check(log, &cfg);
        eprintln!(
            "{label:<44} {:>10} {:>12} {:>6} {:>6} {:>6} {:>9.1?}  {:?}",
            stat(&r, "visited_state_count"),
            stat(&r, "explored_transition_count"),
            stat(&r, "program_count"),
            stat(&r, "reused_clean_program_count"),
            stat(&r, "certified_program_count"),
            t.elapsed(),
            r.verdict,
        );
    }
}

fn main() {
    let plain = pipeline(16, 4, 32, 0);
    let tma = pipeline(16, 4, 32, 4096);
    table("4 warps x 2 stages x 8 iterations", 4, &pipeline(4, 2, 8, 0));
    table("16 warps x 4 stages x 32 iterations", 16, &plain);
    table("same, TMA producer (arrive.expect_tx + async completion)", 16, &tma);

    let mut criterion = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_secs(2))
        .configure_from_args();
    for (group_name, log) in [("pipeline_16x4x32", &plain), ("pipeline_16x4x32_tma", &tma)] {
        let mut group = criterion.benchmark_group(group_name);
        for (label, cfg) in configs(16).into_iter().skip(6) {
            group.bench_with_input(BenchmarkId::from_parameter(label), log, |b, log| b.iter(|| black_box(check(log, &cfg))));
        }
        group.finish();
    }
    criterion.final_summary();
}
