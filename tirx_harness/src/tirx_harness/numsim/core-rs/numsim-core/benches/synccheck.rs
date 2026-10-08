//! Synccheck reduction guards (W6, CLAUDE.md "keep pruning techniques
//! guarded by their criterion benchmarks").
//!
//! Every reduction of the explorer has one row: a scenario generator
//! (`synccheck::build`), the configuration with everything on, and the same
//! configuration with that one technique off. `cargo bench -p numsim-core
//! --bench synccheck` first prints the on/off table (states, time, ratio;
//! table in `docs/development/synccheck-explorer.md` §5.9), then times the
//! "on" configuration of each row with criterion, so a regression in any
//! reduction shows up as a time jump of at least 2x (the off column).
//! `SYNCCHECK_TABLE_ONLY=1` prints the table and skips criterion.

use std::time::{Duration, Instant};

use criterion::{black_box, Criterion};
use numsim_core::observe::RecordingObserver;
use numsim_core::report::Report;
use numsim_core::sync::ResourceInit;
use numsim_core::synccheck::build::{per_lane_arrivals, pipeline, tma_many_waiters, umma_ring};
use numsim_core::synccheck::explore::{Options, Rules};
use numsim_core::synccheck::{check, ProjectionMode, SynccheckConfig};

/// Cap for the "off" runs: hitting it is the state-count cliff.
const BUDGET: u64 = 200_000;

struct Row {
    technique: &'static str,
    scenario: &'static str,
    log: RecordingObserver,
    on: SynccheckConfig,
    off: SynccheckConfig,
}

fn base(warps: u32) -> SynccheckConfig {
    SynccheckConfig {
        mode: ProjectionMode::PerResource,
        certificates: false,
        fingerprints: false,
        explore: Options::ALL,
        state_budget: BUDGET,
        transition_budget: 20 * BUDGET,
        init: ResourceInit { warps_per_cta: warps, cluster_warps: warps, ..ResourceInit::default() },
        ..SynccheckConfig::default()
    }
}

fn rules_off(cfg: &SynccheckConfig, f: impl Fn(&mut Rules)) -> SynccheckConfig {
    let mut c = cfg.clone();
    f(&mut c.explore.rules);
    c
}

fn rows() -> Vec<Row> {
    let mut rows = Vec::new();
    let pipe = || pipeline(6, 2, 8, 0);
    let b = base(6);
    rows.push(Row { technique: "per-resource projection", scenario: "pipeline(6,2,8)", log: pipe(), on: b.clone(), off: SynccheckConfig { mode: ProjectionMode::Whole, ..b.clone() } });
    // Gates are a soundness requirement of per-resource projections: off,
    // the projection over-approximates and reports a false error. The
    // guard is the Clean assertion on the "on" run, not the time ratio.
    rows.push(Row { technique: "HB gates", scenario: "pipeline(6,2,8)", log: pipe(), on: b.clone(), off: SynccheckConfig { hb_gates: false, ..b.clone() } });
    let t16 = base(16);
    let tma_only = |o: Options| SynccheckConfig { explore: o, ..t16.clone() };
    rows.push(Row { technique: "sleep sets (alone)", scenario: "tma_many_waiters(16)", log: tma_many_waiters(16), on: tma_only(Options { sleep_sets: true, ..Options::NONE }), off: tma_only(Options::NONE) });
    rows.push(Row { technique: "strong diamonds (with sleep, no persistent)", scenario: "tma_many_waiters(16)", log: tma_many_waiters(16), on: tma_only(Options { sleep_sets: true, strong_diamonds: true, ..Options::NONE }), off: tma_only(Options { sleep_sets: true, ..Options::NONE }) });
    let p8 = base(8);
    rows.push(Row { technique: "fingerprint dedup", scenario: "pipeline(8,8,64,1024)", log: pipeline(8, 8, 64, 1024), on: SynccheckConfig { fingerprints: true, ..p8.clone() }, off: p8.clone() });
    let umma = base(6);
    rows.push(Row { technique: "causal certificates", scenario: "umma_ring(6,16,16)", log: umma_ring(6, 16, 16), on: SynccheckConfig { certificates: true, ..umma.clone() }, off: umma.clone() });
    let b4 = base(4);
    rows.push(Row { technique: "tx terminal persistent", scenario: "pipeline(4,2,8,1024)", log: pipeline(4, 2, 8, 1024), on: b4.clone(), off: rules_off(&b4, |r| r.tx_terminal = false) });
    // Subsumed by the deferred-completion singleton when that is on.
    let one = rules_off(&base(3), |r| r.deferred_completion = false);
    rows.push(Row { technique: "twin landings (deferred singleton off)", scenario: "per_lane_arrivals(1,2,2,1)", log: per_lane_arrivals(1, 2, 2, 1), on: one.clone(), off: rules_off(&one, |r| r.twin_landings = false) });
    let priv_ = base(5);
    rows.push(Row { technique: "singleton: private async-group issue", scenario: "per_lane_arrivals(4,1,1,4)", log: per_lane_arrivals(4, 1, 1, 4), on: priv_.clone(), off: rules_off(&priv_, |r| r.private_issue = false) });
    let obs = base(9);
    rows.push(Row { technique: "singleton: ready observer", scenario: "per_lane_arrivals(1,8,2,1)", log: per_lane_arrivals(1, 8, 2, 1), on: obs.clone(), off: rules_off(&obs, |r| r.ready_observer = false) });
    let land = base(6);
    rows.push(Row { technique: "singleton: deferred completion", scenario: "per_lane_arrivals(4,2,2,1)", log: per_lane_arrivals(4, 2, 2, 1), on: land.clone(), off: rules_off(&land, |r| r.deferred_completion = false) });
    rows
}

fn stat(r: &Report, key: &str) -> u64 {
    r.coverage.iter().find(|(k, _)| k == key).map_or(0, |(_, v)| *v)
}

fn run(log: &RecordingObserver, cfg: &SynccheckConfig) -> (u64, u64, Duration, String) {
    let t = Instant::now();
    let r = check(log, cfg);
    (stat(&r, "visited_state_count"), stat(&r, "explored_transition_count"), t.elapsed(), format!("{:?}", r.verdict))
}

fn main() {
    let rows = rows();
    eprintln!("| technique | scenario | states on / off | transitions on / off | time on / off | time ratio | verdict on / off |");
    eprintln!("| --- | --- | ---: | ---: | ---: | ---: | --- |");
    for row in &rows {
        let (s_on, x_on, t_on, v_on) = run(&row.log, &row.on);
        assert_eq!(v_on, "Clean", "{}: the reduced search must stay Clean", row.technique);
        let (s_off, x_off, t_off, v_off) = run(&row.log, &row.off);
        let ratio = t_off.as_secs_f64() / t_on.as_secs_f64().max(1e-9);
        eprintln!(
            "| {} | {} | {} / {}{} | {} / {} | {:.1?} / {:.1?} | {:.1}x | {} / {} |",
            row.technique,
            row.scenario,
            s_on,
            s_off,
            if s_off >= BUDGET { " (budget)" } else { "" },
            x_on,
            x_off,
            t_on,
            t_off,
            ratio,
            v_on,
            v_off
        );
    }
    if std::env::var("SYNCCHECK_TABLE_ONLY").is_ok() {
        return;
    }
    let mut c = Criterion::default().sample_size(10).measurement_time(Duration::from_secs(3)).configure_from_args();
    let mut g = c.benchmark_group("synccheck");
    for row in &rows {
        g.bench_function(row.technique, |b| b.iter(|| black_box(check(&row.log, &row.on))));
    }
    g.finish();
    c.final_summary();
}
