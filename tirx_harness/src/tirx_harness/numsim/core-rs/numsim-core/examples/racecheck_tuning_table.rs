//! On/off ratio of each racecheck pruning technique on corpus-shaped runs.
//!
//! Usage: `cargo run --release -p numsim-core --example racecheck_tuning_table [<case>...]` (default: the three cases below).
//! Fixtures `<dir>/<case>.{module,inputs,kw}.json` are the exact native-run
//! arguments of the case's first racecheck phase. `<dir>` is `$RACE_FIXTURES`,
//! or `core-rs/target/race-fixtures`. Missing fixtures are recorded first by
//! `numsim-core/examples/record_race_fixtures.py` (run with `$PY`, else `python3`;
//! source `scripts/dev-env.sh` first). The script caches by module + inputs
//! hash, and fixtures are never committed.
//! Prints one row per switch: wall time of the run with that technique off
//! vs all on (engine time measured separately under NoopObserver and
//! subtracted), plus whether the findings were unchanged. `REPS` (default 3)
//! takes the best of that many runs; `ONLY=NAME,...` limits the switches.
use numsim_core::observe::NoopObserver;
use numsim_core::program::Module;
use numsim_core::racecheck::observer::{RaceObserver, RacecheckConfig};
use numsim_core::racecheck::tuning;
use numsim_core::sched::{self, Inputs, RunConfig};
use std::sync::atomic::Ordering::Relaxed;
use std::time::Instant;

use numsim_core::testutil::fixtures::load;

fn race(m: &Module, i: &Inputs, c: &RunConfig) -> (f64, usize, usize) {
    let mut obs = RaceObserver::new(RacecheckConfig::default());
    let t = Instant::now();
    sched::run_with_config(m, i, &mut obs, c).unwrap();
    let r = obs.finish();
    (t.elapsed().as_secs_f64(), r.findings.len(), r.findings.iter().map(|f| f.attrs.get("occurrences").and_then(|v| v.as_u64()).unwrap_or(1) as usize).sum())
}

const DEFAULT_CASES: [&str; 3] = ["fp16_bf16_gemm", "radix_topk_multi_cta", "gdn_decode_bf16_wide_vec_mtp"];

/// Fixture directory, recording any missing case through the Python side.
fn fixtures(cases: &[String]) -> String {
    let core_rs = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
    let dir = std::env::var("RACE_FIXTURES").unwrap_or_else(|_| core_rs.join("target/race-fixtures").display().to_string());
    let missing: Vec<&String> = cases
        .iter()
        .filter(|c| ["module", "inputs", "kw"].iter().any(|e| !std::path::Path::new(&format!("{dir}/{c}.{e}.json")).exists()))
        .collect();
    if !missing.is_empty() {
        // core-rs -> numsim -> tirx_harness(pkg) -> src -> tirx_harness -> repo
        let repo = core_rs.ancestors().nth(5).unwrap();
        let py = std::env::var("PY").unwrap_or_else(|_| "python3".into());
        let st = std::process::Command::new(py)
            .current_dir(repo.join("tirx_harness"))
            .arg(core_rs.join("numsim-core/examples/record_race_fixtures.py"))
            .arg(&dir)
            .args(missing.iter().map(|s| s.as_str()))
            .status()
            .expect("cannot run the fixture recorder");
        assert!(st.success(), "record_race_fixtures.py failed");
    }
    dir
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let cases: Vec<String> = if a.len() > 1 { a[1..].to_vec() } else { DEFAULT_CASES.iter().map(|s| s.to_string()).collect() };
    let dir = fixtures(&cases);
    // `ONLY=A,B`: measure just these switches.
    let only = std::env::var("ONLY").ok();
    let reps: usize = std::env::var("REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    println!("| case | technique | engine (s) | racecheck on (s) | racecheck off (s) | off/on | findings same |");
    println!("| --- | --- | --- | --- | --- | --- | --- |");
    for case in &cases {
        let (m, i, c) = load(&dir, case);
        let best = |f: &dyn Fn() -> f64| (0..reps).map(|_| f()).fold(f64::INFINITY, f64::min);
        let engine = best(&|| {
            let t = Instant::now();
            sched::run_with_config(&m, &i, &mut NoopObserver, &c).unwrap();
            t.elapsed().as_secs_f64()
        });
        tuning::set_all(true);
        let mut base = (f64::INFINITY, 0, 0);
        for _ in 0..reps {
            let r = race(&m, &i, &c);
            if r.0 < base.0 {
                base = r;
            }
        }
        let on = (base.0 - engine).max(1e-6);
        for (name, s) in tuning::all() {
            if only.as_ref().is_some_and(|o| !o.split(',').any(|x| x == name)) {
                continue;
            }
            s.store(false, Relaxed);
            let mut off = (f64::INFINITY, 0, 0);
            for _ in 0..reps {
                let r = race(&m, &i, &c);
                if r.0 < off.0 {
                    off = r;
                }
            }
            s.store(true, Relaxed);
            let offc = (off.0 - engine).max(1e-6);
            println!(
                "| {case} | {name} | {engine:.3} | {on:.3} | {offc:.3} | {:.2}x | {} |",
                offc / on,
                if (off.1, off.2) == (base.1, base.2) { "yes".to_string() } else { format!("no ({} vs {})", off.1, base.1) }
            );
        }
    }
}
