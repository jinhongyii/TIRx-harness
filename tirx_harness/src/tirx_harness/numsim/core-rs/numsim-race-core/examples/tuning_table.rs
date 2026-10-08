//! On/off ratio of each racecheck pruning technique on corpus-shaped runs.
//!
//! Usage: `tuning_table <fixture-dir> <case>...` where `<dir>/<case>.{module,inputs,kw}.json`
//! were dumped from a corpus racecheck run (`numsim_core_py` mode=racecheck).
//! Prints one row per switch: wall time of the run with that technique off
//! vs all on (engine time measured separately under NoopObserver and
//! subtracted), plus whether the findings were unchanged.
use numsim_core::observe::NoopObserver;
use numsim_core::program::Module;
use numsim_core::racecheck::observer::{RaceObserver, RacecheckConfig};
use numsim_core::racecheck::tuning;
use numsim_core::sched::{self, ArgValue, Backend, Inputs, RunConfig};
use std::sync::atomic::Ordering::Relaxed;
use std::time::Instant;

fn hex(s: &str) -> Vec<u8> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

fn load(dir: &str, case: &str) -> (Module, Inputs, RunConfig) {
    let rd = |ext: &str| std::fs::read_to_string(format!("{dir}/{case}.{ext}.json")).unwrap();
    let module = Module::from_json(&rd("module")).unwrap();
    let v: serde_json::Value = serde_json::from_str(&rd("inputs")).unwrap();
    let kw: serde_json::Value = serde_json::from_str(&rd("kw")).unwrap();
    let mut inputs = Inputs::default();
    for (k, x) in v.as_object().unwrap() {
        let arg = match x["kind"].as_str().unwrap() {
            "buffer" => ArgValue::Buffer { bytes: hex(x["hex"].as_str().unwrap()), valid: None },
            "scalar" => ArgValue::Scalar(x["v"].as_i64().map(|v| v as u64).or_else(|| x["v"].as_u64()).unwrap()),
            "view" => ArgValue::View {
                target: x["target"].as_str().unwrap().into(),
                offset: x["offset"].as_u64().unwrap(),
                len: x["len"].as_u64().unwrap(),
            },
            "pointer" => ArgValue::Pointer { target: x["target"].as_str().unwrap().into(), offset: x["offset"].as_u64().unwrap() },
            other => panic!("input kind {other} not supported by this fixture loader"),
        };
        inputs.args.insert(k.clone(), arg);
    }
    if let Some(h) = kw["host_addrs"].as_object() {
        for (k, a) in h {
            inputs.host_addrs.insert(k.clone(), a.as_u64().unwrap());
        }
    }
    let mut config = RunConfig::default();
    if let Some(w) = kw["workers"].as_u64() {
        config.workers = w as usize;
    }
    (module, inputs, config)
}

fn race(m: &Module, i: &Inputs, c: &RunConfig) -> (f64, usize, usize) {
    let mut obs = RaceObserver::new(RacecheckConfig::default());
    let t = Instant::now();
    sched::run_with_config(m, i, &mut obs, &Backend::Interp, c).unwrap();
    let r = obs.finish();
    (t.elapsed().as_secs_f64(), r.findings.len(), r.findings.iter().map(|f| f.attrs.get("occurrences").and_then(|v| v.as_u64()).unwrap_or(1) as usize).sum())
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let reps: usize = std::env::var("REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    println!("| case | technique | engine (s) | racecheck on (s) | racecheck off (s) | off/on | findings same |");
    println!("| --- | --- | --- | --- | --- | --- | --- |");
    for case in &a[2..] {
        let (m, i, c) = load(&a[1], case);
        let best = |f: &dyn Fn() -> f64| (0..reps).map(|_| f()).fold(f64::INFINITY, f64::min);
        let engine = best(&|| {
            let t = Instant::now();
            sched::run_with_config(&m, &i, &mut NoopObserver, &Backend::Interp, &c).unwrap();
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
