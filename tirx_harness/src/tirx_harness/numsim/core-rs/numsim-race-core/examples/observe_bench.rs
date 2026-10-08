//! Times one module under NoopObserver and RaceObserver.
//! Usage: observe_bench <module.json> <inputs.json> [max_rounds] [race|noop|both]
use numsim_core::observe::NoopObserver;
use numsim_core::program::Module;
use numsim_core::racecheck::observer::{RaceObserver, RacecheckConfig};
use numsim_core::sched::{self, ArgValue, Backend, Inputs, RunConfig};
use std::time::Instant;

fn hex(s: &str) -> Vec<u8> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let module = Module::from_json(&std::fs::read_to_string(&a[1]).unwrap()).unwrap();
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&a[2]).unwrap()).unwrap();
    let mut inputs = Inputs::default();
    for (k, x) in v.as_object().unwrap() {
        let arg = match x["kind"].as_str().unwrap() {
            "buffer" => ArgValue::Buffer { bytes: hex(x["hex"].as_str().unwrap()), valid: None },
            other => panic!("input kind {other}"),
        };
        inputs.args.insert(k.clone(), arg);
    }
    let mut config = RunConfig::default();
    if let Some(r) = a.get(3) {
        config.max_rounds = r.parse().unwrap();
    }
    let mode = a.get(4).map(String::as_str).unwrap_or("both");
    if mode != "race" {
        let t = Instant::now();
        let o = sched::run_with_config(&module, &inputs, &mut NoopObserver, &Backend::Interp, &config).unwrap();
        println!("noop {:.3}s {:?} rounds={}", t.elapsed().as_secs_f64(), o.status, o.stats.rounds);
    }
    if mode != "noop" {
        let mut obs = RaceObserver::new(RacecheckConfig::default());
        let t = Instant::now();
        let o = sched::run_with_config(&module, &inputs, &mut obs, &Backend::Interp, &config).unwrap();
        let run = t.elapsed().as_secs_f64();
        let r = obs.finish();
        println!("race {:.3}s (+finish {:.3}s) {:?} findings={}", run, t.elapsed().as_secs_f64() - run, o.status, r.findings.len());
    }
}
