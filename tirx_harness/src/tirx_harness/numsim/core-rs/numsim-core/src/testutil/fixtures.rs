//! Recorded native-run fixtures (`examples/record_race_fixtures.py`):
//! `<dir>/<case>.{module,inputs,kw}.json`, the exact arguments the Python
//! runner passes to the engine. Shared by the racecheck tuning table and
//! the recorded-stream benches; fixtures are build artefacts under
//! `core-rs/target/race-fixtures` and never committed.

use crate::program::Module;
use crate::sched::{ArgValue, Inputs, RunConfig};

/// Default fixture directory (`$RACE_FIXTURES`, else `core-rs/target/race-fixtures`).
pub fn dir() -> String {
    let core_rs = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
    std::env::var("RACE_FIXTURES").unwrap_or_else(|_| core_rs.join("target/race-fixtures").display().to_string())
}

/// Are all three files of `case` present in `dir`?
pub fn exists(dir: &str, case: &str) -> bool {
    ["module", "inputs", "kw"].iter().all(|e| std::path::Path::new(&format!("{dir}/{case}.{e}.json")).exists())
}

fn hex(s: &str) -> Vec<u8> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

pub fn load(dir: &str, case: &str) -> (Module, Inputs, RunConfig) {
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
            "tensor_map" => ArgValue::TensorMap(hex(x["hex"].as_str().unwrap())),
            "tensor_map_of" => {
                let image: [u8; 128] = hex(x["hex"].as_str().unwrap()).try_into().unwrap();
                ArgValue::TensorMapOf {
                    base: x["base"].as_str().unwrap().into(),
                    offset: x["offset"].as_u64().unwrap(),
                    desc: crate::oplib::TensorMapDesc::decode(&image).unwrap(),
                }
            }
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
    if let Some(b) = kw["loop_budget"].as_u64() {
        config.loop_budget = b;
    }
    if let Some(q) = kw["quantum"].as_u64() {
        config.quantum = q as u32;
    }
    if let Some(s) = kw["seed"].as_u64() {
        config.seed = s;
    }
    (module, inputs, config)
}

