//! Scratch (W2): run a dumped module + inputs.
use numsim_core::arena::BitSet;
use numsim_core::sched::{self, ArgValue, Backend, Inputs, RunConfig};
use std::collections::BTreeMap;
fn unhex(s: &str) -> Vec<u8> { (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect() }
pub fn load(prefix: &str) -> (numsim_core::Module, Inputs) {
    let m: numsim_core::Module = serde_json::from_str(&std::fs::read_to_string(format!("{prefix}.module.json")).unwrap()).unwrap();
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{prefix}.inputs.json")).unwrap()).unwrap();
    let mut args = BTreeMap::new();
    for (k, a) in v.as_object().unwrap() {
        let arg = match a["kind"].as_str().unwrap() {
            "buffer" => {
                let bytes = unhex(a["hex"].as_str().unwrap());
                let valid = a["mask"].as_str().map(|m| {
                    let m = unhex(m);
                    let mut b = BitSet::new(m.len() as u64, false);
                    for (i, x) in m.iter().enumerate() { if *x != 0 { b.set_range(i as u64, 1, true); } }
                    b
                });
                ArgValue::Buffer { bytes, valid }
            }
            "scalar" => ArgValue::Scalar(a["v"].as_i64().map(|x| x as u64).unwrap_or_else(|| a["v"].as_u64().unwrap())),
            "tensor_map" => ArgValue::TensorMap(unhex(a["hex"].as_str().unwrap())),
            "tensor_map_of" => {
                let img: [u8; 128] = unhex(a["hex"].as_str().unwrap()).try_into().unwrap();
                ArgValue::TensorMapOf { base: a["base"].as_str().unwrap().into(), offset: a["offset"].as_u64().unwrap(), desc: numsim_core::oplib::TensorMapDesc::decode(&img).unwrap() }
            }
            "pointer" => ArgValue::Pointer { target: a["target"].as_str().unwrap().into(), offset: a["offset"].as_u64().unwrap() },
            "view" => ArgValue::View { target: a["target"].as_str().unwrap().into(), offset: a["offset"].as_u64().unwrap(), len: a["len"].as_u64().unwrap() },
            _ => panic!(),
        };
        args.insert(k.clone(), arg);
    }
    (m, Inputs { args, ..Default::default() })
}
fn main() {
    let prefix = std::env::args().nth(1).unwrap();
    let out = std::env::args().nth(2).unwrap_or_default();
    let (m, inputs) = load(&prefix);
    for (seed, eager, single) in [(0u64, false, false), (1, false, false), (0, true, false), (0, true, true)] {
        let cfg = RunConfig { seed, single_partition: single, completions: if eager { sched::CompletionPolicy::Eager } else { sched::CompletionPolicy::Seeded }, ..RunConfig::default() };
        let mut obs = numsim_core::observe::NoopObserver;
        let o = sched::run_with_config(&m, &inputs, &mut obs, &Backend::Interp, &cfg).unwrap();
        println!("seed={seed} eager={eager} single={single}: {:?} rounds={} instrs={}", o.status, o.stats.rounds, o.stats.instrs);
        if let Some((b, _)) = o.outputs.buffers.get(&out) {
            println!("  {out}[..32] = {:02x?}", &b[..b.len().min(32)]);
        }
    }
}
