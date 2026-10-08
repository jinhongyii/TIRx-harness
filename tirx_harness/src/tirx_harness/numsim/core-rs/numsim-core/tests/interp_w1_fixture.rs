//! End to end on the `Module` JSON W1's lowering emits for the vector add of
//! `tests/numsim/v2/test_lowering_vector_add.py` (fixture regenerated with
//! `program.to_json()` wrapped as `{"format_version", "kernels": [..]}`).

use numsim_core::observe::RecordingObserver;
use numsim_core::sched::{self, ArgValue, Backend, Inputs, RunConfig, RunStatus};
use numsim_core::Module;

const FIXTURE: &str = include_str!("fixtures/vadd_w1.json");

fn f32_buf(v: impl IntoIterator<Item = f32>) -> ArgValue {
    ArgValue::Buffer { bytes: v.into_iter().flat_map(|x| x.to_le_bytes()).collect(), valid: None }
}

#[test]
fn w1_vector_add_json_runs() {
    let module = Module::from_json(FIXTURE).expect("W1 JSON decodes into the contract Module");
    module.kernels[0].validate().expect("valid");
    let inputs = Inputs {
        args: [
            ("a".to_string(), f32_buf((0..1024).map(|i| i as f32))),
            ("b".to_string(), f32_buf((0..1024).map(|i| 0.25 * i as f32))),
            ("c".to_string(), f32_buf([0.0; 1024])),
        ]
        .into_iter()
        .collect(),
        ..Default::default()
    };
    for seed in [0, 3] {
        let mut obs = RecordingObserver::new();
        let o = sched::run_with_config(&module, &inputs, &mut obs, &Backend::Interp, &RunConfig { seed, ..RunConfig::default() })
            .expect("run starts");
        assert_eq!(o.status, RunStatus::Completed);
        let c: Vec<f32> = o.outputs.buffers["c"].0.chunks(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        for (i, v) in c.iter().enumerate() {
            assert_eq!(*v, 1.25 * i as f32, "c[{i}]");
        }
        assert!(o.diagnostics.is_empty());
    }
}
