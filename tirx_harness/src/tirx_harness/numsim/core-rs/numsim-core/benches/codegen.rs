//! Interp vs codegen (O1 / O3) on a register-bound loop and a memory-bound
//! loop. `cargo bench -p numsim-core --bench codegen`. Generated libraries
//! are cached under `<target>/tmp/numsim-codegen-bench`.

#[path = "../tests/codegen_scenarios/mod.rs"]
mod codegen_scenarios;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use numsim_core::codegen::{self, BuildOptions, OptLevel};
use numsim_core::observe::NoopObserver;
use numsim_core::sched::{self, Backend, RunConfig};
use std::panic::{catch_unwind, AssertUnwindSafe};

fn cache_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("numsim-codegen-bench")
}

fn bench(c: &mut Criterion) {
    let config = RunConfig::default();
    for s in [codegen_scenarios::scalar_loop(2000), codegen_scenarios::memory_heavy(500)] {
        let mut backends = vec![("interp", Backend::Interp)];
        for opt in [OptLevel::O1, OptLevel::O3] {
            let loaded = codegen::build_module(&s.module, &BuildOptions::new(cache_dir()).opt(opt))
                .unwrap_or_else(|e| panic!("{}: {e}", s.name));
            backends.push((if opt == OptLevel::O1 { "codegen-O1" } else { "codegen-O3" }, loaded.backend()));
        }
        let probe = catch_unwind(AssertUnwindSafe(|| {
            sched::run_with_config(&s.module, &s.inputs, &mut NoopObserver, &Backend::Interp, &config)
        }));
        if let Err(p) = probe {
            eprintln!("{}: skipped, interpreter not runnable yet: {}", s.name, codegen::rt::panic_message(&*p));
            continue;
        }
        let mut g = c.benchmark_group(s.name);
        g.sample_size(20);
        for (name, backend) in &backends {
            g.bench_with_input(BenchmarkId::from_parameter(name), backend, |b, backend| {
                b.iter(|| sched::run_with_config(&s.module, &s.inputs, &mut NoopObserver, backend, &config).unwrap())
            });
        }
        g.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
