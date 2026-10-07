//! Differential tests: the codegen backend must be indistinguishable from
//! the interpreter (outputs bit for bit, every observer callback in order,
//! `RunOutcome` including status and stats) across seeds and quanta (small
//! quanta force resumption at arbitrary pcs of the generated step function).
//!
//! Building requires `rustc`/`cargo`; the tests that build are skipped
//! unless `NUMSIM_CODEGEN_TESTS=1`. Cache: `$NUMSIM_CODEGEN_CACHE` or
//! `<target>/tmp/numsim-codegen`.
//!
//! While the interpreter/scheduler bodies are unimplemented (W2), a
//! scenario whose *interpreter* run panics with `not implemented` is
//! reported as PENDING instead of failing.

mod codegen_scenarios;

use codegen_scenarios::Scenario;
use numsim_core::codegen::{self, BuildOptions, LoadedBackend, OptLevel};
use numsim_core::observe::{Access, CtaId, LaunchInfo, Observer, RecordingObserver, SyncEvent, WarpEnd, WarpId};
use numsim_core::sched::{self, Backend, RunConfig, RunOutcome};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::time::Instant;

fn enabled() -> bool {
    std::env::var("NUMSIM_CODEGEN_TESTS").is_ok_and(|v| v == "1")
}

fn cache_dir() -> PathBuf {
    std::env::var_os("NUMSIM_CODEGEN_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("numsim-codegen"))
}

/// Every observer callback, in delivery order, as text.
#[derive(Default)]
struct TraceObserver {
    log: Vec<String>,
}

impl Observer for TraceObserver {
    fn wants_word_history(&self) -> bool {
        true
    }
    fn begin_launch(&mut self, i: &LaunchInfo<'_>) {
        self.log.push(format!("begin_launch k{} {:?}", i.kernel_index, i.shape));
    }
    fn end_launch(&mut self, i: &LaunchInfo<'_>) {
        self.log.push(format!("end_launch k{} {:?}", i.kernel_index, i.shape));
    }
    fn access(&mut self, a: &Access<'_>) {
        self.log.push(format!("access {a:?}"));
    }
    fn sync(&mut self, e: &SyncEvent) {
        self.log.push(format!("sync {e:?}"));
    }
    fn warp_done(&mut self, w: WarpId, end: WarpEnd) {
        self.log.push(format!("warp_done {w:?} {end:?}"));
    }
    fn inbox_drain(&mut self, c: CtaId, r: u64) {
        self.log.push(format!("inbox_drain {c:?} {r}"));
    }
}

#[derive(Debug, PartialEq)]
enum RunResult {
    Ran { outcome: Box<RunOutcome>, trace: Vec<String>, recorded: RecordingObserver },
    RunError(String),
    Panic(String),
}

fn panic_msg(p: Box<dyn std::any::Any + Send>) -> String {
    numsim_core::codegen::rt::panic_message(&*p)
}

fn run(s: &Scenario, backend: &Backend, config: &RunConfig) -> RunResult {
    let r = catch_unwind(AssertUnwindSafe(|| {
        let mut obs = (TraceObserver::default(), RecordingObserver::new());
        let res = sched::run_with_config(&s.module, &s.inputs, &mut obs, backend, config);
        (res, obs)
    }));
    match r {
        Ok((Ok(outcome), (trace, recorded))) => RunResult::Ran { outcome: Box::new(outcome), trace: trace.log, recorded },
        Ok((Err(e), _)) => RunResult::RunError(format!("{e:?}")),
        Err(p) => RunResult::Panic(panic_msg(p)),
    }
}

fn configs() -> Vec<RunConfig> {
    let mut v = Vec::new();
    for seed in [0u64, 7] {
        for quantum in [1u32, 5, 256] {
            v.push(RunConfig { seed, quantum, ..RunConfig::default() });
        }
    }
    v
}

enum Verdict {
    Same,
    Pending(String),
    Differ(String),
}

/// An interpreter panic shows up in the codegen run either as the same
/// panic (raised in host code, e.g. the scheduler) or as an
/// `Internal` error carrying the message (raised in generated code).
fn compare(interp: &RunResult, cg: &RunResult) -> Verdict {
    if let RunResult::Panic(m) = interp {
        if m.contains("not implemented") {
            return Verdict::Pending(m.clone());
        }
        return match cg {
            RunResult::Panic(m2) if m2 == m => Verdict::Same,
            RunResult::Ran { outcome, .. } => match &outcome.status {
                sched::RunStatus::Error(e) if e.message == format!("panic: {m}") => Verdict::Same,
                other => Verdict::Differ(format!("interp panicked ({m}), codegen status {other:?}")),
            },
            other => Verdict::Differ(format!("interp panicked ({m}), codegen {other:?}")),
        };
    }
    if interp == cg {
        return Verdict::Same;
    }
    let detail = match (interp, cg) {
        (RunResult::Ran { outcome: a, trace: ta, .. }, RunResult::Ran { outcome: b, trace: tb, .. }) => {
            if a.status != b.status {
                format!("status: interp {:?} codegen {:?}", a.status, b.status)
            } else if a.outputs != b.outputs {
                "outputs differ".to_string()
            } else if a.stats != b.stats {
                format!("stats: interp {:?} codegen {:?}", a.stats, b.stats)
            } else if let Some(i) = ta.iter().zip(tb.iter()).position(|(x, y)| x != y) {
                format!("observer event {i}: interp `{}` codegen `{}`", ta[i], tb[i])
            } else if ta.len() != tb.len() {
                format!("observer stream length: interp {} codegen {}", ta.len(), tb.len())
            } else {
                "RecordingObserver / sync leftovers differ".to_string()
            }
        }
        _ => format!("interp {interp:?}\ncodegen {cg:?}"),
    };
    Verdict::Differ(detail)
}

fn build(s: &Scenario, opt: OptLevel) -> LoadedBackend {
    let opts = BuildOptions::new(cache_dir()).opt(opt);
    match codegen::build_module(&s.module, &opts) {
        Ok(b) => b,
        Err(e) => panic!("{}: {e}", s.name),
    }
}

fn check_scenarios(opt: OptLevel, names: Option<&[&str]>) {
    let mut pending = Vec::new();
    let mut failures = Vec::new();
    let mut same = 0usize;
    for s in codegen_scenarios::all() {
        if names.is_some_and(|n| !n.contains(&s.name)) {
            continue;
        }
        let t = Instant::now();
        let loaded = build(&s, opt);
        let build_s = t.elapsed().as_secs_f64();
        let cg_backend = loaded.backend();
        let mut verdicts = Vec::new();
        for cfg in configs() {
            let a = run(&s, &Backend::Interp, &cfg);
            let b = run(&s, &cg_backend, &cfg);
            verdicts.push((cfg.seed, cfg.quantum, compare(&a, &b)));
        }
        let mut line = format!("{:<16} {:?} build {:>6.2}s:", s.name, opt, build_s);
        for (seed, q, v) in verdicts {
            match v {
                Verdict::Same => {
                    same += 1;
                    line.push_str(" ok");
                }
                Verdict::Pending(m) => {
                    line.push_str(" pending");
                    pending.push(format!("{} seed={seed} q={q}: {m}", s.name));
                }
                Verdict::Differ(d) => {
                    line.push_str(" DIFF");
                    failures.push(format!("{} seed={seed} q={q}: {d}", s.name));
                }
            }
        }
        eprintln!("{line}");
    }
    eprintln!("codegen equivalence ({opt:?}): {same} identical, {} pending (W2), {} different", pending.len(), failures.len());
    if let Some(p) = pending.first() {
        eprintln!("first pending: {p}");
    }
    assert!(failures.is_empty(), "interp/codegen differ:\n{}", failures.join("\n"));
}

#[test]
fn codegen_matches_interp_o1() {
    if !enabled() {
        eprintln!("skipped: set NUMSIM_CODEGEN_TESTS=1");
        return;
    }
    check_scenarios(OptLevel::O1, None);
}

#[test]
fn codegen_matches_interp_o3() {
    if !enabled() {
        eprintln!("skipped: set NUMSIM_CODEGEN_TESTS=1");
        return;
    }
    check_scenarios(OptLevel::O3, Some(&["vadd", "scalar_loop", "memory_heavy", "break_continue"]));
}

/// Build pipeline without executing: load, ABI fingerprint, module check,
/// cache hit on rebuild, mismatch rejection.
#[test]
fn build_load_and_cache() {
    if !enabled() {
        eprintln!("skipped: set NUMSIM_CODEGEN_TESTS=1");
        return;
    }
    let s = codegen_scenarios::vadd();
    let first = build(&s, OptLevel::O1);
    assert_eq!(first.steps.len(), 1);
    let again = build(&s, OptLevel::O1);
    assert_eq!(again.stats.rustc, None, "second build must hit the cache");
    assert_eq!(again.library, first.library);
    let other = codegen_scenarios::scalar_loop(3);
    assert!(first.check_module(&other.module).is_err(), "library must reject a different module");
    assert!(codegen::load(&first.library, &s.module).is_ok());
    assert!(codegen::load(&first.library, &other.module).is_err());
    let b = codegen::backend_for(&s.module, &BuildOptions::new("/nonexistent"), &cache_dir()).unwrap();
    assert!(matches!(b, Backend::Codegen(ref v) if v.len() == 1));
    let two = build(&codegen_scenarios::two_kernels(), OptLevel::O1);
    assert_eq!(two.steps.len(), 2);
}

/// Compile time per scenario at O1 and O3 (fresh compiles; core rlib cached).
#[test]
fn report_build_times() {
    if !enabled() {
        eprintln!("skipped: set NUMSIM_CODEGEN_TESTS=1");
        return;
    }
    let dir = cache_dir();
    for s in codegen_scenarios::all() {
        let mut row = format!("{:<16} {:>4} instrs", s.name, s.module.kernels.iter().map(|k| k.code.len()).sum::<usize>());
        for opt in [OptLevel::O1, OptLevel::O3] {
            let opts = BuildOptions { force: true, ..BuildOptions::new(&dir).opt(opt) };
            let b = codegen::build_module(&s.module, &opts).unwrap();
            row.push_str(&format!(
                "  {opt:?}: rustc {:>5.2}s ({} KiB src)",
                b.stats.rustc.map(|d| d.as_secs_f64()).unwrap_or(0.0),
                b.stats.source_bytes / 1024
            ));
        }
        eprintln!("{row}");
    }
}
