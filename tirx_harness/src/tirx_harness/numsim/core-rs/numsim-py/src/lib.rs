//! Python bindings for `numsim-core` (W8): the extension module
//! `numsim_core_py`. Build with `--features extension-module` (see
//! `build_dev.sh`).
//!
//! Boundary contract: Python passes a serialized `Module` (JSON or postcard,
//! produced by lowering) plus already-canonicalized inputs; Rust returns raw
//! output bytes, a status dict and, for checker modes, the JSON
//! `report::Report`. No engine types cross the boundary.
//!
//! The engine-facing part ([`decode_module`], [`execute`]) is plain Rust so
//! it is unit-tested without Python. Bodies that are still
//! `unimplemented!()` in `numsim-core` surface as
//! [`ExecuteError::NotImplemented`] (Python: `NotImplementedError`).

pub use numsim_core;

use numsim_core::arena::{BitSet, ValidityPolicy};
use numsim_core::interp::ExecError;
use numsim_core::codegen::{self, BuildOptions, OptLevel};
use numsim_core::observe::{Access, CtaId, LaunchInfo, NoopObserver, Observer, RecordingObserver, SyncEvent, WarpEnd, WarpId};
use numsim_core::program::{Module, ProgramError};
use numsim_core::racecheck::{self, RaceObserver, RacecheckConfig};
use numsim_core::report::Report;
use numsim_core::sched::{self, ArgValue, Backend, Inputs, RunConfig, RunError, RunOutcome, RunStatus};
use numsim_core::synccheck::{self, SynccheckConfig};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::panic::{self, AssertUnwindSafe};

/// Decode a serialized module: JSON when the first non-blank byte is `{`,
/// postcard otherwise. Validates every kernel.
pub fn decode_module(bytes: &[u8]) -> Result<Module, ProgramError> {
    let first = bytes.iter().copied().find(|b| !b.is_ascii_whitespace());
    let module = if first == Some(b'{') {
        let text = std::str::from_utf8(bytes).map_err(|e| ProgramError::Decode(e.to_string()))?;
        Module::from_json(text)?
    } else {
        Module::from_bytes(bytes)?
    };
    for program in &module.kernels {
        program.validate()?;
    }
    Ok(module)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Numsim,
    Racecheck,
    Synccheck,
}

impl Mode {
    pub fn parse(s: &str) -> Option<Mode> {
        match s {
            "numsim" => Some(Mode::Numsim),
            "racecheck" => Some(Mode::Racecheck),
            "synccheck" => Some(Mode::Synccheck),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    Interp,
    Codegen,
}

impl BackendKind {
    pub fn parse(s: &str) -> Option<BackendKind> {
        match s {
            "interp" => Some(BackendKind::Interp),
            "codegen" => Some(BackendKind::Codegen),
            _ => None,
        }
    }
}

/// Everything `run` needs besides the module and inputs.
#[derive(Clone, Debug)]
pub struct RunRequest {
    pub mode: Mode,
    pub backend: BackendKind,
    pub config: RunConfig,
    /// Synccheck state / transition budgets per projection.
    pub state_budget: Option<u64>,
    pub transition_budget: Option<u64>,
    /// Synccheck `EchoLimits` (`max_schedules`, `max_events_per_run`,
    /// `max_total_events`, `max_wall_time_ms`, `max_diagnostic_bytes`).
    pub synccheck_limits: BTreeMap<String, u64>,
    /// Racecheck: stop recording after this many findings (0 = unlimited).
    pub max_findings: usize,
    /// Scheduler worker threads (also copied into `RunConfig::workers`).
    pub workers: u32,
    /// Codegen optimization level (0..=3).
    pub opt_level: u32,
    /// Codegen build cache (default: `$TMPDIR/numsim-codegen`).
    pub codegen_cache_dir: Option<std::path::PathBuf>,
}

impl RunRequest {
    pub fn new(mode: Mode) -> RunRequest {
        RunRequest {
            mode,
            backend: BackendKind::Interp,
            config: RunConfig::default(),
            state_budget: None,
            transition_budget: None,
            synccheck_limits: BTreeMap::new(),
            max_findings: 0,
            workers: 1,
            opt_level: 1,
            codegen_cache_dir: None,
        }
    }
}

/// One output buffer: bytes plus a per-byte validity mask (1 = valid) when
/// any byte is invalid.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputBuffer {
    pub bytes: Vec<u8>,
    pub invalid_mask: Option<Vec<u8>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExecuteResult {
    pub status: Value,
    pub outputs: BTreeMap<String, OutputBuffer>,
    pub stats: Value,
    /// Runtime diagnostics in the v2 payload shape (`kind`, `status`, `site`, ...).
    pub diagnostics: Vec<Value>,
    /// Checker modes: one `report::Report` per launch (JSON), in launch order.
    pub reports: Vec<String>,
    /// Checker modes: the checker's own legacy-shaped payload per launch
    /// (`racecheck::serialize` / `synccheck::serialize`).
    pub payloads: Vec<Value>,
    /// Wall-clock milliseconds: `build` (backend: codegen print/build/load,
    /// 0 for interp), `run` (`sched::run_with_config`, including arena
    /// binding), `check` (checker finish / offline exploration + serialize).
    pub timing: Value,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecuteError {
    /// Rejected before execution (bad program, missing or ill-typed argument).
    Run(String),
    /// A `numsim-core` body is still `unimplemented!()` / `todo!()`.
    NotImplemented(String),
    /// Any other engine panic (an engine bug).
    Panic(String),
}

impl std::fmt::Display for ExecuteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecuteError::Run(m) => write!(f, "{m}"),
            ExecuteError::NotImplemented(m) => write!(f, "numsim-core: {m}"),
            ExecuteError::Panic(m) => write!(f, "numsim-core panicked: {m}"),
        }
    }
}

fn run_error(e: RunError) -> ExecuteError {
    ExecuteError::Run(match e {
        RunError::InvalidProgram(m) => format!("invalid program: {m}"),
        RunError::MissingArg(n) => format!("missing argument {n:?}"),
        RunError::BadArg { name, message } => format!("bad argument {name:?}: {message}"),
        RunError::Backend(m) => format!("backend error: {m}"),
    })
}

/// Run `f`, converting panics into [`ExecuteError`].
pub fn guarded<T>(f: impl FnOnce() -> Result<T, ExecuteError>) -> Result<T, ExecuteError> {
    match panic::catch_unwind(AssertUnwindSafe(f)) {
        Ok(result) => result,
        Err(payload) => {
            let message = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "unknown panic".to_string());
            if message.contains("not implemented") || message.contains("not yet implemented") {
                Err(ExecuteError::NotImplemented(message))
            } else {
                Err(ExecuteError::Panic(message))
            }
        }
    }
}

fn backend_for(module: &Module, request: &RunRequest) -> Result<Backend, ExecuteError> {
    match request.backend {
        BackendKind::Interp => Ok(Backend::Interp),
        BackendKind::Codegen => {
            let cache = request
                .codegen_cache_dir
                .clone()
                .unwrap_or_else(|| std::env::temp_dir().join("numsim-codegen"));
            let mut opts = BuildOptions::new(cache.clone());
            opts.opt_level = match request.opt_level {
                0 => OptLevel::O0,
                1 => OptLevel::O1,
                2 => OptLevel::O2,
                _ => OptLevel::O3,
            };
            codegen::backend_for(module, &opts, &cache).map_err(|e| ExecuteError::Run(format!("codegen backend: {e:?}")))
        }
    }
}

fn exec_error_kind(e: &ExecError) -> (String, &'static str) {
    use numsim_core::interp::ExecErrorKind as K;
    use numsim_core::oplib::OpErrorKind;
    let status = match &e.kind {
        K::Budget | K::Unsupported | K::Op(OpErrorKind::Unsupported) => "incomplete",
        _ => "error",
    };
    let kind = match &e.kind {
        K::OutOfBounds => "out_of_bounds".to_string(),
        K::Uninit => "uninitialized_read".to_string(),
        K::Misaligned => "misaligned".to_string(),
        K::BadAddress => "bad_address".to_string(),
        // Legacy kind strings are kept (W9 review): a named-barrier protocol
        // violation was `named_barrier_contract_mismatch`.
        K::Protocol(numsim_core::sync::SyncError::Named(_)) => "named_barrier_contract_mismatch".to_string(),
        K::Protocol(_) => "sync_protocol_error".to_string(),
        K::Op(OpErrorKind::Unsupported) => "unsupported".to_string(),
        K::Op(OpErrorKind::Invalid) => "invalid_operand".to_string(),
        K::Trap => "trap".to_string(),
        K::Unsupported => "unsupported".to_string(),
        K::Divergence => "divergence".to_string(),
        K::WarpCollectiveDivergence => "warp_collective_divergence".to_string(),
        K::Budget => "budget_exhausted".to_string(),
        K::Internal => "internal_error".to_string(),
    };
    (kind, status)
}

fn snake(name: &str) -> String {
    let mut out = String::new();
    for (i, c) in name.chars().enumerate() {
        if c.is_ascii_uppercase() {
            if i > 0 {
                out.push('_');
            }
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

/// Structured `{protocol, error, fields}` of a sync protocol error, from its
/// serde form (`{"RegPool": {"MissingWarpgroupSync": {"wg": 0}}}`).
fn protocol_details(error: &numsim_core::sync::SyncError) -> serde_json::Map<String, Value> {
    let mut out = serde_json::Map::new();
    let value = serde_json::to_value(error).unwrap_or(Value::Null);
    let (protocol, inner) = match &value {
        Value::Object(m) if m.len() == 1 => {
            let (k, v) = m.iter().next().unwrap();
            (snake(k), v.clone())
        }
        Value::String(s) => (snake(s), Value::Null),
        _ => ("unknown".to_string(), value.clone()),
    };
    let (name, fields) = match &inner {
        Value::Object(m) if m.len() == 1 => {
            let (k, v) = m.iter().next().unwrap();
            (snake(k), v.clone())
        }
        Value::String(s) => (snake(s), Value::Null),
        other => (String::new(), other.clone()),
    };
    out.insert("protocol".into(), json!(protocol));
    out.insert("error".into(), json!(name));
    if let Value::Object(f) = fields {
        for (k, v) in f {
            out.entry(k).or_insert(v);
        }
    } else if !fields.is_null() {
        out.insert("fields".into(), fields);
    }
    out
}

/// The address an arena error message names (`... address 0x... ...`).
fn message_address(message: &str) -> Option<u64> {
    let at = message.find("address 0x")? + "address 0x".len();
    let hex: String = message[at..].chars().take_while(|c| c.is_ascii_hexdigit()).collect();
    u64::from_str_radix(&hex, 16).ok()
}

fn exec_error_json(kernel_index: Option<usize>, e: &ExecError) -> Value {
    use numsim_core::arena::addr::{classify_generic, decode_shared, Generic};
    use numsim_core::interp::ExecErrorKind as K;
    let (kind, status) = exec_error_kind(e);
    let mut m = serde_json::Map::new();
    let mut message = e.message.clone();
    match &e.kind {
        K::Protocol(sync_error) => {
            let details = protocol_details(sync_error);
            let fields: Vec<String> = details
                .iter()
                .filter(|(k, _)| k.as_str() != "protocol" && k.as_str() != "error")
                .map(|(k, v)| format!("{k}={v}"))
                .collect();
            let protocol = details.get("protocol").and_then(Value::as_str).unwrap_or("sync").replace('_', " ");
            let error = details.get("error").and_then(Value::as_str).unwrap_or("").replace('_', " ");
            // Readable message (W11-pin-message 2); the Debug form stays in `detail`.
            let plain = match (
                details.get("protocol").and_then(Value::as_str),
                details.get("error").and_then(Value::as_str),
            ) {
                // Plain words where the variant name alone does not say what
                // happened (coordinator, sweep 6).
                (Some("tcgen"), Some("live_allocations_at_exit")) => Some(match details.get("cta") {
                    Some(cta) => format!("kernel exited with live TMEM allocations (CTA {cta})"),
                    None => "kernel exited with live TMEM allocations".to_string(),
                }),
                _ => None,
            };
            message = if let Some(plain) = plain {
                plain
            } else if fields.is_empty() {
                format!("{protocol} protocol error: {error}")
            } else {
                format!("{protocol} protocol error: {error} ({})", fields.join(", "))
            };
            // Handlers that add context (`pending_count: NotNoComplete`,
            // named-barrier lane masks) keep it, in the text and as `context`.
            if !e.message.is_empty() && e.message != format!("{sync_error:?}") {
                message = format!("{message}; {}", e.message);
                m.insert("context".into(), json!(e.message));
            }
            m.extend(details);
        }
        K::BadAddress => {
            if let Some(addr) = message_address(&e.message) {
                m.insert("address".into(), json!(addr));
                // Name the aperture an address that missed its space decodes
                // into (W11-pin-message 3).
                let aperture = match classify_generic(addr) {
                    Generic::Shared(sa) => {
                        let (rank, offset) = decode_shared(sa);
                        m.insert("shared_cta_rank".into(), json!(rank));
                        m.insert("shared_offset".into(), json!(offset));
                        Some(format!("generic shared window (CTA rank {rank}, offset {offset})"))
                    }
                    Generic::Local(offset) => Some(format!("generic local window (offset {offset})")),
                    Generic::Param(offset) => Some(format!("kernel parameter window (offset {offset})")),
                    Generic::Global(_) | Generic::Unmapped(_) => None,
                };
                if let Some(aperture) = aperture {
                    m.insert("aperture".into(), json!(aperture));
                    message = format!("{message}; the address decodes into the {aperture}");
                }
            }
        }
        _ => {}
    }
    let base = json!({
        "kind": kind,
        "status": status,
        "message": message,
        "detail": format!("{:?}", e.kind),
        "kernel_index": kernel_index.map(|k| k as u32).unwrap_or(e.kernel),
        "source": "run_status",
        "site": if e.site.is_none() { Value::Null } else { json!(e.site.0) },
        "warp": e.warp.0,
        "pc": e.pc.0,
        "lanes": format!("{:?}", e.lanes),
    });
    let Value::Object(mut out) = base else { unreachable!() };
    for (k, v) in m {
        out.entry(k).or_insert(v);
    }
    let mut out = Value::Object(out);
    merge_attrs(&mut out, &e.attrs);
    out
}

/// Merge engine-structured facts (`ExecError::attrs`, `RunStatus::Incomplete
/// attrs`: `faulting_lanes`, `operation`, `operands`, `warp`, `lanes`,
/// `budget`, ...) into a diagnostic without overwriting its own keys.
fn merge_attrs(v: &mut Value, attrs: &std::collections::BTreeMap<String, Value>) {
    if let Value::Object(m) = v {
        for (k, a) in attrs {
            m.entry(k.clone()).or_insert_with(|| a.clone());
        }
    }
}

fn status_json(status: &RunStatus) -> Value {
    match status {
        RunStatus::Completed => json!({"kind": "completed"}),
        RunStatus::Deadlock { blocked } => json!({
            "kind": "deadlock",
            "blocked": blocked
                .iter()
                .map(|(w, r)| json!({"warp": w.0, "resource": serde_json::to_value(r).unwrap_or(Value::Null)}))
                .collect::<Vec<_>>(),
        }),
        RunStatus::Incomplete { reason, site, attrs } => {
            let mut v = json!({
                "kind": "incomplete",
                "reason": reason,
                "site": site.filter(|s| !s.is_none()).map(|s| s.0),
            });
            merge_attrs(&mut v, attrs);
            v
        }
        RunStatus::Error(e) => json!({"kind": "error", "error": exec_error_json(None, e)}),
    }
}

/// A runtime `report::Finding` (e.g. `UninitRead` under `ZeroAndReport`)
/// in the flat legacy diagnostic shape.
fn runtime_finding_json(f: &numsim_core::report::Finding) -> Value {
    let kind = serde_json::to_value(&f.kind).unwrap_or(Value::Null);
    let status = serde_json::to_value(f.status).ok().and_then(|v| v.as_str().map(str::to_lowercase));
    let mut m = serde_json::Map::new();
    m.insert("kind".into(), kind);
    m.insert("status".into(), json!(status.unwrap_or_else(|| "error".into())));
    m.insert("message".into(), json!(f.message));
    for (key, value) in &f.attrs {
        m.entry(key.clone()).or_insert_with(|| value.clone());
    }
    if let Some(e) = f.evidence.iter().find(|e| e.bytes.is_some()).or_else(|| f.evidence.first()) {
        m.insert("kernel_index".into(), json!(e.kernel));
        if !e.site.is_none() {
            m.insert("site".into(), json!(e.site.0));
        }
        if let Some(space) = e.space {
            m.insert("space".into(), serde_json::to_value(space).unwrap_or(Value::Null));
        }
        if let Some(alloc) = e.alloc {
            m.insert("allocation".into(), json!(alloc.0));
        }
        if let Some(b) = e.bytes {
            m.insert("byte_offset".into(), json!(b.start));
            m.insert("byte_len".into(), json!(b.len));
        }
        if let Some(buffer) = &e.buffer {
            m.insert("buffer".into(), json!(buffer));
        }
    }
    Value::Object(m)
}

fn diagnostics_of(outcome: &RunOutcome) -> Vec<Value> {
    let mut out = Vec::new();
    let kernel = outcome.failed_kernel;
    match &outcome.status {
        RunStatus::Error(e) => out.push(exec_error_json(None, e)),
        RunStatus::Deadlock { blocked } => out.push(json!({
            "kind": "deadlock",
            "status": "error",
            "message": "no warp can make progress",
            "source": "run_status",
            "kernel_index": kernel,
            "blocked": blocked.len(),
        })),
        RunStatus::Incomplete { reason, site, attrs } => {
            let mut v = json!({
                "kind": "analysis_incomplete",
                "status": "incomplete",
                "reason": reason,
                "source": "run_status",
                "kernel_index": kernel,
                "site": site.filter(|s| !s.is_none()).map(|s| s.0),
            });
            merge_attrs(&mut v, attrs);
            out.push(v);
        }
        RunStatus::Completed => {}
    }
    for (resource, error) in &outcome.sync_leftovers {
        out.push(json!({
            "kind": "sync_leftover",
            "status": "error",
            "kernel_index": kernel,
            "resource": serde_json::to_value(resource).unwrap_or(Value::Null),
            "message": format!("{error:?}"),
        }));
    }
    out.extend(outcome.diagnostics.iter().map(runtime_finding_json));
    if let Some(subset) = &outcome.subset {
        out.push(json!({
            "kind": "analysis_incomplete",
            "status": "incomplete",
            "reason": "subset_execution",
            "resident_cluster_ids": subset,
        }));
    }
    out
}

fn invalid_mask(valid: &BitSet) -> Option<Vec<u8>> {
    valid.first_clear(0, valid.len())?;
    Some((0..valid.len()).map(|i| u8::from(valid.get(i))).collect())
}

fn outcome_result(outcome: RunOutcome, reports: Vec<String>, payloads: Vec<Value>) -> ExecuteResult {
    let diagnostics = diagnostics_of(&outcome);
    let outputs = outcome
        .outputs
        .buffers
        .into_iter()
        .map(|(name, (bytes, valid))| {
            let invalid_mask = invalid_mask(&valid);
            (name, OutputBuffer { bytes, invalid_mask })
        })
        .collect();
    ExecuteResult {
        status: status_json(&outcome.status),
        outputs,
        stats: json!({
            "instrs": outcome.stats.instrs,
            "rounds": outcome.stats.rounds,
            "completions": outcome.stats.completions,
        }),
        diagnostics,
        reports,
        payloads,
        timing: Value::Null,
    }
}

fn ms(start: std::time::Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1e3
}

/// Splits the recorded protocol log per launch (synccheck checks one launch
/// at a time; `SynccheckConfig::launch` names it).
#[derive(Default)]
struct PerLaunchRecorder {
    current: RecordingObserver,
    kernel: u32,
    launches: Vec<(u32, RecordingObserver)>,
}

impl Observer for PerLaunchRecorder {
    // Every callback is forwarded so the per-launch recorder keeps whatever
    // `RecordingObserver` records (launch shapes included).
    fn enabled(&self) -> bool {
        self.current.enabled()
    }
    fn wants_word_history(&self) -> bool {
        self.current.wants_word_history()
    }
    fn begin_launch(&mut self, info: &LaunchInfo<'_>) {
        self.current = RecordingObserver::new();
        self.kernel = info.kernel_index;
        self.current.begin_launch(info);
    }
    fn end_launch(&mut self, info: &LaunchInfo<'_>) {
        self.current.end_launch(info);
        let log = std::mem::take(&mut self.current);
        self.launches.push((self.kernel, log));
    }
    fn access(&mut self, a: &Access<'_>) {
        self.current.access(a);
    }
    fn sync(&mut self, e: &SyncEvent) {
        self.current.sync(e);
    }
    fn warp_done(&mut self, w: WarpId, end: WarpEnd) {
        self.current.warp_done(w, end);
    }
    fn inbox_drain(&mut self, c: CtaId, r: u64) {
        self.current.inbox_drain(c, r);
    }
}

/// Run `module` in one mode. Never panics: engine panics become errors.
pub fn execute(module: &Module, inputs: &Inputs, request: &RunRequest) -> Result<ExecuteResult, ExecuteError> {
    guarded(|| {
        let started = std::time::Instant::now();
        let backend = backend_for(module, request)?;
        let build_ms = ms(started);
        let config = &request.config;
        let run_started = std::time::Instant::now();
        let run_ms;
        let result: Result<(ExecuteResult, std::time::Instant), ExecuteError> = match request.mode {
            Mode::Numsim => {
                let mut observer = NoopObserver;
                let outcome = sched::run_with_config(module, inputs, &mut observer, &backend, config).map_err(run_error)?;
                run_ms = ms(run_started);
                Ok((outcome_result(outcome, Vec::new(), Vec::new()), std::time::Instant::now()))
            }
            Mode::Racecheck => {
                let mut observer = RaceObserver::new(RacecheckConfig { max_findings: request.max_findings });
                let outcome = sched::run_with_config(module, inputs, &mut observer, &backend, config).map_err(run_error)?;
                run_ms = ms(run_started);
                let check_started = std::time::Instant::now();
                observer.finish_launch(); // no-op when end_launch already finalized it
                let reports: Vec<Report> = racecheck::payload::reports(&observer);
                let payloads = reports.iter().map(racecheck::serialize).collect();
                Ok((outcome_result(outcome, reports.iter().map(Report::to_json).collect(), payloads), check_started))
            }
            Mode::Synccheck => {
                let mut recorder = PerLaunchRecorder::default();
                let outcome = sched::run_with_config(module, inputs, &mut recorder, &backend, config).map_err(run_error)?;
                run_ms = ms(run_started);
                let check_started = std::time::Instant::now();
                let mut reports = Vec::new();
                // One check per launch; the kernel index flows from
                // `SyncEvent.kernel` into `Report.launch` / `Evidence.kernel`.
                for (kernel, log) in &recorder.launches {
                    let mut sc = SynccheckConfig::default();
                    // Exclusive tcgen05 column limit of the target (W6-5).
                    let arch = module.kernels.get(*kernel as usize).and_then(|k| k.arch.as_deref());
                    sc.tcgen_exclusive_max = Some(sched::exclusive_tmem_columns(arch));
                    if let Some(budget) = request.state_budget {
                        sc.state_budget = budget;
                    }
                    if let Some(budget) = request.transition_budget {
                        sc.transition_budget = budget;
                    }
                    for (key, value) in &request.synccheck_limits {
                        let v = *value;
                        match key.as_str() {
                            "max_schedules" => sc.limits.max_schedules = v,
                            "max_events_per_run" => sc.limits.max_events_per_run = v,
                            "max_total_events" => sc.limits.max_total_events = v,
                            "max_wall_time_ms" => sc.limits.max_wall_time_ms = v,
                            "max_diagnostic_bytes" => sc.limits.max_diagnostic_bytes = v,
                            _ => {}
                        }
                    }
                    let mut report = synccheck::check(log, &sc);
                    // A launch with no synchronization events has no kernel
                    // index to infer `launch` from; the recorder knows it.
                    report.launch = *kernel;
                    reports.push(report);
                }
                let payloads = reports.iter().map(synccheck::serialize).collect();
                Ok((outcome_result(outcome, reports.iter().map(Report::to_json).collect(), payloads), check_started))
            }
        };
        let (mut result, check_started) = result?;
        result.timing = json!({"build": build_ms, "run": run_ms, "check": ms(check_started)});
        Ok(result)
    })
}

/// Convenience for building inputs from Rust tests.
pub fn buffer_arg(bytes: Vec<u8>) -> ArgValue {
    ArgValue::Buffer { bytes, valid: None }
}

/// Validity bits from a per-byte 0/1 mask.
pub fn bitset_from_mask(mask: &[u8]) -> BitSet {
    let mut set = BitSet::new(mask.len() as u64, true);
    for (i, b) in mask.iter().enumerate() {
        if *b == 0 {
            set.set_range(i as u64, 1, false);
        }
    }
    set
}

/// `"error" | "allow" | "zero_and_report"`; `None` = `ZeroAndReport`.
pub fn validity_policy(name: Option<&str>, mode: Mode) -> Option<ValidityPolicy> {
    match name {
        Some("error") => Some(ValidityPolicy::Error),
        Some("allow") => Some(ValidityPolicy::Allow),
        Some("zero_and_report") => Some(ValidityPolicy::ZeroAndReport),
        Some(_) => None,
        // Every mode reads uninitialized bytes as zero and reports them
        // (legacy NumSim/Racecheck/Synccheck all surface `uninitialized_read`).
        None => {
            let _ = mode;
            Some(ValidityPolicy::ZeroAndReport)
        }
    }
}

#[cfg(feature = "python")]
mod py {
    use super::*;
    use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyTypeError, PyValueError};
    use pyo3::prelude::*;
    use pyo3::types::{PyBytes, PyDict, PyString, PyTuple};
    use std::sync::Arc;

    fn to_py_err(e: ExecuteError) -> PyErr {
        match e {
            ExecuteError::Run(m) => PyValueError::new_err(m),
            e @ ExecuteError::NotImplemented(_) => PyNotImplementedError::new_err(e.to_string()),
            e @ ExecuteError::Panic(_) => PyRuntimeError::new_err(e.to_string()),
        }
    }

    /// A decoded, validated `numsim_core::Module`.
    #[pyclass(name = "Module", module = "numsim_core_py", frozen)]
    struct PyModuleHandle {
        inner: Arc<Module>,
    }

    #[pymethods]
    impl PyModuleHandle {
        #[getter]
        fn format_version(&self) -> u32 {
            self.inner.format_version
        }
        #[getter]
        fn kernel_names(&self) -> Vec<String> {
            self.inner.kernels.iter().map(|k| k.name.clone()).collect()
        }
        fn __len__(&self) -> usize {
            self.inner.kernels.len()
        }
        /// The module as serde JSON (str).
        fn to_json(&self) -> String {
            self.inner.to_json()
        }
        /// The module as postcard bytes.
        fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
            PyBytes::new(py, &self.inner.to_bytes())
        }
        fn __repr__(&self) -> String {
            format!("numsim_core_py.Module(kernels={:?})", self.kernel_names())
        }
    }

    /// Decode a serialized module (JSON str/bytes or postcard bytes).
    #[pyfunction]
    fn load_module(data: &Bound<'_, PyAny>) -> PyResult<PyModuleHandle> {
        let bytes: Vec<u8> = if let Ok(s) = data.cast::<PyString>() {
            s.to_str()?.as_bytes().to_vec()
        } else if let Ok(b) = data.cast::<PyBytes>() {
            b.as_bytes().to_vec()
        } else {
            data.call_method0("tobytes")?.cast::<PyBytes>()?.as_bytes().to_vec()
        };
        let module = decode_module(&bytes).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyModuleHandle { inner: Arc::new(module) })
    }

    fn object_bytes(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
        if let Ok(b) = obj.cast::<PyBytes>() {
            return Ok(b.as_bytes().to_vec());
        }
        if obj.hasattr("tobytes")? {
            return Ok(obj.call_method0("tobytes")?.cast::<PyBytes>()?.as_bytes().to_vec());
        }
        let builtins = obj.py().import("builtins")?;
        Ok(builtins.getattr("bytes")?.call1((obj,))?.cast::<PyBytes>()?.as_bytes().to_vec())
    }

    fn scalar_bits(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
        if let Ok(v) = obj.extract::<u64>() {
            return Ok(v);
        }
        if let Ok(v) = obj.extract::<i64>() {
            return Ok(v as u64);
        }
        Err(PyTypeError::new_err("scalar arguments must be integer bit patterns (canonicalize floats in Python)"))
    }

    /// Accepted values: `("buffer", data, valid_mask_or_None)`, `("scalar", int)`,
    /// `("tensor_map", data)`, `("tensor_map_of", base_arg, byte_offset, image)`,
    /// `("view", target_buffer_arg, byte_offset, byte_len)`,
    /// `("pointer", target, offset)`, a bytes-like /
    /// numpy array (buffer, all valid), or an int (scalar bits).
    fn arg_value(name: &str, obj: &Bound<'_, PyAny>) -> PyResult<ArgValue> {
        if let Ok(t) = obj.cast::<PyTuple>() {
            let tag: String = t.get_item(0)?.extract()?;
            return match (tag.as_str(), t.len()) {
                ("buffer", 2 | 3) => {
                    let bytes = object_bytes(&t.get_item(1)?)?;
                    let valid = if t.len() == 3 && !t.get_item(2)?.is_none() {
                        let mask = object_bytes(&t.get_item(2)?)?;
                        if mask.len() != bytes.len() {
                            return Err(PyValueError::new_err(format!("{name}: validity mask length differs from data")));
                        }
                        Some(bitset_from_mask(&mask))
                    } else {
                        None
                    };
                    Ok(ArgValue::Buffer { bytes, valid })
                }
                ("scalar", 2) => Ok(ArgValue::Scalar(scalar_bits(&t.get_item(1)?)?)),
                ("tensor_map", 2) => Ok(ArgValue::TensorMap(object_bytes(&t.get_item(1)?)?)),
                ("tensor_map_of", 4) => {
                    // A host descriptor image over another argument: decode
                    // it here; the scheduler re-encodes it against that
                    // argument's engine address (W8-3).
                    let base: String = t.get_item(1)?.extract()?;
                    let offset: u64 = t.get_item(2)?.extract()?;
                    let image = object_bytes(&t.get_item(3)?)?;
                    let image: [u8; 128] = image
                        .try_into()
                        .map_err(|_| PyValueError::new_err(format!("{name}: tensor map image must be 128 bytes")))?;
                    let desc = numsim_core::oplib::TensorMapDesc::decode(&image)
                        .map_err(|e| PyValueError::new_err(format!("{name}: undecodable tensor map: {e:?}")))?;
                    Ok(ArgValue::TensorMapOf { base, offset, desc })
                }
                ("pointer", 3) => Ok(ArgValue::Pointer { target: t.get_item(1)?.extract()?, offset: t.get_item(2)?.extract()? }),
                ("view", 4) => Ok(ArgValue::View {
                    target: t.get_item(1)?.extract()?,
                    offset: t.get_item(2)?.extract()?,
                    len: t.get_item(3)?.extract()?,
                }),
                _ => Err(PyValueError::new_err(format!("{name}: unknown argument form {tag:?}/{}", t.len()))),
            };
        }
        if obj.is_instance_of::<pyo3::types::PyInt>() {
            return Ok(ArgValue::Scalar(scalar_bits(obj)?));
        }
        Ok(ArgValue::Buffer { bytes: object_bytes(obj)?, valid: None })
    }

    fn json_to_py<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyAny>> {
        let json = py.import("json")?;
        json.call_method1("loads", (value.to_string(),))
    }

    /// Run a module. Returns a dict with `status`, `outputs` (name -> bytes),
    /// `invalid` (name -> per-byte validity bytes, only for buffers with
    /// invalid bytes), `stats`, `diagnostics`, and for checker modes
    /// `reports` (one `report::Report` JSON str per launch) and `payloads`
    /// (the checker's legacy-shaped payload dict per launch).
    #[pyfunction]
    #[pyo3(signature = (module, inputs, *, mode="numsim", backend="interp", workers=1, seed=0,
                        loop_budget=None, quantum=None, max_rounds=None, opt_level=1,
                        validity=None, state_budget=None, transition_budget=None, max_findings=0,
                        codegen_cache_dir=None, subset=None, synccheck_limits=None, host_addrs=None))]
    fn run<'py>(
        py: Python<'py>,
        module: &PyModuleHandle,
        inputs: &Bound<'py, PyDict>,
        mode: &str,
        backend: &str,
        workers: u32,
        seed: u64,
        loop_budget: Option<u64>,
        quantum: Option<u32>,
        max_rounds: Option<u64>,
        opt_level: u32,
        validity: Option<&str>,
        state_budget: Option<u64>,
        transition_budget: Option<u64>,
        max_findings: usize,
        codegen_cache_dir: Option<std::path::PathBuf>,
        subset: Option<Vec<u32>>,
        synccheck_limits: Option<BTreeMap<String, u64>>,
        host_addrs: Option<BTreeMap<String, u64>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let mode = Mode::parse(mode).ok_or_else(|| PyValueError::new_err(format!("unknown mode {mode:?}")))?;
        let backend = BackendKind::parse(backend).ok_or_else(|| PyValueError::new_err(format!("unknown backend {backend:?}")))?;
        let mut args = BTreeMap::new();
        for (key, value) in inputs.iter() {
            let name: String = key.extract()?;
            let arg = arg_value(&name, &value)?;
            args.insert(name, arg);
        }
        // Host pointers of buffer arguments: synthetic global addresses keep
        // their low 8 bits (`arena::addr` ruling).
        let inputs = Inputs { args, host_addrs: host_addrs.unwrap_or_default() };
        let mut request = RunRequest::new(mode);
        request.backend = backend;
        request.workers = workers;
        request.config.workers = workers.max(1) as usize;
        request.opt_level = opt_level;
        request.state_budget = state_budget;
        request.transition_budget = transition_budget;
        request.max_findings = max_findings;
        request.codegen_cache_dir = codegen_cache_dir;
        request.config.seed = seed;
        request.config.subset = subset;
        if let Some(limits) = synccheck_limits {
            for key in limits.keys() {
                if !["max_schedules", "max_events_per_run", "max_total_events", "max_wall_time_ms", "max_diagnostic_bytes"].contains(&key.as_str()) {
                    return Err(PyValueError::new_err(format!("unknown synccheck limit {key:?}")));
                }
            }
            request.synccheck_limits = limits;
        }
        request.config.validity = validity_policy(validity, mode)
            .ok_or_else(|| PyValueError::new_err(format!("unknown validity policy {validity:?}")))?;
        if let Some(v) = loop_budget {
            request.config.loop_budget = v;
        }
        if let Some(v) = quantum {
            request.config.quantum = v;
        }
        if let Some(v) = max_rounds {
            request.config.max_rounds = v;
        }
        let module = Arc::clone(&module.inner);
        let result = py.detach(move || execute(&module, &inputs, &request)).map_err(to_py_err)?;

        let out = PyDict::new(py);
        out.set_item("status", json_to_py(py, &result.status)?)?;
        let outputs = PyDict::new(py);
        let invalid = PyDict::new(py);
        for (name, buffer) in &result.outputs {
            outputs.set_item(name, PyBytes::new(py, &buffer.bytes))?;
            if let Some(mask) = &buffer.invalid_mask {
                invalid.set_item(name, PyBytes::new(py, mask))?;
            }
        }
        out.set_item("outputs", outputs)?;
        out.set_item("invalid", invalid)?;
        out.set_item("stats", json_to_py(py, &result.stats)?)?;
        out.set_item("diagnostics", json_to_py(py, &Value::Array(result.diagnostics.clone()))?)?;
        out.set_item("reports", result.reports.clone())?;
        out.set_item("payloads", json_to_py(py, &Value::Array(result.payloads.clone()))?)?;
        out.set_item("timing", json_to_py(py, &result.timing)?)?;
        Ok(out)
    }

    /// Engine global addresses the given inputs would be bound at
    /// (`sched::plan_global_addresses`): `{argument name: address}` for
    /// buffer and view arguments.
    #[pyfunction]
    #[pyo3(signature = (module, inputs, *, host_addrs=None))]
    fn plan_global_addresses(
        module: &PyModuleHandle,
        inputs: &Bound<'_, PyDict>,
        host_addrs: Option<BTreeMap<String, u64>>,
    ) -> PyResult<BTreeMap<String, u64>> {
        let mut args = BTreeMap::new();
        for (key, value) in inputs.iter() {
            let name: String = key.extract()?;
            let arg = arg_value(&name, &value)?;
            args.insert(name, arg);
        }
        let inputs = Inputs { args, host_addrs: host_addrs.unwrap_or_default() };
        let module = Arc::clone(&module.inner);
        guarded(|| sched::plan_global_addresses(&module, &inputs).map_err(run_error)).map_err(to_py_err)
    }

    /// Format version of serialized programs this extension accepts.
    #[pyfunction]
    fn program_format_version() -> u32 {
        numsim_core::program::FORMAT_VERSION
    }

    #[pymodule]
    fn numsim_core_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyModuleHandle>()?;
        m.add_function(wrap_pyfunction!(program_format_version, m)?)?;
        m.add_function(wrap_pyfunction!(load_module, m)?)?;
        m.add_function(wrap_pyfunction!(run, m)?)?;
        m.add_function(wrap_pyfunction!(plan_global_addresses, m)?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numsim_core::dtype::Dtype;
    use numsim_core::program::{CmpOp, SpecialReg};
    use numsim_core::testutil::ProgramBuilder;
    use numsim_core::Ty;

    /// c[i] = a[i] + b[i] over 8 CTAs x 128 threads (the conformance
    /// fixture for the Python layer; see `fixture_is_current`).
    pub fn vector_add() -> Module {
        let mut b = ProgramBuilder::new("vadd", 128);
        b.grid(8, 1, 1);
        b.site("tirx.BufferStore", 10);
        let a = b.global("a", Dtype::F32);
        let bb = b.global("b", Dtype::F32);
        let c = b.global("c", Dtype::F32);
        let bx = b.reg(Ty::U32);
        let tx = b.reg(Ty::U32);
        let i = b.reg(Ty::U32);
        let p = b.reg(Ty::PRED);
        let x = b.reg(Ty::F32);
        let y = b.reg(Ty::F32);
        b.read_special(bx, SpecialReg::CtaLinear);
        b.read_special(tx, SpecialReg::ThreadInCta);
        let k128 = b.k_u32(128);
        b.mul(Ty::U32, i, bx, k128);
        b.add_u32(i, i, tx);
        let n = b.k_u32(1024);
        b.compare(CmpOp::Lt, Ty::U32, p, i, n);
        b.if_(p);
        b.ld_f32(x, a, i);
        b.ld_f32(y, bb, i);
        b.add_f32(x, x, y);
        b.st_f32(c, i, x);
        b.end_if();
        b.exit();
        b.build_module()
    }

    fn fixture_path() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../../../tests/numsim/v2/fixtures/vector_add.module.json")
    }

    #[test]
    fn json_and_postcard_round_trip() {
        let m = vector_add();
        assert_eq!(decode_module(m.to_json().as_bytes()).unwrap(), m);
        assert_eq!(decode_module(&m.to_bytes()).unwrap(), m);
        assert!(matches!(decode_module(b"{\"format_version\": 999, \"kernels\": []}"), Err(ProgramError::Version { .. })));
        assert!(decode_module(b"\x00\x01garbage").is_err());
    }

    /// The Python layer's hand-built fixture is generated from
    /// `vector_add()`; `UPDATE_FIXTURES=1 cargo test -p numsim-py` rewrites it.
    #[test]
    fn fixture_is_current() {
        let path = fixture_path();
        let json = vector_add().to_json() + "\n";
        if std::env::var_os("UPDATE_FIXTURES").is_some() {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, &json).unwrap();
        }
        let current = std::fs::read_to_string(&path).expect("fixture missing; run with UPDATE_FIXTURES=1");
        assert_eq!(current, json, "fixture is stale; run with UPDATE_FIXTURES=1");
    }

    #[test]
    fn execute_reports_unimplemented_bodies_instead_of_panicking() {
        let m = vector_add();
        let mut inputs = Inputs::default();
        for name in ["a", "b", "c"] {
            inputs.args.insert(name.into(), buffer_arg(vec![0u8; 4096]));
        }
        for mode in [Mode::Numsim, Mode::Racecheck, Mode::Synccheck] {
            match execute(&m, &inputs, &RunRequest::new(mode)) {
                Ok(result) => assert!(result.status.get("kind").is_some()),
                Err(ExecuteError::NotImplemented(_)) => {}
                Err(other) => panic!("{mode:?}: {other}"),
            }
        }
    }

    #[test]
    fn protocol_errors_get_structured_details() {
        assert_eq!(snake("MissingWarpgroupSync"), "missing_warpgroup_sync");
        assert_eq!(message_address("Global address 0xfffe00000004 is not mapped"), Some(0xfffe00000004));
        assert_eq!(message_address("no address here"), None);
        let live = numsim_core::sync::SyncError::Tcgen(numsim_core::sync::tcgen::Error::LiveAllocationsAtExit { cta: 0 });
        let details = protocol_details(&live);
        assert_eq!(details.get("protocol"), Some(&json!("tcgen")));
        assert_eq!(details.get("error"), Some(&json!("live_allocations_at_exit")));
        assert_eq!(details.get("cta"), Some(&json!(0)));
    }

    #[test]
    fn validity_mask_round_trip() {
        let set = bitset_from_mask(&[1, 0, 1, 1]);
        assert_eq!(invalid_mask(&set), Some(vec![1, 0, 1, 1]));
        assert_eq!(invalid_mask(&BitSet::new(5, true)), None);
    }
}
