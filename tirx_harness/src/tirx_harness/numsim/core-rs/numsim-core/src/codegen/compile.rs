//! Build, cache and load the generated cdylib.
//!
//! Cache layout under `BuildOptions::cache_dir`:
//! * `core-target-<toolchain>/`: cargo target dir of
//!   `cargo build --release -p numsim-core` from this crate's sources (cargo
//!   decides staleness; an up-to-date check costs ~0.1 s).
//! * `gen-<key>/libnumsim_gen.so`: one `rustc` of the printed `lib.rs`
//!   with `--extern numsim_core=<rlib>`; key = sha256(module bytes, source,
//!   sha256(rlib), `rustc -vV`, opt level, assertion flags, emit version).
//!
//! Entries are written to a temp name and renamed into place, so concurrent
//! builders at worst duplicate work.

use super::emit::{emit_module, EMIT_VERSION};
use super::{abi_fingerprint, ABI_WORDS, CHECK_SYMBOL_PREFIX, STEP_SYMBOL_PREFIX};
use crate::interp::WarpStepFn;
use crate::program::{Module, Program};
use crate::sched::Backend;
use sha2::{Digest, Sha256};
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum OptLevel {
    O0,
    /// Default (checkers: fast compile).
    #[default]
    O1,
    O2,
    /// Perf runs.
    O3,
}

impl OptLevel {
    fn flag(self) -> &'static str {
        match self {
            OptLevel::O0 => "0",
            OptLevel::O1 => "1",
            OptLevel::O2 => "2",
            OptLevel::O3 => "3",
        }
    }
}

#[derive(Clone, Debug)]
pub struct BuildOptions {
    pub cache_dir: PathBuf,
    pub opt_level: OptLevel,
    /// `numsim-core` package directory (sources the rlib is built from;
    /// must be the sources the host was built from).
    pub core_dir: PathBuf,
    /// `rustc` / `cargo` executables (default `$RUSTC`/`$CARGO` or PATH).
    pub rustc: Option<PathBuf>,
    pub cargo: Option<PathBuf>,
    /// Print build timings to stderr (also `NUMSIM_CODEGEN_LOG=1`).
    pub verbose: bool,
    /// Recompile even when the cache has the library (timing runs).
    pub force: bool,
}

impl BuildOptions {
    pub fn new(cache_dir: impl Into<PathBuf>) -> BuildOptions {
        BuildOptions {
            cache_dir: cache_dir.into(),
            opt_level: OptLevel::O1,
            core_dir: PathBuf::from(env!("CARGO_MANIFEST_DIR")),
            rustc: None,
            cargo: None,
            verbose: false,
            force: false,
        }
    }
    pub fn opt(mut self, o: OptLevel) -> BuildOptions {
        self.opt_level = o;
        self
    }
    fn rustc(&self) -> PathBuf {
        self.rustc.clone().or_else(|| std::env::var_os("RUSTC").map(PathBuf::from)).unwrap_or_else(|| "rustc".into())
    }
    fn cargo(&self) -> PathBuf {
        self.cargo.clone().or_else(|| std::env::var_os("CARGO").map(PathBuf::from)).unwrap_or_else(|| "cargo".into())
    }
    fn log(&self) -> bool {
        self.verbose || std::env::var_os("NUMSIM_CODEGEN_LOG").is_some_and(|v| v != "0")
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BuildError(pub String);

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "codegen build: {}", self.0)
    }
}
impl std::error::Error for BuildError {}

fn err(m: impl Into<String>) -> BuildError {
    BuildError(m.into())
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BuildStats {
    pub emit: Duration,
    /// `numsim-core` rlib build (None = cached).
    pub core_build: Option<Duration>,
    /// Generated crate compile (None = cached).
    pub rustc: Option<Duration>,
    pub load: Duration,
    pub source_bytes: usize,
    pub key: String,
}

/// A loaded generated library (never unloaded).
#[derive(Clone, Debug)]
pub struct LoadedBackend {
    /// One step function per kernel, in `Module::kernels` order.
    pub steps: Vec<WarpStepFn>,
    pub library: PathBuf,
    pub stats: BuildStats,
    module_sha256: [u8; 32],
    checks: Vec<fn(&Program) -> bool>,
}

impl LoadedBackend {
    pub fn backend(&self) -> Backend {
        Backend::Codegen(self.steps.clone())
    }
    /// Verify the library was generated for `module` (same bytes, and each
    /// kernel's code length / constant pool / op table match).
    pub fn check_module(&self, module: &Module) -> Result<(), BuildError> {
        if sha256(&module.to_bytes()) != self.module_sha256 {
            return Err(err(format!("{} was built for a different module", self.library.display())));
        }
        if module.kernels.len() != self.checks.len() {
            return Err(err("kernel count mismatch"));
        }
        for (k, (p, check)) in module.kernels.iter().zip(&self.checks).enumerate() {
            if !check(p) {
                return Err(err(format!("kernel {k} ({}) does not match the generated tables", p.name)));
            }
        }
        Ok(())
    }
}

pub(crate) fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// `rustc -vV` of the toolchain `opts` builds with.
pub fn rustc_version(opts: &BuildOptions) -> Result<String, BuildError> {
    let out = Command::new(opts.rustc()).arg("-vV").output().map_err(|e| err(format!("running rustc -vV: {e}")))?;
    if !out.status.success() {
        return Err(err(format!("rustc -vV failed: {}", String::from_utf8_lossy(&out.stderr))));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Build a single-kernel backend.
pub fn build(program: &Program, opts: &BuildOptions) -> Result<LoadedBackend, BuildError> {
    build_module(&Module::new(vec![program.clone()]), opts)
}

/// Emit, compile (or reuse the cache), load and verify.
pub fn build_module(module: &Module, opts: &BuildOptions) -> Result<LoadedBackend, BuildError> {
    let t0 = Instant::now();
    let module_bytes = module.to_bytes();
    let module_sha = sha256(&module_bytes);
    let source = emit_module(module, &module_sha);
    let mut stats = BuildStats { emit: t0.elapsed(), source_bytes: source.len(), ..BuildStats::default() };

    let rustc_v = rustc_version(opts)?;
    let checks_on = cfg!(debug_assertions);
    let core = build_core(opts, &rustc_v, checks_on, &mut stats)?;

    let mut h = Sha256::new();
    for part in [
        format!("numsim-codegen emit={EMIT_VERSION} opt={} checks={checks_on}\n", opts.opt_level.flag()).as_bytes(),
        rustc_v.as_bytes(),
        core.key.as_bytes(),
        &module_bytes,
        source.as_bytes(),
    ] {
        h.update((part.len() as u64).to_le_bytes());
        h.update(part);
    }
    let key = hex(&h.finalize())[..24].to_string();
    stats.key = key.clone();
    let dir = opts.cache_dir.join(format!("gen-{key}"));
    let lib = dir.join(format!("libnumsim_gen.{}", std::env::consts::DLL_EXTENSION));
    if opts.force || !lib.exists() {
        let t = Instant::now();
        let tmp = tmp_dir(&opts.cache_dir, "gen")?;
        let src_path = tmp.join("lib.rs");
        std::fs::write(&src_path, &source).map_err(|e| err(format!("write {}: {e}", src_path.display())))?;
        let out_lib = tmp.join(lib.file_name().unwrap());
        let flag = if checks_on { "on" } else { "off" };
        let output = Command::new(opts.rustc())
            .args(["--edition=2021", "--crate-type=cdylib", "--crate-name=numsim_gen", "--emit=link"])
            .args(["-C", &format!("opt-level={}", opts.opt_level.flag())])
            .args(["-C", &format!("debug-assertions={flag}"), "-C", &format!("overflow-checks={flag}")])
            .args(["-C", "panic=unwind", "-C", "debuginfo=0", "-C", "codegen-units=16"])
            .arg("-L")
            .arg(format!("dependency={}", core.deps.display()))
            .arg("--extern")
            .arg(format!("numsim_core={}", core.rlib.display()))
            .arg("-o")
            .arg(&out_lib)
            .arg(&src_path)
            .output()
            .map_err(|e| err(format!("running rustc: {e}")))?;
        if !output.status.success() {
            let msg = String::from_utf8_lossy(&output.stderr);
            let tail = &msg[msg.len().saturating_sub(12000)..];
            return Err(err(format!("rustc failed for {} (kept for inspection):\n{tail}", src_path.display())));
        }
        if opts.force && dir.exists() {
            // A loaded library may be mapped from `dir`: move it aside, never overwrite in place.
            let _ = std::fs::rename(&dir, tmp_dir(&opts.cache_dir, "old")?.join("gen"));
        }
        publish(&tmp, &dir)?;
        let el = t.elapsed();
        stats.rustc = Some(el);
        {
            eprintln!(
                "[numsim codegen] compiled {} kernel(s), {} KiB source at O{} in {:.2}s -> {}",
                module.kernels.len(),
                source.len() / 1024,
                opts.opt_level.flag(),
                el.as_secs_f64(),
                dir.display()
            );
        }
    }
    let t = Instant::now();
    let loaded = load_library(&lib, module, module_sha, stats.clone())?;
    let mut loaded = loaded;
    loaded.stats.load = t.elapsed();
    loaded.check_module(module)?;
    if opts.log() {
        let s = &loaded.stats;
        eprintln!(
            "[numsim codegen] key {} emit {:.1}ms core {:?} rustc {:?} load {:.1}ms",
            s.key,
            s.emit.as_secs_f64() * 1e3,
            s.core_build,
            s.rustc,
            s.load.as_secs_f64() * 1e3
        );
    }
    Ok(loaded)
}

struct CoreRlib {
    key: String,
    rlib: PathBuf,
    deps: PathBuf,
}

/// Build (or reuse) the `numsim-core` rlib the generated crate links: a
/// `cargo build --release -p numsim-core` into a persistent private target
/// dir (cargo tracks staleness of this crate and its path dependencies; an
/// up-to-date build is a no-op). The rlib's hash keys the generated cache.
fn build_core(opts: &BuildOptions, rustc_v: &str, checks_on: bool, stats: &mut BuildStats) -> Result<CoreRlib, BuildError> {
    let core_dir = &opts.core_dir;
    let ws = core_dir.parent().ok_or_else(|| err("core_dir has no parent workspace"))?;
    let tkey = hex(&sha256(format!("{rustc_v}\nchecks={checks_on}\n{}", ws.display()).as_bytes()))[..16].to_string();
    let target = opts.cache_dir.join(format!("core-target-{tkey}"));
    let flag = if checks_on { "true" } else { "false" };
    let t = Instant::now();
    let output = Command::new(opts.cargo())
        .args(["build", "--release", "--locked", "-p", "numsim-core", "--message-format=json-render-diagnostics"])
        .arg("--manifest-path")
        .arg(ws.join("Cargo.toml"))
        .arg("--target-dir")
        .arg(&target)
        .env_remove("CARGO_TARGET_DIR")
        .env_remove("CARGO_BUILD_TARGET_DIR")
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .env("CARGO_PROFILE_RELEASE_DEBUG", "0")
        .env("CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS", flag)
        .env("CARGO_PROFILE_RELEASE_OVERFLOW_CHECKS", flag)
        .env("CARGO_PROFILE_RELEASE_PANIC", "unwind")
        .env("CARGO_PROFILE_RELEASE_LTO", "false")
        .env("CARGO_PROFILE_RELEASE_INCREMENTAL", "false")
        .output()
        .map_err(|e| err(format!("running cargo: {e}")))?;
    if !output.status.success() {
        let msg = String::from_utf8_lossy(&output.stderr);
        return Err(err(format!("numsim-core rlib build failed:\n{}", &msg[msg.len().saturating_sub(12000)..])));
    }
    let mut rlib = None;
    let mut fresh = true;
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else { continue };
        if v["reason"] != "compiler-artifact" {
            continue;
        }
        fresh &= v["fresh"].as_bool().unwrap_or(false);
        if v["target"]["name"] == "numsim_core" || v["target"]["name"] == "numsim-core" {
            if let Some(files) = v["filenames"].as_array() {
                rlib = files.iter().filter_map(|f| f.as_str()).find(|f| f.ends_with(".rlib")).map(PathBuf::from);
            }
        }
    }
    let rlib = rlib.ok_or_else(|| err("cargo did not report a numsim_core rlib"))?;
    // Cargo reports the uplifted `release/libnumsim_core.rlib`; its
    // dependencies live in `release/deps`.
    let parent = rlib.parent().unwrap();
    let deps = if parent.ends_with("deps") { parent.to_path_buf() } else { parent.join("deps") };
    let bytes = std::fs::read(&rlib).map_err(|e| err(format!("read {}: {e}", rlib.display())))?;
    let key = hex(&sha256(&bytes))[..24].to_string();
    if !fresh {
        let el = t.elapsed();
        stats.core_build = Some(el);
        eprintln!("[numsim codegen] built numsim-core rlib in {:.2}s ({})", el.as_secs_f64(), rlib.display());
    }
    Ok(CoreRlib { key, rlib, deps })
}

fn tmp_dir(cache: &Path, what: &str) -> Result<PathBuf, BuildError> {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let p = cache.join(format!(".tmp-{what}-{}-{nanos}-{n}", std::process::id()));
    std::fs::create_dir_all(&p).map_err(|e| err(format!("mkdir {}: {e}", p.display())))?;
    Ok(p)
}

/// Rename `tmp` to `dst`; if another builder won the race keep theirs.
fn publish(tmp: &Path, dst: &Path) -> Result<(), BuildError> {
    match std::fs::rename(tmp, dst) {
        Ok(()) => Ok(()),
        Err(_) if dst.exists() => {
            let _ = std::fs::remove_dir_all(tmp);
            Ok(())
        }
        Err(e) => Err(err(format!("rename {} -> {}: {e}", tmp.display(), dst.display()))),
    }
}

fn load_library(lib_path: &Path, module: &Module, module_sha: [u8; 32], stats: BuildStats) -> Result<LoadedBackend, BuildError> {
    // SAFETY: the library was produced by `build_module` from this crate's
    // sources; its initializers are std's only. The ABI fingerprint is
    // checked before any boundary type is exchanged.
    let lib = unsafe { libloading::Library::new(lib_path) }.map_err(|e| err(format!("dlopen {}: {e}", lib_path.display())))?;
    unsafe {
        let fp: libloading::Symbol<fn() -> [u64; ABI_WORDS]> =
            lib.get(b"numsim_abi_fingerprint\0").map_err(|e| err(format!("symbol numsim_abi_fingerprint: {e}")))?;
        let theirs = fp();
        let ours = abi_fingerprint();
        if theirs != ours {
            let first = ours.iter().zip(theirs.iter()).position(|(a, b)| a != b).unwrap_or(0);
            return Err(err(format!(
                "ABI mismatch with {} at word {first} (host {:?} vs library {:?}): host and library were built from \
                 different numsim-core sources or toolchains",
                lib_path.display(),
                ours[first],
                theirs[first]
            )));
        }
        let msha: libloading::Symbol<fn() -> [u8; 32]> =
            lib.get(b"numsim_module_sha256\0").map_err(|e| err(format!("symbol numsim_module_sha256: {e}")))?;
        if msha() != module_sha {
            return Err(err(format!("{} embeds a different module hash", lib_path.display())));
        }
        let mut steps = Vec::new();
        let mut checks = Vec::new();
        for k in 0..module.kernels.len() {
            let name = format!("{STEP_SYMBOL_PREFIX}{k}\0");
            let s: libloading::Symbol<WarpStepFn> =
                lib.get(name.as_bytes()).map_err(|e| err(format!("symbol {name}: {e}")))?;
            steps.push(*s);
            let name = format!("{CHECK_SYMBOL_PREFIX}{k}\0");
            let c: libloading::Symbol<fn(&Program) -> bool> =
                lib.get(name.as_bytes()).map_err(|e| err(format!("symbol {name}: {e}")))?;
            checks.push(*c);
        }
        // Never unload: the fn pointers above have no lifetime.
        std::mem::forget(lib);
        Ok(LoadedBackend { steps, library: lib_path.to_path_buf(), stats, module_sha256: module_sha, checks })
    }
}

/// Load an already built library for `module` (W8-1): verifies the ABI
/// fingerprint, the embedded module hash, every `{STEP_SYMBOL_PREFIX}{i}`
/// symbol and each kernel's tables. The library is never unloaded, so the
/// returned backend's step functions stay valid for the process lifetime.
pub fn load(path: &Path, module: &Module) -> Result<Backend, BuildError> {
    let loaded = load_library(path, module, sha256(&module.to_bytes()), BuildStats::default())?;
    loaded.check_module(module)?;
    Ok(loaded.backend())
}

/// Print, build (cached by module/source/toolchain hash) and load in one
/// call (W8-1). `cache_dir` overrides `opts.cache_dir`.
pub fn backend_for(module: &Module, opts: &BuildOptions, cache_dir: &Path) -> Result<Backend, BuildError> {
    let opts = BuildOptions { cache_dir: cache_dir.to_path_buf(), ..opts.clone() };
    Ok(build_module(module, &opts)?.backend())
}
