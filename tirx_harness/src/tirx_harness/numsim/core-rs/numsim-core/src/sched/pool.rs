//! A launch-lifetime worker pool for the parallel phase of a round.
//!
//! Rounds are short (often a few milliseconds of work), so spawning threads
//! per round costs more than it saves. The pool's threads live in a
//! `std::thread::scope` around the launch's round loop and run one
//! `par_for` at a time; the calling thread takes part. Work items are
//! claimed dynamically, so which thread runs which partition varies, but the
//! scheduler never depends on it (results are merged in partition order).
//!
//! `RunConfig::pin_workers` (default off) pins each participant to one CPU
//! of the inherited affinity mask, filling one L3 group (CCD) before the
//! next and SMT siblings last ([`placement`]), and hands each partition
//! first to the participant that ran it last round, idle participants
//! stealing the rest ([`Pool::par_for_sticky`]). Off, the pool makes no
//! affinity calls. Either way results and observer streams are the same.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};

/// One `par_for`: `f(i)` for every `i < n`.
#[derive(Clone, Copy)]
struct Task {
    f: *const (dyn Fn(usize) + Sync + 'static),
    n: usize,
    next: *const AtomicUsize,
    /// Sticky claiming ([`Pool::par_for_sticky`]): the preferred
    /// participant of each index and one claim flag per index.
    sticky: Option<(*const u32, *const AtomicBool)>,
}

thread_local! {
    /// This thread's participant id in a pinned pool: 0 = the calling
    /// thread, 1.. = pool threads in start order.
    static PARTICIPANT: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// The calling thread's participant id (see [`Pool::par_for_sticky`]).
pub(crate) fn participant() -> u32 {
    PARTICIPANT.with(|p| p.get())
}

// SAFETY: a task is only dereferenced between its publication and the
// moment `par_for` observes every worker done, during which the borrowed
// closure and counter outlive it (`par_for` blocks until then).
unsafe impl Send for Task {}

struct State {
    gen: u64,
    task: Option<Task>,
    /// Workers still running the current task.
    busy: usize,
    quit: bool,
}

pub(crate) struct Pool {
    state: Mutex<State>,
    work: Condvar,
    done: Condvar,
    workers: usize,
    /// Next pool-thread participant id.
    ids: AtomicU32,
    /// CPUs in placement order (participant `i` runs on `cores[i % len]`);
    /// empty = not pinned (no affinity calls at all).
    cores: Vec<usize>,
}

impl Pool {
    /// `workers` pool threads (the calling thread is one more
    /// participant); `pin`: [`RunConfig::pin_workers`](super::RunConfig).
    pub(crate) fn new(workers: usize, pin: bool) -> Pool {
        Pool {
            state: Mutex::new(State { gen: 0, task: None, busy: 0, quit: false }),
            work: Condvar::new(),
            done: Condvar::new(),
            workers,
            ids: AtomicU32::new(1),
            cores: if pin { placement() } else { Vec::new() },
        }
    }

    /// Whether participants are pinned (and partitions handed out sticky).
    pub(crate) fn pinned(&self) -> bool {
        !self.cores.is_empty()
    }

    /// Pin the calling thread (participant 0) for the launch; its previous
    /// affinity is restored when the guard drops. `None` when not pinned.
    pub(crate) fn pin_caller(&self) -> Option<RestoreAffinity> {
        let &c = self.cores.first()?;
        PARTICIPANT.with(|p| p.set(0));
        let saved = affinity::get();
        affinity::set(&[c]);
        Some(RestoreAffinity(saved))
    }

    /// Body of a pool thread (spawn it `workers` times inside a scope).
    pub(crate) fn worker(&self) {
        let _fp = FpEnvGuard::enter();
        if self.pinned() {
            let me = self.ids.fetch_add(1, Ordering::Relaxed);
            PARTICIPANT.with(|p| p.set(me));
            affinity::set(&[self.cores[me as usize % self.cores.len()]]);
        }
        let mut seen = 0u64;
        loop {
            let task = {
                let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
                while st.gen == seen && !st.quit {
                    st = self.work.wait(st).unwrap_or_else(|e| e.into_inner());
                }
                if st.quit {
                    return;
                }
                seen = st.gen;
                st.task.expect("published with the generation")
            };
            run(task);
            let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
            st.busy -= 1;
            if st.busy == 0 {
                self.done.notify_all();
            }
        }
    }

    /// Run `f(i)` for `i in 0..n` on the pool and the calling thread;
    /// returns when all are done. `f` must not panic (wrap its body).
    pub(crate) fn par_for(&self, n: usize, f: &(dyn Fn(usize) + Sync)) {
        self.par_for_inner(n, f, None);
    }

    /// [`Pool::par_for`] where index `i` is first offered to participant
    /// `prefs[i]` (`u32::MAX`: none); every participant then claims the
    /// indices nobody took. With pinned participants a partition keeps
    /// running on the core (and L3) that ran it last round while idle
    /// threads still steal for balance. Which thread runs which index never
    /// affects results.
    pub(crate) fn par_for_sticky(&self, prefs: &[u32], f: &(dyn Fn(usize) + Sync)) {
        let claimed: Vec<AtomicBool> = (0..prefs.len()).map(|_| AtomicBool::new(false)).collect();
        self.par_for_inner(prefs.len(), f, Some((prefs.as_ptr(), claimed.as_ptr())));
    }

    fn par_for_inner(&self, n: usize, f: &(dyn Fn(usize) + Sync), sticky: Option<(*const u32, *const AtomicBool)>) {
        let next = AtomicUsize::new(0);
        // SAFETY: lifetime erasure only; `par_for` does not return before
        // every worker has finished with the task (see `Task`).
        let f: *const (dyn Fn(usize) + Sync + 'static) = unsafe { std::mem::transmute(f as *const (dyn Fn(usize) + Sync)) };
        let task = Task { f, n, next: &next, sticky };
        {
            let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
            st.task = Some(task);
            st.gen += 1;
            st.busy = self.workers;
            self.work.notify_all();
        }
        run(task);
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        while st.busy > 0 {
            st = self.done.wait(st).unwrap_or_else(|e| e.into_inner());
        }
        st.task = None;
    }

    /// Stop the pool threads (call before the scope ends).
    pub(crate) fn shutdown(&self) {
        // Tolerate a poisoned lock: this also runs while unwinding.
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        st.quit = true;
        self.work.notify_all();
    }
}

/// The engine's floating-point environment for the current thread
/// (round-to-nearest-even, FTZ/DAZ off, as legacy `fp-env`), restoring the
/// thread's previous environment when dropped: a run never inherits the
/// caller's rounding mode or flush-to-zero, and never leaks its own.
pub(crate) struct FpEnvGuard {
    #[cfg(target_os = "linux")]
    round: core::ffi::c_int,
    #[cfg(target_arch = "x86_64")]
    mxcsr: u32,
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn fegetround() -> core::ffi::c_int;
    fn fesetround(round: core::ffi::c_int) -> core::ffi::c_int;
}

impl FpEnvGuard {
    pub(crate) fn enter() -> FpEnvGuard {
        #[cfg(target_os = "linux")]
        // SAFETY: plain libc calls on the calling thread's environment.
        let round = unsafe { fegetround() };
        #[cfg(target_arch = "x86_64")]
        #[allow(deprecated)]
        // SAFETY: reading / writing MXCSR of the calling thread.
        let mxcsr = unsafe {
            let m = core::arch::x86_64::_mm_getcsr();
            // FTZ (bit 15) and DAZ (bit 6) off.
            core::arch::x86_64::_mm_setcsr(m & !((1 << 15) | (1 << 6)));
            m
        };
        let _ = numsim_oplib::fpenv::set_round_to_nearest();
        FpEnvGuard {
            #[cfg(target_os = "linux")]
            round,
            #[cfg(target_arch = "x86_64")]
            mxcsr,
        }
    }
}

impl Drop for FpEnvGuard {
    fn drop(&mut self) {
        #[cfg(target_os = "linux")]
        // SAFETY: restores the value read in `enter`.
        unsafe {
            fesetround(self.round);
        }
        #[cfg(target_arch = "x86_64")]
        #[allow(deprecated)]
        // SAFETY: restores the value read in `enter`.
        unsafe {
            core::arch::x86_64::_mm_setcsr(self.mxcsr);
        }
    }
}

/// Calls [`Pool::shutdown`] when dropped (including during unwinding).
pub(crate) struct ShutdownGuard<'a>(pub &'a Pool);

impl Drop for ShutdownGuard<'_> {
    fn drop(&mut self) {
        self.0.shutdown();
    }
}

fn run(task: Task) {
    if let Some((prefs, claimed)) = task.sticky {
        // SAFETY: see `Task`; both arrays have `task.n` elements.
        let (prefs, claimed) = unsafe { (std::slice::from_raw_parts(prefs, task.n), std::slice::from_raw_parts(claimed, task.n)) };
        let me = participant();
        let go = |i: usize| {
            if !claimed[i].load(Ordering::Relaxed) && !claimed[i].swap(true, Ordering::AcqRel) {
                let _ = catch_unwind(AssertUnwindSafe(|| unsafe { (*task.f)(i) }));
            }
        };
        for (i, &p) in prefs.iter().enumerate() {
            if p == me {
                go(i);
            }
        }
        // Steal what is left, from a participant-dependent offset.
        let start = (me as usize).wrapping_mul(7919) % task.n.max(1);
        for k in 0..task.n {
            go((start + k) % task.n);
        }
        return;
    }
    loop {
        // SAFETY: see `Task`.
        let i = unsafe { &*task.next }.fetch_add(1, Ordering::Relaxed);
        if i >= task.n {
            break;
        }
        // A panic must not unwind out of a pool thread (the round would
        // never complete); callers wrap their bodies, this is a backstop.
        let _ = catch_unwind(AssertUnwindSafe(|| unsafe { (*task.f)(i) }));
    }
}

/// Restores the calling thread's affinity mask when dropped (also while
/// unwinding); see [`Pool::pin_caller`].
pub(crate) struct RestoreAffinity(Vec<usize>);

impl Drop for RestoreAffinity {
    fn drop(&mut self) {
        affinity::set(&self.0);
    }
}

/// CPU placement for pinned participants: the calling thread's allowed CPUs
/// (an already-restricted mask is honoured; nothing outside it is used)
/// ordered so consecutive participants share an L3 (CCD): physical cores
/// first, grouped by L3 in order of the group's lowest CPU, then every SMT
/// sibling in the same order. With fewer CPUs than participants they wrap
/// round-robin. Empty (no pinning) if the mask cannot be read.
pub(crate) fn placement() -> Vec<usize> {
    let allowed = affinity::get();
    if allowed.is_empty() {
        return Vec::new();
    }
    let read = |cpu: usize, f: &str| std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{cpu}/{f}")).ok();
    let mut keyed: Vec<(bool, usize, usize)> = allowed
        .iter()
        .map(|&c| {
            let l3 = read(c, "cache/index3/shared_cpu_list").map(|s| cpu_list(&s)).and_then(|l| l.first().copied()).unwrap_or(c);
            // A sibling: some lower-numbered SMT sibling is also allowed.
            let sib = read(c, "topology/thread_siblings_list").is_some_and(|s| cpu_list(&s).iter().any(|&o| o < c && allowed.binary_search(&o).is_ok()));
            (sib, l3, c)
        })
        .collect();
    keyed.sort_unstable();
    keyed.into_iter().map(|k| k.2).collect()
}

/// Parse a sysfs CPU list (`0-3,128,130-131`).
fn cpu_list(s: &str) -> Vec<usize> {
    let mut out = Vec::new();
    for part in s.trim().split(',').filter(|p| !p.is_empty()) {
        let mut ends = part.splitn(2, '-').map(|x| x.trim().parse::<usize>());
        match (ends.next(), ends.next()) {
            (Some(Ok(a)), None) => out.push(a),
            (Some(Ok(a)), Some(Ok(b))) if a <= b => out.extend(a..=b),
            _ => {}
        }
    }
    out.sort_unstable();
    out
}

/// Thread affinity (Linux `sched_{get,set}affinity` on the calling thread;
/// elsewhere a no-op with an empty mask, so nothing is ever pinned).
pub(crate) mod affinity {
    #[cfg(target_os = "linux")]
    const WORDS: usize = 64; // 4096 CPUs
    #[cfg(target_os = "linux")]
    unsafe extern "C" {
        fn sched_getaffinity(pid: i32, size: usize, mask: *mut u64) -> i32;
        fn sched_setaffinity(pid: i32, size: usize, mask: *const u64) -> i32;
    }

    /// CPUs the calling thread may run on, ascending (empty on error).
    pub(crate) fn get() -> Vec<usize> {
        #[cfg(target_os = "linux")]
        {
            let mut m = [0u64; WORDS];
            // SAFETY: a valid buffer of `WORDS` words.
            if unsafe { sched_getaffinity(0, WORDS * 8, m.as_mut_ptr()) } != 0 {
                return Vec::new();
            }
            (0..WORDS * 64).filter(|&c| m[c / 64] >> (c % 64) & 1 == 1).collect()
        }
        #[cfg(not(target_os = "linux"))]
        Vec::new()
    }

    /// Restrict the calling thread to `cpus`; failures are ignored (pinning
    /// is a placement hint, never a reason to fail a run).
    pub(crate) fn set(cpus: &[usize]) {
        #[cfg(target_os = "linux")]
        {
            if cpus.is_empty() {
                return;
            }
            let mut m = [0u64; WORDS];
            for &c in cpus.iter().filter(|&&c| c < WORDS * 64) {
                m[c / 64] |= 1 << (c % 64);
            }
            // SAFETY: a valid buffer of `WORDS` words.
            let _ = unsafe { sched_setaffinity(0, WORDS * 8, m.as_ptr()) };
        }
        #[cfg(not(target_os = "linux"))]
        let _ = cpus;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_lists_parse() {
        assert_eq!(cpu_list("0-3,128,130-131\n"), vec![0, 1, 2, 3, 128, 130, 131]);
        assert_eq!(cpu_list(""), Vec::<usize>::new());
    }

    /// Placement stays inside the inherited mask and covers all of it; a
    /// pinned pool with more participants than CPUs runs (round-robin) and
    /// restores the caller's mask.
    #[cfg(target_os = "linux")]
    #[test]
    fn placement_honours_a_restricted_mask() {
        let original = affinity::get();
        assert!(!original.is_empty());
        let mask: Vec<usize> = original.iter().copied().take(2).collect();
        let t = std::thread::spawn(move || {
            affinity::set(&mask);
            let mut p = placement();
            p.sort_unstable();
            assert_eq!(p, mask);
            let pool = Pool::new(4, true);
            let seen: Vec<AtomicUsize> = (0..16).map(|_| AtomicUsize::new(0)).collect();
            let bad = AtomicBool::new(false);
            std::thread::scope(|scope| {
                let restore = pool.pin_caller();
                assert!(restore.is_some());
                for _ in 0..4 {
                    scope.spawn(|| pool.worker());
                }
                let _stop = ShutdownGuard(&pool);
                let prefs: Vec<u32> = (0..16).map(|i| i % 5).collect();
                pool.par_for_sticky(&prefs, &|i| {
                    seen[i].fetch_add(1, Ordering::Relaxed);
                    let now = affinity::get();
                    if now.len() != 1 || !mask.contains(&now[0]) {
                        bad.store(true, Ordering::Relaxed);
                    }
                });
                drop(restore);
            });
            assert!(seen.iter().all(|c| c.load(Ordering::Relaxed) == 1));
            assert!(!bad.load(Ordering::Relaxed), "a participant ran outside its one-CPU pin");
            assert_eq!(affinity::get(), mask);
        });
        t.join().unwrap();
        assert_eq!(affinity::get(), original);
    }

    /// Off: no affinity change anywhere.
    #[cfg(target_os = "linux")]
    #[test]
    fn unpinned_pool_leaves_affinity_alone() {
        let pool = Pool::new(2, false);
        assert!(!pool.pinned() && pool.pin_caller().is_none());
        let before = affinity::get();
        let bad = AtomicBool::new(false);
        std::thread::scope(|scope| {
            for _ in 0..2 {
                scope.spawn(|| pool.worker());
            }
            let _stop = ShutdownGuard(&pool);
            pool.par_for(8, &|_| {
                if affinity::get() != before {
                    bad.store(true, Ordering::Relaxed);
                }
            });
        });
        assert!(!bad.load(Ordering::Relaxed));
    }
}
