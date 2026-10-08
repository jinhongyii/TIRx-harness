//! A launch-lifetime worker pool for the parallel phase of a round.
//!
//! Rounds are short (often a few milliseconds of work), so spawning threads
//! per round costs more than it saves. The pool's threads live in a
//! `std::thread::scope` around the launch's round loop and run one
//! `par_for` at a time; the calling thread takes part. Work items are
//! claimed dynamically, so which thread runs which partition varies, but the
//! scheduler never depends on it (results are merged in partition order).

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};

/// One `par_for`: `f(i)` for every `i < n`.
#[derive(Clone, Copy)]
struct Task {
    f: *const (dyn Fn(usize) + Sync + 'static),
    n: usize,
    next: *const AtomicUsize,
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
}

impl Pool {
    pub(crate) fn new(workers: usize) -> Pool {
        Pool {
            state: Mutex::new(State { gen: 0, task: None, busy: 0, quit: false }),
            work: Condvar::new(),
            done: Condvar::new(),
            workers,
        }
    }

    /// Body of a pool thread (spawn it `workers` times inside a scope).
    pub(crate) fn worker(&self) {
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
        let next = AtomicUsize::new(0);
        // SAFETY: lifetime erasure only; `par_for` does not return before
        // every worker has finished with the task (see `Task`).
        let f: *const (dyn Fn(usize) + Sync + 'static) = unsafe { std::mem::transmute(f as *const (dyn Fn(usize) + Sync)) };
        let task = Task { f, n, next: &next };
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

/// Calls [`Pool::shutdown`] when dropped (including during unwinding).
pub(crate) struct ShutdownGuard<'a>(pub &'a Pool);

impl Drop for ShutdownGuard<'_> {
    fn drop(&mut self) {
        self.0.shutdown();
    }
}

fn run(task: Task) {
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
