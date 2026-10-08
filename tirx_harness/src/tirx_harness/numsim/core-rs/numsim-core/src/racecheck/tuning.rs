//! Switches for the racecheck pruning techniques, so each one's benefit can
//! be measured on corpus-shaped workloads (`numsim-race-core` example
//! `tuning_table`; table in racecheck-semantics.md "Pruning techniques").
//!
//! All switches default to on. They change cost only, never verdicts —
//! except `frontier_eviction`, whose "off" keeps witnesses a later access
//! is already ordered after (more checks, possibly extra duplicate-site
//! findings, never fewer). Process-wide, read with `Relaxed` loads.
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};

macro_rules! switches {
    ($($name:ident: $doc:literal),* $(,)?) => {
        $( #[doc = $doc] pub static $name: AtomicBool = AtomicBool::new(true); )*
        /// `(name, switch)` of every technique.
        pub fn all() -> Vec<(&'static str, &'static AtomicBool)> {
            vec![$((stringify!($name), &$name)),*]
        }
    };
}

switches! {
    EXACT_HIT: "IntervalShadow exact-hit fast path (an access exactly matching one cell skips split/merge).",
    JOIN_MEMO: "Memoized clock-chunk joins (JoinMemo) and dominating-chunk adoption.",
    FRONTIER_EVICTION: "Single witness per actor: a frontier entry the new witness subsumes is evicted.",
    ASYNC_SPAN_MERGE: "Coalesce touching weak spans of one async lane before the checker.",
    ADAPTIVE_GC: "GC period scaled to twice the live shadow cells.",
}

#[inline(always)]
pub fn on(s: &AtomicBool) -> bool {
    s.load(Relaxed)
}

/// Set every switch (benchmarks).
pub fn set_all(v: bool) {
    for (_, s) in all() {
        s.store(v, Relaxed);
    }
}
