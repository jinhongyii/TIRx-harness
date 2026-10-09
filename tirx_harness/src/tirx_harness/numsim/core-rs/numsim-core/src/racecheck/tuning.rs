//! Switches for the racecheck pruning techniques, so each one's benefit can
//! be measured on corpus-shaped workloads (`numsim-core` example
//! `racecheck_tuning_table`; criterion guards in `benches/racecheck.rs`;
//! table in racecheck-semantics.md "Benchmarks").
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
    FRONTIER_EVICTION: "Single witness per actor: a recent frontier entry (newest EVICT_WINDOW) the new witness subsumes and observes is evicted.",
    ASYNC_SPAN_MERGE: "Coalesce touching weak spans of one async lane before the checker.",
    ADAPTIVE_GC: "GC period scaled to twice the live shadow cells.",
    GC_BACKOFF: "GC period doubled (up to 16x) after a collection that retired under 1/64 of the cells it walked and reclaimed under a quarter of the live async slots.",
}

/// Decision 17: `RaceObserver`s fork a child checker per scheduling
/// partition (default on; findings identical either way, not a pruning
/// switch). Decision run on e5d1582, min of 3 interleaved, 16 workers:
/// mega_moe e24 serial 16.60 s, fork/join 7.68 s (2.16x), with the parallel
/// phase-end collector 6.50 s (2.55x); mega_moe medium (2000 rounds) 908 s
/// vs 555 s (1.64x). At 1 worker no fork is offered (W5-17a): 0.98-1.03x on
/// the 7 recorded fixtures.
pub static FORK_JOIN: AtomicBool = AtomicBool::new(true);

/// Decision 17 / W5-17a: a partition with fewer accesses than this is
/// buffered and replayed by the main checker at the merge instead of being
/// given a child checker (cost only; results are identical either way).
pub static FORK_MIN_ACCESSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

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
