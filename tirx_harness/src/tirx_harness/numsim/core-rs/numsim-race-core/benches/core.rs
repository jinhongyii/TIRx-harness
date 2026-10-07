//! Micro-benchmarks for the pruning tricks the core depends on (plan §1
//! "先量后改"). Each bench pairs the trick with the naive alternative so a
//! regression shows up as a ratio, not an absolute time.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use numsim_race_core::clock::{Clock, JoinMemo, Stamp};
use numsim_race_core::input::*;
use numsim_race_core::shadow::IntervalShadow;
use numsim_race_core::Checker;

fn clock_with(actors: u32, base: u32) -> Clock {
    let mut c = Clock::default();
    for a in 0..actors {
        c.raise(a, base + a % 7 + 1);
    }
    c
}

/// Packed (actor, epoch) stamp: one component read vs an O(actors) VC ≤.
fn packed_stamp(c: &mut Criterion) {
    let mut g = c.benchmark_group("packed_stamp_compare");
    let memo = JoinMemo::default();
    for actors in [64u32, 1024] {
        let clock = clock_with(actors, 10);
        let prior_vc = clock_with(actors, 9);
        let stamps: Vec<Stamp> = (0..256).map(|i| Stamp::new(i % actors, 12)).collect();
        g.bench_with_input(BenchmarkId::new("one_component", actors), &actors, |b, _| {
            b.iter(|| stamps.iter().filter(|s| clock.observes(**s, 0)).count())
        });
        g.bench_with_input(BenchmarkId::new("full_vc_leq", actors), &actors, |b, _| {
            b.iter(|| (0..256).filter(|_| black_box(&prior_vc).leq(&clock, &memo)).count())
        });
    }
    g.finish();
}

/// Exact-hit in-place update vs the split/merge path on a fragmented map.
fn exact_hit(c: &mut Criterion) {
    let mut g = c.benchmark_group("interval_shadow_update");
    let mut s: IntervalShadow<u64> = IntervalShadow::new();
    for i in 0..1024u64 {
        s.update(i * 16..i * 16 + 16, |_, c| *c = i);
    }
    g.bench_function("exact_hit", |b| {
        let mut i = 0u64;
        b.iter(|| {
            i = (i + 1) % 1024;
            s.update(i * 16..i * 16 + 16, |_, c| *c = black_box(i));
        })
    });
    g.bench_function("misaligned_split_merge", |b| {
        let mut i = 0u64;
        b.iter(|| {
            i = (i + 1) % 1023;
            s.update(i * 16 + 8..i * 16 + 24, |_, c| *c = black_box(i));
            // restore the original geometry so every iteration splits
            s.update(i * 16..i * 16 + 16, |_, c| *c = i);
            s.update(i * 16 + 16..i * 16 + 32, |_, c| *c = i + 1);
        })
    });
    g.finish();
}

/// Barrier fan-in: N warps join the same payload. With Arc chunks the
/// first join of an incomparable pair computes it, the rest hit the memo or
/// the pointer-equality path.
fn join_memo(c: &mut Criterion) {
    let mut g = c.benchmark_group("clock_join");
    for actors in [128u32, 4096] {
        let payload = clock_with(actors, 100);
        let warps: Vec<Clock> = (0..32).map(|w| {
            let mut c = clock_with(actors, 50);
            c.raise(w, 1000);
            c
        }).collect();
        g.bench_with_input(BenchmarkId::new("chunked_fanin_disjoint_history", actors), &actors, |b, _| {
            let memo = JoinMemo::default();
            b.iter(|| {
                for w in &warps {
                    let mut x = w.clone();
                    x.join(&payload, &memo);
                    black_box(&x);
                }
            })
        });
        let flat_payload: Vec<u32> = (0..actors).map(|a| payload.get(a)).collect();
        let flat_warps: Vec<Vec<u32>> = warps.iter().map(|w| (0..actors).map(|a| w.get(a)).collect()).collect();
        g.bench_with_input(BenchmarkId::new("flat_fanin_disjoint_history", actors), &actors, |b, _| {
            b.iter(|| {
                for w in &flat_warps {
                    let mut x = w.clone();
                    for (s, p) in x.iter_mut().zip(flat_payload.iter()) {
                        *s = (*s).max(*p);
                    }
                    black_box(&x);
                }
            })
        });
        g.bench_with_input(BenchmarkId::new("synchronized_ptr_eq", actors), &actors, |b, _| {
            let memo = JoinMemo::default();
            let mut synced = payload.clone();
            synced.join(&payload, &memo);
            b.iter(|| {
                let mut x = synced.clone();
                black_box(x.join(&payload, &memo));
            })
        });
    }
    // The realistic barrier case: warps synchronised before, so they share
    // every chunk except the one holding their own component.
    for actors in [128u32, 4096] {
        let base = clock_with(actors, 50);
        let warps: Vec<Clock> = (0..32).map(|w| {
            let mut c = base.clone();
            c.raise(w, 1000);
            c
        }).collect();
        let memo = JoinMemo::default();
        let mut payload = base.clone();
        for w in &warps {
            payload.join(w, &memo);
        }
        g.bench_with_input(BenchmarkId::new("chunked_fanin_shared_history", actors), &actors, |b, _| {
            b.iter(|| {
                for w in &warps {
                    let mut x = w.clone();
                    x.join(&payload, &memo);
                    black_box(&x);
                }
            })
        });
        let flat_payload: Vec<u32> = (0..actors).map(|a| payload.get(a)).collect();
        let flat_warps: Vec<Vec<u32>> = warps.iter().map(|w| (0..actors).map(|a| w.get(a)).collect()).collect();
        g.bench_with_input(BenchmarkId::new("flat_fanin_shared_history", actors), &actors, |b, _| {
            b.iter(|| {
                for w in &flat_warps {
                    let mut x = w.clone();
                    for (s, p) in x.iter_mut().zip(flat_payload.iter()) {
                        *s = (*s).max(*p);
                    }
                    black_box(&x);
                }
            })
        });
    }
    g.finish();
}

/// End-to-end: a tiled producer/consumer loop through the whole checker.
fn checker_loop(c: &mut Criterion) {
    let topo = Topology { warps_per_cta: 4, ctas_per_cluster: 1, num_ctas: 1 };
    let mut ev = vec![Event::Sync(SyncEvent::AllocBegin { alloc: 1, space: Space::Shared, size: 1 << 16, cta: 0 })];
    let mut epoch = [0u32; 4];
    let lanes: Vec<u8> = (0..32).collect();
    for it in 0..64u32 {
        for w in 0..4u32 {
            epoch[w as usize] += 1;
            for &l in &lanes {
                let off = (w as u64 * 32 + l as u64) * 16;
                ev.push(Event::Access(Access {
                    who: Who::Lane { warp: w, lane: l, epoch: epoch[w as usize] },
                    alloc: 1,
                    range: off..off + 16,
                    kind: if it % 2 == 0 { AccessKind::Write } else { AccessKind::Read },
                    order: MemOrder::Weak,
                    scope: None,
                    atomic: false,
                    proxy: Proxy::Generic,
                    domain: Some(Domain::SharedCta),
                    site: w,
                }));
            }
        }
        for w in 0..4u32 {
            epoch[w as usize] += 1;
            ev.push(Event::Sync(SyncEvent::Arrive { warp: w, lanes: LaneMask::FULL, obj: 0, phase: it, release: Some(true), scope: None, epoch: epoch[w as usize] }));
        }
        for w in 0..4u32 {
            epoch[w as usize] += 1;
            ev.push(Event::Sync(SyncEvent::Wait { warp: w, lanes: LaneMask::FULL, obj: 0, phase: it, acquire: Some(true), scope: None, epoch: epoch[w as usize] }));
        }
    }
    let n = ev.len();
    c.bench_function(&format!("checker_tile_loop_{n}_events"), |b| {
        b.iter(|| {
            let r = Checker::run(topo, ev.iter().cloned());
            assert!(r.is_clean());
        })
    });
}

criterion_group!(benches, packed_stamp, exact_hit, join_memo, checker_loop);
criterion_main!(benches);
