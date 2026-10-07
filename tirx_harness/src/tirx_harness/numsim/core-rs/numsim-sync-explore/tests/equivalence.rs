//! Reductions must not change verdicts: compare every configuration against
//! the unreduced whole-program search on many small random logs.

use numsim_sync_explore::event::LogBuilder;
use numsim_sync_explore::explore::{Limits, Options};
use numsim_sync_explore::projection::ProjectionMode;
use numsim_sync_explore::{check, CheckConfig, SyncOp, Verdict};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 32) as u32
    }

    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }
}

fn random_log(rng: &mut Rng) -> Vec<numsim_sync_explore::SyncEvent> {
    let warps = 2 + rng.below(2);
    let bars = 1 + rng.below(2);
    let mut log = LogBuilder::new();
    for bar in 0..bars {
        log.push(0, 1, SyncOp::MbarInit { bar, expected: 1 + u32::from(rng.below(4) == 0) });
    }
    if rng.below(4) != 0 {
        for warp in 0..warps {
            log.push(warp, 2, SyncOp::NamedSync { bar: 0, expected: warps * 32, count: 32 });
        }
    }
    for warp in 0..warps {
        let mut parity = [0u8; 2];
        for _ in 0..1 + rng.below(4) {
            let bar = rng.below(bars);
            match rng.below(5) {
                0 | 1 => {
                    log.push(warp, 3, SyncOp::MbarArrive { bar, count: 1, expect_tx: 0 });
                }
                2 | 3 => {
                    log.push(warp, 4, SyncOp::MbarWait { bar, parity: parity[bar as usize] });
                    parity[bar as usize] ^= 1;
                }
                _ => {
                    log.push(warp, 5, SyncOp::NamedSync { bar: 1, expected: 64, count: 32 });
                }
            }
        }
    }
    log.build()
}

/// Structured message passing: per barrier one producer and one consumer,
/// optional named-barrier hand-shake (back-pressure), randomly interleaved
/// per warp across barriers.
fn structured_log(rng: &mut Rng) -> Vec<numsim_sync_explore::SyncEvent> {
    let warps = 2 + rng.below(3);
    let bars = 1 + rng.below(2);
    let mut per_warp: Vec<Vec<Vec<SyncOp>>> = vec![Vec::new(); warps as usize];
    for bar in 0..bars {
        let producer = rng.below(warps);
        let consumer = (producer + 1 + rng.below(warps - 1)) % warps;
        let rounds = 1 + rng.below(3);
        let handshake = rng.below(2) == 0;
        let mut produce = Vec::new();
        let mut consume = Vec::new();
        for round in 0..rounds {
            produce.push(SyncOp::MbarArrive { bar, count: 1, expect_tx: 0 });
            consume.push(SyncOp::MbarWait { bar, parity: (round & 1) as u8 });
            if handshake && round + 1 < rounds {
                let named = 1 + bar;
                produce.push(SyncOp::NamedSync { bar: named, expected: 64, count: 32 });
                consume.push(SyncOp::NamedSync { bar: named, expected: 64, count: 32 });
            }
        }
        per_warp[producer as usize].push(produce);
        per_warp[consumer as usize].push(consume);
    }
    let mut log = LogBuilder::new();
    for bar in 0..bars {
        log.push(0, 1, SyncOp::MbarInit { bar, expected: 1 });
    }
    if rng.below(5) != 0 {
        for warp in 0..warps {
            log.push(warp, 2, SyncOp::NamedSync { bar: 0, expected: warps * 32, count: 32 });
        }
    }
    for (warp, mut lists) in per_warp.into_iter().enumerate() {
        let mut cursors = vec![0usize; lists.len()];
        loop {
            let live = (0..lists.len()).filter(|&i| cursors[i] < lists[i].len()).collect::<Vec<_>>();
            if live.is_empty() {
                break;
            }
            let pick = live[rng.below(live.len() as u32) as usize];
            log.push(warp as u32, 10 + pick as u32, lists[pick][cursors[pick]]);
            cursors[pick] += 1;
        }
        lists.clear();
    }
    log.build()
}

fn verdict(events: &[numsim_sync_explore::SyncEvent], config: CheckConfig) -> Verdict {
    check(events, &config).verdict
}

#[test]
fn reductions_preserve_verdicts_on_random_logs() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let limits = Limits { max_states: 200_000, max_transitions: 2_000_000 };
    let oracle = CheckConfig {
        mode: ProjectionMode::Whole,
        certificates: false,
        fingerprints: false,
        explore: Options::NONE,
        limits,
    };
    let variants = [
        ("whole+all", CheckConfig { explore: Options::ALL, ..oracle }),
        ("components+all", CheckConfig { mode: ProjectionMode::Components, explore: Options::ALL, ..oracle }),
        ("per-resource+none", CheckConfig { mode: ProjectionMode::PerResource, ..oracle }),
        ("per-resource+all", CheckConfig { mode: ProjectionMode::PerResource, explore: Options::ALL, ..oracle }),
        ("default", CheckConfig { limits, ..CheckConfig::default() }),
    ];
    let mut counts = [0usize; 3];
    for case in 0..3000 {
        let events = if case % 2 == 0 { random_log(&mut rng) } else { structured_log(&mut rng) };
        let expected = verdict(&events, oracle);
        counts[expected as usize] += 1;
        for (name, config) in variants {
            let actual = verdict(&events, config);
            if actual != expected {
                for event in &events {
                    eprintln!("  w{} #{} {:?}", event.op.warp, event.op.seq, event.kind);
                }
                eprintln!("oracle: {:#?}", check(&events, &oracle));
                eprintln!("{name}: {:#?}", check(&events, &config));
                panic!("case {case} {name}: {actual:?} != {expected:?}");
            }
        }
    }
    // Make sure the generator exercises both outcomes.
    assert!(counts[Verdict::Clean as usize] > 200, "{counts:?}");
    assert!(counts[Verdict::Error as usize] > 200, "{counts:?}");
    eprintln!("verdict counts [clean, error, incomplete]: {counts:?}");
}
