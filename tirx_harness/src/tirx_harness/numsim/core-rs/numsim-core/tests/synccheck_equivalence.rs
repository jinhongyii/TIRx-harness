//! Reductions must not change verdicts: every configuration agrees with the
//! unreduced whole-program search on random contract logs.

mod synccheck_support;

use numsim_core::observe::RecordingObserver;
use numsim_core::report::Verdict;
use numsim_core::synccheck::build::*;
use numsim_core::synccheck::explore::Options;
use numsim_core::synccheck::{check, serialize, ProjectionMode, SynccheckConfig};
use synccheck_support::*;

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

fn random_log(rng: &mut Rng, warps: u32) -> RecordingObserver {
    let bars = 1 + rng.below(2);
    let mut log = LogBuilder::new();
    for b in 0..bars {
        log.cmd(0, 1, mbar(0, 8 * b), init(1 + u64::from(rng.below(4) == 0)));
    }
    if rng.below(4) != 0 {
        cta_sync(&mut log, 0, &(0..warps).collect::<Vec<_>>(), warps);
    }
    for w in 0..warps {
        let mut parity = [0u64; 2];
        for _ in 0..1 + rng.below(4) {
            let b = rng.below(bars);
            match rng.below(5) {
                0 | 1 => {
                    log.cmd(w, 3, mbar(0, 8 * b), arrive(1));
                }
                2 | 3 => {
                    log.cmd(w, 4, mbar(0, 8 * b), wait(parity[b as usize]));
                    parity[b as usize] ^= 1;
                }
                _ => {
                    log.cmd(w, 5, named_bar(0, 1), bar_sync(w, 64));
                }
            }
        }
    }
    log.build()
}

/// Per barrier one producer and one consumer with an optional named-barrier
/// hand-shake, randomly interleaved per warp.
fn structured_log(rng: &mut Rng, warps: u32) -> RecordingObserver {
    use numsim_core::sync::SyncCmd;
    let bars = 1 + rng.below(2);
    let mut per_warp: Vec<Vec<Vec<(numsim_core::sync::ResourceId, SyncCmd)>>> = vec![Vec::new(); warps as usize];
    for b in 0..bars {
        let producer = rng.below(warps);
        let consumer = (producer + 1 + rng.below(warps - 1)) % warps;
        let rounds = 1 + rng.below(3);
        let handshake = rng.below(2) == 0;
        let (mut p, mut c) = (Vec::new(), Vec::new());
        for round in 0..rounds {
            p.push((mbar(0, 8 * b), arrive(1)));
            c.push((mbar(0, 8 * b), wait(u64::from(round & 1))));
            if handshake && round + 1 < rounds {
                p.push((named_bar(0, 1 + b as u8), bar_sync(producer, 64)));
                c.push((named_bar(0, 1 + b as u8), bar_sync(consumer, 64)));
            }
        }
        per_warp[producer as usize].push(p);
        per_warp[consumer as usize].push(c);
    }
    let mut log = LogBuilder::new();
    for b in 0..bars {
        log.cmd(0, 1, mbar(0, 8 * b), init(1));
    }
    if rng.below(5) != 0 {
        cta_sync(&mut log, 0, &(0..warps).collect::<Vec<_>>(), warps);
    }
    for (w, lists) in per_warp.into_iter().enumerate() {
        let mut cursors = vec![0usize; lists.len()];
        loop {
            let live = (0..lists.len()).filter(|&i| cursors[i] < lists[i].len()).collect::<Vec<_>>();
            if live.is_empty() {
                break;
            }
            let pick = live[rng.below(live.len() as u32) as usize];
            let (res, cmd) = lists[pick][cursors[pick]];
            log.cmd(w as u32, 10 + pick as u32, res, cmd);
            cursors[pick] += 1;
        }
    }
    log.build()
}

#[test]
fn reductions_preserve_verdicts_on_random_logs() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut counts = [0usize; 4];
    for case in 0..1500 {
        let warps = 2 + rng.below(3);
        let init = cta(warps);
        let oracle = SynccheckConfig {
            mode: ProjectionMode::Whole,
            certificates: false,
            fingerprints: false,
            explore: Options::NONE,
            state_budget: 200_000,
            ..config(init)
        };
        let log = if case % 2 == 0 { random_log(&mut rng, warps) } else { structured_log(&mut rng, warps) };
        let expected = check(&log, &oracle).verdict;
        counts[expected as usize] += 1;
        let variants = [
            ("whole+all", SynccheckConfig { explore: Options::ALL, ..oracle.clone() }),
            ("components+all", SynccheckConfig { mode: ProjectionMode::Components, explore: Options::ALL, ..oracle.clone() }),
            ("per-resource+none", SynccheckConfig { mode: ProjectionMode::PerResource, ..oracle.clone() }),
            ("per-resource+all", SynccheckConfig { mode: ProjectionMode::PerResource, explore: Options::ALL, ..oracle.clone() }),
            ("default", SynccheckConfig { state_budget: 200_000, ..config(init) }),
        ];
        for (name, cfg) in variants {
            let actual = check(&log, &cfg);
            assert_eq!(
                actual.verdict,
                expected,
                "case {case} {name}: {log:#?}\noracle {:#}\n{name} {:#}",
                serialize(&check(&log, &oracle)),
                serialize(&actual)
            );
        }
    }
    assert!(counts[Verdict::Clean as usize] > 100, "{counts:?}");
    assert!(counts[Verdict::Error as usize] > 100, "{counts:?}");
}
