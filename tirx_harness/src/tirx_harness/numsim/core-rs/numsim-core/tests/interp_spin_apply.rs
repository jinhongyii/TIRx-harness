//! W6 guards for W13's parked spin-loop "apply": a warp parked on a spin loop
//! whose polled inputs are unchanged replays the recorded iteration's events
//! and register effects instead of re-interpreting it, identically in every
//! mode, so the observer stream is unchanged by construction.
//!
//! Baseline arm (passes today): for each spin scenario, status, outputs,
//! stats, the full observer-stream hash and the racecheck payload are equal
//! at 1, 8 and 32 workers.
//!
//! Apply arm (`#[ignore]` until the switch lands): the same quantities with
//! apply on must equal apply off. W13: implement [`set_spin_apply`] and
//! remove the ignores.
use numsim_core::observe::{Access, NoopObserver, Observer, SyncEvent, WarpEnd, WarpId};
use numsim_core::racecheck::observer::{RaceObserver, RacecheckConfig};
use numsim_core::dtype::{Dtype, Ty};
use numsim_core::program::*;
use numsim_core::sched::{self, RunConfig};
use numsim_core::testutil::scenarios::{self, inputs, u32_buf, Scenario};
use numsim_core::testutil::ProgramBuilder;

/// Hash of every observer callback, in delivery order.
#[derive(Default)]
struct StreamHash {
    h: u64,
    n: u64,
}

impl StreamHash {
    fn mix(&mut self, s: &str) {
        for b in s.bytes() {
            self.h = (self.h ^ b as u64).wrapping_mul(0x100_0000_01b3);
        }
        self.n += 1;
    }
}

impl Observer for StreamHash {
    fn wants_word_history(&self) -> bool {
        true
    }
    fn access(&mut self, a: &Access<'_>) {
        let s = format!("A{a:?}");
        self.mix(&s);
    }
    fn sync(&mut self, e: &SyncEvent) {
        let s = format!("S{e:?}");
        self.mix(&s);
    }
    fn warp_done(&mut self, w: WarpId, e: WarpEnd) {
        let s = format!("D{w:?}{e:?}");
        self.mix(&s);
    }
}

/// Everything the apply must not change for one run configuration.
#[derive(Debug, PartialEq)]
struct Observed {
    status: String,
    outputs: String,
    stats: String,
    stream: (u64, u64),
    race: String,
}

/// TODO(W13): turn the parked-spin apply on or off for the next runs (a
/// tuning switch or a `RunConfig` field).
#[allow(dead_code)]
fn set_spin_apply(_on: bool) {
    unimplemented!("W13: parked spin-loop apply switch")
}

fn observe(s: &Scenario, workers: usize) -> Observed {
    let cfg = RunConfig { workers, ..s.config.clone() };
    let plain = sched::run_with_config(&s.module, &s.inputs, &mut NoopObserver, &cfg).expect("run starts");
    let mut hash = StreamHash::default();
    let watched = sched::run_with_config(&s.module, &s.inputs, &mut hash, &cfg).expect("run starts");
    assert_eq!(format!("{:?}", plain.status), format!("{:?}", watched.status), "{}: the observer changes the run", s.name);
    assert_eq!(plain.outputs, watched.outputs, "{}: the observer changes the outputs", s.name);
    let mut race = RaceObserver::new(RacecheckConfig::default());
    let raced = sched::run_with_config(&s.module, &s.inputs, &mut race, &cfg).expect("run starts");
    assert_eq!(format!("{:?}", plain.status), format!("{:?}", raced.status), "{}: racecheck changes the run", s.name);
    Observed {
        status: format!("{:?}", plain.status),
        outputs: format!("{:?}", plain.outputs),
        stats: format!("{:?}", plain.stats),
        stream: (hash.h, hash.n),
        race: format!("{:?}", race.finish()),
    }
}

/// The spin scenarios:
/// - `load_flag_spin(true)`: warp 0 spins on `ld.acquire` until warp 1 (after
///   a busy loop) publishes; the spin is parked for many rounds, then exits.
/// - `load_flag_spin(false)`: nobody publishes; parked until Deadlock.
/// - `deadlock_spin`, `nested_spin_break`: `try_wait` / `test_wait` spins
///   that park until Deadlock (the nested one parks at the outer iteration).
/// - `bounded_probe_loop`: a loop whose state advances, never parked
///   (apply must not fire).
/// - `cross_cluster_flag(true)`: the polled word is written by another
///   cluster (another partition) in a round in which the spinner polls it.
///   The poll must keep reading the round-start value, be tracked as a shard
///   read for the replay order, and see the write from the next round on.
/// - `cross_cluster_sb()`: store buffering across partitions (replay cycle);
///   no spin, but a guard on the shard-read tracking the apply must keep.
///
/// Two single-CTA clusters (two partitions). CTA 0 lane 0 busy-loops for
/// `delay` iterations (several rounds), then stores `data = 5` and
/// `st.release.gpu flag = 1`. CTA 1 lane 0 spins on `ld.acquire.gpu flag`
/// (parked: its polled input is unchanged for many rounds), then reads
/// `data`. The release lands in a round in which the parked spinner polls
/// the same word from another partition: that round's poll must read the
/// round-start 0, and the next round's poll must see 1.
fn late_cross_cluster_flag(delay: u32) -> Scenario {
    let mut b = ProgramBuilder::new("late_cross_cluster_flag", 32);
    b.grid(2, 1, 1);
    let flag = b.global("flag", Dtype::U32);
    let data = b.global("data", Dtype::U32);
    let out = b.global("out", Dtype::U32);
    let cta = b.reg(Ty::U32);
    let lane = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let k = b.reg(Ty::U32);
    let f = b.reg(Ty::U32);
    let v = b.reg(Ty::U32);
    b.read_special(cta, SpecialReg::CtaLinear);
    b.lane_id(lane);
    let k0 = b.k_u32(0);
    let k1 = b.k_u32(1);
    let k5 = b.k_u32(5);
    let kd = b.k_u32(delay);
    b.compare(CmpOp::Eq, Ty::U32, p, lane, k0);
    b.if_(p);
    b.compare(CmpOp::Eq, Ty::U32, p, cta, k0);
    b.if_(p);
    b.mov(k, k0);
    b.loop_begin();
    b.compare(CmpOp::Lt, Ty::U32, p, k, kd);
    b.loop_if(p);
    b.add_u32(k, k, k1);
    b.loop_end();
    b.st_u32(data, k0, k5);
    b.push(Instr::Store { ty: Ty::U32, buf: flag, offset: k0, value: k1, sem: Sem::Release, scope: Scope::Gpu, mods: MemMods::default() });
    b.else_();
    b.mov(f, k0);
    b.loop_begin();
    b.compare(CmpOp::Eq, Ty::U32, p, f, k0);
    b.loop_if(p);
    b.push(Instr::Load { ty: Ty::U32, dst: f, buf: flag, offset: k0, sem: Sem::Acquire, scope: Scope::Gpu, mods: MemMods::default() });
    b.loop_end();
    b.ld_u32(v, data, k0);
    b.st_u32(out, k0, v);
    b.end_if();
    b.end_if();
    b.exit();
    let mut s = Scenario {
        name: "late_cross_cluster_flag",
        module: b.build_module(),
        inputs: inputs(vec![("flag", u32_buf([0])), ("data", u32_buf([0])), ("out", u32_buf([0]))]),
        config: RunConfig::default(),
    };
    s.config.loop_budget = 1 << 40;
    s
}

fn spin_scenarios() -> Vec<Scenario> {
    vec![
        scenarios::load_flag_spin(true),
        scenarios::load_flag_spin(false),
        scenarios::deadlock_spin(),
        scenarios::nested_spin_break(),
        scenarios::bounded_probe_loop(),
        scenarios::cross_cluster_flag(true),
        scenarios::cross_cluster_sb(),
        late_cross_cluster_flag(2000),
    ]
}

#[test]
fn spin_scenarios_are_worker_independent() {
    for s in spin_scenarios() {
        let base = observe(&s, 1);
        for w in [8usize, 32] {
            assert_eq!(observe(&s, w), base, "{}: differs at {w} workers", s.name);
        }
    }
}

/// The parked spin is reached: `load_flag_spin(false)` and the barrier spins
/// end in Deadlock (not loop-budget exhaustion), and `load_flag_spin(true)`
/// completes. A guard that the scenarios still exercise parking.
#[test]
fn spin_scenarios_park() {
    for (s, want) in [
        (scenarios::load_flag_spin(true), "Completed"),
        (scenarios::load_flag_spin(false), "Deadlock"),
        (scenarios::deadlock_spin(), "Deadlock"),
        (scenarios::nested_spin_break(), "Deadlock"),
        (scenarios::bounded_probe_loop(), "Completed"),
        (late_cross_cluster_flag(2000), "Completed"),
    ] {
        let o = observe(&s, 1);
        assert!(o.status.starts_with(want), "{}: {}", s.name, o.status);
    }
}

#[test]
#[ignore = "TODO(W13): parked spin-loop apply switch not implemented yet"]
fn spin_apply_changes_nothing() {
    for s in spin_scenarios() {
        for w in [1usize, 8, 32] {
            set_spin_apply(false);
            let off = observe(&s, w);
            set_spin_apply(true);
            let on = observe(&s, w);
            assert_eq!(on, off, "{}: apply changes the run or the stream at {w} workers", s.name);
        }
    }
}

/// The late cross-partition publish is seen and orders the data read:
/// `out = 5`, and racecheck reports no race (acquire of the release). The
/// spin on an undeclared flag is an `UndeclaredProtocolWord` review, as for
/// any polled plain word.
#[test]
fn late_cross_cluster_publish_is_seen_and_ordered() {
    let s = late_cross_cluster_flag(2000);
    let o = observe(&s, 8);
    assert!(o.outputs.contains("\"out\": ([5, 0, 0, 0]"), "{}", o.outputs);
    assert!(!o.race.contains("DataRace") && !o.race.contains("ProxyRace"), "{}", o.race);
}
