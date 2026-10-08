//! W6 adversarial review of the partitioned scheduler (934f2b3) against the
//! determinism / observer-independence invariants I7, I8, I10-I13
//! (checker-review notes; CONTRACT_REQUESTS "W6 partition review").
//!
//! Oracle for declared-word numbering (I10): racecheck numbers a declared
//! word's history by delivered write `LaneSpan`s (index 0 = launch value).
//! Every `WaitVerdicts` must therefore have `observed` = writes delivered so
//! far, and each accepted index i >= 1 must name a delivered write whose
//! value satisfies the waiter's predicate.
use numsim_core::dtype::{Dtype, Ty};
use numsim_core::observe::{Access, Actor, NoopObserver, Observer, SyncEvent, SyncKind};
use numsim_core::program::*;
use numsim_core::sched::{self, ArgValue, Inputs, RunConfig, RunOutcome, RunStatus};
use numsim_core::testutil::scenarios::{self, Scenario};
use numsim_core::testutil::ProgramBuilder;
use numsim_core::Module;

fn u32_buf(v: impl IntoIterator<Item = u32>) -> ArgValue {
    scenarios::u32_buf(v)
}

fn u32s(o: &RunOutcome, name: &str) -> Vec<u32> {
    o.outputs.buffers[name].0.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}

/// Records declared-word writes (by writing warp) and verdicts in delivery
/// order, plus a hash of the whole stream.
#[derive(Default)]
struct WordOracle {
    /// Writer warp of each delivered declared-word write `LaneSpan`.
    writers: Vec<u32>,
    /// (waiter warp, observed, accepted indices, writes delivered so far)
    verdicts: Vec<(u32, u32, Vec<usize>, usize)>,
    h: u64,
}

impl WordOracle {
    fn mix(&mut self, s: &str) {
        for b in s.bytes() {
            self.h = (self.h ^ b as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
}

fn warp_of(a: Actor) -> u32 {
    match a {
        Actor::Warp { warp, .. } => warp.0,
        _ => u32::MAX,
    }
}

impl Observer for WordOracle {
    fn wants_word_history(&self) -> bool {
        true
    }
    fn access(&mut self, a: &Access<'_>) {
        if a.declared_word && a.writes() {
            for _ in a.spans.iter() {
                self.writers.push(warp_of(a.actor));
            }
        }
        let s = format!("{:?}/{:?}/{:?}/{:?}/{:?}/{:?}", a.actor, a.site, a.alloc, a.kind, a.spans, a.declared_word);
        self.mix(&s);
    }
    fn sync(&mut self, e: &SyncEvent) {
        if let SyncKind::WaitVerdicts { verdicts, .. } = &e.kind {
            for v in verdicts {
                let acc: Vec<usize> =
                    v.accepted.iter().enumerate().flat_map(|(w, bits)| (0..64).filter(move |b| bits >> b & 1 == 1).map(move |b| w * 64 + b)).collect();
                self.verdicts.push((warp_of(e.actor), v.observed, acc, self.writers.len()));
            }
        }
        let s = format!("{e:?}");
        self.mix(&s);
    }
}

/// `ctas` single-CTA clusters (one warp each). Lane 0 of CTA c stores
/// `c + 1` (release) to the declared word `flag`, then
/// `wait_until(flag == c + 1)`; `out[c]` = the accepted value.
/// Every CTA accepts its own write (it reads its own store first).
fn own_value_flags(ctas: u32) -> Scenario {
    let mut b = ProgramBuilder::new("own_value_flags", 32);
    b.grid(ctas, 1, 1);
    let flag = b.global("flag", Dtype::U32);
    let out = b.global("out", Dtype::U32);
    b.declare_sync_words(flag);
    let lane = b.reg(Ty::U32);
    let cta = b.reg(Ty::U32);
    let mine = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let fa = b.reg(Ty::U64);
    let got = b.reg(Ty::U32);
    let arg = b.reg(Ty::U32);
    let res = b.reg(Ty::PRED);
    b.lane_id(lane);
    b.read_special(cta, SpecialReg::CtaLinear);
    let k0 = b.k_u32(0);
    let k1 = b.k_u32(1);
    b.addr_of(fa, flag, k0);
    b.add_u32(mine, cta, k1);
    b.compare(CmpOp::Eq, Ty::U32, p, lane, k0);
    b.if_(p);
    b.site("publish", 1);
    b.push(Instr::StoreAddr { ty: Ty::U32, addr: fa.into(), space: AddrSpace::Generic, value: mine.into(), sem: Sem::Release, scope: Scope::Gpu, mods: MemMods::default() });
    b.site("wait_until", 2);
    let placeholder = b.push(Instr::Nop);
    b.no_site();
    b.st_u32(out, cta, got);
    b.end_if();
    b.exit();
    let mut prog = b.build();
    let start = Pc(prog.code.len() as u32);
    prog.code.push(Instr::Compare { op: CmpOp::Eq, ty: Ty::U32, dst: res, a: arg.into(), b: mine.into() });
    prog.code_sites.push(numsim_core::site::SiteId::NONE);
    prog.preds.push(PredProgram { arg, start, end: Pc(start.0 + 1), result: res, reads_memory: false });
    prog.code[placeholder.0 as usize] = Instr::WaitUntil {
        dst: got,
        addr: fa.into(),
        ty: Ty::U32,
        space: AddrSpace::Generic,
        sem: Sem::Acquire,
        scope: Scope::Gpu,
        pred: PredId(0),
        captures: vec![mine],
    };
    prog.validate().expect("valid");
    let inputs: Inputs = scenarios::inputs(vec![("flag", u32_buf([0])), ("out", u32_buf(vec![0; ctas as usize]))]);
    let mut config = RunConfig::default();
    config.loop_budget = 1 << 40;
    Scenario { name: "own_value_flags", module: Module::new(vec![prog]), inputs, config }
}

fn run(s: &Scenario, workers: usize, obs: &mut dyn Observer) -> RunOutcome {
    run_seed(s, workers, s.config.seed, obs)
}

fn run_seed(s: &Scenario, workers: usize, seed: u64, obs: &mut dyn Observer) -> RunOutcome {
    let cfg = RunConfig { workers, seed, ..s.config.clone() };
    sched::run_with_config(&s.module, &s.inputs, obs, &cfg).expect("run starts")
}

/// I10/I12 (W6 S-a follow-up): two partitions store to the same declared
/// word in one round and each waits for its own value. Neither poll is a
/// round-start read (each reads its own write), so the replay order is
/// partition order; partition 1's verdict was computed on its local
/// history (launch entries + its own write) and its indices no longer match
/// the merged history, where partition 0's write comes first.
#[test]
#[ignore = "xfail: CONTRACT_REQUESTS W6-P1 (W2) -- verdicts computed on partition-local history"]
fn same_round_writers_each_waiting_on_their_own_value() {
    for workers in [1usize, 8] {
        let s = own_value_flags(2);
        let mut obs = WordOracle::default();
        let o = run(&s, workers, &mut obs);
        assert_eq!(o.status, RunStatus::Completed, "{:?}", o.status);
        assert_eq!(u32s(&o, "out"), vec![1, 2]);
        check_verdicts(&obs);
    }
}

/// Each verdict: `observed` = writes delivered so far, and every accepted
/// index >= 1 names a write by the waiter itself (the only writer of its
/// value in `own_value_flags`).
fn check_verdicts(obs: &WordOracle) {
    assert!(!obs.verdicts.is_empty());
    for (waiter, observed, acc, delivered) in &obs.verdicts {
        assert_eq!(*observed as usize, *delivered, "observed index vs delivered writes (writers {:?}): {:?}", obs.writers, obs.verdicts);
        for &i in acc {
            assert!(i >= 1 && i <= obs.writers.len(), "accepted index {i} out of range: {:?}", obs.verdicts);
            assert_eq!(obs.writers[i - 1], *waiter, "accepted index {i} names warp {}'s write, not the waiter {waiter}'s: writers {:?}", obs.writers[i - 1], obs.writers);
        }
    }
}

/// I10 control: three partitions, the first two storing different values in
/// one round and the third polling for the second one's value.
#[test]
fn two_writers_and_a_poller_number_like_the_delivery() {
    // own_value_flags(3): CTA 2 also waits for its own value; the point is
    // the third partition reading round-start bytes of a word two others
    // write in that round.
    let s = own_value_flags(3);
    let mut a = WordOracle::default();
    let o1 = run(&s, 1, &mut a);
    let mut b = WordOracle::default();
    let o8 = run(&s, 8, &mut b);
    assert_eq!(o1.status, o8.status);
    assert_eq!(o1.outputs, o8.outputs);
    assert_eq!(a.h, b.h, "observer stream differs between 1 and 8 workers");
}

/// I10 with serial-phase writes: CTAs 0 and 1 atomically add to the word
/// (serial points), CTA 2 waits for the count to reach 2.
#[test]
fn wait_satisfied_by_serial_phase_writes_numbers_like_the_delivery() {
    for workers in [1usize, 8] {
        let s = scenarios::atomic_count_wait(3);
        let mut obs = WordOracle::default();
        let o = run(&s, workers, &mut obs);
        assert_eq!(o.status, RunStatus::Completed, "{:?}", o.status);
        assert_eq!(obs.verdicts.len(), 3, "{:?}", obs.verdicts);
        for (_, observed, acc, delivered) in &obs.verdicts {
            assert_eq!(*observed as usize, *delivered, "{:?}", obs.verdicts);
            // count == 3 is the 3rd write: exactly index 3.
            assert_eq!(acc, &vec![3], "{:?} writers {:?}", obs.verdicts, obs.writers);
        }
    }
}

/// I13: overflow is decided on the merged history, identically for any
/// worker count, and a verdict is never computed on a truncated history.
#[test]
fn history_overflow_is_worker_independent() {
    let mut seen = None;
    for workers in [1usize, 8] {
        let s = scenarios::word_history_overflow((1 << 16) + 8);
        let mut obs = WordOracle::default();
        let o = run(&s, workers, &mut obs);
        assert!(matches!(o.status, RunStatus::Incomplete { .. }), "{:?}", o.status);
        let now = (format!("{:?}", o.status), obs.h);
        if let Some(prev) = &seen {
            assert_eq!(prev, &now, "workers {workers}");
        }
        seen = Some(now);
    }
}

/// I7/I8 over every scenario with several clusters or async landings:
/// status, outputs and stats equal with no observer and with a history
/// observer (seeded `land` subsets included), and the history observer's
/// stream equal at 1 and 8 workers.
#[test]
fn every_scenario_is_observer_and_worker_independent() {
    let mut failures = Vec::new();
    for s in scenarios::all() {
        for seed in [0u64, 3, 11] {
            let plain = run_seed(&s, 1, seed, &mut NoopObserver);
            let mut o1 = WordOracle::default();
            let w1 = run_seed(&s, 1, seed, &mut o1);
            let mut o8 = WordOracle::default();
            let w8 = run_seed(&s, 8, seed, &mut o8);
            let key = |o: &RunOutcome| (format!("{:?}", o.status), o.outputs.clone(), o.stats.clone());
            if key(&plain) != key(&w1) {
                failures.push(format!("{} seed {seed}: observer changes the run: {:?} vs {:?}", s.name, plain.status, w1.status));
            }
            if key(&w1) != key(&w8) || o1.h != o8.h {
                failures.push(format!("{} seed {seed}: 1 vs 8 workers differ (stream equal: {})", s.name, o1.h == o8.h));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

const MAX: u32 = numsim_core::interp::aux::MAX_WORD_HISTORY as u32;
const SENTINEL: u32 = 0xFFFF_FFFF;

/// Three single-CTA clusters. Lane 0 of CTA c < 2 does `stores[c]` relaxed
/// stores to the declared word `flag`, then `atoms[c]` relaxed `atom.add`s
/// (serial points), then a release store of `fin[c]`. CTA 2 waits
/// `wait_until(flag == SENTINEL)`. A huge quantum lets each writer's store
/// loop finish in one slice, so both writers' stores land in one round.
fn overflow_writers(stores: [u32; 2], atoms: [u32; 2], finals: [u32; 2]) -> Scenario {
    let mut b = ProgramBuilder::new("overflow_writers", 32);
    b.grid(3, 1, 1);
    let flag = b.global("flag", Dtype::U32);
    let ns = b.global("ns", Dtype::U32);
    let na = b.global("na", Dtype::U32);
    let fin = b.global("fin", Dtype::U32);
    b.declare_sync_words(flag);
    let lane = b.reg(Ty::U32);
    let cta = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let k = b.reg(Ty::U32);
    let n = b.reg(Ty::U32);
    let v = b.reg(Ty::U32);
    let old = b.reg(Ty::U32);
    let fa = b.reg(Ty::U64);
    let got = b.reg(Ty::U32);
    let arg = b.reg(Ty::U32);
    let res = b.reg(Ty::PRED);
    b.lane_id(lane);
    b.read_special(cta, SpecialReg::CtaLinear);
    let k0 = b.k_u32(0);
    let k1 = b.k_u32(1);
    let k2 = b.k_u32(2);
    let ks = b.k_u32(SENTINEL);
    b.addr_of(fa, flag, k0);
    b.compare(CmpOp::Eq, Ty::U32, p, lane, k0);
    b.if_(p);
    b.compare(CmpOp::Lt, Ty::U32, p, cta, k2);
    b.if_(p);
    // Relaxed stores.
    b.ld_u32(n, ns, cta);
    b.mov(k, k0);
    b.loop_begin();
    b.compare(CmpOp::Lt, Ty::U32, p, k, n);
    b.loop_if(p);
    b.add_u32(v, k, k2);
    b.push(Instr::StoreAddr { ty: Ty::U32, addr: fa.into(), space: AddrSpace::Generic, value: v.into(), sem: Sem::Relaxed, scope: Scope::Gpu, mods: MemMods::default() });
    b.add_u32(k, k, k1);
    b.loop_end();
    // Serial-phase RMWs.
    b.ld_u32(n, na, cta);
    b.mov(k, k0);
    b.loop_begin();
    b.compare(CmpOp::Lt, Ty::U32, p, k, n);
    b.loop_if(p);
    b.push(Instr::Atom { op: AtomOp::Add, ty: Ty::U32, dst: Some(old), addr: fa.into(), space: AddrSpace::Global, value: k1, cmp: None, sem: Sem::Relaxed, scope: Scope::Gpu, ftz: false });
    b.add_u32(k, k, k1);
    b.loop_end();
    // Final release store of `fin[cta]`.
    b.ld_u32(v, fin, cta);
    b.push(Instr::StoreAddr { ty: Ty::U32, addr: fa.into(), space: AddrSpace::Generic, value: v.into(), sem: Sem::Release, scope: Scope::Gpu, mods: MemMods::default() });
    b.else_();
    b.site("wait_until", 1);
    let placeholder = b.push(Instr::Nop);
    b.no_site();
    b.end_if();
    b.end_if();
    b.exit();
    let mut prog = b.build();
    let start = Pc(prog.code.len() as u32);
    let Operand::Const(sc) = ks else { unreachable!() };
    prog.code.push(Instr::Compare { op: CmpOp::Eq, ty: Ty::U32, dst: res, a: arg.into(), b: Operand::Const(sc) });
    prog.code_sites.push(numsim_core::site::SiteId::NONE);
    prog.preds.push(PredProgram { arg, start, end: Pc(start.0 + 1), result: res, reads_memory: false });
    prog.code[placeholder.0 as usize] = Instr::WaitUntil {
        dst: got,
        addr: fa.into(),
        ty: Ty::U32,
        space: AddrSpace::Generic,
        sem: Sem::Acquire,
        scope: Scope::Gpu,
        pred: PredId(0),
        captures: vec![],
    };
    prog.validate().expect("valid");
    let inputs: Inputs = scenarios::inputs(vec![("flag", u32_buf([0])), ("ns", u32_buf(stores)), ("na", u32_buf(atoms)), ("fin", u32_buf(finals))]);
    let mut config = RunConfig::default();
    config.loop_budget = 1 << 40;
    config.quantum = 1 << 30;
    Scenario { name: "overflow_writers", module: Module::new(vec![prog]), inputs, config }
}

/// Run at 1 and 8 workers: identical status (which must be `incomplete`)
/// and identical observer stream. Returns the declared-word writes seen.
fn overflow_is_incomplete_everywhere(s: &Scenario) -> usize {
    let mut seen: Option<(String, u64, usize)> = None;
    for workers in [1usize, 8] {
        let mut obs = WordOracle::default();
        let o = run(s, workers, &mut obs);
        match &o.status {
            RunStatus::Incomplete { reason, .. } => assert!(reason.contains("history"), "{reason}"),
            other => panic!("workers {workers}: expected incomplete, got {other:?}"),
        }
        assert!(obs.verdicts.is_empty(), "a verdict over a truncated history: {:?}", obs.verdicts);
        let now = (format!("{:?}", o.status), obs.h, obs.writers.len());
        if let Some(prev) = &seen {
            assert_eq!(prev, &now, "workers {workers} differ from workers 1");
        }
        seen = Some(now);
    }
    seen.unwrap().2
}

/// I13: each writer stays under MAX_WORD_HISTORY in its own partition
/// (MAX/2 + 4 writes each, all in round 0), so only the cross-partition
/// merge crosses it.
#[test]
fn history_overflow_crossed_only_in_the_partition_merge() {
    let each = MAX / 2 + 4;
    let s = overflow_writers([each - 1, each - 1], [0, 0], [SENTINEL, SENTINEL]); // + the sentinel store
    let writes = overflow_is_incomplete_everywhere(&s);
    assert_eq!(writes, 2 * each as usize, "every write is still delivered to the observer");
}

/// I13: the parallel phase brings the history to MAX - 2; the serial-phase
/// atomics (one instruction per round per writer) cross MAX.
#[test]
fn history_overflow_crossed_in_the_serial_phase() {
    // Only CTA 0's last store releases the waiter, after all atomics.
    let s = overflow_writers([MAX - 2, 0], [3, 1], [SENTINEL, 0]);
    overflow_is_incomplete_everywhere(&s);
}

