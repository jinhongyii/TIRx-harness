//! Property tests over random command sequences.
//!
//! Every protocol is checked for these properties:
//!
//! * `step` is total and deterministic.
//! * An `Err` leaves the state unchanged.
//! * Every reachable state satisfies the module's `check_invariants`.
//! * Protocol-specific monotonicity and completion facts hold.
//! * Wherever a module has a `Policy`, Strict refines Numeric: as long as
//!   Strict accepts every command, Numeric returns the same outcomes and
//!   reaches the same state.

use numsim_sync_ref::{
    async_group, cluster, mbarrier, named, setmaxnreg, tcgen, Policy, Protocol, FULL_MASK,
};
use proptest::prelude::*;

fn checked_step<P: Protocol>(state: &mut P::State, cmd: P::Cmd) -> Result<P::Outcome, P::Error> {
    let before = state.clone();
    let mut again = state.clone();
    let result = P::step(state, cmd.clone());
    assert_eq!(
        result,
        P::step(&mut again, cmd),
        "step is not deterministic"
    );
    assert_eq!(*state, again, "step is not deterministic");
    if result.is_err() {
        assert_eq!(*state, before, "Err mutated the state");
    }
    result
}

// ---------------------------------------------------------------- mbarrier

#[derive(Clone, Debug)]
enum MbarTest {
    Raw(mbarrier::Cmd),
    /// Land the `idx`-th outstanding token with transaction bytes.
    LandTx {
        idx: usize,
        bytes: u64,
    },
    /// Land the `idx`-th outstanding token as a deferred arrive-on.
    LandArrive {
        idx: usize,
    },
}

fn mbar_cmd() -> impl Strategy<Value = MbarTest> {
    use mbarrier::Cmd;
    let raw = prop_oneof![
        2 => (1u64..4, any::<bool>()).prop_map(|(count, layout_v1)| Cmd::Init { count, layout_v1 }),
        1 => Just(0u64).prop_map(|count| Cmd::Init { count, layout_v1: false }),
        1 => Just(Cmd::Inval),
        8 => (0u64..3, prop::option::of(0u64..4), prop::bool::weighted(0.1), prop::bool::weighted(0.1))
            .prop_map(|(count, tx, drop, no_complete)| Cmd::Arrive {
                count,
                tx: tx.map(|t| t * 16),
                drop,
                no_complete,
            }),
        3 => (0u64..4).prop_map(|b| Cmd::ExpectTx { bytes: b * 16 }),
        2 => (0u64..2).prop_map(|count| Cmd::IncPending { count }),
        4 => Just(Cmd::Issue),
        1 => (0u64..4, 0u64..4).prop_map(|(gen, b)| Cmd::CompleteTx { gen, bytes: b * 16 }),
        6 => (0u64..2).prop_map(|parity| Cmd::TestParity { parity }),
        4 => (0u64..2).prop_map(|parity| Cmd::WaitParity { parity }),
        1 => Just(2u64).prop_map(|parity| Cmd::TestParity { parity }),
        2 => (0u64..5).prop_map(|gen| Cmd::TestState { gen }),
    ];
    prop_oneof![
        8 => raw.prop_map(MbarTest::Raw),
        3 => (0usize..3, 0u64..4).prop_map(|(idx, b)| MbarTest::LandTx { idx, bytes: b * 16 }),
        2 => (0usize..3).prop_map(|idx| MbarTest::LandArrive { idx }),
    ]
}

fn resolve_mbar(s: &mbarrier::State, t: &MbarTest) -> mbarrier::Cmd {
    let token = |idx: usize| {
        let gens: Vec<u64> = s
            .outstanding
            .iter()
            .flat_map(|(&g, &n)| std::iter::repeat_n(g, n as usize))
            .collect();
        if gens.is_empty() {
            s.gen
        } else {
            gens[idx % gens.len()]
        }
    };
    match *t {
        MbarTest::Raw(cmd) => cmd,
        MbarTest::LandTx { idx, bytes } => mbarrier::Cmd::CompleteTx {
            gen: token(idx),
            bytes,
        },
        MbarTest::LandArrive { idx } => mbarrier::Cmd::DeferredArrive {
            gen: token(idx),
            count: 1,
        },
    }
}

fn mbar_check_transition(
    before: &mbarrier::State,
    cmd: mbarrier::Cmd,
    result: &Result<mbarrier::Outcome, mbarrier::Error>,
    after: &mbarrier::State,
) {
    use mbarrier::{Cmd, Outcome};
    mbarrier::check_invariants(after).unwrap_or_else(|e| panic!("{e}: {after:?}"));
    let Ok(outcome) = result else { return };
    let reset = matches!(cmd, Cmd::Init { .. } | Cmd::Inval);
    if !reset {
        assert!(after.gen >= before.gen, "phase went backwards");
        assert!(after.gen <= before.gen + 1, "phase skipped a generation");
        assert!(
            after.last_completed >= before.last_completed,
            "completion history regressed"
        );
    }
    let completed = match *outcome {
        Outcome::Arrived { completed, .. } => completed,
        Outcome::Landed { completed } => completed.is_some(),
        _ => false,
    };
    if completed {
        // A completion needs a pending phase: the barrier was not already
        // complete, or a roll-over to a fresh phase happened first.
        assert!(after.complete);
        assert!(!before.complete || after.gen == before.gen + 1);
        assert_eq!(
            after.tx_completed, after.tx_expected,
            "tx unbalanced at flip"
        );
        assert_eq!(after.arrived, after.required());
    } else if !reset {
        assert!(
            after.last_completed == before.last_completed,
            "phase completed without reporting it"
        );
    }
    match (cmd, outcome) {
        (Cmd::TestParity { parity } | Cmd::WaitParity { parity }, Outcome::Ready { gen }) => {
            assert_eq!(parity, before.completed_parity());
            assert_eq!(*gen, before.last_completed);
        }
        (Cmd::TestParity { parity }, Outcome::NotReady)
        | (Cmd::WaitParity { parity }, Outcome::Blocked) => {
            assert_ne!(parity, before.completed_parity());
        }
        (Cmd::Issue, Outcome::Issued { gen }) => {
            assert_eq!(*gen, before.gen + u64::from(before.complete));
        }
        _ => {}
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2048))]

    #[test]
    fn mbarrier_properties(cmds in prop::collection::vec(mbar_cmd(), 0..60)) {
        let mut numeric = mbarrier::State::new(Policy::Numeric);
        let mut strict = mbarrier::State::new(Policy::Strict);
        let mut strict_alive = true;
        // Bytes landed per generation must equal the tx-count the phase
        // completed with (no byte lost or double-counted across buffering).
        let mut landed = std::collections::BTreeMap::<u64, u64>::new();
        for t in &cmds {
            let cmd = resolve_mbar(&numeric, t);
            let before = numeric.clone();
            let n = checked_step::<mbarrier::Mbarrier>(&mut numeric, cmd);
            mbar_check_transition(&before, cmd, &n, &numeric);
            if n.is_ok() {
                match cmd {
                    mbarrier::Cmd::Init { .. } | mbarrier::Cmd::Inval => landed.clear(),
                    mbarrier::Cmd::CompleteTx { gen, bytes } => *landed.entry(gen).or_default() += bytes,
                    _ => {}
                }
                if numeric.complete && !(before.complete && before.gen == numeric.gen) {
                    prop_assert_eq!(numeric.tx_completed, landed.get(&numeric.gen).copied().unwrap_or(0));
                }
            }
            if strict_alive {
                let strict_before = strict.clone();
                let s = checked_step::<mbarrier::Mbarrier>(&mut strict, cmd);
                mbar_check_transition(&strict_before, cmd, &s, &strict);
                match s {
                    Ok(o) => {
                        prop_assert_eq!(Ok(o), n, "strict accepted what numeric did not");
                        let mut projected = strict.clone();
                        projected.policy = Policy::Numeric;
                        prop_assert_eq!(&projected, &numeric);
                    }
                    Err(_) => strict_alive = false,
                }
            }
        }
    }
}

#[test]
fn mbarrier_tma_pipeline_round_trip() {
    use mbarrier::{Cmd, Outcome};
    for policy in [Policy::Numeric, Policy::Strict] {
        let mut s = mbarrier::State::new(policy);
        mbarrier::step(
            &mut s,
            Cmd::Init {
                count: 1,
                layout_v1: false,
            },
        )
        .unwrap();
        // Parity 1 is vacuously complete right after init.
        assert_eq!(
            mbarrier::step(&mut s, Cmd::TestParity { parity: 1 }),
            Ok(Outcome::Ready { gen: None })
        );
        for round in 0..4u64 {
            let parity = round & 1;
            let arrived = mbarrier::step(
                &mut s,
                Cmd::Arrive {
                    count: 1,
                    tx: Some(64),
                    drop: false,
                    no_complete: false,
                },
            )
            .unwrap();
            assert_eq!(
                arrived,
                Outcome::Arrived {
                    gen: round,
                    pending_before: 1,
                    completed: false
                }
            );
            let Outcome::Issued { gen } = mbarrier::step(&mut s, Cmd::Issue).unwrap() else {
                panic!()
            };
            assert_eq!(
                mbarrier::step(&mut s, Cmd::WaitParity { parity }),
                Ok(Outcome::Blocked)
            );
            assert_eq!(
                mbarrier::step(&mut s, Cmd::CompleteTx { gen, bytes: 64 }),
                Ok(Outcome::Landed {
                    completed: Some(round)
                })
            );
            assert_eq!(
                mbarrier::step(&mut s, Cmd::WaitParity { parity }),
                Ok(Outcome::Ready { gen: Some(round) })
            );
        }
    }
}

#[test]
fn mbarrier_parked_waiter_consumes_at_completion() {
    // Strict: a wait blocked before completion consumes the phase when it
    // completes, so the producer may reuse the barrier before the waiter
    // is rescheduled (strict_mbarrier.rs:1656-1695).
    use mbarrier::{Cmd, Outcome};
    let arrive = Cmd::Arrive {
        count: 1,
        tx: None,
        drop: false,
        no_complete: false,
    };
    let mut s = mbarrier::State::new(Policy::Strict);
    mbarrier::step(
        &mut s,
        Cmd::Init {
            count: 1,
            layout_v1: false,
        },
    )
    .unwrap();
    assert_eq!(
        mbarrier::step(&mut s, Cmd::WaitParity { parity: 0 }),
        Ok(Outcome::Blocked)
    );
    mbarrier::step(&mut s, arrive).unwrap();
    assert!(s.consumed);
    assert_eq!(
        mbarrier::step(&mut s, arrive),
        Ok(Outcome::Arrived {
            gen: 1,
            pending_before: 1,
            completed: true
        })
    );
}

#[test]
fn mbarrier_strict_rejects_phase_reuse_before_wait() {
    use mbarrier::{Cmd, Error, Op};
    let arrive = Cmd::Arrive {
        count: 1,
        tx: None,
        drop: false,
        no_complete: false,
    };
    let mut strict = mbarrier::State::new(Policy::Strict);
    let mut numeric = mbarrier::State::new(Policy::Numeric);
    for s in [&mut strict, &mut numeric] {
        mbarrier::step(
            s,
            Cmd::Init {
                count: 1,
                layout_v1: false,
            },
        )
        .unwrap();
        mbarrier::step(s, arrive).unwrap();
    }
    assert_eq!(
        mbarrier::step(&mut strict, arrive),
        Err(Error::ReuseBeforeConsumption {
            op: Op::Arrive,
            gen: 0
        })
    );
    assert!(mbarrier::step(&mut numeric, arrive).is_ok());
}

#[test]
fn mbarrier_early_bytes_then_expectation() {
    // complete_tx may land before expect_tx while arrivals are incomplete; the
    // phase flips only when both arrivals and bytes balance.
    use mbarrier::{Cmd, Outcome};
    let mut s = mbarrier::State::new(Policy::Strict);
    mbarrier::step(
        &mut s,
        Cmd::Init {
            count: 1,
            layout_v1: false,
        },
    )
    .unwrap();
    let Outcome::Issued { gen } = mbarrier::step(&mut s, Cmd::Issue).unwrap() else {
        panic!()
    };
    assert_eq!(
        mbarrier::step(&mut s, Cmd::CompleteTx { gen, bytes: 32 }),
        Ok(Outcome::Landed { completed: None })
    );
    assert_eq!(
        mbarrier::step(
            &mut s,
            Cmd::Arrive {
                count: 1,
                tx: Some(16),
                drop: false,
                no_complete: false
            }
        ),
        Err(mbarrier::Error::TxOverDelivery {
            expected: 16,
            completed: 32
        })
    );
    assert_eq!(
        mbarrier::step(
            &mut s,
            Cmd::Arrive {
                count: 1,
                tx: Some(32),
                drop: false,
                no_complete: false
            }
        ),
        Ok(Outcome::Arrived {
            gen: 0,
            pending_before: 1,
            completed: true
        })
    );
}

#[test]
fn mbarrier_late_deferred_arrival_is_rejected() {
    // Legacy engine lands this arrival on the next generation in release
    // builds (hardware_barriers.rs:2523-2524); both policies reject it here.
    use mbarrier::{Cmd, Error, Outcome};
    for policy in [Policy::Numeric, Policy::Strict] {
        let mut s = mbarrier::State::new(policy);
        mbarrier::step(
            &mut s,
            Cmd::Init {
                count: 1,
                layout_v1: false,
            },
        )
        .unwrap();
        let Outcome::Issued { gen } = mbarrier::step(&mut s, Cmd::Issue).unwrap() else {
            panic!()
        };
        mbarrier::step(
            &mut s,
            Cmd::Arrive {
                count: 1,
                tx: None,
                drop: false,
                no_complete: false,
            },
        )
        .unwrap();
        assert_eq!(
            mbarrier::step(&mut s, Cmd::DeferredArrive { gen, count: 1 }),
            Err(Error::CompletionAfterComplete { gen: 0 })
        );
    }
}

// ---------------------------------------------------------------- named

fn named_contribution() -> impl Strategy<Value = named::Contribution> {
    (
        0u32..4,
        prop::sample::select(vec![FULL_MASK, FULL_MASK, 0x0000_ffff, 0xffff_0000, 1, 0]),
        prop::sample::select(vec![32u64, 64, 64, 96, 128, 48]),
        any::<bool>(),
        prop::option::weighted(0.15, prop::sample::select(vec![FULL_MASK, 1u32])),
        0u32..2,
    )
        .prop_map(
            |(warp, mask, count, aligned, elect_entry, site)| named::Contribution {
                warp,
                mask,
                count,
                aligned,
                elect_entry,
                site,
            },
        )
}

fn named_cmd() -> impl Strategy<Value = named::Cmd> {
    prop_oneof![
        3 => named_contribution().prop_map(named::Cmd::Arrive),
        4 => named_contribution().prop_map(named::Cmd::Sync),
        3 => (0u64..4).prop_map(|gen| named::Cmd::Resume { gen }),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2048))]

    #[test]
    fn named_properties(cmds in prop::collection::vec(named_cmd(), 0..60)) {
        let mut numeric = named::State::new(Policy::Numeric);
        let mut strict = named::State::new(Policy::Strict);
        let mut strict_alive = true;
        for &cmd in &cmds {
            let before = numeric.clone();
            let n = checked_step::<named::Named>(&mut numeric, cmd);
            named::check_invariants(&numeric).unwrap();
            prop_assert!(numeric.gen >= before.gen && numeric.gen <= before.gen + 1);
            if let Ok(named::Outcome::Ready { gen }) = n {
                prop_assert!(gen < numeric.gen || numeric.complete);
            }
            if strict_alive {
                let s = checked_step::<named::Named>(&mut strict, cmd);
                named::check_invariants(&strict).unwrap();
                match s {
                    Ok(o) => {
                        prop_assert_eq!(Ok(o), n);
                        prop_assert_eq!(strict.gen, numeric.gen);
                        prop_assert_eq!(strict.arrived, numeric.arrived);
                        prop_assert_eq!(&strict.lanes, &numeric.lanes);
                    }
                    Err(_) => strict_alive = false,
                }
            }
        }
    }
}

#[test]
fn named_producer_consumer_arrive_then_sync() {
    // 32-thread producer arrives, 128-thread consumers sync: count 160.
    use named::{Cmd, Contribution, Outcome};
    let c = |warp, count| Contribution {
        warp,
        mask: FULL_MASK,
        count,
        aligned: true,
        elect_entry: None,
        site: 0,
    };
    let mut s = named::State::new(Policy::Strict);
    for w in 0..4 {
        assert_eq!(
            named::step(&mut s, Cmd::Sync(c(w, 160))),
            Ok(Outcome::Registered { gen: 0 })
        );
    }
    assert_eq!(
        named::step(&mut s, Cmd::Resume { gen: 0 }),
        Ok(Outcome::Blocked)
    );
    assert_eq!(
        named::step(&mut s, Cmd::Arrive(c(4, 160))),
        Ok(Outcome::Arrived {
            gen: 0,
            completed: true
        })
    );
    assert_eq!(
        named::step(&mut s, Cmd::Resume { gen: 0 }),
        Ok(Outcome::Ready { gen: 0 })
    );
    assert_eq!(named::quiescent(&s), Ok(()));
}

// ---------------------------------------------------------------- cluster

fn cluster_cmd() -> impl Strategy<Value = cluster::Cmd> {
    let mask = prop::sample::select(vec![
        FULL_MASK,
        FULL_MASK,
        FULL_MASK,
        0x0000_ffff,
        0xffff_0000,
        0,
    ]);
    prop_oneof![
        (0u32..4, mask.clone(), any::<bool>()).prop_map(|(warp, mask, aligned)| {
            cluster::Cmd::Arrive {
                warp,
                mask,
                aligned,
            }
        }),
        (0u32..4, mask, any::<bool>()).prop_map(|(warp, mask, aligned)| cluster::Cmd::Wait {
            warp,
            mask,
            aligned
        }),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2048))]

    #[test]
    fn cluster_properties(participants in 1u32..4, cmds in prop::collection::vec(cluster_cmd(), 0..60)) {
        let mut numeric = cluster::State::new(Policy::Numeric, participants);
        let mut strict = cluster::State::new(Policy::Strict, participants);
        let mut strict_alive = true;
        for &cmd in &cmds {
            let before = numeric.clone();
            let n = checked_step::<cluster::Cluster>(&mut numeric, cmd);
            cluster::check_invariants(&numeric).unwrap();
            prop_assert!(numeric.gen >= before.gen && numeric.gen <= before.gen + 1);
            if let Ok(cluster::Outcome::Ready { gen }) = n {
                prop_assert!(gen < numeric.gen, "wait released before its generation completed");
            }
            if strict_alive {
                let s = checked_step::<cluster::Cluster>(&mut strict, cmd);
                match s {
                    Ok(o) => {
                        prop_assert_eq!(Ok(o), n);
                        let mut projected = strict.clone();
                        projected.policy = Policy::Numeric;
                        prop_assert_eq!(&projected, &numeric);
                    }
                    Err(_) => strict_alive = false,
                }
            }
        }
    }
}

// ---------------------------------------------------------------- async groups

#[derive(Clone, Debug)]
enum AgTest {
    Raw(async_group::Cmd),
    Fire { idx: usize, full: bool },
}

fn ag_cmd() -> impl Strategy<Value = AgTest> {
    use async_group::Cmd;
    prop_oneof![
        4 => Just(AgTest::Raw(Cmd::Issue)),
        3 => Just(AgTest::Raw(Cmd::Commit)),
        2 => Just(AgTest::Raw(Cmd::ArriveOn)),
        3 => (0u64..3, any::<bool>()).prop_map(|(n, read)| AgTest::Raw(Cmd::Wait { n, read })),
        1 => Just(AgTest::Raw(Cmd::Exit)),
        5 => (0usize..4, any::<bool>()).prop_map(|(idx, full)| AgTest::Fire { idx, full }),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2048))]

    #[test]
    fn async_group_properties(bulk in any::<bool>(), cmds in prop::collection::vec(ag_cmd(), 0..60)) {
        use async_group::{Cmd, Milestone, Outcome};
        let domain = if bulk { async_group::Domain::Bulk } else { async_group::Domain::CpAsync };
        let mut s = async_group::State::new(domain);
        let mut released = 0u32;
        let mut attached = 0u32;
        for t in &cmds {
            let cmd = match *t {
                AgTest::Raw(cmd) => cmd,
                AgTest::Fire { idx, full } => Cmd::Complete {
                    ordinal: s.groups.get(idx).map_or(99, |g| g.ordinal),
                    milestone: if full { Milestone::FullyDone } else { Milestone::ReadsDone },
                },
            };
            let before = s.clone();
            let r = checked_step::<async_group::AsyncGroup>(&mut s, cmd);
            async_group::check_invariants(&s).unwrap();
            match (cmd, r) {
                (Cmd::Wait { n, read }, Ok(Outcome::Ready { .. })) => {
                    let prefix = async_group::wait_prefix_len(&before, n);
                    let need = if read { Milestone::ReadsDone } else { Milestone::FullyDone };
                    prop_assert!(before.groups.iter().take(prefix).all(|g| g.milestone >= need));
                }
                (Cmd::ArriveOn, Ok(Outcome::ArriveOn { group: Some(_) })) => attached += 1,
                (Cmd::Complete { .. }, Ok(Outcome::Completed { arrivals })) => released += arrivals,
                _ => {}
            }
            let held: u32 = s.groups.iter().map(|g| g.arrivals).sum();
            prop_assert_eq!(attached, released + held, "deferred arrive-on lost or duplicated");
        }
    }
}

// ---------------------------------------------------------------- tcgen

fn tcgen_cmd() -> impl Strategy<Value = tcgen::Cmd> {
    use tcgen::{Cmd, Who};
    let who = prop_oneof![
        Just(Who::One(0)),
        Just(Who::One(1)),
        Just(Who::Pair),
        Just(Who::One(7))
    ];
    let cols = prop::sample::select(vec![32u32, 64, 96, 128, 256, 512, 16, 1024]);
    prop_oneof![
        4 => (who.clone(), cols.clone(), prop::bool::weighted(0.2))
            .prop_map(|(who, columns, exclusive)| Cmd::Alloc { who, columns, exclusive }),
        3 => (who.clone(), prop::sample::select(vec![0u32, 32, 64, 128, 256]), cols)
            .prop_map(|(who, taddr, columns)| Cmd::Dealloc { who, taddr, columns }),
        1 => who.prop_map(|who| Cmd::Relinquish { who }),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2048))]

    #[test]
    fn tcgen_properties(cmds in prop::collection::vec(tcgen_cmd(), 0..40)) {
        let mut s = tcgen::State::default();
        for &cmd in &cmds {
            let before = s.clone();
            let r = checked_step::<tcgen::Tcgen>(&mut s, cmd);
            tcgen::check_invariants(&s).unwrap();
            if let (tcgen::Cmd::Alloc { who, .. }, Ok(_)) = (cmd, &r) {
                let idx: &[usize] = match who { tcgen::Who::Pair => &[0, 1], tcgen::Who::One(0) => &[0], _ => &[1] };
                for &i in idx {
                    prop_assert!(!before.ctas[i].relinquished, "alloc after relinquish");
                }
            }
            for (b, a) in before.ctas.iter().zip(&s.ctas) {
                prop_assert!(!b.relinquished || a.relinquished, "relinquish undone");
                prop_assert!(b.cta_group.is_none() || b.cta_group == a.cta_group, "cta_group changed");
            }
        }
    }
}

// ---------------------------------------------------------------- setmaxnreg

fn setmax_cmd() -> impl Strategy<Value = setmaxnreg::Cmd> {
    use setmaxnreg::Cmd;
    let count = prop::sample::select(vec![
        24u32, 40, 64, 96, 120, 128, 168, 232, 240, 256, 20, 100,
    ]);
    prop_oneof![
        1 => count.clone().prop_map(|count| Cmd::Configure { count }),
        6 => (0u32..3, any::<bool>(), count).prop_map(|(wg, inc, count)| Cmd::Set { wg, inc, count }),
        3 => (0u32..3).prop_map(|wg| Cmd::WarpgroupSync { wg }),
        2 => (0u32..3).prop_map(|wg| Cmd::Grant { wg }),
        2 => (0u32..3).prop_map(|wg| Cmd::Poll { wg }),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2048))]

    #[test]
    fn setmaxnreg_properties(warps in prop::sample::select(vec![4u32, 8, 12, 10]),
                             cmds in prop::collection::vec(setmax_cmd(), 0..60)) {
        let mut s = setmaxnreg::State::new(warps);
        for &cmd in &cmds {
            let r = checked_step::<setmaxnreg::Setmaxnreg>(&mut s, cmd);
            setmaxnreg::check_invariants(&s).unwrap();
            if let (setmaxnreg::Cmd::Grant { wg }, Ok(_)) = (cmd, r) {
                prop_assert!(s.pending[wg as usize].is_none());
            }
        }
    }
}
