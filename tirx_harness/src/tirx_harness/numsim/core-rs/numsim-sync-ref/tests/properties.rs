//! Property tests over random command sequences.
//!
//! Every protocol is checked for these properties:
//!
//! * `step` is total and deterministic.
//! * An `Err` leaves the state unchanged.
//! * Every reachable state satisfies the module's `check_invariants`.
//! * Protocol-specific monotonicity and completion facts hold.
//! * mbarrier: Strict refines Numeric. As long as Strict accepts every
//!   command, Numeric returns the same outcomes and reaches the same state.

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
fn mbarrier_phase_reuse_before_wait() {
    // PTX §9.7.15.16.5.1: an arrive-on in the next phase needs a successful
    // wait first (every policy). Extending this to expect_tx is Strict only.
    use mbarrier::{Cmd, Error, Op};
    let arrive = Cmd::Arrive {
        count: 1,
        tx: None,
        drop: false,
        no_complete: false,
    };
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
        mbarrier::step(&mut s, arrive).unwrap();
        assert_eq!(
            mbarrier::step(&mut s, arrive),
            Err(Error::ReuseBeforeConsumption {
                op: Op::Arrive,
                gen: 0
            })
        );
        assert_eq!(
            mbarrier::step(&mut s, Cmd::IncPending { count: 1 }),
            Err(Error::ReuseBeforeConsumption {
                op: Op::IncPending,
                gen: 0
            })
        );
        let expect = mbarrier::step(&mut s, Cmd::ExpectTx { bytes: 16 });
        match policy {
            Policy::Strict => assert_eq!(
                expect,
                Err(Error::ReuseBeforeConsumption {
                    op: Op::ExpectTx,
                    gen: 0
                })
            ),
            Policy::Numeric => assert!(expect.is_ok()),
        }
    }
}

#[test]
fn mbarrier_reinit_and_limits() {
    use mbarrier::{Cmd, Error};
    for policy in [Policy::Numeric, Policy::Strict] {
        let mut s = mbarrier::State::new(policy);
        mbarrier::step(
            &mut s,
            Cmd::Init {
                count: 2,
                layout_v1: false,
            },
        )
        .unwrap();
        // PTX §9.7.15.16.12: init on a valid object is UB under every policy.
        assert_eq!(
            mbarrier::step(
                &mut s,
                Cmd::Init {
                    count: 2,
                    layout_v1: false
                }
            ),
            Err(Error::ReinitWithoutInval)
        );
        // PTX §9.7.15.16.17: drop to an expected count of zero is UB.
        assert_eq!(
            mbarrier::step(
                &mut s,
                Cmd::Arrive {
                    count: 2,
                    tx: None,
                    drop: true,
                    no_complete: false
                }
            ),
            Err(Error::DropUnderflow {
                expected: 2,
                count: 2
            })
        );
        // Signed tx-count: the state range is checked, not the u32 operand.
        assert_eq!(
            mbarrier::step(&mut s, Cmd::ExpectTx { bytes: 1 << 20 }),
            Err(Error::TxCountOutOfRange { tx_count: 1 << 20 })
        );
        mbarrier::step(
            &mut s,
            Cmd::ExpectTx {
                bytes: (1 << 20) - 1,
            },
        )
        .unwrap();
        assert_eq!(
            mbarrier::step(
                &mut s,
                Cmd::Arrive {
                    count: 2,
                    tx: None,
                    drop: false,
                    no_complete: true
                }
            ),
            Err(Error::NoCompleteWouldComplete {
                count: 2,
                pending: 2
            })
        );
        mbarrier::step(&mut s, Cmd::Inval).unwrap();
        assert_eq!(
            mbarrier::step(
                &mut s,
                Cmd::Init {
                    count: 512,
                    layout_v1: true
                }
            ),
            Err(Error::InvalidCount {
                count: 512,
                limit: 511
            })
        );
    }
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
        0u32..5,
        prop::sample::select(vec![FULL_MASK, FULL_MASK, FULL_MASK, 0x0000_ffff, 1, 0]),
        prop::sample::select(vec![FULL_MASK, FULL_MASK, FULL_MASK, 0x0000_ffff]),
        prop::sample::select(vec![32u64, 64, 64, 96, 128, 160, 48]),
        any::<bool>(),
    )
        .prop_map(|(warp, mask, live, count, aligned)| named::Contribution {
            warp,
            mask,
            live,
            count,
            aligned,
        })
}

fn named_cmd() -> impl Strategy<Value = named::Cmd> {
    prop_oneof![
        3 => named_contribution().prop_map(named::Cmd::Arrive),
        4 => named_contribution().prop_map(named::Cmd::Sync),
        1 => named_contribution().prop_map(named::Cmd::Red),
        3 => (0u64..4).prop_map(|gen| named::Cmd::Resume { gen }),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2048))]

    #[test]
    fn named_properties(cmds in prop::collection::vec(named_cmd(), 0..60)) {
        let mut s = named::State::default();
        for &cmd in &cmds {
            let before = s.clone();
            let r = checked_step::<named::Named>(&mut s, cmd);
            named::check_invariants(&s).unwrap();
            prop_assert!(s.gen >= before.gen && s.gen <= before.gen + 1);
            match (cmd, r) {
                (_, Ok(named::Outcome::Ready { gen })) => {
                    prop_assert!(gen < s.gen || s.complete);
                }
                (named::Cmd::Arrive(c) | named::Cmd::Sync(c) | named::Cmd::Red(c), Ok(_)) => {
                    prop_assert_eq!(c.mask, c.live, "partial warp accepted");
                    prop_assert!(c.count.is_multiple_of(32));
                }
                _ => {}
            }
        }
    }
}

#[test]
fn named_producer_consumer_arrive_then_sync() {
    // A 32-thread producer arrives and 128 consumer threads sync: count 160.
    use named::{Cmd, Contribution, Outcome};
    let c = |warp| Contribution {
        warp,
        mask: FULL_MASK,
        live: FULL_MASK,
        count: 160,
        aligned: true,
    };
    let mut s = named::State::default();
    for w in 0..4 {
        assert_eq!(
            named::step(&mut s, Cmd::Sync(c(w))),
            Ok(Outcome::Registered { gen: 0 })
        );
    }
    assert_eq!(
        named::step(&mut s, Cmd::Resume { gen: 0 }),
        Ok(Outcome::Blocked)
    );
    assert_eq!(
        named::step(&mut s, Cmd::Arrive(c(4))),
        Ok(Outcome::Arrived {
            gen: 0,
            completed: true
        })
    );
    assert_eq!(
        named::step(&mut s, Cmd::Resume { gen: 0 }),
        Ok(Outcome::Ready { gen: 0 })
    );
    assert_eq!(named::exit_lint(&s), None);
}

#[test]
fn named_elected_lane_and_red_mixing_are_errors() {
    use named::{Cmd, Contribution, Error};
    let mut s = named::State::default();
    let elected = Contribution {
        warp: 0,
        mask: 1,
        live: FULL_MASK,
        count: 64,
        aligned: false,
    };
    assert_eq!(
        named::step(&mut s, Cmd::Arrive(elected)),
        Err(Error::PartialWarp {
            mask: 1,
            live: FULL_MASK
        })
    );
    // Lanes that exited do not have to participate.
    let partial_live = Contribution {
        warp: 0,
        mask: 0xffff,
        live: 0xffff,
        count: 64,
        aligned: false,
    };
    named::step(&mut s, Cmd::Sync(partial_live)).unwrap();
    let red = Contribution {
        warp: 1,
        mask: FULL_MASK,
        live: FULL_MASK,
        count: 64,
        aligned: false,
    };
    assert_eq!(
        named::step(&mut s, Cmd::Red(red)),
        Err(Error::RedMixed { warp: 1 })
    );
    assert!(named::exit_lint(&s).is_some());
}

// ---------------------------------------------------------------- cluster

fn cluster_cmd() -> impl Strategy<Value = cluster::Cmd> {
    let mask = prop::sample::select(vec![FULL_MASK, FULL_MASK, FULL_MASK, 0x0000_ffff, 0]);
    prop_oneof![
        4 => (0u32..4, mask.clone(), any::<bool>())
            .prop_map(|(warp, mask, aligned)| cluster::Cmd::Arrive { warp, mask, aligned }),
        4 => (0u32..4, mask.clone(), any::<bool>())
            .prop_map(|(warp, mask, aligned)| cluster::Cmd::Wait { warp, mask, aligned }),
        1 => (0u32..4, prop::sample::select(vec![FULL_MASK, 0xffff_0000]))
            .prop_map(|(warp, lanes)| cluster::Cmd::Exit { warp, lanes }),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2048))]

    #[test]
    fn cluster_properties(participants in 1u32..4, cmds in prop::collection::vec(cluster_cmd(), 0..60)) {
        let mut s = cluster::State::new(participants);
        for &cmd in &cmds {
            let before = s.clone();
            let r = checked_step::<cluster::Cluster>(&mut s, cmd);
            cluster::check_invariants(&s).unwrap();
            prop_assert!(s.gen >= before.gen && s.gen <= before.gen + 1);
            match (cmd, r) {
                (_, Ok(cluster::Outcome::Ready { gen })) => {
                    prop_assert!(gen < s.gen, "wait released before its generation completed");
                }
                (cluster::Cmd::Arrive { warp, mask, .. } | cluster::Cmd::Wait { warp, mask, .. }, Ok(_)) => {
                    prop_assert_eq!(mask, before.live[warp as usize], "partial warp accepted");
                }
                _ => {}
            }
        }
    }
}

#[test]
fn cluster_exit_releases_waiters() {
    use cluster::{Cmd, Outcome};
    let mut s = cluster::State::new(2);
    let arrive = Cmd::Arrive {
        warp: 0,
        mask: FULL_MASK,
        aligned: true,
    };
    let wait = Cmd::Wait {
        warp: 0,
        mask: FULL_MASK,
        aligned: true,
    };
    assert!(matches!(
        cluster::step(&mut s, arrive),
        Ok(Outcome::Arrived {
            completed: false,
            ..
        })
    ));
    assert_eq!(cluster::step(&mut s, wait), Ok(Outcome::Blocked));
    assert_eq!(
        cluster::step(
            &mut s,
            Cmd::Exit {
                warp: 1,
                lanes: FULL_MASK
            }
        ),
        Ok(Outcome::Exited { completed: true })
    );
    assert_eq!(cluster::step(&mut s, wait), Ok(Outcome::Ready { gen: 0 }));
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
                (Cmd::Wait { n, read }, Ok(Outcome::Ready { acquired, .. })) => {
                    let need = if read { Milestone::ReadsDone } else { Milestone::FullyDone };
                    prop_assert_eq!(acquired, need, ".read must never acquire destinations");
                    let prefix = async_group::wait_prefix_len(&before, n);
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
    let cols = prop::sample::select(vec![32u32, 64, 96, 128, 256, 512, 576, 16, 1024]);
    prop_oneof![
        4 => (who.clone(), cols.clone(), prop::bool::weighted(0.2))
            .prop_map(|(who, columns, exclusive)| Cmd::Alloc { who, columns, exclusive }),
        3 => (who.clone(), prop::sample::select(vec![0u32, 32, 64, 128, 256]), cols, prop::bool::weighted(0.2))
            .prop_map(|(who, taddr, columns, exclusive)| Cmd::Dealloc { who, taddr, columns, exclusive }),
        1 => who.prop_map(|who| Cmd::Relinquish { who }),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2048))]

    #[test]
    fn tcgen_properties(sm107 in any::<bool>(), cmds in prop::collection::vec(tcgen_cmd(), 0..40)) {
        let mut s = tcgen::State::new(if sm107 { 576 } else { 512 });
        let mut kernel = tcgen::KernelState::default();
        for &cmd in &cmds {
            let who = match cmd {
                tcgen::Cmd::Alloc { who, .. } | tcgen::Cmd::Dealloc { who, .. } | tcgen::Cmd::Relinquish { who } => who,
            };
            if tcgen::use_cta_group(&mut kernel, who.group()).is_err() {
                continue;
            }
            let before = s.clone();
            let r = checked_step::<tcgen::Tcgen>(&mut s, cmd);
            tcgen::check_invariants(&s).unwrap();
            if let (tcgen::Cmd::Alloc { .. }, Ok(tcgen::Outcome::Blocked)) = (cmd, &r) {
                prop_assert_eq!(&before, &s, "blocked alloc mutated state");
            }
            if let (tcgen::Cmd::Alloc { who, .. }, Ok(tcgen::Outcome::Allocated { .. })) = (cmd, &r) {
                let idx: &[usize] = match who { tcgen::Who::Pair => &[0, 1], tcgen::Who::One(0) => &[0], _ => &[1] };
                for &i in idx {
                    prop_assert!(!before.ctas[i].relinquished, "alloc after relinquish");
                }
            }
            for (b, a) in before.ctas.iter().zip(&s.ctas) {
                prop_assert!(!b.relinquished || a.relinquished, "relinquish undone");
                prop_assert!(b.last_alloc_columns.is_none() || a.last_alloc_columns <= b.last_alloc_columns,
                    "allocation width increased");
            }
        }
    }
}

#[test]
fn tcgen_alloc_blocks_until_dealloc_and_group_is_kernel_wide() {
    use tcgen::{Cmd, Error, Outcome, Who};
    let mut s = tcgen::State::default();
    let alloc = |columns| Cmd::Alloc {
        who: Who::One(0),
        columns,
        exclusive: false,
    };
    assert_eq!(
        tcgen::step(&mut s, alloc(512)),
        Ok(Outcome::Allocated { base: 0 })
    );
    assert_eq!(tcgen::step(&mut s, alloc(256)), Ok(Outcome::Blocked));
    tcgen::step(
        &mut s,
        Cmd::Dealloc {
            who: Who::One(0),
            taddr: 0,
            columns: 512,
            exclusive: false,
        },
    )
    .unwrap();
    assert_eq!(
        tcgen::step(&mut s, alloc(256)),
        Ok(Outcome::Allocated { base: 0 })
    );
    assert_eq!(
        tcgen::step(&mut s, alloc(512)),
        Err(Error::AllocationSizeIncrease {
            previous: 256,
            requested: 512
        })
    );
    let mut kernel = tcgen::KernelState::default();
    tcgen::use_cta_group(&mut kernel, 1).unwrap();
    assert_eq!(
        tcgen::use_cta_group(&mut kernel, 2),
        Err(Error::CtaGroupMismatch {
            established: 1,
            requested: 2
        })
    );
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

/// V2C-14: warps 4-5 of a 6-warp CTA form no warpgroup. Their `setmaxnreg`
/// is `IncompleteWarpgroup` (PTX 9.7.21.5: UB unless every warp of the
/// warpgroup executes it), but crediting their aligned `bar.sync` is a no-op.
#[test]
fn setmaxnreg_trailing_partial_warpgroup() {
    use setmaxnreg::{step, Cmd, Error, Outcome, State};
    let mut s = State::new(6);
    assert_eq!(step(&mut s, Cmd::WarpgroupSync { wg: 1 }), Ok(Outcome::Done));
    assert_eq!(step(&mut s, Cmd::Set { wg: 1, inc: false, count: 64 }), Err(Error::IncompleteWarpgroup { wg: 1 }));
    assert_eq!(step(&mut s, Cmd::Set { wg: 0, inc: false, count: 64 }), Ok(Outcome::Applied { count: 64 }));
    assert_eq!(step(&mut s, Cmd::WarpgroupSync { wg: 2 }), Ok(Outcome::Done));
}

/// Partial-warp ruling (sync-isa-answers Q3/Q5): non-aligned pieces of one
/// warp gather into one full-mask arrival; `.aligned` stays immediate; the
/// missing lanes exiting or reaching another barrier id is `PartialWarp`.
#[test]
fn named_partial_warp_gathers() {
    use named::{gather, Contribution, Cmd, Error, Flavor, Gather, GatherCmd, GatherOutcome, Outcome, State};
    let exec = |id, mask, aligned| GatherCmd::Execute { id, flavor: Flavor::Sync, count: 64, mask, live: FULL_MASK, aligned };
    let mut g = Gather { warp: 0, pending: None };
    assert_eq!(gather(&mut g, exec(1, 0x0000_ffff, false)), Ok(GatherOutcome::Wait));
    assert_eq!(gather(&mut g, exec(1, 0xffff_0000, false)), Ok(GatherOutcome::Arrive { mask: FULL_MASK }));
    // The one arrival the barrier sees is a full-warp contribution.
    let mut s = State::default();
    let full = |warp| Contribution { warp, mask: FULL_MASK, live: FULL_MASK, count: 64, aligned: false };
    assert_eq!(named::step(&mut s, Cmd::Sync(full(0))), Ok(Outcome::Registered { gen: 0 }));
    assert_eq!(named::step(&mut s, Cmd::Sync(full(1))), Ok(Outcome::Ready { gen: 0 }));
    // `.aligned` partial warp: immediate error.
    let mut g = Gather { warp: 0, pending: None };
    assert_eq!(gather(&mut g, exec(1, 1, true)), Err(Error::PartialWarp { mask: 1, live: FULL_MASK }));
    // Missing lanes reach another id, or exit.
    assert_eq!(gather(&mut g, exec(1, 1, false)), Ok(GatherOutcome::Wait));
    assert_eq!(gather(&mut g.clone(), exec(2, 0xfffe, false)), Err(Error::PartialWarp { mask: 1, live: FULL_MASK }));
    assert_eq!(gather(&mut g, GatherCmd::Exit { live: 1 }), Err(Error::PartialWarp { mask: 1, live: 1 }));
    assert_eq!(gather(&mut Gather::default(), GatherCmd::Exit { live: 0 }), Ok(GatherOutcome::Idle));
}

fn gather_cmd() -> impl Strategy<Value = named::GatherCmd> {
    let mask = prop::sample::select(vec![FULL_MASK, 0x0000_ffff, 0xffff_0000, 0xff00_0000, 1, 0]);
    prop_oneof![
        8 => (0u32..3, 0u8..3, prop::sample::select(vec![32u64, 64]), mask.clone(),
              prop::sample::select(vec![FULL_MASK, FULL_MASK, 0x0000_ffff]), any::<bool>())
            .prop_map(|(id, f, count, mask, live, aligned)| named::GatherCmd::Execute {
                id,
                flavor: [named::Flavor::Arrive, named::Flavor::Sync, named::Flavor::Red][f as usize],
                count,
                mask,
                live,
                aligned,
            }),
        1 => mask.prop_map(|live| named::GatherCmd::Exit { live }),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2048))]

    #[test]
    fn named_gather_properties(cmds in prop::collection::vec(gather_cmd(), 0..40)) {
        let mut g = named::Gather::default();
        for &cmd in &cmds {
            let before = g.clone();
            match named::gather(&mut g, cmd) {
                Err(_) => prop_assert_eq!(&g, &before, "errors leave the gather unchanged"),
                Ok(named::GatherOutcome::Arrive { mask }) => {
                    let named::GatherCmd::Execute { live, .. } = cmd else { unreachable!() };
                    prop_assert_eq!(mask, live);
                    prop_assert!(g.pending.is_none());
                }
                Ok(named::GatherOutcome::Wait) => {
                    let named::GatherCmd::Execute { aligned, live, .. } = cmd else { unreachable!() };
                    prop_assert!(!aligned);
                    let p = g.pending.expect("pending");
                    prop_assert!(p.lanes != live && p.lanes & !live == 0 && p.lanes != 0);
                }
                Ok(named::GatherOutcome::Idle) => prop_assert!(before.pending.is_none()),
            }
        }
    }
}

/// tcgen05 exclusive ruling (sync-isa-answers, "Exclusive allocation"): a
/// CTA holding a live `.exclusive` allocation may not allocate again ("This
/// must be the only live allocation, until it is deallocated"); the peer
/// CTA's allocation waits for the exclusive one.
#[test]
fn tcgen_alloc_while_exclusive_is_an_error() {
    use tcgen::{Cmd, Error, Outcome, Who};
    let mut s = tcgen::State::new(576);
    let alloc = |cta, columns, exclusive| Cmd::Alloc { who: Who::One(cta), columns, exclusive };
    assert_eq!(tcgen::step(&mut s, alloc(0, 96, true)), Ok(Outcome::Allocated { base: 0 }));
    assert_eq!(tcgen::step(&mut s, alloc(0, 32, false)), Err(Error::AllocWhileExclusive { cta: 0 }));
    assert_eq!(tcgen::step(&mut s, alloc(0, 32, true)), Err(Error::AllocWhileExclusive { cta: 0 }));
    tcgen::step(&mut s, Cmd::Dealloc { who: Who::One(0), taddr: 0, columns: 96, exclusive: true }).unwrap();
    assert_eq!(tcgen::step(&mut s, alloc(0, 32, false)), Ok(Outcome::Allocated { base: 0 }));
}

/// sync-semantics §2.9: a quiescent launch whose open phase has every
/// arrival but fewer bytes than `expect_tx`, with nothing in flight, is
/// `TxUnderDelivered` (legacy "transactions=48/52").
#[test]
fn mbarrier_stuck_reports_tx_under_delivery() {
    use mbarrier::{step, stuck, Cmd, Error, State};
    let mut s = State::new(Policy::Numeric);
    step(&mut s, Cmd::Init { count: 1, layout_v1: false }).unwrap();
    assert_eq!(stuck(&s), None, "arrival missing: ordinary deadlock");
    step(&mut s, Cmd::Arrive { count: 1, tx: Some(52), drop: false, no_complete: false }).unwrap();
    step(&mut s, Cmd::Issue).unwrap();
    assert_eq!(stuck(&s), None, "a transaction is still in flight");
    step(&mut s, Cmd::CompleteTx { gen: 0, bytes: 48 }).unwrap();
    assert_eq!(stuck(&s), Some(Error::TxUnderDelivered { gen: 0, expected: 52, completed: 48 }));
}
