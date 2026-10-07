//! Differential tests: production `numsim_core::sync` against the executable
//! spec `numsim_sync_ref`.
//!
//! Random command sequences are fed to both implementations. After every
//! command these must be identical (compared through `Debug`, because the
//! types are mechanical copies with identical derives):
//!
//! * the result, `Ok(outcome)` or `Err(error)`;
//! * the full state;
//! * the exit checks (`quiescent`, `exit_lint`).
//!
//! The production `check_invariants` must also hold. mbarrier runs under both
//! policies.
//!
//! The `both!` macro evaluates one token tree in both crates' namespaces, so
//! the two command values are built from the same source text.

use numsim_core::sync as prod;
use numsim_sync_ref as spec;
use proptest::prelude::*;

const CASES: u32 = 4096;
const MAX_LEN: usize = 80;

macro_rules! both {
    ($module:ident, $($e:tt)*) => {{
        let r = { use spec::$module::*; $($e)* };
        let c = { use prod::$module::*; $($e)* };
        (r, c)
    }};
}

fn dbg<T: std::fmt::Debug>(v: &T) -> String {
    format!("{v:?}")
}

/// Compare one step.
macro_rules! diff_step {
    ($module:ident, $rs:expr, $cs:expr, $rc:expr, $cc:expr, $ctx:expr) => {{
        let r = spec::$module::step(&mut $rs, $rc);
        let c = prod::$module::step(&mut $cs, $cc);
        prop_assert_eq!(dbg(&r), dbg(&c), "result differs at {}", $ctx);
        prop_assert_eq!(dbg(&$rs), dbg(&$cs), "state differs at {}", $ctx);
        prop_assert_eq!(
            dbg(&spec::$module::quiescent(&$rs)),
            dbg(&prod::$module::quiescent(&$cs)),
            "quiescent differs at {}",
            $ctx
        );
        if let Err(e) = prod::$module::check_invariants(&$cs) {
            prop_assert!(false, "production invariant `{}` broken at {}", e, $ctx);
        }
    }};
}

// ---------------------------------------------------------------- mbarrier

#[derive(Clone, Debug)]
enum MbarOp {
    Init(u64, bool),
    Inval,
    Arrive(u64, Option<u64>, bool, bool),
    ExpectTx(u64),
    IncPending(u64),
    Issue,
    /// Land the `idx`-th outstanding token (or a raw generation).
    Tx(usize, u64),
    Deferred(usize, u64),
    RawTx(u64, u64),
    Test(u64),
    Wait(u64),
    State(u64),
}

fn mbar_op() -> impl Strategy<Value = MbarOp> {
    prop_oneof![
        2 => (prop::sample::select(vec![0u64, 1, 2, 3, 511, 512]), prop::bool::weighted(0.2)).prop_map(|(c, v)| MbarOp::Init(c, v)),
        1 => Just(MbarOp::Inval),
        9 => (0u64..4, prop::option::of(prop::sample::select(vec![0u64, 16, 64, (1 << 20) - 16, 1 << 20])), prop::bool::weighted(0.1), prop::bool::weighted(0.1))
            .prop_map(|(c, t, d, n)| MbarOp::Arrive(c, t, d, n)),
        3 => prop::sample::select(vec![0u64, 16, 48, (1 << 20) - 1, 1 << 21]).prop_map(MbarOp::ExpectTx),
        2 => prop::sample::select(vec![0u64, 1, 2, 600]).prop_map(MbarOp::IncPending),
        4 => Just(MbarOp::Issue),
        4 => (0usize..4, prop::sample::select(vec![0u64, 16, 32, 64, 1 << 20])).prop_map(|(i, b)| MbarOp::Tx(i, b)),
        2 => (0usize..4, 0u64..3).prop_map(|(i, c)| MbarOp::Deferred(i, c)),
        1 => (0u64..5, 0u64..64).prop_map(|(g, b)| MbarOp::RawTx(g, b)),
        6 => (0u64..3).prop_map(MbarOp::Test),
        4 => (0u64..2).prop_map(MbarOp::Wait),
        2 => (0u64..5).prop_map(MbarOp::State),
    ]
}

fn token(s: &spec::mbarrier::State, idx: usize) -> u64 {
    let gens: Vec<u64> = s.outstanding.iter().flat_map(|(&g, &n)| std::iter::repeat_n(g, n as usize)).collect();
    if gens.is_empty() {
        s.gen
    } else {
        gens[idx % gens.len()]
    }
}

fn mbar_cmds(s: &spec::mbarrier::State, op: &MbarOp) -> (spec::mbarrier::Cmd, prod::mbarrier::Cmd) {
    match *op {
        MbarOp::Init(count, layout_v1) => both!(mbarrier, Cmd::Init { count, layout_v1 }),
        MbarOp::Inval => both!(mbarrier, Cmd::Inval),
        MbarOp::Arrive(count, tx, drop, no_complete) => both!(mbarrier, Cmd::Arrive { count, tx, drop, no_complete }),
        MbarOp::ExpectTx(bytes) => both!(mbarrier, Cmd::ExpectTx { bytes }),
        MbarOp::IncPending(count) => both!(mbarrier, Cmd::IncPending { count }),
        MbarOp::Issue => both!(mbarrier, Cmd::Issue),
        MbarOp::Tx(i, bytes) => {
            let gen = token(s, i);
            both!(mbarrier, Cmd::CompleteTx { gen, bytes })
        }
        MbarOp::Deferred(i, count) => {
            let gen = token(s, i);
            both!(mbarrier, Cmd::DeferredArrive { gen, count })
        }
        MbarOp::RawTx(gen, bytes) => both!(mbarrier, Cmd::CompleteTx { gen, bytes }),
        MbarOp::Test(parity) => both!(mbarrier, Cmd::TestParity { parity }),
        MbarOp::Wait(parity) => both!(mbarrier, Cmd::WaitParity { parity }),
        MbarOp::State(gen) => both!(mbarrier, Cmd::TestState { gen }),
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(CASES))]

    #[test]
    fn mbarrier_matches_reference(strict in any::<bool>(), ops in prop::collection::vec(mbar_op(), 0..MAX_LEN)) {
        let (mut rs, mut cs) = if strict {
            (spec::mbarrier::State::new(spec::Policy::Strict), prod::mbarrier::State::new(prod::Policy::Strict))
        } else {
            (spec::mbarrier::State::new(spec::Policy::Numeric), prod::mbarrier::State::new(prod::Policy::Numeric))
        };
        for (i, op) in ops.iter().enumerate() {
            let (rc, cc) = mbar_cmds(&rs, op);
            diff_step!(mbarrier, rs, cs, rc, cc, format!("#{i} {op:?}"));
        }
    }
}

// ---------------------------------------------------------------- named

fn named_op() -> impl Strategy<Value = (u8, u32, u32, u32, u64, u64)> {
    (
        0u8..4,
        0u32..5,
        prop::sample::select(vec![u32::MAX, u32::MAX, u32::MAX, 0xffff, 1, 0]),
        prop::sample::select(vec![u32::MAX, u32::MAX, 0xffff]),
        prop::sample::select(vec![32u64, 64, 64, 96, 128, 160, 0, 48]),
        0u64..4,
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(CASES))]

    #[test]
    fn named_matches_reference(ops in prop::collection::vec(named_op(), 0..MAX_LEN)) {
        let mut rs = spec::named::State::default();
        let mut cs = prod::named::State::default();
        for (i, &(kind, warp, mask, live, count, gen)) in ops.iter().enumerate() {
            let (rc, cc) = match kind {
                0 => both!(named, Cmd::Arrive(Contribution { warp, mask, live, count })),
                1 => both!(named, Cmd::Sync(Contribution { warp, mask, live, count })),
                2 => both!(named, Cmd::Red(Contribution { warp, mask, live, count })),
                _ => both!(named, Cmd::Resume { gen }),
            };
            diff_step!(named, rs, cs, rc, cc, format!("#{i}"));
            prop_assert_eq!(dbg(&spec::named::exit_lint(&rs)), dbg(&prod::named::exit_lint(&cs)));
        }
    }
}

// ---------------------------------------------------------------- cluster

proptest! {
    #![proptest_config(ProptestConfig::with_cases(CASES))]

    #[test]
    fn cluster_matches_reference(
        participants in 0u32..5,
        ops in prop::collection::vec((0u8..3, 0u32..5, prop::sample::select(vec![u32::MAX, u32::MAX, 0xffff, 0xffff_0000, 0]), any::<bool>()), 0..MAX_LEN),
    ) {
        let mut rs = spec::cluster::State::new(participants);
        let mut cs = prod::cluster::State::new(participants);
        for (i, &(kind, warp, mask, aligned)) in ops.iter().enumerate() {
            let (rc, cc) = match kind {
                0 => both!(cluster, Cmd::Arrive { warp, mask, aligned }),
                1 => both!(cluster, Cmd::Wait { warp, mask, aligned }),
                _ => both!(cluster, Cmd::Exit { warp, lanes: mask }),
            };
            diff_step!(cluster, rs, cs, rc, cc, format!("#{i}"));
        }
    }
}

// ---------------------------------------------------------------- async groups

proptest! {
    #![proptest_config(ProptestConfig::with_cases(CASES))]

    #[test]
    fn async_group_matches_reference(
        bulk in any::<bool>(),
        ops in prop::collection::vec((0u8..7, 0usize..5, 0u64..4, any::<bool>()), 0..MAX_LEN),
    ) {
        let (mut rs, mut cs) = if bulk {
            (spec::async_group::State::new(spec::async_group::Domain::Bulk), prod::async_group::State::new(prod::async_group::Domain::Bulk))
        } else {
            (spec::async_group::State::new(spec::async_group::Domain::CpAsync), prod::async_group::State::new(prod::async_group::Domain::CpAsync))
        };
        for (i, &(kind, idx, n, flag)) in ops.iter().enumerate() {
            let ordinal = rs.groups.get(idx).map_or(n + 90, |g| g.ordinal);
            let (rc, cc) = match kind {
                0 | 1 => both!(async_group, Cmd::Issue),
                2 => both!(async_group, Cmd::Commit),
                3 => both!(async_group, Cmd::ArriveOn),
                4 => both!(async_group, Cmd::Wait { n, read: flag }),
                5 => both!(async_group, Cmd::Exit),
                _ => {
                    let (rm, cm) = match n {
                        0 => both!(async_group, Milestone::Pending),
                        1 | 2 => both!(async_group, Milestone::ReadsDone),
                        _ => both!(async_group, Milestone::FullyDone),
                    };
                    (spec::async_group::Cmd::Complete { ordinal, milestone: rm },
                     prod::async_group::Cmd::Complete { ordinal, milestone: cm })
                }
            };
            diff_step!(async_group, rs, cs, rc, cc, format!("#{i}"));
            prop_assert_eq!(dbg(&spec::async_group::exit_lint(&rs)), dbg(&prod::async_group::exit_lint(&cs)));
            for k in 0..4 {
                prop_assert_eq!(spec::async_group::wait_prefix_len(&rs, k), prod::async_group::wait_prefix_len(&cs, k));
            }
        }
    }
}

// ---------------------------------------------------------------- tcgen

proptest! {
    #![proptest_config(ProptestConfig::with_cases(CASES))]

    #[test]
    fn tcgen_matches_reference(
        sm107 in any::<bool>(),
        ops in prop::collection::vec((
            0u8..3,
            0u8..4,
            prop::sample::select(vec![32u32, 64, 96, 128, 160, 256, 512, 576, 16, 1024]),
            prop::sample::select(vec![0u32, 32, 64, 128, 256]),
            prop::bool::weighted(0.25),
        ), 0..MAX_LEN),
    ) {
        let max = if sm107 { 576 } else { 512 };
        let mut rs = spec::tcgen::State::new(max);
        let mut cs = prod::tcgen::State::new(max);
        for (i, &(kind, w, columns, taddr, exclusive)) in ops.iter().enumerate() {
            let (rw, cw) = match w {
                0 => both!(tcgen, Who::One(0)),
                1 => both!(tcgen, Who::One(1)),
                2 => both!(tcgen, Who::Pair),
                _ => both!(tcgen, Who::One(5)),
            };
            let (rc, cc) = match kind {
                0 => (spec::tcgen::Cmd::Alloc { who: rw, columns, exclusive }, prod::tcgen::Cmd::Alloc { who: cw, columns, exclusive }),
                1 => (spec::tcgen::Cmd::Dealloc { who: rw, taddr, columns, exclusive }, prod::tcgen::Cmd::Dealloc { who: cw, taddr, columns, exclusive }),
                _ => (spec::tcgen::Cmd::Relinquish { who: rw }, prod::tcgen::Cmd::Relinquish { who: cw }),
            };
            diff_step!(tcgen, rs, cs, rc, cc, format!("#{i}"));
        }
    }

    #[test]
    fn tcgen_kernel_and_work_match_reference(ops in prop::collection::vec((0u8..7, 0u8..4), 0..MAX_LEN)) {
        let mut rk = spec::tcgen::KernelState::default();
        let mut ck = prod::tcgen::KernelState::default();
        let mut rw = spec::tcgen::WorkState::default();
        let mut cw = prod::tcgen::WorkState::default();
        for &(kind, g) in &ops {
            if kind == 6 {
                prop_assert_eq!(dbg(&spec::tcgen::use_cta_group(&mut rk, g)), dbg(&prod::tcgen::use_cta_group(&mut ck, g)));
                prop_assert_eq!(dbg(&rk), dbg(&ck));
                continue;
            }
            let (rc, cc) = match kind {
                0 => both!(tcgen, WorkCmd::Issue),
                1 => both!(tcgen, WorkCmd::Load),
                2 => both!(tcgen, WorkCmd::Store),
                3 => both!(tcgen, WorkCmd::Commit),
                4 => both!(tcgen, WorkCmd::WaitLd),
                _ => both!(tcgen, WorkCmd::WaitSt),
            };
            prop_assert_eq!(dbg(&spec::tcgen::work_step(&mut rw, rc)), dbg(&prod::tcgen::work_step(&mut cw, cc)));
            prop_assert_eq!(dbg(&rw), dbg(&cw));
        }
    }
}

// ---------------------------------------------------------------- setmaxnreg

proptest! {
    #![proptest_config(ProptestConfig::with_cases(CASES))]

    #[test]
    fn setmaxnreg_matches_reference(
        warps in prop::sample::select(vec![0u32, 4, 8, 10, 12, 16]),
        ops in prop::collection::vec((
            0u8..6,
            0u32..4,
            any::<bool>(),
            prop::sample::select(vec![24u32, 40, 64, 96, 120, 128, 168, 232, 240, 256, 20, 100, 512]),
        ), 0..MAX_LEN),
    ) {
        let mut rs = spec::setmaxnreg::State::new(warps);
        let mut cs = prod::setmaxnreg::State::new(warps);
        for (i, &(kind, wg, inc, count)) in ops.iter().enumerate() {
            let (rc, cc) = match kind {
                0 => both!(setmaxnreg, Cmd::Configure { count }),
                1 | 2 => both!(setmaxnreg, Cmd::Set { wg, inc, count }),
                3 => both!(setmaxnreg, Cmd::WarpgroupSync { wg }),
                4 => both!(setmaxnreg, Cmd::Grant { wg }),
                _ => both!(setmaxnreg, Cmd::Poll { wg }),
            };
            diff_step!(setmaxnreg, rs, cs, rc, cc, format!("#{i}"));
            prop_assert_eq!(spec::setmaxnreg::enabled_grants(&rs), prod::setmaxnreg::enabled_grants(&cs));
            prop_assert_eq!(spec::setmaxnreg::initial_total(&rs), prod::setmaxnreg::initial_total(&cs));
        }
    }
}

// ---------------------------------------------------------------- SyncTable

/// `SyncTable::enabled` is exactly "applying now is not premature": for an
/// mbarrier completion it is false only when the reference would reject the
/// landing as `FutureNotBufferable`; for a milestone or grant it is true
/// exactly when the reference applies it without error.
mod table {
    use super::*;
    use prod::{Completion, Resource, ResourceId, ResourceInit, Step, SyncCmd, SyncTable};
    use numsim_core::arena::AllocId;
    use numsim_core::observe::CtaId;

    fn mbar_id() -> ResourceId {
        ResourceId::Mbarrier { cta: CtaId(0), alloc: AllocId(0), offset: 0 }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(CASES))]

        #[test]
        fn table_mbarrier_steps_and_enabled(strict in any::<bool>(), ops in prop::collection::vec(mbar_op(), 0..MAX_LEN), probe in 0u64..6) {
            let policy = if strict { prod::Policy::Strict } else { prod::Policy::Numeric };
            let rpolicy = if strict { spec::Policy::Strict } else { spec::Policy::Numeric };
            let mut table = SyncTable::new(ResourceInit { policy, ..ResourceInit::default() });
            let mut rs = spec::mbarrier::State::new(rpolicy);
            let id = mbar_id();
            for op in &ops {
                let (rc, cc) = mbar_cmds(&rs, op);
                let r = spec::mbarrier::step(&mut rs, rc);
                let before = table.get(id).cloned();
                let c = table.step(id, SyncCmd::Mbarrier(cc));
                match (&r, &c) {
                    (Ok(spec::mbarrier::Outcome::Blocked), Ok(Step::Blocked(b))) => prop_assert_eq!(*b, id),
                    (Ok(ro), Ok(Step::Done(prod::Outcome::Mbarrier(co)))) => prop_assert_eq!(dbg(ro), dbg(co)),
                    (Err(re), Err(prod::SyncError::Mbarrier(ce))) => {
                        prop_assert_eq!(dbg(re), dbg(ce));
                        prop_assert_eq!(dbg(&before), dbg(&table.get(id).cloned()), "table changed on error");
                    }
                    _ => prop_assert!(false, "{:?} vs {:?}", r, c),
                }
                if let Some(Resource::Mbarrier(cs)) = table.get(id) {
                    prop_assert_eq!(dbg(&rs), dbg(cs));
                }
            }
            // Probe `enabled` with an arbitrary transaction completion.
            let completion = Completion::MbarTx { res: id, gen: probe, bytes: 16 };
            let mut probe_state = rs.clone();
            *probe_state.outstanding.entry(probe).or_default() += 1;
            let landing = spec::mbarrier::step(&mut probe_state, spec::mbarrier::Cmd::CompleteTx { gen: probe, bytes: 16 });
            if table.get(id).is_some() {
                let premature = matches!(landing, Err(spec::mbarrier::Error::FutureNotBufferable { .. }));
                prop_assert_eq!(table.enabled(&completion), !premature, "{:?}", landing);
            }
        }

        #[test]
        fn table_milestone_and_grant_enabled(
            ops in prop::collection::vec((0u8..5, 0usize..4, any::<bool>()), 0..MAX_LEN),
            warps in prop::sample::select(vec![4u32, 8]),
        ) {
            let mut table = SyncTable::new(ResourceInit { warps_per_cta: warps, ..ResourceInit::default() });
            let ag = ResourceId::AsyncGroup { warp: numsim_core::observe::WarpId(0), lane: 0, domain: prod::async_group::Domain::Bulk };
            let pool = ResourceId::RegPool { cta: CtaId(0) };
            for &(kind, idx, flag) in &ops {
                let cmd = match kind {
                    0 => SyncCmd::AsyncGroup(prod::async_group::Cmd::Issue),
                    1 => SyncCmd::AsyncGroup(prod::async_group::Cmd::Commit),
                    2 => SyncCmd::RegPool(prod::setmaxnreg::Cmd::Set { wg: (idx % 2) as u32, inc: flag, count: if flag { 256 } else { 64 } }),
                    3 => SyncCmd::RegPool(prod::setmaxnreg::Cmd::WarpgroupSync { wg: (idx % 2) as u32 }),
                    _ => {
                        // Try every candidate completion; enabled must agree with apply.
                        let mut candidates = Vec::new();
                        if let Some(Resource::AsyncGroup(s)) = table.get(ag) {
                            for g in &s.groups {
                                for m in [prod::async_group::Milestone::ReadsDone, prod::async_group::Milestone::FullyDone] {
                                    candidates.push(Completion::GroupMilestone { res: ag, ordinal: g.ordinal, milestone: m });
                                }
                            }
                        }
                        for wg in 0..2 {
                            candidates.push(Completion::SetmaxGrant { res: pool, wg });
                        }
                        for c in candidates {
                            let mut trial = table.clone();
                            let applied = trial.apply_completion(c);
                            prop_assert_eq!(table.enabled(&c), applied.is_ok(), "{:?} -> {:?}", c, applied);
                        }
                        continue;
                    }
                };
                let id = if matches!(cmd, SyncCmd::AsyncGroup(_)) { ag } else { pool };
                let _ = table.step(id, cmd);
            }
        }
    }
}
