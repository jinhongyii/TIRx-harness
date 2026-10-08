//! Stable kind strings: today's payload `kind` per protocol error
//! (`sync_check_python.rs:1979-2145`) and effect names (`sync_check.rs:109-135`).
//! Kinds marked NEW have no legacy counterpart (protocols the legacy strict
//! models did not cover, or finer reference-model errors).

use crate::report::FindingKind;
use crate::sync::{async_group, cluster, mbarrier, named, setmaxnreg, tcgen, SyncCmd, SyncError};

pub fn error_kind(e: &SyncError) -> &'static str {
    match e {
        SyncError::Mbarrier(e) => match e {
            mbarrier::Error::Uninitialized => "mbarrier_use_before_init",
            mbarrier::Error::InvalidCount { .. } => "mbarrier_invalid_expected_arrivals",
            mbarrier::Error::InvalidPhase { .. } => "mbarrier_invalid_phase",
            mbarrier::Error::InvalidStateToken { .. } => "mbarrier_invalid_state_token", // NEW
            mbarrier::Error::ReinitActive => "mbarrier_reinit_while_active",
            mbarrier::Error::ReinitWithoutInval => "mbarrier_reinit_without_inval",
            mbarrier::Error::ReinitBeforeConsumption { .. } => "mbarrier_reinit_before_consumption",
            mbarrier::Error::InvalWithOutstanding => "mbarrier_invalidate_with_outstanding_work",
            mbarrier::Error::ArrivalOverflow { .. } => "mbarrier_arrival_overflow",
            mbarrier::Error::DropUnderflow { .. } => "mbarrier_drop_underflow", // NEW
            mbarrier::Error::NoCompleteWouldComplete { .. } => "mbarrier_no_complete_violated", // NEW
            mbarrier::Error::TxCountOutOfRange { .. } => "mbarrier_counter_overflow",
            mbarrier::Error::TxOverDelivery { .. } => "mbarrier_transaction_over_delivery",
            mbarrier::Error::PendingOverflow { .. } => "mbarrier_pending_overflow", // NEW
            mbarrier::Error::ReuseBeforeConsumption { op: mbarrier::Op::ExpectTx, .. } => {
                "mbarrier_expect_tx_before_consumption"
            }
            mbarrier::Error::ReuseBeforeConsumption { .. } => "mbarrier_arrive_before_consumption",
            mbarrier::Error::UnknownToken { .. } => "mbarrier_unknown_completion_token",
            mbarrier::Error::StaleCompletion { .. } => "mbarrier_stale_completion",
            mbarrier::Error::CompletionAfterComplete { .. } => "mbarrier_completion_after_generation_complete",
            mbarrier::Error::FutureNotBufferable { .. } => "mbarrier_future_completion_not_bufferable",
            mbarrier::Error::IncompleteAtExit { .. } => "mbarrier_incomplete_at_exit", // NEW
            mbarrier::Error::TxUnderDelivered { .. } => "mbarrier_tx_underdelivered", // NEW
        },
        SyncError::Named(e) => match e {
            named::Error::InvalidCount { .. } => "named_barrier_invalid_expected_arrivals",
            named::Error::PartialWarp { .. } => "named_barrier_invalid_arrival_count",
            named::Error::ContractMismatch { .. } => "named_barrier_contract_mismatch",
            named::Error::Duplicate { .. } => "named_barrier_duplicate_contribution",
            named::Error::RedMixed { .. } => "named_barrier_red_mixed", // NEW
            named::Error::ArrivalOverflow { .. } => "named_barrier_arrival_overflow",
            named::Error::ResumeFuture { .. } => "named_barrier_resume_without_registration",
        },
        SyncError::Cluster(e) => match e {
            cluster::Error::UnexpectedParticipant { .. } => "cluster_barrier_unexpected_participant",
            cluster::Error::PartialWarp { .. } => "cluster_barrier_partial_warp",
            cluster::Error::EarlyArrival { .. } => "cluster_barrier_early_arrival",
            cluster::Error::WaitBeforeArrival { .. } => "cluster_barrier_wait_before_arrival",
            cluster::Error::DuplicateWait { .. } => "cluster_barrier_duplicate_wait",
        },
        SyncError::AsyncGroup(e) => match e {
            async_group::Error::InvalidForm => "async_group_invalid_form", // NEW
            async_group::Error::NotEnabled { .. } => "async_group_milestone_not_enabled", // NEW
            async_group::Error::PendingAtExit { .. } => "async_group_pending_at_exit", // NEW
        },
        SyncError::Tcgen(e) => match e {
            tcgen::Error::InvalidCtaGroup => "tcgen_invalid_cta_group", // NEW
            tcgen::Error::CtaGroupMismatch { .. } => "tcgen_cta_group_mismatch", // NEW
            tcgen::Error::InvalidColumns { .. } => "tcgen_invalid_columns", // NEW
            tcgen::Error::AllocAfterRelinquish => "tcgen_alloc_after_relinquish", // NEW
            tcgen::Error::AllocationSizeIncrease { .. } => "tcgen_allocation_size_increase", // NEW
            tcgen::Error::DeallocationMismatch { .. } => "tcgen_deallocation_mismatch", // NEW
            tcgen::Error::LiveAllocationsAtExit { .. } => "tcgen_live_allocations_at_exit", // NEW
            tcgen::Error::AllocWhileExclusive { .. } => "tcgen_alloc_while_exclusive", // NEW
        },
        SyncError::RegPool(e) => match e {
            setmaxnreg::Error::InvalidCount { .. } => "setmaxnreg_invalid_count", // NEW
            setmaxnreg::Error::ConfigureConflict => "setmaxnreg_configure_conflict", // NEW
            setmaxnreg::Error::IncompleteWarpgroup { .. } => "setmaxnreg_incomplete_warpgroup", // NEW
            setmaxnreg::Error::MissingWarpgroupSync { .. } => "setmaxnreg_missing_warpgroup_sync",
            setmaxnreg::Error::InvalidDirection { .. } => "setmaxnreg_invalid_direction", // NEW
            setmaxnreg::Error::WarpgroupPending { .. } => "setmaxnreg_warpgroup_pending", // NEW
            setmaxnreg::Error::GrantNotEnabled { .. } => "setmaxnreg_grant_not_enabled", // NEW
            setmaxnreg::Error::PendingAtExit { .. } => "setmaxnreg_pool_deadlock",
        },
        SyncError::WrongResource { .. } => "sync_wrong_resource", // NEW (engine bug)
    }
}

/// Today's `FixedSyncProtocolKind` Debug names (payload `protocol`).
pub fn protocol_name(e: &SyncError) -> &'static str {
    match e {
        SyncError::Mbarrier(_) => "Mbarrier",
        SyncError::Named(_) => "NamedBarrier",
        SyncError::Cluster(_) => "ClusterBarrier",
        SyncError::RegPool(_) => "Setmaxnreg",
        SyncError::Tcgen(_) => "TcgenLifecycle",
        SyncError::AsyncGroup(_) => "AsyncGroup", // NEW
        SyncError::WrongResource { .. } => "Internal",
    }
}

/// W3's per-protocol mapping (`SyncError::finding_kind`).
pub fn finding_kind(e: &SyncError) -> FindingKind {
    e.finding_kind()
}

pub fn protocol_finding_kind(protocol: &str) -> FindingKind {
    match protocol {
        "Mbarrier" => FindingKind::MbarrierMisuse,
        "NamedBarrier" | "ClusterBarrier" => FindingKind::BarrierMismatch,
        "Setmaxnreg" => FindingKind::RegPoolMisuse,
        "TcgenLifecycle" => FindingKind::TmemMisuse,
        "AsyncGroup" => FindingKind::AsyncGroupMisuse,
        _ => FindingKind::RuntimeError,
    }
}

/// `SyncCheckEffectKind::name()` of a protocol command.
pub fn effect_name(cmd: &SyncCmd) -> &'static str {
    match cmd {
        SyncCmd::Mbarrier(c) => match c {
            mbarrier::Cmd::Init { .. } => "mbarrier.init",
            mbarrier::Cmd::Inval => "mbarrier.inval",
            mbarrier::Cmd::Arrive { .. } => "mbarrier.arrive",
            mbarrier::Cmd::ExpectTx { .. } => "mbarrier.expect_tx",
            mbarrier::Cmd::IncPending { .. } => "cp.async.mbarrier.arrive",
            mbarrier::Cmd::Issue => "mbarrier.completion_issue",
            mbarrier::Cmd::CompleteTx { .. } | mbarrier::Cmd::DeferredArrive { .. } => "mbarrier.complete_tx",
            mbarrier::Cmd::TestParity { .. } | mbarrier::Cmd::WaitParity { .. } | mbarrier::Cmd::TestState { .. } => {
                "mbarrier.wait"
            }
        },
        SyncCmd::Named(c) => match c {
            named::Cmd::Arrive(_) => "bar.arrive.register",
            named::Cmd::Sync(_) | named::Cmd::Red(_) => "bar.sync.register",
            named::Cmd::Resume { .. } => "bar.sync.resume",
        },
        SyncCmd::Cluster(c) => match c {
            cluster::Cmd::Arrive { .. } => "barrier.cluster.arrive",
            cluster::Cmd::Wait { .. } => "barrier.cluster.wait.register",
            cluster::Cmd::Exit { .. } => "barrier.cluster.exit", // NEW
        },
        SyncCmd::AsyncGroup(_) => "cp.async.group", // NEW
        SyncCmd::Tcgen(c) => match c {
            tcgen::Cmd::Alloc { .. } => "tcgen05.alloc.register",
            tcgen::Cmd::Dealloc { .. } => "tcgen05.dealloc.register",
            tcgen::Cmd::Relinquish { .. } => "tcgen05.relinquish_alloc_permit.register",
        },
        SyncCmd::TcgenWork(_) => "tcgen05.commit.issue",
        SyncCmd::TcgenGroup(_) => "tcgen05.cta_group", // W3-3
        SyncCmd::RegPool(_) => "setmaxnreg.register",
    }
}

/// Does command `cmd` belong to the protocol that raised `e`?
pub fn same_protocol(cmd: &SyncCmd, e: &SyncError) -> bool {
    matches!(
        (cmd, e),
        (SyncCmd::Mbarrier(_), SyncError::Mbarrier(_))
            | (SyncCmd::Named(_), SyncError::Named(_))
            | (SyncCmd::Cluster(_), SyncError::Cluster(_))
            | (SyncCmd::AsyncGroup(_), SyncError::AsyncGroup(_))
            | (SyncCmd::Tcgen(_), SyncError::Tcgen(_))
            | (SyncCmd::RegPool(_), SyncError::RegPool(_))
    )
}
