//! `SyncTable::step_all` guard (W6, from W13's fp16_bf16_gemm profile, where
//! `step_all` was 16% of engine time). The rows replay the batch shapes the
//! engine sends for a cp.async GEMM mainloop: per-lane `AsyncGroup`
//! resources, one batch of 32 commands per warp instruction.
//!
//! * `sync/step_all/cp_async_32lanes`: per stage, `Issue`, `Commit`, both
//!   milestones (through `apply_completion`) and `Wait{n:0}`, each as a
//!   32-lane batch.
//! * `sync/step_all/blocked_wait_32lanes`: a 32-lane `Wait` whose last lane
//!   is not done, so the batch blocks and rolls back (spin retries).
//!
//! End-to-end guard: `interp_hot` `corpus_numsim/fp16_bf16_gemm`.
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use numsim_core::observe::WarpId;
use numsim_core::sync::async_group::{Cmd, Domain, Milestone};
use numsim_core::sync::{Completion, ResourceId, ResourceInit, Step, SyncCmd, SyncTable};

const LANES: u8 = 32;
const STAGES: u64 = 64;

fn lane(l: u8) -> ResourceId {
    ResourceId::AsyncGroup { warp: WarpId(0), lane: l, domain: Domain::CpAsync }
}

fn batch(cmd: Cmd) -> Vec<(ResourceId, SyncCmd)> {
    (0..LANES).map(|l| (lane(l), SyncCmd::AsyncGroup(cmd))).collect()
}

fn complete(t: &mut SyncTable, l: u8, ordinal: u64) {
    for milestone in [Milestone::ReadsDone, Milestone::FullyDone] {
        let r = t.apply_completion(Completion::GroupMilestone { res: lane(l), ordinal, milestone }).expect("milestone");
        assert!(matches!(r, Step::Done(_)));
    }
}

fn mainloop() -> SyncTable {
    let mut t = SyncTable::new(ResourceInit::default());
    let (issue, commit, wait) = (batch(Cmd::Issue), batch(Cmd::Commit), batch(Cmd::Wait { n: 0, read: false }));
    for stage in 0..STAGES {
        assert!(matches!(t.step_all(&issue), Ok(Step::Done(_))));
        assert!(matches!(t.step_all(&commit), Ok(Step::Done(_))));
        for l in 0..LANES {
            complete(&mut t, l, stage);
        }
        assert!(matches!(t.step_all(&wait), Ok(Step::Done(_))));
    }
    t
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("sync/step_all");
    g.throughput(Throughput::Elements(STAGES * 3));
    g.bench_function("cp_async_32lanes", |b| b.iter(mainloop));
    // Every lane has one committed group; all but the last lane completed it.
    let mut t = SyncTable::new(ResourceInit::default());
    assert!(matches!(t.step_all(&batch(Cmd::Issue)), Ok(Step::Done(_))));
    assert!(matches!(t.step_all(&batch(Cmd::Commit)), Ok(Step::Done(_))));
    for l in 0..LANES - 1 {
        complete(&mut t, l, 0);
    }
    let wait = batch(Cmd::Wait { n: 0, read: false });
    g.throughput(Throughput::Elements(1));
    g.bench_function("blocked_wait_32lanes", |b| {
        b.iter(|| assert!(matches!(t.step_all(&wait), Ok(Step::Blocked(_)))))
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
