//! Synthetic synchronization logs for tests and benchmarks.

use crate::event::{LogBuilder, SyncEvent, SyncOp};

/// One producer warp (warp 0) and `warps - 1` consumer warps sharing a
/// `stages`-deep ring of full/empty mbarriers, `iterations` trips.
///
/// * full[s]  = mbarrier `s`, expected 1 arrival (+ `tma_bytes` if TMA);
/// * empty[s] = mbarrier `stages + s`, expected one arrival per consumer;
/// * named barrier 0 publishes the inits (`cta_sync`).
#[derive(Clone, Copy, Debug)]
pub struct PipelineShape {
    pub warps: u32,
    pub stages: u32,
    pub iterations: u32,
    /// When non-zero the producer uses `arrive.expect_tx` + an async completion.
    pub tma_bytes: u32,
}

pub fn pipeline(shape: PipelineShape) -> Vec<SyncEvent> {
    let PipelineShape {
        warps,
        stages,
        iterations,
        tma_bytes,
    } = shape;
    assert!(warps >= 2 && stages >= 1);
    let consumers = warps - 1;
    let full = |s: u32| s;
    let empty = |s: u32| stages + s;
    let cta_threads = warps * 32;
    let mut log = LogBuilder::new();
    for s in 0..stages {
        log.push(0, 1, SyncOp::MbarInit { bar: full(s), expected: 1 });
        log.push(0, 2, SyncOp::MbarInit { bar: empty(s), expected: consumers });
    }
    for warp in 0..warps {
        log.push(warp, 3, SyncOp::NamedSync { bar: 0, expected: cta_threads, count: 32 });
    }
    for i in 0..iterations {
        let s = i % stages;
        let round = i / stages;
        if round >= 1 {
            log.push(0, 10, SyncOp::MbarWait { bar: empty(s), parity: ((round - 1) & 1) as u8 });
        }
        if tma_bytes > 0 {
            log.push(0, 11, SyncOp::MbarArrive { bar: full(s), count: 1, expect_tx: tma_bytes });
            log.push(0, 12, SyncOp::MbarTxIssue { bar: full(s), tx: tma_bytes });
        } else {
            log.push(0, 11, SyncOp::MbarArrive { bar: full(s), count: 1, expect_tx: 0 });
        }
    }
    for warp in 1..warps {
        for i in 0..iterations {
            let s = i % stages;
            let round = i / stages;
            log.push(warp, 20, SyncOp::MbarWait { bar: full(s), parity: (round & 1) as u8 });
            log.push(warp, 21, SyncOp::MbarArrive { bar: empty(s), count: 1, expect_tx: 0 });
        }
    }
    log.build()
}
