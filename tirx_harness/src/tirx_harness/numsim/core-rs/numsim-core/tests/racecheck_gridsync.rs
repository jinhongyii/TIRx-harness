//! mega_moe workspace reset after a release/acquire grid rendezvous.
#[path = "racecheck_common/mod.rs"]
mod common;
use common::*;

fn grid_reset(rendezvous: bool) -> Report {
    // CTA0 = warps 0,1; CTA1 = warps 2,3.
    let mut k = K::new(2, 1, 2);
    k.declare(GMEM2, 0..4); // the polled ring counter F
    k.declare(GMEM, 0..4); // the grid-sync counter C
    k.a(3, 0, st(MemOrder::Release, Scope::Gpu), GMEM2, 0..4); // F history 1
    k.wait_until(2, 0, GMEM2, 0..4, Scope::Gpu, 0b10, 1); // CTA1 polls F
    if rendezvous {
        k.bar(1, &[2, 3]); // CTA-local join (dispatch + load-A)
        k.a(3, 0, atom(MemOrder::Release, Scope::Gpu), GMEM, 0..4); // C history 1
        k.a(0, 0, atom(MemOrder::Release, Scope::Gpu), GMEM, 0..4); // C history 2
        k.wait_until(0, 0, GMEM, 0..4, Scope::Gpu, 0b100, 2); // CTA0 sees the last add
    }
    k.st(0, 0, GMEM2, 0..4); // reset F
    k.run()
}

#[test]
fn reset_after_grid_rendezvous_is_ordered() {
    let r = grid_reset(true);
    assert!(r.findings.is_empty(), "{r:#?}");
    assert!(!grid_reset(false).findings.is_empty());
}
