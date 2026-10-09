//! W5-15: `alias_stale_read` names each access by its own pointer operand.
use numsim_core::arena::{AllocId, Space};
use numsim_core::racecheck::checker::{AdvisoryKind, Checker, FindingKind};
use numsim_core::racecheck::input::*;
use numsim_core::site::SiteId;
use std::sync::Arc;

fn acc(epoch: u32, kind: AccessKind, site: u32, operand: u8) -> Event {
    Event::Access(Access {
        seq: epoch as u64,
        who: Who::Lane { warp: 0, lane: 0, epoch },
        alloc: AllocId(1),
        range: 0..16,
        kind,
        order: MemOrder::Weak,
        scope: None,
        atomic: false,
        returns_value: false,
        proxy: Proxy::Generic,
        domain: Some(Domain::SharedCta),
        site: SiteId(site),
        operand,
    })
}

fn run(per_operand: bool, read_operand: u8) -> usize {
    let mut c = Checker::new(Topology { warps_per_cta: 1, ctas_per_cluster: 1, num_ctas: 1 });
    c.event(Event::Sync(SyncEvent::AllocBegin { alloc: AllocId(1), space: Space::Shared, size: 64, cta: 0 }));
    // site 1 writes `image`; site 2 (a copy: operand 0 = `dst`, operand 1 =
    // `image`) reads the same bytes through operand 1.
    Arc::make_mut(&mut c.site_buffer).insert(SiteId(1), Arc::from("image"));
    Arc::make_mut(&mut c.site_buffer).insert(SiteId(2), Arc::from("dst"));
    if per_operand {
        Arc::make_mut(&mut c.operand_buffer).insert((SiteId(1), 0), Arc::from("image"));
        Arc::make_mut(&mut c.operand_buffer).insert((SiteId(2), 0), Arc::from("dst"));
        Arc::make_mut(&mut c.operand_buffer).insert((SiteId(2), 1), Arc::from("image"));
    }
    c.event(acc(1, AccessKind::Write, 1, 0));
    c.event(acc(2, AccessKind::Read, 2, read_operand));
    let r = c.finish();
    r.findings.iter().filter(|f| matches!(f.kind, FindingKind::Advisory { kind: AdvisoryKind::AliasStaleRead })).count()
}

#[test]
fn operand_names_the_access() {
    // Per-operand names: the read is through `image`, no advisory.
    assert_eq!(run(true, 1), 0);
    // Without them, operand 1 has no known name (never operand 0's): none.
    assert_eq!(run(false, 1), 0);
    // Positive control: a read through operand 0 (`dst`) of `image`'s bytes.
    assert_eq!(run(true, 0), 1);
    assert_eq!(run(false, 0), 1);
}
