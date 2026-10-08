//! W9-public-API engine bugs (CONTRACT_REQUESTS.md "W9-public-API"):
//! discard alignment, st.bulk size, isspacep.shared::cta rank, generic
//! shared window bits of `mapa.u64`.

use numsim_core::interp::ExecErrorKind;
use numsim_core::observe::RecordingObserver;
use numsim_core::sched::{self, Backend, RunOutcome, RunStatus};
use numsim_core::testutil::scenarios::{self, Scenario};

fn run(s: &Scenario) -> RunOutcome {
    let mut obs = RecordingObserver::new();
    sched::run_with_config(&s.module, &s.inputs, &mut obs, &Backend::Interp, &s.config).expect("run starts")
}

fn error(o: &RunOutcome) -> (ExecErrorKind, String) {
    match &o.status {
        RunStatus::Error(e) => (e.kind.clone(), e.message.clone()),
        other => panic!("expected an error, got {other:?}"),
    }
}

fn u32s(o: &RunOutcome, name: &str) -> Vec<u32> {
    o.outputs.buffers[name].0.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}

#[test]
fn discard_requires_a_128_byte_aligned_line() {
    assert_eq!(run(&scenarios::discard_at(0)).status, RunStatus::Completed);
    let (kind, msg) = error(&run(&scenarios::discard_at(1)));
    assert_eq!(kind, ExecErrorKind::Misaligned);
    assert!(msg.contains("discard requires a 128-byte aligned address"), "{msg}");
}

#[test]
fn st_bulk_size_is_a_multiple_of_8_up_to_16_mib() {
    assert_eq!(run(&scenarios::st_bulk_size(16)).status, RunStatus::Completed);
    for size in [1u32, 24 << 20] {
        let (_, msg) = error(&run(&scenarios::st_bulk_size(size)));
        assert!(msg.contains(&format!("st.bulk byte count {size} must be a multiple of 8 with maximum 16777216 on lane 0")), "{msg}");
    }
}

#[test]
fn isspacep_shared_cta_needs_the_own_rank_and_mapa_uses_the_hardware_window() {
    let o = run(&scenarios::mapa_isspacep());
    assert_eq!(o.status, RunStatus::Completed);
    let out = u32s(&o, "out");
    for cta in 0..2u32 {
        let r = &out[(cta * 8) as usize..(cta * 8 + 8) as usize];
        // Own rank: .shared::cta 1, .shared::cluster 1; peer: 0, 1.
        assert_eq!([r[0], r[1], r[4], r[5]], [1, 1, 0, 1], "cta {cta}: {r:?}");
        // Generic shared window base 0xFFFE_0000_0000, rank in bits 24.
        assert_eq!(r[2], 0x0000_fffe, "cta {cta}");
        assert_eq!(r[6], 0x0000_fffe, "cta {cta}");
        assert_eq!(r[3] >> 24, cta, "cta {cta}");
        assert_eq!(r[7] >> 24, 1 - cta, "cta {cta}");
    }
}
