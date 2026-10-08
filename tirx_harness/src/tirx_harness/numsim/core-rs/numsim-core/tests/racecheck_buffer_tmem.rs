//! W2-19 (deltas T14): buffer-form TMEM accesses (warp actor,
//! `Proxy::Tcgen`, synchronous in the engine) are ordered by hb.
use numsim_core::racecheck::observer::{RaceObserver, RacecheckConfig};
use numsim_core::report::Status;
use numsim_core::sched::{self, Backend, RunStatus};
use numsim_core::testutil::scenarios;

#[test]
fn buffer_form_tmem_scenarios_are_race_free() {
    let mut seen = 0;
    for s in scenarios::all().into_iter().chain(scenarios::special()) {
        if !matches!(s.name, "implicit_tmem" | "tmem_subword") {
            continue;
        }
        seen += 1;
        let mut obs = RaceObserver::new(RacecheckConfig::default());
        let o = sched::run_with_config(&s.module, &s.inputs, &mut obs, &Backend::Interp, &s.config).unwrap();
        assert_eq!(o.status, RunStatus::Completed, "{}", s.name);
        let report = obs.finish();
        let errors: Vec<_> = report.findings.iter().filter(|f| f.status == Status::Error).collect();
        assert!(errors.is_empty(), "{}: {errors:#?}", s.name);
    }
    assert_eq!(seen, 2);
}
