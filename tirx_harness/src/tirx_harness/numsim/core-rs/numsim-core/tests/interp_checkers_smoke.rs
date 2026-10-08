//! The event streams the interpreter emits for every scenario are consumable
//! by racecheck (online) and synccheck (offline): no panics, and the
//! race-free scenarios report no race.

use numsim_core::observe::RecordingObserver;
use numsim_core::racecheck::observer::{RaceObserver, RacecheckConfig};
use numsim_core::report::{FindingKind, Status};
use numsim_core::sched::{self, Backend, RunStatus};
use numsim_core::synccheck::{self, SynccheckConfig};
use numsim_core::testutil::scenarios;

#[test]
fn checkers_consume_interp_events() {
    let only = std::env::var("SCEN").ok();
    for s in scenarios::all() {
        if only.as_deref().is_some_and(|o| o != s.name) {
            continue;
        }
        eprintln!("== {}", s.name);
        let mut obs = (RaceObserver::new(RacecheckConfig::default()), RecordingObserver::new());
        let o = sched::run_with_config(&s.module, &s.inputs, &mut obs, &Backend::Interp, &s.config).unwrap();
        let (race, log) = obs;
        let report = race.finish();
        if o.status == RunStatus::Completed {
            let races: Vec<_> = report
                .findings
                .iter()
                .filter(|f| f.status == Status::Error && matches!(f.kind, FindingKind::DataRace | FindingKind::ProxyRace | FindingKind::AsyncRace))
                .collect();
            assert!(races.is_empty(), "{}: unexpected races {races:#?}", s.name);
        }
        eprintln!("   racecheck done");
        // Small budgets: this checks consumability, not exhaustive coverage.
        let cfg = SynccheckConfig { state_budget: 20_000, transition_budget: 200_000, ..SynccheckConfig::default() };
        let _ = synccheck::check(&log, &cfg);
        eprintln!("   synccheck done");
    }
}
