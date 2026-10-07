//! Findings and verdicts shared by Racecheck, Synccheck and NumSim runtime
//! errors. The Python report layer renders these; it never inspects engine
//! internals (plan 2.7). Serialized as JSON (serde) across the pyo3 boundary.
//!
//! Status rules (numsim/AGENTS.md): `Error` requires proof; `Review` is one
//! precise advisory; `Incomplete` means coverage cannot support a claim and
//! is never success.

use crate::arena::{AllocId, ByteSpan, Space};
use crate::observe::Actor;
use crate::site::SiteId;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Status {
    Error,
    Review,
    Incomplete,
}

/// Finding kinds. Checkers may add variants through the coordinator; the
/// snake_case serde name is the stable public identifier.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FindingKind {
    // racecheck
    DataRace,
    ProxyRace,
    AsyncRace,
    UninitRead,
    // runtime / numsim
    OutOfBounds,
    Misaligned,
    Trap,
    RuntimeError,
    // synccheck
    Deadlock,
    /// Two complete interleavings reached different final protocol states (synccheck Phase B).
    NonConfluent,
    /// Memory reused while an unfinished async op may still access it (W5-3).
    AsyncLifetime,
    /// TMEM lifetime advisory (W5-3).
    TmemLifetimeReview,
    /// Release/acquire pair whose scopes do not mutually include each other (W5-3).
    ScopeMismatch,
    /// Strong load observed an unordered morally-strong write on a word not declared for `wait_until` (W5-3).
    UndeclaredProtocolWord,
    /// Async-proxy writers from several CTAs ordered only by base causality (W5-3).
    CrossCtaAsyncOrder,
    BarrierMismatch,
    MbarrierMisuse,
    AsyncGroupMisuse,
    TmemMisuse,
    RegPoolMisuse,
    UnwaitedAsync,
    // coverage
    BudgetExhausted,
    Unsupported,
    /// Escape hatch for a new kind during development; must be promoted
    /// to a real variant before merge.
    Other(String),
}

/// One piece of evidence: an actor's event at a site, optionally on bytes.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Evidence {
    /// Role in the finding (`"first_access"`, `"second_access"`, `"waiter"`, ...).
    pub role: String,
    /// Index of the kernel (within the `Module`) whose `sites` table `site` indexes (W8-2).
    pub kernel: u32,
    pub site: SiteId,
    pub actor: Option<Actor>,
    /// Buffer name and allocation-relative bytes, when relevant.
    pub buffer: Option<String>,
    /// Memory space and allocation of `bytes` (W8-2; legacy payloads always carry `space`).
    pub space: Option<Space>,
    pub alloc: Option<AllocId>,
    pub bytes: Option<ByteSpan>,
    pub detail: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Finding {
    pub kind: FindingKind,
    pub status: Status,
    pub message: String,
    /// Structured, tool-specific attributes (legacy payload keys such as
    /// `reason`, `cause`, `access_pair`, `ordering_*`), serialized verbatim
    /// into the payload (W6-1 item 2, contract review item 8).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub attrs: BTreeMap<String, serde_json::Value>,
    /// Every causal source site (deduplicated, sorted).
    pub sites: Vec<SiteId>,
    pub evidence: Vec<Evidence>,
}

/// Overall verdict of one check.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Clean,
    Review,
    Incomplete,
    Error,
}

impl Verdict {
    /// Error > Incomplete > Review > Clean.
    pub fn of(findings: &[Finding]) -> Verdict {
        let mut v = Verdict::Clean;
        for f in findings {
            let fv = match f.status {
                Status::Error => Verdict::Error,
                Status::Incomplete => Verdict::Incomplete,
                Status::Review => Verdict::Review,
            };
            if fv > v {
                v = fv;
            }
        }
        v
    }
}

/// The payload returned to Python for one check.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Report {
    pub tool: String,
    /// Launch (kernel index within the `Module`) this report covers; one `Report` per launch (W8-2).
    pub launch: u32,
    pub verdict: Verdict,
    pub findings: Vec<Finding>,
    /// What was covered (e.g. explored states, launches), free-form.
    pub coverage: Vec<(String, u64)>,
    /// Non-numeric coverage facts (termination kind, algorithm name, ...).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub meta: BTreeMap<String, serde_json::Value>,
}

impl Report {
    pub fn new(tool: &str, findings: Vec<Finding>) -> Report {
        Report { tool: tool.to_string(), launch: 0, verdict: Verdict::of(&findings), findings, coverage: Vec::new(), meta: BTreeMap::new() }
    }
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("Report is serializable")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn verdict_order() {
        let f = |status| Finding {
            kind: FindingKind::DataRace,
            status,
            message: String::new(),
            sites: vec![],
            evidence: vec![],
        };
        assert_eq!(Verdict::of(&[]), Verdict::Clean);
        assert_eq!(Verdict::of(&[f(Status::Review), f(Status::Incomplete)]), Verdict::Incomplete);
        assert_eq!(Verdict::of(&[f(Status::Error), f(Status::Incomplete)]), Verdict::Error);
        let j = Report::new("racecheck", vec![f(Status::Review)]).to_json();
        assert!(j.contains("\"data_race\""));
    }
}
