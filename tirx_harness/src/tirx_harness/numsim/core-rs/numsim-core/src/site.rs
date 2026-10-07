//! Static source sites.
//!
//! Sites are allocated in lowering order and interned by (IR node, role);
//! their identity is their index. Every instruction that can produce a
//! finding or an event has one (`Program::code_sites[pc]`); pure ALU
//! instructions use `SiteId::NONE`. Snapshots compare findings by kind +
//! span + byte overlap, never by site index (W1 B.9).

use serde::{Deserialize, Serialize};
use std::fmt;

/// Index into `Program::sites`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SiteId(pub u32);

impl SiteId {
    pub const NONE: SiteId = SiteId(u32::MAX);
    pub const fn is_none(self) -> bool {
        self.0 == u32::MAX
    }
}

impl fmt::Display for SiteId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_none() {
            f.write_str("s?")
        } else {
            write!(f, "s{}", self.0)
        }
    }
}

/// One source span (1-based lines/columns; 0 = unknown).
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Span {
    #[serde(deserialize_with = "crate::program::required")]
    pub file: Option<String>,
    pub line: u32,
    pub col: u32,
    pub end_line: u32,
    pub end_col: u32,
}

/// Static facts about one site.
/// Serde: unknown fields are rejected and `Option` fields must be present
/// (`null` for none), like every program type.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SiteInfo {
    /// IR node `type_key` (`tirx.Call`, `tirx.BufferStore`, ...).
    pub kind: String,
    /// Flattened `SequentialSpan` chain (tirx-lite helpers chain call sites),
    /// innermost first.
    pub spans: Vec<Span>,
    /// Canonical op name (`tirx.ptx.mbarrier`), empty for statements.
    pub op_name: String,
    /// Short TVMScript rendering (<= 200 chars, children elided).
    pub text: String,
    #[serde(deserialize_with = "crate::program::required")]
    pub dtype: Option<String>,
    #[serde(deserialize_with = "crate::program::required")]
    pub buffer: Option<String>,
}
