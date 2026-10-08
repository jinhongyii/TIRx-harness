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
/// (`null` for none), like every program type. `buffers` (README decision
/// 15) may be absent while lowering still emits only the legacy `buffer`;
/// it is then derived as `[buffer]`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(from = "SiteInfoWire", into = "SiteInfoWire")]
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
    pub dtype: Option<String>,
    /// Buffer named by each pointer operand of the site, in operand order
    /// (`None`: a raw pointer with no buffer). `observe::Access::operand`
    /// indexes it (W5-15).
    pub buffers: Vec<Option<String>>,
}

impl SiteInfo {
    /// The first pointer operand's buffer (transition accessor for the
    /// pre-W5-15 single `buffer` field).
    pub fn buffer(&self) -> Option<&str> {
        self.buffers.first().and_then(|b| b.as_deref())
    }
    /// Buffer of pointer operand `operand`.
    pub fn buffer_of(&self, operand: u8) -> Option<&str> {
        self.buffers.get(operand as usize).and_then(|b| b.as_deref())
    }
}

/// JSON shape of [`SiteInfo`]: `buffers` is the contract field; the legacy
/// `buffer` is still accepted (and written, `= buffers[0]`) during the
/// transition.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SiteInfoWire {
    kind: String,
    spans: Vec<Span>,
    op_name: String,
    text: String,
    #[serde(deserialize_with = "crate::program::required")]
    dtype: Option<String>,
    #[serde(default)]
    buffer: Option<String>,
    #[serde(default)]
    buffers: Option<Vec<Option<String>>>,
}

impl From<SiteInfoWire> for SiteInfo {
    fn from(w: SiteInfoWire) -> SiteInfo {
        let buffers = match (w.buffers, w.buffer) {
            (Some(b), _) => b,
            (None, Some(b)) => vec![Some(b)],
            (None, None) => Vec::new(),
        };
        SiteInfo { kind: w.kind, spans: w.spans, op_name: w.op_name, text: w.text, dtype: w.dtype, buffers }
    }
}

impl From<SiteInfo> for SiteInfoWire {
    fn from(s: SiteInfo) -> SiteInfoWire {
        let buffer = s.buffer().map(str::to_string);
        SiteInfoWire { kind: s.kind, spans: s.spans, op_name: s.op_name, text: s.text, dtype: s.dtype, buffer, buffers: Some(s.buffers) }
    }
}
