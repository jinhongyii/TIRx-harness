//! Shadow cell: the write and read frontiers of one byte range, holding
//! packed 16-byte witnesses.
//!
//! A frontier is an antichain of witnesses: a new witness evicts every prior
//! it *observes* (happens-after) **and whose conflict contract it subsumes**
//! (same kind and, for strong accesses, same scope / proxy / exact span).
//! The contract condition keeps morally-strong exemptions sound: an
//! `atom.gpu` that happens after a plain `st` must not evict it, because a
//! later remote `atom.gpu` is exempt against the atom but races the store.
//!
//! In the race-free steady state each frontier holds one witness (`One`), so
//! a check is O(1) packed-stamp compares.

use std::ops::Range;
use std::sync::Arc;

use super::clock::Stamp;
use super::input::{AccessKind, Domain, Proxy, Scope};
use super::knowledge::Heads;

/// One retained access in 16 bytes: the packed `(actor, epoch)` stamp plus
/// one word of metadata.
///
/// `bits` (LSB first): lane 5 | proxy 3 | domain 2 | kind 2 | scope 3 |
/// atomic 1 | wide 1 | len 15 | start 32. A span that does not fit (start
/// ≥ 4 GiB or len ≥ 32 KiB) sets `wide` and stores an index into the
/// checker's wide-span table in `start`. The site, the performing warp and
/// the async op are recovered from the stamp (per-warp epoch→site table,
/// async actor table), so they cost no witness bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Witness {
    pub stamp: Stamp,
    bits: u64,
}

const LANE_SHIFT: u32 = 0;
const PROXY_SHIFT: u32 = 5;
const DOMAIN_SHIFT: u32 = 8;
const KIND_SHIFT: u32 = 10;
const SCOPE_SHIFT: u32 = 12;
const ATOMIC_BIT: u64 = 1 << 15;
const WIDE_BIT: u64 = 1 << 16;
const LEN_SHIFT: u32 = 17;
const LEN_MAX: u64 = (1 << 15) - 1;
const START_SHIFT: u32 = 32;
/// Proxy and domain fields.
const VIEW_CLASS_MASK: u64 = (7 << PROXY_SHIFT) | (3 << DOMAIN_SHIFT);
/// Mask of the span fields (wide, len, start).
const SPAN_MASK: u64 = !((1u64 << 16) - 1);

fn proxy_code(p: Proxy) -> u64 {
    match p {
        Proxy::Generic => 0,
        Proxy::Async => 1,
        Proxy::TensorMap => 2,
        Proxy::ReadOnly => 3,
        Proxy::Tcgen => 4,
    }
}

fn proxy_of(c: u64) -> Proxy {
    match c {
        0 => Proxy::Generic,
        1 => Proxy::Async,
        2 => Proxy::TensorMap,
        3 => Proxy::ReadOnly,
        _ => Proxy::Tcgen,
    }
}

/// Spans that do not fit the compact encoding, deduplicated: the table is
/// bounded by the number of distinct wide spans (review R3).
#[derive(Clone, Debug, Default)]
pub struct WideSpans {
    pub spans: Vec<(u64, u64)>,
    index: std::collections::HashMap<(u64, u64), u64>,
}

impl WideSpans {
    fn intern(&mut self, span: (u64, u64)) -> u64 {
        if let Some(&i) = self.index.get(&span) {
            return i;
        }
        let i = self.spans.len() as u64;
        self.spans.push(span);
        self.index.insert(span, i);
        i
    }
}

#[allow(clippy::too_many_arguments)]
impl Witness {
    pub fn pack(
        stamp: Stamp,
        lane: u8,
        proxy: Proxy,
        domain: Option<Domain>,
        kind: AccessKind,
        scope: Option<Scope>,
        atomic: bool,
        span: (u64, u64),
        wide: &mut WideSpans,
    ) -> Witness {
        let mut bits = (lane as u64 & 31) << LANE_SHIFT;
        bits |= proxy_code(proxy) << PROXY_SHIFT;
        bits |= (match domain {
            None => 0,
            Some(Domain::Global) => 1,
            Some(Domain::SharedCta) => 2,
            Some(Domain::SharedCluster) => 3,
        }) << DOMAIN_SHIFT;
        bits |= (match kind {
            AccessKind::Read => 0,
            AccessKind::Write => 1,
            AccessKind::Rmw => 2,
        }) << KIND_SHIFT;
        bits |= (match scope {
            None => 0,
            Some(Scope::Cta) => 1,
            Some(Scope::Cluster) => 2,
            Some(Scope::Gpu) => 3,
            Some(Scope::Sys) => 4,
        }) << SCOPE_SHIFT;
        if atomic {
            bits |= ATOMIC_BIT;
        }
        let len = span.1 - span.0;
        if span.0 < (1 << 32) && len <= LEN_MAX {
            bits |= (len << LEN_SHIFT) | (span.0 << START_SHIFT);
        } else {
            let i = wide.intern(span);
            bits |= WIDE_BIT | (i << START_SHIFT);
        }
        Witness { stamp, bits }
    }

    #[inline(always)]
    pub fn lane(&self) -> u8 {
        ((self.bits >> LANE_SHIFT) & 31) as u8
    }
    #[inline(always)]
    pub fn proxy(&self) -> Proxy {
        proxy_of((self.bits >> PROXY_SHIFT) & 7)
    }
    #[inline(always)]
    pub fn domain(&self) -> Option<Domain> {
        match (self.bits >> DOMAIN_SHIFT) & 3 {
            0 => None,
            1 => Some(Domain::Global),
            2 => Some(Domain::SharedCta),
            _ => Some(Domain::SharedCluster),
        }
    }
    #[inline(always)]
    pub fn kind(&self) -> AccessKind {
        match (self.bits >> KIND_SHIFT) & 3 {
            0 => AccessKind::Read,
            1 => AccessKind::Write,
            _ => AccessKind::Rmw,
        }
    }
    #[inline(always)]
    pub fn scope(&self) -> Option<Scope> {
        match (self.bits >> SCOPE_SHIFT) & 7 {
            0 => None,
            1 => Some(Scope::Cta),
            2 => Some(Scope::Cluster),
            3 => Some(Scope::Gpu),
            _ => Some(Scope::Sys),
        }
    }
    #[inline(always)]
    pub fn atomic(&self) -> bool {
        self.bits & ATOMIC_BIT != 0
    }
    #[inline(always)]
    pub fn writes(&self) -> bool {
        (self.bits >> KIND_SHIFT) & 3 != 0
    }
    pub fn span(&self, wide: &WideSpans) -> (u64, u64) {
        let start = self.bits >> START_SHIFT;
        if self.bits & WIDE_BIT != 0 {
            wide.spans[start as usize]
        } else {
            (start, start + ((self.bits >> LEN_SHIFT) & LEN_MAX))
        }
    }
    /// Equal spans (exact for compact witnesses; wide spans compare by value).
    pub fn same_span(&self, o: &Witness, wide: &WideSpans) -> bool {
        if (self.bits | o.bits) & WIDE_BIT == 0 {
            self.bits & SPAN_MASK == o.bits & SPAN_MASK
        } else {
            self.span(wide) == o.span(wide)
        }
    }
    /// Same dynamic event (same instruction and lane).
    #[inline(always)]
    pub fn same_event(&self, o: &Witness) -> bool {
        self.stamp == o.stamp && self.lane() == o.lane()
    }

    /// Same proxy and same address window: the views a future access uses
    /// to judge the two are the same (bridges are keyed by the prior's
    /// window), so transitivity through `self` covers `prior`.
    #[inline(always)]
    pub fn same_view_class(&self, o: &Witness) -> bool {
        (self.bits ^ o.bits) & VIEW_CLASS_MASK == 0
    }

    /// `self` (newer) may stand in for `prior` in a frontier.
    pub fn subsumes_contract(&self, prior: &Witness, wide: &WideSpans) -> bool {
        if self.kind() != prior.kind() || !self.same_view_class(prior) {
            return false;
        }
        match self.scope() {
            None => prior.scope().is_none() || !prior.atomic(),
            Some(s) => {
                prior.scope() == Some(s)
                    && prior.atomic() == self.atomic()
                    && prior.proxy() == self.proxy()
                    && self.same_span(prior, wide)
            }
        }
    }
}

/// A frontier entry. Writes may carry the release heads a reader that
/// reads-from them acquires (own release / fence-release head plus the heads
/// inherited through an RMW observation chain).
///
/// `base` is the part inherited from writes that precede the whole
/// instruction. Sibling lanes of one same-address RMW instruction are
/// independent threads with unconstrained coherence order (PTX §8.9.1), so
/// whoever reads from a sibling group may only rely on `base`.
#[derive(Clone, Debug)]
pub struct Entry {
    pub w: Witness,
    pub rel: Option<Heads>,
    pub base: Option<Heads>,
}

fn heads_eq(a: &Option<Heads>, b: &Option<Heads>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
        _ => false,
    }
}

impl PartialEq for Entry {
    fn eq(&self, o: &Self) -> bool {
        self.w == o.w && heads_eq(&self.rel, &o.rel) && heads_eq(&self.base, &o.base)
    }
}

/// The heads a reader of `e` may rely on (see [`Entry::base`]).
pub fn effective_heads(f: &Frontier, e: &Entry) -> Option<Heads> {
    let siblings = f.as_slice().iter().any(|o| o.w.stamp == e.w.stamp && o.w.lane() != e.w.lane());
    if siblings {
        e.base.clone()
    } else {
        e.rel.clone()
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub enum Frontier {
    #[default]
    Empty,
    One(Entry),
    Many(Vec<Entry>),
}

/// How many of a frontier's newest entries a record tries to evict.
pub const EVICT_WINDOW: usize = 8;

impl Frontier {
    pub fn as_slice(&self) -> &[Entry] {
        match self {
            Frontier::Empty => &[],
            Frontier::One(e) => std::slice::from_ref(e),
            Frontier::Many(v) => v,
        }
    }

    pub fn last(&self) -> Option<&Entry> {
        self.as_slice().last()
    }

    pub fn is_empty(&self) -> bool {
        matches!(self, Frontier::Empty)
    }

    /// Insert `new`, evicting recent priors it observes and subsumes.
    #[inline]
    pub fn record(&mut self, new: Entry, wide: &WideSpans, mut observed: impl FnMut(&Witness) -> bool) {
        let nw = new.w;
        let eviction = super::tuning::on(&super::tuning::FRONTIER_EVICTION);
        let mut evict = |p: &Witness| eviction && nw.subsumes_contract(p, wide) && (p.same_event(&nw) || observed(p));
        match self {
            Frontier::Empty => *self = Frontier::One(new),
            Frontier::One(prior) => {
                if evict(&prior.w) {
                    *prior = new; // the steady state: in place
                } else {
                    let Frontier::One(prior) = std::mem::take(self) else { unreachable!() };
                    *self = Frontier::Many(vec![prior, new]);
                }
            }
            Frontier::Many(v) => {
                // Only the newest EVICT_WINDOW entries are candidates: a
                // full scan made every record O(frontier) and read-shared
                // cells (thousands of unordered readers) quadratic. The
                // candidates that matter (the same lane's previous access
                // in a loop, siblings of the same instruction) are recent;
                // older dominated entries are left to GC. Evicting a subset
                // is always sound (an evicted entry is subsumed and
                // observed by `new`).
                let start = v.len().saturating_sub(EVICT_WINDOW);
                let mut keep = start;
                for r in start..v.len() {
                    if !evict(&v[r].w) {
                        v.swap(keep, r);
                        keep += 1;
                    }
                }
                v.truncate(keep);
                v.push(new);
                if v.len() == 1 {
                    let e = v.pop().unwrap();
                    *self = Frontier::One(e);
                }
            }
        }
    }

    pub fn retain(&mut self, mut keep: impl FnMut(&Entry) -> bool) {
        match self {
            Frontier::Empty => {}
            Frontier::One(e) => {
                if !keep(e) {
                    *self = Frontier::Empty;
                }
            }
            Frontier::Many(v) => {
                v.retain(|e| keep(e));
                match v.len() {
                    0 => *self = Frontier::Empty,
                    1 => {
                        let e = v.pop().unwrap();
                        *self = Frontier::One(e);
                    }
                    _ => {}
                }
            }
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Cell {
    pub writes: Frontier,
    pub reads: Frontier,
}

impl Cell {
    pub fn is_empty(&self) -> bool {
        self.writes.is_empty() && self.reads.is_empty()
    }
}

pub fn overlap(a: (u64, u64), r: &Range<u64>) -> Range<u64> {
    a.0.max(r.start)..a.1.min(r.end)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn witness_is_16_bytes_and_roundtrips() {
        assert_eq!(std::mem::size_of::<Witness>(), 16);
        let mut wide = WideSpans::default();
        let w = Witness::pack(
            Stamp::new(7, 9),
            31,
            Proxy::Tcgen,
            Some(Domain::SharedCluster),
            AccessKind::Rmw,
            Some(Scope::Sys),
            true,
            (4096, 4096 + 64),
            &mut wide,
        );
        assert_eq!((w.lane(), w.proxy(), w.domain(), w.kind(), w.scope(), w.atomic()), (31, Proxy::Tcgen, Some(Domain::SharedCluster), AccessKind::Rmw, Some(Scope::Sys), true));
        assert_eq!(w.span(&wide), (4096, 4160));
        let big = Witness::pack(Stamp::new(1, 1), 0, Proxy::Generic, None, AccessKind::Read, None, false, (1 << 33, (1 << 33) + (1 << 20)), &mut wide);
        assert_eq!(big.span(&wide), (1 << 33, (1 << 33) + (1 << 20)));
        assert_eq!(wide.spans.len(), 1);
    }
}
