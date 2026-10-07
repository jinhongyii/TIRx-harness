//! Shadow cell: the write and read frontiers of one byte range.
//!
//! A frontier is an antichain of witnesses: a new witness evicts every prior
//! it *observes* (happens-after) **and whose conflict contract it subsumes**
//! (same kind and, for strong accesses, same scope / proxy / exact span).
//! The contract condition is what keeps morally-strong exemptions sound: an
//! `atom.gpu` that happens after a plain `st` must not evict it, because a
//! later remote `atom.gpu` is exempt against the atom but races the store.
//!
//! In the race-free steady state each frontier holds one witness (`One`), so
//! a check is O(1) packed-stamp compares; it grows only with genuinely
//! concurrent readers / morally-strong writers.

use std::ops::Range;
use std::sync::Arc;

use crate::clock::Stamp;
use crate::knowledge::Rel;
use crate::input::{AccessKind, Domain, Proxy, Scope, SiteId};

/// One retained access. 48 bytes; the legacy compact form packs the
/// equivalent into 16 (see spec §3.1) — a later optimisation, not semantics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Witness {
    pub stamp: Stamp,
    pub lane: u8,
    pub proxy: Proxy,
    pub domain: Option<Domain>,
    pub kind: AccessKind,
    /// Strong scope (scoped relaxed/acquire/release or atomic); `None` = weak.
    pub scope: Option<Scope>,
    pub atomic: bool,
    pub span: (u64, u64),
    pub site: SiteId,
    /// Warp performing it (issuing warp for async actors), for scope tests
    /// and for attributing async accesses to their issuer in evidence.
    pub warp: u32,
}

impl Witness {
    pub fn writes(&self) -> bool {
        !matches!(self.kind, AccessKind::Read)
    }

    /// `self` (newer) may stand in for `prior` in a frontier.
    pub fn subsumes_contract(&self, prior: &Witness) -> bool {
        if self.kind != prior.kind {
            return false;
        }
        match self.scope {
            None => prior.scope.is_none() || !prior.atomic,
            Some(s) => {
                prior.scope == Some(s)
                    && prior.atomic == self.atomic
                    && prior.proxy == self.proxy
                    && prior.span == self.span
            }
        }
    }
}

/// A frontier entry. Writes may carry the release payload a reader that
/// reads-from them acquires (release head, fence-release head, or the
/// accumulated release sequence of an RMW chain).
#[derive(Clone, Debug)]
pub struct Entry {
    pub w: Witness,
    pub rel: Option<Arc<Rel>>,
}

impl PartialEq for Entry {
    fn eq(&self, o: &Self) -> bool {
        self.w == o.w
            && match (&self.rel, &o.rel) {
                (None, None) => true,
                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                _ => false,
            }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub enum Frontier {
    #[default]
    Empty,
    One(Entry),
    Many(Vec<Entry>),
}

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

    pub fn clear(&mut self) {
        *self = Frontier::Empty;
    }

    /// Insert `new`, evicting priors it observes and subsumes. `observed`
    /// is the happens-before test against the new access's knowledge.
    #[inline]
    pub fn record(&mut self, new: Entry, mut observed: impl FnMut(&Witness) -> bool) {
        match self {
            Frontier::Empty => *self = Frontier::One(new),
            Frontier::One(prior) => {
                if new.w.subsumes_contract(&prior.w) && (prior.w.stamp == new.w.stamp && prior.w.lane == new.w.lane || observed(&prior.w)) {
                    *prior = new; // the exact-hit steady state: in place
                } else {
                    let prior = std::mem::replace(self, Frontier::Empty);
                    let Frontier::One(prior) = prior else { unreachable!() };
                    *self = Frontier::Many(vec![prior, new]);
                }
            }
            Frontier::Many(v) => {
                v.retain(|p| !(new.w.subsumes_contract(&p.w) && (p.w.stamp == new.w.stamp && p.w.lane == new.w.lane || observed(&p.w))));
                v.push(new);
                if v.len() == 1 {
                    let e = v.pop().unwrap();
                    *self = Frontier::One(e);
                }
            }
        }
    }

    pub fn retain(&mut self, mut keep: impl FnMut(&Entry) -> bool) {
        let mut v: Vec<Entry> = std::mem::take(self).into_vec();
        v.retain(|e| keep(e));
        *self = Frontier::from_vec(v);
    }

    fn into_vec(self) -> Vec<Entry> {
        match self {
            Frontier::Empty => vec![],
            Frontier::One(e) => vec![e],
            Frontier::Many(v) => v,
        }
    }

    fn from_vec(mut v: Vec<Entry>) -> Self {
        match v.len() {
            0 => Frontier::Empty,
            1 => Frontier::One(v.pop().unwrap()),
            _ => Frontier::Many(v),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Cell {
    pub writes: Frontier,
    pub reads: Frontier,
}

pub fn overlap(a: (u64, u64), r: &Range<u64>) -> Range<u64> {
    a.0.max(r.start)..a.1.min(r.end)
}
