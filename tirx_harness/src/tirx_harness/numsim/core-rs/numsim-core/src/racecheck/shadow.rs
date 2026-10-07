//! `IntervalShadow<C>`: per-allocation byte-range map from disjoint segments
//! to cells.
//!
//! * Segments are `[start, end)`, keyed by `start` in a `BTreeMap`.
//! * **Exact-hit fast path**: an access whose range is exactly one existing
//!   segment updates that cell in place — no split, no reinsert, no merge.
//!   Tile loops revisit the same shapes, so after warm-up nearly every access
//!   takes this path: one O(log n) lookup and zero allocations, versus up to
//!   two splits + k removes + reinserts + neighbour merges otherwise.
//! * Otherwise the range is split at its two boundaries, gaps are filled with
//!   `C::default()`, the update runs per covered segment, and equal adjacent
//!   segments inside and at the edges of the range are merged, so the map
//!   does not fragment monotonically.
//!
//! The update closure sees each covered sub-range with its cell and may
//! record findings; check and record happen in one pass (the checker's
//! rule is check-then-record per segment, which is order independent across
//! segments because segments are disjoint).

use std::collections::BTreeMap;
use std::ops::Range;

#[derive(Clone, Debug, Default)]
pub struct IntervalShadow<C> {
    map: BTreeMap<u64, (u64, C)>,
    pub exact_hits: u64,
    pub slow_updates: u64,
}

impl<C: Clone + Default + PartialEq> IntervalShadow<C> {
    pub fn new() -> Self {
        IntervalShadow { map: BTreeMap::new(), exact_hits: 0, slow_updates: 0 }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn segments(&self) -> impl Iterator<Item = (Range<u64>, &C)> {
        self.map.iter().map(|(s, (e, c))| (*s..*e, c))
    }

    /// Read-only visit of every existing segment overlapping `range`,
    /// clipped to it (gaps are skipped: a gap is a default cell).
    pub fn visit(&self, range: Range<u64>, mut f: impl FnMut(Range<u64>, &C)) {
        if range.start >= range.end {
            return;
        }
        if let Some((s, (e, c))) = self.map.range(..=range.start).next_back() {
            if *e > range.start {
                f(range.start.max(*s)..range.end.min(*e), c);
            }
        }
        for (s, (e, c)) in self.map.range(range.start + 1..range.end) {
            f(*s..range.end.min(*e), c);
        }
    }

    /// Apply `f` to every sub-range of `range`, materialising gaps.
    pub fn update(&mut self, range: Range<u64>, mut f: impl FnMut(Range<u64>, &mut C)) {
        if range.start >= range.end {
            return;
        }
        if let Some((end, cell)) = self.map.get_mut(&range.start) {
            if *end == range.end {
                self.exact_hits += 1;
                f(range, cell);
                return;
            }
        }
        self.slow_updates += 1;
        self.split_at(range.start);
        self.split_at(range.end);
        // Collect the covered segments and gaps in order.
        let mut pieces: Vec<(u64, u64, C)> = Vec::new();
        let mut cursor = range.start;
        let keys: Vec<u64> = self.map.range(range.start..range.end).map(|(k, _)| *k).collect();
        for k in keys {
            let (e, c) = self.map.remove(&k).unwrap();
            if k > cursor {
                pieces.push((cursor, k, C::default()));
            }
            pieces.push((k, e, c));
            cursor = e;
        }
        if cursor < range.end {
            pieces.push((cursor, range.end, C::default()));
        }
        for (s, e, c) in pieces.iter_mut() {
            f(*s..*e, c);
        }
        // Merge equal neighbours inside the range.
        let mut merged: Vec<(u64, u64, C)> = Vec::with_capacity(pieces.len());
        for p in pieces {
            match merged.last_mut() {
                Some(last) if last.1 == p.0 && last.2 == p.2 => last.1 = p.1,
                _ => merged.push(p),
            }
        }
        for (s, e, c) in merged {
            self.map.insert(s, (e, c));
        }
        self.merge_at(range.start);
        self.merge_at(range.end);
    }

    /// Mutable visit of every segment; segments for which `f` returns
    /// `false` are dropped (GC). Returns the number dropped.
    pub fn retain_mut(&mut self, mut f: impl FnMut(&mut C) -> bool) -> usize {
        let before = self.map.len();
        self.map.retain(|_, (_, c)| f(c));
        before - self.map.len()
    }

    /// Remove everything in `range` (allocation end / reuse).
    pub fn clear(&mut self, range: Range<u64>) {
        self.split_at(range.start);
        self.split_at(range.end);
        let keys: Vec<u64> = self.map.range(range.start..range.end).map(|(k, _)| *k).collect();
        for k in keys {
            self.map.remove(&k);
        }
    }

    fn split_at(&mut self, at: u64) {
        let Some((&s, (e, _))) = self.map.range(..at).next_back() else { return };
        if *e <= at {
            return;
        }
        let e = *e;
        let cell = self.map.get(&s).unwrap().1.clone();
        self.map.get_mut(&s).unwrap().0 = at;
        self.map.insert(at, (e, cell));
    }

    /// Merge the segment ending at `at` with the one starting at `at` if equal.
    fn merge_at(&mut self, at: u64) {
        let Some((&ls, (le, lc))) = self.map.range(..at).next_back() else { return };
        if *le != at {
            return;
        }
        let Some((re, rc)) = self.map.get(&at) else { return };
        if lc != rc {
            return;
        }
        let re = *re;
        self.map.remove(&at);
        self.map.get_mut(&ls).unwrap().0 = re;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_and_merge() {
        let mut s: IntervalShadow<u32> = IntervalShadow::new();
        s.update(0..16, |_, c| *c = 1);
        s.update(4..8, |_, c| *c = 2);
        assert_eq!(s.len(), 3);
        s.update(4..8, |_, c| *c = 1);
        assert_eq!(s.exact_hits, 1);
        // Exact hit does not merge; the next slow update at the edges does.
        s.update(0..16, |_, c| *c = 1);
        assert_eq!(s.len(), 1);
        let mut seen = vec![];
        s.visit(2..20, |r, c| seen.push((r, *c)));
        assert_eq!(seen, vec![(2..16, 1)]);
    }

    #[test]
    fn gaps_are_materialised() {
        let mut s: IntervalShadow<u32> = IntervalShadow::new();
        s.update(8..12, |_, c| *c = 5);
        let mut ranges = vec![];
        s.update(0..16, |r, c| {
            ranges.push((r.clone(), *c));
            *c += 1;
        });
        assert_eq!(ranges, vec![(0..8, 0), (8..12, 5), (12..16, 0)]);
        assert_eq!(s.len(), 3);
    }
}
