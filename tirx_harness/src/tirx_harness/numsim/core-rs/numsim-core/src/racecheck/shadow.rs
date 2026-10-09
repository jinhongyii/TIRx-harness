//! `IntervalShadow<C>`: per-allocation byte-range map from disjoint segments
//! to cells.
//!
//! * Segments are `[start, end)`, keyed by `start` in a `BTreeMap` whose
//!   values are `(end, slot)`; the cells live in a slab (`cells`, free list
//!   `free`). A racecheck `Cell` is ~80 bytes: stored inline, every node
//!   insert/remove shifted up to eleven of them, and the split/merge path
//!   (TMA/MMA ranges alternating with per-lane ranges over the same bytes)
//!   was ~0.75 us per update on cudnn gemm_proj_rope (W16).
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
    map: BTreeMap<u64, (u64, u32)>,
    cells: Vec<C>,
    free: Vec<u32>,
    pub exact_hits: u64,
    pub slow_updates: u64,
}

impl<C: Clone + Default + PartialEq> IntervalShadow<C> {
    pub fn new() -> Self {
        IntervalShadow { map: BTreeMap::new(), cells: Vec::new(), free: Vec::new(), exact_hits: 0, slow_updates: 0 }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn segments(&self) -> impl Iterator<Item = (Range<u64>, &C)> {
        self.map.iter().map(|(s, (e, i))| (*s..*e, &self.cells[*i as usize]))
    }

    #[inline]
    fn alloc_slot(&mut self, c: C) -> u32 {
        match self.free.pop() {
            Some(i) => {
                self.cells[i as usize] = c;
                i
            }
            None => {
                self.cells.push(c);
                (self.cells.len() - 1) as u32
            }
        }
    }

    #[inline]
    fn free_slot(&mut self, i: u32) {
        self.cells[i as usize] = C::default(); // drop what the cell held
        self.free.push(i);
    }

    /// Read-only visit of every existing segment overlapping `range`,
    /// clipped to it (gaps are skipped: a gap is a default cell).
    pub fn visit(&self, range: Range<u64>, mut f: impl FnMut(Range<u64>, &C)) {
        if range.start >= range.end {
            return;
        }
        if let Some((s, (e, i))) = self.map.range(..=range.start).next_back() {
            if *e > range.start {
                f(range.start.max(*s)..range.end.min(*e), &self.cells[*i as usize]);
            }
        }
        for (s, (e, i)) in self.map.range(range.start + 1..range.end) {
            f(*s..range.end.min(*e), &self.cells[*i as usize]);
        }
    }

    /// Apply `f` to every sub-range of `range`, materialising gaps.
    pub fn update(&mut self, range: Range<u64>, mut f: impl FnMut(Range<u64>, &mut C)) {
        if range.start >= range.end {
            return;
        }
        if !super::tuning::on(&super::tuning::EXACT_HIT) {
        } else if let Some(&(end, i)) = self.map.get(&range.start) {
            if end == range.end {
                self.exact_hits += 1;
                f(range, &mut self.cells[i as usize]);
                return;
            }
        }
        self.slow_updates += 1;
        // Gap fast path: no segment overlaps `range` (a first touch; 63% of
        // the slow updates on cudnn gemm_proj_rope, whose global shadow
        // holds ~0.3M segments). Same result as the general path below
        // (both splits are no-ops, one default piece, then the two edge
        // merges) with two tree descents instead of about nine.
        let left = self.map.range(..range.end).next_back().map(|(s, (e, i))| (*s, *e, *i));
        if left.is_none_or(|(_, e, _)| e <= range.start) {
            let i = self.alloc_slot(C::default());
            f(range.clone(), &mut self.cells[i as usize]);
            let (key, cur) = match left {
                Some((ls, le, li)) if le == range.start && self.cells[li as usize] == self.cells[i as usize] => {
                    self.map.get_mut(&ls).unwrap().0 = range.end;
                    self.free_slot(i);
                    (ls, li)
                }
                _ => {
                    self.map.insert(range.start, (range.end, i));
                    (range.start, i)
                }
            };
            if let Some(&(re, ri)) = self.map.get(&range.end) {
                if self.cells[cur as usize] == self.cells[ri as usize] {
                    self.map.remove(&range.end);
                    self.free_slot(ri);
                    self.map.get_mut(&key).unwrap().0 = re;
                }
            }
            return;
        }
        let left = self.split_at(range.start);
        self.split_at(range.end);
        // One pass over the covered segments, gaps materialised in place
        // (a default cell in a fresh slot, inserted after the pass): update
        // each piece in order, and group equal neighbours into runs (the
        // sequential merge of the pieces: a run of equal cells becomes its
        // first segment). The first key at or past the end is the right
        // neighbour. Same pieces, callback order and layout as collecting
        // the pieces, merging them, and reinserting them.
        let mut runs: Vec<(u64, u64, u32, bool)> = Vec::new(); // (start, end, slot, new)
        let mut drop: Vec<(u64, u32, bool)> = Vec::new();
        let mut right: Option<(u64, u32)> = None;
        {
            let cells = &mut self.cells;
            let free = &mut self.free;
            let mut piece = |s: u64, e: u64, i: u32, new: bool, cells: &mut Vec<C>, runs: &mut Vec<(u64, u64, u32, bool)>, drop: &mut Vec<(u64, u32, bool)>| {
                f(s..e, &mut cells[i as usize]);
                match runs.last_mut() {
                    Some(r) if r.1 == s && cells[r.2 as usize] == cells[i as usize] => {
                        r.1 = e;
                        drop.push((s, i, new));
                    }
                    _ => runs.push((s, e, i, new)),
                }
            };
            let mut fresh = |cells: &mut Vec<C>| -> u32 {
                match free.pop() {
                    Some(i) => {
                        cells[i as usize] = C::default();
                        i
                    }
                    None => {
                        cells.push(C::default());
                        (cells.len() - 1) as u32
                    }
                }
            };
            let mut cursor = range.start;
            for (k, (e, i)) in self.map.range(range.start..) {
                if *k >= range.end {
                    if *k == range.end {
                        right = Some((*e, *i));
                    }
                    break;
                }
                if *k > cursor {
                    let g = fresh(cells);
                    piece(cursor, *k, g, true, cells, &mut runs, &mut drop);
                }
                piece(*k, *e, *i, false, cells, &mut runs, &mut drop);
                cursor = *e;
            }
            if cursor < range.end {
                let g = fresh(cells);
                piece(cursor, range.end, g, true, cells, &mut runs, &mut drop);
            }
        }
        for (k, i, new) in drop {
            if !new {
                self.map.remove(&k);
            }
            self.free_slot(i);
        }
        for r in &runs {
            if r.3 {
                self.map.insert(r.0, (r.1, r.2));
            } else {
                let slot = self.map.get_mut(&r.0).unwrap();
                if slot.0 != r.1 {
                    slot.0 = r.1;
                }
            }
        }
        let mut runs: Vec<(u64, u64, u32)> = runs.into_iter().map(|(s, e, i, _)| (s, e, i)).collect();
        // Edge merges: the segment ending at `start` with the first run,
        // then the last run with the segment starting at `end`.
        if let Some((ls, le, li)) = left {
            let (fk, fe, fi) = runs[0];
            if le == range.start && self.cells[li as usize] == self.cells[fi as usize] {
                self.map.remove(&fk);
                self.free_slot(fi);
                self.map.get_mut(&ls).unwrap().0 = fe;
                runs[0] = (ls, fe, li);
            }
        }
        let (lk, le, li) = *runs.last().unwrap();
        debug_assert_eq!(le, range.end);
        if let Some((re, ri)) = right {
            if self.cells[li as usize] == self.cells[ri as usize] {
                self.map.remove(&range.end);
                self.free_slot(ri);
                self.map.get_mut(&lk).unwrap().0 = re;
            }
        }
    }

    /// Mutable visit of every segment; segments for which `f` returns
    /// `false` are dropped (GC). Returns the number dropped.
    pub fn retain_mut(&mut self, mut f: impl FnMut(&mut C) -> bool) -> usize {
        let before = self.map.len();
        let cells = &mut self.cells;
        let free = &mut self.free;
        self.map.retain(|_, (_, i)| {
            let keep = f(&mut cells[*i as usize]);
            if !keep {
                cells[*i as usize] = C::default();
                free.push(*i);
            }
            keep
        });
        before - self.map.len()
    }

    /// Remove everything in `range` (allocation end / reuse).
    pub fn clear(&mut self, range: Range<u64>) {
        self.split_at(range.start);
        self.split_at(range.end);
        let keys: Vec<(u64, u32)> = self.map.range(range.start..range.end).map(|(k, (_, i))| (*k, *i)).collect();
        for (k, i) in keys {
            self.map.remove(&k);
            self.free_slot(i);
        }
    }

    /// Split the segment straddling `at`; returns the segment that now
    /// precedes `at` (`(start, end, slot)`), if any.
    fn split_at(&mut self, at: u64) -> Option<(u64, u64, u32)> {
        let (&s, &(e, i)) = self.map.range(..at).next_back()?;
        if e <= at {
            return Some((s, e, i));
        }
        let cell = self.cells[i as usize].clone();
        let j = self.alloc_slot(cell);
        self.map.get_mut(&s).unwrap().0 = at;
        self.map.insert(at, (e, j));
        Some((s, at, i))
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

    /// The slab layout with its gap fast path and one-pass merge produces
    /// exactly the segments (and the callback sequence) of the original
    /// inline-cell algorithm, kept here as the reference (W16).
    #[test]
    fn matches_reference_layout() {
        #[derive(Default)]
        struct Reference(BTreeMap<u64, (u64, u8)>);
        impl Reference {
            fn split_at(&mut self, at: u64) {
                let Some((&s, &(e, c))) = self.0.range(..at).next_back() else { return };
                if e <= at {
                    return;
                }
                self.0.get_mut(&s).unwrap().0 = at;
                self.0.insert(at, (e, c));
            }
            fn merge_at(&mut self, at: u64) {
                let Some((&ls, &(le, lc))) = self.0.range(..at).next_back() else { return };
                if le != at {
                    return;
                }
                let Some(&(re, rc)) = self.0.get(&at) else { return };
                if lc != rc {
                    return;
                }
                self.0.remove(&at);
                self.0.get_mut(&ls).unwrap().0 = re;
            }
            fn update(&mut self, r: Range<u64>, mut f: impl FnMut(Range<u64>, &mut u8)) {
                if let Some((end, c)) = self.0.get_mut(&r.start) {
                    if *end == r.end {
                        f(r, c);
                        return;
                    }
                }
                self.split_at(r.start);
                self.split_at(r.end);
                let mut pieces: Vec<(u64, u64, u8)> = Vec::new();
                let mut cursor = r.start;
                let keys: Vec<u64> = self.0.range(r.start..r.end).map(|(k, _)| *k).collect();
                for k in keys {
                    let (e, c) = self.0.remove(&k).unwrap();
                    if k > cursor {
                        pieces.push((cursor, k, 0));
                    }
                    pieces.push((k, e, c));
                    cursor = e;
                }
                if cursor < r.end {
                    pieces.push((cursor, r.end, 0));
                }
                for (s, e, c) in pieces.iter_mut() {
                    f(*s..*e, c);
                }
                let mut merged: Vec<(u64, u64, u8)> = Vec::new();
                for p in pieces {
                    match merged.last_mut() {
                        Some(last) if last.1 == p.0 && last.2 == p.2 => last.1 = p.1,
                        _ => merged.push(p),
                    }
                }
                for (s, e, c) in merged {
                    self.0.insert(s, (e, c));
                }
                self.merge_at(r.start);
                self.merge_at(r.end);
            }
        }
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        for round in 0..200 {
            let mut a: IntervalShadow<u8> = IntervalShadow::new();
            let mut b = Reference::default();
            for _ in 0..400 {
                let start = rnd(256);
                let len = 1 + rnd(if round % 2 == 0 { 8 } else { 64 });
                let mode = rnd(4);
                let v = rnd(3) as u8;
                let (mut ca, mut cb) = (Vec::new(), Vec::new());
                a.update(start..start + len, |r, c| {
                    ca.push((r.clone(), *c));
                    *c = match mode { 0 => v, 1 => *c, 2 => c.wrapping_add(1) % 3, _ => (r.start % 2) as u8 };
                });
                b.update(start..start + len, |r, c| {
                    cb.push((r.clone(), *c));
                    *c = match mode { 0 => v, 1 => *c, 2 => c.wrapping_add(1) % 3, _ => (r.start % 2) as u8 };
                });
                assert_eq!(ca, cb);
                if rnd(16) == 0 {
                    let cs = rnd(256);
                    let r = cs..cs + 1 + rnd(32);
                    a.clear(r.clone());
                    b.split_at(r.start);
                    b.split_at(r.end);
                    let keys: Vec<u64> = b.0.range(r.clone()).map(|(k, _)| *k).collect();
                    for k in keys {
                        b.0.remove(&k);
                    }
                }
                let sa: Vec<(Range<u64>, u8)> = a.segments().map(|(r, c)| (r, *c)).collect();
                let sb: Vec<(Range<u64>, u8)> = b.0.iter().map(|(s, (e, c))| (*s..*e, *c)).collect();
                assert_eq!(sa, sb);
            }
        }
    }
}
