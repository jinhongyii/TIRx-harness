//! Vector clocks over a dense actor index.
//!
//! Actor index space: `0..num_warps` are warps of every CTA of the launch
//! (so cross-CTA actors are ordinary components), `num_warps..` are async
//! virtual actors allocated at issue. One representation serves shared
//! memory, TMEM and global memory.
//!
//! Three representation tricks, each with an operation-count reason:
//!
//! 1. **Packed stamps** ([`Stamp`]): an access is `(actor, epoch)` in one
//!    `u64`. "Is this access ordered before me?" is one component read and
//!    one compare (`clock[actor] >= epoch`), O(1), instead of an O(actors)
//!    vector-clock comparison (FastTrack's epoch optimisation).
//! 2. **Chunked, `Arc`-shared epochs** ([`Epochs`]): components live in
//!    immutable `CHUNK`-wide chunks shared between clocks. Async actors are
//!    never recycled, so a flat vector grows with every op ever issued and
//!    every join/copy would cost O(all ops). With chunks a join first compares
//!    chunk pointers (equal → skip), then dominance (one side adopts the
//!    other's chunk, so synchronised clocks converge on shared storage), and
//!    only genuinely incomparable chunk pairs are merged — once, thanks to the
//!    join memo. A copy is a refcount bump.
//! 3. **Join memo** ([`JoinMemo`]): incomparable chunk joins are memoised by
//!    the pair of chunk identities (weakly held). Barrier fan-in makes many
//!    warps join the same two payload chunks; after the first join the rest
//!    are a hash lookup and share the result chunk, which in turn keeps later
//!    joins on the pointer-equality fast path.
//!
//! Lane precision: a warp component means "every lane of that warp up to this
//! epoch". When only some lanes released (a single-lane `st.release`, an
//! elected mbarrier arrive after divergent lane work), the clock carries a
//! sparse per-lane vector for that warp ([`LaneEntries`]). It is sparse
//! because the common case (whole-warp release after convergence) needs none.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::{Arc, Weak};

pub type ActorId = u32;
pub type Epoch = u32;

/// `(epoch << 32) | actor`, the unit of all shadow witnesses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Stamp(pub u64);

impl Stamp {
    #[inline(always)]
    pub const fn new(actor: ActorId, epoch: Epoch) -> Self {
        Stamp(((epoch as u64) << 32) | actor as u64)
    }
    #[inline(always)]
    pub const fn actor(self) -> ActorId {
        self.0 as u32
    }
    #[inline(always)]
    pub const fn epoch(self) -> Epoch {
        (self.0 >> 32) as u32
    }
}

pub const CHUNK: usize = 32;

#[derive(Debug)]
pub struct Chunk {
    id: u64,
    e: [Epoch; CHUNK],
}

thread_local! {
    static NEXT_CHUNK_ID: Cell<u64> = const { Cell::new(1) };
}

impl Chunk {
    fn new(e: [Epoch; CHUNK]) -> Arc<Self> {
        let id = NEXT_CHUNK_ID.with(|c| {
            let id = c.get();
            c.set(id + 1);
            id
        });
        Arc::new(Chunk { id, e })
    }
    #[inline]
    fn dominates(&self, other: &Chunk) -> bool {
        // Branch-free reduction: the compiler vectorises the 32-wide compare.
        self.e
            .iter()
            .zip(other.e.iter())
            .fold(true, |all, (l, r)| all & (l >= r))
    }
}

enum ChunkJoin {
    Current,
    Incoming,
    New(Arc<Chunk>),
}

/// Memo of incomparable chunk joins, keyed by `(id_a, id_b)`.
#[derive(Default)]
pub struct JoinMemo {
    map: RefCell<HashMap<(u64, u64), Weak<Chunk>>>,
    inserts: Cell<u32>,
    pub hits: Cell<u64>,
    pub misses: Cell<u64>,
}

const MEMO_PRUNE_EVERY: u32 = 1 << 10;

impl JoinMemo {
    fn join(&self, cur: &Arc<Chunk>, inc: &Arc<Chunk>) -> ChunkJoin {
        if Arc::ptr_eq(cur, inc) || cur.dominates(inc) {
            return ChunkJoin::Current;
        }
        if inc.dominates(cur) {
            return ChunkJoin::Incoming;
        }
        let key = (cur.id, inc.id);
        if let Some(hit) = self.map.borrow().get(&key).and_then(Weak::upgrade) {
            self.hits.set(self.hits.get() + 1);
            return ChunkJoin::New(hit);
        }
        self.misses.set(self.misses.get() + 1);
        let mut e = cur.e;
        for (s, i) in e.iter_mut().zip(inc.e.iter()) {
            *s = (*s).max(*i);
        }
        let chunk = Chunk::new(e);
        let mut map = self.map.borrow_mut();
        map.insert(key, Arc::downgrade(&chunk));
        let n = self.inserts.get() + 1;
        if n >= MEMO_PRUNE_EVERY {
            // A dead Weak still pins the chunk allocation; prune on a cadence.
            map.retain(|_, w| w.strong_count() != 0);
            self.inserts.set(0);
        } else {
            self.inserts.set(n);
        }
        ChunkJoin::New(chunk)
    }
}

/// Scalar components in shared immutable chunks. `None` chunk = zeros.
#[derive(Clone, Debug, Default)]
pub struct Epochs {
    pub(crate) chunks: Option<Arc<[Option<Arc<Chunk>>]>>,
}

impl Epochs {
    #[inline(always)]
    pub fn get(&self, actor: ActorId) -> Epoch {
        let i = actor as usize;
        match &self.chunks {
            Some(c) => c
                .get(i / CHUNK)
                .and_then(|c| c.as_ref())
                .map_or(0, |c| c.e[i % CHUNK]),
            None => 0,
        }
    }

    pub fn ptr_eq(&self, other: &Self) -> bool {
        match (&self.chunks, &other.chunks) {
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            (None, None) => true,
            _ => false,
        }
    }

    pub fn raise(&mut self, actor: ActorId, epoch: Epoch) {
        if self.get(actor) >= epoch {
            return;
        }
        let i = actor as usize;
        let ci = i / CHUNK;
        let mut v: Vec<Option<Arc<Chunk>>> = self.chunks.as_deref().map(<[_]>::to_vec).unwrap_or_default();
        if v.len() <= ci {
            v.resize(ci + 1, None);
        }
        let mut e = v[ci].as_ref().map_or([0; CHUNK], |c| c.e);
        e[i % CHUNK] = epoch;
        v[ci] = Some(Chunk::new(e));
        self.chunks = Some(v.into());
    }

    /// `self ⊔= other`. Returns whether `self` changed.
    pub fn join(&mut self, other: &Self, memo: &JoinMemo) -> bool {
        if self.ptr_eq(other) {
            return false;
        }
        let Some(inc) = &other.chunks else { return false };
        let Some(cur) = &self.chunks else {
            self.chunks = Some(inc.clone());
            return true;
        };
        // Fast path: the incoming side dominates chunk-wise (the barrier
        // fan-in steady state). Adopt its slice outright, so synchronised
        // clocks converge on one storage and later joins are pointer-equal.
        if inc.len() >= cur.len()
            && cur.iter().zip(inc.iter()).all(|(c, i)| match (c, i) {
                (None, _) => true,
                (Some(_), None) => false,
                (Some(c), Some(i)) => Arc::ptr_eq(c, i) || i.dominates(c),
            })
        {
            self.chunks = Some(inc.clone());
            return true;
        }
        let mut out: Option<Vec<Option<Arc<Chunk>>>> = None;
        let n = cur.len().max(inc.len());
        for ci in 0..n {
            let c = cur.get(ci).and_then(|c| c.as_ref());
            let i = inc.get(ci).and_then(|c| c.as_ref());
            let replacement = match (c, i) {
                (_, None) => None,
                (None, Some(i)) => Some(i.clone()),
                (Some(c), Some(i)) => match memo.join(c, i) {
                    ChunkJoin::Current => None,
                    ChunkJoin::Incoming => Some(i.clone()),
                    ChunkJoin::New(n) => Some(n),
                },
            };
            if let Some(r) = replacement {
                let v = out.get_or_insert_with(|| {
                    let mut v = cur.to_vec();
                    v.resize(n, None);
                    v
                });
                v[ci] = Some(r);
            }
        }
        if let Some(v) = out {
            if inc.len() >= cur.len() && v.iter().zip(inc.iter()).all(|(a, b)| opt_ptr_eq(a, b)) {
                // Fully adopted the incoming storage: share its slice too.
                self.chunks = Some(inc.clone());
            } else {
                self.chunks = Some(v.into());
            }
            true
        } else {
            if inc.len() > cur.len() {
                let mut v = cur.to_vec();
                v.resize(inc.len(), None);
                self.chunks = Some(v.into());
            }
            false
        }
    }

    /// `self ⊑ other` (every component covered).
    pub fn leq(&self, other: &Self, memo: &JoinMemo) -> bool {
        if self.ptr_eq(other) {
            return true;
        }
        let Some(cur) = &self.chunks else { return true };
        cur.iter().enumerate().all(|(ci, own)| {
            let Some(own) = own else { return true };
            match other.chunks.as_ref().and_then(|o| o.get(ci)).and_then(|c| c.as_ref()) {
                Some(theirs) => matches!(memo.join(theirs, own), ChunkJoin::Current),
                None => own.e.iter().all(|e| *e == 0),
            }
        })
    }

    pub fn nonzero(&self) -> impl Iterator<Item = (ActorId, Epoch)> + '_ {
        self.chunks.iter().flat_map(|c| c.iter().enumerate()).filter_map(|(ci, c)| c.as_ref().map(|c| (ci, c))).flat_map(
            |(ci, c)| {
                c.e.iter()
                    .enumerate()
                    .filter(|(_, e)| **e != 0)
                    .map(move |(s, e)| ((ci * CHUNK + s) as ActorId, *e))
            },
        )
    }
}

fn opt_ptr_eq(a: &Option<Arc<Chunk>>, b: &Option<Arc<Chunk>>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
        (None, None) => true,
        _ => false,
    }
}

/// Sparse per-lane components for warps that released from a lane subset.
/// Sorted by actor; an entry dominated by the scalar component is dropped.
pub type LaneVec = [Epoch; 32];
pub type LaneEntries = Vec<(ActorId, LaneVec)>;

/// A vector clock: scalar components plus sparse lane-precise warp entries.
#[derive(Clone, Debug, Default)]
pub struct Clock {
    pub epochs: Epochs,
    pub lanes: Option<Arc<LaneEntries>>,
}

impl Clock {
    /// No component at all (never joined or raised).
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.epochs.chunks.is_none() && self.lanes.is_none()
    }

    #[inline(always)]
    pub fn get(&self, actor: ActorId) -> Epoch {
        self.epochs.get(actor)
    }

    /// The single-component happens-before test for a packed stamp. `lane` is
    /// the lane of the prior access (0 for async actors).
    #[inline(always)]
    pub fn observes(&self, stamp: Stamp, lane: u8) -> bool {
        let (a, e) = (stamp.actor(), stamp.epoch());
        if self.epochs.get(a) >= e {
            return true;
        }
        match &self.lanes {
            None => false,
            Some(l) => match l.binary_search_by_key(&a, |(x, _)| *x) {
                Ok(i) => l[i].1[lane as usize] >= e,
                Err(_) => false,
            },
        }
    }

    pub fn raise(&mut self, actor: ActorId, epoch: Epoch) {
        self.epochs.raise(actor, epoch);
        self.normalize_actor(actor);
    }

    /// Raise a warp's lane vector (lane-precise knowledge).
    pub fn raise_lanes(&mut self, actor: ActorId, v: &LaneVec) {
        let min = *v.iter().min().unwrap();
        if min > 0 {
            self.epochs.raise(actor, min);
        }
        let scalar = self.epochs.get(actor);
        if v.iter().all(|e| *e <= scalar) {
            self.normalize_actor(actor);
            return;
        }
        let entries = Arc::make_mut(self.lanes.get_or_insert_with(Default::default));
        match entries.binary_search_by_key(&actor, |(x, _)| *x) {
            Ok(i) => {
                for (s, n) in entries[i].1.iter_mut().zip(v.iter()) {
                    *s = (*s).max(*n);
                }
            }
            Err(i) => entries.insert(i, (actor, *v)),
        }
        self.normalize_actor(actor);
    }

    /// Dominated-entry GC: drop a lane entry the scalar covers.
    fn normalize_actor(&mut self, actor: ActorId) {
        let scalar = self.epochs.get(actor);
        if let Some(l) = &mut self.lanes {
            if let Ok(i) = l.binary_search_by_key(&actor, |(x, _)| *x) {
                if l[i].1.iter().all(|e| *e <= scalar) {
                    let l = Arc::make_mut(l);
                    l.remove(i);
                }
            }
            if l.is_empty() {
                self.lanes = None;
            }
        }
    }

    pub fn join(&mut self, other: &Clock, memo: &JoinMemo) -> bool {
        let mut changed = self.epochs.join(&other.epochs, memo);
        if let Some(ol) = &other.lanes {
            let same = matches!(&self.lanes, Some(sl) if Arc::ptr_eq(sl, ol));
            if !same {
                for (a, v) in ol.iter() {
                    let before = self.lanes.clone();
                    self.raise_lanes(*a, v);
                    changed |= !lanes_ptr_eq(&before, &self.lanes);
                }
            }
        }
        if changed {
            // Scalar growth can dominate lane entries.
            if let Some(l) = self.lanes.clone() {
                for (a, _) in l.iter() {
                    self.normalize_actor(*a);
                }
            }
        }
        changed
    }

    /// Component-wise minimum (lane entries dropped: a lower bound). Used by
    /// the dominated-frontier GC; O(nonzero components).
    pub fn meet(&self, other: &Clock) -> Clock {
        let mut out = Clock::default();
        if self.epochs.ptr_eq(&other.epochs) {
            out.epochs = self.epochs.clone();
            return out;
        }
        for (a, e) in self.epochs.nonzero() {
            let m = e.min(other.get(a));
            if m > 0 {
                out.epochs.raise(a, m);
            }
        }
        out
    }

    pub fn leq(&self, other: &Clock, memo: &JoinMemo) -> bool {
        if !self.epochs.leq(&other.epochs, memo) {
            return false;
        }
        match &self.lanes {
            None => true,
            Some(l) => l.iter().all(|(a, v)| {
                v.iter().enumerate().all(|(lane, e)| other.observes(Stamp::new(*a, *e), lane as u8) || *e == 0)
            }),
        }
    }
}

fn lanes_ptr_eq(a: &Option<Arc<LaneEntries>>, b: &Option<Arc<LaneEntries>>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
        (None, None) => true,
        _ => false,
    }
}
