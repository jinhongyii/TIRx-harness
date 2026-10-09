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

/// Process-global chunk identities: a checker may migrate between threads
/// (pyo3 `Send`), so ids must never repeat across threads (review R6).
static NEXT_CHUNK_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

impl Chunk {
    fn new(e: [Epoch; CHUNK]) -> Arc<Self> {
        let id = NEXT_CHUNK_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
        let memo = super::tuning::on(&super::tuning::JOIN_MEMO);
        if Arc::ptr_eq(cur, inc) || cur.dominates(inc) {
            return ChunkJoin::Current;
        }
        if inc.dominates(cur) {
            return ChunkJoin::Incoming;
        }
        let key = (cur.id, inc.id);
        if !memo {
            let mut e = cur.e;
            for (s, i) in e.iter_mut().zip(inc.e.iter()) {
                *s = (*s).max(*i);
            }
            return ChunkJoin::New(Chunk::new(e));
        }
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
        // A dead Weak still pins the chunk allocation; prune on a cadence
        // proportional to the map (amortised O(1) per insert: a fixed
        // cadence made every prune O(map) and the run quadratic once the
        // live map outgrew it — mega_moe medium).
        if n >= MEMO_PRUNE_EVERY.max((map.len() / 2) as u32) {
            map.retain(|_, w| w.strong_count() != 0);
            self.inserts.set(0);
        } else {
            self.inserts.set(n);
        }
        ChunkJoin::New(chunk)
    }
}

/// Chunks per group: the second level of [`Epochs`].
pub const GROUP: usize = 16;

/// `GROUP` chunk slots (`None` = zeros), shared between clocks like chunks.
pub type Group = [Option<Arc<Chunk>>; GROUP];

/// Scalar components in shared immutable chunks, two levels deep: a slice of
/// shared groups of shared chunks. `None` group or chunk = zeros. With ~10K
/// actors (persistent kernels over 148 CTAs with many async ops) a flat
/// chunk slice made every join and every copy-on-change O(all chunks); with
/// groups both skip pointer-equal groups and copy one group plus the top
/// slice (O(chunks / GROUP + GROUP)).
#[derive(Clone, Debug, Default)]
pub struct Epochs {
    pub(crate) groups: Option<Arc<[Option<Arc<Group>>]>>,
}

const EMPTY_GROUP: Group = [const { None }; GROUP];

fn group_dominates(i: &Group, c: &Group) -> bool {
    c.iter().zip(i.iter()).all(|(c, i)| match (c, i) {
        (None, _) => true,
        (Some(_), None) => false,
        (Some(c), Some(i)) => Arc::ptr_eq(c, i) || i.dominates(c),
    })
}

fn opt_arc_eq<T>(a: &Option<Arc<T>>, b: &Option<Arc<T>>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
        (None, None) => true,
        _ => false,
    }
}

/// `c ⊔ i` for one group; `None` = `c` unchanged.
fn join_group(c: &Arc<Group>, i: &Arc<Group>, memo: &JoinMemo) -> Option<Arc<Group>> {
    let mut out: Option<Group> = None;
    for s in 0..GROUP {
        let replacement = match (&c[s], &i[s]) {
            (_, None) => None,
            (None, Some(i)) => Some(i.clone()),
            (Some(c), Some(i)) => match memo.join(c, i) {
                ChunkJoin::Current => None,
                ChunkJoin::Incoming => Some(i.clone()),
                ChunkJoin::New(n) => Some(n),
            },
        };
        if let Some(r) = replacement {
            out.get_or_insert_with(|| (**c).clone())[s] = Some(r);
        }
    }
    let g = out?;
    if g.iter().zip(i.iter()).all(|(a, b)| opt_arc_eq(a, b)) {
        Some(i.clone())
    } else {
        Some(Arc::new(g))
    }
}

impl Epochs {
    #[inline(always)]
    pub fn get(&self, actor: ActorId) -> Epoch {
        let i = actor as usize;
        let ci = i / CHUNK;
        match &self.groups {
            Some(g) => g
                .get(ci / GROUP)
                .and_then(|g| g.as_ref())
                .and_then(|g| g[ci % GROUP].as_ref())
                .map_or(0, |c| c.e[i % CHUNK]),
            None => 0,
        }
    }

    pub fn ptr_eq(&self, other: &Self) -> bool {
        match (&self.groups, &other.groups) {
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            (None, None) => true,
            _ => false,
        }
    }

    /// Every chunk slot in flat order: `(chunk index, chunk)`.
    fn chunk_slots(&self) -> impl Iterator<Item = (usize, &Option<Arc<Chunk>>)> + '_ {
        self.groups.iter().flat_map(|t| t.iter().enumerate()).flat_map(|(gi, g)| {
            g.iter().flat_map(|g| g.iter()).enumerate().map(move |(s, c)| (gi * GROUP + s, c))
        })
    }

    fn chunk(&self, ci: usize) -> Option<&Arc<Chunk>> {
        self.groups.as_ref()?.get(ci / GROUP)?.as_ref()?[ci % GROUP].as_ref()
    }

    pub fn raise(&mut self, actor: ActorId, epoch: Epoch) {
        if self.get(actor) >= epoch {
            return;
        }
        let i = actor as usize;
        let ci = i / CHUNK;
        let gi = ci / GROUP;
        let mut top: Vec<Option<Arc<Group>>> = self.groups.as_deref().map(<[_]>::to_vec).unwrap_or_default();
        if top.len() <= gi {
            top.resize(gi + 1, None);
        }
        let mut g: Group = top[gi].as_deref().cloned().unwrap_or(EMPTY_GROUP);
        let mut e = g[ci % GROUP].as_ref().map_or([0; CHUNK], |c| c.e);
        e[i % CHUNK] = epoch;
        g[ci % GROUP] = Some(Chunk::new(e));
        top[gi] = Some(Arc::new(g));
        self.groups = Some(top.into());
    }

    /// `self ⊔= other`. Returns whether `self` changed.
    pub fn join(&mut self, other: &Self, memo: &JoinMemo) -> bool {
        if self.ptr_eq(other) {
            return false;
        }
        let Some(inc) = &other.groups else { return false };
        let Some(cur) = &self.groups else {
            self.groups = Some(inc.clone());
            return true;
        };
        // Fast path: the incoming side dominates group-wise (the barrier
        // fan-in steady state). Adopt its slice outright, so synchronised
        // clocks converge on one storage and later joins are pointer-equal.
        if super::tuning::on(&super::tuning::JOIN_MEMO)
            && inc.len() >= cur.len()
            && cur.iter().zip(inc.iter()).all(|(c, i)| match (c, i) {
                (None, _) => true,
                (Some(_), None) => false,
                (Some(c), Some(i)) => Arc::ptr_eq(c, i) || group_dominates(i, c),
            })
        {
            self.groups = Some(inc.clone());
            return true;
        }
        let mut out: Option<Vec<Option<Arc<Group>>>> = None;
        let n = cur.len().max(inc.len());
        for gi in 0..n {
            let c = cur.get(gi).and_then(|c| c.as_ref());
            let i = inc.get(gi).and_then(|c| c.as_ref());
            let replacement = match (c, i) {
                (_, None) => None,
                (None, Some(i)) => Some(i.clone()),
                (Some(c), Some(i)) if Arc::ptr_eq(c, i) => None,
                (Some(c), Some(i)) => join_group(c, i, memo),
            };
            if let Some(r) = replacement {
                let v = out.get_or_insert_with(|| {
                    let mut v = cur.to_vec();
                    v.resize(n, None);
                    v
                });
                v[gi] = Some(r);
            }
        }
        if let Some(v) = out {
            if inc.len() >= cur.len() && v.iter().zip(inc.iter()).all(|(a, b)| opt_arc_eq(a, b)) {
                // Fully adopted the incoming storage: share its slice too.
                self.groups = Some(inc.clone());
            } else {
                self.groups = Some(v.into());
            }
            true
        } else {
            if inc.len() > cur.len() {
                let mut v = cur.to_vec();
                v.resize(inc.len(), None);
                self.groups = Some(v.into());
            }
            false
        }
    }

    /// Chunk-wise rebuild: `f(actor, epoch)` per slot, one allocation per
    /// touched chunk (O(chunks · CHUNK), never per-component `raise`).
    pub fn map_chunks(&self, mut f: impl FnMut(ActorId, Epoch) -> Epoch) -> Epochs {
        let Some(top) = &self.groups else { return Epochs::default() };
        let v: Vec<Option<Arc<Group>>> = top
            .iter()
            .enumerate()
            .map(|(gi, g)| {
                let g = g.as_ref()?;
                let mut out = EMPTY_GROUP;
                let mut same = true;
                for (s, c) in g.iter().enumerate() {
                    let ci = gi * GROUP + s;
                    out[s] = c.as_ref().and_then(|c| {
                        let mut e = c.e;
                        let mut any = false;
                        for (k, x) in e.iter_mut().enumerate() {
                            if *x != 0 {
                                *x = f((ci * CHUNK + k) as ActorId, *x);
                                any |= *x != 0;
                            }
                        }
                        if e == c.e {
                            Some(c.clone())
                        } else if any {
                            Some(Chunk::new(e))
                        } else {
                            None
                        }
                    });
                    same &= opt_arc_eq(&out[s], c);
                }
                if same {
                    Some(g.clone())
                } else if out.iter().any(|c| c.is_some()) {
                    Some(Arc::new(out))
                } else {
                    None
                }
            })
            .collect();
        Epochs { groups: Some(v.into()) }
    }

    /// Component-wise minimum, chunk by chunk.
    pub fn meet(&self, other: &Epochs) -> Epochs {
        let (Some(a), Some(b)) = (&self.groups, &other.groups) else { return Epochs::default() };
        let v: Vec<Option<Arc<Group>>> = a
            .iter()
            .zip(b.iter())
            .map(|(gx, gy)| match (gx, gy) {
                (Some(gx), Some(gy)) if Arc::ptr_eq(gx, gy) => Some(gx.clone()),
                (Some(gx), Some(gy)) => {
                    let mut out = EMPTY_GROUP;
                    for s in 0..GROUP {
                        out[s] = match (&gx[s], &gy[s]) {
                            (Some(x), Some(y)) if Arc::ptr_eq(x, y) || y.dominates(x) => Some(x.clone()),
                            (Some(x), Some(y)) if x.dominates(y) => Some(y.clone()),
                            (Some(x), Some(y)) => {
                                let mut e = x.e;
                                for (s, o) in e.iter_mut().zip(y.e.iter()) {
                                    *s = (*s).min(*o);
                                }
                                Some(Chunk::new(e))
                            }
                            _ => None,
                        };
                    }
                    Some(Arc::new(out))
                }
                _ => None,
            })
            .collect();
        Epochs { groups: Some(v.into()) }
    }

    /// `self ⊑ other` (every component covered).
    pub fn leq(&self, other: &Self, memo: &JoinMemo) -> bool {
        if self.ptr_eq(other) {
            return true;
        }
        let _ = memo;
        let Some(cur) = &self.groups else { return true };
        cur.iter().enumerate().all(|(gi, g)| {
            let Some(g) = g else { return true };
            if let Some(Some(theirs)) = other.groups.as_ref().and_then(|o| o.get(gi)) {
                if Arc::ptr_eq(theirs, g) {
                    return true;
                }
            }
            // Pointer equality, then a direct dominance test: a comparison
            // never allocates or touches the memo.
            g.iter().enumerate().all(|(s, own)| {
                let Some(own) = own else { return true };
                match other.chunk(gi * GROUP + s) {
                    Some(theirs) => Arc::ptr_eq(theirs, own) || theirs.dominates(own),
                    None => own.e.iter().all(|e| *e == 0),
                }
            })
        })
    }

    pub fn nonzero(&self) -> impl Iterator<Item = (ActorId, Epoch)> + '_ {
        self.chunk_slots().filter_map(|(ci, c)| c.as_ref().map(|c| (ci, c))).flat_map(|(ci, c)| {
            c.e.iter()
                .enumerate()
                .filter(|(_, e)| **e != 0)
                .map(move |(s, e)| ((ci * CHUNK + s) as ActorId, *e))
        })
    }
}

/// Sparse per-lane components for warps that released from a lane subset.
/// An entry dominated by the scalar component is dropped.
pub type LaneVec = [Epoch; 32];

/// log2 of the actors per lane block. A block covers exactly the actors of
/// one scalar chunk, so a block's entries can only become dominated when that
/// chunk changes.
const LBLK_SHIFT: u32 = 5;
const _: () = assert!(1 << LBLK_SHIFT == CHUNK);

/// Lane blocks per lane group: the second level of [`Lanes`].
const LGROUP: usize = 16;

/// One warp's lane-precise components, stored inline with their maximum.
///
/// The vector is a plain value: a block rebuild copies entries instead of
/// bumping a shared reference count. Lane entries are learned from release
/// payloads that every checker partition of a phase reads, so a shared
/// count turned each rebuild into cross-core cache-line traffic (mega_moe
/// e24: 59% of the children's time at 16 workers was lane code, against 36%
/// at one worker). `max` makes the "dominated by the scalar" test O(1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LaneEntry {
    actor: ActorId,
    max: Epoch,
    v: LaneVec,
}

impl LaneEntry {
    #[inline]
    fn new(actor: ActorId, v: LaneVec) -> LaneEntry {
        LaneEntry { actor, max: *v.iter().max().unwrap(), v }
    }
}

/// One block's entries (actors `bi << LBLK_SHIFT ..`), sorted by actor.
type LaneBlock = Vec<LaneEntry>;

/// `LGROUP` block slots (`None` = no entries), shared between clocks.
type LaneGroup = [Option<Arc<LaneBlock>>; LGROUP];

const EMPTY_LANE_GROUP: LaneGroup = [const { None }; LGROUP];

/// Lane entries in shared blocks of `1 << LBLK_SHIFT` actors, two levels deep
/// like [`Epochs`]. With hundreds of lane-precise warps (elected-lane arrives
/// and releases in persistent kernels) a join that changes one block copies
/// one group and the top slice, never every block; a join skips
/// pointer-equal groups and blocks.
#[derive(Clone, Debug)]
pub struct Lanes {
    groups: Arc<[Option<Arc<LaneGroup>>]>,
}

impl Lanes {
    #[inline]
    fn block(&self, bi: usize) -> Option<&Arc<LaneBlock>> {
        self.groups.get(bi / LGROUP)?.as_ref()?[bi % LGROUP].as_ref()
    }

    #[inline]
    fn entry(&self, actor: ActorId) -> Option<&LaneEntry> {
        let b = self.block((actor >> LBLK_SHIFT) as usize)?;
        b.iter().find(|e| e.actor == actor)
    }

    #[inline]
    pub fn get(&self, actor: ActorId) -> Option<&LaneVec> {
        self.entry(actor).map(|e| &e.v)
    }

    pub fn iter(&self) -> impl Iterator<Item = (ActorId, &LaneVec)> + '_ {
        self.blocks().flat_map(|(_, b)| b.iter().map(|e| (e.actor, &e.v)))
    }

    /// Every non-empty block: `(block index, block)`.
    fn blocks(&self) -> impl Iterator<Item = (usize, &Arc<LaneBlock>)> + '_ {
        self.groups
            .iter()
            .enumerate()
            .filter_map(|(gi, g)| g.as_ref().map(|g| (gi, g)))
            .flat_map(|(gi, g)| g.iter().enumerate().filter_map(move |(s, b)| b.as_ref().map(|b| (gi * LGROUP + s, b))))
    }

    pub fn len(&self) -> usize {
        self.blocks().map(|(_, b)| b.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks().next().is_none()
    }

    fn ptr_eq(&self, o: &Lanes) -> bool {
        Arc::ptr_eq(&self.groups, &o.groups)
    }

    /// Replace blocks (`repl` sorted by block index, each at most once);
    /// `None` result = no entries left.
    fn with_blocks(cur: Option<&Lanes>, repl: Vec<(usize, Option<Arc<LaneBlock>>)>) -> Option<Lanes> {
        let mut top: Vec<Option<Arc<LaneGroup>>> = cur.map(|l| l.groups.to_vec()).unwrap_or_default();
        let mut i = 0;
        while i < repl.len() {
            let gi = repl[i].0 / LGROUP;
            if top.len() <= gi {
                top.resize(gi + 1, None);
            }
            let mut g: LaneGroup = top[gi].as_deref().cloned().unwrap_or(EMPTY_LANE_GROUP);
            while i < repl.len() && repl[i].0 / LGROUP == gi {
                let (bi, b) = &repl[i];
                g[bi % LGROUP] = b.clone().filter(|b| !b.is_empty());
                i += 1;
            }
            top[gi] = g.iter().any(Option::is_some).then(|| Arc::new(g));
        }
        while matches!(top.last(), Some(None)) {
            top.pop();
        }
        (!top.is_empty()).then(|| Lanes { groups: top.into() })
    }
}

/// A vector clock: scalar components plus sparse lane-precise warp entries.
#[derive(Clone, Debug, Default)]
pub struct Clock {
    pub epochs: Epochs,
    pub lanes: Option<Lanes>,
}

impl Clock {
    /// No component at all (never joined or raised).
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.epochs.groups.is_none() && self.lanes.is_none()
    }

    #[inline(always)]
    pub fn get(&self, actor: ActorId) -> Epoch {
        self.epochs.get(actor)
    }

    /// Same storage (so equal), without comparing components.
    pub fn ptr_eq(&self, o: &Clock) -> bool {
        self.epochs.ptr_eq(&o.epochs)
            && match (&self.lanes, &o.lanes) {
                (None, None) => true,
                (Some(a), Some(b)) => a.ptr_eq(b),
                _ => false,
            }
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
            Some(l) => l.get(a).is_some_and(|v| v[lane as usize] >= e),
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
        let bi = (actor >> LBLK_SHIFT) as usize;
        let mut b: LaneBlock = self.lanes.as_ref().and_then(|l| l.block(bi)).map(|b| (**b).clone()).unwrap_or_default();
        match b.binary_search_by_key(&actor, |e| e.actor) {
            Ok(i) => {
                let mut m = b[i].v;
                for (s, n) in m.iter_mut().zip(v.iter()) {
                    *s = (*s).max(*n);
                }
                b[i] = LaneEntry::new(actor, m);
                if b[i].max <= scalar {
                    b.remove(i);
                }
            }
            Err(i) => b.insert(i, LaneEntry::new(actor, *v)),
        }
        self.lanes = Lanes::with_blocks(self.lanes.as_ref(), vec![(bi, Some(Arc::new(b)))]);
    }

    /// Dominated-entry GC: drop a lane entry the scalar covers.
    fn normalize_actor(&mut self, actor: ActorId) {
        let scalar = self.epochs.get(actor);
        let Some(l) = &self.lanes else { return };
        let bi = (actor >> LBLK_SHIFT) as usize;
        let Some(b) = l.block(bi) else { return };
        if let Ok(i) = b.binary_search_by_key(&actor, |e| e.actor) {
            if b[i].max <= scalar {
                let mut nb = (**b).clone();
                nb.remove(i);
                self.lanes = Lanes::with_blocks(Some(l), vec![(bi, Some(Arc::new(nb)))]);
            }
        }
    }

    /// Does this clock already cover incoming entry `n` (given its own entry
    /// for the actor, if any)?
    #[inline]
    fn covers_entry(&self, n: &LaneEntry, own: Option<&LaneEntry>) -> bool {
        let scalar = self.epochs.get(n.actor);
        if n.max <= scalar {
            return true;
        }
        match own {
            Some(o) if o.v == n.v => true,
            Some(o) => n.v.iter().zip(o.v.iter()).all(|(e, m)| *e <= scalar || *e <= *m),
            None => false,
        }
    }

    pub fn join(&mut self, other: &Clock, memo: &JoinMemo) -> bool {
        // The scalar slice before the join (one reference), so the
        // normalisation below revisits only blocks whose chunk changed.
        let before = if self.lanes.is_some() { self.epochs.groups.clone() } else { None };
        let scalar_changed = self.epochs.join(&other.epochs, memo);
        let mut changed = scalar_changed;
        if let Some(ol) = &other.lanes {
            let same = matches!(&self.lanes, Some(sl) if sl.ptr_eq(ol));
            if !same {
                let mut repl: Vec<(usize, Option<Arc<LaneBlock>>)> = Vec::new();
                for (gi, og) in ol.groups.iter().enumerate() {
                    let Some(og) = og else { continue };
                    let mg = self.lanes.as_ref().and_then(|l| l.groups.get(gi)).and_then(|g| g.as_ref());
                    if mg.is_some_and(|mg| Arc::ptr_eq(mg, og)) {
                        continue;
                    }
                    for (s, ob) in og.iter().enumerate() {
                        let Some(ob) = ob else { continue };
                        let bi = gi * LGROUP + s;
                        let mb = mg.and_then(|g| g[s].as_ref());
                        if mb.is_some_and(|mb| Arc::ptr_eq(mb, ob)) {
                            continue;
                        }
                        let mine: &[LaneEntry] = mb.map_or(&[], |b| b.as_slice());
                        // Which incoming entries this clock does not cover
                        // yet: one merge walk, one scalar lookup per actor.
                        let mut j = 0;
                        let mut any = false;
                        for n in ob.iter() {
                            while j < mine.len() && mine[j].actor < n.actor {
                                j += 1;
                            }
                            let own = (j < mine.len() && mine[j].actor == n.actor).then(|| &mine[j]);
                            if !self.covers_entry(n, own) {
                                any = true;
                                break;
                            }
                        }
                        if !any {
                            continue;
                        }
                        // Rebuild this block: merge, raise scalars to each
                        // new vector's minimum (as `raise_lanes`), drop
                        // dominated entries.
                        let mut out: LaneBlock = Vec::with_capacity(mine.len() + ob.len());
                        let mut j = 0;
                        for n in ob.iter() {
                            while j < mine.len() && mine[j].actor < n.actor {
                                out.push(mine[j]);
                                j += 1;
                            }
                            let own = (j < mine.len() && mine[j].actor == n.actor).then(|| mine[j]);
                            if own.is_some() {
                                j += 1;
                            }
                            if self.covers_entry(n, own.as_ref()) {
                                if let Some(o) = own {
                                    out.push(o);
                                }
                                continue;
                            }
                            let mut merged = n.v;
                            if let Some(o) = &own {
                                for (m, p) in merged.iter_mut().zip(o.v.iter()) {
                                    *m = (*m).max(*p);
                                }
                            }
                            let min = *n.v.iter().min().unwrap();
                            if min > 0 {
                                self.epochs.raise(n.actor, min);
                            }
                            let scalar = self.epochs.get(n.actor);
                            let e = LaneEntry::new(n.actor, merged);
                            if e.max > scalar {
                                out.push(e);
                            }
                        }
                        out.extend_from_slice(&mine[j..]);
                        changed = true;
                        let nb = if out.as_slice() == ob.as_slice() { ob.clone() } else { Arc::new(out) };
                        repl.push((bi, Some(nb)));
                    }
                }
                if !repl.is_empty() {
                    self.lanes = Lanes::with_blocks(self.lanes.as_ref(), repl);
                }
            }
        }
        if scalar_changed {
            // Scalar growth can dominate lane entries (a lane-only change
            // was normalised by the rebuild itself).
            if let Some(l) = &self.lanes {
                let epochs = &self.epochs;
                let old = Epochs { groups: before };
                let mut repl = Vec::new();
                for (bi, b) in l.blocks() {
                    // Entries are normalised against the scalar before the
                    // join; a block whose chunk (the same 32 actors) is
                    // unchanged cannot have become dominated.
                    let unchanged = match (old.chunk(bi), epochs.chunk(bi)) {
                        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                        (None, None) => true,
                        _ => false,
                    };
                    if unchanged {
                        continue;
                    }
                    if b.iter().any(|e| e.max <= epochs.get(e.actor)) {
                        let nb: LaneBlock = b.iter().filter(|e| e.max > epochs.get(e.actor)).copied().collect();
                        repl.push((bi, Some(Arc::new(nb))));
                    }
                }
                if !repl.is_empty() {
                    self.lanes = Lanes::with_blocks(Some(l), repl);
                }
            }
        }
        changed
    }

    /// The components whose actor satisfies `keep` (lane entries included).
    pub fn filter(&self, mut keep: impl FnMut(ActorId) -> bool) -> Clock {
        let mut out = Clock { epochs: self.epochs.map_chunks(|base, e| if keep(base) { e } else { 0 }), lanes: None };
        if let Some(l) = &self.lanes {
            for (a, v) in l.iter() {
                if keep(a) {
                    out.raise_lanes(a, v);
                }
            }
        }
        out
    }

    /// Component-wise minimum (lane entries dropped: a lower bound). Used by
    /// the dominated-frontier GC; O(nonzero components).
    pub fn meet(&self, other: &Clock) -> Clock {
        if self.epochs.ptr_eq(&other.epochs) {
            return Clock { epochs: self.epochs.clone(), lanes: None };
        }
        Clock { epochs: self.epochs.meet(&other.epochs), lanes: None }
    }

    pub fn leq(&self, other: &Clock, memo: &JoinMemo) -> bool {
        if !self.epochs.leq(&other.epochs, memo) {
            return false;
        }
        match &self.lanes {
            None => true,
            Some(l) => l.iter().all(|(a, v)| {
                v.iter().enumerate().all(|(lane, e)| other.observes(Stamp::new(a, *e), lane as u8) || *e == 0)
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference model: a clock is `K(actor, lane)`, joins are pointwise max.
    #[derive(Clone)]
    struct Model(Vec<[Epoch; 32]>);

    impl Model {
        fn k(&self, a: usize, l: usize) -> Epoch {
            self.0[a][l]
        }
    }

    fn agrees(c: &Clock, m: &Model) {
        for a in 0..m.0.len() {
            for l in 0..32 {
                let k = m.k(a, l);
                let s = |e: Epoch| Stamp::new(a as ActorId, e);
                if k > 0 {
                    assert!(c.observes(s(k), l as u8), "actor {a} lane {l}: misses {k}");
                }
                assert!(!c.observes(s(k + 1), l as u8), "actor {a} lane {l}: claims {}", k + 1);
            }
        }
    }

    /// Random raises, lane raises and joins over actors spanning several
    /// chunks, lane blocks and lane groups, against the pointwise model.
    #[test]
    fn lanes_and_joins_match_the_pointwise_model() {
        const ACTORS: usize = 1100; // > 2 lane groups of 512 actors
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = move |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        let memo = JoinMemo::default();
        let mut clocks: Vec<(Clock, Model)> = (0..6).map(|_| (Clock::default(), Model(vec![[0; 32]; ACTORS]))).collect();
        for step in 0..4000 {
            let i = rnd(clocks.len() as u64) as usize;
            // A few hot actors so entries collide, plus the whole range.
            let a = if rnd(2) == 0 { rnd(40) as usize * 29 % ACTORS } else { rnd(ACTORS as u64) as usize };
            match rnd(4) {
                0 => {
                    let e = rnd(50) as Epoch + 1;
                    let (c, m) = &mut clocks[i];
                    c.raise(a as ActorId, e);
                    for l in 0..32 {
                        m.0[a][l] = m.0[a][l].max(e);
                    }
                }
                1 | 2 => {
                    let mut v = [0; 32];
                    for x in v.iter_mut() {
                        *x = rnd(50) as Epoch;
                    }
                    let (c, m) = &mut clocks[i];
                    c.raise_lanes(a as ActorId, &v);
                    for (x, y) in m.0[a].iter_mut().zip(v) {
                        *x = (*x).max(y);
                    }
                }
                _ => {
                    let j = rnd(clocks.len() as u64) as usize;
                    let (oc, om) = clocks[j].clone();
                    let (c, m) = &mut clocks[i];
                    c.join(&oc, &memo);
                    for (x, y) in m.0.iter_mut().zip(om.0.iter()) {
                        for l in 0..32 {
                            x[l] = x[l].max(y[l]);
                        }
                    }
                }
            }
            if step % 500 == 499 {
                for (c, m) in &clocks {
                    agrees(c, m);
                }
            }
        }
        for (c, m) in &clocks {
            agrees(c, m);
            // No entry the scalar already covers is kept.
            if let Some(l) = &c.lanes {
                for (a, v) in l.iter() {
                    assert!(v.iter().any(|e| *e > c.get(a)), "dominated entry kept for actor {a}");
                }
            }
        }
    }
}
