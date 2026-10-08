//! Arena: bytes + 1 validity bit per byte + views/OOB. Nothing else.
//!
//! Analysis metadata (shadow cells, clocks) is keyed by `(AllocId, range)`
//! and lives in the checkers, never here (plan 2.4). The arena knows
//! nothing about warps, ordering or proxies.
//!
//! # Allocations
//!
//! * Global: one allocation per host buffer / scratch buffer, owner `Launch`.
//!   Each gets a synthetic virtual address range (see [`addr`]); separate
//!   allocations are separated by an unmapped guard gap so pointer overrun
//!   is an OOB error, never a silent hit on a neighbour.
//! * Shared: one allocation per CTA covering the whole shared window
//!   (static + dynamic), owner `Cta`.
//! * Local: one allocation per warp, lane-major: lane `l` owns bytes
//!   `[l * per_lane, (l + 1) * per_lane)`.
//! * Param: one allocation per launch holding kernel params (tensor maps).
//! * Tmem: one allocation per CTA, 128 lanes x 512 columns x 4 bytes,
//!   lane-major (see [`addr::tmem_byte_offset`]).
//! * Reg: *metadata-only* allocation per warp (no bytes) so async register
//!   writers (tcgen05.ld) can be described as spans to checkers: register
//!   `r` lane `l` is span `[(r * 32 + l) * 8, +8)`.
//!
//! # Validity
//!
//! Each byte has a valid bit. Writes set it; `invalidate` clears it
//! (`discard`, uninitialized allocations). Reading an invalid byte is an
//! [`ArenaError::Uninit`] under [`ValidityPolicy::Error`] (the NumSim
//! default) and allowed under [`ValidityPolicy::Allow`].

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Allocation space (where bytes physically live).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Space {
    Global,
    Shared,
    Local,
    Param,
    Tmem,
    Reg,
}

/// Index of an allocation in the arena (stable for the arena's lifetime).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct AllocId(pub u32);

impl fmt::Display for AllocId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "alloc{}", self.0)
    }
}

/// Who owns an allocation (determines lifetime and visibility).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Owner {
    /// Lives for the whole launch / module run (global, param).
    Launch,
    /// CTA-private (shared window, tmem). Global CTA index.
    Cta(u32),
    /// Warp-private (local, reg). Global warp index.
    Warp(u32),
}

/// A byte range `[start, start + len)` relative to a view (or allocation).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ByteSpan {
    pub start: u64,
    pub len: u64,
}

impl ByteSpan {
    pub const fn new(start: u64, len: u64) -> ByteSpan {
        ByteSpan { start, len }
    }
    pub const fn end(self) -> u64 {
        self.start + self.len
    }
    pub const fn overlaps(self, o: ByteSpan) -> bool {
        self.start < o.end() && o.start < self.end()
    }
    pub const fn shifted(self, by: u64) -> ByteSpan {
        ByteSpan { start: self.start + by, len: self.len }
    }
    /// Sort and merge overlapping/adjacent spans in place (canonical form
    /// used in `observe::Access::spans`).
    pub fn coalesce(spans: &mut Vec<ByteSpan>) {
        spans.retain(|s| s.len > 0);
        spans.sort_unstable();
        let mut out: Vec<ByteSpan> = Vec::with_capacity(spans.len());
        for s in spans.drain(..) {
            match out.last_mut() {
                Some(last) if s.start <= last.end() => {
                    let end = last.end().max(s.end());
                    last.len = end - last.start;
                }
                _ => out.push(s),
            }
        }
        *spans = out;
    }
}

/// A fixed-length bitset (validity bits).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BitSet {
    words: Vec<u64>,
    len: u64,
}

impl BitSet {
    pub fn new(len: u64, value: bool) -> BitSet {
        let n = len.div_ceil(64) as usize;
        let mut b = BitSet { words: vec![if value { u64::MAX } else { 0 }; n], len };
        b.trim();
        b
    }
    fn trim(&mut self) {
        let r = self.len % 64;
        if r != 0 {
            if let Some(w) = self.words.last_mut() {
                *w &= (1u64 << r) - 1;
            }
        }
    }
    pub fn len(&self) -> u64 {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn get(&self, i: u64) -> bool {
        (self.words[(i / 64) as usize] >> (i % 64)) & 1 == 1
    }
    pub fn set_range(&mut self, start: u64, len: u64, value: bool) {
        let mut i = start;
        let end = start + len;
        while i < end {
            let w = (i / 64) as usize;
            let bit = i % 64;
            let n = (64 - bit).min(end - i);
            let m = if n == 64 { u64::MAX } else { ((1u64 << n) - 1) << bit };
            if value {
                self.words[w] |= m;
            } else {
                self.words[w] &= !m;
            }
            i += n;
        }
    }
    /// First index in `[start, start+len)` whose bit is set.
    pub fn first_set(&self, start: u64, len: u64) -> Option<u64> {
        let mut i = start;
        let end = start + len;
        while i < end {
            let w = (i / 64) as usize;
            let bit = i % 64;
            let n = (64 - bit).min(end - i);
            let m = if n == 64 { u64::MAX } else { ((1u64 << n) - 1) << bit };
            let set = self.words[w] & m;
            if set != 0 {
                return Some(w as u64 * 64 + set.trailing_zeros() as u64);
            }
            i += n;
        }
        None
    }

    /// First index in `[start, start+len)` whose bit is clear.
    pub fn first_clear(&self, start: u64, len: u64) -> Option<u64> {
        let mut i = start;
        let end = start + len;
        while i < end {
            let w = (i / 64) as usize;
            let bit = i % 64;
            let n = (64 - bit).min(end - i);
            let m = if n == 64 { u64::MAX } else { ((1u64 << n) - 1) << bit };
            let missing = !self.words[w] & m;
            if missing != 0 {
                return Some(w as u64 * 64 + missing.trailing_zeros() as u64);
            }
            i += n;
        }
        None
    }
}

/// One allocation.
#[derive(Clone, Debug)]
pub struct Allocation {
    pub space: Space,
    pub owner: Owner,
    /// Buffer / role name for reports.
    pub name: String,
    /// Address of byte 0 in its space's address domain (global VA for
    /// `Global`, 0 for the others).
    pub base: u64,
    pub size: u64,
    /// Empty for metadata-only allocations.
    pub bytes: Vec<u8>,
    pub valid: BitSet,
    pub metadata_only: bool,
}

/// A window into an allocation. Spans passed with a view are relative to
/// `offset` and must lie within `len`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct View {
    pub alloc: AllocId,
    pub offset: u64,
    pub len: u64,
}

impl View {
    /// Absolute (allocation-relative) span of a view-relative span.
    pub const fn absolute(self, s: ByteSpan) -> ByteSpan {
        s.shifted(self.offset)
    }
}

/// What reading an invalid (never written / discarded) byte does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ValidityPolicy {
    /// Return [`ArenaError::Uninit`] (NumSim default).
    #[default]
    Error,
    /// Return whatever bytes are stored (zero for fresh allocations).
    Allow,
    /// Read invalid bytes as zero; the engine reports one `UninitRead`
    /// review finding per read range (NumSim-mode default, W8-5).
    ZeroAndReport,
}

/// Initial contents of a new allocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Init {
    /// Zero bytes, all invalid.
    Uninit,
    /// Zero bytes, all valid.
    Zeroed,
    /// Given bytes (length must equal size), all valid.
    Bytes(Vec<u8>),
    /// Given bytes and explicit validity.
    BytesWithValidity(Vec<u8>, BitSet),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArenaError {
    OutOfBounds { alloc: AllocId, span: ByteSpan, size: u64 },
    Uninit { alloc: AllocId, offset: u64 },
    /// An address that resolves to no allocation.
    BadAddress { space: Space, addr: u64 },
    /// Data access to a metadata-only allocation.
    MetadataOnly { alloc: AllocId },
    /// `dst`/`src` length differs from the total span length.
    LengthMismatch { expected: u64, got: u64 },
}

impl fmt::Display for ArenaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArenaError::OutOfBounds { alloc, span, size } => write!(
                f,
                "out-of-bounds access [{}, {}) to {alloc} of {size} bytes",
                span.start,
                span.end()
            ),
            ArenaError::Uninit { alloc, offset } => {
                write!(f, "read of uninitialized byte {offset} of {alloc}")
            }
            ArenaError::BadAddress { space, addr } => write!(f, "{space:?} address {addr:#x} is not mapped"),
            ArenaError::MetadataOnly { alloc } => write!(f, "{alloc} has no bytes"),
            ArenaError::LengthMismatch { expected, got } => {
                write!(f, "buffer length {got} != span total {expected}")
            }
        }
    }
}

impl std::error::Error for ArenaError {}

/// Bytes per copy-on-write stripe of a shard's global-memory overlay.
pub const STRIPE: u64 = 4096;

/// One copy-on-write stripe of a shared allocation inside a shard.
#[derive(Clone, Debug)]
struct Stripe {
    bytes: Vec<u8>,
    valid: BitSet,
    /// Bytes this shard wrote (merged into the base in partition order).
    written: BitSet,
}

/// Shard state (see [`Arena::make_shard`]): the base arena's allocations
/// addressed in place through a raw element pointer. Shared allocations
/// (global, param) are read-only and written through a copy-on-write stripe
/// overlay; private allocations (shared windows, TMEM, local, register
/// metadata) are accessed directly: each belongs to exactly one partition.
#[derive(Clone, Debug)]
struct Shard {
    /// `base.allocs.as_mut_ptr()` and its length. Only individual elements
    /// are ever referenced, never the whole slice.
    allocs: *mut Allocation,
    len: usize,
    /// The base arena's global index (read-only while shards exist).
    global_index: *const Vec<(u64, u64, AllocId)>,
    overlay: HashMap<(AllocId, u64), Stripe>,
}

// SAFETY: the scheduler keeps the base arena alive and does not touch it
// while shards exist (`make_shard` .. `merge_shard`); shards read shared
// allocations and write only their own partition's private allocations,
// which are disjoint across partitions.
unsafe impl Send for Shard {}

/// Is an allocation of this space shared across partitions (read through
/// the overlay inside a shard)?
#[inline]
pub const fn is_shared_space(space: Space) -> bool {
    matches!(space, Space::Global | Space::Param)
}

/// What a span write stores.
enum Put<'a> {
    Bytes(&'a [u8]),
    Fill(u8),
    Invalidate,
    /// Bytes with explicit per-byte validity.
    BytesValid(&'a [u8], &'a [bool]),
}

/// The arena: all allocations of one module run.
///
/// # Shards (CTA parallelism)
///
/// [`Arena::make_shard`] creates a shard for one scheduling partition. The
/// shard reads and writes the partition's private allocations (shared
/// windows, TMEM, local, register metadata) in place, and reads shared
/// allocations (global, param) as of the shard's creation plus its own
/// writes, which go to a copy-on-write [`STRIPE`]-byte overlay.
/// [`Arena::merge_shard`] applies the overlay's written bytes to the base;
/// merging shards in a fixed order gives deterministic results. All public
/// methods behave identically on a shard and on a plain arena, except that
/// [`Arena::get`] returns the base's (pre-overlay) bytes for a shared
/// allocation (use [`Arena::read_raw`]) and [`Arena::get_mut`] /
/// [`Arena::alloc`] are not available for shared allocations.
#[derive(Clone, Debug, Default)]
pub struct Arena {
    allocs: Vec<Allocation>,
    policy: ValidityPolicy,
    next_global_va: u64,
    /// Global allocations sorted by base (bases are monotonically assigned).
    global_index: Vec<(u64, u64, AllocId)>,
    shard: Option<Box<Shard>>,
}

impl Arena {
    pub fn new(policy: ValidityPolicy) -> Arena {
        Arena { allocs: Vec::new(), policy, next_global_va: addr::GLOBAL_VA_BASE, global_index: Vec::new(), shard: None }
    }

    pub fn policy(&self) -> ValidityPolicy {
        self.policy
    }

    pub fn set_policy(&mut self, p: ValidityPolicy) {
        self.policy = p;
    }

    /// Is this arena a shard?
    pub fn is_shard(&self) -> bool {
        self.shard.is_some()
    }

    /// Is `id` read through the overlay (a shared allocation inside a shard)?
    #[inline]
    pub fn is_overlaid(&self, id: AllocId) -> bool {
        self.shard.is_some() && is_shared_space(self.get(id).space)
    }

    /// Create an allocation. Global allocations get a fresh VA range.
    pub fn alloc(&mut self, space: Space, owner: Owner, name: &str, size: u64, init: Init) -> AllocId {
        assert!(self.shard.is_none(), "allocations cannot be created inside a shard");
        let id = AllocId(self.allocs.len() as u32);
        let metadata_only = space == Space::Reg;
        let (bytes, valid) = if metadata_only {
            (Vec::new(), BitSet::new(0, false))
        } else {
            match init {
                Init::Uninit => (vec![0u8; size as usize], BitSet::new(size, false)),
                Init::Zeroed => (vec![0u8; size as usize], BitSet::new(size, true)),
                Init::Bytes(b) => {
                    assert_eq!(b.len() as u64, size, "Init::Bytes length must equal size");
                    (b, BitSet::new(size, true))
                }
                Init::BytesWithValidity(b, v) => {
                    assert_eq!(b.len() as u64, size);
                    assert_eq!(v.len(), size);
                    (b, v)
                }
            }
        };
        let base = if space == Space::Global {
            let base = align_up(self.next_global_va, addr::GLOBAL_ALIGN);
            self.next_global_va = base + size.max(1) + addr::GLOBAL_GUARD;
            self.global_index.push((base, base + size, id));
            base
        } else {
            0
        };
        self.allocs.push(Allocation {
            space,
            owner,
            name: name.to_string(),
            base,
            size,
            bytes,
            valid,
            metadata_only,
        });
        id
    }

    /// The allocation (metadata; for a shared allocation inside a shard the
    /// bytes are the base's, without this shard's writes: use
    /// [`Arena::read_raw`]).
    #[inline]
    pub fn get(&self, id: AllocId) -> &Allocation {
        match &self.shard {
            None => &self.allocs[id.0 as usize],
            Some(sh) => {
                assert!((id.0 as usize) < sh.len, "allocation out of range");
                // SAFETY: see `Shard`; one element, in bounds.
                unsafe { &*sh.allocs.add(id.0 as usize) }
            }
        }
    }

    /// Mutable allocation; inside a shard only for private allocations.
    pub fn get_mut(&mut self, id: AllocId) -> &mut Allocation {
        match &mut self.shard {
            None => &mut self.allocs[id.0 as usize],
            Some(sh) => {
                assert!((id.0 as usize) < sh.len, "allocation out of range");
                // SAFETY: see `Shard`; private allocations belong to this
                // shard's partition only.
                let a = unsafe { &mut *sh.allocs.add(id.0 as usize) };
                assert!(!is_shared_space(a.space), "shared allocations are read-only inside a shard");
                a
            }
        }
    }

    pub fn len(&self) -> usize {
        match &self.shard {
            None => self.allocs.len(),
            Some(sh) => sh.len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = (AllocId, &Allocation)> {
        (0..self.len()).map(move |i| (AllocId(i as u32), self.get(AllocId(i as u32))))
    }

    /// A view of the whole allocation.
    pub fn view(&self, id: AllocId) -> View {
        View { alloc: id, offset: 0, len: self.get(id).size }
    }

    /// A sub-view; errors if it exceeds the allocation.
    pub fn subview(&self, id: AllocId, offset: u64, len: u64) -> Result<View, ArenaError> {
        let size = self.get(id).size;
        if offset.checked_add(len).is_none_or(|e| e > size) {
            return Err(ArenaError::OutOfBounds { alloc: id, span: ByteSpan::new(offset, len), size });
        }
        Ok(View { alloc: id, offset, len })
    }

    /// Check every span lies inside the view and the view inside its allocation.
    pub fn check_oob(&self, view: View, spans: &[ByteSpan]) -> Result<(), ArenaError> {
        let size = self.get(view.alloc).size;
        let view_end = view.offset.checked_add(view.len);
        if view_end.is_none_or(|e| e > size) {
            return Err(ArenaError::OutOfBounds {
                alloc: view.alloc,
                span: ByteSpan::new(view.offset, view.len),
                size,
            });
        }
        for s in spans {
            if s.start.checked_add(s.len).is_none_or(|e| e > view.len) {
                return Err(ArenaError::OutOfBounds { alloc: view.alloc, span: view.absolute(*s), size });
            }
        }
        Ok(())
    }

    fn total(spans: &[ByteSpan]) -> u64 {
        spans.iter().map(|s| s.len).sum()
    }

    /// Visit `abs` of `id` as `(bytes, valid, offset_in_alloc)` pieces,
    /// overlay-aware.
    fn pieces(&self, id: AllocId, abs: ByteSpan, mut f: impl FnMut(&[u8], &BitSet, u64, u64, u64)) {
        // f(bytes, valid, index_of_first_byte_in_bytes/valid, alloc_offset, len)
        if let Some(sh) = self.shard.as_ref().filter(|_| self.is_overlaid(id)) {
            let base = self.get(id);
            let mut i = abs.start;
            while i < abs.end() {
                let stripe = i / STRIPE;
                let lo = i % STRIPE;
                let n = (STRIPE - lo).min(abs.end() - i);
                match sh.overlay.get(&(id, stripe)) {
                    Some(st) => f(&st.bytes, &st.valid, lo, i, n),
                    None => f(&base.bytes, &base.valid, i, i, n),
                }
                i += n;
            }
        } else {
            let a = self.get(id);
            f(&a.bytes, &a.valid, abs.start, abs.start, abs.len);
        }
    }

    /// First invalid byte of `abs` of `id`.
    fn first_clear(&self, id: AllocId, abs: ByteSpan) -> Option<u64> {
        let mut out = None;
        self.pieces(id, abs, |_, valid, at, off, n| {
            if out.is_none() {
                if let Some(x) = valid.first_clear(at, n) {
                    out = Some(off + (x - at));
                }
            }
        });
        out
    }

    /// Store into `abs` of `id` (overlay-aware; copy-on-write stripes).
    fn put(&mut self, id: AllocId, abs: ByteSpan, what: Put<'_>) {
        let overlaid = self.is_overlaid(id);
        let apply = |bytes: &mut [u8], valid: &mut BitSet, at: u64, pos: usize, n: u64, what: &Put<'_>| {
            let r = at as usize..(at + n) as usize;
            match what {
                Put::Bytes(src) => {
                    bytes[r].copy_from_slice(&src[pos..pos + n as usize]);
                    valid.set_range(at, n, true);
                }
                Put::Fill(b) => {
                    bytes[r].fill(*b);
                    valid.set_range(at, n, true);
                }
                Put::Invalidate => valid.set_range(at, n, false),
                Put::BytesValid(src, v) => {
                    bytes[r].copy_from_slice(&src[pos..pos + n as usize]);
                    for k in 0..n {
                        valid.set_range(at + k, 1, v[pos + k as usize]);
                    }
                }
            }
        };
        if !overlaid {
            let a = self.get_mut(id);
            apply(&mut a.bytes, &mut a.valid, abs.start, 0, abs.len, &what);
            return;
        }
        let sh = self.shard.as_mut().expect("overlaid implies shard");
        // SAFETY: see `Shard`; a shared allocation, read-only.
        let base = unsafe { &*sh.allocs.add(id.0 as usize) };
        let mut i = abs.start;
        while i < abs.end() {
            let stripe = i / STRIPE;
            let lo = i % STRIPE;
            let n = (STRIPE - lo).min(abs.end() - i);
            let st = sh.overlay.entry((id, stripe)).or_insert_with(|| {
                let s0 = stripe * STRIPE;
                let len = STRIPE.min(base.size - s0);
                let mut valid = BitSet::new(len, false);
                for k in 0..len {
                    if base.valid.get(s0 + k) {
                        valid.set_range(k, 1, true);
                    }
                }
                Stripe {
                    bytes: base.bytes[s0 as usize..(s0 + len) as usize].to_vec(),
                    valid,
                    written: BitSet::new(len, false),
                }
            });
            apply(&mut st.bytes, &mut st.valid, lo, (i - abs.start) as usize, n, &what);
            st.written.set_range(lo, n, true);
            i += n;
        }
    }

    /// Read the concatenation of `spans` (view-relative, in order) into `dst`.
    pub fn read(&self, view: View, spans: &[ByteSpan], dst: &mut [u8]) -> Result<(), ArenaError> {
        self.check_oob(view, spans)?;
        let total = Self::total(spans);
        if total != dst.len() as u64 {
            return Err(ArenaError::LengthMismatch { expected: total, got: dst.len() as u64 });
        }
        if self.get(view.alloc).metadata_only {
            return Err(ArenaError::MetadataOnly { alloc: view.alloc });
        }
        let mut pos = 0usize;
        for s in spans {
            let abs = view.absolute(*s);
            if self.policy == ValidityPolicy::Error {
                if let Some(off) = self.first_clear(view.alloc, abs) {
                    return Err(ArenaError::Uninit { alloc: view.alloc, offset: off });
                }
            }
            let zero = self.policy == ValidityPolicy::ZeroAndReport;
            self.pieces(view.alloc, abs, |bytes, valid, at, off, n| {
                let d = pos + (off - abs.start) as usize;
                dst[d..d + n as usize].copy_from_slice(&bytes[at as usize..(at + n) as usize]);
                if zero {
                    let mut i = at;
                    while let Some(x) = valid.first_clear(i, at + n - i) {
                        dst[d + (x - at) as usize] = 0;
                        i = x + 1;
                    }
                }
            });
            pos += s.len as usize;
        }
        Ok(())
    }

    /// Bytes of `span` (allocation-relative) regardless of validity
    /// (out-of-range bytes read as zero).
    pub fn read_raw(&self, id: AllocId, span: ByteSpan) -> Vec<u8> {
        let size = self.get(id).size;
        let mut out = vec![0u8; span.len as usize];
        let end = span.end().min(size);
        if span.start < end && !self.get(id).metadata_only {
            let abs = ByteSpan::new(span.start, end - span.start);
            self.pieces(id, abs, |bytes, _, at, off, n| {
                let d = (off - span.start) as usize;
                out[d..d + n as usize].copy_from_slice(&bytes[at as usize..(at + n) as usize]);
            });
        }
        out
    }

    /// First invalid byte (allocation-relative) of a view-relative span.
    pub fn first_invalid(&self, view: View, span: ByteSpan) -> Option<u64> {
        let a = self.get(view.alloc);
        if a.metadata_only {
            return None;
        }
        let abs = view.absolute(span);
        if abs.end() > a.size {
            return None;
        }
        self.first_clear(view.alloc, abs)
    }

    /// Raw copy between allocations that carries validity (async payloads):
    /// destination bytes become valid exactly where the source was valid.
    pub fn copy_with_validity(&mut self, src: (AllocId, ByteSpan), dst: (AllocId, u64)) -> Result<(), ArenaError> {
        let (sid, sspan) = src;
        let (did, doff) = dst;
        self.check_oob(self.view(sid), &[sspan])?;
        self.check_oob(self.view(did), &[ByteSpan::new(doff, sspan.len)])?;
        if self.get(sid).metadata_only {
            return Err(ArenaError::MetadataOnly { alloc: sid });
        }
        if self.get(did).metadata_only {
            return Err(ArenaError::MetadataOnly { alloc: did });
        }
        let n = sspan.len as usize;
        let mut bytes = vec![0u8; n];
        let mut valid = vec![false; n];
        self.pieces(sid, sspan, |b, v, at, off, len| {
            let d = (off - sspan.start) as usize;
            bytes[d..d + len as usize].copy_from_slice(&b[at as usize..(at + len) as usize]);
            for k in 0..len {
                valid[d + k as usize] = v.get(at + k);
            }
        });
        self.put(did, ByteSpan::new(doff, sspan.len), Put::BytesValid(&bytes, &valid));
        Ok(())
    }

    /// Reset an allocation to all-invalid zero bytes (CTA-private memory
    /// reused by a later CTA).
    pub fn reset(&mut self, id: AllocId) {
        let a = self.get_mut(id);
        a.bytes.fill(0);
        let n = a.size;
        a.valid.set_range(0, n, false);
    }

    /// Write the concatenation of `src` into `spans` (view-relative, in
    /// order) and mark those bytes valid.
    pub fn write(&mut self, view: View, spans: &[ByteSpan], src: &[u8]) -> Result<(), ArenaError> {
        self.check_oob(view, spans)?;
        let total = Self::total(spans);
        if total != src.len() as u64 {
            return Err(ArenaError::LengthMismatch { expected: total, got: src.len() as u64 });
        }
        if self.get(view.alloc).metadata_only {
            return Err(ArenaError::MetadataOnly { alloc: view.alloc });
        }
        let mut pos = 0usize;
        for s in spans {
            let n = s.len as usize;
            self.put(view.alloc, view.absolute(*s), Put::Bytes(&src[pos..pos + n]));
            pos += n;
        }
        Ok(())
    }

    /// Fill spans with `byte` and mark them valid (st.bulk, zero fill).
    pub fn fill(&mut self, view: View, spans: &[ByteSpan], byte: u8) -> Result<(), ArenaError> {
        self.check_oob(view, spans)?;
        if self.get(view.alloc).metadata_only {
            return Err(ArenaError::MetadataOnly { alloc: view.alloc });
        }
        for s in spans {
            self.put(view.alloc, view.absolute(*s), Put::Fill(byte));
        }
        Ok(())
    }

    /// Clear validity of spans (`discard`, explicit uninit).
    pub fn invalidate(&mut self, view: View, spans: &[ByteSpan]) -> Result<(), ArenaError> {
        self.check_oob(view, spans)?;
        if self.get(view.alloc).metadata_only {
            return Err(ArenaError::MetadataOnly { alloc: view.alloc });
        }
        for s in spans {
            self.put(view.alloc, view.absolute(*s), Put::Invalidate);
        }
        Ok(())
    }

    /// Are all bytes of `spans` valid?
    pub fn is_valid(&self, view: View, spans: &[ByteSpan]) -> Result<bool, ArenaError> {
        self.check_oob(view, spans)?;
        if self.get(view.alloc).metadata_only {
            return Ok(true);
        }
        Ok(spans.iter().all(|s| self.first_clear(view.alloc, view.absolute(*s)).is_none()))
    }

    /// Resolve a global virtual address range to `(alloc, offset)`. The whole
    /// `[va, va+len)` must lie in one allocation.
    pub fn resolve_global(&self, va: u64, len: u64) -> Result<(AllocId, u64), ArenaError> {
        let index = match &self.shard {
            None => &self.global_index,
            // SAFETY: see `Shard`.
            Some(sh) => unsafe { &*sh.global_index },
        };
        let i = index.partition_point(|(base, _, _)| *base <= va);
        if i == 0 {
            return Err(ArenaError::BadAddress { space: Space::Global, addr: va });
        }
        let (base, end, id) = index[i - 1];
        if va.checked_add(len).is_none_or(|e| e > end) {
            if va < end {
                return Err(ArenaError::OutOfBounds {
                    alloc: id,
                    span: ByteSpan::new(va - base, len),
                    size: end - base,
                });
            }
            return Err(ArenaError::BadAddress { space: Space::Global, addr: va });
        }
        Ok((id, va - base))
    }

    /// Split off a shard (see [`Shard`]): it addresses this arena's
    /// allocations in place, buffering writes to shared allocations
    /// (global, param) in a copy-on-write overlay. `private` lists the
    /// partition's own allocations (checked in debug builds). While any
    /// shard is alive this arena must not be used, modified or dropped, and
    /// different shards must own disjoint private allocations (the
    /// scheduler's round structure guarantees both).
    pub fn make_shard(&mut self, private: &[AllocId]) -> Arena {
        assert!(self.shard.is_none(), "nested shards are not supported");
        debug_assert!(private.iter().all(|id| !is_shared_space(self.allocs[id.0 as usize].space)));
        Arena {
            allocs: Vec::new(),
            policy: self.policy,
            next_global_va: 0,
            global_index: Vec::new(),
            shard: Some(Box::new(Shard {
                allocs: self.allocs.as_mut_ptr(),
                len: self.allocs.len(),
                global_index: &self.global_index as *const _,
                overlay: HashMap::new(),
            })),
        }
    }

    /// Drop a shard's writes to shared allocations (its private writes
    /// were made in place).
    pub fn discard_shard(&mut self, shard: Arena) {
        let _ = shard.shard.expect("not a shard");
    }

    /// Apply a shard's writes to shared allocations. Shards merged later
    /// overwrite bytes written by shards merged earlier (byte granularity).
    pub fn merge_shard(&mut self, shard: Arena) {
        let sh = *shard.shard.expect("not a shard");
        let mut stripes: Vec<((AllocId, u64), Stripe)> = sh.overlay.into_iter().collect();
        stripes.sort_by_key(|(k, _)| *k);
        for ((id, stripe), st) in stripes {
            let a = &mut self.allocs[id.0 as usize];
            let s0 = stripe * STRIPE;
            let len = st.written.len();
            let mut k = 0;
            while k < len {
                match st.written.first_set(k, len - k) {
                    None => break,
                    Some(x) => {
                        let end = st.written.first_clear(x, len - x).unwrap_or(len);
                        let (lo, hi) = (x as usize, end as usize);
                        a.bytes[(s0 + x) as usize..(s0 + end) as usize].copy_from_slice(&st.bytes[lo..hi]);
                        for b in x..end {
                            a.valid.set_range(s0 + b, 1, st.valid.get(b));
                        }
                        k = end;
                    }
                }
            }
        }
    }
}

fn align_up(x: u64, a: u64) -> u64 {
    x.div_ceil(a) * a
}

/// Address-value encodings (what a pointer register holds).
///
/// * Global: synthetic VA starting at [`addr::GLOBAL_VA_BASE`]; allocation bases
///   are [`addr::GLOBAL_ALIGN`]-aligned, separated by [`addr::GLOBAL_GUARD`].
/// * Shared (`.shared::cta` and `.shared::cluster`, one encoding, as on
///   hardware): 32-bit `rank << 24 | offset`, where `rank` is the owning
///   CTA's rank in its cluster and `offset < 2^24` the byte offset in its
///   window. `cvta.to.shared` of the executing CTA's window yields its own
///   rank tag, so `mapa(p, own_rank) == p` and the CUTLASS pair-leader idiom
///   `p & 0xFEFF_FFFF` names the even CTA of the pair. `.shared::cta`
///   accesses require `rank == own rank` (else a bad-address error);
///   `.shared::cluster` accesses route by rank. In a cluster-less launch
///   every rank tag is 0.
/// * Generic: global VAs as is; distributed shared memory at
///   [`addr::GENERIC_SHARED_BASE`] + shared address (so 64-bit `mapa` results
///   and generic ld/st to a peer's window resolve); local at
///   [`addr::GENERIC_LOCAL_BASE`] + offset; kernel parameters at
///   [`addr::GENERIC_PARAM_BASE`] + offset.
/// * Tensor memory: `lane << 16 | column` (PTX taddr).
pub mod addr {
    /// First global VA.
    pub const GLOBAL_VA_BASE: u64 = 0x0000_1000_0000_0000;
    /// Alignment of every global allocation base.
    pub const GLOBAL_ALIGN: u64 = 1 << 12;
    /// Unmapped gap after every global allocation.
    pub const GLOBAL_GUARD: u64 = 1 << 16;
    /// Generic aperture of the cluster's distributed shared memory.
    pub const GENERIC_SHARED_BASE: u64 = 0x0000_7f00_0000_0000;
    /// Generic aperture of the executing thread's local memory.
    pub const GENERIC_LOCAL_BASE: u64 = 0x0000_7e00_0000_0000;
    /// Generic aperture of the launch's kernel-parameter block
    /// (`__grid_constant__` tensor maps); param-space addresses are offsets
    /// in the block.
    pub const GENERIC_PARAM_BASE: u64 = 0x0000_7d00_0000_0000;
    /// Size of each generic aperture.
    pub const APERTURE: u64 = 1 << 32;
    /// Bits of window offset in a shared address.
    pub const SHARED_OFFSET_BITS: u32 = 24;
    /// Largest shared window offset + 1.
    pub const SHARED_WINDOW_MAX: u32 = 1 << SHARED_OFFSET_BITS;
    pub const TMEM_LANES: u32 = 128;
    pub const TMEM_COLS: u32 = 512;

    /// Which space a generic address points into.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Generic {
        Global(u64),
        /// A shared address (`rank << 24 | offset`).
        Shared(u32),
        Local(u32),
        /// Offset in the kernel-parameter block.
        Param(u32),
        Unmapped(u64),
    }

    pub fn classify_generic(va: u64) -> Generic {
        if (GENERIC_SHARED_BASE..GENERIC_SHARED_BASE + APERTURE).contains(&va) {
            Generic::Shared((va - GENERIC_SHARED_BASE) as u32)
        } else if (GENERIC_LOCAL_BASE..GENERIC_LOCAL_BASE + APERTURE).contains(&va) {
            Generic::Local((va - GENERIC_LOCAL_BASE) as u32)
        } else if (GENERIC_PARAM_BASE..GENERIC_PARAM_BASE + APERTURE).contains(&va) {
            Generic::Param((va - GENERIC_PARAM_BASE) as u32)
        } else if va >= GLOBAL_VA_BASE && va < GENERIC_LOCAL_BASE {
            Generic::Global(va)
        } else {
            Generic::Unmapped(va)
        }
    }

    /// Generic address of a shared address (`rank << 24 | offset`).
    pub const fn generic_from_shared(shared: u32) -> u64 {
        GENERIC_SHARED_BASE + shared as u64
    }

    pub const fn generic_from_local(offset: u32) -> u64 {
        GENERIC_LOCAL_BASE + offset as u64
    }

    /// Encode a shared address; `None` if `offset >= 2^24` or `rank > 255`.
    pub const fn shared_addr(rank: u32, offset: u32) -> Option<u32> {
        if offset >= SHARED_WINDOW_MAX || rank > 0xff {
            None
        } else {
            Some((rank << SHARED_OFFSET_BITS) | offset)
        }
    }

    /// Decode a shared address into `(rank, window offset)`.
    pub const fn decode_shared(a: u32) -> (u32, u32) {
        (a >> SHARED_OFFSET_BITS, a & (SHARED_WINDOW_MAX - 1))
    }

    pub const fn tmem_addr(lane: u32, col: u32) -> u32 {
        (lane << 16) | (col & 0xffff)
    }

    pub const fn tmem_decode(taddr: u32) -> (u32, u32) {
        (taddr >> 16, taddr & 0xffff)
    }

    /// Byte offset of (lane, column) in a CTA's tmem allocation.
    pub const fn tmem_byte_offset(lane: u32, col: u32) -> u64 {
        (lane as u64 * TMEM_COLS as u64 + col as u64) * 4
    }

    /// Size in bytes of a CTA's tmem allocation.
    pub const TMEM_BYTES: u64 = TMEM_LANES as u64 * TMEM_COLS as u64 * 4;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_write_validity() {
        let mut a = Arena::new(ValidityPolicy::Error);
        let id = a.alloc(Space::Shared, Owner::Cta(0), "smem", 256, Init::Uninit);
        let v = a.view(id);
        let mut buf = [0u8; 4];
        assert!(matches!(a.read(v, &[ByteSpan::new(0, 4)], &mut buf), Err(ArenaError::Uninit { .. })));
        a.write(v, &[ByteSpan::new(0, 2), ByteSpan::new(8, 2)], &[1, 2, 3, 4]).unwrap();
        a.read(v, &[ByteSpan::new(8, 2), ByteSpan::new(0, 2)], &mut buf).unwrap();
        assert_eq!(buf, [3, 4, 1, 2]);
        assert!(matches!(a.check_oob(v, &[ByteSpan::new(250, 8)]), Err(ArenaError::OutOfBounds { .. })));
        a.invalidate(v, &[ByteSpan::new(0, 1)]).unwrap();
        assert!(!a.is_valid(v, &[ByteSpan::new(0, 2)]).unwrap());
        a.set_policy(ValidityPolicy::Allow);
        a.read(v, &[ByteSpan::new(0, 4)], &mut buf).unwrap();
    }

    #[test]
    fn shards_overlay_and_merge_in_order() {
        let mut a = Arena::new(ValidityPolicy::Error);
        let g = a.alloc(Space::Global, Owner::Launch, "g", 3 * STRIPE, Init::Zeroed);
        let s0 = a.alloc(Space::Shared, Owner::Cta(0), "s0", 64, Init::Uninit);
        let s1 = a.alloc(Space::Shared, Owner::Cta(1), "s1", 64, Init::Uninit);
        let vg = a.view(g);
        let mut sh0 = a.make_shard(&[s0]);
        let mut sh1 = a.make_shard(&[s1]);
        // Each shard sees the snapshot plus its own writes.
        sh0.write(vg, &[ByteSpan::new(STRIPE - 2, 4)], &[1, 2, 3, 4]).unwrap();
        sh1.write(vg, &[ByteSpan::new(STRIPE, 2)], &[9, 9]).unwrap();
        let mut b = [0u8; 4];
        sh0.read(vg, &[ByteSpan::new(STRIPE - 2, 4)], &mut b).unwrap();
        assert_eq!(b, [1, 2, 3, 4]);
        sh1.read(vg, &[ByteSpan::new(STRIPE - 2, 4)], &mut b).unwrap();
        assert_eq!(b, [0, 0, 9, 9]);
        sh0.write(sh0.view(s0), &[ByteSpan::new(0, 1)], &[7]).unwrap();
        assert!(sh1.read(sh1.view(s1), &[ByteSpan::new(0, 1)], &mut b[..1]).is_err());
        a.merge_shard(sh0);
        a.merge_shard(sh1);
        a.read(vg, &[ByteSpan::new(STRIPE - 2, 4)], &mut b).unwrap();
        // Later shard wins where both wrote (byte granularity).
        assert_eq!(b, [1, 2, 9, 9]);
        a.read(a.view(s0), &[ByteSpan::new(0, 1)], &mut b[..1]).unwrap();
        assert_eq!(b[0], 7);
    }

    #[test]
    fn global_resolution() {
        let mut a = Arena::new(ValidityPolicy::Error);
        let x = a.alloc(Space::Global, Owner::Launch, "x", 100, Init::Zeroed);
        let y = a.alloc(Space::Global, Owner::Launch, "y", 100, Init::Zeroed);
        let bx = a.get(x).base;
        let by = a.get(y).base;
        assert_eq!(a.resolve_global(bx + 10, 4).unwrap(), (x, 10));
        assert_eq!(a.resolve_global(by, 100).unwrap(), (y, 0));
        assert!(a.resolve_global(bx + 98, 4).is_err());
        assert!(a.resolve_global(bx + 200, 4).is_err());
        assert!(a.resolve_global(0, 4).is_err());
    }

    #[test]
    fn coalesce_spans() {
        let mut s = vec![ByteSpan::new(8, 4), ByteSpan::new(0, 4), ByteSpan::new(4, 2), ByteSpan::new(20, 0)];
        ByteSpan::coalesce(&mut s);
        assert_eq!(s, vec![ByteSpan::new(0, 6), ByteSpan::new(8, 4)]);
    }

    #[test]
    fn shared_addr_roundtrip() {
        let a = addr::shared_addr(3, 0x1230).unwrap();
        assert_eq!(a, 3 << 24 | 0x1230);
        assert_eq!(addr::decode_shared(a), (3, 0x1230));
        // CUTLASS pair-leader idiom: clearing bit 24 names the even CTA.
        assert_eq!(addr::decode_shared(addr::shared_addr(1, 64).unwrap() & 0xFEFF_FFFF), (0, 64));
        assert_eq!(addr::shared_addr(0, 1 << 24), None);
        assert!(matches!(addr::classify_generic(addr::generic_from_shared(a)), addr::Generic::Shared(x) if x == a));
    }

    #[test]
    fn bitset_ranges() {
        let mut b = BitSet::new(200, false);
        b.set_range(3, 130, true);
        assert_eq!(b.first_clear(3, 130), None);
        assert_eq!(b.first_clear(0, 10), Some(0));
        assert_eq!(b.first_clear(100, 50), Some(133));
    }
}
