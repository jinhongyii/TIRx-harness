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
use std::fmt;

/// Allocation space (where bytes physically live).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
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

/// The arena: all allocations of one module run.
#[derive(Clone, Debug, Default)]
pub struct Arena {
    allocs: Vec<Allocation>,
    policy: ValidityPolicy,
    next_global_va: u64,
    /// Global allocations sorted by base (bases are monotonically assigned).
    global_index: Vec<(u64, u64, AllocId)>,
}

impl Arena {
    pub fn new(policy: ValidityPolicy) -> Arena {
        Arena { allocs: Vec::new(), policy, next_global_va: addr::GLOBAL_VA_BASE, global_index: Vec::new() }
    }

    pub fn policy(&self) -> ValidityPolicy {
        self.policy
    }

    pub fn set_policy(&mut self, p: ValidityPolicy) {
        self.policy = p;
    }

    /// Create an allocation. Global allocations get a fresh VA range.
    pub fn alloc(&mut self, space: Space, owner: Owner, name: &str, size: u64, init: Init) -> AllocId {
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

    pub fn get(&self, id: AllocId) -> &Allocation {
        &self.allocs[id.0 as usize]
    }

    pub fn get_mut(&mut self, id: AllocId) -> &mut Allocation {
        &mut self.allocs[id.0 as usize]
    }

    pub fn len(&self) -> usize {
        self.allocs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.allocs.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (AllocId, &Allocation)> {
        self.allocs.iter().enumerate().map(|(i, a)| (AllocId(i as u32), a))
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

    /// Read the concatenation of `spans` (view-relative, in order) into `dst`.
    pub fn read(&self, view: View, spans: &[ByteSpan], dst: &mut [u8]) -> Result<(), ArenaError> {
        self.check_oob(view, spans)?;
        let total = Self::total(spans);
        if total != dst.len() as u64 {
            return Err(ArenaError::LengthMismatch { expected: total, got: dst.len() as u64 });
        }
        let a = self.get(view.alloc);
        if a.metadata_only {
            return Err(ArenaError::MetadataOnly { alloc: view.alloc });
        }
        let mut pos = 0usize;
        for s in spans {
            let abs = view.absolute(*s);
            if self.policy == ValidityPolicy::Error {
                if let Some(off) = a.valid.first_clear(abs.start, abs.len) {
                    return Err(ArenaError::Uninit { alloc: view.alloc, offset: off });
                }
            }
            let n = s.len as usize;
            dst[pos..pos + n].copy_from_slice(&a.bytes[abs.start as usize..abs.end() as usize]);
            pos += n;
        }
        Ok(())
    }

    /// Write the concatenation of `src` into `spans` (view-relative, in
    /// order) and mark those bytes valid.
    pub fn write(&mut self, view: View, spans: &[ByteSpan], src: &[u8]) -> Result<(), ArenaError> {
        self.check_oob(view, spans)?;
        let total = Self::total(spans);
        if total != src.len() as u64 {
            return Err(ArenaError::LengthMismatch { expected: total, got: src.len() as u64 });
        }
        let id = view.alloc;
        let a = self.get_mut(id);
        if a.metadata_only {
            return Err(ArenaError::MetadataOnly { alloc: id });
        }
        let mut pos = 0usize;
        for s in spans {
            let abs = view.absolute(*s);
            let n = s.len as usize;
            a.bytes[abs.start as usize..abs.end() as usize].copy_from_slice(&src[pos..pos + n]);
            a.valid.set_range(abs.start, abs.len, true);
            pos += n;
        }
        Ok(())
    }

    /// Fill spans with `byte` and mark them valid (st.bulk, zero fill).
    pub fn fill(&mut self, view: View, spans: &[ByteSpan], byte: u8) -> Result<(), ArenaError> {
        self.check_oob(view, spans)?;
        let id = view.alloc;
        let a = self.get_mut(id);
        if a.metadata_only {
            return Err(ArenaError::MetadataOnly { alloc: id });
        }
        for s in spans {
            let abs = view.absolute(*s);
            a.bytes[abs.start as usize..abs.end() as usize].fill(byte);
            a.valid.set_range(abs.start, abs.len, true);
        }
        Ok(())
    }

    /// Clear validity of spans (`discard`, explicit uninit).
    pub fn invalidate(&mut self, view: View, spans: &[ByteSpan]) -> Result<(), ArenaError> {
        self.check_oob(view, spans)?;
        let id = view.alloc;
        let a = self.get_mut(id);
        if a.metadata_only {
            return Err(ArenaError::MetadataOnly { alloc: id });
        }
        for s in spans {
            let abs = view.absolute(*s);
            a.valid.set_range(abs.start, abs.len, false);
        }
        Ok(())
    }

    /// Are all bytes of `spans` valid?
    pub fn is_valid(&self, view: View, spans: &[ByteSpan]) -> Result<bool, ArenaError> {
        self.check_oob(view, spans)?;
        let a = self.get(view.alloc);
        Ok(spans.iter().all(|s| {
            let abs = view.absolute(*s);
            a.metadata_only || a.valid.first_clear(abs.start, abs.len).is_none()
        }))
    }

    /// Resolve a global virtual address range to `(alloc, offset)`. The whole
    /// `[va, va+len)` must lie in one allocation.
    pub fn resolve_global(&self, va: u64, len: u64) -> Result<(AllocId, u64), ArenaError> {
        let i = self.global_index.partition_point(|(base, _, _)| *base <= va);
        if i == 0 {
            return Err(ArenaError::BadAddress { space: Space::Global, addr: va });
        }
        let (base, end, id) = self.global_index[i - 1];
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
}

fn align_up(x: u64, a: u64) -> u64 {
    x.div_ceil(a) * a
}

/// Address-value encodings (what a pointer register holds).
///
/// * Global: synthetic VA starting at [`addr::GLOBAL_VA_BASE`]; allocation bases
///   are [`addr::GLOBAL_ALIGN`]-aligned, separated by [`addr::GLOBAL_GUARD`].
/// * Shared (`.shared::cta`): 32-bit offset in the executing CTA's window.
/// * Shared cluster (`.shared::cluster`): 32-bit; bits `[0, 24)` = window
///   offset, bits `[24, 32)` = `rank + 1` of the target CTA, or 0 meaning
///   "the executing CTA". So every `.shared::cta` address is also a valid
///   `.shared::cluster` address naming the executing CTA (PTX property).
/// * Generic: global VAs as is; shared at [`addr::GENERIC_SHARED_BASE`] + window
///   offset (executing CTA); local at [`addr::GENERIC_LOCAL_BASE`] + offset.
/// * Tensor memory: `lane << 16 | column` (PTX taddr).
pub mod addr {
    /// First global VA.
    pub const GLOBAL_VA_BASE: u64 = 0x0000_1000_0000_0000;
    /// Alignment of every global allocation base.
    pub const GLOBAL_ALIGN: u64 = 1 << 12;
    /// Unmapped gap after every global allocation.
    pub const GLOBAL_GUARD: u64 = 1 << 16;
    /// Generic aperture of the executing CTA's shared window.
    pub const GENERIC_SHARED_BASE: u64 = 0x0000_7f00_0000_0000;
    /// Generic aperture of the executing thread's local memory.
    pub const GENERIC_LOCAL_BASE: u64 = 0x0000_7e00_0000_0000;
    /// Size of each generic aperture.
    pub const APERTURE: u64 = 1 << 32;
    /// Bits of window offset in a shared::cluster address.
    pub const CLUSTER_OFFSET_BITS: u32 = 24;
    pub const TMEM_LANES: u32 = 128;
    pub const TMEM_COLS: u32 = 512;

    /// Which space a generic address points into.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Generic {
        Global(u64),
        Shared(u32),
        Local(u32),
        Unmapped(u64),
    }

    pub fn classify_generic(va: u64) -> Generic {
        if (GENERIC_SHARED_BASE..GENERIC_SHARED_BASE + APERTURE).contains(&va) {
            Generic::Shared((va - GENERIC_SHARED_BASE) as u32)
        } else if (GENERIC_LOCAL_BASE..GENERIC_LOCAL_BASE + APERTURE).contains(&va) {
            Generic::Local((va - GENERIC_LOCAL_BASE) as u32)
        } else if va >= GLOBAL_VA_BASE && va < GENERIC_LOCAL_BASE {
            Generic::Global(va)
        } else {
            Generic::Unmapped(va)
        }
    }

    pub const fn generic_from_shared(offset: u32) -> u64 {
        GENERIC_SHARED_BASE + offset as u64
    }

    pub const fn generic_from_local(offset: u32) -> u64 {
        GENERIC_LOCAL_BASE + offset as u64
    }

    /// Encode a shared::cluster address. `rank = None` = executing CTA.
    pub const fn shared_cluster(rank: Option<u32>, offset: u32) -> u32 {
        let r = match rank {
            Some(r) => r + 1,
            None => 0,
        };
        (r << CLUSTER_OFFSET_BITS) | (offset & ((1 << CLUSTER_OFFSET_BITS) - 1))
    }

    /// Decode a shared::cluster address into (rank or self, window offset).
    pub const fn decode_shared_cluster(a: u32) -> (Option<u32>, u32) {
        let r = a >> CLUSTER_OFFSET_BITS;
        let off = a & ((1 << CLUSTER_OFFSET_BITS) - 1);
        if r == 0 {
            (None, off)
        } else {
            (Some(r - 1), off)
        }
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
    fn cluster_addr_roundtrip() {
        let a = addr::shared_cluster(Some(3), 0x1230);
        assert_eq!(addr::decode_shared_cluster(a), (Some(3), 0x1230));
        assert_eq!(addr::decode_shared_cluster(0x40), (None, 0x40));
        assert!(matches!(addr::classify_generic(addr::generic_from_shared(16)), addr::Generic::Shared(16)));
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
