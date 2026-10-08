//! Helpers shared by the handler families (and by the scheduler's async
//! landing path): operand decoding, address resolution, memory access with
//! observer emission, sync-event emission and error mapping.

use super::{BufBinding, ExecCtx, ExecError, ExecErrorKind, LaunchAux, LaunchCounters};
use crate::arena::{addr, AllocId, Arena, ArenaError, ByteSpan, Space, ValidityPolicy, View};
use crate::dtype::Ty;
use crate::observe::{
    Access, AccessKind, Actor, AsyncTarget, Collective, Counts, CtaId, LaneSpan, Observer, ProtocolCmd, ProtocolStatus, SyncEvent,
    SyncKind, Window,
};
use crate::oplib::{OpError, OpResult, PtxIo};
use crate::program::{AddrSpace, Buf, Operand, Proxy, Reg, Scope, Sem};
use crate::site::SiteId;
use crate::sync::{ResourceId, SyncCmd, SyncError};
use crate::value::{WarpMask, WarpValue};

pub use crate::arena::addr::GENERIC_PARAM_BASE;

/// Placeholder for an op `oplib::resolve_ptx` rejected at load.
pub fn unresolved_ptx(_io: &mut PtxIo<'_>) -> OpResult {
    Err(OpError::unsupported("unresolved generic op"))
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

pub fn err(ctx: &ExecCtx<'_>, kind: ExecErrorKind, lanes: WarpMask, msg: impl Into<String>) -> ExecError {
    let mut e = ctx.error(kind, msg);
    e.lanes = lanes;
    e
}

pub fn arena_err(ctx: &ExecCtx<'_>, e: ArenaError, lanes: WarpMask) -> ExecError {
    let kind = match e {
        ArenaError::OutOfBounds { .. } => ExecErrorKind::OutOfBounds,
        ArenaError::Uninit { .. } => ExecErrorKind::Uninit,
        ArenaError::BadAddress { .. } => ExecErrorKind::BadAddress,
        ArenaError::MetadataOnly { .. } | ArenaError::LengthMismatch { .. } => ExecErrorKind::Internal,
    };
    let name = match e {
        ArenaError::OutOfBounds { alloc, .. } | ArenaError::Uninit { alloc, .. } => {
            format!(" ({})", ctx.arena.get(alloc).name)
        }
        _ => String::new(),
    };
    err(ctx, kind, lanes, format!("{e}{name}"))
}

pub fn op_err(ctx: &ExecCtx<'_>, e: OpError) -> ExecError {
    ctx.error(ExecErrorKind::Op(e.kind), e.message)
}

pub fn sync_err(ctx: &ExecCtx<'_>, e: SyncError) -> ExecError {
    let msg = format!("{e:?}");
    if e.is_infrastructure() {
        ctx.error(ExecErrorKind::Internal, msg)
    } else {
        ctx.error(ExecErrorKind::Protocol(e), msg)
    }
}

/// `Unsupported` (fails closed as incomplete) when any lane is active.
pub fn unsupported(ctx: &ExecCtx<'_>, what: &str) -> ExecError {
    ctx.error(ExecErrorKind::Unsupported, format!("not modeled: {what}"))
}

// ---------------------------------------------------------------------------
// Operands
// ---------------------------------------------------------------------------

#[inline]
pub fn operand_ty(ctx: &ExecCtx<'_>, o: Operand) -> Ty {
    match o {
        Operand::Reg(r) => ctx.program.regs[r.0 as usize].ty,
        Operand::Const(c) => ctx.program.consts[c.0 as usize].ty,
    }
}

#[inline]
pub fn reg_ty(ctx: &ExecCtx<'_>, r: Reg) -> Ty {
    ctx.program.regs[r.0 as usize].ty
}

/// Slot `i` of operand `o` in one lane.
#[inline]
pub fn lane_slot(ctx: &ExecCtx<'_>, o: Operand, i: u32, lane: usize) -> u64 {
    match o {
        Operand::Reg(r) => ctx.warp.regs.get(ctx.slot(r) + i)[lane],
        Operand::Const(c) => {
            let k = ctx.program.consts[c.0 as usize].bits;
            match i {
                0 => k as u64,
                1 => (k >> 64) as u64,
                _ => 0,
            }
        }
    }
}

/// Low 64 bits of an operand in one lane.
#[inline]
pub fn lane_val(ctx: &ExecCtx<'_>, o: Operand, lane: usize) -> u64 {
    lane_slot(ctx, o, 0, lane)
}

/// Sign- or zero-extend raw bits of an integer of type `ty`.
#[inline]
pub fn extend(ty: Ty, v: u64) -> i64 {
    let bits = ty.bits().min(64);
    if bits == 64 {
        return v as i64;
    }
    let v = v & ((1u64 << bits) - 1);
    if ty.elem.is_signed_int() {
        let sh = 64 - bits;
        ((v << sh) as i64) >> sh
    } else {
        v as i64
    }
}

/// Integer value of an operand in one lane (sign-extended per its type).
#[inline]
pub fn lane_int(ctx: &ExecCtx<'_>, o: Operand, lane: usize) -> i64 {
    extend(operand_ty(ctx, o), lane_val(ctx, o, lane))
}

/// Little-endian bytes of operand `o` (as `ty`) in one lane.
pub fn lane_bytes(ctx: &ExecCtx<'_>, o: Operand, ty: Ty, lane: usize, out: &mut [u8]) {
    let n = ty.mem_bytes() as usize;
    let mut i = 0usize;
    let mut slot = 0u32;
    while i < n {
        let w = lane_slot(ctx, o, slot, lane).to_le_bytes();
        let k = (n - i).min(8);
        out[i..i + k].copy_from_slice(&w[..k]);
        i += k;
        slot += 1;
    }
}

/// Write little-endian bytes into register `r` (all its slots) in one lane;
/// bytes beyond `src` are zero.
pub fn write_lane_bytes(ctx: &mut ExecCtx<'_>, r: Reg, lane: usize, src: &[u8]) {
    let ty = reg_ty(ctx, r);
    let base = ctx.slot(r);
    let bits = ty.bits();
    for s in 0..ty.slots() {
        let mut w = [0u8; 8];
        let lo = (s * 8) as usize;
        if lo < src.len() {
            let k = (src.len() - lo).min(8);
            w[..k].copy_from_slice(&src[lo..lo + k]);
        }
        let mut v = u64::from_le_bytes(w);
        let rem = bits.saturating_sub(s * 64);
        if rem < 64 {
            v &= (1u64 << rem) - 1;
        }
        ctx.warp.regs.get_mut(base + s)[lane] = v;
    }
}

/// [`write_lane_bytes`] for every lane of `mask` at once (W13): lane `l`'s
/// bytes are `rows[row[l] * stride + lo..row[l] * stride + hi]`. One
/// register-type lookup per register instead of one per lane; same
/// per-lane effect.
#[allow(clippy::too_many_arguments)]
pub fn write_lanes_bytes(ctx: &mut ExecCtx<'_>, r: Reg, mask: WarpMask, rows: &[u8], stride: usize, row: &[usize; 32], lo: usize, hi: usize) {
    let ty = reg_ty(ctx, r);
    let base = ctx.slot(r);
    let bits = ty.bits();
    let len = hi - lo;
    for s in 0..ty.slots() {
        let off = (s * 8) as usize;
        let rem = bits.saturating_sub(s * 64);
        let m = if rem < 64 { (1u64 << rem) - 1 } else { u64::MAX };
        let dst = ctx.warp.regs.get_mut(base + s);
        let k = len.saturating_sub(off).min(8);
        let word = |at: usize| -> u64 {
            match k {
                0 => 0,
                4 => u32::from_le_bytes(rows[at..at + 4].try_into().unwrap()) as u64,
                8 => u64::from_le_bytes(rows[at..at + 8].try_into().unwrap()),
                _ => {
                    let mut w = [0u8; 8];
                    w[..k].copy_from_slice(&rows[at..at + k]);
                    u64::from_le_bytes(w)
                }
            }
        };
        for l in mask.lanes() {
            dst[l] = word(row[l] * stride + lo + off) & m;
        }
    }
}

/// Write a <= 64-bit raw value to `r` in one lane (other slots cleared).
#[inline]
pub fn write_lane(ctx: &mut ExecCtx<'_>, r: Reg, lane: usize, v: u64) {
    let ty = reg_ty(ctx, r);
    let base = ctx.slot(r);
    let bits = ty.bits();
    let v = if bits < 64 { v & ((1u64 << bits) - 1) } else { v };
    ctx.warp.regs.get_mut(base)[lane] = v;
    for s in 1..ty.slots() {
        ctx.warp.regs.get_mut(base + s)[lane] = 0;
    }
}

/// [`write_lane`] for every lane of `mask` at once (W13): one register-type
/// lookup instead of one per lane; same per-lane effect.
#[inline]
pub fn write_lanes(ctx: &mut ExecCtx<'_>, r: Reg, v: &WarpValue<u64>, mask: WarpMask) {
    let ty = reg_ty(ctx, r);
    let base = ctx.slot(r);
    let bits = ty.bits();
    let m = if bits < 64 { (1u64 << bits) - 1 } else { u64::MAX };
    let mut w = *v;
    for x in w.iter_mut() {
        *x &= m;
    }
    write_masked(ctx.warp.regs.get_mut(base), &w, mask);
    for s in 1..ty.slots() {
        write_masked(ctx.warp.regs.get_mut(base + s), &[0u64; 32], mask);
    }
}

/// Gather all slots of an operand (`n` slots) into `out`.
#[inline]
pub fn gather(ctx: &ExecCtx<'_>, o: Operand, n: u32, out: &mut [WarpValue<u64>; 4]) {
    for i in 0..n {
        out[i as usize] = ctx.read_slot(o, i);
    }
}

/// Write `n` slots of `r` under `mask`.
#[inline]
pub fn scatter(ctx: &mut ExecCtx<'_>, r: Reg, n: u32, v: &[WarpValue<u64>], mask: WarpMask) {
    let base = ctx.slot(r);
    for i in 0..n {
        write_masked(ctx.warp.regs.get_mut(base + i), &v[i as usize], mask);
    }
}

/// Masked lane blend (branch-free for full and empty masks).
#[inline(always)]
pub fn write_masked(dst: &mut WarpValue<u64>, v: &WarpValue<u64>, mask: WarpMask) {
    if mask.is_all() {
        *dst = *v;
    } else {
        let m = mask.bits();
        for (l, d) in dst.iter_mut().enumerate() {
            if (m >> l) & 1 == 1 {
                *d = v[l];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Address resolution
// ---------------------------------------------------------------------------

/// A resolved byte location.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Loc {
    pub alloc: AllocId,
    pub offset: u64,
    pub window: Option<Window>,
    /// The location is another CTA's shared window (shared::cluster).
    pub remote: Option<CtaId>,
}

impl Loc {
    #[inline]
    pub fn span(&self, len: u64) -> ByteSpan {
        ByteSpan::new(self.offset, len)
    }
}

fn bounds(ctx: &ExecCtx<'_>, alloc: AllocId, offset: u64, len: u64, lane: usize) -> Result<(), ExecError> {
    let size = ctx.arena.get(alloc).size;
    if offset.checked_add(len).is_none_or(|e| e > size) {
        return Err(arena_err(
            ctx,
            ArenaError::OutOfBounds { alloc, span: ByteSpan::new(offset, len), size },
            WarpMask::lane(lane),
        ));
    }
    Ok(())
}

/// Shared window of cluster rank `rank`.
fn shared_loc(ctx: &ExecCtx<'_>, rank: u32, off: u32, len: u64, lane: usize, window: Window) -> Result<Loc, ExecError> {
    let Some(&alloc) = ctx.cta.cluster_smem.get(rank as usize) else {
        return Err(err(
            ctx,
            ExecErrorKind::BadAddress,
            WarpMask::lane(lane),
            format!("shared address names CTA rank {rank} outside the cluster of {}", ctx.cta.cluster_smem.len()),
        ));
    };
    bounds(ctx, alloc, off as u64, len, lane)?;
    let remote = (rank != ctx.cta.rank_in_cluster).then(|| ctx.cta.cluster_ctas[rank as usize]);
    Ok(Loc { alloc, offset: off as u64, window: Some(window), remote })
}

/// The executing CTA's shared address of window offset `off`.
pub fn own_shared(ctx: &ExecCtx<'_>, off: u64) -> u64 {
    ((ctx.cta.rank_in_cluster as u64) << addr::SHARED_OFFSET_BITS) | (off & (addr::SHARED_WINDOW_MAX as u64 - 1))
}

/// Resolve an address value in `space` for `lane`, checking `len` bytes are
/// in bounds.
/// [`resolve`] for a data access (`ld`/`st`/`atom`/`red`/`st.bulk`/
/// `discard`) through a raw pointer: a null generic or global pointer is in
/// no aperture and faults as a null dereference before any window match.
/// (Synchronization operands keep the W1 rule that a zero-extended 32-bit
/// shared::cluster address, which may be 0, names that location.)
pub fn resolve_data(ctx: &ExecCtx<'_>, space: AddrSpace, a: u64, lane: usize, len: u64) -> Result<Loc, ExecError> {
    if a == 0 && matches!(space, AddrSpace::Generic | AddrSpace::Global) {
        return Err(err(
            ctx,
            ExecErrorKind::BadAddress,
            WarpMask::lane(lane),
            format!("null pointer dereference: {space:?} address 0x0 is in no aperture"),
        ));
    }
    resolve(ctx, space, a, lane, len)
}

/// A global/generic integer address that names no bound allocation
/// (W12-gaps 1 ruling): `incomplete` (`integer_address_without_binding`),
/// since the memory may exist outside the launch's bindings. An address in
/// an allocation's trailing guard gap is an out-of-bounds use of that
/// allocation (a kernel error).
fn unbound_global(ctx: &ExecCtx<'_>, va: u64, len: u64, lanes: WarpMask) -> ExecError {
    // A null pointer, or an address that decodes into another state space's
    // generic window (shared, local, param), is a provable state-space
    // mismatch, not unknown memory: a `bad_address` error (W11).
    if va == 0 || !matches!(addr::classify_generic(va), addr::Generic::Global(_) | addr::Generic::Unmapped(_)) {
        return arena_err(ctx, ArenaError::BadAddress { space: Space::Global, addr: va }, lanes);
    }
    if let Some((alloc, base, end)) = ctx.arena.global_neighbor(va) {
        let e = ArenaError::OutOfBounds { alloc, span: ByteSpan::new(va - base, len), size: end - base };
        return arena_err(ctx, e, lanes);
    }
    let mut e = err(
        ctx,
        ExecErrorKind::Unsupported,
        lanes,
        format!("integer_address_without_binding: global address {va:#x} names no bound allocation"),
    );
    e.attrs.insert("address".into(), serde_json::json!(va));
    e
}

pub fn resolve(ctx: &ExecCtx<'_>, space: AddrSpace, a: u64, lane: usize, len: u64) -> Result<Loc, ExecError> {
    let lanes = WarpMask::lane(lane);
    match space {
        AddrSpace::Global => {
            // `ld.global` of a kernel-parameter generic address (e.g. the
            // address of a `__grid_constant__` tensor map) reads the param
            // buffer (legal on hardware; stores there fail in `mem_write`).
            if let addr::Generic::Param(off) = addr::classify_generic(a) {
                return resolve(ctx, AddrSpace::Param, off as u64, lane, len);
            }
            let (alloc, offset) = match ctx.arena.resolve_global(a, len) {
                Ok(x) => x,
                Err(ArenaError::BadAddress { .. }) => return Err(unbound_global(ctx, a, len, lanes)),
                Err(e) => return Err(arena_err(ctx, e, lanes)),
            };
            Ok(Loc { alloc, offset, window: Some(Window::Global), remote: None })
        }
        AddrSpace::Shared => {
            if a >> 32 != 0 {
                return Err(err(ctx, ExecErrorKind::BadAddress, lanes, format!("shared::cta address {a:#x} exceeds 32 bits")));
            }
            // Every shared::cta address is a valid shared::cluster address:
            // a value whose rank tag names another CTA of the cluster (a
            // `mapa` result or cluster arithmetic reaching a shared::cta
            // operand) is decoded as that shared::cluster location (V2C-30).
            let (rank, off) = addr::decode_shared(a as u32);
            let w = if rank == ctx.cta.rank_in_cluster { Window::SharedCta } else { Window::SharedCluster };
            shared_loc(ctx, rank, off, len, lane, w)
        }
        AddrSpace::SharedCluster => {
            let (rank, off) = addr::decode_shared(a as u32);
            shared_loc(ctx, rank, off, len, lane, Window::SharedCluster)
        }
        AddrSpace::Generic => {
            match addr::classify_generic(a) {
                addr::Generic::Global(va) => resolve(ctx, AddrSpace::Global, va, lane, len),
                addr::Generic::Shared(sa) => {
                    let (rank, off) = addr::decode_shared(sa);
                    let w = if rank == ctx.cta.rank_in_cluster { Window::SharedCta } else { Window::SharedCluster };
                    shared_loc(ctx, rank, off, len, lane, w)
                }
                addr::Generic::Local(off) => resolve(ctx, AddrSpace::Local, off as u64, lane, len),
                addr::Generic::Param(off) => resolve(ctx, AddrSpace::Param, off as u64, lane, len),
                // A 32-bit shared::cluster window address (`rank << 24 |
                // offset`, e.g. a `mapa.shared::cluster` result) zero-extended
                // into a u64 and used where a generic pointer is expected:
                // decoded as that shared::cluster location (W1 ruling).
                addr::Generic::Unmapped(va) if va >> 32 == 0 => {
                    let (rank, off) = addr::decode_shared(va as u32);
                    let w = if rank == ctx.cta.rank_in_cluster { Window::SharedCta } else { Window::SharedCluster };
                    shared_loc(ctx, rank, off, len, lane, w)
                }
                addr::Generic::Unmapped(va) => Err(unbound_global(ctx, va, len, lanes)),
            }
        }
        AddrSpace::Local => {
            let Some(alloc) = ctx.warp.local else {
                return Err(err(ctx, ExecErrorKind::BadAddress, lanes, "local address but the warp has no local memory"));
            };
            let per = ctx.loaded.local_per_lane;
            if a.checked_add(len).is_none_or(|e| e > per) {
                return Err(arena_err(
                    ctx,
                    ArenaError::OutOfBounds { alloc, span: ByteSpan::new(a, len), size: per },
                    lanes,
                ));
            }
            Ok(Loc { alloc, offset: lane as u64 * per + a, window: None, remote: None })
        }
        AddrSpace::Param => {
            let alloc = ctx.cta.params;
            bounds(ctx, alloc, a, len, lane)?;
            Ok(Loc { alloc, offset: a, window: None, remote: None })
        }
        AddrSpace::Tmem => {
            let (tl, col) = addr::tmem_decode(a as u32);
            if tl >= addr::TMEM_LANES || col >= addr::TMEM_COLS {
                return Err(err(ctx, ExecErrorKind::BadAddress, lanes, format!("tmem address {a:#x} out of range")));
            }
            let alloc = ctx.cta.tmem;
            let off = addr::tmem_byte_offset(tl, col);
            bounds(ctx, alloc, off, len, lane)?;
            Ok(Loc { alloc, offset: off, window: None, remote: None })
        }
        AddrSpace::Const => Err(unsupported(ctx, "const state space")),
    }
}

/// Resolve `buf[idx]` (element index) for `lane`, `len` bytes.
pub fn resolve_buf(ctx: &ExecCtx<'_>, buf: Buf, idx: i64, lane: usize, len: u64) -> Result<Loc, ExecError> {
    let lanes = WarpMask::lane(lane);
    let decl = &ctx.program.buffers[buf.0 as usize];
    // Sub-byte elements start at bit `idx * bits`; a non-byte-aligned
    // access is Misaligned (contract).
    let Some(bit) = decl.bit_offset(idx) else {
        return Err(err(ctx, ExecErrorKind::OutOfBounds, lanes, format!("{}[{idx}]: offset overflows", decl.name)));
    };
    if bit.rem_euclid(8) != 0 {
        return Err(err(ctx, ExecErrorKind::Misaligned, lanes, format!("{}[{idx}]: sub-byte element is not byte-aligned", decl.name)));
    }
    let byte = bit.div_euclid(8);
    let oob = |ctx: &ExecCtx<'_>, alloc: AllocId, size: u64| {
        let start = byte;
        let msg = format!("{}[{idx}]: element out of bounds of {size} bytes", decl.name);
        let mut e = err(ctx, ExecErrorKind::OutOfBounds, lanes, msg);
        if start >= 0 {
            e.message = format!("{} (byte {start}, alloc {alloc})", e.message);
        }
        e
    };
    let in_range = |size: u64| byte >= 0 && (byte as u64).checked_add(len).is_some_and(|e| e <= size);
    match ctx.buffers[buf.0 as usize] {
        BufBinding::View(v) => {
            if !in_range(v.len) {
                return Err(oob(ctx, v.alloc, v.len));
            }
            let a = ctx.arena.get(v.alloc);
            let window = match a.space {
                Space::Global => Some(Window::Global),
                Space::Shared => Some(Window::SharedCta),
                _ => None,
            };
            Ok(Loc { alloc: v.alloc, offset: v.offset + byte as u64, window, remote: None })
        }
        BufBinding::SharedWindow { offset, len: blen } => {
            if !in_range(blen) {
                return Err(oob(ctx, ctx.cta.smem, blen));
            }
            let off = offset as u64 + byte as u64;
            bounds(ctx, ctx.cta.smem, off, len, lane)?;
            Ok(Loc { alloc: ctx.cta.smem, offset: off, window: Some(Window::SharedCta), remote: None })
        }
        BufBinding::Local { offset, per_lane } => {
            let Some(alloc) = ctx.warp.local else {
                return Err(err(ctx, ExecErrorKind::Internal, lanes, "local buffer without local allocation"));
            };
            if !in_range(per_lane) {
                return Err(oob(ctx, alloc, per_lane));
            }
            let off = lane as u64 * ctx.loaded.local_per_lane + offset + byte as u64;
            Ok(Loc { alloc, offset: off, window: None, remote: None })
        }
        BufBinding::Reg { offset, per_lane } => {
            let Some(alloc) = ctx.warp.regbuf else {
                return Err(err(ctx, ExecErrorKind::Internal, lanes, "register buffer without its allocation"));
            };
            if !in_range(per_lane) {
                return Err(oob(ctx, alloc, per_lane));
            }
            let off = lane as u64 * ctx.loaded.reg_per_lane + offset + byte as u64;
            Ok(Loc { alloc, offset: off, window: None, remote: None })
        }
        BufBinding::Tmem { base_col, cols, base_reg } => {
            let (base_lane, base_col) = match base_reg {
                Some(r) => {
                    let t = uniform_over(ctx, Operand::Reg(r), ctx.warp.active)? as u32;
                    let (l, c) = addr::tmem_decode(t);
                    (l, base_col + c)
                }
                None => (0, base_col),
            };
            let b = byte;
            // 8/16-bit elements pack `32 / bits` per 32-bit cell: element
            // `idx` is in cell `idx / per_cell` at byte `(idx % per_cell) *
            // bits / 8` (contract batch 4); the access must stay in its cell
            // (a sub-word store is then a read-modify-write of the cell).
            let bits = decl.dtype.elem.bits();
            if matches!(bits, 8 | 16) && len > 4 {
                return Err(unsupported(ctx, &format!("{}[{idx}]: a sub-word tmem vector spanning cells is not modelled", decl.name)));
            }
            if !matches!(bits, 8 | 16 | 32) {
                return Err(unsupported(ctx, &format!("{}[{idx}]: {bits}-bit tmem elements are not modelled", decl.name)));
            }
            let packed = matches!(bits, 8 | 16) && len < 4;
            let sub = if packed { b.rem_euclid(4) as u64 } else { 0 };
            let bad = if packed { b < 0 || sub + len > 4 } else { b < 0 || b % 4 != 0 || !len.is_multiple_of(4) };
            if bad {
                return Err(err(ctx, ExecErrorKind::Misaligned, lanes, format!("{}[{idx}]: tmem access is not 32-bit aligned", decl.name)));
            }
            let word = (b / 4) as u64;
            // `cols` is the row length in 32-bit cells (the view's last
            // dimension, contract batch 4): element `idx` of an 8/16-bit view
            // is lane `idx / (cols * per_cell)`, cell `(idx % (cols *
            // per_cell)) / per_cell` — the byte offset `b` / 4 already is
            // that linear cell index.
            let cols = cols.max(1) as u64;
            let tl = ((base_lane as u64 + word / cols) % addr::TMEM_LANES as u64) as u32;
            let col = base_col as u64 + word % cols;
            let ncols = len.div_ceil(4).max(1);
            if word % cols + ncols > cols || col + ncols > addr::TMEM_COLS as u64 {
                return Err(unsupported(ctx, &format!("{}[{idx}]: tmem access crosses a lane row", decl.name)));
            }
            // A warp accesses only the 32 TMEM lanes of its sub-partition
            // (warp id % 4): anything else is a kernel bug.
            if tl / 32 != ctx.warp.warp_in_cta % 4 {
                return Err(err(
                    ctx,
                    ExecErrorKind::BadAddress,
                    lanes,
                    format!("{}[{idx}]: tmem lane {tl} is outside warp {}'s sub-partition", decl.name, ctx.warp.warp_in_cta),
                ));
            }
            if !tmem_live(ctx, col as u32, ncols as u32) {
                // A TMEM access outside every live tcgen05 allocation is a
                // kernel bug (allocation state is tracked exactly).
                return Err(err(
                    ctx,
                    ExecErrorKind::BadAddress,
                    lanes,
                    format!("{}[{idx}]: tmem column {col} is not in a live tcgen05 allocation", decl.name),
                ));
            }
            let off = addr::tmem_byte_offset(tl, col as u32) + sub;
            bounds(ctx, ctx.cta.tmem, off, len, lane)?;
            Ok(Loc { alloc: ctx.cta.tmem, offset: off, window: None, remote: None })
        }
        BufBinding::Unbound => Err(err(
            ctx,
            ExecErrorKind::BadAddress,
            lanes,
            format!("buffer {} is not bound (missing host argument?)", decl.name),
        )),
    }
}

/// Columns `[col, col+n)` lie in a live tcgen05 allocation of this CTA.
///
/// With `Requirements::implicit_tmem` (TMEM views without any
/// `tcgen05.alloc`, legacy semantics) the kernel owns the whole TMEM.
pub fn tmem_live(ctx: &ExecCtx<'_>, col: u32, n: u32) -> bool {
    if ctx.program.requirements.implicit_tmem {
        return col.checked_add(n).is_some_and(|e| e <= addr::TMEM_COLS);
    }
    let rank = ctx.cta.rank_in_cluster;
    let id = crate::sync::ResourceId::TcgenLifecycle { cluster: ctx.cta.cluster, pair_rank: (rank >> 1) as u8 };
    match ctx.sync.get(id) {
        Some(crate::sync::Resource::Tcgen(s)) => s.ctas[(rank & 1) as usize]
            .allocations
            .iter()
            .any(|a| a.base <= col && col + n <= a.base + a.columns),
        _ => false,
    }
}

/// Generic address of `buf` byte `byte` (pointer arithmetic, no bounds check).
pub fn buf_generic_addr(ctx: &ExecCtx<'_>, buf: Buf, byte: i64) -> Result<u64, ExecError> {
    let b = byte as u64;
    Ok(match ctx.buffers[buf.0 as usize] {
        BufBinding::View(v) => {
            let a = ctx.arena.get(v.alloc);
            match a.space {
                Space::Global => a.base.wrapping_add(v.offset).wrapping_add(b),
                Space::Param => GENERIC_PARAM_BASE.wrapping_add(v.offset).wrapping_add(b),
                Space::Shared => addr::GENERIC_SHARED_BASE.wrapping_add(own_shared(ctx, v.offset.wrapping_add(b))),
                _ => return Err(unsupported(ctx, "address of a non-addressable buffer")),
            }
        }
        BufBinding::SharedWindow { offset, .. } => {
            addr::GENERIC_SHARED_BASE.wrapping_add(own_shared(ctx, (offset as u64).wrapping_add(b)))
        }
        BufBinding::Local { offset, .. } => addr::GENERIC_LOCAL_BASE.wrapping_add(offset).wrapping_add(b),
        BufBinding::Tmem { .. } => return Err(unsupported(ctx, "address of a tensor-memory buffer")),
        BufBinding::Reg { .. } => return Err(unsupported(ctx, "address of a register-space buffer")),
        BufBinding::Unbound => {
            let name = &ctx.program.buffers[buf.0 as usize].name;
            return Err(ctx.error(ExecErrorKind::BadAddress, format!("buffer {name} is not bound")));
        }
    })
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

#[inline]
pub fn whole(arena: &Arena, alloc: AllocId) -> View {
    View { alloc, offset: 0, len: arena.get(alloc).size }
}

/// Read `out.len()` bytes at `loc`.
#[inline]
pub fn mem_read(ctx: &mut ExecCtx<'_>, loc: Loc, lane: usize, out: &mut [u8]) -> Result<(), ExecError> {
    let span = loc.span(out.len() as u64);
    let v = whole(ctx.arena, loc.alloc);
    if ctx.arena.policy() == ValidityPolicy::ZeroAndReport && ctx.arena.first_invalid(v, span).is_some() {
        report_uninit(ctx, loc.alloc, span, lane);
    }
    if let Err(e) = ctx.arena.read(v, &[span], out) {
        return Err(arena_err(ctx, e, WarpMask::lane(lane)));
    }
    if let Some(c) = ctx.aux.capture_reads.as_mut() {
        c.push((loc.alloc, span));
    }
    Ok(())
}

/// Record one `UninitRead` review finding for a read range.
pub fn report_uninit(ctx: &mut ExecCtx<'_>, alloc: AllocId, span: ByteSpan, lane: usize) {
    let site = ctx.site();
    if !ctx.aux.uninit_seen.insert((site, alloc, span)) {
        return;
    }
    // Coalesce with the previous report of the same instruction and
    // allocation when the ranges touch (one finding per read range).
    if let Some(last) = ctx.aux.diagnostics.last_mut() {
        if let Some(ev) = last.evidence.first_mut() {
            if ev.site == site && ev.alloc == Some(alloc) && ev.kernel == ctx.aux.kernel {
                if let Some(b) = ev.bytes.as_mut() {
                    if span.start <= b.end() && b.start <= span.end() {
                        let start = b.start.min(span.start);
                        let end = b.end().max(span.end());
                        *b = ByteSpan::new(start, end - start);
                        let name = ctx.arena.get(alloc).name.clone();
                        last.message = format!("read of uninitialized bytes [{start}, {end}) of {name} (read as zero)");
                        return;
                    }
                }
            }
        }
    }
    let a = ctx.arena.get(alloc);
    let f = uninit_finding(ctx.aux.kernel, site, Some(ctx.actor()), a.space, alloc, &a.name, span, lane);
    ctx.aux.diagnostics.push(f);
}

pub fn uninit_finding(
    kernel: u32,
    site: SiteId,
    actor: Option<Actor>,
    space: Space,
    alloc: AllocId,
    name: &str,
    span: ByteSpan,
    lane: usize,
) -> crate::report::Finding {
    use crate::report::{Evidence, Finding, FindingKind, Status};
    Finding {
        kind: FindingKind::UninitRead,
        status: Status::Review,
        message: format!("read of uninitialized bytes [{}, {}) of {name} (read as zero)", span.start, span.end()),
        attrs: Default::default(),
        sites: if site.is_none() { vec![] } else { vec![site] },
        evidence: vec![Evidence {
            role: "read".into(),
            kernel,
            site,
            actor,
            buffer: Some(name.to_string()),
            space: Some(space),
            alloc: Some(alloc),
            bytes: Some(span),
            detail: Some(format!("lane {lane}")),
        }],
    }
}

/// Write `src` at `loc` (marks valid).
#[inline]
pub fn mem_write(ctx: &mut ExecCtx<'_>, loc: Loc, lane: usize, src: &[u8]) -> Result<(), ExecError> {
    let span = loc.span(src.len() as u64);
    if ctx.arena.get(loc.alloc).space == Space::Param {
        return Err(err(
            ctx,
            ExecErrorKind::BadAddress,
            WarpMask::lane(lane),
            format!("store to byte {} of the kernel parameter space, which is read-only", loc.offset),
        ));
    }
    readonly_write(ctx, loc.alloc, span, lane)?;
    let v = whole(ctx.arena, loc.alloc);
    if let Err(e) = ctx.arena.write(v, &[span], src) {
        return Err(arena_err(ctx, e, WarpMask::lane(lane)));
    }
    if !ctx.aux.words.is_empty() {
        ctx.aux.words.log_lane(loc.alloc, span, src);
    }
    Ok(())
}

/// Message of a readonly-proxy conflict (legacy wording).
pub fn readonly_conflict_message(name: &str, alloc: AllocId, byte: u64) -> String {
    format!("write overlaps readonly bytes (readonly-proxy load): global allocation {} ({name}), byte offset {byte}", alloc.0)
}

/// A global write of `span` of `alloc` by `lane`: an error when the kernel
/// read any of those bytes through the readonly proxy (PTX: such bytes stay
/// read-only for the whole kernel; legacy `ReadonlyProxyWriteConflict`).
#[inline]
pub fn readonly_write(ctx: &mut ExecCtx<'_>, alloc: AllocId, span: ByteSpan, lane: usize) -> Result<(), ExecError> {
    if !ctx.arena.readonly_tracking() {
        return Ok(());
    }
    match ctx.arena.note_global_write(alloc, span) {
        Ok(()) => Ok(()),
        Err(b) => {
            let m = readonly_conflict_message(&ctx.arena.get(alloc).name, alloc, b);
            Err(err(ctx, ExecErrorKind::BadAddress, WarpMask::lane(lane), m))
        }
    }
}

/// A readonly-proxy (`ld.global.nc`) read of `span` of `alloc` by `lane`:
/// an error when the kernel already wrote any of those bytes.
#[inline]
pub fn readonly_read(ctx: &mut ExecCtx<'_>, alloc: AllocId, span: ByteSpan, lane: usize) -> Result<(), ExecError> {
    if !ctx.arena.readonly_tracking() {
        return Ok(());
    }
    match ctx.arena.note_readonly_read(alloc, span) {
        Ok(()) => Ok(()),
        Err(b) => {
            let m = readonly_conflict_message(&ctx.arena.get(alloc).name, alloc, b);
            Err(err(ctx, ExecErrorKind::BadAddress, WarpMask::lane(lane), m))
        }
    }
}

/// What an access looks like apart from its spans.
#[derive(Clone, Copy, Debug)]
pub struct AccessSpec {
    pub actor: Actor,
    pub site: SiteId,
    pub kind: AccessKind,
    pub sem: Sem,
    pub scope: Scope,
    pub atomic: bool,
    pub returns_value: bool,
    pub proxy: Proxy,
    /// `observe::Access::operand`.
    pub operand: u8,
}

/// Per-instruction access collector: `(alloc, window, lane span)` in lane
/// order. Emits one `Access` per (alloc, window).
#[derive(Default)]
pub struct Accesses {
    pub items: Vec<(AllocId, Option<Window>, LaneSpan)>,
}

impl Accesses {
    #[inline]
    pub fn push(&mut self, loc: Loc, lane: u8, len: u64) {
        self.items.push((loc.alloc, loc.window, LaneSpan { lane, span: loc.span(len) }));
    }
}

/// Emit the collected accesses (after a write has been performed, so the
/// declared-word history can snapshot the post-image).
pub fn emit_accesses(
    observer: &mut dyn Observer,
    counters: &mut LaunchCounters,
    aux: &mut LaunchAux,
    arena: &Arena,
    spec: AccessSpec,
    acc: &mut Accesses,
) {
    if acc.items.is_empty() {
        return;
    }
    // Stable grouping by (alloc, window), preserving lane order inside.
    // Items are usually pushed in this order already (W13: skip the sort,
    // a stable sort leaves sorted input unchanged).
    if !acc.items.is_sorted_by_key(|a| (a.0, window_key(a.1), a.2)) {
        acc.items.sort_by_key(|a| (a.0, window_key(a.1), a.2));
    }
    let mut spans: Vec<LaneSpan> = Vec::with_capacity(acc.items.len());
    let mut i = 0;
    while i < acc.items.len() {
        let (alloc, window, _) = acc.items[i];
        spans.clear();
        let mut j = i;
        while j < acc.items.len() && acc.items[j].0 == alloc && acc.items[j].1 == window {
            spans.push(acc.items[j].2);
            j += 1;
        }
        let declared = aux.wants_history && !aux.words.is_empty() && {
            let raw: Vec<ByteSpan> = spans.iter().map(|s| s.span).collect();
            aux.words.overlaps(alloc, &raw)
        };
        let a = Access {
            seq: counters.next_access_seq(),
            actor: spec.actor,
            site: spec.site,
            alloc,
            space: arena.get(alloc).space,
            kind: spec.kind,
            sem: spec.sem,
            scope: spec.scope,
            atomic: spec.atomic,
            returns_value: spec.returns_value,
            proxy: spec.proxy,
            window,
            spans: &spans,
            declared_word: declared,
            operand: spec.operand,
        };
        observer.access(&a);
        i = j;
    }
    acc.items.clear();
}

/// Declared-word logging of an async landing's writes (async landings,
/// fills): one entry per span, in the order `emit_accesses` reports them
/// (sorted by allocation, window, lane span). Runs in every mode, so
/// declared-word counts and overflow never depend on the observer (W13-1);
/// warp writes are logged per lane at `mem_write`.
pub fn log_async_writes(aux: &mut LaunchAux, arena: &Arena, items: &mut [(AllocId, Option<Window>, LaneSpan)]) {
    if aux.words.is_empty() {
        return;
    }
    items.sort_by_key(|a| (a.0, window_key(a.1), a.2));
    for &(alloc, _, ls) in items.iter() {
        aux.words.log_from_arena(arena, alloc, ls.span);
    }
}

fn window_key(w: Option<Window>) -> u8 {
    match w {
        None => 0,
        Some(Window::Global) => 1,
        Some(Window::SharedCta) => 2,
        Some(Window::SharedCluster) => 3,
    }
}

/// Emit an instruction's accesses through `ctx`.
pub fn emit(ctx: &mut ExecCtx<'_>, spec: AccessSpec, acc: &mut Accesses) {
    if !ctx.observing {
        acc.items.clear();
        return;
    }
    emit_accesses(ctx.observer, ctx.counters, ctx.aux, ctx.arena, spec, acc);
}

/// Default spec for a warp instruction.
pub fn spec(ctx: &ExecCtx<'_>, kind: AccessKind, sem: Sem, scope: Scope, proxy: Proxy) -> AccessSpec {
    AccessSpec { actor: ctx.actor(), site: ctx.site(), kind, sem, scope, atomic: false, returns_value: false, proxy, operand: 0 }
}

// ---------------------------------------------------------------------------
// Sync events
// ---------------------------------------------------------------------------

/// Emit a non-protocol sync event for the current warp instruction.
pub fn sync_event(ctx: &mut ExecCtx<'_>, lanes: WarpMask, kind: SyncKind) {
    if !ctx.observing {
        return;
    }
    let e = SyncEvent {
        kernel: ctx.aux.kernel,
        actor: ctx.actor(),
        seq: ctx.warp.sync_seq,
        site: ctx.site(),
        frames: ctx.warp.loop_frames(ctx.program),
        lanes,
        kind,
    };
    ctx.observer.sync(&e);
}

/// Emit a sync event on behalf of async op `op` (write side), as its
/// landing does.
pub fn async_event(ctx: &mut ExecCtx<'_>, op: crate::sync::AsyncId, kind: SyncKind) {
    if !ctx.observing {
        return;
    }
    let e = SyncEvent {
        kernel: ctx.aux.kernel,
        actor: Actor::Async { op, side: crate::observe::Side::Write },
        seq: 0,
        site: ctx.site(),
        frames: Vec::new(),
        lanes: WarpMask::NONE,
        kind,
    };
    ctx.observer.sync(&e);
}

/// Everything of a `SyncKind::Protocol` event except the commands.
#[derive(Clone, Debug, Default)]
pub struct ProtoExtra {
    pub counts: Counts,
    pub collective: Option<Collective>,
    pub issued: Vec<AsyncTarget>,
    pub observed_parity: Option<u8>,
}

/// Emit a committed protocol event (increments the warp's protocol seq).
/// `extra.counts` / `observed_parity` apply to every target; use
/// [`protocol_cmds`] for per-target values.
pub fn protocol(ctx: &mut ExecCtx<'_>, lanes: WarpMask, cmds: Vec<(ResourceId, SyncCmd)>, extra: ProtoExtra) {
    protocol_status(ctx, lanes, cmds, extra, ProtocolStatus::Committed);
}

/// Emit a committed protocol event with per-target counts.
pub fn protocol_cmds(
    ctx: &mut ExecCtx<'_>,
    lanes: WarpMask,
    cmds: Vec<ProtocolCmd>,
    collective: Option<Collective>,
    issued: Vec<AsyncTarget>,
) {
    let seq = ctx.warp.sync_seq;
    ctx.warp.sync_seq += 1;
    if !ctx.observing {
        return;
    }
    let e = SyncEvent {
        kernel: ctx.aux.kernel,
        actor: ctx.actor(),
        seq,
        site: ctx.site(),
        frames: ctx.warp.loop_frames(ctx.program),
        lanes,
        kind: SyncKind::Protocol { cmds, collective, issued, status: ProtocolStatus::Committed },
    };
    ctx.observer.sync(&e);
}

pub fn protocol_status(
    ctx: &mut ExecCtx<'_>,
    lanes: WarpMask,
    cmds: Vec<(ResourceId, SyncCmd)>,
    extra: ProtoExtra,
    status: ProtocolStatus,
) {
    let seq = ctx.warp.sync_seq;
    ctx.warp.sync_seq += 1;
    if !ctx.observing {
        return;
    }
    let cmds = cmds
        .into_iter()
        .map(|(res, cmd)| ProtocolCmd { res, cmd, counts: extra.counts, observed_parity: extra.observed_parity })
        .collect();
    let e = SyncEvent {
        kernel: ctx.aux.kernel,
        actor: ctx.actor(),
        seq,
        site: ctx.site(),
        frames: ctx.warp.loop_frames(ctx.program),
        lanes,
        kind: SyncKind::Protocol { cmds, collective: extra.collective, issued: extra.issued, status },
    };
    ctx.observer.sync(&e);
}

/// Step one command; on error emit a `Failed` protocol event and return the
/// kernel error.
pub fn step(
    ctx: &mut ExecCtx<'_>,
    res: ResourceId,
    cmd: SyncCmd,
) -> Result<crate::sync::Step<crate::sync::Outcome>, ExecError> {
    match ctx.sync.step(res, cmd) {
        Ok(s) => Ok(s),
        Err(e) => {
            let lanes = ctx.active();
            protocol_status(ctx, lanes, vec![(res, cmd)], ProtoExtra::default(), ProtocolStatus::Failed(e.clone()));
            Err(sync_err(ctx, e))
        }
    }
}

/// All-or-nothing multi-target step with the same error reporting.
pub fn step_all(
    ctx: &mut ExecCtx<'_>,
    cmds: &[(ResourceId, SyncCmd)],
) -> Result<crate::sync::Step<Vec<crate::sync::Outcome>>, ExecError> {
    match ctx.sync.step_all(cmds) {
        Ok(s) => Ok(s),
        Err(e) => {
            let lanes = ctx.active();
            protocol_status(ctx, lanes, cmds.to_vec(), ProtoExtra::default(), ProtocolStatus::Failed(e.clone()));
            Err(sync_err(ctx, e))
        }
    }
}

/// Lane-uniform value of an operand over `mask` (Divergence otherwise).
pub fn uniform_over(ctx: &ExecCtx<'_>, o: Operand, mask: WarpMask) -> Result<u64, ExecError> {
    let Some(first) = mask.first() else { return Ok(lane_val(ctx, o, 0)) };
    let v = lane_val(ctx, o, first);
    for l in mask.lanes() {
        if lane_val(ctx, o, l) != v {
            return Err(err(ctx, ExecErrorKind::Divergence, mask, format!("operand {o} is not uniform across lanes")));
        }
    }
    Ok(v)
}
