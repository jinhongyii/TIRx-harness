//! Loads, stores, atomics and address arithmetic.
//!
//! Every lane resolves its own location (exact OOB per buffer / allocation);
//! accesses are collected per lane and emitted as one `Access` per
//! allocation after the memory effect (racecheck-semantics §3 rows 1-2,
//! 24-26: plain accesses carry their `sem`/`scope`; atomics are `Rmw`).
//! Remote shared::cluster stores are delivered through the outbox.

use super::HResult;
use crate::arena::{addr, Space};
use crate::dtype::{Dtype, Ty};
use crate::interp::support::{self, lane_bytes, lane_int, lane_val, write_lane, write_lane_bytes, Accesses, Loc};
use crate::interp::{ExecCtx, ExecError, ExecErrorKind, Flow};
use crate::observe::AccessKind;
use crate::oplib;
use crate::program::*;
use crate::value::WarpMask;

/// Natural alignment requirement of an `n`-byte access.
#[inline]
pub fn check_align(ctx: &ExecCtx<'_>, loc: Loc, n: u64, lane: usize) -> Result<(), ExecError> {
    let align = n.next_power_of_two().clamp(1, 32);
    let base = ctx.arena.get(loc.alloc).base;
    if (base.wrapping_add(loc.offset)) % align != 0 {
        return Err(support::err(
            ctx,
            ExecErrorKind::Misaligned,
            WarpMask::lane(lane),
            format!("{n}-byte access at offset {} of {} is not {align}-byte aligned", loc.offset, ctx.arena.get(loc.alloc).name),
        ));
    }
    Ok(())
}

/// A non-weak load inside a loop polls memory another actor may change
/// (`while (ld.acquire(flag) == 0)`): record it for spin parking, which
/// still requires the loop to reach a register fixed point (control.rs).
#[inline]
fn note_load_poll(ctx: &mut ExecCtx<'_>, sem: Sem, alloc: crate::arena::AllocId, offset: u64) {
    if sem != Sem::Weak && ctx.warp.frames.iter().any(|f| matches!(f.kind, crate::interp::FrameKind::Loop { .. })) {
        ctx.note_failed_poll(crate::sync::ResourceId::Word { alloc, offset });
    }
}

/// `ld.global.nc ... ldu`-style uniform loads (`MemMods::uniform`): every
/// active lane must read the same address (PTX `ldu`).
#[inline]
fn check_uniform(ctx: &ExecCtx<'_>, mods: &MemMods, first: Loc, loc: Loc, lane: usize) -> Result<(), ExecError> {
    if mods.uniform && (first.alloc != loc.alloc || first.offset != loc.offset) {
        return Err(support::err(
            ctx,
            ExecErrorKind::Op(crate::oplib::OpErrorKind::Invalid),
            WarpMask::lane(lane),
            "ldu: the address is not uniform across the active lanes",
        ));
    }
    Ok(())
}

#[inline]
fn load_proxy(mods: &MemMods) -> Proxy {
    if mods.nc {
        Proxy::ReadOnly
    } else {
        Proxy::Generic
    }
}

/// Fast-path target of a buffer access: the allocation, the buffer's byte
/// base in it, its byte length, and the window. `None` = use the general
/// path (local/tmem/unbound buffers, overlaid allocations, sub-byte dtypes).
#[inline]
fn fast_target(ctx: &ExecCtx<'_>, buf: Buf) -> Option<(crate::arena::AllocId, u64, u64, Option<crate::observe::Window>)> {
    use crate::interp::BufBinding;
    if ctx.program.buffers[buf.0 as usize].dtype.bits() % 8 != 0 {
        return None;
    }
    let (alloc, base, len) = match ctx.buffers[buf.0 as usize] {
        BufBinding::View(v) => (v.alloc, v.offset, v.len),
        BufBinding::SharedWindow { offset, len } => (ctx.cta.smem, offset as u64, len),
        _ => return None,
    };
    // Overlaid allocations and wait_until predicate evaluation (which
    // records the bytes it read) take the general path.
    if ctx.arena.is_overlaid(alloc) || ctx.aux.capture_reads.is_some() {
        return None;
    }
    let a = ctx.arena.get(alloc);
    if a.metadata_only || a.space == crate::arena::Space::Param || base.checked_add(len).is_none_or(|e| e > a.size) {
        return None;
    }
    let window = match a.space {
        crate::arena::Space::Global => Some(crate::observe::Window::Global),
        crate::arena::Space::Shared => Some(crate::observe::Window::SharedCta),
        _ => None,
    };
    Some((alloc, base, len, window))
}

/// Byte offset of `buf[idx]` for an `n`-byte access within `len`, or
/// `None` (out of bounds / misaligned: the general path reports it).
#[inline(always)]
fn fast_offset(ctx: &ExecCtx<'_>, buf: Buf, idx: i64, n: u64, len: u64, abs_base: u64) -> Option<u64> {
    let eb = (ctx.program.buffers[buf.0 as usize].dtype.bits() / 8) as i64;
    let byte = idx.checked_mul(eb)?;
    if byte < 0 || (byte as u64).checked_add(n)? > len {
        return None;
    }
    let off = abs_base + byte as u64;
    let align = n.next_power_of_two().clamp(1, 32);
    (off % align == 0).then_some(off)
}

#[inline]
pub fn load(ctx: &mut ExecCtx<'_>, ty: Ty, dst: Reg, buf: Buf, offset: Operand, sem: Sem, scope: Scope, mods: MemMods) -> HResult {
    active_or_next!(ctx);
    let n = ty.mem_bytes() as u64;
    if n <= 8 && ty.slots() == 1 && !mods.uniform && !(mods.nc && ctx.arena.readonly_tracking()) {
        if let Some((alloc, base, len, window)) = fast_target(ctx, buf) {
            let active = ctx.warp.active;
            let mut vals = [0u64; 32];
            let mut ok = true;
            {
                let a = ctx.arena.get(alloc);
                let allow = ctx.arena.policy() == crate::arena::ValidityPolicy::Allow;
                for l in active.lanes() {
                    let idx = lane_int(ctx, offset, l);
                    match fast_offset(ctx, buf, idx, n, len, base) {
                        Some(off) if allow || a.valid.first_clear(off, n).is_none() => {
                            let mut b = [0u8; 8];
                            b[..n as usize].copy_from_slice(&a.bytes[off as usize..(off + n) as usize]);
                            vals[l] = u64::from_le_bytes(b);
                        }
                        _ => {
                            ok = false;
                            break;
                        }
                    }
                }
            }
            if ok {
                let s = ctx.slot(dst);
                support::write_masked(ctx.warp.regs.get_mut(s), &vals, active);
                if sem != Sem::Weak {
                    let l0 = active.lanes().next().expect("active");
                    let off = fast_offset(ctx, buf, lane_int(ctx, offset, l0), n, len, base).expect("checked");
                    note_load_poll(ctx, sem, alloc, off);
                }
                if ctx.observing {
                    let mut acc = Accesses::default();
                    for l in active.lanes() {
                        let idx = lane_int(ctx, offset, l);
                        let off = fast_offset(ctx, buf, idx, n, len, base).expect("checked");
                        acc.push(Loc { alloc, offset: off, window, remote: None }, l as u8, n);
                    }
                    let sp = support::spec(ctx, AccessKind::Read, sem, scope, load_proxy(&mods));
                    support::emit(ctx, sp, &mut acc);
                }
                return Ok(Flow::Next);
            }
            // Some lane needs the general path (error / uninit report).
        }
    }
    if let Some(eb) = sub_byte(ctx, ty, buf) {
        return load_sub_byte(ctx, dst, buf, offset, sem, scope, mods, eb);
    }
    let n = ty.mem_bytes() as u64;
    let mut acc = Accesses::default();
    let mut bytes = [0u8; 32];
    let mut first = None;
    for l in ctx.warp.active.lanes() {
        let idx = lane_int(ctx, offset, l);
        let loc = support::resolve_buf(ctx, buf, idx, l, n)?;
        check_align(ctx, loc, n, l)?;
        support::mem_read(ctx, loc, l, &mut bytes[..n as usize])?;
        if mods.nc {
            support::readonly_read(ctx, loc.alloc, loc.span(n), l)?;
        }
        write_lane_bytes(ctx, dst, l, &bytes[..n as usize]);
        check_uniform(ctx, &mods, *first.get_or_insert(loc), loc, l)?;
        if ctx.observing {
            acc.push(loc, l as u8, n);
        }
    }
    if let Some(loc) = first {
        note_load_poll(ctx, sem, loc.alloc, loc.offset);
    }
    let proxy = if matches!(ctx.buffers[buf.0 as usize], crate::interp::BufBinding::Tmem { .. }) { Proxy::Tcgen } else { load_proxy(&mods) };
    let sp = support::spec(ctx, AccessKind::Read, sem, scope, proxy);
    support::emit(ctx, sp, &mut acc);
    Ok(Flow::Next)
}

#[inline]
pub fn store(ctx: &mut ExecCtx<'_>, ty: Ty, buf: Buf, offset: Operand, value: Operand, sem: Sem, scope: Scope, mods: MemMods) -> HResult {
    active_or_next!(ctx);
    let _ = mods;
    let n = ty.mem_bytes() as u64;
    if n <= 8 && ty.slots() == 1 && !(ctx.aux.wants_history && !ctx.aux.words.is_empty()) && !ctx.arena.readonly_tracking() {
        if let Some((alloc, base, len, window)) = fast_target(ctx, buf) {
            let active = ctx.warp.active;
            let mut offs = [0u64; 32];
            let mut ok = true;
            for l in active.lanes() {
                let idx = lane_int(ctx, offset, l);
                match fast_offset(ctx, buf, idx, n, len, base) {
                    Some(off) => offs[l] = off,
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                let vals = ctx.read(value);
                {
                    let a = ctx.arena.get_mut(alloc);
                    for l in active.lanes() {
                        let off = offs[l] as usize;
                        a.bytes[off..off + n as usize].copy_from_slice(&vals[l].to_le_bytes()[..n as usize]);
                        a.valid.set_range(offs[l], n, true);
                    }
                }
                if ctx.observing {
                    let mut acc = Accesses::default();
                    for l in active.lanes() {
                        acc.push(Loc { alloc, offset: offs[l], window, remote: None }, l as u8, n);
                    }
                    let sp = support::spec(ctx, AccessKind::Write, sem, scope, Proxy::Generic);
                    support::emit(ctx, sp, &mut acc);
                }
                return Ok(Flow::Next);
            }
        }
    }
    if let Some(eb) = sub_byte(ctx, ty, buf) {
        return store_sub_byte(ctx, ty, buf, offset, value, sem, scope, eb);
    }
    let n = ty.mem_bytes() as u64;
    let mut acc = Accesses::default();
    let mut bytes = [0u8; 32];
    for l in ctx.warp.active.lanes() {
        let idx = lane_int(ctx, offset, l);
        let loc = support::resolve_buf(ctx, buf, idx, l, n)?;
        check_align(ctx, loc, n, l)?;
        lane_bytes(ctx, value, ty, l, &mut bytes);
        support::mem_write(ctx, loc, l, &bytes[..n as usize])?;
        if ctx.observing {
            acc.push(loc, l as u8, n);
        }
    }
    let proxy = if matches!(ctx.buffers[buf.0 as usize], crate::interp::BufBinding::Tmem { .. }) { Proxy::Tcgen } else { Proxy::Generic };
    let sp = support::spec(ctx, AccessKind::Write, sem, scope, proxy);
    support::emit(ctx, sp, &mut acc);
    Ok(Flow::Next)
}

/// Element width of a scalar access to a sub-byte buffer (`float4`,
/// `int4`, ...): legacy packs `8 / bits` elements per byte, element `i` at
/// bit `i * bits` (low bits first). `None` for every other access.
fn sub_byte(ctx: &ExecCtx<'_>, ty: Ty, buf: Buf) -> Option<u32> {
    let eb = ctx.program.buffers[buf.0 as usize].dtype.elem.bits();
    (eb < 8 && 8 % eb == 0 && ty.slots() == 1 && ty.elem.bits() == eb).then_some(eb)
}

/// The byte holding sub-byte element `idx` (resolved as the byte-aligned
/// element that starts it) and the element's bit shift in it.
fn sub_byte_loc(ctx: &ExecCtx<'_>, buf: Buf, idx: i64, lane: usize, eb: u32) -> Result<(Loc, u32), ExecError> {
    let per = (8 / eb) as i64;
    let first = idx.div_euclid(per) * per;
    let loc = support::resolve_buf(ctx, buf, first, lane, 1)?;
    Ok((loc, (idx.rem_euclid(per) as u32) * eb))
}

#[allow(clippy::too_many_arguments)]
fn load_sub_byte(ctx: &mut ExecCtx<'_>, dst: Reg, buf: Buf, offset: Operand, sem: Sem, scope: Scope, mods: MemMods, eb: u32) -> HResult {
    let mut acc = Accesses::default();
    for l in ctx.warp.active.lanes() {
        let idx = lane_int(ctx, offset, l);
        let (loc, shift) = sub_byte_loc(ctx, buf, idx, l, eb)?;
        let mut b = [0u8; 1];
        support::mem_read(ctx, loc, l, &mut b)?;
        if mods.nc {
            support::readonly_read(ctx, loc.alloc, loc.span(1), l)?;
        }
        let v = (b[0] >> shift) & ((1u16 << eb) - 1) as u8;
        write_lane_bytes(ctx, dst, l, &[v]);
        if ctx.observing {
            acc.push(loc, l as u8, 1);
        }
    }
    let sp = support::spec(ctx, AccessKind::Read, sem, scope, load_proxy(&mods));
    support::emit(ctx, sp, &mut acc);
    Ok(Flow::Next)
}

/// Sub-byte store: read-modify-write of the holding byte, in lane order (two
/// lanes storing elements of one byte both land).
#[allow(clippy::too_many_arguments)]
fn store_sub_byte(ctx: &mut ExecCtx<'_>, ty: Ty, buf: Buf, offset: Operand, value: Operand, sem: Sem, scope: Scope, eb: u32) -> HResult {
    let mut acc = Accesses::default();
    let mask = ((1u16 << eb) - 1) as u8;
    for l in ctx.warp.active.lanes() {
        let idx = lane_int(ctx, offset, l);
        let (loc, shift) = sub_byte_loc(ctx, buf, idx, l, eb)?;
        let old = ctx.arena.read_raw(loc.alloc, loc.span(1))[0];
        let mut v = [0u8; 32];
        lane_bytes(ctx, value, ty, l, &mut v);
        let new = (old & !(mask << shift)) | ((v[0] & mask) << shift);
        support::mem_write(ctx, loc, l, &[new])?;
        if ctx.observing {
            acc.push(loc, l as u8, 1);
        }
    }
    let sp = support::spec(ctx, AccessKind::Write, sem, scope, Proxy::Generic);
    support::emit(ctx, sp, &mut acc);
    Ok(Flow::Next)
}

#[inline]
pub fn load_addr(ctx: &mut ExecCtx<'_>, ty: Ty, dst: Reg, a: Operand, space: AddrSpace, sem: Sem, scope: Scope, mods: MemMods) -> HResult {
    active_or_next!(ctx);
    let n = ty.mem_bytes() as u64;
    let mut acc = Accesses::default();
    let mut bytes = [0u8; 32];
    let mut first = None;
    for l in ctx.warp.active.lanes() {
        let v = lane_val(ctx, a, l);
        let loc = support::resolve(ctx, space, v, l, n)?;
        check_align(ctx, loc, n, l)?;
        support::mem_read(ctx, loc, l, &mut bytes[..n as usize])?;
        if mods.nc {
            support::readonly_read(ctx, loc.alloc, loc.span(n), l)?;
        }
        write_lane_bytes(ctx, dst, l, &bytes[..n as usize]);
        check_uniform(ctx, &mods, *first.get_or_insert(loc), loc, l)?;
        if ctx.observing {
            acc.push(loc, l as u8, n);
        }
    }
    if let Some(loc) = first {
        note_load_poll(ctx, sem, loc.alloc, loc.offset);
    }
    let sp = support::spec(ctx, AccessKind::Read, sem, scope, load_proxy(&mods));
    support::emit(ctx, sp, &mut acc);
    Ok(Flow::Next)
}

#[inline]
pub fn store_addr(ctx: &mut ExecCtx<'_>, ty: Ty, a: Operand, space: AddrSpace, value: Operand, sem: Sem, scope: Scope, mods: MemMods) -> HResult {
    active_or_next!(ctx);
    let _ = mods;
    let n = ty.mem_bytes() as u64;
    let mut acc = Accesses::default();
    let mut bytes = [0u8; 32];
    for l in ctx.warp.active.lanes() {
        let v = lane_val(ctx, a, l);
        let loc = support::resolve(ctx, space, v, l, n)?;
        check_align(ctx, loc, n, l)?;
        lane_bytes(ctx, value, ty, l, &mut bytes);
        support::mem_write(ctx, loc, l, &bytes[..n as usize])?;
        if ctx.observing {
            acc.push(loc, l as u8, n);
        }
    }
    let sp = support::spec(ctx, AccessKind::Write, sem, scope, Proxy::Generic);
    support::emit(ctx, sp, &mut acc);
    Ok(Flow::Next)
}

#[inline]
pub fn addr_of(ctx: &mut ExecCtx<'_>, dst: Reg, buf: Buf, offset: Operand) -> HResult {
    active_or_next!(ctx);
    let bits = ctx.program.buffers[buf.0 as usize].dtype.bits() as i64;
    for l in ctx.warp.active.lanes() {
        let idx = lane_int(ctx, offset, l);
        let byte = idx.wrapping_mul(bits).div_euclid(8);
        let a = support::buf_generic_addr(ctx, buf, byte)?;
        write_lane(ctx, dst, l, a);
    }
    Ok(Flow::Next)
}

/// Flush an f32 subnormal to a sign-preserving zero.
#[inline]
fn ftz32(bits: u64) -> u64 {
    let b = bits as u32;
    if b & 0x7f80_0000 == 0 {
        (b & 0x8000_0000) as u64
    } else {
        bits
    }
}

/// One element of an atomic read-modify-write.
pub fn rmw_elem(op: AtomOp, elem: Dtype, old: u64, val: u64, cmp: u64, ftz: bool) -> oplib::OpResult<u64> {
    let bits = elem.bits();
    let m = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
    let bin = |b: BinOp, x: u64, y: u64| -> oplib::OpResult<u64> {
        let a = [[x; 32]];
        let c = [[y; 32]];
        let mut out = [[0u64; 32]];
        oplib::binary(b, Ty::scalar(elem), &a, &c, &mut out, WarpMask::lane(0))?;
        Ok(out[0][0])
    };
    Ok(match op {
        AtomOp::Add => {
            if ftz && elem == Dtype::F32 {
                ftz32(bin(BinOp::Add, ftz32(old), ftz32(val))?)
            } else {
                bin(BinOp::Add, old, val)?
            }
        }
        AtomOp::Min => bin(BinOp::Min, old, val)?,
        AtomOp::Max => bin(BinOp::Max, old, val)?,
        AtomOp::And => old & val,
        AtomOp::Or => old | val,
        AtomOp::Xor => old ^ val,
        AtomOp::Exch => val,
        AtomOp::Cas => {
            if old & m == cmp & m {
                val
            } else {
                old
            }
        }
        AtomOp::Inc => {
            if old & m >= val & m {
                0
            } else {
                old + 1
            }
        }
        AtomOp::Dec => {
            if old & m == 0 || old & m > val & m {
                val
            } else {
                old - 1
            }
        }
    } & m)
}

/// Apply `op` element-wise over little-endian byte buffers (`dst = op(dst, src)`).
pub fn rmw_bytes(op: AtomOp, elem: Dtype, dst: &mut [u8], src: &[u8], cmp: &[u8], ftz: bool) -> oplib::OpResult {
    let eb = elem.mem_bytes() as usize;
    if eb > 8 {
        // `.b128` atomics (PTX §9.7.13.5): only exch and cas exist; both
        // are bytewise on whole 16-byte elements.
        let mut i = 0;
        while i + eb <= dst.len() {
            match op {
                AtomOp::Exch => dst[i..i + eb].copy_from_slice(&src[i..i + eb]),
                AtomOp::Cas => {
                    if cmp.len() >= i + eb && dst[i..i + eb] == cmp[i..i + eb] {
                        dst[i..i + eb].copy_from_slice(&src[i..i + eb]);
                    }
                }
                _ => return Err(oplib::OpError::unsupported(format!("atom.{op:?} on a {eb}-byte element (only exch/cas have .b128 forms)"))),
            }
            i += eb;
        }
        return Ok(());
    }
    let get = |b: &[u8], i: usize| {
        let mut w = [0u8; 8];
        w[..eb].copy_from_slice(&b[i..i + eb]);
        u64::from_le_bytes(w)
    };
    let mut i = 0;
    while i + eb <= dst.len() {
        let c = if cmp.len() >= i + eb { get(cmp, i) } else { 0 };
        let v = rmw_elem(op, elem, get(dst, i), get(src, i), c, ftz)?;
        dst[i..i + eb].copy_from_slice(&v.to_le_bytes()[..eb]);
        i += eb;
    }
    Ok(())
}

#[inline]
pub fn atom(
    ctx: &mut ExecCtx<'_>,
    op: AtomOp,
    ty: Ty,
    dst: Option<Reg>,
    a: Operand,
    space: AddrSpace,
    value: Operand,
    cmp: Option<Operand>,
    sem: Sem,
    scope: Scope,
    ftz: bool,
) -> HResult {
    active_or_next!(ctx);
    let n = ty.mem_bytes() as u64;
    if ty.elem.bits() < 8 {
        return Err(support::unsupported(ctx, "sub-byte atomics"));
    }
    // A read-modify-write of memory shared across partitions (global) is
    // a serial point inside an arena shard: the scheduler re-executes this
    // instruction after the round's merge, in partition order, so no
    // update is lost and results do not depend on the worker count.
    if ctx.arena.is_shard() {
        for l in ctx.warp.active.lanes() {
            let v = lane_val(ctx, a, l);
            let loc = support::resolve(ctx, space, v, l, n)?;
            if ctx.arena.is_overlaid(loc.alloc) {
                ctx.aux.serial_request = true;
                return Ok(Flow::Yield(ctx.pc()));
            }
        }
    }
    let mut acc = Accesses::default();
    let mut old = [0u8; 32];
    for l in ctx.warp.active.lanes() {
        let v = lane_val(ctx, a, l);
        let loc = support::resolve(ctx, space, v, l, n)?;
        check_align(ctx, loc, n, l)?;
        let k = n as usize;
        support::mem_read(ctx, loc, l, &mut old[..k])?;
        let mut val = [0u8; 32];
        lane_bytes(ctx, value, ty, l, &mut val);
        let mut c = [0u8; 32];
        if let Some(cmp) = cmp {
            lane_bytes(ctx, cmp, ty, l, &mut c);
        }
        let mut new = old;
        // Legacy `atomic_f32(.., space)` (W4): only global (incl. generic
        // resolving to global) `.add.f32` without `.noftz` flushes subnormals;
        // shared-memory float atomics keep them.
        let ftz = ftz && ctx.arena.get(loc.alloc).space != crate::arena::Space::Shared;
        rmw_bytes(op, ty.elem, &mut new[..k], &val[..k], &c[..k], ftz).map_err(|e| support::op_err(ctx, e))?;
        support::mem_write(ctx, loc, l, &new[..k])?;
        if let Some(d) = dst {
            write_lane_bytes(ctx, d, l, &old[..k]);
        }
        if ctx.observing {
            acc.push(loc, l as u8, n);
        }
    }
    let sem = if sem == Sem::Weak { Sem::Relaxed } else { sem };
    let mut sp = support::spec(ctx, AccessKind::Rmw, sem, scope, Proxy::Generic);
    sp.atomic = true;
    sp.returns_value = dst.is_some();
    support::emit(ctx, sp, &mut acc);
    Ok(Flow::Next)
}

/// Largest `st.bulk` byte count (legacy `memory_ops.rs`).
const ST_BULK_MAX: i64 = 16 * 1024 * 1024;

#[inline]
pub fn st_bulk(ctx: &mut ExecCtx<'_>, a: Operand, space: AddrSpace, size: Operand) -> HResult {
    active_or_next!(ctx);
    let mut acc = Accesses::default();
    for l in ctx.warp.active.lanes() {
        let v = lane_val(ctx, a, l);
        // The size operand is signed in the 64-bit forms: read it as i64.
        let n_signed = lane_int(ctx, size, l);
        if n_signed < 0 || n_signed % 8 != 0 || n_signed > ST_BULK_MAX {
            return Err(support::err(
                ctx,
                ExecErrorKind::Op(crate::oplib::OpErrorKind::Invalid),
                WarpMask::lane(l),
                format!("st.bulk byte count {n_signed} must be a multiple of 8 with maximum {ST_BULK_MAX} on lane {l}"),
            ));
        }
        let n = n_signed as u64;
        if v % 8 != 0 {
            return Err(support::err(ctx, ExecErrorKind::Misaligned, WarpMask::lane(l), format!("st.bulk address {v:#x} is not 8-byte aligned on lane {l}")));
        }
        let loc = support::resolve(ctx, space, v, l, n)?;
        support::readonly_write(ctx, loc.alloc, loc.span(n), l)?;
        let view = support::whole(ctx.arena, loc.alloc);
        if let Err(e) = ctx.arena.fill(view, &[loc.span(n)], 0) {
            return Err(support::arena_err(ctx, e, WarpMask::lane(l)));
        }
        if ctx.aux.wants_history {
            ctx.aux.words.log_from_arena(ctx.arena, loc.alloc, loc.span(n));
        }
        if ctx.observing {
            acc.push(loc, l as u8, n);
        }
    }
    let sp = support::spec(ctx, AccessKind::Write, Sem::Weak, Scope::Cta, Proxy::Generic);
    support::emit(ctx, sp, &mut acc);
    Ok(Flow::Next)
}

#[inline]
pub fn discard(ctx: &mut ExecCtx<'_>, a: Operand, space: AddrSpace, size: u32) -> HResult {
    active_or_next!(ctx);
    let mut acc = Accesses::default();
    for l in ctx.warp.active.lanes() {
        let v = lane_val(ctx, a, l);
        // `discard.L2 [a], 128` works on one 128-byte line (legacy check).
        if v % 128 != 0 {
            return Err(support::err(
                ctx,
                ExecErrorKind::Misaligned,
                WarpMask::lane(l),
                format!("discard requires a 128-byte aligned address (got {v:#x} on lane {l})"),
            ));
        }
        let loc = support::resolve(ctx, space, v, l, size as u64)?;
        let view = support::whole(ctx.arena, loc.alloc);
        if let Err(e) = ctx.arena.invalidate(view, &[loc.span(size as u64)]) {
            return Err(support::arena_err(ctx, e, WarpMask::lane(l)));
        }
        if ctx.aux.wants_history {
            ctx.aux.words.log_from_arena(ctx.arena, loc.alloc, loc.span(size as u64));
        }
        if ctx.observing {
            acc.push(loc, l as u8, size as u64);
        }
    }
    let sp = support::spec(ctx, AccessKind::Write, Sem::Weak, Scope::Gpu, Proxy::Generic);
    support::emit(ctx, sp, &mut acc);
    Ok(Flow::Next)
}

/// Generic address of a `space`-relative address, or None if the space has
/// no generic window in this model.
fn to_generic(ctx: &ExecCtx<'_>, space: AddrSpace, a: u64) -> Result<u64, ExecError> {
    Ok(match space {
        AddrSpace::Global | AddrSpace::Generic => a,
        AddrSpace::Shared | AddrSpace::SharedCluster => addr::generic_from_shared(a as u32),
        AddrSpace::Local => addr::generic_from_local(a as u32),
        AddrSpace::Param => support::GENERIC_PARAM_BASE + a,
        AddrSpace::Const | AddrSpace::Tmem => return Err(support::unsupported(ctx, "cvta of const/tmem")),
    })
}

/// `space`-relative address of a generic address (undefined result, not a
/// fault, when the address is in another window: wrapping difference).
fn from_generic(ctx: &ExecCtx<'_>, space: AddrSpace, a: u64) -> Result<u64, ExecError> {
    Ok(match space {
        AddrSpace::Global | AddrSpace::Generic => a,
        AddrSpace::Shared | AddrSpace::SharedCluster => a.wrapping_sub(addr::GENERIC_SHARED_BASE) & 0xffff_ffff,
        AddrSpace::Local => a.wrapping_sub(addr::GENERIC_LOCAL_BASE) & 0xffff_ffff,
        AddrSpace::Param => a.wrapping_sub(support::GENERIC_PARAM_BASE),
        AddrSpace::Const | AddrSpace::Tmem => return Err(support::unsupported(ctx, "cvta to const/tmem")),
    })
}

#[inline]
pub fn cvta(ctx: &mut ExecCtx<'_>, dst: Reg, src: Operand, space: AddrSpace, to_generic: bool) -> HResult {
    active_or_next!(ctx);
    for l in ctx.warp.active.lanes() {
        let a = lane_val(ctx, src, l);
        let v = if to_generic { self::to_generic(ctx, space, a)? } else { from_generic(ctx, space, a)? };
        write_lane(ctx, dst, l, v);
    }
    Ok(Flow::Next)
}

#[inline]
pub fn isspacep(ctx: &mut ExecCtx<'_>, dst: Reg, src: Operand, space: AddrSpace) -> HResult {
    active_or_next!(ctx);
    for l in ctx.warp.active.lanes() {
        let a = lane_val(ctx, src, l);
        let g = addr::classify_generic(a);
        let in_param = matches!(g, addr::Generic::Param(_));
        let yes = match space {
            AddrSpace::Global => matches!(g, addr::Generic::Global(_)),
            // `.shared::cta`: the executing CTA's own window only (a peer's
            // window, e.g. from `mapa`, is `.shared::cluster`).
            AddrSpace::Shared => matches!(g, addr::Generic::Shared(sh) if addr::decode_shared(sh).0 == ctx.cta.rank_in_cluster),
            AddrSpace::SharedCluster => matches!(g, addr::Generic::Shared(_)),
            AddrSpace::Local => matches!(g, addr::Generic::Local(_)),
            AddrSpace::Param => in_param,
            AddrSpace::Generic => true,
            AddrSpace::Const | AddrSpace::Tmem => false,
        };
        write_lane(ctx, dst, l, yes as u64);
    }
    Ok(Flow::Next)
}

#[inline]
pub fn mapa(ctx: &mut ExecCtx<'_>, dst: Reg, src: Operand, rank: Operand, space: AddrSpace) -> HResult {
    active_or_next!(ctx);
    let n = ctx.launch.ctas_per_cluster();
    for l in ctx.warp.active.lanes() {
        let a = lane_val(ctx, src, l);
        let r = lane_val(ctx, rank, l) as u32;
        if r >= n {
            return Err(support::err(ctx, ExecErrorKind::BadAddress, WarpMask::lane(l), format!("mapa rank {r} >= cluster size {n}")));
        }
        let v = match space {
            AddrSpace::Shared | AddrSpace::SharedCluster => {
                let off = addr::decode_shared(a as u32).1;
                addr::shared_addr(r, off).unwrap_or(0) as u64
            }
            AddrSpace::Generic => match addr::classify_generic(a) {
                addr::Generic::Shared(sa) => {
                    let off = addr::decode_shared(sa).1;
                    addr::generic_from_shared(addr::shared_addr(r, off).unwrap_or(0))
                }
                _ => {
                    return Err(support::err(ctx, ExecErrorKind::BadAddress, WarpMask::lane(l), "mapa of a non-shared generic address"))
                }
            },
            _ => return Err(support::unsupported(ctx, "mapa in this state space")),
        };
        write_lane(ctx, dst, l, v);
    }
    Ok(Flow::Next)
}

#[inline]
pub fn getctarank(ctx: &mut ExecCtx<'_>, dst: Reg, src: Operand, space: AddrSpace) -> HResult {
    active_or_next!(ctx);
    for l in ctx.warp.active.lanes() {
        let a = lane_val(ctx, src, l);
        let r = match space {
            AddrSpace::SharedCluster | AddrSpace::Shared => addr::decode_shared(a as u32).0,
            _ => match addr::classify_generic(a) {
                addr::Generic::Shared(sa) => addr::decode_shared(sa).0,
                _ => {
                    return Err(support::err(ctx, ExecErrorKind::BadAddress, WarpMask::lane(l), "getctarank of a non-shared address"))
                }
            },
        };
        write_lane(ctx, dst, l, r as u64);
    }
    let _ = Space::Shared;
    Ok(Flow::Next)
}
