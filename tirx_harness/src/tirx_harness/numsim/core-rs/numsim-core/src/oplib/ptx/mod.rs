//! `resolve_ptx`: interned `OpKey` -> `PtxFn`.
//!
//! # Key format
//! `OpKey.name` is the TIRx op (`tirx.ptx.cvt`, `tirx.cuda.make_float2`).
//! `OpKey.mods` entries are `"slot=token"` pairs as W1's decoder produces
//! them (`("rnd","rn")` -> `"rnd=rn"`); a bare `"token"` (empty slot) is also
//! accepted and matched by token. Empty tokens are omitted.
//!
//! # Operands
//! `PtxIo.dsts` = write operands in OpKey operand order (a value-returning
//! call's result first); `PtxIo.srcs` = read (and read-write) operands in
//! order, each register spanning `ty.slots()` slots. Multi-lane PTX
//! operands (`{a, b}` of `mov.b64`, `f32x4` of `cvt.rs`) arrive as one
//! register per lane. Register values keep their carrier `Ty`; each op
//! interprets the low bits per the PTX type in `mods` and writes the
//! destination zero-extended (signed PTX results narrower than the carrier
//! are sign-extended to the carrier width, then truncated to it).
//!
//! # Dispatch
//! `PtxFn` is a plain `fn` pointer, so parameterized forms are resolved once
//! into a boxed closure, interned process-wide by (key, tys), and returned as
//! one of `SLOTS` pre-instantiated trampolines (`tramp::<I>`). Hot
//! parameterless forms return direct fn items. Exhausting the slots fails
//! closed (`Unsupported`).

mod cvt;
mod helpers;
mod io;
mod warp;

pub(super) mod alu;

pub(super) use io::{Mods, Operands};

use super::{OpError, OpResult, PtxFn, PtxIo};
use crate::dtype::Ty;
use crate::program::OpKey;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// A resolved, parameterized op.
pub(super) type Op = Box<dyn Fn(&mut PtxIo<'_>) -> OpResult + Send + Sync>;

/// Distinct parameterized (key, tys) forms per process.
pub(super) const SLOTS: usize = 2048;

static TABLE: [OnceLock<Op>; SLOTS] = [const { OnceLock::new() }; SLOTS];
static TRAMPOLINES: [PtxFn; SLOTS] = include!("tramp_table.rs");

fn tramp<const I: usize>(io: &mut PtxIo<'_>) -> OpResult {
    match TABLE[I].get() {
        Some(op) => op(io),
        None => Err(OpError::invalid("unresolved PTX trampoline slot")),
    }
}

/// Resolution outcome of a family resolver.
pub(super) enum Resolved {
    /// A direct fn item (no captured parameters).
    Direct(PtxFn),
    /// A parameterized closure; interned behind a trampoline.
    Boxed(Op),
}

fn intern(identity: String, op: Op) -> OpResult<PtxFn> {
    static INDEX: OnceLock<Mutex<HashMap<String, usize>>> = OnceLock::new();
    let mut index = INDEX
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map_err(|_| OpError::invalid("PTX intern table poisoned"))?;
    if let Some(&slot) = index.get(&identity) {
        return Ok(TRAMPOLINES[slot]);
    }
    let slot = index.len();
    if slot >= SLOTS {
        return Err(OpError::unsupported(format!(
            "more than {SLOTS} distinct parameterized PTX forms in one process"
        )));
    }
    TABLE[slot]
        .set(op)
        .map_err(|_| OpError::invalid("PTX trampoline slot reused"))?;
    index.insert(identity, slot);
    Ok(TRAMPOLINES[slot])
}

/// Resolve an op for the given operand types.
pub(super) fn resolve(key: &OpKey, dst_tys: &[Ty], src_tys: &[Ty]) -> OpResult<PtxFn> {
    let mods = Mods::parse(&key.mods);
    let ops = Operands::new(dst_tys, src_tys);
    let name = key.name.as_str();
    let resolved = if let Some(found) = helpers::resolve(name, &mods, &ops)? {
        found
    } else if let Some(found) = cvt::resolve(name, &mods, &ops)? {
        found
    } else if let Some(found) = warp::resolve(name, &mods, &ops)? {
        found
    } else if let Some(found) = alu::resolve(name, &mods, &ops)? {
        found
    } else {
        return Err(OpError::unsupported(format!(
            "no oplib implementation for {} {:?}",
            key.name, key.mods
        )));
    };
    match resolved {
        Resolved::Direct(f) => Ok(f),
        Resolved::Boxed(op) => intern(format!("{}|{:?}|{:?}|{:?}", key.name, key.mods, dst_tys, src_tys), op),
    }
}

/// Every op name the resolver recognises (some forms of a name may still be
/// rejected by modifier/type checks). Used for coverage reports.
pub(in crate::oplib) fn known_ops() -> Vec<&'static str> {
    let mut names = Vec::new();
    names.extend_from_slice(helpers::NAMES);
    names.extend_from_slice(cvt::NAMES);
    names.extend_from_slice(warp::NAMES);
    names.extend_from_slice(alu::NAMES);
    names.sort_unstable();
    names.dedup();
    names
}
