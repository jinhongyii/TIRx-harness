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
//! Parameterless hot forms resolve to `PtxFn::new(fn item)`. Parameterized
//! forms resolve once into a boxed closure capturing the parsed modifiers and
//! operand layout; it is interned process-wide by (key, tys) and leaked so the
//! returned `PtxFn` stays `Copy` (bounded by the number of distinct forms).

mod cvt;
mod helpers;
mod hints;
mod io;
mod vector;
mod warp;

pub(super) mod alu;

pub(super) use io::{Mods, Operands};

use super::{OpError, OpResult, PtxFn, PtxIo};
use crate::dtype::Ty;
use crate::program::OpKey;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// A resolved, parameterized op.
pub(super) type Op = Box<super::PtxOp>;

/// A plain op fn item (the `Direct` form).
pub(in crate::oplib) type DirectFn = fn(&mut PtxIo<'_>) -> OpResult;

/// Resolution outcome of a family resolver.
pub(super) enum Resolved {
    /// A direct fn item (no captured parameters).
    Direct(DirectFn),
    /// A parameterized closure; interned and leaked once per distinct form.
    Boxed(Op),
}

fn intern(identity: String, op: Op) -> OpResult<PtxFn> {
    static INDEX: OnceLock<Mutex<HashMap<String, &'static super::PtxOp>>> = OnceLock::new();
    let mut index = INDEX
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map_err(|_| OpError::invalid("PTX intern table poisoned"))?;
    let op: &'static super::PtxOp = *index.entry(identity).or_insert_with(|| Box::leak(op));
    Ok(PtxFn::from_static(op))
}

/// Resolve an op for the given operand types.
pub(super) fn resolve(key: &OpKey, dst_tys: &[Ty], src_tys: &[Ty]) -> OpResult<PtxFn> {
    let mods = Mods::parse(&key.mods);
    let ops = Operands::new(dst_tys, src_tys);
    let name = key.name.as_str();
    let resolved = if let Some(found) = vector::resolve(name, &mods, &ops)? {
        found
    } else if let Some(found) = hints::resolve(name, &mods, &ops)? {
        found
    } else if let Some(found) = helpers::resolve(name, &mods, &ops)? {
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
        Resolved::Direct(f) => Ok(PtxFn::new(f)),
        Resolved::Boxed(op) => intern(format!("{}|{:?}|{:?}|{:?}", key.name, key.mods, dst_tys, src_tys), op),
    }
}

/// Every op name the resolver recognises (some forms of a name may still be
/// rejected by modifier/type checks). Used for coverage reports.
pub(in crate::oplib) fn known_ops() -> Vec<&'static str> {
    let mut names = Vec::new();
    names.extend_from_slice(vector::NAMES);
    names.extend_from_slice(hints::NAMES);
    names.extend_from_slice(helpers::NAMES);
    names.extend_from_slice(cvt::NAMES);
    names.extend_from_slice(warp::NAMES);
    names.extend_from_slice(alu::NAMES);
    names.sort_unstable();
    names.dedup();
    names
}
