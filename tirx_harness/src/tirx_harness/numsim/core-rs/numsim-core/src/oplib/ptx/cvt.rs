//! `tirx.ptx.cvt*`: every TVM PTX-table `cvt` form.
//!
//! Resolution builds the exact PTX spelling from the `OpKey` modifiers
//! (`cvt.<rnd>.<relu>.<satfinite>.<ftz>.<sat>.<pzo>.<scaled>.<dtype>.<atype>`,
//! or `cvt.pack.sat.<convert>.s32{.b32}`), parses it once with
//! [`CvtSpelling::parse`], probes it once (illegal spellings fail closed as
//! `Unsupported`), and captures it. Each lane then fills a [`CvtOperands`]
//! from the sources and runs [`CvtSpelling::execute`].
//!
//! Source order is the TVM table operand order, which is the PTX operand
//! order: `a` (the UPPER element of two-primary packs, ISA 9.7.10.24:65-68),
//! `b`, then `rbits`, then the optional `scale_factor` (present exactly when
//! the `scaled` slot is written). `abef:f32x4` arrives as four registers
//! `a, b, e, f` (`a` in the most significant field) -> `CvtOperands.{a,b,c,d}`.

use super::{Mods, Operands, Resolved};
use crate::oplib::{OpError, OpResult, PtxIo};
use numsim_oplib::cvt::{CvtOperands, CvtSpelling, CvtType};

mod hot;
mod table;
#[cfg(test)]
mod tests;

pub(in crate::oplib) use table::NAMES;

/// Source-register layout of one table op (scale factor excluded).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::oplib) enum Layout {
    /// `d, a`
    Unary,
    /// `d, a, b`
    Pair,
    /// `d, a, b, rbits`
    PairRbits,
    /// `d, {a, b, e, f}, rbits`
    Quad,
    /// `cvt.pack.sat.{u16,s16}.s32 d, a, b`
    Pack,
    /// `cvt.pack.sat.{u8,s8,u4,s4,u2,s2}.s32.b32 d, a, b, c`
    PackC,
}

impl Layout {
    pub(in crate::oplib) fn sources(self) -> usize {
        match self {
            Layout::Unary => 1,
            Layout::Pair | Layout::Pack => 2,
            Layout::PairRbits | Layout::PackC => 3,
            Layout::Quad => 5,
        }
    }
}

/// One modifier slot of a table op.
pub(in crate::oplib) struct Slot {
    pub name: &'static str,
    pub choices: &'static [&'static str],
    pub optional: bool,
}

/// One `tirx.ptx.cvt*` table op.
pub(in crate::oplib) struct OpSpec {
    pub name: &'static str,
    pub layout: Layout,
    pub slots: &'static [Slot],
}

pub(in crate::oplib) fn spec(name: &str) -> Option<&'static OpSpec> {
    table::SPECS.iter().find(|spec| spec.name == name)
}

/// Assign every modifier to a slot (by slot name, or by token for bare
/// tokens); unknown slots/tokens, repeats and missing required slots fail
/// closed.
fn assign(spec: &OpSpec, mods: &Mods) -> OpResult<Vec<Option<String>>> {
    let mut tokens: Vec<Option<String>> = vec![None; spec.slots.len()];
    for (slot, token) in &mods.pairs {
        let index = if slot.is_empty() {
            spec.slots
                .iter()
                .enumerate()
                .position(|(i, s)| tokens[i].is_none() && s.choices.contains(&token.as_str()))
        } else {
            spec.slots.iter().position(|s| s.name == slot)
        };
        let Some(index) = index else {
            return Err(OpError::unsupported(format!(
                "{}: unmodeled modifier {slot}={token}",
                spec.name
            )));
        };
        if !spec.slots[index].choices.contains(&token.as_str()) {
            return Err(OpError::unsupported(format!(
                "{}: modifier {}={token} not in {:?}",
                spec.name, spec.slots[index].name, spec.slots[index].choices
            )));
        }
        if tokens[index].replace(token.clone()).is_some() {
            return Err(OpError::unsupported(format!(
                "{}: repeated modifier slot {}",
                spec.name, spec.slots[index].name
            )));
        }
    }
    for (slot, token) in spec.slots.iter().zip(&tokens) {
        if !slot.optional && token.is_none() {
            return Err(OpError::unsupported(format!(
                "{}: missing modifier slot `{}`",
                spec.name, slot.name
            )));
        }
    }
    Ok(tokens)
}

fn token<'a>(spec: &OpSpec, tokens: &'a [Option<String>], slot: &str) -> Option<&'a str> {
    spec.slots
        .iter()
        .position(|s| s.name == slot)
        .and_then(|i| tokens[i].as_deref())
}

/// The exact PTX spelling of a table op with these slot tokens.
pub(in crate::oplib) fn spelling(spec: &OpSpec, tokens: &[Option<String>]) -> OpResult<String> {
    if matches!(spec.layout, Layout::Pack | Layout::PackC) {
        let convert = token(spec, tokens, "convert")
            .ok_or_else(|| OpError::unsupported(format!("{}: missing `convert`", spec.name)))?;
        let tail = if spec.layout == Layout::PackC {
            ".b32"
        } else {
            ""
        };
        return Ok(format!("cvt.pack.sat.{convert}.s32{tail}"));
    }
    let mut text = String::from("cvt");
    for (slot, token) in spec.slots.iter().zip(tokens) {
        if slot.name == "dtype" || slot.name == "atype" {
            continue;
        }
        if let Some(token) = token {
            text.push('.');
            text.push_str(token);
        }
    }
    for slot in ["dtype", "atype"] {
        let value = token(spec, tokens, slot)
            .ok_or_else(|| OpError::unsupported(format!("{}: missing `{slot}`", spec.name)))?;
        text.push('.');
        text.push_str(value);
    }
    Ok(text)
}

/// Destination payload width and signedness of a parsed spelling.
fn destination(parsed: &CvtSpelling) -> (u32, bool) {
    if parsed.pack.is_some() {
        return (32, false);
    }
    match parsed.dst {
        CvtType::Int(kind) => (kind.bits(), kind.signed()),
        other => (other.bits(), false),
    }
}

/// Fill one lane's operands from the sources.
#[inline]
fn gather(
    layout: Layout,
    scaled: bool,
    ops: &Operands,
    io: &PtxIo<'_>,
    lane: usize,
) -> CvtOperands {
    let mut operands = CvtOperands::unary(ops.src(io, 0, lane));
    match layout {
        Layout::Unary => {}
        Layout::Pair | Layout::Pack => operands.b = ops.src(io, 1, lane),
        Layout::PairRbits => {
            operands.b = ops.src(io, 1, lane);
            operands.rbits = ops.src(io, 2, lane) as u32;
        }
        Layout::PackC => {
            operands.b = ops.src(io, 1, lane);
            operands.c = ops.src(io, 2, lane);
        }
        Layout::Quad => {
            operands.b = ops.src(io, 1, lane);
            operands.c = ops.src(io, 2, lane);
            operands.d = ops.src(io, 3, lane);
            operands.rbits = ops.src(io, 4, lane) as u32;
        }
    }
    if scaled {
        operands.scale = ops.src(io, layout.sources(), lane) as u16;
    }
    operands
}

pub(in crate::oplib) fn resolve(
    name: &str,
    mods: &Mods,
    ops: &Operands,
) -> OpResult<Option<Resolved>> {
    let Some(spec) = spec(name) else {
        return Ok(None);
    };
    let tokens = assign(spec, mods)?;
    let text = spelling(spec, &tokens)?;
    let parsed =
        CvtSpelling::parse(&text).map_err(|e| OpError::unsupported(format!("{name}: {}", e.0)))?;
    // Every rejection in `execute` is form-level; probing once moves it to
    // resolve time so an illegal spelling never reaches a lane.
    parsed
        .execute(&CvtOperands::default())
        .map_err(|e| OpError::unsupported(format!("{name} ({text}): {}", e.0)))?;
    let scaled = token(spec, &tokens, "scaled").is_some();
    let layout = spec.layout;
    ops.arity(1, layout.sources() + usize::from(scaled), name)?;
    let (dst_bits, signed) = destination(&parsed);
    // A carrier wider than the PTX result is extended per the result's
    // signedness (128-bit carriers: both slots, `Operands::put`).
    if ops.dst_tys[0].bits() < dst_bits || ops.dst_tys[0].slots() > 2 {
        return Err(OpError::unsupported(format!(
            "{name} ({text}): destination carrier {:?} cannot hold a {dst_bits}-bit result",
            ops.dst_tys[0]
        )));
    }
    // A 128-bit source carrier holds the operand in its low slot (cvt reads
    // at most 64 source bits, truncating the carrier like legacy).
    if ops.src_tys.iter().any(|ty| ty.slots() > 2) {
        return Err(OpError::unsupported(format!(
            "{name} ({text}): source carriers {:?} wider than 128 bits",
            ops.src_tys
        )));
    }
    if let Some(direct) = hot::select(&parsed, ops) {
        return Ok(Some(Resolved::Direct(direct)));
    }
    let ops = ops.clone();
    Ok(Some(Resolved::Boxed(Box::new(
        move |io: &mut PtxIo<'_>| {
            for lane in 0..numsim_types::WARP_SIZE {
                if !io.mask.contains(lane) {
                    continue;
                }
                let operands = gather(layout, scaled, &ops, io, lane);
                let result = parsed.execute(&operands)?;
                ops.put(io, 0, lane, result, dst_bits, signed);
            }
            Ok(())
        },
    ))))
}
