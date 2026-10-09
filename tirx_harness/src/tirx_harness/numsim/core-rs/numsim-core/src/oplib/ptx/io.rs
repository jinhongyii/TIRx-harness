//! Modifier parsing and operand-slot access shared by all PTX families.

use super::super::{OpError, OpResult, PtxIo};
use crate::dtype::Ty;

/// Parsed `OpKey.mods`: `(slot, token)` pairs (`slot` empty for bare tokens).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(in crate::oplib) struct Mods {
    pub pairs: Vec<(String, String)>,
}

impl Mods {
    /// Split `OpKey.mods` into `(slot, token)` pairs (`slot=token`, or a bare token with an
    /// empty slot); empty strings are dropped.
    pub fn parse(mods: &[String]) -> Mods {
        Mods {
            pairs: mods
                .iter()
                .filter(|m| !m.is_empty())
                .map(|m| match m.split_once('=') {
                    Some((slot, token)) => (slot.to_string(), token.to_string()),
                    None => (String::new(), m.clone()),
                })
                .collect(),
        }
    }
    /// Token of a named slot (`"rnd"`, `"type"`, `"dtype"`, ...).
    pub fn get(&self, slot: &str) -> Option<&str> {
        self.pairs.iter().find(|(s, _)| s == slot).map(|(_, t)| t.as_str())
    }
    /// Whether any slot carries `token` (for flag slots: `ftz`, `sat`, ...).
    pub fn has(&self, token: &str) -> bool {
        self.pairs.iter().any(|(_, t)| t == token)
    }
    /// Required slot or an `Unsupported` error naming it.
    pub fn req(&self, slot: &str, op: &str) -> OpResult<&str> {
        self.get(slot)
            .ok_or_else(|| OpError::unsupported(format!("{op}: missing modifier slot `{slot}` in {:?}", self.pairs)))
    }
    /// Assign every mod to a slot of `slots` = `(name, choices, optional)` in
    /// table order: `slot=token` pairs by name (token must be a choice), bare
    /// tokens (what W1's `mod_tokens` emits today) to the first unassigned
    /// slot at or after the previous one whose choices contain them. Unknown
    /// tokens/slots, duplicates and missing required slots fail closed.
    pub fn normalize(&self, slots: &[(&str, &[&str], bool)], op: &str) -> OpResult<Mods> {
        let mut assigned: Vec<Option<String>> = vec![None; slots.len()];
        let mut cursor = 0usize;
        for (slot, token) in &self.pairs {
            let index = if slot.is_empty() {
                (cursor..slots.len())
                    .find(|&i| assigned[i].is_none() && slots[i].1.contains(&token.as_str()))
                    .ok_or_else(|| OpError::unsupported(format!("{op}: unmodeled modifier `{token}`")))?
            } else {
                let i = slots
                    .iter()
                    .position(|(s, _, _)| s == slot)
                    .ok_or_else(|| OpError::unsupported(format!("{op}: unmodeled modifier {slot}={token}")))?;
                if !slots[i].1.contains(&token.as_str()) {
                    return Err(OpError::unsupported(format!("{op}: {slot}={token} outside {:?}", slots[i].1)));
                }
                i
            };
            if assigned[index].is_some() {
                return Err(OpError::unsupported(format!("{op}: repeated modifier slot {}", slots[index].0)));
            }
            assigned[index] = Some(token.clone());
            cursor = index + 1;
        }
        let mut pairs = Vec::new();
        for ((name, _, optional), token) in slots.iter().zip(assigned) {
            match token {
                Some(token) => pairs.push((name.to_string(), token)),
                None if !optional => {
                    return Err(OpError::unsupported(format!("{op}: missing modifier slot `{name}`")))
                }
                None => {}
            }
        }
        Ok(Mods { pairs })
    }
}

/// Slot offsets and carrier types of one op's operands (captured by closures).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(in crate::oplib) struct Operands {
    pub dst_tys: Vec<Ty>,
    pub src_tys: Vec<Ty>,
    pub dst_off: Vec<usize>,
    pub src_off: Vec<usize>,
}

impl Operands {
    /// Operand layout from destination/source types: each operand's register-slot offset.
    pub fn new(dst_tys: &[Ty], src_tys: &[Ty]) -> Operands {
        let offsets = |tys: &[Ty]| {
            let mut at = 0usize;
            tys.iter()
                .map(|ty| {
                    let here = at;
                    at += ty.slots() as usize;
                    here
                })
                .collect::<Vec<_>>()
        };
        Operands {
            dst_off: offsets(dst_tys),
            src_off: offsets(src_tys),
            dst_tys: dst_tys.to_vec(),
            src_tys: src_tys.to_vec(),
        }
    }
    /// Require exactly `dsts` destination and `srcs` source registers.
    pub fn arity(&self, dsts: usize, srcs: usize, op: &str) -> OpResult<()> {
        if self.dst_tys.len() != dsts || self.src_tys.len() != srcs {
            return Err(OpError::unsupported(format!(
                "{op}: expected {dsts} dst / {srcs} src registers, got {:?} / {:?}",
                self.dst_tys, self.src_tys
            )));
        }
        Ok(())
    }
    /// Low 64 bits of source `i` in `lane`.
    #[inline]
    pub fn src(&self, io: &PtxIo<'_>, i: usize, lane: usize) -> u64 {
        io.srcs[self.src_off[i]][lane]
    }
    /// Full (up to 128-bit) value of source `i`.
    #[inline]
    pub fn src128(&self, io: &PtxIo<'_>, i: usize, lane: usize) -> u128 {
        let off = self.src_off[i];
        let lo = io.srcs[off][lane] as u128;
        if self.src_tys[i].slots() > 1 {
            lo | ((io.srcs[off + 1][lane] as u128) << 64)
        } else {
            lo
        }
    }
    /// Write `bits` (a PTX result of `ptx_bits` width, `signed` = sign-extend
    /// into a wider carrier) to destination `i`.
    #[inline]
    pub fn put(&self, io: &mut PtxIo<'_>, i: usize, lane: usize, bits: u64, ptx_bits: u32, signed: bool) {
        let ty = self.dst_tys[i];
        let carrier = ty.bits().min(64);
        let mut value = if ptx_bits >= 64 { bits } else { bits & ((1u64 << ptx_bits) - 1) };
        if signed && ptx_bits < 64 && ptx_bits > 0 && (value >> (ptx_bits - 1)) & 1 == 1 {
            value |= !0u64 << ptx_bits;
        }
        if carrier < 64 {
            value &= (1u64 << carrier) - 1;
        }
        io.dsts[self.dst_off[i]][lane] = value;
        if ty.slots() > 1 {
            // A 128-bit carrier (int128/uint128/b128): the upper slot is the
            // extension of the result (legacy, W11-4).
            let negative = signed && ptx_bits > 0 && ptx_bits <= 64 && (value >> 63) & 1 == 1;
            io.dsts[self.dst_off[i] + 1][lane] = if negative { u64::MAX } else { 0 };
        }
    }
    /// Write a 128-bit value to destination `i` (two slots when the carrier has them).
    #[inline]
    pub fn put128(&self, io: &mut PtxIo<'_>, i: usize, lane: usize, bits: u128) {
        let off = self.dst_off[i];
        io.dsts[off][lane] = bits as u64;
        if self.dst_tys[i].slots() > 1 {
            io.dsts[off + 1][lane] = (bits >> 64) as u64;
        }
    }
}
