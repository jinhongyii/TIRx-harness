//! `ldmatrix` / `stmatrix` fragment maps (every legacy form).

use std::cell::RefCell;

use super::super::{OpError, OpResult};
use crate::program::{MatrixFmt, MatrixShape};
use crate::value::WarpValue;
use numsim_oplib::layout::matrix as lib;
use numsim_oplib::types::{OpError as LibError, OpResult as LibResult};

/// One contiguous access through a providing lane's row address:
/// `byte_len` bytes at `byte_delta` past the row pointer of lane
/// `provider_lane`; `shift` is the element's bit offset in those bytes
/// (b6/b4 formats).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MatrixAccess {
    pub provider_lane: usize,
    pub byte_delta: usize,
    pub byte_len: usize,
    pub shift: usize,
}

impl From<lib::MatrixAccess> for MatrixAccess {
    fn from(access: lib::MatrixAccess) -> MatrixAccess {
        MatrixAccess {
            provider_lane: access.provider_lane,
            byte_delta: access.byte_delta,
            byte_len: access.byte_len,
            shift: access.shift,
        }
    }
}

/// Keeps the caller's closure error (kind + message) across `numsim_oplib`.
struct Bridge(RefCell<Option<OpError>>);

impl Bridge {
    fn new() -> Bridge {
        Bridge(RefCell::new(None))
    }
    fn fail(&self, error: OpError) -> LibError {
        let message = error.message.clone();
        self.0.borrow_mut().get_or_insert(error);
        LibError::message(format!("matrix operand access failed: {message}"))
    }
    fn lift<T>(&self, result: LibResult<T>) -> OpResult<T> {
        result.map_err(|error| self.0.borrow_mut().take().unwrap_or_else(|| error.into()))
    }
}

/// One `ldmatrix` form (legacy `Ldmatrix<COUNT, TRANSPOSE, SOURCE_BITS,
/// SIGNED>` as the frontend mapped PTX onto it).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LdMatrixPlan {
    /// Destination registers per lane (`num`, or `2 * num` for m16n16).
    pub registers: usize,
    pub transpose: bool,
    /// 16 (`m8n8.b16`), 8 (`m16n16.b8`), 6 / 4 (`b8x16.b6x16_p32` /
    /// `b4x16_p64`, and `m8n16.s8.s4`).
    pub source_bits: usize,
    /// `m8n16.s8.s4`: each destination byte is the sign-extended nibble.
    pub signed: bool,
    /// Lanes `0..providers` supply row addresses.
    pub providers: usize,
    /// Bytes each provider row contributes (16 for b16/b8, 12 for b6, 8 for
    /// b4; row padding is never read).
    pub row_bytes: usize,
}

/// Validate an `Instr::LdMatrix` form and return its plan.
pub fn ldmatrix_plan(
    shape: MatrixShape,
    num: u8,
    trans: bool,
    fmt: MatrixFmt,
) -> OpResult<LdMatrixPlan> {
    if !matches!(num, 1 | 2 | 4) {
        return Err(OpError::invalid(format!(
            "ldmatrix .x{num} is not x1/x2/x4"
        )));
    }
    let num = usize::from(num);
    let (registers, source_bits, signed) = match (shape, fmt, trans) {
        (MatrixShape::M8N8, MatrixFmt::B16, _) => (num, 16, false),
        (MatrixShape::M16N16, MatrixFmt::B8, true) => (2 * num, 8, false),
        (MatrixShape::M8N16, MatrixFmt::S8S4, false) => (num, 4, true),
        (MatrixShape::M8N16, MatrixFmt::B6x16P32, false) => (num, 6, false),
        (MatrixShape::M8N16, MatrixFmt::B4x16P64, false) => (num, 4, false),
        (MatrixShape::M16N16, MatrixFmt::B6x16P32, true) => (2 * num, 6, false),
        (MatrixShape::M16N16, MatrixFmt::B4x16P64, true) => (2 * num, 4, false),
        (shape, fmt, trans) => {
            return Err(OpError::invalid(format!(
                "ldmatrix has no {shape:?}.{fmt:?} form with trans={trans}"
            )))
        }
    };
    if registers > 4 {
        return Err(OpError::invalid(format!(
            "ldmatrix {shape:?} .x{num} needs {registers} registers (max 4)"
        )));
    }
    Ok(LdMatrixPlan {
        registers,
        transpose: trans,
        source_bits,
        signed,
        providers: registers * 8,
        row_bytes: if source_bits == 16 {
            16
        } else {
            16 * source_bits / 8
        },
    })
}

impl LdMatrixPlan {
    /// Every access feeding consumer `lane` (footprint form; b8 formats check
    /// that each provider row is 16-byte aligned).
    pub fn accesses(
        &self,
        lane: usize,
        row_address: impl Fn(usize) -> OpResult<u64>,
    ) -> OpResult<Vec<MatrixAccess>> {
        let bridge = Bridge::new();
        let row_base = |provider: usize| -> LibResult<usize> {
            let address = row_address(provider).map_err(|error| bridge.fail(error))?;
            usize::try_from(address)
                .map_err(|_| bridge.fail(OpError::invalid("ldmatrix row address overflow")))
        };
        let accesses = bridge.lift(lib::ldmatrix_lane_accesses(
            self.registers,
            lane,
            self.transpose,
            self.source_bits,
            row_base,
        ))?;
        Ok(accesses.into_iter().map(MatrixAccess::from).collect())
    }
}

/// Destination registers of an `ldmatrix` (legacy
/// `raw_ldmatrix_b16_fragments` / `raw_ldmatrix_b8_fragments` plus the
/// s4 sign extension). `row_address(provider)` is the provider lane's row
/// address (b8 formats require 16-byte alignment); `read(provider,
/// byte_delta, len)` returns `len` bytes past that row pointer.
pub fn ldmatrix_fragments(
    plan: &LdMatrixPlan,
    row_address: impl Fn(usize) -> OpResult<u64>,
    mut read: impl FnMut(usize, usize, usize) -> OpResult<Vec<u8>>,
) -> OpResult<Vec<WarpValue<u32>>> {
    let bridge = Bridge::new();
    let read = |provider: usize, delta: usize, len: usize| -> LibResult<Vec<u8>> {
        read(provider, delta, len).map_err(|error| bridge.fail(error))
    };
    let mut fragments = if plan.source_bits == 16 {
        let zeros = [0_i64; 32];
        bridge.lift(lib::ldmatrix_b16_fragments(
            &zeros,
            1,
            plan.registers,
            plan.transpose,
            read,
        ))?
    } else {
        let row_base = |provider: usize| -> LibResult<usize> {
            let address = row_address(provider).map_err(|error| bridge.fail(error))?;
            usize::try_from(address)
                .map_err(|_| bridge.fail(OpError::invalid("ldmatrix row address overflow")))
        };
        bridge.lift(lib::ldmatrix_b8_fragments(
            plan.registers,
            plan.transpose,
            plan.source_bits,
            row_base,
            read,
        ))?
    };
    if plan.signed {
        for fragment in &mut fragments {
            for word in fragment.iter_mut() {
                *word = u32::from_le_bytes(
                    word.to_le_bytes()
                        .map(|byte| ((byte as i8) << 4 >> 4) as u8),
                );
            }
        }
    }
    Ok(fragments)
}

/// One `stmatrix` form (legacy `StmatrixDescriptor`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StMatrixPlan {
    /// Source registers per lane.
    pub registers: usize,
    shape: lib::StmatrixShape,
    /// Lanes `0..providers` supply 16-byte-aligned row addresses.
    pub providers: usize,
}

/// Validate an `Instr::StMatrix` form: `m8n8` (b16, optional `.trans`) or
/// `m16n8` (b8, `.trans` required).
pub fn stmatrix_plan(shape: MatrixShape, num: u8, trans: bool) -> OpResult<StMatrixPlan> {
    if !matches!(num, 1 | 2 | 4) {
        return Err(OpError::invalid(format!(
            "stmatrix .x{num} is not x1/x2/x4"
        )));
    }
    let shape = match (shape, trans) {
        (MatrixShape::M8N8, transpose) => lib::StmatrixShape::M8n8B16 { transpose },
        (MatrixShape::M16N8, true) => lib::StmatrixShape::M16n8B8Transposed,
        (shape, trans) => {
            return Err(OpError::invalid(format!(
                "stmatrix has no {shape:?} form with trans={trans}"
            )))
        }
    };
    let registers = usize::from(num);
    Ok(StMatrixPlan {
        registers,
        shape,
        providers: registers * 8,
    })
}

/// Writes of an `stmatrix` (legacy `raw_stmatrix` order: register-major,
/// then source lane): `(provider_lane, byte_delta, bytes)` past the
/// provider's row pointer. `sources[r]` is source register `r` of all lanes.
pub fn stmatrix_writes(
    plan: &StMatrixPlan,
    sources: &[WarpValue<u32>],
    row_address: impl Fn(usize) -> OpResult<u64>,
) -> OpResult<Vec<(usize, usize, lib::WriteBytes)>> {
    if sources.len() != plan.registers {
        return Err(OpError::invalid(format!(
            "stmatrix needs {} source registers, got {}",
            plan.registers,
            sources.len()
        )));
    }
    let bridge = Bridge::new();
    let row = |provider: usize| -> LibResult<usize> {
        let address = row_address(provider).map_err(|error| bridge.fail(error))?;
        usize::try_from(address)
            .map_err(|_| bridge.fail(OpError::invalid("stmatrix row address overflow")))
    };
    let refs: Vec<&WarpValue<u32>> = sources.iter().collect();
    bridge.lift(lib::stmatrix_writes(plan.shape, &refs, row))
}
