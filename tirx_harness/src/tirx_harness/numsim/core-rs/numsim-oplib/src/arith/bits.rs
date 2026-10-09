//! Bit manipulation: clmad, bfe/bfi/bfind/bmsk, brev/clz/cnot/popc, shf,
//! szext, and/or/xor/not, lop3, shl/shr, prmt, fns.
//!
//! Moved from legacy `engine-rs/src/runtime/instructions/reg.rs`
//! (`carryless_product_u64`, `bit_field_extract`, `bit_field_insert`,
//! `most_significant_non_sign_bit`, `bit_mask`, `funnel_shift`,
//! `zero_extend_32`, `sign_extend_32`, `bit_variants!`, `lop3_word`,
//! `lop3_bool_variant!`, `shift_variant!`, `fns_spec`), `prmt.rs`, and
//! `runtime/warp_ops.rs::warp_fns_b32` (per-lane base check).

use super::BoolOp;
use crate::types::{OpError, OpResult};

fn carryless_product_u64(lhs: u64, rhs: u64) -> u128 {
    let mut product = 0_u128;
    for bit in 0..64 {
        if lhs & (1_u64 << bit) != 0 {
            product ^= u128::from(rhs) << bit;
        }
    }
    product
}

/// `clmad.lo.u64`.
pub fn clmad_lo(lhs: u64, rhs: u64, addend: u64) -> u64 {
    (carryless_product_u64(lhs, rhs) as u64) ^ addend
}

/// `clmad.hi.u64`.
pub fn clmad_hi(lhs: u64, rhs: u64, addend: u64) -> u64 {
    ((carryless_product_u64(lhs, rhs) >> 64) as u64) ^ addend
}

pub(crate) fn low_mask_u64(width: u32) -> u64 {
    if width >= u64::BITS {
        u64::MAX
    } else {
        (1_u64 << width) - 1
    }
}

/// PTX defines `bfe`/`bfi` position and length only in `0..=255`.
pub fn check_bit_field_controls(instruction: &str, position: u32, length: u32) -> OpResult<()> {
    if position > 255 || length > 255 {
        return Err(OpError::message(format!(
            "{instruction} position/length is outside the defined 0..255 range: {position}/{length}"
        )));
    }
    Ok(())
}

/// Raw `bfe` on a `width`-bit carrier held in a `u64`. Controls are masked to
/// eight bits (callers validate active lanes with
/// [`check_bit_field_controls`]).
pub fn bit_field_extract(value: u64, position: u32, length: u32, width: u32, signed: bool) -> u64 {
    let position = position & 0xff;
    let length = length & 0xff;
    if length == 0 {
        return 0;
    }

    let copied = if position >= width {
        0
    } else {
        length.min(width - position)
    };
    let copied_mask = low_mask_u64(copied);
    let mut result = if copied == 0 {
        0
    } else {
        (value >> position) & copied_mask
    };

    if signed {
        let sign_position = (position + length - 1).min(width - 1);
        if value & (1_u64 << sign_position) != 0 {
            result |= !copied_mask;
        }
    }
    result & low_mask_u64(width)
}

/// `bfe.u32`.
pub fn bfe_u32(value: u32, position: u32, length: u32) -> OpResult<u32> {
    check_bit_field_controls("bfe", position, length)?;
    Ok(bit_field_extract(u64::from(value), position, length, 32, false) as u32)
}

/// `bfe.u64`.
pub fn bfe_u64(value: u64, position: u32, length: u32) -> OpResult<u64> {
    check_bit_field_controls("bfe", position, length)?;
    Ok(bit_field_extract(value, position, length, 64, false))
}

/// `bfe.s32`.
pub fn bfe_s32(value: i32, position: u32, length: u32) -> OpResult<i32> {
    check_bit_field_controls("bfe", position, length)?;
    Ok(bit_field_extract(value as u64, position, length, 32, true) as i32)
}

/// `bfe.s64`.
pub fn bfe_s64(value: i64, position: u32, length: u32) -> OpResult<i64> {
    check_bit_field_controls("bfe", position, length)?;
    Ok(bit_field_extract(value as u64, position, length, 64, true) as i64)
}

/// Raw `bfi` on a `width`-bit carrier held in a `u64`.
pub fn bit_field_insert(source: u64, base: u64, position: u32, length: u32, width: u32) -> u64 {
    let position = position & 0xff;
    let length = length & 0xff;
    let copied = if position >= width {
        0
    } else {
        length.min(width - position)
    };
    if copied == 0 {
        return base & low_mask_u64(width);
    }
    let source_mask = low_mask_u64(copied);
    let insertion_mask = source_mask << position;
    ((base & !insertion_mask) | ((source & source_mask) << position)) & low_mask_u64(width)
}

/// `bfi.b32`.
pub fn bfi_b32(source: u32, base: u32, position: u32, length: u32) -> OpResult<u32> {
    check_bit_field_controls("bfi", position, length)?;
    Ok(bit_field_insert(u64::from(source), u64::from(base), position, length, 32) as u32)
}

/// `bfi.b64`.
pub fn bfi_b64(source: u64, base: u64, position: u32, length: u32) -> OpResult<u64> {
    check_bit_field_controls("bfi", position, length)?;
    Ok(bit_field_insert(source, base, position, length, 64))
}

/// Position of the most significant non-sign bit, or `u32::MAX` if none.
pub fn most_significant_non_sign_bit(value: u64, width: u32, signed: bool) -> u32 {
    let width_mask = low_mask_u64(width);
    let mut bits = value & width_mask;
    if signed && bits & (1_u64 << (width - 1)) != 0 {
        bits = !bits & width_mask;
    }
    if bits == 0 {
        u32::MAX
    } else {
        u64::BITS - 1 - bits.leading_zeros()
    }
}

fn bfind(value: u64, width: u32, signed: bool, shift_amount: bool) -> u32 {
    let position = most_significant_non_sign_bit(value, width, signed);
    if shift_amount && position != u32::MAX {
        width - 1 - position
    } else {
        position
    }
}

/// `bfind{.shiftamt}.u32`.
pub fn bfind_u32(value: u32, shift_amount: bool) -> u32 {
    bfind(u64::from(value), 32, false, shift_amount)
}

/// `bfind{.shiftamt}.u64`.
pub fn bfind_u64(value: u64, shift_amount: bool) -> u32 {
    bfind(value, 64, false, shift_amount)
}

/// `bfind{.shiftamt}.s32`.
pub fn bfind_s32(value: i32, shift_amount: bool) -> u32 {
    bfind(value as u64, 32, true, shift_amount)
}

/// `bfind{.shiftamt}.s64`.
pub fn bfind_s64(value: i64, shift_amount: bool) -> u32 {
    bfind(value as u64, 64, true, shift_amount)
}

/// `bmsk.{clamp,wrap}.b32`.
pub fn bmsk_b32(position: u32, width: u32, clamp: bool) -> u32 {
    let position = if clamp {
        if position >= 32 {
            return 0;
        }
        position
    } else {
        position & 0x1f
    };
    let width = if clamp { width.min(32) } else { width & 0x1f };
    if width == 0 {
        return 0;
    }
    let width = width.min(32 - position);
    (low_mask_u64(width) as u32) << position
}

/// `brev.b32`.
pub fn brev_b32(value: u32) -> u32 {
    value.reverse_bits()
}

/// `brev.b64`.
pub fn brev_b64(value: u64) -> u64 {
    value.reverse_bits()
}

/// `clz.b32`.
pub fn clz_b32(value: u32) -> u32 {
    value.leading_zeros()
}

/// `clz.b64`.
pub fn clz_b64(value: u64) -> u32 {
    value.leading_zeros()
}

/// `popc.b32`.
pub fn popc_b32(value: u32) -> u32 {
    value.count_ones()
}

/// `popc.b64`.
pub fn popc_b64(value: u64) -> u32 {
    value.count_ones()
}

/// `cnot.b16`.
pub fn cnot_b16(value: u16) -> u16 {
    u16::from(value == 0)
}

/// `cnot.b32`.
pub fn cnot_b32(value: u32) -> u32 {
    u32::from(value == 0)
}

/// `cnot.b64`.
pub fn cnot_b64(value: u64) -> u64 {
    u64::from(value == 0)
}

/// `shf.{l,r}.{clamp,wrap}.b32` funnel shift of `high:low`.
pub fn shf_b32(low: u32, high: u32, shift: u32, left: bool, clamp: bool) -> u32 {
    let shift = if clamp { shift.min(32) } else { shift & 0x1f };
    let joined = (u64::from(high) << 32) | u64::from(low);
    if left {
        ((joined << shift) >> 32) as u32
    } else {
        (joined >> shift) as u32
    }
}

/// `szext.{clamp,wrap}.u32` (zero extension from bit `width`).
pub fn szext_u32(value: u32, width: u32, clamp: bool) -> u32 {
    if clamp && width >= 32 {
        return value;
    }
    value & (low_mask_u64(width & 0x1f) as u32)
}

/// `szext.{clamp,wrap}.s32` (sign extension from bit `width - 1`).
pub fn szext_s32(value: i32, width: u32, clamp: bool) -> i32 {
    if clamp && width >= 32 {
        return value;
    }
    let width = width & 0x1f;
    if width == 0 {
        return 0;
    }
    let mask = low_mask_u64(width) as u32;
    let bits = value as u32 & mask;
    if bits & (1_u32 << (width - 1)) != 0 {
        (bits | !mask) as i32
    } else {
        bits as i32
    }
}

/// `and.{pred,b16,b32,b64,...}`.
pub fn and<T: std::ops::BitAnd<Output = T>>(lhs: T, rhs: T) -> T {
    lhs & rhs
}

/// `or.{pred,b16,b32,b64,...}`.
pub fn or<T: std::ops::BitOr<Output = T>>(lhs: T, rhs: T) -> T {
    lhs | rhs
}

/// `xor.{pred,b16,b32,b64,...}`.
pub fn xor<T: std::ops::BitXor<Output = T>>(lhs: T, rhs: T) -> T {
    lhs ^ rhs
}

/// `not.{pred,b16,b32,b64,...}`.
pub fn not<T: std::ops::Not<Output = T>>(value: T) -> T {
    !value
}

/// `lop3.b32` with immediate truth table `lut` (PTX `a, b, c` row order).
pub fn lop3_b32(a: u32, b: u32, c: u32, lut: u8) -> u32 {
    let mut result = 0_u32;
    for row in 0_u8..8 {
        if lut & (1_u8 << row) == 0 {
            continue;
        }
        let a_mask = if row & 0b100 == 0 { !a } else { a };
        let b_mask = if row & 0b010 == 0 { !b } else { b };
        let c_mask = if row & 0b001 == 0 { !c } else { c };
        result |= a_mask & b_mask & c_mask;
    }
    result
}

/// `lop3.{and,or}.b32 d|p, a, b, c, lut, q`: returns the data result and the
/// predicate `(d != 0) BoolOp q`. PTX defines only `.and`/`.or`.
pub fn lop3_bool_b32(
    a: u32,
    b: u32,
    c: u32,
    lut: u8,
    bool_op: BoolOp,
    q: bool,
) -> OpResult<(u32, bool)> {
    if bool_op == BoolOp::Xor {
        return Err(OpError::message("lop3 BoolOp must be .and or .or"));
    }
    let data = lop3_b32(a, b, c, lut);
    Ok((data, bool_op.apply(data != 0, q)))
}

/// Integer carrier of `shl`/`shr`.
///
/// PTX states for both mnemonics that "Shift amounts greater than the register
/// width N are clamped to N" (PTX ISA 9.7.8.8 `shl`, 9.7.8.9 `shr`), so an
/// amount of N or more produces the fully shifted-out result. `shr` fill
/// follows PTX's own split: signed shifts fill with the sign bit, unsigned and
/// untyped (`.b*`) shifts fill with 0 -- bind `.b*` to the unsigned carrier.
pub trait PtxShift: Copy {
    fn checked_shl(self, shift: u32) -> Option<Self>;
    fn checked_shr(self, shift: u32) -> Option<Self>;
    /// Value left by an out-of-range right shift.
    fn shr_fill(self) -> Self;
}

macro_rules! ptx_shift {
    ($scalar:ty, $fill:expr) => {
        impl PtxShift for $scalar {
            fn checked_shl(self, shift: u32) -> Option<Self> {
                <$scalar>::checked_shl(self, shift)
            }
            fn checked_shr(self, shift: u32) -> Option<Self> {
                <$scalar>::checked_shr(self, shift)
            }
            fn shr_fill(self) -> Self {
                let fill: fn($scalar) -> $scalar = $fill;
                fill(self)
            }
        }
    };
}

ptx_shift!(u16, |_value| 0);
ptx_shift!(u32, |_value| 0);
ptx_shift!(u64, |_value| 0);
ptx_shift!(i16, |value| value >> (i16::BITS - 1));
ptx_shift!(i32, |value| value >> (i32::BITS - 1));
ptx_shift!(i64, |value| value >> (i64::BITS - 1));

/// `shl.{b16,b32,b64,s32,s64,u32,u64}` with PTX clamping.
pub fn shl<T: PtxShift + Default>(value: T, shift: u32) -> T {
    value.checked_shl(shift).unwrap_or_default()
}

/// `shr.{b16,u16,s16,b32,s32,u32,b64,s64,u64}` with PTX clamping and fill.
pub fn shr<T: PtxShift>(value: T, shift: u32) -> T {
    value.checked_shr(shift).unwrap_or_else(|| value.shr_fill())
}

/// Byte-selection mode of `prmt.b32`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrmtMode {
    /// Unmodified generic form: four 4-bit selectors, bit 3 replicates sign.
    Generic,
    F4e,
    B4e,
    Rc8,
    Ecl,
    Ecr,
    Rc16,
}

/// `prmt.b32{.mode}`. All modes select bytes from the same `b:a` register
/// pair; specialized modes derive selectors from `c[1:0]` and only the generic
/// form can replicate a sign.
pub fn prmt_b32(a: u32, b: u32, control: u32, mode: PrmtMode) -> u32 {
    let select = |c: u32, byte: u32| -> u32 {
        match mode {
            PrmtMode::Generic => (c >> (4 * byte)) & 15,
            PrmtMode::F4e => (c & 3) + byte,
            PrmtMode::B4e => ((c & 3) + 8 - byte) & 7,
            PrmtMode::Rc8 => c & 3,
            PrmtMode::Ecl => byte.max(c & 3),
            PrmtMode::Ecr => byte.min(c & 3),
            PrmtMode::Rc16 => ((c & 1) * 2) + (byte & 1),
        }
    };
    let source = (u64::from(b) << 32) | u64::from(a);
    let mut output = 0_u32;
    for byte in 0..4 {
        let selector = select(control, byte);
        let value = ((source >> ((selector & 7) * 8)) & 0xff) as u32;
        let value = if selector & 8 != 0 {
            if value & 0x80 != 0 {
                0xff
            } else {
                0
            }
        } else {
            value
        };
        output |= value << (byte * 8);
    }
    output
}

/// `fns.b32`: find the `offset`-th set bit of `mask` relative to `base`.
/// A base outside `0..=31` is undefined and fails closed.
pub fn fns_b32(mask: u32, base: u32, offset: i32) -> OpResult<u32> {
    if base >= 32 {
        return Err(OpError::message(
            "fns.b32 base is outside the defined 0..31 range",
        ));
    }
    Ok(crate::scalar::ptx_fns_b32(mask, base, offset))
}
