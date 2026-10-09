//! Monomorphic fast paths for scalar (one-slot) TIR ALU forms.
//!
//! The generic path (`tir.rs` `map_elems` over `alu::*`) decodes every lane
//! through `u128` element extraction and a runtime `(op, dtype)` match, which
//! costs ~300-800 ns per 32-lane call. Here the `(op, dtype)` match happens
//! once per call; each arm is a tight loop over `[u64; 32]` on the native Rust
//! type that LLVM can vectorize, followed by a masked blend. Every fast form
//! is bit-identical to the generic one (checked by `tir/tests.rs`
//! `fast_paths_match_the_generic_path`); anything not listed returns `None`
//! and falls through to the generic path. Lanes outside `mask` are never
//! written; value errors (division by zero) are reported for active lanes
//! only, with the generic path's messages.

use super::super::{OpError, OpResult};
use numsim_oplib::scalar::det;
use crate::dtype::{Dtype, Ty};
use crate::program::{BinOp, CmpOp, Rounding, TerOp, UnOp};
use crate::value::{WarpMask, WarpValue};
use numsim_oplib::cvt;
use numsim_oplib::scalar as sc;

type W = WarpValue<u64>;

/// Copy `r` into `out` for the lanes of `mask`.
#[inline(always)]
fn blend(out: &mut W, r: &W, mask: WarpMask) {
    if mask.is_all() {
        *out = *r;
        return;
    }
    let m = mask.bits();
    for l in 0..32 {
        if (m >> l) & 1 == 1 {
            out[l] = r[l];
        }
    }
}

/// One-slot scalar operand of `ty`, if this is a fast-path shape.
#[inline(always)]
fn scalar(ty: Ty, slots: usize) -> bool {
    ty.lanes == 1 && ty.elem.bits() <= 64 && slots >= 1
}

/// A native lane type with the register encoding of its `Dtype`.
trait Lane: Copy {
    fn get(x: u64) -> Self;
    fn put(self) -> u64;
}

macro_rules! lane {
    ($t:ty, $u:ty) => {
        impl Lane for $t {
            #[inline(always)]
            fn get(x: u64) -> Self {
                x as $u as $t
            }
            #[inline(always)]
            fn put(self) -> u64 {
                self as $u as u64
            }
        }
    };
}
lane!(u8, u8);
lane!(u16, u16);
lane!(u32, u32);
lane!(u64, u64);
lane!(i8, u8);
lane!(i16, u16);
lane!(i32, u32);
lane!(i64, u64);

impl Lane for f32 {
    #[inline(always)]
    fn get(x: u64) -> Self {
        f32::from_bits(x as u32)
    }
    #[inline(always)]
    fn put(self) -> u64 {
        u64::from(self.to_bits())
    }
}

impl Lane for f64 {
    #[inline(always)]
    fn get(x: u64) -> Self {
        f64::from_bits(x)
    }
    #[inline(always)]
    fn put(self) -> u64 {
        self.to_bits()
    }
}

#[inline(always)]
fn map1<A: Lane, R: Lane>(a: &W, out: &mut W, mask: WarpMask, f: impl Fn(A) -> R) {
    let mut r = [0u64; 32];
    for l in 0..32 {
        r[l] = f(A::get(a[l])).put();
    }
    blend(out, &r, mask);
}

#[inline(always)]
fn map2<T: Lane>(a: &W, b: &W, out: &mut W, mask: WarpMask, f: impl Fn(T, T) -> T) {
    let mut r = [0u64; 32];
    for l in 0..32 {
        r[l] = f(T::get(a[l]), T::get(b[l])).put();
    }
    blend(out, &r, mask);
}

#[inline(always)]
fn map3<T: Lane>(a: &W, b: &W, c: &W, out: &mut W, mask: WarpMask, f: impl Fn(T, T, T) -> T) {
    let mut r = [0u64; 32];
    for l in 0..32 {
        r[l] = f(T::get(a[l]), T::get(b[l]), T::get(c[l])).put();
    }
    blend(out, &r, mask);
}

#[inline(always)]
fn cmp_mask<T: Lane>(a: &W, b: &W, mask: WarpMask, f: impl Fn(T, T) -> bool) -> WarpMask {
    let mut bits = 0u32;
    for l in 0..32 {
        bits |= u32::from(f(T::get(a[l]), T::get(b[l]))) << l;
    }
    WarpMask(bits & mask.bits())
}

/// Checked per-active-lane op (division family).
#[inline(always)]
fn try2<T: Lane>(a: &W, b: &W, out: &mut W, mask: WarpMask, f: impl Fn(T, T) -> OpResult<T>) -> OpResult {
    for l in mask.lanes() {
        out[l] = f(T::get(a[l]), T::get(b[l]))?.put();
    }
    Ok(())
}

macro_rules! int_binary {
    ($t:ty, $signed:expr, $d:expr, $op:expr, $a:expr, $b:expr, $out:expr, $mask:expr) => {{
        let (a, b, out, mask) = ($a, $b, $out, $mask);
        let d: Dtype = $d;
        match $op {
            BinOp::Add => map2::<$t>(a, b, out, mask, |x, y| x.wrapping_add(y)),
            BinOp::Sub => map2::<$t>(a, b, out, mask, |x, y| x.wrapping_sub(y)),
            BinOp::Mul => map2::<$t>(a, b, out, mask, |x, y| x.wrapping_mul(y)),
            BinOp::Min => map2::<$t>(a, b, out, mask, |x, y| x.min(y)),
            BinOp::Max => map2::<$t>(a, b, out, mask, |x, y| x.max(y)),
            BinOp::And => map2::<$t>(a, b, out, mask, |x, y| x & y),
            BinOp::Or => map2::<$t>(a, b, out, mask, |x, y| x | y),
            BinOp::Xor => map2::<$t>(a, b, out, mask, |x, y| x ^ y),
            BinOp::Shl => map2::<$t>(a, b, out, mask, |x, y| x.wrapping_shl(y as u32)),
            BinOp::Shr => map2::<$t>(a, b, out, mask, |x, y| x.wrapping_shr(y as u32)),
            BinOp::Div | BinOp::Mod => {
                let div = $op == BinOp::Div;
                let what = if div { "division" } else { "remainder" };
                return Some(try2::<$t>(a, b, out, mask, |x, y| {
                    if y == 0 || ($signed && x == <$t>::MIN && y as i128 == -1) {
                        return Err(OpError::invalid(format!(
                            "invalid truncating {what}: {} / {} ({d})",
                            x as i128, y as i128
                        )));
                    }
                    Ok(if div { x / y } else { x % y })
                }));
            }
            BinOp::FloorDiv | BinOp::FloorMod => {
                let div = $op == BinOp::FloorDiv;
                return Some(try2::<$t>(a, b, out, mask, |x, y| {
                    if $signed {
                        // Legacy: `floor_*_i64(a as i64, b as i64)? as T`.
                        let (x, y) = (x as i64, y as i64);
                        let r = if div { sc::floor_div_i64(x, y) } else { sc::floor_mod_i64(x, y) };
                        Ok(r.map_err(OpError::from)? as $t)
                    } else {
                        if y == 0 {
                            return Err(OpError::invalid(format!(
                                "invalid unsigned floor division by zero ({d})"
                            )));
                        }
                        Ok(if div { x / y } else { x % y })
                    }
                }));
            }
            _ => return None,
        }
    }};
}

macro_rules! per_int_type {
    ($d:expr, $m:ident, $($args:tt)*) => {
        match $d {
            Dtype::U8 => $m!(u8, false, $($args)*),
            Dtype::U16 => $m!(u16, false, $($args)*),
            Dtype::U32 => $m!(u32, false, $($args)*),
            Dtype::U64 => $m!(u64, false, $($args)*),
            Dtype::S8 => $m!(i8, true, $($args)*),
            Dtype::S16 => $m!(i16, true, $($args)*),
            Dtype::S32 => $m!(i32, true, $($args)*),
            Dtype::S64 => $m!(i64, true, $($args)*),
            _ => return None,
        }
    };
}

/// (decode, encode) of a 16-bit float format.
type HalfCodec = (fn(u16) -> f32, fn(f32) -> u16);

/// f16/bf16 codec pair (the legacy codecs the generic path uses).
#[inline(always)]
fn half_codec(d: Dtype) -> HalfCodec {
    if d == Dtype::F16 {
        (cvt::fp16_bits_to_f32, cvt::f32_to_fp16_bits)
    } else {
        (cvt::bf16_bits_to_f32, cvt::f32_to_bf16_bits)
    }
}

#[inline(always)]
fn half2(a: &W, b: &W, out: &mut W, mask: WarpMask, d: Dtype, f: impl Fn(f32, f32) -> f32) {
    if d == Dtype::F16 {
        use crate::oplib::simd::{f16_to_f32, f32_to_f16_rne};
        let (x, y) = (f16_to_f32(&a.map(|v| v as u16)), f16_to_f32(&b.map(|v| v as u16)));
        let r: [f32; 32] = std::array::from_fn(|l| f(x[l], y[l]));
        blend(out, &f32_to_f16_rne(&r).map(u64::from), mask);
        return;
    }
    let (dec, enc) = half_codec(d);
    let mut r = [0u64; 32];
    for l in 0..32 {
        r[l] = u64::from(enc(f(dec(a[l] as u16), dec(b[l] as u16))));
    }
    blend(out, &r, mask);
}

pub(super) fn binary(op: BinOp, ty: Ty, a: &[W], b: &[W], out: &mut [W], mask: WarpMask) -> Option<OpResult> {
    if !scalar(ty, a.len().min(b.len()).min(out.len())) {
        return None;
    }
    let (a, b, out) = (&a[0], &b[0], &mut out[0]);
    let d = ty.elem;
    match d {
        Dtype::F32 => match op {
            BinOp::Add => map2::<f32>(a, b, out, mask, |x, y| sc::pin_nan2_f32(x, y, x + y)),
            BinOp::Sub => map2::<f32>(a, b, out, mask, |x, y| sc::pin_nan2_f32(x, y, x - y)),
            BinOp::Mul => map2::<f32>(a, b, out, mask, |x, y| sc::pin_nan2_f32(x, y, x * y)),
            BinOp::Div => map2::<f32>(a, b, out, mask, |x, y| sc::pin_nan2_f32(x, y, x / y)),
            BinOp::Min => map2::<f32>(a, b, out, mask, sc::cuda_f32_min),
            BinOp::Max => map2::<f32>(a, b, out, mask, sc::cuda_f32_max),
            _ => return None,
        },
        Dtype::F64 => match op {
            BinOp::Add => map2::<f64>(a, b, out, mask, |x, y| sc::pin_nan2_f64(x, y, x + y)),
            BinOp::Sub => map2::<f64>(a, b, out, mask, |x, y| sc::pin_nan2_f64(x, y, x - y)),
            BinOp::Mul => map2::<f64>(a, b, out, mask, |x, y| sc::pin_nan2_f64(x, y, x * y)),
            BinOp::Div => map2::<f64>(a, b, out, mask, |x, y| sc::pin_nan2_f64(x, y, x / y)),
            BinOp::Min => map2::<f64>(a, b, out, mask, sc::cuda_f64_min),
            BinOp::Max => map2::<f64>(a, b, out, mask, sc::cuda_f64_max),
            _ => return None,
        },
        // Generic: `(x op y) & 1` on the low byte (W4).
        Dtype::Pred => match op {
            BinOp::And => map2::<u64>(a, b, out, mask, |x, y| x & y & 1),
            BinOp::Or => map2::<u64>(a, b, out, mask, |x, y| (x | y) & 1),
            BinOp::Xor => map2::<u64>(a, b, out, mask, |x, y| (x ^ y) & 1),
            _ => return None,
        },
        Dtype::F16 | Dtype::BF16 => match op {
            BinOp::Add => half2(a, b, out, mask, d, |x, y| sc::pin_nan2_f32(x, y, x + y)),
            BinOp::Sub => half2(a, b, out, mask, d, |x, y| sc::pin_nan2_f32(x, y, x - y)),
            BinOp::Mul => half2(a, b, out, mask, d, |x, y| sc::pin_nan2_f32(x, y, x * y)),
            BinOp::Div => half2(a, b, out, mask, d, |x, y| sc::pin_nan2_f32(x, y, x / y)),
            BinOp::Min => half2(a, b, out, mask, d, sc::cuda_f32_min),
            BinOp::Max => half2(a, b, out, mask, d, sc::cuda_f32_max),
            _ => return None,
        },
        _ => {
            macro_rules! go {
                ($t:ty, $signed:expr, $($rest:tt)*) => {
                    int_binary!($t, $signed, d, op, a, b, out, mask)
                };
            }
            per_int_type!(d, go,)
        }
    }
    Some(Ok(()))
}

pub(super) fn ternary(op: TerOp, ty: Ty, a: &[W], b: &[W], c: &[W], out: &mut [W], mask: WarpMask) -> Option<OpResult> {
    if !scalar(ty, a.len().min(b.len()).min(c.len()).min(out.len())) {
        return None;
    }
    let TerOp::Fma = op;
    let (a, b, c, out) = (&a[0], &b[0], &c[0], &mut out[0]);
    match ty.elem {
        Dtype::F32 => {
            let r = crate::oplib::simd::fma_f32(&a.map(f32::get), &b.map(f32::get), &c.map(f32::get));
            blend(out, &r.map(f32::put), mask);
        }
        Dtype::F64 => {
            let r = crate::oplib::simd::fma_f64(&a.map(f64::get), &b.map(f64::get), &c.map(f64::get));
            blend(out, &r.map(f64::put), mask);
        }
        d => {
            macro_rules! go {
                ($t:ty, $signed:expr, $($rest:tt)*) => {
                    map3::<$t>(a, b, c, out, mask, |x, y, z| x.wrapping_mul(y).wrapping_add(z))
                };
            }
            per_int_type!(d, go,)
        }
    }
    Some(Ok(()))
}

pub(super) fn unary(op: UnOp, ty: Ty, a: &[W], out: &mut [W], mask: WarpMask) -> Option<OpResult> {
    if !scalar(ty, a.len().min(out.len())) {
        return None;
    }
    let (a, out) = (&a[0], &mut out[0]);
    match (ty.elem, op) {
        (Dtype::F32, UnOp::Neg) => map1::<u32, u32>(a, out, mask, |x| x ^ 0x8000_0000),
        (Dtype::F32, UnOp::Abs) => map1::<u32, u32>(a, out, mask, |x| x & 0x7fff_ffff),
        (Dtype::F32, UnOp::Sqrt) => map1::<f32, f32>(a, out, mask, f32::sqrt),
        (Dtype::F32, UnOp::Exp) => map1::<f32, f32>(a, out, mask, det::exp_f32),
        (Dtype::F32, UnOp::Exp2) => map1::<f32, f32>(a, out, mask, det::exp2_f32),
        (Dtype::F32, UnOp::Log) => map1::<f32, f32>(a, out, mask, det::ln_f32),
        (Dtype::F32, UnOp::Log2) => map1::<f32, f32>(a, out, mask, det::log2_f32),
        (Dtype::F32, UnOp::Rsqrt) => map1::<f32, f32>(a, out, mask, |x| 1.0_f32 / x.sqrt()),
        (Dtype::F64, UnOp::Neg) => map1::<u64, u64>(a, out, mask, |x| x ^ (1 << 63)),
        (Dtype::F64, UnOp::Abs) => map1::<u64, u64>(a, out, mask, |x| x & !(1 << 63)),
        (Dtype::F16 | Dtype::BF16, UnOp::Neg) => map1::<u16, u16>(a, out, mask, |x| x ^ 0x8000),
        (Dtype::F16 | Dtype::BF16, UnOp::Abs) => map1::<u16, u16>(a, out, mask, |x| x & 0x7fff),
        (Dtype::Pred, UnOp::Not | UnOp::BitNot) => map1::<u64, u64>(a, out, mask, |x| (x & 1) ^ 1),
        (d, UnOp::Neg | UnOp::Abs | UnOp::Not | UnOp::BitNot) => {
            macro_rules! go {
                ($t:ty, $signed:expr, $($rest:tt)*) => {
                    match op {
                        UnOp::Neg => map1::<$t, $t>(a, out, mask, |x| x.wrapping_neg()),
                        UnOp::Abs => map1::<$t, $t>(a, out, mask, |x| if $signed { (x as i64).wrapping_abs() as $t } else { x }),
                        _ => map1::<$t, $t>(a, out, mask, |x| !x),
                    }
                };
            }
            per_int_type!(d, go,)
        }
        _ => return None,
    }
    Some(Ok(()))
}

pub(super) fn compare(op: CmpOp, ty: Ty, a: &[W], b: &[W], mask: WarpMask) -> Option<OpResult<WarpMask>> {
    if !scalar(ty, a.len().min(b.len())) {
        return None;
    }
    let (a, b) = (&a[0], &b[0]);
    macro_rules! by_op {
        ($t:ty, $a:expr, $b:expr) => {
            match op {
                CmpOp::Eq => cmp_mask::<$t>($a, $b, mask, |x, y| x == y),
                CmpOp::Ne => cmp_mask::<$t>($a, $b, mask, |x, y| x != y),
                CmpOp::Lt => cmp_mask::<$t>($a, $b, mask, |x, y| x < y),
                CmpOp::Le => cmp_mask::<$t>($a, $b, mask, |x, y| x <= y),
                CmpOp::Gt => cmp_mask::<$t>($a, $b, mask, |x, y| x > y),
                CmpOp::Ge => cmp_mask::<$t>($a, $b, mask, |x, y| x >= y),
            }
        };
    }
    let r = match ty.elem {
        Dtype::F32 => by_op!(f32, a, b),
        Dtype::F64 => by_op!(f64, a, b),
        Dtype::Pred => {
            let (pa, pb) = (a.map(|x| x & 1), b.map(|x| x & 1));
            match op {
                CmpOp::Eq => cmp_mask::<u64>(&pa, &pb, mask, |x, y| x == y),
                CmpOp::Ne => cmp_mask::<u64>(&pa, &pb, mask, |x, y| x != y),
                _ => return None,
            }
        }
        Dtype::F16 | Dtype::BF16 => {
            let (dec, _) = half_codec(ty.elem);
            let (fa, fb) = (a.map(|x| u64::from(dec(x as u16).to_bits())), b.map(|x| u64::from(dec(x as u16).to_bits())));
            by_op!(f32, &fa, &fb)
        }
        d => {
            macro_rules! go {
                ($t:ty, $signed:expr, $($rest:tt)*) => {
                    by_op!($t, a, b)
                };
            }
            per_int_type!(d, go,)
        }
    };
    Some(Ok(r))
}

/// Fast scalar casts (`Rounding::Default`, no `sat`) with the generic path's
/// legacy C semantics: int<->int wrap/extend, int->float RN, float->int
/// truncate + saturate (NaN -> 0), f32<->f64, f32<->f16/bf16 via the legacy
/// codecs, int -> f16/bf16 through f32.
pub(super) fn cast(from: Ty, to: Ty, rnd: Rounding, sat: bool, src: &[W], out: &mut [W], mask: WarpMask) -> Option<OpResult> {
    if rnd != Rounding::Default || sat || !scalar(from, src.len()) || !scalar(to, out.len()) || from.elem == to.elem {
        return None;
    }
    let (a, out) = (&src[0], &mut out[0]);
    // Source as f64 / i64 / u64 helpers via a closure per source type.
    macro_rules! to_dst {
        ($s:ty, $conv_f32:expr, $conv_f64:expr) => {{
            let get = |x: u64| <$s as Lane>::get(x);
            match to.elem {
                Dtype::F32 => map1::<u64, f32>(a, out, mask, |x| $conv_f32(get(x))),
                Dtype::F64 => map1::<u64, f64>(a, out, mask, |x| $conv_f64(get(x))),
                Dtype::F16 => {
                    let v: [f32; 32] = std::array::from_fn(|l| $conv_f32(get(a[l])));
                    blend(out, &crate::oplib::simd::f32_to_f16_rne(&v).map(u64::from), mask);
                }
                Dtype::BF16 => map1::<u64, u16>(a, out, mask, |x| cvt::f32_to_bf16_bits($conv_f32(get(x)))),
                Dtype::U8 => map1::<u64, u8>(a, out, mask, |x| get(x) as u8),
                Dtype::U16 => map1::<u64, u16>(a, out, mask, |x| get(x) as u16),
                Dtype::U32 => map1::<u64, u32>(a, out, mask, |x| get(x) as u32),
                Dtype::U64 => map1::<u64, u64>(a, out, mask, |x| get(x) as u64),
                Dtype::S8 => map1::<u64, i8>(a, out, mask, |x| get(x) as i8),
                Dtype::S16 => map1::<u64, i16>(a, out, mask, |x| get(x) as i16),
                Dtype::S32 => map1::<u64, i32>(a, out, mask, |x| get(x) as i32),
                Dtype::S64 => map1::<u64, i64>(a, out, mask, |x| get(x) as i64),
                // Generic: `x != 0` (NaN != 0) as 0/1 (W4).
                Dtype::Pred => map1::<u64, u64>(a, out, mask, |x| u64::from(get(x) != <$s>::default())),
                _ => return None,
            }
        }};
    }
    match from.elem {
        Dtype::U8 => to_dst!(u8, |v| v as f32, |v| v as f64),
        Dtype::U16 => to_dst!(u16, |v| v as f32, |v| v as f64),
        Dtype::U32 => to_dst!(u32, |v| v as f32, |v| v as f64),
        Dtype::U64 => to_dst!(u64, |v| v as f32, |v| v as f64),
        Dtype::S8 => to_dst!(i8, |v| v as f32, |v| v as f64),
        Dtype::S16 => to_dst!(i16, |v| v as f32, |v| v as f64),
        Dtype::S32 => to_dst!(i32, |v| v as f32, |v| v as f64),
        Dtype::S64 => to_dst!(i64, |v| v as f32, |v| v as f64),
        Dtype::F32 => to_dst!(f32, |v| v, |v: f32| v as f64),
        Dtype::F64 => to_dst!(f64, |v: f64| v as f32, |v| v),
        // Generic: the low bit as an integer (W4); float destinations stay generic.
        Dtype::Pred => {
            if !to.elem.is_int() {
                return None;
            }
            map1::<u64, u64>(a, out, mask, |x| x & 1)
        }
        Dtype::F16 => {
            if !matches!(to.elem, Dtype::F32 | Dtype::F64) {
                return None;
            }
            to_dst!(u16, cvt::fp16_bits_to_f32, |v| cvt::fp16_bits_to_f32(v) as f64)
        }
        Dtype::BF16 => {
            if !matches!(to.elem, Dtype::F32 | Dtype::F64) {
                return None;
            }
            to_dst!(u16, cvt::bf16_bits_to_f32, |v| cvt::bf16_bits_to_f32(v) as f64)
        }
        _ => return None,
    }
    Some(Ok(()))
}
