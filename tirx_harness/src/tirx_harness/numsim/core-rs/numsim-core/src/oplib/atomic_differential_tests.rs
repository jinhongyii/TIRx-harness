//! Differential test: the engine's atomic RMW (`interp::handlers::mem::rmw_elem`,
//! the single implementation) against the legacy-ported atomic kernels that
//! used to live in `numsim_oplib::atomic::rmw` (now the `legacy` oracle below), per element, over edge values of every dtype the
//! atomics accept. CAS/Exch act on the whole operand in the engine (a 128-bit
//! CAS is all-or-nothing), so only the per-element ops are compared here.
//!
//! Lives in-crate because `interp::handlers::mem` is `pub(crate)`. Owner: W4. Agreed with W2 (2026-10-08) before retiring `atomic/rmw.rs`.

use crate::interp::handlers::mem::rmw_elem;
use crate::program::AtomOp;
use crate::Dtype;
use legacy::{self as lib, AtomicOp, AtomicSpace};

/// The legacy-ported atomic kernels formerly in `numsim_oplib::atomic::rmw`
/// (retired 2026-10-08, W4/W2), kept verbatim as the independent oracle for
/// the engine's `rmw_elem`. Not used outside this test.
#[allow(dead_code)]
mod legacy {
    use numsim_oplib::cvt::formats::{
        bf16_bits_to_f32, f32_to_bf16_bits, f32_to_fp16_bits, fp16_bits_to_f32,
    };
    use numsim_oplib::scalar::{
        add_f32, add_f32_ftz, ptx_max_f32, ptx_min_f32, F32RoundingMode, U64x2,
    };
    use numsim_oplib::types::{OpError, OpResult};

    /// PTX/CUDA atomic operation (legacy `RawAtomicOperation`).
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum AtomicOp {
        Add,
        /// `atom.add.noftz.f32` (PTX 9.4): round-to-nearest without flushing.
        AddNoFtz,
        BitAnd,
        BitOr,
        BitXor,
        Exchange,
        /// `.inc`: wraps to zero once `old >= operand`.
        Increment,
        /// `.dec`: reloads `operand` when `old == 0 || old > operand`.
        Decrement,
        Minimum,
        Maximum,
    }

    /// PTX state space of the atomic's target (only `.f32` add depends on it).
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum AtomicSpace {
        Global,
        Shared,
    }

    fn unsupported<T>(operation: AtomicOp) -> OpResult<T> {
        Err(OpError::message(format!(
            "atomic operation {operation:?} is invalid for this scalar type"
        )))
    }

    /// `atom/red.{add,min,max}.s32`.
    pub fn atomic_i32(operation: AtomicOp, old: i32, operand: i32) -> OpResult<i32> {
        match operation {
            AtomicOp::Add => Ok(old.wrapping_add(operand)),
            AtomicOp::Minimum => Ok(old.min(operand)),
            AtomicOp::Maximum => Ok(old.max(operand)),
            _ => unsupported(operation),
        }
    }

    /// `atom/red.{min,max}.s64`.
    pub fn atomic_i64(operation: AtomicOp, old: i64, operand: i64) -> OpResult<i64> {
        match operation {
            AtomicOp::Minimum => Ok(old.min(operand)),
            AtomicOp::Maximum => Ok(old.max(operand)),
            _ => unsupported(operation),
        }
    }

    /// `atom/red.{add,and,or,xor,exch,inc,dec,min,max}.{u32,b32}`.
    pub fn atomic_u32(operation: AtomicOp, old: u32, operand: u32) -> OpResult<u32> {
        match operation {
            AtomicOp::Add => Ok(old.wrapping_add(operand)),
            AtomicOp::BitAnd => Ok(old & operand),
            AtomicOp::BitOr => Ok(old | operand),
            AtomicOp::BitXor => Ok(old ^ operand),
            AtomicOp::Exchange => Ok(operand),
            AtomicOp::Increment => Ok(if old >= operand { 0 } else { old + 1 }),
            AtomicOp::Decrement => Ok(if old == 0 || old > operand {
                operand
            } else {
                old - 1
            }),
            AtomicOp::Minimum => Ok(old.min(operand)),
            AtomicOp::Maximum => Ok(old.max(operand)),
            AtomicOp::AddNoFtz => unsupported(operation),
        }
    }

    /// `atom/red.{add,and,or,xor,exch,min,max}.{u64,b64}`.
    pub fn atomic_u64(operation: AtomicOp, old: u64, operand: u64) -> OpResult<u64> {
        match operation {
            AtomicOp::Add => Ok(old.wrapping_add(operand)),
            AtomicOp::BitAnd => Ok(old & operand),
            AtomicOp::BitOr => Ok(old | operand),
            AtomicOp::BitXor => Ok(old ^ operand),
            AtomicOp::Exchange => Ok(operand),
            AtomicOp::Minimum => Ok(old.min(operand)),
            AtomicOp::Maximum => Ok(old.max(operand)),
            _ => unsupported(operation),
        }
    }

    /// `atom.exch.b128`.
    pub fn atomic_u64x2(operation: AtomicOp, _old: U64x2, operand: U64x2) -> OpResult<U64x2> {
        match operation {
            AtomicOp::Exchange => Ok(operand),
            _ => unsupported(operation),
        }
    }

    /// `atom/red.add{.noftz}.f32` and CUDA `atomicAdd(float*)`.
    ///
    /// A global `.add` flushes subnormal inputs and result (measured); `.noftz`
    /// rounds to nearest without flushing; a shared `.add` is a plain IEEE add.
    pub fn atomic_f32(
        operation: AtomicOp,
        old: f32,
        operand: f32,
        space: AtomicSpace,
    ) -> OpResult<f32> {
        match operation {
            AtomicOp::Add if space == AtomicSpace::Global => {
                Ok(add_f32_ftz(old, operand, F32RoundingMode::Nearest))
            }
            AtomicOp::AddNoFtz => Ok(add_f32(old, operand, F32RoundingMode::Nearest)),
            AtomicOp::Add => Ok(old + operand),
            _ => unsupported(operation),
        }
    }

    /// `atom/red.add.f64` and CUDA `atomicAdd(double*)`.
    pub fn atomic_f64(operation: AtomicOp, old: f64, operand: f64) -> OpResult<f64> {
        match operation {
            AtomicOp::Add => Ok(old + operand),
            _ => unsupported(operation),
        }
    }

    /// CUDA `atomicAdd(__half*)` / `atom.add.noftz.f16` on raw payloads.
    pub fn atomic_add_f16(old: u16, operand: u16) -> u16 {
        f32_to_fp16_bits(fp16_bits_to_f32(old) + fp16_bits_to_f32(operand))
    }

    /// CUDA `atomicAdd(__nv_bfloat16*)` / `atom.add.noftz.bf16` on raw payloads.
    pub fn atomic_add_bf16(old: u16, operand: u16) -> u16 {
        f32_to_bf16_bits(bf16_bits_to_f32(old) + bf16_bits_to_f32(operand))
    }

    /// CUDA `atomicAdd(__half2*)`: each 16-bit component is added independently
    /// (element 0 in the low half).
    pub fn atomic_add_f16x2(old: u32, operand: u32) -> u32 {
        let lane = |shift: u32| {
            u32::from(atomic_add_f16(
                (old >> shift) as u16,
                (operand >> shift) as u16,
            ))
        };
        lane(0) | (lane(16) << 16)
    }

    /// CUDA `atomicAdd(__nv_bfloat162*)`.
    pub fn atomic_add_bf16x2(old: u32, operand: u32) -> u32 {
        let lane = |shift: u32| {
            u32::from(atomic_add_bf16(
                (old >> shift) as u16,
                (operand >> shift) as u16,
            ))
        };
        lane(0) | (lane(16) << 16)
    }

    /// One element of `atom/red.{add,min,max}.noftz.{f16,bf16}{x2}{.vN}`.
    ///
    /// `min`/`max` follow `ptx_{min,max}_f32` without `.ftz`/`.NaN` propagation.
    pub fn atomic_half(operation: AtomicOp, old: u16, operand: u16, bf16: bool) -> OpResult<u16> {
        type Codec = (fn(u16) -> f32, fn(f32) -> u16);
        let (decode, encode): Codec = if bf16 {
            (bf16_bits_to_f32, f32_to_bf16_bits)
        } else {
            (fp16_bits_to_f32, f32_to_fp16_bits)
        };
        let (old, operand) = (decode(old), decode(operand));
        let result = match operation {
            AtomicOp::Add => old + operand,
            AtomicOp::Minimum => ptx_min_f32(old, operand, false, false),
            AtomicOp::Maximum => ptx_max_f32(old, operand, false, false),
            _ => return unsupported(operation),
        };
        Ok(encode(result))
    }

    /// Half-vector atomics (2/4/8 elements): each element updates independently.
    pub fn atomic_half_vector<const N: usize>(
        operation: AtomicOp,
        old: [u16; N],
        operand: [u16; N],
        bf16: bool,
    ) -> OpResult<[u16; N]> {
        if !matches!(N, 2 | 4 | 8) {
            return Err(OpError::message(
                "half-vector atomics require global memory and 2/4/8 half elements",
            ));
        }
        let mut result = [0_u16; N];
        for index in 0..N {
            result[index] = atomic_half(operation, old[index], operand[index], bf16)?;
        }
        Ok(result)
    }

    /// One `.f32` component of a global vector add (`noftz` selects `.noftz`).
    fn vector_f32_component(old: f32, operand: f32, noftz: bool) -> f32 {
        let operation = if noftz {
            AtomicOp::AddNoFtz
        } else {
            AtomicOp::Add
        };
        atomic_f32(operation, old, operand, AtomicSpace::Global).expect("add is valid for f32")
    }

    /// CUDA `atomicAdd(float2*)` / `atom.add{.noftz}.v2.f32` (global only);
    /// component 0 is the low 32 bits.
    pub fn atomic_add_f32x2(old: u64, operand: u64, noftz: bool) -> u64 {
        let lane = |shift: u32| {
            u64::from(
                vector_f32_component(
                    f32::from_bits((old >> shift) as u32),
                    f32::from_bits((operand >> shift) as u32),
                    noftz,
                )
                .to_bits(),
            ) << shift
        };
        lane(0) | lane(32)
    }

    /// CUDA `atomicAdd(float4*)` / `atom.add{.noftz}.v4.f32` (global only).
    pub fn atomic_add_f32x4(old: [f32; 4], operand: [f32; 4], noftz: bool) -> [f32; 4] {
        std::array::from_fn(|index| vector_f32_component(old[index], operand[index], noftz))
    }

    /// `atom.cas.b{16,32,64,128}` / CUDA `atomicCAS`: whole-value bitwise
    /// compare; returns the value stored afterwards (the caller returns `old`).
    pub fn atomic_cas_bytes(old: &[u8], compare: &[u8], value: &[u8]) -> Vec<u8> {
        if old == compare {
            value.to_vec()
        } else {
            old.to_vec()
        }
    }

    /// [`atomic_cas_bytes`] for payloads up to 64 bits.
    pub fn atomic_cas_u64(old: u64, compare: u64, value: u64) -> u64 {
        if old == compare {
            value
        } else {
            old
        }
    }
}

const F32_EDGES: &[u32] = &[
    0x0000_0000,
    0x8000_0000,
    0x0000_0001,
    0x8000_0001,
    0x007f_ffff,
    0x0080_0000,
    0x3f80_0000,
    0xbf80_0000,
    0x7f7f_ffff,
    0xff7f_ffff,
    0x7f80_0000,
    0xff80_0000,
    0x7fc0_0000,
    0xffc0_0001,
    0x7f80_0123,
    0x3400_0000,
    0x4b80_0000,
];
const F64_EDGES: &[u64] = &[
    0,
    0x8000_0000_0000_0000,
    1,
    0x000f_ffff_ffff_ffff,
    0x3ff0_0000_0000_0000,
    0xbff0_0000_0000_0000,
    0x7fef_ffff_ffff_ffff,
    0x7ff0_0000_0000_0000,
    0xfff0_0000_0000_0000,
    0x7ff8_0000_0000_0000,
    0xfff8_0000_0000_0042,
    0x7ff0_0000_0000_0007,
    0x3ca0_0000_0000_0000,
];
const H16_EDGES: &[u16] = &[
    0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x7bff, 0xfbff, 0x7c00, 0xfc00,
    0x7e00, 0xfe01, 0x7c01, 0x3f80, 0xbf80, 0x0080, 0x007f, 0x7f7f, 0x7f80, 0xff80, 0x7fc0, 0x7f81,
];
const I_EDGES: &[u64] = &[
    0,
    1,
    2,
    3,
    0x7f,
    0x80,
    0xffff,
    0x7fff_ffff,
    0x8000_0000,
    0xffff_fffe,
    0xffff_ffff,
    0x1_0000_0000,
    0x7fff_ffff_ffff_ffff,
    0x8000_0000_0000_0000,
    u64::MAX,
];

thread_local! {
    static COMPARED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn lib_op(op: AtomOp) -> AtomicOp {
    match op {
        AtomOp::Add => AtomicOp::Add,
        AtomOp::Min => AtomicOp::Minimum,
        AtomOp::Max => AtomicOp::Maximum,
        AtomOp::Inc => AtomicOp::Increment,
        AtomOp::Dec => AtomicOp::Decrement,
        AtomOp::And => AtomicOp::BitAnd,
        AtomOp::Or => AtomicOp::BitOr,
        AtomOp::Xor => AtomicOp::BitXor,
        AtomOp::Exch => AtomicOp::Exchange,
        AtomOp::Cas => unreachable!("whole-operand op"),
    }
}

/// One disagreement, formatted for the report.
fn record(out: &mut Vec<String>, what: &str, old: u64, val: u64, engine: u64, legacy: u64) {
    COMPARED.with(|c| c.set(c.get() + 1));
    if engine != legacy {
        out.push(format!(
            "{what}: old={old:#x} val={val:#x} engine={engine:#x} oplib={legacy:#x}"
        ));
    }
}

fn disagreements() -> Vec<String> {
    use AtomOp::*;
    let mut out = Vec::new();
    // Integers.
    for &op in &[Add, Min, Max, Inc, Dec, And, Or, Xor, Exch] {
        for &a in I_EDGES {
            for &b in I_EDGES {
                let (a32, b32) = (a as u32, b as u32);
                if let Ok(l) = lib::atomic_u32(lib_op(op), a32, b32) {
                    let e = rmw_elem(op, Dtype::U32, a32.into(), b32.into(), 0, false).unwrap();
                    record(
                        &mut out,
                        &format!("u32 {op:?}"),
                        a32.into(),
                        b32.into(),
                        e,
                        l.into(),
                    );
                }
                if let Ok(l) = lib::atomic_i32(lib_op(op), a32 as i32, b32 as i32) {
                    let e = rmw_elem(op, Dtype::S32, a32.into(), b32.into(), 0, false).unwrap();
                    record(
                        &mut out,
                        &format!("s32 {op:?}"),
                        a32.into(),
                        b32.into(),
                        e,
                        u64::from(l as u32),
                    );
                }
                if let Ok(l) = lib::atomic_u64(lib_op(op), a, b) {
                    let e = rmw_elem(op, Dtype::U64, a, b, 0, false).unwrap();
                    record(&mut out, &format!("u64 {op:?}"), a, b, e, l);
                }
                if let Ok(l) = lib::atomic_i64(lib_op(op), a as i64, b as i64) {
                    let e = rmw_elem(op, Dtype::S64, a, b, 0, false).unwrap();
                    record(&mut out, &format!("s64 {op:?}"), a, b, e, l as u64);
                }
            }
        }
    }
    // f32 add: global `.add` (FTZ), `.noftz`, and shared `.add` (no FTZ).
    for &a in F32_EDGES {
        for &b in F32_EDGES {
            let (fa, fb) = (f32::from_bits(a), f32::from_bits(b));
            let g = lib::atomic_f32(AtomicOp::Add, fa, fb, AtomicSpace::Global).unwrap();
            let e = rmw_elem(Add, Dtype::F32, a.into(), b.into(), 0, true).unwrap();
            record(
                &mut out,
                "f32 add global(ftz)",
                a.into(),
                b.into(),
                e,
                g.to_bits().into(),
            );
            let n = lib::atomic_f32(AtomicOp::AddNoFtz, fa, fb, AtomicSpace::Global).unwrap();
            let e = rmw_elem(Add, Dtype::F32, a.into(), b.into(), 0, false).unwrap();
            record(
                &mut out,
                "f32 add.noftz",
                a.into(),
                b.into(),
                e,
                n.to_bits().into(),
            );
            // Delta D13: with both operands NaN the engine keeps `old`'s NaN
            // (PTX operand order `*a = old + b`, D8 first-NaN rule, quieted);
            // the legacy shared form returned the host's choice (`val`'s).
            let s = if fa.is_nan() && fb.is_nan() {
                f32::from_bits(a | 0x0040_0000)
            } else {
                lib::atomic_f32(AtomicOp::Add, fa, fb, AtomicSpace::Shared).unwrap()
            };
            record(
                &mut out,
                "f32 add shared",
                a.into(),
                b.into(),
                e,
                s.to_bits().into(),
            );
        }
    }
    // f64 add.
    for &a in F64_EDGES {
        for &b in F64_EDGES {
            // Delta D13, as for f32 above.
            let l = if f64::from_bits(a).is_nan() && f64::from_bits(b).is_nan() {
                f64::from_bits(a | 0x0008_0000_0000_0000)
            } else {
                lib::atomic_f64(AtomicOp::Add, f64::from_bits(a), f64::from_bits(b)).unwrap()
            };
            let e = rmw_elem(Add, Dtype::F64, a, b, 0, false).unwrap();
            record(&mut out, "f64 add", a, b, e, l.to_bits());
        }
    }
    // f16 / bf16 add, min, max (the `atom.*.noftz.{f16,bf16}` element rule).
    for (dtype, bf16) in [(Dtype::F16, false), (Dtype::BF16, true)] {
        for &op in &[Add, Min, Max] {
            for &a in H16_EDGES {
                for &b in H16_EDGES {
                    let l = lib::atomic_half(lib_op(op), a, b, bf16).unwrap();
                    let e = rmw_elem(op, dtype, a.into(), b.into(), 0, false).unwrap();
                    record(
                        &mut out,
                        &format!("{dtype:?} {op:?}"),
                        a.into(),
                        b.into(),
                        e,
                        l.into(),
                    );
                }
            }
        }
    }
    out
}

#[test]
fn engine_atomic_rmw_matches_the_legacy_oplib_kernels() {
    let found = disagreements();
    let mut classes: std::collections::BTreeMap<String, usize> = Default::default();
    for line in &found {
        *classes
            .entry(line.split(':').next().unwrap().to_string())
            .or_default() += 1;
    }
    for line in found.iter().take(60) {
        eprintln!("{line}");
    }
    eprintln!("classes: {classes:?}");
    assert!(
        found.is_empty(),
        "{} disagreements: {classes:?}",
        found.len()
    );
    // Every class above was exercised (ints x 9 ops, f32 x 3 modes, f64, half x 2 x 3).
    let compared = COMPARED.with(|c| c.get());
    assert!(compared > 9_000, "only {compared} comparisons");
}
