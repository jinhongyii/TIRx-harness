//! Reviewed `func_call` helpers at the contract boundary. Expected values are
//! hand-derived from the legacy helper semantics (and the legacy tests'
//! vectors), not recomputed through the kernels.

use crate::dtype::Ty;
use crate::oplib::{resolve_ptx, OpError, OpErrorKind, PtxIo};
use crate::program::OpKey;
use numsim_types::{WarpMask, WarpValue, WARP_SIZE};

const LANE: usize = 3;

fn digest(helper: &str) -> String {
    let d = super::REVIEWED.iter().find(|r| r.0 == helper).unwrap().1;
    format!("source_sha256={d}")
}

fn run_with(helper: &str, mods: Vec<String>, dsts: &[Ty], srcs: &[(Ty, u64)]) -> Result<u64, OpError> {
    let key = OpKey { name: format!("tirx.cuda.func_call.{helper}"), mods };
    let src_tys: Vec<Ty> = srcs.iter().map(|s| s.0).collect();
    let f = resolve_ptx(&key, dsts, &src_tys)?;
    let values: Vec<WarpValue<u64>> = srcs.iter().map(|&(_, v)| [v; WARP_SIZE]).collect();
    let mut out = vec![[0x5555_5555_5555_5555u64; WARP_SIZE]; 1];
    let mut io = PtxIo { dsts: &mut out, dst_tys: dsts, srcs: &values, src_tys: &src_tys, mask: WarpMask::lane(LANE) };
    f.call(&mut io)?;
    assert_eq!(out[0][LANE + 1], 0x5555_5555_5555_5555, "inactive lane written");
    Ok(out[0][LANE])
}

fn run(helper: &str, dsts: &[Ty], srcs: &[(Ty, u64)]) -> u64 {
    run_with(helper, vec![digest(helper)], dsts, srcs).unwrap()
}

fn f32b(v: f32) -> u64 {
    u64::from(v.to_bits())
}

#[test]
fn shl_u32_clamp_gives_zero_for_counts_of_32_and_more() {
    let shl = |v: u32, s: u32| run("shl_u32_clamp", &[Ty::U32], &[(Ty::U32, v.into()), (Ty::U32, s.into())]);
    assert_eq!(shl(0xFFFF_FFFF, 0), 0xFFFF_FFFF);
    assert_eq!(shl(3, 7), 0x180);
    assert_eq!(shl(0x8000_0001, 31), 0x8000_0000);
    for count in [32, 33, 63, 255] {
        assert_eq!(shl(0xDEAD_BEEF, count), 0, "count {count}");
    }
}

#[test]
fn combine_int_frac_ex2_adds_the_shifted_integer_bits() {
    // (0x4B40_0005 << 23) mod 2^32 = 0x0280_0000; + 0x3F80_0000.
    let r = run("combine_int_frac_ex2", &[Ty::F32], &[(Ty::F32, 0x4B40_0005), (Ty::F32, 0x3F80_0000)]);
    assert_eq!(r, 0x4200_0000);
    // Wraps: 0x0000_01FF << 23 = 0xFF80_0000; + 0x0100_0000 = 0x0080_0000 (mod 2^32).
    let r = run("combine_int_frac_ex2", &[Ty::F32], &[(Ty::F32, 0x1FF), (Ty::F32, 0x0100_0000)]);
    assert_eq!(r, 0x0080_0000);
}

#[test]
fn flashkda_fmaf_rn_rounds_once() {
    // (1 + 2^-12)(1 - 2^-12) - 1 = -2^-24 exactly; a separate multiply would round to 0.
    let a = f32b(1.0 + 2f32.powi(-12));
    let b = f32b(1.0 - 2f32.powi(-12));
    let r = run("flashkda_fmaf_rn", &[Ty::F32], &[(Ty::F32, a), (Ty::F32, b), (Ty::F32, f32b(-1.0))]);
    assert_eq!(r, 0xB380_0000);
}

#[test]
fn flashkda_rsqrtf_is_reciprocal_of_ieee_sqrt() {
    let rsqrt = |x: f32| run("flashkda_rsqrtf", &[Ty::F32], &[(Ty::F32, f32b(x))]);
    assert_eq!(rsqrt(0.25), 0x4000_0000);
    assert_eq!(rsqrt(16.0), 0x3E80_0000);
    // 1/sqrt(2): sqrt(2) rounds to 0x3FB504F3, then 1/that rounds to 0x3F3504F3.
    assert_eq!(rsqrt(2.0), 0x3F35_04F3);
}

#[test]
fn approx_helpers_use_the_legacy_representatives() {
    let tanh = |x: f32| run("flashkda_tanh_approx", &[Ty::F32], &[(Ty::F32, f32b(x))]);
    assert_eq!(tanh(0.0), 0);
    assert_eq!(tanh(-0.0), 0x8000_0000);
    assert_eq!(tanh(20.0), f32b(1.0));
    let lg2 = |bits: u64| run("gdn_lg2_approx_ftz", &[Ty::F32], &[(Ty::F32, bits)]);
    assert_eq!(lg2(f32b(4.0)), f32b(2.0));
    assert_eq!(lg2(f32b(0.25)), f32b(-2.0));
    // Smallest subnormal flushes to +0, so lg2 gives -inf.
    assert_eq!(lg2(1), 0xFF80_0000);
}

#[test]
fn fma_scale_sub_f32x2_computes_both_packed_lanes() {
    let pack = |lo: f32, hi: f32| (f32b(hi) << 32) | f32b(lo);
    let r = run(
        "tvm_builtin_fma_scale_sub_f32x2",
        &[Ty::U64],
        &[(Ty::U64, pack(2.0, 1.5)), (Ty::U64, pack(0.5, 2.0)), (Ty::U64, pack(0.25, -0.75))],
    );
    // x: 2*0.5 - 0.25 = 0.75; y: 1.5*2 + 0.75 = 3.75.
    assert_eq!(r, 0x4070_0000_3F40_0000);
}

#[test]
fn smem_desc_add_16b_offset_wraps_only_the_low_word() {
    let r = run(
        "tvm_builtin_smem_desc_add_16B_offset",
        &[Ty::U64],
        &[(Ty::U64, 0xFEDC_BA98_FFFF_FFF0), (Ty::S32, u64::from(-41i32 as u32))],
    );
    assert_eq!(r, 0xFEDC_BA98_FFFF_FFC7);
}

#[test]
fn helpers_fail_closed_without_the_reviewed_body_digest() {
    let srcs = [(Ty::U32, 1u64), (Ty::U32, 1)];
    let kind = |mods: Vec<String>| run_with("shl_u32_clamp", mods, &[Ty::U32], &srcs).unwrap_err().kind;
    assert_eq!(kind(vec![]), OpErrorKind::Unsupported);
    assert_eq!(kind(vec!["source_sha256=0000000000000000".into()]), OpErrorKind::Unsupported);
    assert_eq!(kind(vec![digest("shl_u32_clamp"), "x=y".into()]), OpErrorKind::Unsupported);
    // Wrong carrier widths.
    let err = run_with("shl_u32_clamp", vec![digest("shl_u32_clamp")], &[Ty::U64], &srcs).unwrap_err();
    assert_eq!(err.kind, OpErrorKind::Unsupported);
    // Pointer-based cast helpers touch memory: not a value op.
    let err = run_with("tvm_builtin_cast_float32x2_float16x2", vec![], &[], &[(Ty::U64, 0), (Ty::U64, 0)]).unwrap_err();
    assert_eq!(err.kind, OpErrorKind::Unsupported);
}
