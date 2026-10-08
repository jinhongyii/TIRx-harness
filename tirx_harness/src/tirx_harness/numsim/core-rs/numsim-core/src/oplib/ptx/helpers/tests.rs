//! Bit-exact helper cases at the contract boundary (`resolve_ptx` + `PtxIo`).
//! Expected values are hand-derived, not recomputed through the kernels.

use crate::arena::addr;
use crate::dtype::{Dtype, Ty};
use crate::oplib::{resolve_ptx, OpError, OpErrorKind, PtxIo};
use crate::program::OpKey;
use numsim_types::{WarpMask, WarpValue, WARP_SIZE};

const LANE: usize = 5;

/// Run `name` on lane `LANE` only; returns the first destination's slots.
fn run(name: &str, mods: &[&str], dsts: &[Ty], srcs: &[(Ty, u128)]) -> Result<Vec<u64>, OpError> {
    let key = OpKey {
        name: name.into(),
        mods: mods.iter().map(|m| m.to_string()).collect(),
    };
    let src_tys: Vec<Ty> = srcs.iter().map(|s| s.0).collect();
    let f = resolve_ptx(&key, dsts, &src_tys)?;
    let mut values: Vec<WarpValue<u64>> = Vec::new();
    for &(ty, v) in srcs {
        for slot in 0..ty.slots() {
            values.push([(v >> (64 * slot)) as u64; WARP_SIZE]);
        }
    }
    let slots: usize = dsts.iter().map(|t| t.slots() as usize).sum();
    let mut out = vec![[0x5555_5555_5555_5555u64; WARP_SIZE]; slots];
    let mut io = PtxIo {
        dsts: &mut out,
        dst_tys: dsts,
        srcs: &values,
        src_tys: &src_tys,
        mask: WarpMask::lane(LANE),
    };
    f.call(&mut io)?;
    assert!(
        out.iter().all(|s| s[LANE + 1] == 0x5555_5555_5555_5555),
        "inactive lane written"
    );
    Ok(out
        .iter()
        .take(dsts[0].slots() as usize)
        .map(|s| s[LANE])
        .collect())
}

fn one(name: &str, dst: Ty, srcs: &[(Ty, u128)]) -> u64 {
    run(name, &[], &[dst], srcs).unwrap_or_else(|e| panic!("{name}: {e}"))[0]
}

fn f(v: f32) -> (Ty, u128) {
    (Ty::F32, u128::from(v.to_bits()))
}

fn u32v(v: u32) -> (Ty, u128) {
    (Ty::U32, u128::from(v))
}

fn u64v(v: u64) -> (Ty, u128) {
    (Ty::U64, u128::from(v))
}

fn kind(r: Result<Vec<u64>, OpError>) -> Option<OpErrorKind> {
    r.err().map(|e| e.kind)
}

#[test]
fn float2_moves() {
    assert_eq!(
        one("tirx.cuda.make_float2", Ty::U64, &[f(1.0), f(2.0)]),
        0x4000_0000_3f80_0000
    );
    assert_eq!(
        one(
            "tirx.cuda.float2_x",
            Ty::F32,
            &[u64v(0x4000_0000_3f80_0000)]
        ),
        0x3f80_0000
    );
    assert_eq!(
        one(
            "tirx.cuda.float2_y",
            Ty::F32,
            &[u64v(0x4000_0000_3f80_0000)]
        ),
        0x4000_0000
    );
    assert_eq!(
        one("tirx.cuda.float_as_uint", Ty::U32, &[f(1.5)]),
        0x3fc0_0000
    );
    assert_eq!(
        one("tirx.cuda.uint_as_float", Ty::F32, &[u32v(0x7fc0_0001)]),
        0x7fc0_0001
    );
    // Wrong carrier widths fail closed.
    assert_eq!(
        kind(run(
            "tirx.cuda.make_float2",
            &[],
            &[Ty::U32],
            &[f(1.0), f(2.0)]
        )),
        Some(OpErrorKind::Unsupported)
    );
}

#[test]
fn ffs_u32() {
    assert_eq!(one("tirx.cuda.ffs_u32", Ty::S32, &[u32v(0)]), 0);
    assert_eq!(one("tirx.cuda.ffs_u32", Ty::S32, &[u32v(0b1000)]), 4);
    assert_eq!(one("tirx.cuda.ffs_u32", Ty::S32, &[u32v(0x8000_0000)]), 32);
}

#[test]
fn bf16x2_pack_unpack() {
    assert_eq!(
        one("tirx.cuda.float22bfloat162_rn", Ty::U32, &[f(1.0), f(-2.0)]),
        0xc000_3f80
    );
    // Ties to even on the dropped half; NaN -> 0x7fff.
    let tie = f32::from_bits(0x3f80_8000);
    let up = f32::from_bits(0x3f81_8000);
    assert_eq!(
        one("tirx.cuda.float22bfloat162_rn", Ty::U32, &[f(tie), f(up)]),
        0x3f82_3f80
    );
    assert_eq!(
        one(
            "tirx.cuda.float22bfloat162_rn",
            Ty::U32,
            &[f(f32::NAN), f(0.0)]
        ),
        0x0000_7fff
    );
    assert_eq!(
        one(
            "tirx.cuda.float22bfloat162_rn_from_float2",
            Ty::U32,
            &[u64v(0xc000_0000_3f80_0000)]
        ),
        0xc000_3f80
    );
    assert_eq!(
        one("tirx.cuda.bfloat1622float2", Ty::U64, &[u32v(0xc000_3f80)]),
        0xc000_0000_3f80_0000
    );
}

#[test]
fn hmin2_hmax2_bf16x2() {
    let a = u32v(0x4000_3f80); // hi 2.0, lo 1.0
    let b = u32v(0x3f80_c000); // hi 1.0, lo -2.0
    assert_eq!(one("tirx.cuda.hmin2", Ty::U32, &[a, b]), 0x3f80_c000);
    assert_eq!(one("tirx.cuda.hmax2", Ty::U32, &[a, b]), 0x4000_3f80);
    // float16x2 carriers select the f16x2 form: hi 2.0 / lo 1.0 vs hi 1.0 / lo -2.0.
    let a = (Ty::F16X2, 0x4000_3c00);
    let b = (Ty::F16X2, 0x3c00_c000);
    assert_eq!(one("tirx.cuda.hmin2", Ty::U32, &[a, b]), 0x3c00_c000);
    assert_eq!(one("tirx.cuda.hmax2", Ty::U32, &[a, b]), 0x4000_3c00);
}

#[test]
fn f32x2_arith_and_fdividef() {
    let x = u64v(0x4000_0000_3fc0_0000); // (1.5, 2.0)
    let y = u64v(0xc040_0000_4000_0000); // (2.0, -3.0)
    assert_eq!(
        one("tirx.cuda.fmul2_rn", Ty::U64, &[x, y]),
        0xc0c0_0000_4040_0000
    ); // (3, -6)
    assert_eq!(
        one("tirx.cuda.fadd2_rn", Ty::U64, &[x, y]),
        0xbf80_0000_4060_0000
    ); // (3.5, -1)
    assert_eq!(
        one("tirx.cuda.fdividef", Ty::F32, &[f(1.0), f(3.0)]),
        0x3eaa_aaab
    );
}

#[test]
fn fp8x4_e4m3_from_float4() {
    assert_eq!(
        one(
            "tirx.cuda.fp8x4_e4m3_from_float4",
            Ty::U32,
            &[f(1.0), f(-2.0), f(448.0), f(0.5)]
        ),
        0x307e_c038
    );
}

#[test]
fn half_and_bf16_to_float() {
    assert_eq!(
        one("tirx.cuda.half2float", Ty::F32, &[(Ty::F16, 0x3c00)]),
        0x3f80_0000
    );
    assert_eq!(
        one("tirx.cuda.half2float", Ty::F32, &[(Ty::F16, 0x7e01)]),
        0x7fff_ffff
    );
    assert_eq!(
        one("tirx.cuda.bfloat162float", Ty::F32, &[(Ty::BF16, 0x3fc0)]),
        0x3fc0_0000
    );
    assert_eq!(
        one("tirx.cuda.bfloat162float", Ty::F32, &[(Ty::BF16, 0xffc1)]),
        0x7fff_ffff
    );
}

#[test]
fn clock64_is_zero() {
    assert_eq!(one("tirx.cuda.clock64", Ty::U64, &[]), 0);
}

#[test]
fn tmem_address_and_runtime_descriptor() {
    assert_eq!(
        one(
            "tirx.cuda.get_tmem_addr",
            Ty::U32,
            &[u32v(0x0010_0020), u32v(0x10), u32v(8)]
        ),
        0x0020_0028
    );
    let minus_one = (Ty::S32, 0xffff_ffff);
    assert_eq!(
        one(
            "tirx.cuda.get_tmem_addr",
            Ty::U32,
            &[u32v(0x0001_0005), minus_one, u32v(0)]
        ),
        0x0000_0005
    );
    assert_eq!(
        one(
            "tirx.cuda.runtime_instr_desc",
            Ty::U32,
            &[u32v(0xa5a5_5a5a), u32v(3)]
        ),
        0xe5a5_5a7a
    );
}

#[test]
fn matrix_descriptor() {
    let generic = u128::from(addr::generic_from_shared(0x1230));
    let args = |a: (Ty, u128)| [a, (Ty::S32, 1), (Ty::S32, 64), (Ty::S32, 3)];
    let expected = 0x4000_4040_0001_0123;
    assert_eq!(
        one(
            "tirx.cuda.tcgen05_encode_matrix_descriptor",
            Ty::U64,
            &args((Ty::U64, generic))
        ),
        expected
    );
    assert_eq!(
        one(
            "tirx.cuda.tcgen05_encode_matrix_descriptor",
            Ty::U64,
            &args(u32v(0x1230))
        ),
        expected
    );
    // Negative swizzle: zero layout type.
    let neg = [
        (Ty::U32, 0x1230),
        (Ty::S32, 1),
        (Ty::S32, 64),
        (Ty::S32, 0xffff_ffff),
    ];
    assert_eq!(
        one("tirx.cuda.tcgen05_encode_matrix_descriptor", Ty::U64, &neg),
        expected & !(7 << 61)
    );
    let global = [
        (Ty::U64, u128::from(addr::GLOBAL_VA_BASE)),
        (Ty::S32, 1),
        (Ty::S32, 64),
        (Ty::S32, 3),
    ];
    let r = run(
        "tirx.cuda.tcgen05_encode_matrix_descriptor",
        &[],
        &[Ty::U64],
        &global,
    );
    assert_eq!(kind(r), Some(OpErrorKind::Invalid));
}

#[test]
fn instr_descriptors() {
    let i = |v: u128| (Ty::S32, v);
    let b = |v: u128| (Ty::PRED, v);
    // f32 <- bf16 x bf16, M128 N256 K16, cta_group 1: c_format 1<<4, a/b format 1<<7 / 1<<10,
    // N>>3 = 32 at bit 17, M>>4 = 8 at bit 24.
    let dense = [
        i(128),
        i(256),
        i(16),
        b(0),
        b(0),
        i(1),
        b(0),
        b(0),
        b(0),
        b(0),
    ];
    for mods in [
        &["float32", "bfloat16", "bfloat16"][..],
        &["arg3=bfloat16", "arg1=float32", "arg2=bfloat16"][..],
    ] {
        let got = run(
            "tirx.cuda.tcgen05_encode_instr_descriptor",
            mods,
            &[Ty::U32],
            &dense,
        )
        .unwrap();
        assert_eq!(got[0], 0x0840_0490);
    }
    // trans_b on bf16 sets bit 16.
    let trans = [
        i(128),
        i(256),
        i(16),
        b(0),
        b(1),
        i(1),
        b(0),
        b(0),
        b(0),
        b(0),
    ];
    let got = run(
        "tirx.cuda.tcgen05_encode_instr_descriptor",
        &["float32", "bfloat16", "bfloat16"],
        &[Ty::U32],
        &trans,
    );
    assert_eq!(got.unwrap()[0], 0x0841_0490);
    // Bad shape: an operand-value error.
    let bad = [
        i(128),
        i(250),
        i(16),
        b(0),
        b(0),
        i(1),
        b(0),
        b(0),
        b(0),
        b(0),
    ];
    let r = run(
        "tirx.cuda.tcgen05_encode_instr_descriptor",
        &["float32", "bfloat16", "bfloat16"],
        &[Ty::U32],
        &bad,
    );
    assert!(r.is_err());
    // Missing dtype string: fail closed at resolve.
    let r = run(
        "tirx.cuda.tcgen05_encode_instr_descriptor",
        &["float32", "bfloat16"],
        &[Ty::U32],
        &dense,
    );
    assert_eq!(kind(r), Some(OpErrorKind::Unsupported));

    // Block-scaled mxf8f6f4: e4m3 x e4m3 with ue8m0 scales, M128 N256 K32 ->
    // a/b format 0, scale_format 1 at bit 23, N>>3 at 17, M>>4 at 24.
    let scaled = [
        u32v(0),
        u32v(0),
        i(128),
        i(256),
        i(32),
        b(0),
        b(0),
        i(1),
        b(0),
        b(0),
        b(0),
    ];
    let mods = [
        "float32",
        "float8_e4m3fn",
        "float8_e4m3fn",
        "float8_e8m0fnu",
        "float8_e8m0fnu",
    ];
    let got = run(
        "tirx.cuda.tcgen05_encode_instr_descriptor_block_scaled",
        &mods,
        &[Ty::U32],
        &scaled,
    )
    .unwrap();
    let expected = numsim_oplib::tcgen05::encode::encode_block_scaled_instr_descriptor_fields(
        mods[0], mods[1], mods[2], mods[3], mods[4], 128, 256, 32, false, false, 1, false, false,
        false,
    )
    .unwrap();
    assert_eq!(got[0], expected as u64);
    assert_eq!(got[0] & 0xff7e_0000, (8 << 24) | (32 << 17));
}

#[test]
fn fma_single_rounding() {
    let a = f32::from_bits(0x3f80_0800); // 1 + 2^-12
    let c = -f32::from_bits(0x3f80_1000); // -(1 + 2^-11)
    assert_eq!(one("tirx.fma", Ty::F32, &[f(a), f(a), f(c)]), 0x3380_0000); // 2^-24, not 0
    let a = f64::from_bits(0x3ff0_0000_0000_0001); // 1 + 2^-52
    let c = -f64::from_bits(0x3ff0_0000_0000_0002); // -(1 + 2^-51)
    let d = (Ty::F64, u128::from(a.to_bits()));
    assert_eq!(
        one(
            "tirx.fma",
            Ty::F64,
            &[d, d, (Ty::F64, u128::from(c.to_bits()))]
        ),
        0x3970_0000_0000_0000
    ); // 2^-104
}

#[test]
fn reinterpret_moves_bits_between_equal_width_carriers() {
    assert_eq!(
        one("tirx.reinterpret", Ty::F32, &[u32v(0xbf80_0000)]),
        0xbf80_0000
    );
    let wide = 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210_u128;
    let got = run(
        "tirx.reinterpret",
        &[],
        &[Ty::B128],
        &[(Ty::vector(Dtype::U32, 4), wide)],
    )
    .unwrap();
    assert_eq!(got, vec![wide as u64, (wide >> 64) as u64]);
    let r = run("tirx.reinterpret", &[], &[Ty::U64], &[u32v(1)]);
    assert_eq!(kind(r), Some(OpErrorKind::Unsupported));
}

#[test]
fn memory_and_encoding_dependent_helpers_fail_closed() {
    for name in [
        "tirx.cuda.float22half2",
        "tirx.cuda.float8tohalf8",
        "tirx.cuda.half8tofloat8",
    ] {
        let r = run(name, &[], &[Ty::U64], &[u64v(0), u64v(0)]);
        assert_eq!(kind(r), Some(OpErrorKind::Unsupported), "{name}");
    }
    let r = run(
        "tirx.cuda.sm100_2sm_leader_smem_addr",
        &[],
        &[Ty::U32],
        &[u32v(0)],
    );
    assert_eq!(kind(r), Some(OpErrorKind::Unsupported));
    // Modifiers on a no-string helper.
    assert_eq!(
        kind(run(
            "tirx.cuda.float_as_uint",
            &["rn"],
            &[Ty::U32],
            &[f(1.0)]
        )),
        Some(OpErrorKind::Unsupported)
    );
}
