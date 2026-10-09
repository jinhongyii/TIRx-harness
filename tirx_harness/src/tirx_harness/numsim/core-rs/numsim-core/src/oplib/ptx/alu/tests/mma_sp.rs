//! Bit-exact `mma.sp{::ordered_metadata}.sync.aligned.m16n8k*.row.col`
//! through `resolve_ptx` (resolver: `ptx/warp.rs::resolve_mma_sp`).
//!
//! The expected D comes from an independent dense reference: compressed A
//! plus 2:4 metadata are expanded into a dense 16xK matrix and multiplied
//! with dense B in exact integer arithmetic (all data are small integers, so
//! every f16/bf16/e4m3 operand and f16/f32 accumulator value is exact).
//! Fragment layouts are written here from the PTX ISA tables (sparse A,
//! dense B, C/D, metadata selector geometry), not taken from numsim-oplib;
//! they agree with `numsim-oplib/src/mma/sparse.rs` (`sparse_a_owner`,
//! `sparse_metadata_location`) and its `sparse_f16_selects_the_metadata_rows_of_b`
//! case, which is reproduced below through the resolver.

use super::*;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Elem {
    F16,
    Bf16,
    E4m3,
    S8,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Acc {
    F32,
    F16,
    S32,
}

#[derive(Clone, Copy, Debug)]
struct Case {
    name: &'static str,
    k: usize,
    elem: Elem,
    acc: Acc,
    sat: bool,
    ordered: bool,
    selector: u32,
    seed: i64,
    /// Added to every C value (C stays in i32; the sums cross the i32 range).
    c_bias: i64,
}

impl Elem {
    fn bits(self) -> usize {
        if matches!(self, Elem::F16 | Elem::Bf16) {
            16
        } else {
            8
        }
    }
    fn token(self) -> &'static str {
        match self {
            Elem::F16 => "f16",
            Elem::Bf16 => "bf16",
            Elem::E4m3 => "e4m3",
            Elem::S8 => "s8",
        }
    }
    /// Encoding of a small integer (|v| <= 15).
    fn encode(self, v: i64) -> u32 {
        match self {
            Elem::F16 => f16_bits(v),
            Elem::Bf16 => (((v as f32).to_bits()) >> 16) & 0xffff,
            Elem::E4m3 => {
                if v == 0 {
                    return 0;
                }
                let sign = if v < 0 { 0x80 } else { 0 };
                let mag = v.unsigned_abs() as u32;
                let e = 31 - mag.leading_zeros(); // floor(log2)
                let mant = (mag - (1 << e)) << (3 - e);
                sign | ((e + 7) << 3) | mant
            }
            Elem::S8 => u32::from(v as i8 as u8),
        }
    }
}

/// IEEE binary16 bits of a small integer.
fn f16_bits(v: i64) -> u32 {
    if v == 0 {
        return 0;
    }
    let sign = if v < 0 { 0x8000 } else { 0 };
    let mag = v.unsigned_abs() as u32;
    let e = 31 - mag.leading_zeros();
    let mant = (mag - (1 << e)) << (10 - e);
    sign | ((e + 15) << 10) | mant
}

const PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

/// Write `value` (`bits` wide) as element `index` of `lane`'s fragment.
fn place(regs: &mut [WarpValue<u64>], lane: usize, index: usize, bits: usize, value: u32) {
    let per = 32 / bits;
    let shift = (index % per) * bits;
    regs[index / per][lane] |= u64::from(value) << shift;
}

/// Sparse A `(lane, element index)` of compressed column `cc` in `row`
/// (PTX "sparse mma .m16n8k16/.m16n8k32 A" tables).
fn a_owner(elem: Elem, row: usize, cc: usize) -> (usize, usize) {
    let (group, rh) = (row % 8, row / 8);
    if elem.bits() == 16 {
        (4 * group + (cc % 8) / 2, 4 * (cc / 8) + 2 * rh + cc % 2)
    } else {
        (4 * group + (cc % 16) / 4, 8 * (cc / 16) + 4 * rh + cc % 4)
    }
}

/// Dense B `(lane, element index)` of `(k, n)`.
fn b_owner(elem: Elem, k: usize, n: usize) -> (usize, usize) {
    if elem.bits() == 16 {
        (4 * n + (k % 8) / 2, 2 * (k / 8) + k % 2)
    } else {
        (4 * n + (k % 16) / 4, 4 * (k / 16) + k % 4)
    }
}

/// C/D `(lane, element index)` of `(row, col)` (4 elements per lane).
fn cd_owner(row: usize, col: usize) -> (usize, usize) {
    (4 * (row % 8) + col / 2, 2 * (row / 8) + col % 2)
}

/// Metadata `(lane, nibble)` of `(row, chunk)` for the selector geometry.
fn meta_owner(chunks: usize, selector: usize, row: usize, chunk: usize) -> (usize, usize) {
    let (group, rh) = (row % 8, row / 8);
    match chunks {
        4 => (4 * group + selector, 4 * rh + chunk),
        8 => (4 * group + 2 * selector + chunk / 4, 4 * rh + chunk % 4),
        _ => (4 * group + chunk / 4, 4 * rh + chunk % 4),
    }
}

fn mods(case: &Case) -> Vec<&'static str> {
    let shape = match case.k {
        16 => "m16n8k16",
        32 => "m16n8k32",
        _ => "m16n8k64",
    };
    let (dtype, ctype) = match case.acc {
        Acc::F32 => ("f32", "f32"),
        Acc::F16 => ("f16", "f16"),
        Acc::S32 => ("s32", "s32"),
    };
    let mut out = vec![
        if case.ordered {
            "sp::ordered_metadata"
        } else {
            "sp"
        },
        "sync",
        "aligned",
        shape,
        "row",
        "col",
    ];
    if case.sat {
        out.push("satfinite");
    }
    out.extend([dtype, case.elem.token(), case.elem.token(), ctype]);
    out
}

struct Built {
    srcs: Vec<WarpValue<u64>>,
    expected: Vec<WarpValue<u64>>,
    nd: usize,
    /// D elements whose exact sum is outside i32.
    overflowed: usize,
}

#[allow(clippy::needless_range_loop)]
fn build(case: &Case) -> Built {
    let k = case.k;
    let elem = case.elem;
    let chunks = k / 4;
    let ck = k / 2;
    let seed = case.seed;
    let small = |x: i64, m: i64| x.rem_euclid(m) - m / 2;
    // Compressed A values, metadata codes, dense expansion.
    let mut dense_a = vec![vec![0i64; k]; 16];
    let mut a_regs = vec![[0u64; 32]; 16 * ck * elem.bits() / 1024];
    let mut meta = [0xffff_ffffu64; 32];
    let mut meta_written = [false; 32];
    for row in 0..16 {
        for chunk in 0..chunks {
            let (mut first, mut second) =
                PAIRS[(row * 5 + chunk * 3 + seed as usize) % PAIRS.len()];
            if !case.ordered && (row + chunk) % 2 == 1 {
                std::mem::swap(&mut first, &mut second);
            }
            let (lane, nibble) = meta_owner(chunks, case.selector as usize, row, chunk);
            if !meta_written[lane] {
                meta[lane] = 0;
                meta_written[lane] = true;
            }
            meta[lane] |= ((first | (second << 2)) as u64) << (4 * nibble);
            for (packed, pos) in [first, second].into_iter().enumerate() {
                let cc = 2 * chunk + packed;
                let v = small(row as i64 * 7 + cc as i64 * 3 + seed, 7);
                dense_a[row][4 * chunk + pos] = v;
                let (lane, index) = a_owner(elem, row, cc);
                place(&mut a_regs, lane, index, elem.bits(), elem.encode(v));
            }
        }
    }
    // Lanes of the metadata register that no (row, chunk) reads must not
    // matter: leave them as an invalid all-ones code.
    let mut b_regs = vec![[0u64; 32]; 8 * k * elem.bits() / 1024];
    let mut dense_b = vec![vec![0i64; 8]; k];
    for (inner, row) in dense_b.iter_mut().enumerate() {
        for (n, slot) in row.iter_mut().enumerate() {
            let v = small(inner as i64 * 5 + n as i64 * 3 + seed, 7);
            *slot = v;
            let (lane, index) = b_owner(elem, inner, n);
            place(&mut b_regs, lane, index, elem.bits(), elem.encode(v));
        }
    }
    let nc = if case.acc == Acc::F16 { 2 } else { 4 };
    let mut c_regs = vec![[0u64; 32]; nc];
    let mut d_regs = vec![[0u64; 32]; nc];
    let mut overflowed = 0;
    for row in 0..16 {
        for col in 0..8 {
            let c = small(row as i64 + 2 * col as i64 + seed, 17) + case.c_bias;
            let sum: i64 = (0..k)
                .map(|i| dense_a[row][i] * dense_b[i][col])
                .sum::<i64>()
                + c;
            overflowed += usize::from(i32::try_from(sum).is_err());
            let (lane, index) = cd_owner(row, col);
            let (cbits, dbits) = match case.acc {
                Acc::F32 => ((c as f32).to_bits(), (sum as f32).to_bits()),
                Acc::F16 => (f16_bits(c), f16_bits(sum)),
                Acc::S32 => {
                    let d = if case.sat {
                        sum.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
                    } else {
                        sum as i32
                    };
                    (c as i32 as u32, d as u32)
                }
            };
            let bits = if case.acc == Acc::F16 { 16 } else { 32 };
            place(&mut c_regs, lane, index, bits, cbits);
            place(&mut d_regs, lane, index, bits, dbits);
        }
    }
    let mut srcs = a_regs;
    srcs.extend(b_regs);
    srcs.extend(c_regs);
    srcs.push(meta);
    srcs.push([u64::from(case.selector); 32]);
    Built {
        srcs,
        expected: d_regs,
        nd: nc,
        overflowed,
    }
}

fn execute(case: &Case, srcs: &[WarpValue<u64>], nd: usize) -> OpResult<Vec<WarpValue<u64>>> {
    let dst_tys = vec![U32; nd];
    let src_tys = vec![U32; srcs.len()];
    let f = resolve_ptx(&key(case.name, &mods(case)), &dst_tys, &src_tys)?;
    let mut dsts = vec![[POISON; 32]; nd];
    let mut io = PtxIo {
        dsts: &mut dsts,
        dst_tys: &dst_tys,
        srcs,
        src_tys: &src_tys,
        mask: WarpMask::ALL,
    };
    f.call(&mut io)?;
    Ok(dsts)
}

fn check(case: Case) -> usize {
    let built = build(&case);
    let got = execute(&case, &built.srcs, built.nd).unwrap_or_else(|e| panic!("{case:?}: {e}"));
    for (reg, (g, e)) in got.iter().zip(&built.expected).enumerate() {
        for lane in 0..32 {
            assert_eq!(g[lane], e[lane], "{case:?}: D reg {reg} lane {lane}");
        }
    }
    built.overflowed
}

const BASE: Case = Case {
    name: "mma_sp",
    k: 16,
    elem: Elem::F16,
    acc: Acc::F32,
    sat: false,
    ordered: true,
    selector: 0,
    seed: 0,
    c_bias: 0,
};

#[test]
fn mma_sp_f16_bf16_f32_accumulator() {
    for elem in [Elem::F16, Elem::Bf16] {
        for ordered in [true, false] {
            // m16n8k16: one metadata thread per group, selector 0..=3.
            for selector in [0, 3] {
                check(Case {
                    elem,
                    ordered,
                    selector,
                    seed: 1 + i64::from(selector),
                    ..BASE
                });
            }
            // m16n8k32 (`_pair`): metadata thread pair, selector 0 or 1.
            for selector in [0, 1] {
                check(Case {
                    name: "mma_sp_pair",
                    k: 32,
                    elem,
                    ordered,
                    selector,
                    seed: 5 + i64::from(selector),
                    ..BASE
                });
            }
        }
    }
}

#[test]
fn mma_sp_f16_accumulator() {
    for selector in [1, 2] {
        check(Case {
            name: "mma_sp_f16acc",
            acc: Acc::F16,
            selector,
            seed: 3,
            ..BASE
        });
    }
    for selector in [0, 1] {
        check(Case {
            name: "mma_sp_f16acc_pair",
            k: 32,
            acc: Acc::F16,
            selector,
            seed: 4,
            ..BASE
        });
    }
}

#[test]
fn mma_sp_all_e4m3() {
    for ordered in [true, false] {
        check(Case {
            name: "mma_sp_all",
            k: 64,
            elem: Elem::E4m3,
            ordered,
            seed: 2,
            ..BASE
        });
    }
}

#[test]
fn mma_sp_int_s8_wrap_and_satfinite() {
    let int = Case {
        elem: Elem::S8,
        acc: Acc::S32,
        ..BASE
    };
    for selector in [0, 1] {
        for sat in [false, true] {
            check(Case {
                name: "mma_sp_int_pair",
                k: 32,
                sat,
                selector,
                seed: 6,
                ..int
            });
        }
    }
    for sat in [false, true] {
        check(Case {
            name: "mma_sp_int_all",
            k: 64,
            sat,
            seed: 7,
            ..int
        });
        // Sums past i32::MAX: wrap without, clamp with `.satfinite`.
        let high = Case {
            name: "mma_sp_int_all",
            k: 64,
            sat,
            seed: 7,
            c_bias: i64::from(i32::MAX) - 8,
            ..int
        };
        let low = Case {
            name: "mma_sp_int_pair",
            k: 32,
            sat,
            selector: 1,
            seed: 8,
            c_bias: i64::from(i32::MIN) + 8,
            ..int
        };
        assert!(
            check(high) > 0 && check(low) > 0,
            "saturation cases must overflow"
        );
    }
}

#[test]
fn mma_sp_matches_the_oplib_metadata_row_case() {
    // numsim-oplib `sparse_f16_selects_the_metadata_rows_of_b`: row 0 stores
    // (1, 2) in chunk 0 at dense positions 1 and 3 (code 0b1101 everywhere),
    // B[inner, 0] = inner + 1, so D[0,0] = 1*2 + 2*4 = 10 and D[1,0] = 0.
    let mut a = vec![[0u64; 32]; 2];
    for (packed, v) in [(0, 1), (1, 2)] {
        let (lane, index) = a_owner(Elem::F16, 0, packed);
        place(&mut a, lane, index, 16, f16_bits(v));
    }
    let mut b = vec![[0u64; 32]; 2];
    for inner in 0..16 {
        let (lane, index) = b_owner(Elem::F16, inner, 0);
        place(&mut b, lane, index, 16, f16_bits(inner as i64 + 1));
    }
    let mut srcs = a;
    srcs.extend(b);
    srcs.extend(vec![[0u64; 32]; 4]);
    srcs.push([0xdddd_dddd; 32]);
    srcs.push([0; 32]);
    let d = execute(&BASE, &srcs, 4).unwrap();
    let (lane, index) = cd_owner(0, 0);
    assert_eq!(d[index][lane], u64::from(10.0_f32.to_bits()));
    let (lane, index) = cd_owner(1, 0);
    assert_eq!(d[index][lane], 0);
}

#[test]
fn mma_sp_rejects_bad_metadata_and_selectors() {
    let built = build(&BASE);
    // Ordered metadata with a descending code is an operand error.
    let mut srcs = built.srcs.clone();
    let meta = srcs.len() - 2;
    srcs[meta] = [0x6666_6669; 32]; // nibble 0x9 = (1, 2) ok; 0x6 = (2, 1) descending
    assert_eq!(
        execute(&BASE, &srcs, 4).unwrap_err().kind,
        OpErrorKind::Invalid
    );
    // A repeated position is invalid in either variant.
    srcs[meta] = [0x5555_5555; 32];
    let plain = Case {
        ordered: false,
        ..BASE
    };
    assert_eq!(
        execute(&plain, &srcs, 4).unwrap_err().kind,
        OpErrorKind::Invalid
    );
    // Selector outside the geometry (k16: 0..=3; pair: 0..=1).
    let mut srcs = built.srcs.clone();
    let sel = srcs.len() - 1;
    srcs[sel] = [4; 32];
    assert_eq!(
        execute(&BASE, &srcs, 4).unwrap_err().kind,
        OpErrorKind::Invalid
    );
    // Wrong operand count fails closed at resolve time.
    assert_eq!(
        execute(&BASE, &built.srcs[1..], 4).unwrap_err().kind,
        OpErrorKind::Unsupported
    );
    // bf16 with an f16 accumulator has no legacy variant.
    let bad = Case {
        elem: Elem::Bf16,
        acc: Acc::F16,
        ..BASE
    };
    let built = build(&Case {
        elem: Elem::Bf16,
        ..BASE
    });
    let s = &built.srcs; // A(2) B(2) C(4) e f
    let srcs: Vec<WarpValue<u64>> = s[..4]
        .iter()
        .chain(&s[4..6])
        .chain(&s[8..])
        .copied()
        .collect();
    assert_eq!(
        execute(&bad, &srcs, 2).unwrap_err().kind,
        OpErrorKind::Invalid
    );
}
