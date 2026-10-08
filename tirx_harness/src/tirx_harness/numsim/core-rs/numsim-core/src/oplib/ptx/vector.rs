//! Lowering-internal vector ops (W1 lowering conventions §C.3):
//!
//! * `numsim.pack` / `numsim.unpack` with mod `ty=<Elem>x<N>` (the whole
//!   vector type, `Dtype` debug name): `pack` concatenates its source
//!   registers (element 0 / low bits first) into one vector register;
//!   `unpack` splits one vector register into its destination registers in
//!   order. Pieces may be scalars or sub-vectors; their bit widths must sum to
//!   the vector's. Pure bit moves (sub-byte elements included).
//! * `tirx.cuda.{float22half2,float8tohalf8,half8tofloat8}.value`: the pure
//!   element conversions behind the pointer helpers (lowering does
//!   `LoadAddr` + `Ptx` + `StoreAddr`), exactly as legacy
//!   `cuda_helper.rs` emitted them: f32 -> f16 with `cuda_f32_to_fp16_bits`,
//!   f16 -> f32 with `cuda_fp16_bits_to_f32` (CUDA NaN canonicalisation).

use super::{Mods, Operands, Resolved};
use crate::dtype::{Dtype, Ty};
use crate::oplib::{OpError, OpResult, PtxIo};
use numsim_oplib::scalar::{cuda_f32_to_fp16_bits, cuda_fp16_bits_to_f32};

pub(in crate::oplib) const NAMES: &[&str] = &[
    "numsim.pack",
    "numsim.unpack",
    "tirx.cuda.float22half2.value",
    "tirx.cuda.float8tohalf8.value",
    "tirx.cuda.half8tofloat8.value",
];

type Bits = [u64; 4];

fn read(ops: &Operands, io: &PtxIo<'_>, i: usize, lane: usize) -> Bits {
    let mut v = [0u64; 4];
    let off = ops.src_off[i];
    for (k, w) in v.iter_mut().enumerate().take(ops.src_tys[i].slots() as usize) {
        *w = io.srcs[off + k][lane];
    }
    mask(&mut v, ops.src_tys[i].bits());
    v
}

fn write(ops: &Operands, io: &mut PtxIo<'_>, i: usize, lane: usize, mut v: Bits) {
    mask(&mut v, ops.dst_tys[i].bits());
    let off = ops.dst_off[i];
    for (k, w) in v.iter().enumerate().take(ops.dst_tys[i].slots() as usize) {
        io.dsts[off + k][lane] = *w;
    }
}

fn mask(v: &mut Bits, bits: u32) {
    for (k, w) in v.iter_mut().enumerate() {
        let lo = k as u32 * 64;
        if bits <= lo {
            *w = 0;
        } else if bits < lo + 64 {
            *w &= (1u64 << (bits - lo)) - 1;
        }
    }
}

/// `width` bits of `v` starting at bit `at`.
fn extract(v: &Bits, at: u32, width: u32) -> Bits {
    let mut out = [0u64; 4];
    for bit in 0..width {
        let src = at + bit;
        if (v[(src / 64) as usize] >> (src % 64)) & 1 == 1 {
            out[(bit / 64) as usize] |= 1 << (bit % 64);
        }
    }
    out
}

fn insert(v: &mut Bits, at: u32, width: u32, piece: &Bits) {
    for bit in 0..width {
        if (piece[(bit / 64) as usize] >> (bit % 64)) & 1 == 1 {
            let dst = at + bit;
            v[(dst / 64) as usize] |= 1 << (dst % 64);
        }
    }
}

fn parse_ty(token: &str, name: &str) -> OpResult<Ty> {
    let bad = || OpError::unsupported(format!("{name}: bad vector type `{token}`"));
    let (elem, lanes) = token.rsplit_once('x').ok_or_else(bad)?;
    let lanes: u8 = lanes.parse().map_err(|_| bad())?;
    let elem = Dtype::ALL_FOR_PACK.iter().copied().find(|d| format!("{d:?}") == elem).ok_or_else(bad)?;
    Ok(Ty::vector(elem, lanes))
}

trait PackDtypes {
    const ALL_FOR_PACK: [Dtype; 32];
}

impl PackDtypes for Dtype {
    const ALL_FOR_PACK: [Dtype; 32] = {
        use Dtype::*;
        [
            Pred, U8, U16, U32, U64, S8, S16, S32, S64, B128, F16, BF16, TF32, F32, F64, E4M3, E5M2, UE8M0, UE4M3,
            UE5M3, E2M3, E3M2, S2F6, E2M1, U4, S4, U6, E3M4, E4M3Ieee, E4M3B11Fnuz, E4M3Fnuz, E5M2Fnuz,
        ]
    };
}

pub(in crate::oplib) fn resolve(name: &str, mods: &Mods, ops: &Operands) -> OpResult<Option<Resolved>> {
    let ops = ops.clone();
    Ok(Some(match name {
        "numsim.pack" | "numsim.unpack" => {
            let token = mods
                .pairs
                .iter()
                .find(|(slot, _)| slot == "ty" || slot.is_empty())
                .map(|(_, t)| t.as_str())
                .ok_or_else(|| OpError::unsupported(format!("{name}: missing `ty=<Elem>xN`")))?;
            if mods.pairs.len() != 1 {
                return Err(OpError::unsupported(format!("{name}: unexpected modifiers {:?}", mods.pairs)));
            }
            let vector = parse_ty(token, name)?;
            let pack = name == "numsim.pack";
            let (whole, pieces) = if pack { (&ops.dst_tys, &ops.src_tys) } else { (&ops.src_tys, &ops.dst_tys) };
            if whole.len() != 1 || pieces.is_empty() || whole[0].bits() != vector.bits() {
                return Err(OpError::unsupported(format!(
                    "{name}: {vector} needs one {} bit register, got {:?}",
                    vector.bits(),
                    whole
                )));
            }
            let total: u32 = pieces.iter().map(|t| t.bits()).sum();
            if total != vector.bits() || vector.bits() > 256 {
                return Err(OpError::unsupported(format!("{name}: pieces {pieces:?} do not tile {vector}")));
            }
            let widths: Vec<u32> = pieces.iter().map(|t| t.bits()).collect();
            if pack {
                Resolved::Boxed(Box::new(move |io| {
                    for lane in io.mask.lanes() {
                        let mut v = [0u64; 4];
                        let mut at = 0;
                        for (i, &w) in widths.iter().enumerate() {
                            insert(&mut v, at, w, &read(&ops, io, i, lane));
                            at += w;
                        }
                        write(&ops, io, 0, lane, v);
                    }
                    Ok(())
                }))
            } else {
                Resolved::Boxed(Box::new(move |io| {
                    for lane in io.mask.lanes() {
                        let v = read(&ops, io, 0, lane);
                        let mut at = 0;
                        for (i, &w) in widths.iter().enumerate() {
                            write(&ops, io, i, lane, extract(&v, at, w));
                            at += w;
                        }
                    }
                    Ok(())
                }))
            }
        }
        "tirx.cuda.float22half2.value" | "tirx.cuda.float8tohalf8.value" | "tirx.cuda.half8tofloat8.value" => {
            if !mods.pairs.is_empty() {
                return Err(OpError::unsupported(format!("{name}: unexpected modifiers {:?}", mods.pairs)));
            }
            let (src, dst) = match name {
                "tirx.cuda.float22half2.value" => (Ty::vector(Dtype::F32, 2), Ty::vector(Dtype::F16, 2)),
                "tirx.cuda.float8tohalf8.value" => (Ty::vector(Dtype::F32, 8), Ty::vector(Dtype::F16, 8)),
                _ => (Ty::vector(Dtype::F16, 8), Ty::vector(Dtype::F32, 8)),
            };
            if ops.src_tys.len() != 1 || ops.dst_tys.len() != 1 || ops.src_tys[0].bits() != src.bits() || ops.dst_tys[0].bits() != dst.bits() {
                return Err(OpError::unsupported(format!(
                    "{name}: expected {src} -> {dst}, got {:?} -> {:?}",
                    ops.src_tys, ops.dst_tys
                )));
            }
            let to_half = src.elem == Dtype::F32;
            let lanes = src.lanes as u32;
            Resolved::Boxed(Box::new(move |io| {
                for lane in io.mask.lanes() {
                    let v = read(&ops, io, 0, lane);
                    let mut out = [0u64; 4];
                    for e in 0..lanes {
                        if to_half {
                            let x = extract(&v, e * 32, 32)[0] as u32;
                            let h = u64::from(cuda_f32_to_fp16_bits(f32::from_bits(x)));
                            insert(&mut out, e * 16, 16, &[h, 0, 0, 0]);
                        } else {
                            let h = extract(&v, e * 16, 16)[0] as u16;
                            let f = u64::from(cuda_fp16_bits_to_f32(h).to_bits());
                            insert(&mut out, e * 32, 32, &[f, 0, 0, 0]);
                        }
                    }
                    write(&ops, io, 0, lane, out);
                }
                Ok(())
            }))
        }
        _ => return Ok(None),
    }))
}

#[cfg(test)]
mod tests {
    use crate::dtype::{Dtype, Ty};
    use crate::oplib::{resolve_ptx, OpErrorKind, PtxIo};
    use crate::program::OpKey;
    use crate::value::{WarpMask, WarpValue};

    fn key(name: &str, mods: &[&str]) -> OpKey {
        OpKey { name: name.into(), mods: mods.iter().map(|m| m.to_string()).collect() }
    }

    fn run(key: &OpKey, dst_tys: &[Ty], src_tys: &[Ty], srcs: &[WarpValue<u64>]) -> Vec<WarpValue<u64>> {
        let f = resolve_ptx(key, dst_tys, src_tys).unwrap();
        let mut dsts = vec![[0u64; 32]; dst_tys.iter().map(|t| t.slots() as usize).sum()];
        let mut io = PtxIo { dsts: &mut dsts, dst_tys, srcs, src_tys, mask: WarpMask(0b101) };
        f.call(&mut io).unwrap();
        dsts
    }

    #[test]
    fn pack_and_unpack_round_trip_vectors_and_subvectors() {
        let f32x4 = Ty::vector(Dtype::F32, 4);
        let lanes: Vec<WarpValue<u64>> = (0..4u64).map(|i| [0x4000_0000 + i; 32]).collect();
        let packed = run(&key("numsim.pack", &["ty=F32x4"]), &[f32x4], &[Ty::F32; 4], &lanes);
        assert_eq!(packed[0][0], 0x4000_0001_4000_0000);
        assert_eq!(packed[1][2], 0x4000_0003_4000_0002);
        assert_eq!(packed[0][1], 0, "inactive lanes untouched");
        // Split into two f32x2 halves, then into four scalars.
        let halves = run(&key("numsim.unpack", &["ty=F32x4"]), &[Ty::vector(Dtype::F32, 2); 2], &[f32x4], &packed);
        assert_eq!((halves[0][0], halves[1][0]), (packed[0][0], packed[1][0]));
        let scalars = run(&key("numsim.unpack", &["ty=F32x4"]), &[Ty::F32; 4], &[f32x4], &packed);
        assert_eq!(scalars.iter().map(|s| s[2]).collect::<Vec<_>>(), [0x4000_0000, 0x4000_0001, 0x4000_0002, 0x4000_0003]);
        // Broadcast of a sub-byte element: e2m1 x 4 = 16 bits.
        let fp4 = Ty::vector(Dtype::E2M1, 4);
        let b = run(&key("numsim.pack", &["ty=E2M1x4"]), &[fp4], &[Ty::scalar(Dtype::E2M1); 4], &[[0x1f; 32]; 4]);
        assert_eq!(b[0][0], 0xffff);
        // Bad tilings fail closed.
        let bad = resolve_ptx(&key("numsim.pack", &["ty=F32x4"]), &[f32x4], &[Ty::F32; 3]).unwrap_err();
        assert_eq!(bad.kind, OpErrorKind::Unsupported);
        assert!(resolve_ptx(&key("numsim.pack", &["ty=Q9x4"]), &[f32x4], &[Ty::F32; 4]).is_err());
    }

    #[test]
    fn value_converters_match_legacy_cuda_helpers() {
        // float22half2: (1.0, -inf) -> 0xfc00_3c00; NaN canonicalises to 0x7fff.
        let src = [(f32::NEG_INFINITY.to_bits() as u64) << 32 | 1.0f32.to_bits() as u64; 32];
        let out = run(&key("tirx.cuda.float22half2.value", &[]), &[Ty::F16X2], &[Ty::vector(Dtype::F32, 2)], &[src]);
        assert_eq!(out[0][0], 0xfc00_3c00);
        let nan = [f32::NAN.to_bits() as u64; 32];
        let out = run(&key("tirx.cuda.float22half2.value", &[]), &[Ty::F16X2], &[Ty::vector(Dtype::F32, 2)], &[nan]);
        assert_eq!(out[0][0] & 0xffff, 0x7fff);
        // half8tofloat8 then float8tohalf8 round-trips exact halves.
        let h: Vec<WarpValue<u64>> = vec![[0x4000_3c00_bc00_0001; 32], [0x7c00_8000_3555_c000; 32]];
        let f = run(&key("tirx.cuda.half8tofloat8.value", &[]), &[Ty::vector(Dtype::F32, 8)], &[Ty::vector(Dtype::F16, 8)], &h);
        assert_eq!(f[0][0] as u32, f32::from_bits(0x3380_0000).to_bits()); // 2^-24 subnormal
        assert_eq!((f[0][0] >> 32) as u32, (-1.0f32).to_bits());
        let back = run(&key("tirx.cuda.float8tohalf8.value", &[]), &[Ty::vector(Dtype::F16, 8)], &[Ty::vector(Dtype::F32, 8)], &f);
        assert_eq!((back[0][0], back[1][0]), (h[0][0], h[1][0]));
    }
}
