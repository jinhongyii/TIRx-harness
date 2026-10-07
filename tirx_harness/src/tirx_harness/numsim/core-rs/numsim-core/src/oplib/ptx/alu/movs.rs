//! Moves and vector packs, `createpolicy`, and register sparse
//! (de)compression (legacy `move_variant!`, `mov_pack`/`mov_unpack`
//! variants, `createpolicy_spec`, `SpCompress`, `SpDecompress`).

use super::{flat, int_type, lanes, map, mask, try_map, Md};
use crate::oplib::ptx::{Operands, Resolved};
use crate::oplib::{OpError, OpResult, PtxIo};
use numsim_oplib::arith::{self, SpCompressShape, SpDecompressShape};
use numsim_oplib::cvt;

pub(super) fn resolve(op: &str, m: &Md, ops: &Operands) -> OpResult<Resolved> {
    match op {
        "mov" => {
            let ty = m.get("type");
            if ty == "pred" {
                return map::<1, _>(m, ops, 1, false, |[a]| u64::from(a != 0));
            }
            let (bits, signed) = match ty {
                "f32" => (32, false),
                "f64" => (64, false),
                other => int_type(m, other)?,
            };
            if !signed && bits >= 32 && flat(ops, 1, 1, bits) {
                return Ok(Resolved::Direct(if bits == 32 {
                    mov_direct::<32>
                } else {
                    mov_direct::<64>
                }));
            }
            let all = mask(bits);
            map::<1, _>(m, ops, bits, signed, move |[a]| a & all)
        }
        "mov_pack_b16x2" => map::<2, _>(m, ops, 32, false, |[a, b]| {
            u64::from(arith::mov_pack_b32(a as u16, b as u16))
        }),
        "mov_pack_b32x2" => {
            if flat(ops, 1, 2, 64) {
                return Ok(Resolved::Direct(mov_pack_b32x2_direct));
            }
            map::<2, _>(m, ops, 64, false, |[a, b]| {
                arith::mov_pack_b64(a as u32, b as u32)
            })
        }
        "mov_pack_b16x4" => map::<4, _>(m, ops, 64, false, |[a, b, c, d]| {
            cvt::ptx_mov_pack_b16x4([a as u16, b as u16, c as u16, d as u16])
        }),
        "mov_unpack_b16x2" => lanes::<1, 2, _>(m, ops, [(16, false); 2], |[a]| {
            let (lo, hi) = arith::mov_unpack_b32(a as u32);
            Ok([u64::from(lo), u64::from(hi)])
        }),
        "mov_unpack_b32x2" => {
            if flat(ops, 2, 1, 32) {
                return Ok(Resolved::Direct(mov_unpack_b32x2_direct));
            }
            lanes::<1, 2, _>(m, ops, [(32, false); 2], |[a]| {
                let (lo, hi) = arith::mov_unpack_b64(a);
                Ok([u64::from(lo), u64::from(hi)])
            })
        }
        "mov_unpack_b16x4" => lanes::<1, 4, _>(m, ops, [(16, false); 4], |[a]| {
            Ok(cvt::ptx_mov_unpack_b16x4(a).map(u64::from))
        }),
        "mov_pack_b32x4" | "mov_pack_b64x2" => {
            let n = if op == "mov_pack_b32x4" { 4 } else { 2 };
            ops.arity(1, n, m.op)?;
            let ops = ops.clone();
            let b32 = n == 4;
            Ok(Resolved::Boxed(Box::new(move |io: &mut PtxIo<'_>| {
                for lane in io.mask.lanes() {
                    let words = if b32 {
                        cvt::ptx_mov_pack_b32x4(std::array::from_fn(|i| {
                            ops.src(io, i, lane) as u32
                        }))
                    } else {
                        arith::mov_pack_b128(ops.src(io, 0, lane), ops.src(io, 1, lane))
                    };
                    ops.put128(
                        io,
                        0,
                        lane,
                        u128::from(words[0]) | (u128::from(words[1]) << 64),
                    );
                }
                Ok(())
            })))
        }
        "mov_unpack_b32x4" | "mov_unpack_b64x2" => {
            let n = if op == "mov_unpack_b32x4" { 4 } else { 2 };
            ops.arity(n, 1, m.op)?;
            let ops = ops.clone();
            Ok(Resolved::Boxed(Box::new(move |io: &mut PtxIo<'_>| {
                for lane in io.mask.lanes() {
                    let value = ops.src128(io, 0, lane);
                    let words = [value as u64, (value >> 64) as u64];
                    if n == 4 {
                        for (i, word) in cvt::ptx_mov_unpack_b32x4(words).into_iter().enumerate() {
                            ops.put(io, i, lane, u64::from(word), 32, false);
                        }
                    } else {
                        for (i, word) in cvt::ptx_mov_unpack_b64x2(words).into_iter().enumerate() {
                            ops.put(io, i, lane, word, 64, false);
                        }
                    }
                }
                Ok(())
            })))
        }
        // Policy bits are opaque; zero is the private representative of every
        // valid policy (legacy `createpolicy_spec`). Inputs are validated.
        "createpolicy_fraction" => try_map::<1, _>(m, ops, 64, false, |[f]| {
            Ok(arith::createpolicy_fraction(f32::from_bits(f as u32))?)
        }),
        "createpolicy_fractional" => try_map::<0, _>(m, ops, 64, false, |[]| {
            Ok(arith::createpolicy_fraction(1.0)?)
        }),
        "createpolicy_cvt" => map::<1, _>(m, ops, 64, false, |[p]| arith::createpolicy_cvt(p)),
        "createpolicy_range" => try_map::<3, _>(m, ops, 64, false, |[_addr, primary, total]| {
            Ok(arith::createpolicy_range(primary as u32, total as u32)?)
        }),
        "spcompress" => {
            let shape = SpCompressShape {
                elem_bits: shape_field(m, "elemsize", "b")?,
                index_bits: shape_field(m, "idxsize", "b")?,
                num: shape_field(m, "num", "x")?,
            };
            shape.validate()?;
            let (inputs, outputs) = (shape.data_registers(), shape.output_registers());
            ops.arity(outputs, inputs + 1, m.op)?;
            let ops = ops.clone();
            Ok(Resolved::Boxed(Box::new(move |io: &mut PtxIo<'_>| {
                for lane in io.mask.lanes() {
                    let data: Vec<u32> = (0..inputs).map(|i| ops.src(io, i, lane) as u32).collect();
                    let words = arith::spcompress(shape, &data, ops.src(io, inputs, lane) as u32)?;
                    for (i, word) in words.into_iter().enumerate() {
                        ops.put(io, i, lane, u64::from(word), 32, false);
                    }
                }
                Ok(())
            })))
        }
        "spdecompress" => {
            let factor = m.get("spfactor");
            let (src, dst) = factor
                .strip_prefix("sp::")
                .and_then(|f| f.split_once(':'))
                .and_then(|(s, d)| Some((s.parse::<usize>().ok()?, d.parse::<usize>().ok()?)))
                .ok_or_else(|| OpError::unsupported(format!("{}: spfactor `{factor}`", m.op)))?;
            let shape = SpDecompressShape {
                elem_bits: shape_field(m, "elemsize", "b")?,
                index_bits: shape_field(m, "idxsize", "b")?,
                src,
                dst,
                num: shape_field(m, "num", "x")?,
            };
            if shape.validate().is_err() {
                return m.unsupported("spdecompress shape outside the PTX 9.4 domain");
            }
            let (meta, comp, data) = (
                shape.metadata_registers(),
                shape.compressed_registers(),
                shape.data_registers(),
            );
            ops.arity(data, meta + comp, m.op)?;
            let ops = ops.clone();
            Ok(Resolved::Boxed(Box::new(move |io: &mut PtxIo<'_>| {
                for lane in io.mask.lanes() {
                    let metadata: Vec<u32> =
                        (0..meta).map(|i| ops.src(io, i, lane) as u32).collect();
                    let compressed: Vec<u32> = (meta..meta + comp)
                        .map(|i| ops.src(io, i, lane) as u32)
                        .collect();
                    let words = arith::spdecompress(shape, &metadata, &compressed)?;
                    for (i, word) in words.into_iter().enumerate() {
                        ops.put(io, i, lane, u64::from(word), 32, false);
                    }
                }
                Ok(())
            })))
        }
        _ => m.unsupported("not an ALU op"),
    }
}

fn shape_field(m: &Md, slot: &str, prefix: &str) -> OpResult<usize> {
    let token = m.get(slot);
    match token.strip_prefix(prefix).and_then(|t| t.parse().ok()) {
        Some(value) => Ok(value),
        None => m.unsupported(format!("{slot} `{token}`")),
    }
}

/// Unsigned `mov.b{32,64}` / `.u{32,64}` / `.f{32,64}` (validated by `flat`).
fn mov_direct<const BITS: u32>(io: &mut PtxIo<'_>) -> OpResult {
    for lane in io.mask.lanes() {
        let value = io.srcs[0][lane];
        io.dsts[0][lane] = if BITS >= 64 {
            value
        } else {
            value & ((1u64 << BITS) - 1)
        };
    }
    Ok(())
}

/// `mov.b64 d, {lo, hi}` (validated by `flat`).
fn mov_pack_b32x2_direct(io: &mut PtxIo<'_>) -> OpResult {
    for lane in io.mask.lanes() {
        io.dsts[0][lane] = arith::mov_pack_b64(io.srcs[0][lane] as u32, io.srcs[1][lane] as u32);
    }
    Ok(())
}

/// `mov.b64 {lo, hi}, s` (validated by `flat`).
fn mov_unpack_b32x2_direct(io: &mut PtxIo<'_>) -> OpResult {
    for lane in io.mask.lanes() {
        let (lo, hi) = arith::mov_unpack_b64(io.srcs[0][lane]);
        io.dsts[0][lane] = u64::from(lo);
        io.dsts[1][lane] = u64::from(hi);
    }
    Ok(())
}
