//! Pure warp-collective PTX forms resolved through `Instr::Ptx`: `shfl.sync`,
//! `vote.sync`, `redux.sync`, `match.sync`, `elect.sync`, `activemask`,
//! `movmatrix`, and dense `mma.sync` (README decision 1). They read all lanes
//! of `srcs` and write only `io.mask`; `io.mask` is the executing (active)
//! set and each lane's `membermask` source is its participant mask, exactly
//! as legacy validated them (`numsim_oplib::warp::validate_participants`).

use super::{Mods, Operands, Resolved};
use crate::oplib::{OpError, OpResult, PtxIo};
use crate::value::WarpValue;
use numsim_oplib::mma as mma;
use numsim_oplib::warp as lib;

pub(in crate::oplib) const NAMES: &[&str] = &[
    "tirx.ptx.shfl_sync",
    "tirx.ptx.shfl_sync_p",
    "tirx.ptx.vote_sync",
    "tirx.ptx.vote_sync_ballot",
    "tirx.ptx.redux_sync",
    "tirx.ptx.redux_sync_bitwise",
    "tirx.ptx.redux_sync_f32",
    "tirx.ptx.match_any_sync",
    "tirx.ptx.match_all_sync",
    "tirx.ptx.match_all_sync_p",
    "tirx.ptx.elect_sync",
    "tirx.ptx.activemask",
    "tirx.ptx.movmatrix",
    "tirx.ptx.mma",
    "tirx.ptx.mma_f16acc",
    "tirx.ptx.mma_f16c_f32d",
    "tirx.ptx.mma_f64",
    "tirx.ptx.mma_int",
    "tirx.ptx.mma_sp",
    "tirx.ptx.mma_sp_pair",
    "tirx.ptx.mma_sp_all",
    "tirx.ptx.mma_sp_f16acc",
    "tirx.ptx.mma_sp_f16acc_pair",
    "tirx.ptx.mma_sp_int_all",
    "tirx.ptx.mma_sp_int_pair",
];

fn lanes32(ops: &Operands, io: &PtxIo<'_>, i: usize) -> WarpValue<u32> {
    std::array::from_fn(|l| ops.src(io, i, l) as u32)
}

fn boxed(f: impl Fn(&mut PtxIo<'_>) -> OpResult + Send + Sync + 'static) -> Resolved {
    Resolved::Boxed(Box::new(f))
}

type Slot = (&'static str, &'static [&'static str], bool);

const MMA_SHAPES: &[&str] = &["m8n8k4", "m16n8k4", "m16n8k8", "m16n8k16", "m16n8k32", "m8n8k16", "m8n8k32", "m16n8k64", "m8n8k128", "m16n8k128", "m16n8k256"];
const LAYOUT: &[&str] = &["row", "col"];
const MMA_TYPES: &[&str] = &["f16", "bf16", "tf32", "e4m3", "e5m2", "f32", "f64", "s32", "u8", "s8", "u4", "s4", "b1"];
const RND: &[&str] = &["rn", "rz", "rm", "rp"];

/// TVM PTX table modifier slots of the ops this family owns (in table order).
fn slots(short: &str) -> Option<&'static [Slot]> {
    const SHFL: &[Slot] = &[("mode", &["up", "down", "bfly", "idx"], false), ("type", &["b32"], false)];
    const VOTE: &[Slot] = &[("mode", &["all", "any", "uni"], false), ("type", &["pred"], false)];
    const BALLOT: &[Slot] = &[("mode", &["ballot"], false), ("type", &["b32"], false)];
    const REDUX: &[Slot] = &[("op", &["add", "min", "max"], false), ("type", &["u32", "s32"], false)];
    const REDUX_BIT: &[Slot] = &[("op", &["and", "or", "xor"], false), ("type", &["b32"], false)];
    const REDUX_F32: &[Slot] =
        &[("op", &["min", "max"], false), ("abs", &["abs"], true), ("nan", &["NaN"], true), ("type", &["f32"], false)];
    const MATCH_ALL: &[Slot] = &[("mode", &["all"], false), ("sync", &["sync"], false), ("type", &["b32", "b64"], false)];
    const MATCH_ANY: &[Slot] = &[("mode", &["any"], false), ("sync", &["sync"], false), ("type", &["b32", "b64"], false)];
    const NONE: &[Slot] = &[];
    const ACTIVE: &[Slot] = &[("type", &["b32"], false)];
    const MOVM: &[Slot] = &[
        ("sync", &["sync"], false),
        ("aligned", &["aligned"], false),
        ("shape", &["m8n8"], false),
        ("trans", &["trans"], false),
        ("type", &["b16"], false),
    ];
    const MMA: &[Slot] = &[
        ("sync", &["sync"], false),
        ("aligned", &["aligned"], false),
        ("shape", MMA_SHAPES, false),
        ("alayout", LAYOUT, false),
        ("blayout", LAYOUT, false),
        ("satfinite", &["satfinite"], true),
        ("dtype", MMA_TYPES, false),
        ("atype", MMA_TYPES, false),
        ("btype", MMA_TYPES, false),
        ("ctype", MMA_TYPES, false),
        ("rnd", RND, true),
        ("bitop", &["xor", "and"], true),
        ("popc", &["popc"], true),
    ];
    const MMA_SP: &[Slot] = &[
        ("spvariant", &["sp", "sp::ordered_metadata"], false),
        ("sync", &["sync"], false),
        ("aligned", &["aligned"], false),
        ("shape", MMA_SHAPES, false),
        ("alayout", LAYOUT, false),
        ("blayout", LAYOUT, false),
        ("satfinite", &["satfinite"], true),
        ("dtype", MMA_TYPES, false),
        ("atype", MMA_TYPES, false),
        ("btype", MMA_TYPES, false),
        ("ctype", MMA_TYPES, false),
    ];
    Some(match short {
        "mma_sp" | "mma_sp_pair" | "mma_sp_all" | "mma_sp_f16acc" | "mma_sp_f16acc_pair" | "mma_sp_int_all"
        | "mma_sp_int_pair" => MMA_SP,
        "shfl_sync" | "shfl_sync_p" => SHFL,
        "vote_sync" => VOTE,
        "vote_sync_ballot" => BALLOT,
        "redux_sync" => REDUX,
        "redux_sync_bitwise" => REDUX_BIT,
        "redux_sync_f32" => REDUX_F32,
        "match_any_sync" => MATCH_ANY,
        "match_all_sync" | "match_all_sync_p" => MATCH_ALL,
        "elect_sync" => NONE,
        "activemask" => ACTIVE,
        "movmatrix" => MOVM,
        "mma" | "mma_f16acc" | "mma_f16c_f32d" | "mma_f64" | "mma_int" => MMA,
        _ => return None,
    })
}

pub(in crate::oplib) fn resolve(name: &str, mods: &Mods, ops: &Operands) -> OpResult<Option<Resolved>> {
    let Some(short) = name.strip_prefix("tirx.ptx.") else {
        return Ok(None);
    };
    let Some(table) = slots(short) else {
        return Ok(None);
    };
    let normalized = mods.normalize(table, name)?;
    let mods = &normalized;
    let ops = ops.clone();
    Ok(Some(match short {
        "shfl_sync" | "shfl_sync_p" => {
            let with_p = short == "shfl_sync_p";
            ops.arity(if with_p { 2 } else { 1 }, 4, name)?;
            let mode = match mods.req("mode", name)? {
                "up" => lib::ShuffleMode::Up,
                "down" => lib::ShuffleMode::Down,
                "bfly" => lib::ShuffleMode::Xor,
                "idx" => lib::ShuffleMode::Index,
                other => return Err(OpError::unsupported(format!("{name}: mode {other}"))),
            };
            boxed(move |io| {
                let values = lanes32(&ops, io, 0);
                let (selectors, controls, members) = (lanes32(&ops, io, 1), lanes32(&ops, io, 2), lanes32(&ops, io, 3));
                let (out, pred) = lib::shfl_sync(io.mask, &members, &values, &selectors, &controls, mode)?;
                for l in io.mask.lanes() {
                    ops.put(io, 0, l, u64::from(out[l]), 32, false);
                    if with_p {
                        ops.put(io, 1, l, u64::from(pred[l]), 1, false);
                    }
                }
                Ok(())
            })
        }
        "vote_sync" | "vote_sync_ballot" => {
            ops.arity(1, 2, name)?;
            let mode = mods.req("mode", name)?.to_string();
            boxed(move |io| {
                let preds: WarpValue<bool> = std::array::from_fn(|l| ops.src(io, 0, l) & 1 != 0);
                let members = lanes32(&ops, io, 1);
                let out: WarpValue<u64> = match mode.as_str() {
                    "all" => lib::vote_all(io.mask, &members, &preds)?.map(u64::from),
                    "any" => lib::vote_any(io.mask, &members, &preds)?.map(u64::from),
                    "uni" => lib::vote_uni(io.mask, &members, &preds)?.map(u64::from),
                    "ballot" => lib::vote_ballot(io.mask, &members, &preds)?.map(u64::from),
                    other => return Err(OpError::unsupported(format!("vote.sync.{other}"))),
                };
                let bits = if mode == "ballot" { 32 } else { 1 };
                for l in io.mask.lanes() {
                    ops.put(io, 0, l, out[l], bits, false);
                }
                Ok(())
            })
        }
        "redux_sync" | "redux_sync_bitwise" | "redux_sync_f32" => {
            ops.arity(1, 2, name)?;
            let op = mods.req("op", name)?.to_string();
            let ty = mods.req("type", name)?.to_string();
            let abs = mods.has("abs");
            let nan = mods.has("NaN");
            if abs {
                return Err(OpError::unsupported(format!("{name}: .abs was not modeled by legacy")));
            }
            let int_op = |op: &str| -> OpResult<lib::ReduxIntOp> {
                Ok(match op {
                    "add" => lib::ReduxIntOp::Add,
                    "min" => lib::ReduxIntOp::Min,
                    "max" => lib::ReduxIntOp::Max,
                    "and" => lib::ReduxIntOp::And,
                    "or" => lib::ReduxIntOp::Or,
                    "xor" => lib::ReduxIntOp::Xor,
                    other => return Err(OpError::unsupported(format!("redux.sync.{other}"))),
                })
            };
            match ty.as_str() {
                "u32" | "b32" => {
                    let op = int_op(&op)?;
                    boxed(move |io| {
                        let out = lib::redux_sync_u32(io.mask, &lanes32(&ops, io, 1), &lanes32(&ops, io, 0), op)?;
                        for l in io.mask.lanes() {
                            ops.put(io, 0, l, u64::from(out[l]), 32, false);
                        }
                        Ok(())
                    })
                }
                "s32" => {
                    let op = int_op(&op)?;
                    boxed(move |io| {
                        let values: WarpValue<i32> = lanes32(&ops, io, 0).map(|v| v as i32);
                        let out = lib::redux_sync_i32(io.mask, &lanes32(&ops, io, 1), &values, op)?;
                        for l in io.mask.lanes() {
                            ops.put(io, 0, l, u64::from(out[l] as u32), 32, true);
                        }
                        Ok(())
                    })
                }
                "f32" => {
                    let op = match (op.as_str(), nan) {
                        ("min", false) => lib::ReduxF32Op::Min,
                        ("max", false) => lib::ReduxF32Op::Max,
                        ("min", true) => lib::ReduxF32Op::MinNan,
                        ("max", true) => lib::ReduxF32Op::MaxNan,
                        (other, _) => return Err(OpError::unsupported(format!("redux.sync.{other}.f32"))),
                    };
                    boxed(move |io| {
                        let values: WarpValue<f32> = lanes32(&ops, io, 0).map(f32::from_bits);
                        let out = lib::redux_sync_f32(io.mask, &lanes32(&ops, io, 1), &values, op)?;
                        for l in io.mask.lanes() {
                            ops.put(io, 0, l, u64::from(out[l].to_bits()), 32, false);
                        }
                        Ok(())
                    })
                }
                other => return Err(OpError::unsupported(format!("{name}.{other}"))),
            }
        }
        "match_any_sync" | "match_all_sync" | "match_all_sync_p" => {
            let with_p = short == "match_all_sync_p";
            ops.arity(if with_p { 2 } else { 1 }, 2, name)?;
            let wide = match mods.req("type", name)? {
                "b32" => false,
                "b64" => true,
                other => return Err(OpError::unsupported(format!("{name}.{other}"))),
            };
            let any = short == "match_any_sync";
            boxed(move |io| {
                let values: WarpValue<u64> =
                    std::array::from_fn(|l| if wide { ops.src(io, 0, l) } else { ops.src(io, 0, l) & 0xffff_ffff });
                let members = lanes32(&ops, io, 1);
                if any {
                    let out = lib::match_any(io.mask, &members, &values)?;
                    for l in io.mask.lanes() {
                        ops.put(io, 0, l, u64::from(out[l]), 32, false);
                    }
                } else {
                    let (out, pred) = lib::match_all(io.mask, &members, &values)?;
                    for l in io.mask.lanes() {
                        ops.put(io, 0, l, u64::from(out[l]), 32, false);
                        if with_p {
                            ops.put(io, 1, l, u64::from(pred[l]), 1, false);
                        }
                    }
                }
                Ok(())
            })
        }
        "elect_sync" => {
            ops.arity(2, 1, name)?;
            boxed(move |io| {
                let (leader, pred) = lib::elect_sync(io.mask, &lanes32(&ops, io, 0))?;
                for l in io.mask.lanes() {
                    ops.put(io, 0, l, u64::from(leader[l]), 32, false);
                    ops.put(io, 1, l, u64::from(pred[l]), 1, false);
                }
                Ok(())
            })
        }
        "activemask" => {
            ops.arity(1, 0, name)?;
            boxed(move |io| {
                let bits = u64::from(io.mask.bits());
                for l in io.mask.lanes() {
                    ops.put(io, 0, l, bits, 32, false);
                }
                Ok(())
            })
        }
        "movmatrix" => {
            ops.arity(1, 1, name)?;
            boxed(move |io| {
                let out = lib::movmatrix_m8n8_trans_b16(io.mask, &lanes32(&ops, io, 0))?;
                for l in io.mask.lanes() {
                    ops.put(io, 0, l, u64::from(out[l]), 32, false);
                }
                Ok(())
            })
        }
        "mma" | "mma_f16acc" | "mma_f16c_f32d" | "mma_f64" | "mma_int" => resolve_mma(short, name, mods, ops)?,
        s if s.starts_with("mma_sp") => resolve_mma_sp(name, mods, ops)?,
        _ => return Ok(None),
    }))
}

/// Register counts (a, b, c, d) of one dense `mma.sync` form.
fn mma_counts(m: usize, k: usize, a_bits: usize, b_bits: usize, c_bits: usize, d_bits: usize) -> (usize, usize, usize, usize) {
    (m * k * a_bits / 1024, 8 * k * b_bits / 1024, m * 8 * c_bits / 1024, m * 8 * d_bits / 1024)
}

fn shape(mods: &Mods, name: &str) -> OpResult<(usize, usize)> {
    let text = mods.req("shape", name)?;
    let parse = || -> Option<(usize, usize)> {
        let rest = text.strip_prefix('m')?;
        let (m, rest) = rest.split_once('n')?;
        let (_n, k) = rest.split_once('k')?;
        Some((m.parse().ok()?, k.parse().ok()?))
    };
    parse().ok_or_else(|| OpError::unsupported(format!("{name}: shape {text}")))
}

fn resolve_mma(short: &str, name: &str, mods: &Mods, ops: Operands) -> OpResult<Resolved> {
    let (m, k) = shape(mods, name)?;
    let alayout = mods.req("alayout", name)?;
    let blayout = mods.req("blayout", name)?;
    let atype = mods.req("atype", name)?.to_string();
    let btype = mods.req("btype", name)?.to_string();
    let m8n8k4 = (m, k) == (8, 4) && short != "mma_f64";
    if !m8n8k4 && (alayout != "row" || blayout != "col") {
        return Err(OpError::unsupported(format!("{name}: .{alayout}.{blayout} layout")));
    }
    let width = |t: &str| -> usize {
        match t {
            "f16" | "bf16" => 16,
            "tf32" | "f32" | "s32" => 32,
            "f64" => 64,
            "e4m3" | "e5m2" | "u8" | "s8" => 8,
            "u4" | "s4" => 4,
            "b1" => 1,
            _ => 0,
        }
    };
    let (ab, bb) = (width(&atype), width(&btype));
    if ab == 0 || bb == 0 {
        return Err(OpError::unsupported(format!("{name}: types {atype}/{btype}")));
    }
    let f8 = |t: &str| match t {
        "e4m3" => Some(mma::MatrixF8Type::E4M3),
        "e5m2" => Some(mma::MatrixF8Type::E5M2),
        _ => None,
    };
    match short {
        "mma" | "mma_f16c_f32d" | "mma_f16acc" if m8n8k4 => {
            let lay = |t: &str| if t == "row" { mma::MatrixLayout::Row } else { mma::MatrixLayout::Col };
            let (al, bl) = (lay(alayout), lay(blayout));
            let acc = |t: &str| if t == "f16" { mma::MatrixAccumulatorType::Fp16 } else { mma::MatrixAccumulatorType::Fp32 };
            let d_ty = acc(mods.req("dtype", name)?);
            let c_ty = acc(mods.req("ctype", name)?);
            if atype != "f16" || btype != "f16" {
                return Err(OpError::unsupported(format!("{name}: m8n8k4 requires f16 operands")));
            }
            let c_count = if c_ty == mma::MatrixAccumulatorType::Fp16 { 4 } else { 8 };
            let d_count = if d_ty == mma::MatrixAccumulatorType::Fp16 { 4 } else { 8 };
            ops.arity(d_count, 4 + c_count, name)?;
            Ok(Resolved::Boxed(Box::new(move |io| {
                let (a, b, c) = (collect(&ops, io, 0, 2), collect(&ops, io, 2, 2), collect(&ops, io, 4, c_count));
                let out = mma::mma_sync_m8n8k4_f16(&a, &b, Some(&c), al, bl, d_ty, c_ty)?;
                let regs: Vec<WarpValue<u32>> = match out {
                    mma::MmaFloatOutput::F32(v) => v.into_iter().map(|r| r.map(f32::to_bits)).collect(),
                    mma::MmaFloatOutput::PackedF16(v) => v,
                };
                write_regs(&ops, io, &regs);
                Ok(())
            })))
        }
        "mma" => {
            let (na, nb, nc, nd) = mma_counts(16, k, ab, bb, 32, 32);
            ops.arity(nd, na + nb + nc, name)?;
            let kind = match (atype.as_str(), btype.as_str()) {
                ("f16", "f16") => 0,
                ("bf16", "bf16") => 1,
                ("tf32", "tf32") => 2,
                (a, b) if f8(a).is_some() && f8(b).is_some() => 3,
                _ => return Err(OpError::unsupported(format!("{name}: {atype}/{btype}"))),
            };
            if m != 16 {
                return Err(OpError::unsupported(format!("{name}: m{m}")));
            }
            let (fa, fb) = (f8(&atype), f8(&btype));
            Ok(Resolved::Boxed(Box::new(move |io| {
                let (a, b, c) = (collect(&ops, io, 0, na), collect(&ops, io, na, nb), collect(&ops, io, na + nb, nc));
                let out = match kind {
                    0 => mma::mma_sync_f32_b16(&a, &b, Some(&c), k, mma::MatrixB16Type::Fp16)?,
                    1 => mma::mma_sync_f32_b16(&a, &b, Some(&c), k, mma::MatrixB16Type::Bf16)?,
                    2 => mma::mma_sync_f32_tf32(&a, &b, Some(&c), k)?,
                    _ => mma::mma_sync_f32_f8(&a, &b, Some(&c), k, fa.unwrap(), fb.unwrap())?,
                };
                let regs: Vec<WarpValue<u32>> = out.into_iter().map(|r| r.map(f32::to_bits)).collect();
                write_regs(&ops, io, &regs);
                Ok(())
            })))
        }
        "mma_f16acc" => {
            let (na, nb, nc, nd) = mma_counts(16, k, ab, bb, 16, 16);
            ops.arity(nd, na + nb + nc, name)?;
            if m != 16 {
                return Err(OpError::unsupported(format!("{name}: m{m}")));
            }
            let (fa, fb) = (f8(&atype), f8(&btype));
            let half = atype == "f16" && btype == "f16";
            if !half && (fa.is_none() || fb.is_none()) {
                return Err(OpError::unsupported(format!("{name}: {atype}/{btype}")));
            }
            Ok(Resolved::Boxed(Box::new(move |io| {
                let (a, b, c) = (collect(&ops, io, 0, na), collect(&ops, io, na, nb), collect(&ops, io, na + nb, nc));
                let regs = if half {
                    mma::mma_sync_f16_f16(&a, &b, Some(&c), k)?
                } else {
                    mma::mma_sync_f16_f8(&a, &b, Some(&c), k, fa.unwrap(), fb.unwrap())?
                };
                write_regs(&ops, io, &regs);
                Ok(())
            })))
        }
        "mma_f64" => {
            let rounding = match mods.get("rnd") {
                None | Some("rn") => numsim_oplib::scalar::F32RoundingMode::Nearest,
                Some("rz") => numsim_oplib::scalar::F32RoundingMode::Zero,
                Some("rm") => numsim_oplib::scalar::F32RoundingMode::Down,
                Some("rp") => numsim_oplib::scalar::F32RoundingMode::Up,
                Some(other) => return Err(OpError::unsupported(format!("{name}: .{other}"))),
            };
            let (na, nb, nc, nd) = mma_counts(m, k, 64, 64, 64, 64);
            let (na, nb) = (na / 2, nb / 2); // one f64 per register, not per 32-bit half
            let (nc, nd) = (nc / 2, nd / 2);
            ops.arity(nd, na + nb + nc, name)?;
            Ok(Resolved::Boxed(Box::new(move |io| {
                let grab = |io: &PtxIo<'_>, start: usize, count: usize| -> Vec<WarpValue<f64>> {
                    (start..start + count).map(|i| std::array::from_fn(|l| f64::from_bits(ops.src(io, i, l)))).collect()
                };
                let (a, b, c) = (grab(io, 0, na), grab(io, na, nb), grab(io, na + nb, nc));
                let out = mma::mma_sync_f64(&a, &b, Some(&c), m, k, rounding)?;
                for (i, reg) in out.iter().enumerate() {
                    for l in io.mask.lanes() {
                        ops.put(io, i, l, reg[l].to_bits(), 64, false);
                    }
                }
                Ok(())
            })))
        }
        "mma_int" => {
            let int = |t: &str| match t {
                "s8" => Some(mma::MatrixPackedIntType::I8),
                "u8" => Some(mma::MatrixPackedIntType::U8),
                "s4" => Some(mma::MatrixPackedIntType::I4),
                "u4" => Some(mma::MatrixPackedIntType::U4),
                "b1" => Some(mma::MatrixPackedIntType::B1),
                _ => None,
            };
            let (Some(ta), Some(tb)) = (int(&atype), int(&btype)) else {
                return Err(OpError::unsupported(format!("{name}: {atype}/{btype}")));
            };
            let saturate = mods.has("satfinite");
            let bit_op = match mods.get("bitop") {
                None => None,
                Some("xor") => Some(mma::MatrixBitOp::Xor),
                Some("and") => Some(mma::MatrixBitOp::And),
                Some(other) => return Err(OpError::unsupported(format!("{name}: .{other}"))),
            };
            let (na, nb, nc, nd) = mma_counts(m, k, ab, bb, 32, 32);
            ops.arity(nd, na + nb + nc, name)?;
            Ok(Resolved::Boxed(Box::new(move |io| {
                let (a, b, c) = (collect(&ops, io, 0, na), collect(&ops, io, na, nb), collect(&ops, io, na + nb, nc));
                let out = mma::mma_sync_packed_integer(&a, &b, Some(&c), m, k, ta, tb, saturate, bit_op)?;
                let regs: Vec<WarpValue<u32>> = out.into_iter().map(|r| r.map(|v| v as u32)).collect();
                write_regs(&ops, io, &regs);
                Ok(())
            })))
        }
        _ => Err(OpError::unsupported(format!("{name}: m{m}k{k}"))),
    }
}

/// `mma.sp{::ordered_metadata}.sync` via `numsim_oplib::mma::mma_sp_sync`.
/// Sources: A regs, B regs, C regs, metadata `e`, selector `f` (uniform; the
/// first executing lane's value is used).
fn resolve_mma_sp(name: &str, mods: &Mods, ops: Operands) -> OpResult<Resolved> {
    let (m, k) = shape(mods, name)?;
    if m != 16 || mods.req("alayout", name)? != "row" || mods.req("blayout", name)? != "col" {
        return Err(OpError::unsupported(format!("{name}: only m16 row.col is modeled")));
    }
    let operand = |t: &str| -> OpResult<mma::MatrixSparseOperandType> {
        Ok(match t {
            "f16" => mma::MatrixSparseOperandType::Fp16,
            "bf16" => mma::MatrixSparseOperandType::Bf16,
            "tf32" => mma::MatrixSparseOperandType::Tf32,
            "s8" => mma::MatrixSparseOperandType::I8,
            "u8" => mma::MatrixSparseOperandType::U8,
            "s4" => mma::MatrixSparseOperandType::I4,
            "u4" => mma::MatrixSparseOperandType::U4,
            "e4m3" => mma::MatrixSparseOperandType::E4M3,
            "e5m2" => mma::MatrixSparseOperandType::E5M2,
            other => return Err(OpError::unsupported(format!("{name}: operand type {other}"))),
        })
    };
    let (ta, tb) = (operand(mods.req("atype", name)?)?, operand(mods.req("btype", name)?)?);
    let acc = match mods.req("ctype", name)? {
        "f16" => mma::MatrixSparseAccumulatorType::Fp16,
        "f32" => mma::MatrixSparseAccumulatorType::Fp32,
        "s32" => mma::MatrixSparseAccumulatorType::I32,
        other => return Err(OpError::unsupported(format!("{name}: accumulator {other}"))),
    };
    let saturate = mods.has("satfinite");
    let ordered = mods.get("spvariant") == Some("sp::ordered_metadata");
    let (na, nb) = mma::sparse_fragment_counts(k, ta)?;
    let nc = if acc == mma::MatrixSparseAccumulatorType::Fp16 { 2 } else { 4 };
    ops.arity(nc, na + nb + nc + 2, name)?;
    Ok(Resolved::Boxed(Box::new(move |io| {
        let (a, b, c) = (collect(&ops, io, 0, na), collect(&ops, io, na, nb), collect(&ops, io, na + nb, nc));
        let metadata: WarpValue<u32> = std::array::from_fn(|l| ops.src(io, na + nb + nc, l) as u32);
        let lane = io.mask.first().unwrap_or(0);
        let selector = ops.src(io, na + nb + nc + 1, lane) as usize;
        let out = mma::mma_sp_sync(&a, &b, &c, &metadata, selector, k, ta, tb, acc, saturate, ordered)?;
        let regs: Vec<WarpValue<u32>> = match out {
            mma::MmaSparseOutput::Float(mma::MmaFloatOutput::F32(v)) => v.into_iter().map(|r| r.map(f32::to_bits)).collect(),
            mma::MmaSparseOutput::Float(mma::MmaFloatOutput::PackedF16(v)) => v,
            mma::MmaSparseOutput::I32(v) => v.into_iter().map(|r| r.map(|x| x as u32)).collect(),
        };
        write_regs(&ops, io, &regs);
        Ok(())
    })))
}

fn collect(ops: &Operands, io: &PtxIo<'_>, start: usize, count: usize) -> Vec<WarpValue<u32>> {
    (start..start + count).map(|i| std::array::from_fn(|l| ops.src(io, i, l) as u32)).collect()
}

fn write_regs(ops: &Operands, io: &mut PtxIo<'_>, regs: &[WarpValue<u32>]) {
    for (i, reg) in regs.iter().enumerate() {
        for l in io.mask.lanes() {
            ops.put(io, i, l, u64::from(reg[l]), 32, false);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::dtype::Ty;
    use crate::oplib::{resolve_ptx, PtxIo};
    use crate::program::OpKey;
    use crate::value::{WarpMask, WarpValue};

    fn key(name: &str, mods: &[&str]) -> OpKey {
        OpKey { name: name.into(), mods: mods.iter().map(|m| m.to_string()).collect() }
    }

    fn run(key: &OpKey, dst_tys: &[Ty], src_tys: &[Ty], srcs: &[WarpValue<u64>], mask: WarpMask) -> Vec<WarpValue<u64>> {
        let f = resolve_ptx(key, dst_tys, src_tys).unwrap();
        let mut dsts = vec![[0u64; 32]; dst_tys.iter().map(|t| t.slots() as usize).sum()];
        let mut io = PtxIo { dsts: &mut dsts, dst_tys, srcs, src_tys, mask };
        f.call(&mut io).unwrap();
        dsts
    }

    #[test]
    fn shfl_vote_redux_match_elect_through_resolve_ptx() {
        let full = [0xffff_ffffu64; 32];
        let values: WarpValue<u64> = std::array::from_fn(|l| 1000 + l as u64);
        let out = run(
            &key("tirx.ptx.shfl_sync_p", &["mode=down", "type=b32"]),
            &[Ty::U32, Ty::PRED],
            &[Ty::U32, Ty::U32, Ty::U32, Ty::U32],
            &[values, [2; 32], [0x1f; 32], full],
            WarpMask::ALL,
        );
        assert_eq!((out[0][0], out[1][0]), (1002, 1));
        assert_eq!((out[0][31], out[1][31]), (1031, 0));

        let preds: WarpValue<u64> = std::array::from_fn(|l| (l % 2) as u64);
        let ballot = run(&key("tirx.ptx.vote_sync_ballot", &["mode=ballot", "type=b32"]), &[Ty::U32], &[Ty::PRED, Ty::U32], &[preds, full], WarpMask::ALL);
        assert_eq!(ballot[0][7], 0xaaaa_aaaa);

        let redux = run(&key("tirx.ptx.redux_sync", &["op=add", "type=u32"]), &[Ty::U32], &[Ty::U32, Ty::U32], &[values, full], WarpMask::ALL);
        assert_eq!(redux[0][5], (1000..1032).sum::<u64>());
        let neg: WarpValue<u64> = std::array::from_fn(|l| (-(l as i32)) as u32 as u64);
        let rmin = run(&key("tirx.ptx.redux_sync", &["op=min", "type=s32"]), &[Ty::S32], &[Ty::S32, Ty::U32], &[neg, full], WarpMask::ALL);
        assert_eq!(rmin[0][0], (-31i32) as u32 as u64);

        let keys: WarpValue<u64> = std::array::from_fn(|l| (l % 4) as u64);
        let any = run(&key("tirx.ptx.match_any_sync", &["mode=any", "sync=sync", "type=b32"]), &[Ty::U32], &[Ty::U32, Ty::U32], &[keys, full], WarpMask::ALL);
        assert_eq!(any[0][1], 0x2222_2222);

        let elect = run(&key("tirx.ptx.elect_sync", &[]), &[Ty::U32, Ty::PRED], &[Ty::U32], &[[0xffff_fff0; 32]], WarpMask(0xffff_fff0));
        assert_eq!((elect[0][9], elect[1][4], elect[1][9]), (4, 1, 0));

        let bad = resolve_ptx(&key("tirx.ptx.redux_sync_f32", &["op=min", "abs=abs", "type=f32"]), &[Ty::F32], &[Ty::F32, Ty::U32]);
        assert!(bad.is_err());
        // Bare tokens (W1 `mod_tokens`) resolve to the same form.
        let bare = run(&key("tirx.ptx.redux_sync", &["add", "u32"]), &[Ty::U32], &[Ty::U32, Ty::U32], &[values, full], WarpMask::ALL);
        assert_eq!(bare[0][3], redux[0][3]);
        assert!(resolve_ptx(&key("tirx.ptx.redux_sync", &["mul", "u32"]), &[Ty::U32], &[Ty::U32, Ty::U32]).is_err());
    }

    #[test]
    fn mma_sync_bf16_m16n8k16_matches_reference_through_resolve_ptx() {
        // A = 1.0 everywhere, B = 2.0, C = 0.5: every D element = 16*2 + 0.5.
        let one_bf16 = 0x3f80u64 | (0x3f80u64 << 16);
        let two_bf16 = 0x4000u64 | (0x4000u64 << 16);
        let half = 0.5f32.to_bits() as u64;
        let mut srcs = vec![[one_bf16; 32]; 4];
        srcs.extend(vec![[two_bf16; 32]; 2]);
        srcs.extend(vec![[half; 32]; 4]);
        let src_tys = vec![Ty::U32; 6].into_iter().chain(vec![Ty::F32; 4]).collect::<Vec<_>>();
        let out = run(
            &key("tirx.ptx.mma", &["sync=sync", "aligned=aligned", "shape=m16n8k16", "alayout=row", "blayout=col", "dtype=f32", "atype=bf16", "btype=bf16", "ctype=f32"]),
            &[Ty::F32; 4],
            &src_tys,
            &srcs,
            WarpMask::ALL,
        );
        for reg in &out {
            assert!(reg.iter().all(|&v| f32::from_bits(v as u32) == 32.5));
        }
    }
}
