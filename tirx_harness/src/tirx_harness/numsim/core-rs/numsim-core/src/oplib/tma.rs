//! Tensor-map descriptor codec and TMA address plans.
//!
//! Everything delegates to `numsim_oplib::tma`:
//!
//! * The 128-byte encoding is `numsim_oplib::tma::TensorMapImage`'s private
//!   NumSim image (host-address form: `allocation_id` = the global VA,
//!   `host_address` flag set, `base_byte_offset` 0). The image has no L2
//!   promotion field, so `l2_promotion` lives in the reserved descriptor
//!   tail at byte [`L2_PROMOTION_BYTE`] (the image decoder ignores the tail).
//! * `tensormap.replace` goes through `TensorMapImage::replace_field`.
//! * Plans build a `TensorMapLayout` (via `TensorMapImage::materialize`, which
//!   runs every descriptor-range check) and convert the byte runs of
//!   `plan_tiled_g2s` / `plan_gather4_g2s` / `plan_im2col_g2s` /
//!   `plan_im2col_s2g` into matched `ByteSpan` lists.
//!
//! Contract encodings of [`TensorMapDesc`] fields:
//! * `swizzle`: 0 none, 1 = 32B, 2 = 64B, 3 = 128B (16B atoms), 4 = 128B with
//!   32B atoms, 5 = 128B with 32B atoms + 8B flip, 6 = 128B with 64B atoms
//!   (the `CUtensorMapSwizzle` numbering), 7 = 96B (sm_103a
//!   `tensormap.replace.swizzle_mode` 4; NumSim extension).
//!   `swizzle_atomicity` (0 = implied by `swizzle`, 1 = 32B, 2 = 32B + 8B
//!   flip, 3 = 64B) names atomicities codes 4..6 cannot (decode emits it only
//!   then; codes 4..6 plus a different nonzero override are invalid).
//! * `interleave`: 0 none, 1 = 16B, 2 = 32B (`CUtensorMapInterleave`).
//! * `oob_fill`: 0 zero, 1 NaN-request-zero-FMA.
//! * `global_stride[i]` is the byte stride of dimension `i + 1`;
//!   `global_stride[4]` must be 0.
//! * `elem`: `Pred`=bool, `U8`, `S8`, `E4M3`, `UE8M0`, `S16`, `U16`, `F16`,
//!   `BF16`, `S32`, `U32`, `F32`, `TF32`, `F64`, `S64`, `U64`, `E2M1`
//!   (packed 16U4_ALIGN8B layout). The image types U6, U32x2, F32_FTZ and
//!   TF32_FTZ have no `Dtype` and decode as `Unsupported`.
//! * `im2col`: the image's im2col extension (corners, `wide`); `None` =
//!   tiled. An im2col-mode plan on a map without a box uses a zero
//!   bounding box (lower = upper = 0, `wide` from the mode).

use super::{Im2colBox, OpError, OpResult, TensorMapDesc, TmaFill, TmaPlan, TmaPlanDir};
use crate::arena::ByteSpan;
use crate::dtype::Dtype;
use crate::program::{TmaMode, TmapField};
use numsim_oplib::tma::{
    plan_gather4_g2s, plan_im2col_g2s, plan_im2col_s2g, plan_tiled_g2s, plan_tiled_s2g, ByteRun,
    Fp4SharedLayout, G2sPlan, Im2colMode, S2gPlan, SwizzleAtomicity, TensorMapElementType,
    TensorMapFillMode, TensorMapIm2col, TensorMapImage, TensorMapLayout,
    TENSOR_MAP_DESCRIPTOR_BYTES,
};

/// Reserved descriptor-tail byte that carries `l2_promotion`.
pub(crate) const L2_PROMOTION_BYTE: usize = TENSOR_MAP_DESCRIPTOR_BYTES - 1;

// ---------------------------------------------------------------------------
// Field mappings
// ---------------------------------------------------------------------------

fn elem_to_image(elem: Option<Dtype>) -> OpResult<(TensorMapElementType, Option<Fp4SharedLayout>)> {
    use TensorMapElementType as E;
    let Some(elem) = elem else {
        return Err(OpError::invalid("TensorMap has no element type"));
    };
    Ok(match elem {
        Dtype::Pred => (E::Bool, None),
        Dtype::U8 => (E::U8, None),
        Dtype::S8 => (E::I8, None),
        Dtype::E4M3 => (E::Float8E4M3Fn, None),
        Dtype::UE8M0 => (E::Float8E8M0Fnu, None),
        Dtype::S16 => (E::I16, None),
        Dtype::U16 => (E::U16, None),
        Dtype::F16 => (E::F16, None),
        Dtype::BF16 => (E::Bf16, None),
        Dtype::S32 => (E::I32, None),
        Dtype::U32 => (E::U32, None),
        Dtype::F32 => (E::F32, None),
        Dtype::TF32 => (E::Tf32, None),
        Dtype::F64 => (E::F64, None),
        Dtype::S64 => (E::I64, None),
        Dtype::U64 => (E::U64, None),
        Dtype::E2M1 => (E::Float4E2M1Fn, Some(Fp4SharedLayout::Align8Packed)),
        other => {
            return Err(OpError::unsupported(format!(
                "TensorMap element type {other:?} has no CUtensorMap data type"
            )))
        }
    })
}

fn elem_from_image(elem: TensorMapElementType, fp4: Option<Fp4SharedLayout>) -> OpResult<Dtype> {
    use TensorMapElementType as E;
    Ok(match elem {
        E::Bool => Dtype::Pred,
        E::U8 => Dtype::U8,
        E::I8 => Dtype::S8,
        E::Float8E4M3Fn => Dtype::E4M3,
        E::Float8E8M0Fnu => Dtype::UE8M0,
        E::I16 => Dtype::S16,
        E::U16 => Dtype::U16,
        E::F16 => Dtype::F16,
        E::Bf16 => Dtype::BF16,
        E::I32 => Dtype::S32,
        E::U32 => Dtype::U32,
        E::F32 => Dtype::F32,
        E::Tf32 => Dtype::TF32,
        E::F64 => Dtype::F64,
        E::I64 => Dtype::S64,
        E::U64 => Dtype::U64,
        E::Float4E2M1Fn if fp4 == Some(Fp4SharedLayout::Align8Packed) => Dtype::E2M1,
        other => {
            return Err(OpError::unsupported(format!(
                "TensorMap element type {other} (shared layout {fp4:?}) is not representable as a Dtype"
            )))
        }
    })
}

fn swizzle_to_image(code: u8) -> OpResult<(Option<usize>, SwizzleAtomicity)> {
    Ok(match code {
        0 => (None, SwizzleAtomicity::B16),
        1 => (Some(32), SwizzleAtomicity::B16),
        2 => (Some(64), SwizzleAtomicity::B16),
        3 => (Some(128), SwizzleAtomicity::B16),
        4 => (Some(128), SwizzleAtomicity::B32),
        5 => (Some(128), SwizzleAtomicity::B32Flip8),
        6 => (Some(128), SwizzleAtomicity::B64),
        7 => (Some(96), SwizzleAtomicity::B16),
        _ => {
            return Err(OpError::invalid(format!(
                "TensorMap swizzle code {code} is invalid"
            )))
        }
    })
}

fn atomicity_from_code(code: u8) -> OpResult<SwizzleAtomicity> {
    Ok(match code {
        0 => SwizzleAtomicity::B16,
        1 => SwizzleAtomicity::B32,
        2 => SwizzleAtomicity::B32Flip8,
        3 => SwizzleAtomicity::B64,
        _ => {
            return Err(OpError::invalid(format!(
                "TensorMap swizzle atomicity code {code} is invalid"
            )))
        }
    })
}

fn atomicity_code(atomicity: SwizzleAtomicity) -> u8 {
    match atomicity {
        SwizzleAtomicity::B16 => 0,
        SwizzleAtomicity::B32 => 1,
        SwizzleAtomicity::B32Flip8 => 2,
        SwizzleAtomicity::B64 => 3,
    }
}

/// `(swizzle, swizzle_atomicity)` contract codes of an image swizzle.
fn swizzle_from_image(bytes: Option<usize>, atomicity: SwizzleAtomicity) -> OpResult<(u8, u8)> {
    let width = match bytes {
        None => 0,
        Some(32) => 1,
        Some(64) => 2,
        Some(128) => 3,
        Some(96) => 7,
        Some(other) => {
            return Err(OpError::invalid(format!(
                "TensorMap swizzle {other}B has no code"
            )))
        }
    };
    Ok(match (width, atomicity) {
        (3, SwizzleAtomicity::B32) => (4, 0),
        (3, SwizzleAtomicity::B32Flip8) => (5, 0),
        (3, SwizzleAtomicity::B64) => (6, 0),
        (width, atomicity) => (width, atomicity_code(atomicity)),
    })
}

/// Image swizzle of the contract `(swizzle, swizzle_atomicity)` codes.
fn swizzle_of_desc(desc: &TensorMapDesc) -> OpResult<(Option<usize>, SwizzleAtomicity)> {
    let (bytes, implied) = swizzle_to_image(desc.swizzle)?;
    if desc.swizzle_atomicity == 0 {
        return Ok((bytes, implied));
    }
    let explicit = atomicity_from_code(desc.swizzle_atomicity)?;
    if (4..=6).contains(&desc.swizzle) && explicit != implied {
        return Err(OpError::invalid(format!(
            "TensorMap swizzle code {} conflicts with swizzle_atomicity {}",
            desc.swizzle, desc.swizzle_atomicity
        )));
    }
    Ok((bytes, explicit))
}

fn interleave_to_image(code: u8) -> OpResult<Option<usize>> {
    match code {
        0 => Ok(None),
        1 => Ok(Some(16)),
        2 => Ok(Some(32)),
        _ => Err(OpError::invalid(format!(
            "TensorMap interleave code {code} is invalid"
        ))),
    }
}

fn interleave_from_image(bytes: Option<usize>) -> OpResult<u8> {
    match bytes {
        None => Ok(0),
        Some(16) => Ok(1),
        Some(32) => Ok(2),
        Some(other) => Err(OpError::invalid(format!(
            "TensorMap interleave {other}B has no code"
        ))),
    }
}

fn to_usize(value: u64, what: &str) -> OpResult<usize> {
    usize::try_from(value)
        .map_err(|_| OpError::invalid(format!("TensorMap {what} {value} does not fit usize")))
}

/// Structural conversion (no range validation: `TensorMapImage::encode` and
/// `materialize` own those). Unused axes (`>= rank`) with a zero
/// dimension/box/element stride become 1, the image's canonical filler.
fn desc_to_image(desc: &TensorMapDesc) -> OpResult<TensorMapImage> {
    let rank = usize::from(desc.rank);
    if !(1..=5).contains(&rank) {
        return Err(OpError::invalid(format!(
            "TensorMap rank must be in 1..=5, got {rank}"
        )));
    }
    let (element_type, fp4_shared_layout) = elem_to_image(desc.elem)?;
    let (swizzle_bytes, swizzle_atomicity) = swizzle_of_desc(desc)?;
    let fill_mode = match desc.oob_fill {
        0 => TensorMapFillMode::Zero,
        1 => TensorMapFillMode::OobNan,
        other => {
            return Err(OpError::invalid(format!(
                "TensorMap OOB fill code {other} is invalid"
            )))
        }
    };
    if desc.global_stride[4] != 0 {
        return Err(OpError::invalid(
            "TensorMap global_stride[4] must be zero (strides cover dims 1..5)",
        ));
    }
    let unused_one = |axis: usize, value: u64| if axis >= rank && value == 0 { 1 } else { value };
    let mut physical_global_shape = [1_usize; 5];
    let mut box_shape = [1_usize; 5];
    let mut element_strides = [1_usize; 5];
    for axis in 0..5 {
        physical_global_shape[axis] =
            to_usize(unused_one(axis, desc.global_dim[axis]), "global dimension")?;
        box_shape[axis] = to_usize(
            unused_one(axis, u64::from(desc.box_dim[axis])),
            "box dimension",
        )?;
        element_strides[axis] = to_usize(
            unused_one(axis, u64::from(desc.element_stride[axis])),
            "element stride",
        )?;
    }
    let mut physical_global_strides = [0_usize; 4];
    for (axis, stride) in physical_global_strides.iter_mut().enumerate() {
        *stride = to_usize(desc.global_stride[axis], "global stride")?;
    }
    Ok(TensorMapImage {
        allocation_id: desc.global_address,
        base_byte_offset: 0,
        host_address: true,
        rank,
        physical_global_shape,
        physical_global_strides,
        box_shape,
        element_strides,
        element_type,
        interleave_bytes: interleave_to_image(desc.interleave)?,
        fp4_shared_layout,
        swizzle_bytes,
        swizzle_atomicity,
        fill_mode,
        im2col: desc.im2col.map(|b| TensorMapIm2col {
            lower: b.lower,
            upper: b.upper,
            wide: b.wide,
        }),
    })
}

fn image_to_desc(image: &TensorMapImage, l2_promotion: u8) -> OpResult<TensorMapDesc> {
    if !image.host_address || image.base_byte_offset != 0 {
        return Err(OpError::unsupported(
            "TensorMap image is relocated to an allocation id, not a global virtual address",
        ));
    }
    let rank = u8::try_from(image.rank).map_err(|_| OpError::invalid("TensorMap rank overflow"))?;
    let (swizzle, swizzle_atomicity) =
        swizzle_from_image(image.swizzle_bytes, image.swizzle_atomicity)?;
    let mut desc = TensorMapDesc {
        global_address: image.allocation_id,
        rank,
        elem: Some(elem_from_image(
            image.element_type,
            image.fp4_shared_layout,
        )?),
        interleave: interleave_from_image(image.interleave_bytes)?,
        swizzle,
        swizzle_atomicity,
        l2_promotion,
        oob_fill: u8::from(image.fill_mode == TensorMapFillMode::OobNan),
        im2col: image.im2col.as_ref().map(|b| Im2colBox {
            lower: b.lower,
            upper: b.upper,
            wide: b.wide,
        }),
        ..TensorMapDesc::default()
    };
    for axis in 0..5 {
        desc.global_dim[axis] = image.physical_global_shape[axis] as u64;
        desc.box_dim[axis] = u32::try_from(image.box_shape[axis])
            .map_err(|_| OpError::invalid("TensorMap box dimension overflow"))?;
        desc.element_stride[axis] = u32::try_from(image.element_strides[axis])
            .map_err(|_| OpError::invalid("TensorMap element stride overflow"))?;
    }
    for axis in 0..4 {
        desc.global_stride[axis] = image.physical_global_strides[axis] as u64;
    }
    Ok(desc)
}

// ---------------------------------------------------------------------------
// encode / decode / replace
// ---------------------------------------------------------------------------

pub(super) fn try_encode(desc: &TensorMapDesc) -> OpResult<[u8; 128]> {
    let image = desc_to_image(desc)?;
    let mut bytes = [0_u8; 128];
    image.write_descriptor(&mut bytes)?;
    bytes[L2_PROMOTION_BYTE] = desc.l2_promotion;
    Ok(bytes)
}

/// Encode; a descriptor outside the image's ranges (or with an unmappable
/// field) encodes as all zeros, which [`decode`] rejects (invalid magic).
pub(super) fn encode(desc: &TensorMapDesc) -> [u8; 128] {
    try_encode(desc).unwrap_or([0; 128])
}

pub(super) fn decode(bytes: &[u8; 128]) -> OpResult<TensorMapDesc> {
    let image = TensorMapImage::decode_descriptor(bytes)?;
    image_to_desc(&image, bytes[L2_PROMOTION_BYTE])
}

/// `tensormap.replace` with PTX field encodings (value = the PTX `new_val`;
/// `ord` = dimension index for per-dimension fields). Cross-field legality is
/// checked when the map is next encoded/planned.
pub(super) fn replace(
    desc: &mut TensorMapDesc,
    field: TmapField,
    ord: Option<u8>,
    value: u64,
) -> OpResult {
    let index = ord.map(usize::from);
    let per_dim = matches!(
        field,
        TmapField::BoxDim
            | TmapField::GlobalDim
            | TmapField::GlobalStride
            | TmapField::ElementStride
    );
    if per_dim != index.is_some() {
        return Err(OpError::invalid(format!(
            "tensormap.replace {field:?} {} an ordinal",
            if per_dim { "requires" } else { "does not take" }
        )));
    }
    if field == TmapField::GlobalAddress {
        if !value.is_multiple_of(16) {
            return Err(OpError::invalid(
                "tensormap.replace global_address must be 16-byte aligned",
            ));
        }
        desc.global_address = value;
        return Ok(());
    }
    let mut image = desc_to_image(desc)?;
    let value_usize = to_usize(value, "replacement value")?;
    let name = match field {
        TmapField::SwizzleAtomicity => {
            image.swizzle_atomicity = match value {
                0 => SwizzleAtomicity::B16,
                1 => SwizzleAtomicity::B32,
                2 => SwizzleAtomicity::B32Flip8,
                3 => SwizzleAtomicity::B64,
                _ => {
                    return Err(OpError::invalid(format!(
                        "tensormap.replace swizzle_atomicity value {value} is invalid"
                    )))
                }
            };
            None
        }
        TmapField::GlobalAddress => None,
        TmapField::Rank => Some("rank"),
        TmapField::BoxDim => Some("box_dim"),
        TmapField::GlobalDim => Some("global_dim"),
        TmapField::GlobalStride => Some("global_stride"),
        TmapField::GlobalStrideUpper => {
            return Err(OpError::invalid(
                "GlobalStrideUpper is an override operand: use TensorMapDesc::apply_overrides",
            ))
        }
        TmapField::ElementStride => Some("element_stride"),
        TmapField::ElemType => Some("elemtype"),
        TmapField::InterleaveLayout => Some("interleave_layout"),
        TmapField::SwizzleMode => Some("swizzle_mode"),
        TmapField::FillMode => Some("fill_mode"),
    };
    if let Some(name) = name {
        image.replace_field(name, index, value_usize)?;
    }
    *desc = image_to_desc(&image, desc.l2_promotion)?;
    Ok(())
}

/// Per-instruction overrides (legacy `override_tensor_map`).
pub(super) fn apply_overrides(desc: &mut TensorMapDesc, overrides: &[(TmapField, Option<u8>, u64)]) -> OpResult {
    let mut dims: Vec<(u8, u64)> = Vec::new();
    let mut lowers: Vec<(u8, u64)> = Vec::new();
    let mut upper: Option<u64> = None;
    for &(field, ord, value) in overrides {
        match field {
            TmapField::GlobalDim => dims.push((ord.ok_or_else(|| OpError::invalid("GlobalDim override needs an ordinal"))?, value)),
            TmapField::GlobalStride => {
                lowers.push((ord.ok_or_else(|| OpError::invalid("GlobalStride override needs an ordinal"))?, value))
            }
            TmapField::GlobalStrideUpper => {
                if ord.is_some() || upper.replace(value).is_some() {
                    return Err(OpError::invalid("GlobalStrideUpper: exactly one, without an ordinal"));
                }
            }
            _ => replace(desc, field, ord, value)?,
        }
    }
    if dims.is_empty() && lowers.is_empty() && upper.is_none() {
        return Ok(());
    }
    dims.sort_unstable();
    lowers.sort_unstable();
    let ordered = |v: &[(u8, u64)]| v.iter().enumerate().all(|(i, (o, _))| usize::from(*o) == i);
    if !ordered(&dims) || !ordered(&lowers) {
        return Err(OpError::invalid("TMA override ordinals must be 0..rank without gaps or repeats"));
    }
    let rank = usize::from(desc.rank);
    let mut image = desc_to_image(desc)?;
    let as_i64 = |v: u64| i64::try_from(v).map_err(|_| OpError::invalid("TMA override operand out of range"));
    let dims = dims.iter().map(|&(_, v)| as_i64(v)).collect::<OpResult<Vec<_>>>()?;
    let lowers = lowers.iter().map(|&(_, v)| as_i64(v)).collect::<OpResult<Vec<_>>>()?;
    let zeros = vec![0i64; rank];
    image.apply_overrides(rank, &dims, &lowers, as_i64(upper.unwrap_or(0))?, &zeros)?;
    *desc = image_to_desc(&image, desc.l2_promotion)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Plans
// ---------------------------------------------------------------------------

fn push_span(spans: &mut Vec<ByteSpan>, start: u64, len: u64) {
    if len == 0 {
        return;
    }
    if let Some(last) = spans.last_mut() {
        if last.end() == start {
            last.len += len;
            return;
        }
    }
    spans.push(ByteSpan::new(start, len));
}

fn offset(base: u64, delta: usize, what: &str) -> OpResult<u64> {
    u64::try_from(delta)
        .ok()
        .and_then(|delta| base.checked_add(delta))
        .ok_or_else(|| OpError::invalid(format!("TMA {what} address overflow")))
}

/// Pair two run lists that index one payload. `primary` runs are walked in
/// order; each payload byte is matched with the `secondary` run covering it.
/// Returns (matched primary spans, matched secondary spans, unmatched primary
/// spans) with the matched lists in the same (concatenation) order.
fn pair_runs(
    primary: &[ByteRun],
    primary_base: u64,
    secondary: &[ByteRun],
    secondary_base: u64,
) -> OpResult<(Vec<ByteSpan>, Vec<ByteSpan>, Vec<ByteSpan>)> {
    let mut sorted = secondary.to_vec();
    sorted.sort_unstable_by_key(|run| run.payload_offset);
    let mut matched_primary = Vec::new();
    let mut matched_secondary = Vec::new();
    let mut unmatched = Vec::new();
    for run in primary {
        let mut done = 0_usize;
        while done < run.byte_len {
            let payload = run.payload_offset + done;
            let remaining = run.byte_len - done;
            // Last secondary run starting at or before `payload`.
            let index = sorted.partition_point(|candidate| candidate.payload_offset <= payload);
            let covering = index
                .checked_sub(1)
                .map(|i| sorted[i])
                .filter(|candidate| payload < candidate.payload_offset + candidate.byte_len);
            let primary_start = offset(primary_base, run.byte_offset + done, "plan")?;
            match covering {
                Some(candidate) => {
                    let inside = payload - candidate.payload_offset;
                    let len = remaining.min(candidate.byte_len - inside);
                    push_span(&mut matched_primary, primary_start, len as u64);
                    let secondary_start =
                        offset(secondary_base, candidate.byte_offset + inside, "plan")?;
                    push_span(&mut matched_secondary, secondary_start, len as u64);
                    done += len;
                }
                None => {
                    let next = sorted
                        .get(index)
                        .map_or(usize::MAX, |candidate| candidate.payload_offset);
                    let len = remaining.min(next - payload);
                    push_span(&mut unmatched, primary_start, len as u64);
                    done += len;
                }
            }
        }
    }
    Ok((matched_primary, matched_secondary, unmatched))
}

fn load_plan(
    plan: G2sPlan,
    element_type: TensorMapElementType,
    global_address: u64,
    smem_offset: u64,
) -> OpResult<TmaPlan> {
    let (smem, global, smem_oob_fill) = pair_runs(
        &plan.destination_runs,
        smem_offset,
        &plan.source_runs,
        global_address,
    )?;
    let (fill, fill_pattern) = match plan.fill_mode {
        TensorMapFillMode::Zero => (TmaFill::Zero, Vec::new()),
        TensorMapFillMode::OobNan => {
            // `materialize_g2s_payload`: every 16 bits of a unit read 0x7ff7.
            if plan.geometry.unit_bytes < 2 || !plan.geometry.unit_bytes.is_multiple_of(2) {
                return Err(OpError::invalid(
                    "TensorMap OOB-NaN fill requires an even floating-point element width",
                ));
            }
            (
                TmaFill::NanRequestZeroFma,
                numsim_oplib::scalar::PTX_OOB_NAN.to_le_bytes().to_vec(),
            )
        }
    };
    Ok(TmaPlan {
        global,
        smem,
        smem_oob_fill,
        bytes: plan.payload_len as u64,
        fill,
        fill_pattern,
        tf32_round: matches!(
            element_type,
            TensorMapElementType::Tf32 | TensorMapElementType::Tf32Ftz
        ),
    })
}

fn store_plan(plan: S2gPlan, global_address: u64, smem_offset: u64) -> OpResult<TmaPlan> {
    if !plan.destination_bits.is_empty() {
        return Err(OpError::unsupported(
            "sub-byte (FP4/U6) TMA store fragments are not representable in TmaPlan",
        ));
    }
    let (global, smem, _) = pair_runs(
        &plan.destination_runs,
        global_address,
        &plan.source_runs,
        smem_offset,
    )?;
    Ok(TmaPlan {
        global,
        smem,
        bytes: plan.payload_len as u64,
        ..TmaPlan::default()
    })
}

/// `cp.async.bulk.tensor.2d.global.shared::cta.tile::scatter4`: the mirror of
/// `plan_gather4_g2s` (four rows `[col, row_i]` of a rank-2 map whose box is
/// one row) with the tiled-store checks. No legacy oracle exists (legacy did
/// not model scatter4); PTX defines the same four-row box as gather4.
fn plan_scatter4_s2g(
    map: &TensorMapLayout,
    column: i64,
    rows: &[i64],
    base: usize,
) -> OpResult<S2gPlan> {
    if map.global_shape.len() != 2 || map.box_shape.len() != 2 || map.box_shape[1] != 1 {
        return Err(OpError::invalid(
            "TensorMap scatter4 requires a rank-2 map with outer box extent one",
        ));
    }
    if column < 0 || rows.iter().any(|row| *row < 0) {
        return Err(OpError::invalid(
            "tiled TMA store requires nonnegative starting coordinates",
        ));
    }
    if matches!(map.transfer_element_bits(), 4 | 6) {
        return Err(OpError::unsupported(
            "sub-byte (FP4/U6) TMA scatter4 store fragments are not modeled",
        ));
    }
    map.validate_swizzle_direction(false)?;
    let geometry = map.geometry()?;
    let template = &map.transfer_template;
    let mut source_runs = Vec::new();
    let mut destination_runs = Vec::new();
    for (index, &row) in rows.iter().enumerate() {
        let payload_base = index * template.payload_len;
        for mut run in template.bind_global(map, &[column, row])? {
            run.payload_offset += payload_base;
            destination_runs.push(run);
        }
        for unit in 0..geometry.inner_units {
            let byte_offset = map.shared_byte_offset(
                index,
                unit * geometry.unit_stride_bytes,
                geometry.inner_row_bytes,
                base,
            )?;
            source_runs.push(ByteRun {
                byte_offset,
                payload_offset: payload_base + unit * geometry.unit_bytes,
                byte_len: geometry.unit_bytes,
            });
        }
    }
    Ok(S2gPlan {
        source_runs,
        destination_runs,
        destination_bits: Vec::new(),
        unit_bytes: geometry.unit_bytes.min(16),
        payload_len: template.payload_len * 4,
    })
}

/// Address generation for one direction (see `oplib::tma_plan_dir`).
///
/// Load: `Tile`, `TileGather4`, `Im2col`, `Im2colW`, `Im2colW128`.
/// Store (and reduce): `Tile`, `TileScatter4`, `Im2col`/`Im2colNoOffs`
/// (spatial, no offsets), `Im2colW`. Other combinations are invalid.
pub(super) fn plan(
    map: &TensorMapDesc,
    dir: TmaPlanDir,
    mode: TmaMode,
    coords: &[i64],
    im2col_offsets: &[i64],
    smem_offset: u64,
) -> OpResult<TmaPlan> {
    let im2col_mode = match (dir, mode) {
        (_, TmaMode::Tile)
        | (TmaPlanDir::Load, TmaMode::TileGather4)
        | (TmaPlanDir::Store, TmaMode::TileScatter4) => None,
        (TmaPlanDir::Load, TmaMode::Im2col) => Some(Im2colMode::Spatial),
        (TmaPlanDir::Store, TmaMode::Im2col | TmaMode::Im2colNoOffs) => Some(Im2colMode::Spatial),
        (_, TmaMode::Im2colW) => Some(Im2colMode::Wide),
        (TmaPlanDir::Load, TmaMode::Im2colW128) => Some(Im2colMode::Wide128),
        (dir, mode) => {
            return Err(OpError::invalid(format!(
                "TMA {mode:?} is not a valid {dir:?} mode"
            )))
        }
    };
    let mut image = desc_to_image(map)?;
    if let Some(im2col_mode) = im2col_mode {
        let wide = im2col_mode != Im2colMode::Spatial;
        match &image.im2col {
            None => {
                image.im2col = Some(TensorMapIm2col {
                    lower: [0; 3],
                    upper: [0; 3],
                    wide,
                })
            }
            Some(config) if config.wide != wide => {
                return Err(OpError::invalid(
                    "im2col instruction/descriptor layout mismatch (wide)",
                ))
            }
            Some(_) => {}
        }
    }
    image.host_address = false;
    image.allocation_id = 0;
    let layout = image.materialize(image.rank, usize::MAX, map.global_address)?;
    let base =
        usize::try_from(smem_offset).map_err(|_| OpError::invalid("TMA shared offset overflow"))?;
    let element_type = layout.element_type;
    let address = map.global_address;
    let check_rank = |expected: usize| {
        if coords.len() != expected {
            return Err(OpError::invalid(format!(
                "{mode:?} TMA expects {expected} coordinates, got {}",
                coords.len()
            )));
        }
        Ok(())
    };
    match (dir, mode, im2col_mode) {
        (TmaPlanDir::Load, TmaMode::TileGather4, _) => {
            check_rank(5)?;
            let plan = plan_gather4_g2s(&layout, coords[0], &coords[1..], base)?;
            load_plan(plan, element_type, address, smem_offset)
        }
        (TmaPlanDir::Store, TmaMode::TileScatter4, _) => {
            check_rank(5)?;
            let plan = plan_scatter4_s2g(&layout, coords[0], &coords[1..], base)?;
            store_plan(plan, address, smem_offset)
        }
        (TmaPlanDir::Load, _, None) => {
            check_rank(layout.rank())?;
            let plan = plan_tiled_g2s(&layout, coords, base)?;
            load_plan(plan, element_type, address, smem_offset)
        }
        (TmaPlanDir::Store, _, None) => {
            check_rank(layout.rank())?;
            store_plan(plan_tiled_s2g(&layout, coords, base)?, address, smem_offset)
        }
        (TmaPlanDir::Load, _, Some(im2col_mode)) => {
            check_rank(layout.rank())?;
            let plan = plan_im2col_g2s(&layout, coords, im2col_mode, im2col_offsets, base)?;
            load_plan(plan, element_type, address, smem_offset)
        }
        (TmaPlanDir::Store, _, Some(im2col_mode)) => {
            check_rank(layout.rank())?;
            if !im2col_offsets.is_empty() {
                return Err(OpError::invalid("im2col TMA store takes no im2col offsets"));
            }
            let plan = plan_im2col_s2g(&layout, coords, im2col_mode, base)?;
            store_plan(plan, address, smem_offset)
        }
    }
}

#[cfg(test)]
mod tests;
