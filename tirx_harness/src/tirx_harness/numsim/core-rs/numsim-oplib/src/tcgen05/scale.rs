//! Block-scale factor decoding and application for tcgen05 block-scaled MMA.
//!
//! Legacy source: `engine-rs/src/runtime/tcgen_ops.rs`
//! (`raw_tcgen05_decode_{ue8m0,ue5m3,ue4m3}_scale`, `raw_tcgen05_read_block_scale`,
//! `raw_tcgen05_mxf8_scale_values`, the scale loop of
//! `raw_tcgen05_gather_mxf8f6f4_matrix`).
//!
//! TMEM reads are re-expressed as a caller closure
//! `read_byte(lane, column, byte_in_cell) -> OpResult<u8>`.

use super::layouts::{block_scale_location, ScaleLayout};
use crate::cvt::{
    float8_e4m3fn_bits_to_f32, float8_e8m0fnu_bits_to_f32, narrow_float_bits_to_f32_checked,
    FLOAT8_UE5M3,
};
use crate::types::{OpError, OpResult};

/// How one block-scale byte in TMEM becomes the factor the MMA multiplies by.
pub type ScaleDecoder = fn(u8) -> OpResult<f32>;

/// UE8M0 block scale: exactly `2^(bits - 127)` (code 0 is the f32 subnormal `2^-127`);
/// `0xff` is NaN (`f32::NAN`). Never errors.
pub fn decode_ue8m0_scale(bits: u8) -> OpResult<f32> {
    Ok(float8_e8m0fnu_bits_to_f32(bits))
}

/// UE5M3 (unsigned, NaN-only specials) block scale, exact including subnormals; the
/// all-ones code is NaN (`f32::NAN`). Never errors.
pub fn decode_ue5m3_scale(bits: u8) -> OpResult<f32> {
    Ok(narrow_float_bits_to_f32_checked(bits, FLOAT8_UE5M3).unwrap_or(f32::NAN))
}

/// `ue4m3` is a 7-bit unsigned format whose MSB is padding PTX ISA 5.2.3
/// requires to be zero; a set padding bit fails closed.
pub fn decode_ue4m3_scale(bits: u8) -> OpResult<f32> {
    if bits & 0x80 != 0 {
        return Err(OpError::message(format!(
            "raw mxf4nvf4 ue4m3 scale byte {bits:#04x} sets the padding MSB"
        )));
    }
    Ok(float8_e4m3fn_bits_to_f32(bits))
}

/// One block-scale factor: its TMEM byte and the format-exact decode.
#[allow(clippy::too_many_arguments)]
pub fn read_block_scale(
    read_byte: &mut impl FnMut(usize, usize, usize) -> OpResult<u8>,
    address: u32,
    scale_id: usize,
    matrix_row: usize,
    vector_index: usize,
    decode: ScaleDecoder,
    matrix_rows: usize,
    lanes_per_column: usize,
) -> OpResult<f32> {
    let (lane, column, byte) = block_scale_location(
        address,
        scale_id,
        matrix_row,
        vector_index,
        matrix_rows,
        lanes_per_column,
    )?;
    decode(read_byte(lane, column, byte)?)
}

/// `(locations (lane, column) in read order, lane_end, column_end)`.
pub type ScaleLocations = (Vec<(usize, usize)>, usize, usize);

/// Every replicated scale location of `rows` rows, in read order, plus the
/// `(lane_end, column_end)` rectangle legacy validates before reading.
pub fn mxf8_scale_locations(
    address: u32,
    scale_id: usize,
    rows: usize,
    layout: ScaleLayout,
) -> OpResult<ScaleLocations> {
    if scale_id >= 4 {
        return Err(OpError::message("raw TCGEN scale byte is outside TMEM"));
    }
    let mut locations = Vec::with_capacity(rows * layout.replicas());
    let mut lane_end = 0;
    let mut column_end = 0;
    for row in 0..rows {
        for replica in 0..layout.replicas() {
            let location = layout.location(address, row, replica)?;
            lane_end = lane_end.max(location.0 + 1);
            column_end = column_end.max(location.1 + 1);
            locations.push(location);
        }
    }
    Ok((locations, lane_end, column_end))
}

/// Select the common value of each row's replicas and decode it as UE8M0.
/// `bytes` is in [`mxf8_scale_locations`] order.
pub fn mxf8_scale_values_from_bytes(bytes: &[u8], replicas: usize) -> OpResult<Vec<f32>> {
    let mut values = Vec::with_capacity(bytes.len() / replicas);
    for copies in bytes.chunks_exact(replicas) {
        let mut common = None;
        for &bits in copies {
            if common.is_some_and(|value| value != bits) {
                return Err(OpError::message("raw TCGEN block-scale replicas disagree"));
            }
            common = Some(bits);
        }
        values.push(decode_ue8m0_scale(common.expect("scale has replicas"))?);
    }
    Ok(values)
}

/// Read every required copy before selecting the common scale value
/// (legacy `raw_tcgen05_mxf8_scale_values`).
pub fn mxf8_scale_values(
    read_byte: &mut impl FnMut(usize, usize, usize) -> OpResult<u8>,
    address: u32,
    scale_id: usize,
    rows: usize,
    layout: ScaleLayout,
) -> OpResult<Vec<f32>> {
    let (locations, _, _) = mxf8_scale_locations(address, scale_id, rows, layout)?;
    let bytes = locations
        .iter()
        .map(|&(lane, column)| read_byte(lane, column, scale_id))
        .collect::<OpResult<Vec<_>>>()?;
    mxf8_scale_values_from_bytes(&bytes, layout.replicas())
}

/// Multiply each row of a `rows x k` block by its scale (indexed with
/// `scale_row_base + row`) and optionally negate, in legacy order.
pub fn apply_row_scales(
    values: &mut [f32],
    k: usize,
    scales: &[f32],
    scale_row_base: usize,
    negate: bool,
) {
    for (row, values) in values.chunks_exact_mut(k).enumerate() {
        let scale = scales[scale_row_base + row];
        for value in values {
            *value *= scale;
            if negate {
                *value = -*value;
            }
        }
    }
}

/// SFB copies of a joint-row CTA pair must agree bit-for-bit.
pub fn check_joint_scales(previous: Option<&[f32]>, scales: &[f32]) -> OpResult<()> {
    if previous.is_some_and(|previous| {
        previous
            .iter()
            .zip(scales)
            .any(|(a, b)| a.to_bits() != b.to_bits())
    }) {
        return Err(OpError::message(
            "raw TCGEN SFB copies disagree across CTAs",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ue4m3_scale_decodes_as_unsigned_e4m3_and_rejects_a_set_padding_bit() {
        assert_eq!(decode_ue4m3_scale(0x38).unwrap(), 1.0_f32);
        assert_eq!(decode_ue4m3_scale(0x3c).unwrap(), 1.5_f32);
        assert_eq!(decode_ue4m3_scale(0x00).unwrap(), 0.0_f32);
        assert!(decode_ue4m3_scale(0x7f).unwrap().is_nan());
        let error = decode_ue4m3_scale(0xb8).unwrap_err();
        assert!(error.to_string().contains("sets the padding MSB"));
    }

    #[test]
    fn mxf8_scale_replicas_must_agree() {
        assert_eq!(
            mxf8_scale_values_from_bytes(&[127, 127, 128, 128], 2).unwrap(),
            vec![1.0, 2.0]
        );
        assert!(mxf8_scale_values_from_bytes(&[127, 128], 2).is_err());
    }
}
