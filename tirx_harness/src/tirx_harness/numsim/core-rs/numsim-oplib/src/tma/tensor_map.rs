//! Validated TensorMap layout (descriptor metadata without a memory view),
//! transfer geometry, and box-coordinate helpers.
//!
//! Legacy source: `engine-rs/src/runtime/tensor_map.rs`
//! (`RuntimeTensorMap::new_with_layout`, `validate_swizzle_direction`,
//! `validate_u6_origin`, `RuntimeTensorMapImage::{from_tensor_map,
//! materialize}`, `TensorMapGeometry`, `tensor_map_geometry*`,
//! `tensor_map_outer_coordinates`, `tensor_map_global_coordinates*`,
//! `tensor_map_coordinates_in_bounds`, `tensor_map_global_byte_offset`).
//!
//! The legacy map owned a `BufferView`; here the caller supplies only the
//! view's absolute base address (for alignment) and its byte length (for the
//! global-span check). All global byte offsets produced by this module are
//! relative to the start of that view.

use std::sync::Arc;

use super::descriptor::{
    analysis_incomplete, Fp4SharedLayout, TensorMapElementType, TensorMapFillMode, TensorMapIm2col,
    TensorMapImage, MAX_BOX_DIMENSION, MAX_ELEMENT_STRIDE, MAX_GLOBAL_DIMENSION, MAX_GLOBAL_STRIDE,
};
use super::swizzle::{shared_byte_offset, SwizzleAtomicity};
use super::tiled::TransferTemplate;
use crate::types::{OpError, OpResult};

/// Construction parameters of a TensorMap (legacy `RuntimeTensorMap::new*`
/// argument list minus the view).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorMapSpec {
    pub global_shape: Vec<usize>,
    /// Byte strides of axes 1.. (length `rank - 1`).
    pub global_strides: Vec<usize>,
    /// `rank` entries for tiled maps, `[channels, pixels]` for im2col.
    pub box_shape: Vec<usize>,
    pub element_strides: Vec<usize>,
    pub element_bits: usize,
    pub element_type: TensorMapElementType,
    pub fp4_shared_layout: Option<Fp4SharedLayout>,
    pub swizzle_bytes: Option<usize>,
    pub swizzle_atomicity: SwizzleAtomicity,
    pub fill_mode: TensorMapFillMode,
    pub interleave_bytes: Option<usize>,
    pub im2col: Option<TensorMapIm2col>,
}

impl TensorMapSpec {
    /// Tiled, non-interleaved spec with default (16B) swizzle atomicity.
    #[allow(clippy::too_many_arguments)]
    pub fn tiled(
        global_shape: Vec<usize>,
        global_strides: Vec<usize>,
        box_shape: Vec<usize>,
        element_strides: Vec<usize>,
        element_type: TensorMapElementType,
        fp4_shared_layout: Option<Fp4SharedLayout>,
        swizzle_bytes: Option<usize>,
        fill_mode: TensorMapFillMode,
    ) -> Self {
        Self {
            global_shape,
            global_strides,
            box_shape,
            element_strides,
            element_bits: element_type.bits(),
            element_type,
            fp4_shared_layout,
            swizzle_bytes,
            swizzle_atomicity: SwizzleAtomicity::B16,
            fill_mode,
            interleave_bytes: None,
            im2col: None,
        }
    }
}

/// Validated TensorMap metadata plus its compiled transfer template.
#[derive(Clone, Debug)]
pub struct TensorMapLayout {
    pub global_shape: Vec<usize>,
    pub global_strides: Vec<usize>,
    pub physical_global_shape: [usize; 5],
    pub physical_global_strides: [usize; 4],
    pub box_shape: Vec<usize>,
    pub element_strides: Vec<usize>,
    /// Box extent in elements traversed along each axis (after element
    /// strides and interleave/im2col adjustment).
    pub traversal_shape: Vec<usize>,
    pub element_bits: usize,
    pub interleave_bytes: Option<usize>,
    pub element_type: TensorMapElementType,
    pub fp4_shared_layout: Option<Fp4SharedLayout>,
    pub swizzle_bytes: Option<usize>,
    pub swizzle_atomicity: SwizzleAtomicity,
    pub fill_mode: TensorMapFillMode,
    pub transfer_template: Arc<TransferTemplate>,
    pub im2col: Option<TensorMapIm2col>,
}

/// Per-transfer packing of box elements into shared-memory units.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorMapGeometry {
    pub packed_elements: usize,
    pub unit_bytes: usize,
    pub unit_stride_bytes: usize,
    pub inner_units: usize,
    pub inner_row_bytes: usize,
    pub outer_count: usize,
}

/// Transfer geometry (packing, unit bytes/stride, inner units, outer rows) of a box with
/// `traversal_shape` and element width; FP4 needs its shared layout. No numerics.
pub fn tensor_map_geometry_from_metadata(
    traversal_shape: &[usize],
    element_bits: usize,
    fp4_shared_layout: Option<Fp4SharedLayout>,
) -> OpResult<TensorMapGeometry> {
    let (packed_elements, unit_bytes, unit_stride_bytes) = match (element_bits, fp4_shared_layout) {
        (4, Some(Fp4SharedLayout::Align8Packed)) => (2_usize, 1_usize, 1_usize),
        (4, Some(Fp4SharedLayout::Align16Padded)) => (16_usize, 8_usize, 16_usize),
        (6, None) => (16_usize, 12_usize, 16_usize),
        (4, None) => {
            return Err(OpError::message(
                "FP4 TensorMap is missing its shared layout",
            ));
        }
        (8 | 16 | 32 | 64 | 128 | 256, None) | (128 | 256, Some(Fp4SharedLayout::Align8Packed)) => {
            // Interleave transfers whole byte slices even for packed FP4;
            // only non-interleaved four-bit units need nibble processing.
            let bytes = element_bits / 8;
            (1_usize, bytes, bytes)
        }
        (bits, layout) => {
            return Err(OpError::message(format!(
                "unsupported TensorMap geometry for {bits}-bit elements and {layout:?}"
            )));
        }
    };
    let inner_units = traversal_shape[0]
        .checked_add(packed_elements - 1)
        .ok_or_else(|| OpError::message("TensorMap inner box size overflow"))?
        / packed_elements;
    let inner_row_bytes = inner_units
        .checked_mul(unit_stride_bytes)
        .ok_or_else(|| OpError::message("TensorMap inner row byte size overflow"))?;
    let outer_count = traversal_shape[1..]
        .iter()
        .try_fold(1_usize, |product, extent| {
            product
                .checked_mul(*extent)
                .ok_or_else(|| OpError::message("TensorMap outer box size overflow"))
        })?;
    Ok(TensorMapGeometry {
        packed_elements,
        unit_bytes,
        unit_stride_bytes,
        inner_units,
        inner_row_bytes,
        outer_count,
    })
}

/// TMA transaction-byte count for one transfer unit.
pub fn tensor_map_transaction_bytes(unit_bytes: usize) -> OpResult<u64> {
    u64::try_from(unit_bytes)
        .map_err(|_| OpError::message("TensorMap transaction byte count overflow"))
}

impl TensorMapLayout {
    /// Validate a TensorMap against hardware descriptor ranges and compile its
    /// transfer template. `base_address` is the absolute global address of
    /// the view start (alignment checks); `view_byte_len` bounds the span.
    pub fn new(spec: TensorMapSpec, base_address: u64, view_byte_len: usize) -> OpResult<Self> {
        let TensorMapSpec {
            global_shape,
            global_strides,
            box_shape,
            mut element_strides,
            element_bits,
            element_type,
            fp4_shared_layout,
            swizzle_bytes,
            swizzle_atomicity,
            fill_mode,
            interleave_bytes,
            im2col,
        } = spec;
        let rank = global_shape.len();
        if rank == 0
            || rank > 5
            || global_strides.len() + 1 != rank
            || box_shape.len() != if im2col.is_some() { 2 } else { rank }
            || element_strides.len() != rank
            || global_shape.contains(&0)
            || box_shape.contains(&0)
        {
            return Err(OpError::message(
                "TensorMap rank/shape/stride metadata is inconsistent",
            ));
        }
        if swizzle_bytes.is_some_and(|width| width != 128)
            && swizzle_atomicity != SwizzleAtomicity::B16
        {
            return Err(OpError::message(
                "non-default swizzle atomicity requires 128B swizzle",
            ));
        }
        if swizzle_atomicity == SwizzleAtomicity::B32Flip8
            && (element_type == TensorMapElementType::U6
                || fp4_shared_layout == Some(Fp4SharedLayout::Align16Padded))
        {
            return Err(OpError::message(
                "padded FP4 and U6 TensorMaps do not support 8B flip",
            ));
        }
        if im2col.as_ref().is_some_and(|config| config.wide)
            && matches!(
                swizzle_atomicity,
                SwizzleAtomicity::B32Flip8 | SwizzleAtomicity::B64
            )
        {
            return Err(OpError::message(
                "wide im2col does not support 8B flip or 64B atomicity",
            ));
        }
        let dtype_bits = element_type.bits();
        if dtype_bits != element_bits {
            return Err(OpError::message(format!(
                "TensorMap logical dtype {element_type} has {dtype_bits} bits, but descriptor declares {element_bits} bits"
            )));
        }
        if global_shape
            .iter()
            .any(|dimension| *dimension as u128 > MAX_GLOBAL_DIMENSION)
        {
            return Err(OpError::message(
                "TensorMap global dimensions must be at most 2^32",
            ));
        }
        if im2col.is_none()
            && box_shape
                .iter()
                .any(|dimension| *dimension > MAX_BOX_DIMENSION)
        {
            return Err(OpError::message(
                "TensorMap box dimensions must be in 1..=256",
            ));
        }
        if let Some(config) = &im2col {
            super::im2col::validate_im2col_bounds(
                config,
                &global_shape,
                &box_shape,
                interleave_bytes,
                swizzle_bytes,
            )?;
        }
        let address_aligned = |alignment: u64| base_address.is_multiple_of(alignment);
        if let Some(bytes) = interleave_bytes {
            if swizzle_bytes == Some(96) {
                return Err(OpError::message("96B swizzle does not support interleave"));
            }
            if !matches!(bytes, 16 | 32) || rank < 3 {
                return Err(OpError::message(
                    "TensorMap interleave requires rank 3..=5 and 16B or 32B slices",
                ));
            }
            if element_strides[0] == 0 {
                return Err(OpError::message(
                    "interleaved TensorMap axis-zero stride must be nonzero",
                ));
            }
            if bytes == 32
                && (swizzle_bytes != Some(32)
                    || !address_aligned(32)
                    || global_strides.iter().any(|s| s % 32 != 0))
            {
                return Err(OpError::message(
                    "32B interleave requires 32B swizzle, address and strides",
                ));
            }
        }
        // Without interleave hardware ignores axis zero, including its zero sentinel.
        if element_strides[0] > MAX_ELEMENT_STRIDE
            || element_strides[1..]
                .iter()
                .any(|stride| *stride == 0 || *stride > MAX_ELEMENT_STRIDE)
        {
            return Err(OpError::message(
                "TensorMap element strides must be in 1..=8 (axis zero may be zero when interleave is disabled)",
            ));
        }
        if interleave_bytes.is_none() {
            element_strides[0] = 1;
        }
        if global_strides
            .iter()
            .any(|stride| *stride == 0 || *stride as u128 >= MAX_GLOBAL_STRIDE || stride % 16 != 0)
        {
            return Err(OpError::message(
                "TensorMap global byte strides must be non-zero multiples of 16 below 2^40",
            ));
        }
        match (element_bits, fp4_shared_layout) {
            (4, Some(_)) | (6 | 8 | 16 | 32 | 64, None) => {}
            (4, None) => {
                return Err(OpError::message(
                    "FP4 TensorMap is missing its shared layout",
                ));
            }
            (_, Some(layout)) => {
                return Err(OpError::message(format!(
                    "non-FP4 TensorMap cannot use FP4 shared layout {layout:?}"
                )));
            }
            (_, None) => {
                return Err(OpError::message(format!(
                    "TensorMap has unsupported {element_bits}-bit elements"
                )));
            }
        }
        if fill_mode == TensorMapFillMode::OobNan && !element_type.supports_oob_nan() {
            return Err(OpError::message(
                "TensorMap OOB-NaN fill requires a 16/32/64-bit floating-point dtype",
            ));
        }
        if swizzle_bytes.is_some_and(|bytes| !matches!(bytes, 32 | 64 | 96 | 128)) {
            return Err(OpError::message(
                "TensorMap swizzle must be 32, 64, 96, or 128 bytes",
            ));
        }
        if !address_aligned(16) {
            return Err(OpError::message(
                "TensorMap global address must be 16-byte aligned",
            ));
        }
        if element_bits == 6 {
            if interleave_bytes.is_some() {
                return Err(OpError::message("U6 TensorMap does not support interleave"));
            }
            if !global_shape[0].is_multiple_of(128) || box_shape[0] != 128 {
                return Err(OpError::message(
                    "SM100 U6 TensorMap requires dimension zero in multiples of 128 and box zero 128",
                ));
            }
            if !address_aligned(32) || global_strides.iter().any(|stride| stride % 32 != 0) {
                return Err(OpError::message(
                    "SM100 U6 TensorMap address and strides must be 32-byte aligned",
                ));
            }
            if swizzle_bytes.is_some_and(|bytes| bytes != 128) {
                return Err(OpError::message(
                    "SM100 U6 TensorMap supports only none or 128B swizzle",
                ));
            }
        }
        match fp4_shared_layout {
            Some(Fp4SharedLayout::Align8Packed) => {
                if !global_shape[0].is_multiple_of(2) {
                    return Err(OpError::message(
                        "align8 packed FP4 TensorMap global dimension zero must be a multiple of 2",
                    ));
                }
            }
            Some(Fp4SharedLayout::Align16Padded) => {
                // Interleaved padded dimensions do not yet have a validated
                // unit contract; do not apply the non-interleaved 128 rule.
                if interleave_bytes.is_some() {
                    return Err(analysis_incomplete("tma_padded_fp4_interleave_unmodeled"));
                }
                if !global_shape[0].is_multiple_of(128) || box_shape[0] != 128 {
                    return Err(OpError::message(
                        "align16 padded FP4 TensorMap requires global dimension zero to be a multiple of 128 and box dimension zero to equal 128",
                    ));
                }
                if !address_aligned(32) || global_strides.iter().any(|stride| stride % 32 != 0) {
                    return Err(OpError::message(
                        "align16 padded FP4 TensorMap address and global strides must be 32-byte aligned",
                    ));
                }
                if swizzle_bytes.is_some_and(|bytes| bytes != 128) {
                    return Err(OpError::message(
                        "align16 padded FP4 TensorMap only supports 128B swizzle",
                    ));
                }
            }
            None => {}
        }

        let transfer_bits = interleave_bytes.map_or(element_bits, |bytes| bytes * 8);
        let inner_global_bytes = global_shape[0]
            .checked_mul(transfer_bits)
            .and_then(|bits| bits.checked_add(7))
            .map(|bits| bits / 8)
            .ok_or_else(|| OpError::message("TensorMap inner global size overflow"))?;
        let mut varying_outer_axes = global_strides
            .iter()
            .copied()
            .zip(global_shape.iter().copied().skip(1))
            .enumerate()
            .filter_map(|(axis, (stride, dimension))| {
                (dimension > 1).then_some((stride, dimension, axis))
            })
            .collect::<Vec<_>>();
        varying_outer_axes.sort_unstable_by_key(|(stride, _, axis)| (*stride, *axis));
        let mut occupied_span = inner_global_bytes;
        for (stride, dimension, axis) in varying_outer_axes {
            if stride < occupied_span {
                return Err(OpError::message(format!(
                    "TensorMap global stride {axis} is {stride} bytes, which overlaps the prior {occupied_span}-byte span"
                )));
            }
            occupied_span = (dimension - 1)
                .checked_mul(stride)
                .and_then(|axis_span| occupied_span.checked_add(axis_span))
                .ok_or_else(|| OpError::message("TensorMap global span overflow"))?;
        }
        let required_byte_len = global_strides
            .iter()
            .zip(global_shape.iter().skip(1))
            .try_fold(inner_global_bytes, |span, (stride, dimension)| {
                span.checked_add(
                    (dimension - 1)
                        .checked_mul(*stride)
                        .ok_or_else(|| OpError::message("TensorMap global span overflow"))?,
                )
                .ok_or_else(|| OpError::message("TensorMap global span overflow"))
            })?;
        if view_byte_len < required_byte_len {
            return Err(OpError::message(format!(
                "TensorMap requires {required_byte_len} global bytes, but its view has {view_byte_len}"
            )));
        }

        let descriptor_inner_bytes = box_shape[0]
            .checked_mul(transfer_bits)
            .and_then(|bits| bits.checked_add(7))
            .map(|bits| bits / 8)
            .ok_or_else(|| OpError::message("TensorMap inner box size overflow"))?;
        if descriptor_inner_bytes % 16 != 0 {
            return Err(OpError::message(
                "TensorMap inner box transfer size must be a multiple of 16 bytes without interleave",
            ));
        }
        let shared_inner_bytes =
            if element_bits == 6 || fp4_shared_layout == Some(Fp4SharedLayout::Align16Padded) {
                box_shape[0]
            } else {
                descriptor_inner_bytes
            };
        if let Some(swizzle_bytes) = swizzle_bytes.filter(|_| interleave_bytes.is_none()) {
            if shared_inner_bytes > swizzle_bytes {
                return Err(OpError::message(format!(
                    "TensorMap row uses {shared_inner_bytes} bytes, exceeding {swizzle_bytes}B swizzle"
                )));
            }
        }

        let mut traversal_shape = if im2col.is_some() {
            box_shape.clone()
        } else {
            box_shape
                .iter()
                .zip(&element_strides)
                .map(|(dimension, stride)| dimension.div_ceil(*stride))
                .collect::<Vec<_>>()
        };
        if interleave_bytes.is_some() && im2col.is_none() {
            // Interleaved descriptors traverse spatial axes and N; the
            // penultimate coordinate selects one channel slice.
            traversal_shape[rank - 2] = 1;
        }
        if interleave_bytes.is_some() && im2col.is_some() {
            // Each im2col pixel selects exactly one channel slice; hardware
            // ignores channels-per-pixel in an interleaved descriptor.
            traversal_shape[0] = 1;
        }
        // Im2col selects pixel origins at issue time; the ordinary template
        // remains the sole owner of channel packing and global OOB handling.
        let template_shape = if im2col.is_some() {
            let mut shape = vec![1; rank];
            shape[0] = traversal_shape[0];
            shape
        } else {
            traversal_shape.clone()
        };
        let transfer_template = Arc::new(TransferTemplate::compile(
            &template_shape,
            &element_strides,
            &global_strides,
            transfer_bits,
            fp4_shared_layout,
            swizzle_bytes,
            swizzle_atomicity,
        )?);
        let mut physical_global_shape = [1_usize; 5];
        physical_global_shape[..rank].copy_from_slice(&global_shape);
        let mut physical_global_strides = [0_usize; 4];
        physical_global_strides[..global_strides.len()].copy_from_slice(&global_strides);
        Ok(Self {
            global_shape,
            global_strides,
            physical_global_shape,
            physical_global_strides,
            box_shape,
            element_strides,
            traversal_shape,
            element_bits,
            interleave_bytes,
            element_type,
            fp4_shared_layout,
            swizzle_bytes,
            swizzle_atomicity,
            fill_mode,
            transfer_template,
            im2col,
        })
    }

    /// Number of tensor dimensions.
    pub const fn rank(&self) -> usize {
        self.global_shape.len()
    }

    /// Bits moved per traversed element (the slice width when interleaved).
    pub fn transfer_element_bits(&self) -> usize {
        self.interleave_bytes
            .map_or(self.element_bits, |bytes| bytes * 8)
    }

    /// Reject shared layouts that a transfer in this direction cannot use.
    /// The descriptor stays usable by cache hints and predicated-off copies.
    pub fn validate_swizzle_direction(&self, load: bool) -> OpResult<()> {
        if self.interleave_bytes == Some(16) && self.swizzle_bytes.is_some() {
            return Err(analysis_incomplete("tma_swizzled_16b_interleave_unmodeled"));
        }
        if self.swizzle_bytes.is_none() || self.swizzle_atomicity == SwizzleAtomicity::B16 {
            return Ok(());
        }
        if !load && self.swizzle_atomicity == SwizzleAtomicity::B32Flip8 {
            return Err(OpError::message(
                "8B flip is only valid for global-to-shared tensor copies",
            ));
        }
        if load && self.swizzle_atomicity == SwizzleAtomicity::B64 {
            if self.element_type == TensorMapElementType::U6
                || self.fp4_shared_layout == Some(Fp4SharedLayout::Align16Padded)
            {
                return Err(OpError::message(
                    "64B atomicity loads are invalid for U6 and padded FP4 TensorMaps",
                ));
            }
            // The independent SM100 load probe traps; store is GPU-validated.
            return Err(analysis_incomplete("tma_64b_atomicity_load_unmodeled"));
        }
        Ok(())
    }

    /// SM100 U6 maps require the inner box origin to be a multiple of 128 elements.
    pub fn validate_u6_origin(&self, origin: &[i64]) -> OpResult<()> {
        if self.element_bits == 6
            && origin
                .first()
                .is_some_and(|value| value.rem_euclid(128) != 0)
        {
            return Err(OpError::message(
                "SM100 U6 TensorMap inner origin must be a multiple of 128",
            ));
        }
        Ok(())
    }

    /// FP4 inner-origin alignment shared by tiled loads and gather4.
    pub(crate) fn fp4_origin_alignment(&self) -> OpResult<Option<i64>> {
        if self.transfer_element_bits() != 4 {
            return Ok(None);
        }
        match self.fp4_shared_layout {
            Some(Fp4SharedLayout::Align8Packed) => Ok(Some(2)),
            Some(Fp4SharedLayout::Align16Padded) => Ok(Some(128)),
            None => Err(OpError::message(
                "FP4 TensorMap is missing its shared layout",
            )),
        }
    }

    /// Transfer geometry of this map; errors when the rank metadata is inconsistent.
    pub fn geometry(&self) -> OpResult<TensorMapGeometry> {
        if self.global_shape.is_empty()
            || self.box_shape.len()
                != if self.im2col.is_some() {
                    2
                } else {
                    self.rank()
                }
            || self.global_shape.len() != self.element_strides.len()
            || self.global_strides.len() + 1 != self.global_shape.len()
        {
            return Err(OpError::message("TensorMap rank metadata is inconsistent"));
        }
        tensor_map_geometry_from_metadata(
            &self.traversal_shape,
            self.transfer_element_bits(),
            self.fp4_shared_layout,
        )
    }

    /// Box coordinates (axis 0 = 0) of outer row `linear`, axis 1 fastest.
    pub fn outer_coordinates(&self, mut linear: usize) -> Vec<usize> {
        let mut coordinates = vec![0_usize; self.traversal_shape.len()];
        for (axis, coordinate) in coordinates.iter_mut().enumerate().skip(1) {
            *coordinate = linear % self.traversal_shape[axis];
            linear /= self.traversal_shape[axis];
        }
        coordinates
    }

    /// Global coordinates of box element (`inner_element`, `outer`) from `origin`, scaled by
    /// the element strides. Errors on a rank mismatch or overflow.
    pub fn global_coordinates(
        &self,
        origin: &[i64],
        inner_element: usize,
        outer: &[usize],
    ) -> OpResult<Vec<i64>> {
        if origin.len() != self.global_shape.len() {
            return Err(OpError::message(format!(
                "TensorMap expected {} coordinates, got {}",
                self.global_shape.len(),
                origin.len()
            )));
        }
        let mut result = vec![0_i64; origin.len()];
        self.global_coordinates_into(origin, inner_element, outer, &mut result)?;
        Ok(result)
    }

    /// [`Self::global_coordinates`] into a caller-provided buffer of rank length.
    pub fn global_coordinates_into(
        &self,
        origin: &[i64],
        inner_element: usize,
        outer: &[usize],
        result: &mut [i64],
    ) -> OpResult<()> {
        if origin.len() != self.global_shape.len() {
            return Err(OpError::message(format!(
                "TensorMap expected {} coordinates, got {}",
                self.global_shape.len(),
                origin.len()
            )));
        }
        if result.len() != origin.len() {
            return Err(OpError::message(format!(
                "TensorMap coordinate scratch has {} entries, expected {}",
                result.len(),
                origin.len()
            )));
        }
        for axis in 0..origin.len() {
            let local = if axis == 0 {
                inner_element
            } else {
                outer[axis]
            };
            let delta = local
                .checked_mul(self.element_strides[axis])
                .and_then(|value| i64::try_from(value).ok())
                .ok_or_else(|| OpError::message("TensorMap coordinate delta overflow"))?;
            result[axis] = origin[axis]
                .checked_add(delta)
                .ok_or_else(|| OpError::message("TensorMap coordinate overflow"))?;
        }
        Ok(())
    }

    /// TMA OOB predicate: every coordinate lies in `0..global_shape[axis]`.
    pub fn coordinates_in_bounds(&self, coordinates: &[i64]) -> bool {
        coordinates
            .iter()
            .zip(&self.global_shape)
            .all(|(coordinate, extent)| {
                *coordinate >= 0 && usize::try_from(*coordinate).is_ok_and(|value| value < *extent)
            })
    }

    /// `(byte offset, bit shift)` of an in-bounds element relative to the view.
    pub fn global_byte_offset(&self, coordinates: &[i64]) -> OpResult<(usize, usize)> {
        if !self.coordinates_in_bounds(coordinates) {
            return Err(OpError::message(
                "TensorMap coordinate is outside global shape",
            ));
        }
        let inner = usize::try_from(coordinates[0])
            .map_err(|_| OpError::message("negative TensorMap inner coordinate"))?;
        let inner_bits = inner
            .checked_mul(self.transfer_element_bits())
            .ok_or_else(|| OpError::message("TensorMap inner bit offset overflow"))?;
        let mut byte_offset = inner_bits / 8;
        for (axis, coordinate) in coordinates.iter().enumerate().skip(1) {
            let coordinate = usize::try_from(*coordinate)
                .map_err(|_| OpError::message("negative TensorMap coordinate"))?;
            byte_offset = byte_offset
                .checked_add(
                    coordinate
                        .checked_mul(self.global_strides[axis - 1])
                        .ok_or_else(|| OpError::message("TensorMap stride offset overflow"))?,
                )
                .ok_or_else(|| OpError::message("TensorMap byte offset overflow"))?;
        }
        Ok((byte_offset, inner_bits % 8))
    }

    /// Shared byte offset of a box position under this map's swizzle.
    pub fn shared_byte_offset(
        &self,
        outer_linear: usize,
        inner_byte: usize,
        inner_row_bytes: usize,
        absolute_base: usize,
    ) -> OpResult<usize> {
        shared_byte_offset(
            self.swizzle_bytes,
            self.swizzle_atomicity,
            outer_linear,
            inner_byte,
            inner_row_bytes,
            absolute_base,
        )
    }

    /// The descriptor image of this layout at `(allocation_id, base_byte_offset)`.
    pub fn to_image(&self, allocation_id: u64, base_byte_offset: usize) -> TensorMapImage {
        let mut box_shape = [1_usize; 5];
        box_shape[..self.box_shape.len()].copy_from_slice(&self.box_shape);
        let mut element_strides = [1_usize; 5];
        element_strides[..self.rank()].copy_from_slice(&self.element_strides);
        TensorMapImage {
            allocation_id,
            base_byte_offset,
            host_address: false,
            rank: self.rank(),
            physical_global_shape: self.physical_global_shape,
            physical_global_strides: self.physical_global_strides,
            box_shape,
            element_strides,
            element_type: self.element_type,
            interleave_bytes: self.interleave_bytes,
            fp4_shared_layout: self.fp4_shared_layout,
            swizzle_bytes: self.swizzle_bytes,
            swizzle_atomicity: self.swizzle_atomicity,
            fill_mode: self.fill_mode,
            im2col: self.im2col.clone(),
        }
    }
}

impl TensorMapImage {
    /// Materialize the image into a validated layout. `allocation_byte_len`
    /// is the full length of allocation `allocation_id`; `allocation_base_address`
    /// is its absolute address (0 when unobserved). The resulting view starts
    /// at `base_byte_offset` and extends to the end of the allocation.
    pub fn materialize(
        &self,
        expected_rank: usize,
        allocation_byte_len: usize,
        allocation_base_address: u64,
    ) -> OpResult<TensorMapLayout> {
        if self.host_address {
            return Err(OpError::message(
                "NumSim TensorMap host address was not relocated before use",
            ));
        }
        if expected_rank != self.rank {
            return Err(OpError::message(format!(
                "raw TMA rank {expected_rank} disagrees with TensorMap image rank {}",
                self.rank
            )));
        }
        if self.base_byte_offset > allocation_byte_len {
            return Err(OpError::message(format!(
                "TensorMap base byte offset {} exceeds allocation length {}",
                self.base_byte_offset, allocation_byte_len
            )));
        }
        let global_shape = self.physical_global_shape[..expected_rank].to_vec();
        let mut global_strides =
            self.physical_global_strides[..expected_rank.saturating_sub(1)].to_vec();
        for (axis, stride) in global_strides.iter_mut().enumerate() {
            if *stride == 0 && global_shape[axis + 1] == 1 {
                *stride = 16;
            }
        }
        TensorMapLayout::new(
            TensorMapSpec {
                global_shape,
                global_strides,
                box_shape: self.box_shape[..if self.im2col.is_some() {
                    2
                } else {
                    expected_rank
                }]
                    .to_vec(),
                element_strides: self.element_strides[..expected_rank].to_vec(),
                element_bits: self.element_type.bits(),
                element_type: self.element_type,
                fp4_shared_layout: self.fp4_shared_layout,
                swizzle_bytes: self.swizzle_bytes,
                swizzle_atomicity: self.swizzle_atomicity,
                fill_mode: self.fill_mode,
                interleave_bytes: self.interleave_bytes,
                im2col: self.im2col.clone(),
            },
            allocation_base_address.wrapping_add(self.base_byte_offset as u64),
            allocation_byte_len - self.base_byte_offset,
        )
    }
}

#[cfg(test)]
#[path = "tensor_map_tests.rs"]
pub(crate) mod tests;
