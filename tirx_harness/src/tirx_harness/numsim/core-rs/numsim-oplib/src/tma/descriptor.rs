//! NumSim's private, byte-addressable TensorMap (CUtensorMap) image: element
//! types, encode/decode, canonical-candidate recognition, `tensormap.replace`
//! field updates, TMA attribute overrides, and reduction-op resolution.
//!
//! Legacy sources: `engine-rs/src/runtime/tensor_map.rs`
//! (`RuntimeTensorMapImage`, `TensorMapElementType`, `RawTmaReductionOp`,
//! `RuntimeTensorMap::with_overrides`) and
//! `engine-rs/src/runtime/tensor_map_registry.rs` (`descriptor_address`).
//!
//! NVIDIA deliberately leaves the hardware CUtensorMap layout opaque. NumSim
//! keeps the architectural 128-byte object and ordering footprint, while its
//! supported TensorMap semantics are self-contained in the first 64 bytes
//! (80 for im2col). The remaining bytes are reserved and ignored by direct
//! decoding, while automatic descriptor discovery requires their canonical
//! zero value.

use std::fmt;

use super::swizzle::SwizzleAtomicity;
use crate::types::{OpError, OpResult};

/// Architectural CUtensorMap object size.
pub const TENSOR_MAP_DESCRIPTOR_BYTES: usize = 128;
/// Bytes of a tiled NumSim TensorMap image (im2col images use 80).
pub const TENSOR_MAP_PAYLOAD_BYTES: usize = 64;
/// Format tag (byte 63) of an ordinary tiled image.
pub const TENSOR_MAP_MAGIC: u8 = 0xA7;
const TENSOR_MAP_FLAG_MAGIC: u8 = 0x80;
pub const MAX_GLOBAL_DIMENSION: u128 = 1_u128 << 32;
pub const MAX_GLOBAL_STRIDE: u128 = 1_u128 << 40;
pub const MAX_BOX_DIMENSION: usize = 256;
pub const MAX_ELEMENT_STRIDE: usize = 8;

/// Whether a format tag byte carries the TensorMap magic (A5/A6/A7 plus
/// im2col and atomicity bits).
pub fn tensor_map_has_magic(byte: u8) -> bool {
    matches!(byte & !0x58, 0xA5..=0xA7)
}

/// Payload length implied by a format tag byte.
pub fn tensor_map_payload_bytes(tag: u8) -> usize {
    if tag & 0x10 != 0 {
        80
    } else {
        TENSOR_MAP_PAYLOAD_BYTES
    }
}

/// Error text for a missing analysis contract (legacy
/// `EngineError::analysis_incomplete(kind)` Display).
pub fn analysis_incomplete(kind: &'static str) -> OpError {
    OpError::message(format!("{kind} requires an unmodeled analysis contract"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Fp4SharedLayout {
    Align8Packed,
    Align16Padded,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TensorMapFillMode {
    Zero,
    /// OOB elements read as the PTX canonical NaN (`0x7ff7` per 16 bits),
    /// a.k.a. "NaN request zero FMA".
    OobNan,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TensorMapElementType {
    Float4E2M1Fn,
    U6,
    Bool,
    I8,
    U8,
    Float8E4M3Fn,
    Float8E8M0Fnu,
    I16,
    U16,
    F16,
    Bf16,
    I32,
    U32,
    F32,
    Tf32,
    F64,
    I64,
    U64,
    U32x2,
    F32Ftz,
    Tf32Ftz,
}

impl TensorMapElementType {
    pub const fn bits(self) -> usize {
        match self {
            Self::Float4E2M1Fn => 4,
            Self::U6 => 6,
            Self::Bool | Self::I8 | Self::U8 | Self::Float8E4M3Fn | Self::Float8E8M0Fnu => 8,
            Self::I16 | Self::U16 | Self::F16 | Self::Bf16 => 16,
            Self::I32 | Self::U32 | Self::F32 | Self::Tf32 | Self::F32Ftz | Self::Tf32Ftz => 32,
            Self::F64 | Self::I64 | Self::U64 | Self::U32x2 => 64,
        }
    }

    pub const fn supports_oob_nan(self) -> bool {
        matches!(
            self,
            Self::F16
                | Self::Bf16
                | Self::F32
                | Self::Tf32
                | Self::F64
                | Self::F32Ftz
                | Self::Tf32Ftz
        )
    }

    /// NumSim private image code (bits 3..8 of byte 59).
    pub const fn image_code(self) -> u8 {
        match self {
            Self::Float4E2M1Fn => 0,
            Self::Bool => 1,
            Self::I8 => 2,
            Self::U8 => 3,
            Self::Float8E4M3Fn => 4,
            Self::Float8E8M0Fnu => 5,
            Self::I16 => 6,
            Self::U16 => 7,
            Self::F16 => 8,
            Self::Bf16 => 9,
            Self::I32 => 10,
            Self::U32 => 11,
            Self::F32 => 12,
            Self::Tf32 => 13,
            Self::F64 => 14,
            Self::I64 => 15,
            Self::U64 => 16,
            Self::U32x2 => 17,
            Self::F32Ftz => 18,
            Self::Tf32Ftz => 19,
            Self::U6 => 20,
        }
    }

    pub fn from_image_code(code: u8) -> OpResult<Self> {
        Ok(match code {
            0 => Self::Float4E2M1Fn,
            1 => Self::Bool,
            2 => Self::I8,
            3 => Self::U8,
            4 => Self::Float8E4M3Fn,
            5 => Self::Float8E8M0Fnu,
            6 => Self::I16,
            7 => Self::U16,
            8 => Self::F16,
            9 => Self::Bf16,
            10 => Self::I32,
            11 => Self::U32,
            12 => Self::F32,
            13 => Self::Tf32,
            14 => Self::F64,
            15 => Self::I64,
            16 => Self::U64,
            17 => Self::U32x2,
            18 => Self::F32Ftz,
            19 => Self::Tf32Ftz,
            20 => Self::U6,
            _ => {
                return Err(OpError::message(format!(
                    "NumSim TensorMap image has unknown element-type code {code}"
                )));
            }
        })
    }

    /// PTX `tensormap.replace.elemtype` field encoding (not CUtensorMapDataType
    /// nor the private image code).
    pub fn from_ptx_elemtype(value: usize) -> OpResult<Self> {
        Ok(match value {
            0 => Self::U8,
            1 => Self::U16,
            2 => Self::U32,
            3 => Self::I32,
            4 => Self::U64,
            5 => Self::I64,
            6 => Self::F16,
            7 => Self::F32,
            8 => Self::F32Ftz,
            9 => Self::F64,
            10 => Self::Bf16,
            11 => Self::Tf32,
            12 => Self::Tf32Ftz,
            _ => {
                return Err(OpError::message(format!(
                    "TensorMap PTX element type {value} is not modeled"
                )))
            }
        })
    }
}

impl fmt::Display for TensorMapElementType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Float4E2M1Fn => "float4_e2m1fn",
            Self::U6 => "uint6",
            Self::Bool => "bool",
            Self::I8 => "int8",
            Self::U8 => "uint8",
            Self::Float8E4M3Fn => "float8_e4m3fn",
            Self::Float8E8M0Fnu => "float8_e8m0fnu",
            Self::I16 => "int16",
            Self::U16 => "uint16",
            Self::F16 => "float16",
            Self::Bf16 => "bfloat16",
            Self::I32 => "int32",
            Self::U32 => "uint32",
            Self::F32 => "float32",
            Self::Tf32 => "tf32",
            Self::F64 => "float64",
            Self::I64 => "int64",
            Self::U64 => "uint64",
            Self::U32x2 => "uint32x2",
            Self::F32Ftz => "float32_ftz",
            Self::Tf32Ftz => "tf32_ftz",
        })
    }
}

/// Im2col adds spatial bounds to a two-dimensional (channels, pixels) box.
/// Channels/pixels remain owned by the box shape, not duplicated here.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorMapIm2col {
    pub lower: [i16; 3],
    pub upper: [i16; 3],
    pub wide: bool,
}

/// Decoded NumSim TensorMap image (all fields, including inactive axes).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorMapImage {
    /// Allocation id, or a host address when `host_address` is set.
    pub allocation_id: u64,
    pub base_byte_offset: usize,
    pub host_address: bool,
    pub rank: usize,
    pub physical_global_shape: [usize; 5],
    pub physical_global_strides: [usize; 4],
    pub box_shape: [usize; 5],
    pub element_strides: [usize; 5],
    pub element_type: TensorMapElementType,
    pub interleave_bytes: Option<usize>,
    pub fp4_shared_layout: Option<Fp4SharedLayout>,
    pub swizzle_bytes: Option<usize>,
    pub swizzle_atomicity: SwizzleAtomicity,
    pub fill_mode: TensorMapFillMode,
    pub im2col: Option<TensorMapIm2col>,
}

impl TensorMapImage {
    pub fn encode(&self) -> OpResult<Vec<u8>> {
        if self.rank == 0 || self.rank > 5 {
            return Err(OpError::message(format!(
                "NumSim TensorMap image rank must be in 1..=5, got {}",
                self.rank
            )));
        }
        let base_byte_offset = u64::try_from(self.base_byte_offset)
            .map_err(|_| OpError::message("TensorMap base byte offset does not fit u64"))?;
        let mut bytes = vec![
            0_u8;
            if self.im2col.is_some() {
                80
            } else {
                TENSOR_MAP_PAYLOAD_BYTES
            }
        ];
        bytes[0..8].copy_from_slice(&self.allocation_id.to_le_bytes());
        bytes[8..16].copy_from_slice(&base_byte_offset.to_le_bytes());

        for (axis, dimension) in self.physical_global_shape.iter().copied().enumerate() {
            if dimension == 0 || dimension as u128 > MAX_GLOBAL_DIMENSION {
                return Err(OpError::message(format!(
                    "TensorMap global dimension {axis}={dimension} is outside 1..=2^32"
                )));
            }
            let encoded = if dimension as u128 == MAX_GLOBAL_DIMENSION {
                0_u32
            } else {
                u32::try_from(dimension).expect("bounded TensorMap dimension fits u32")
            };
            let start = 16 + axis * 4;
            bytes[start..start + 4].copy_from_slice(&encoded.to_le_bytes());
        }

        let mut encoded_strides = [0_u64; 4];
        for (axis, stride) in self.physical_global_strides.iter().copied().enumerate() {
            if stride as u128 >= MAX_GLOBAL_STRIDE || (stride != 0 && stride % 16 != 0) {
                return Err(OpError::message(format!(
                    "TensorMap global stride {axis}={stride} is not zero or a 16-byte multiple below 2^40"
                )));
            }
            encoded_strides[axis] =
                u64::try_from(stride as u128 >> 4).expect("bounded TensorMap stride units fit u64");
        }
        for pair in 0..2 {
            let packed = u128::from(encoded_strides[pair * 2])
                | (u128::from(encoded_strides[pair * 2 + 1]) << 36);
            let start = 36 + pair * 9;
            bytes[start..start + 9].copy_from_slice(&packed.to_le_bytes()[..9]);
        }

        for (axis, dimension) in self.box_shape.iter().copied().enumerate() {
            let limit = if self.im2col.is_some() && axis == 1 {
                1024
            } else {
                MAX_BOX_DIMENSION
            };
            if dimension == 0 || dimension > limit {
                return Err(OpError::message(format!(
                    "TensorMap box dimension {axis}={dimension} is outside 1..={limit}"
                )));
            }
            bytes[54 + axis] = ((dimension - 1) & 255) as u8;
        }

        bytes[59] = u8::try_from(self.rank).expect("bounded TensorMap rank fits u8")
            | (self.element_type.image_code() << 3);
        let fp4 = match self.fp4_shared_layout {
            None => 0_u8,
            Some(Fp4SharedLayout::Align8Packed) => 1,
            Some(Fp4SharedLayout::Align16Padded) => 2,
        };
        let swizzle = match self.swizzle_bytes {
            None => 0_u8,
            Some(32) => 1,
            Some(64) => 2,
            Some(128) => 3,
            Some(96) => 4,
            Some(value) => {
                return Err(OpError::message(format!(
                    "NumSim TensorMap image cannot encode {value}B swizzle"
                )));
            }
        };
        bytes[60] = fp4
            | ((swizzle & 3) << 2)
            | ((swizzle & 4) << 4)
            | (u8::from(self.fill_mode == TensorMapFillMode::OobNan) << 4)
            | (u8::from(self.host_address) << 5)
            | TENSOR_MAP_FLAG_MAGIC;

        let mut encoded_element_strides = 0_u16;
        for (axis, stride) in self.element_strides.iter().copied().enumerate() {
            if stride == 0 || stride > MAX_ELEMENT_STRIDE {
                return Err(OpError::message(format!(
                    "TensorMap element stride {axis}={stride} is outside 1..=8"
                )));
            }
            encoded_element_strides |= u16::try_from(stride - 1)
                .expect("bounded TensorMap element stride minus one fits u16")
                << (axis * 3);
        }
        bytes[61..63].copy_from_slice(&encoded_element_strides.to_le_bytes());
        // The format tag retains A7 for ordinary tiled maps; A6/A5 encode
        // 16B/32B interleave without competing metadata in the reserved tail.
        bytes[63] = match self.interleave_bytes {
            None => TENSOR_MAP_MAGIC,
            Some(16) => 0xA6,
            Some(32) => 0xA5,
            _ => return Err(OpError::message("TensorMap interleave must be 16B or 32B")),
        };
        bytes[63] |= self.swizzle_atomicity.tag_bits();
        if let Some(im2col) = &self.im2col {
            bytes[63] |= 0x10;
            for (i, value) in im2col.lower.iter().chain(&im2col.upper).enumerate() {
                bytes[64 + i * 2..66 + i * 2].copy_from_slice(&value.to_le_bytes());
            }
            bytes[76] = ((self.box_shape[1] - 1) >> 8) as u8 | (u8::from(im2col.wide) << 2);
        }
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8]) -> OpResult<Self> {
        if bytes.len() < TENSOR_MAP_PAYLOAD_BYTES
            || bytes.len() != tensor_map_payload_bytes(bytes[63])
        {
            return Err(OpError::message(format!(
                "NumSim TensorMap payload length disagrees with format tag: {} bytes",
                bytes.len()
            )));
        }
        if !tensor_map_has_magic(bytes[63]) {
            return Err(OpError::message("NumSim TensorMap image has invalid magic"));
        }
        let rank = usize::from(bytes[59] & 0b111);
        if rank == 0 || rank > 5 {
            return Err(OpError::message(format!(
                "NumSim TensorMap image rank must be in 1..=5, got {rank}"
            )));
        }
        let element_type = TensorMapElementType::from_image_code(bytes[59] >> 3)?;
        let interleave_bytes = match bytes[63] & !0x58 {
            0xA6 => Some(16),
            0xA5 => Some(32),
            _ => None,
        };
        let flags = bytes[60];
        if flags & 0x80 != TENSOR_MAP_FLAG_MAGIC {
            return Err(OpError::message(
                "NumSim TensorMap image has invalid flag magic",
            ));
        }
        let fp4_shared_layout = match flags & 0b11 {
            0 => None,
            1 => Some(Fp4SharedLayout::Align8Packed),
            2 => Some(Fp4SharedLayout::Align16Padded),
            value => {
                return Err(OpError::message(format!(
                    "NumSim TensorMap image has unknown FP4 layout code {value}"
                )));
            }
        };
        let swizzle_bytes = match ((flags >> 2) & 0b11) | ((flags >> 4) & 4) {
            0 => None,
            1 => Some(32),
            2 => Some(64),
            3 => Some(128),
            4 => Some(96),
            _ => {
                return Err(OpError::message(
                    "NumSim TensorMap image has invalid swizzle code",
                ))
            }
        };
        let fill_mode = if flags & (1 << 4) == 0 {
            TensorMapFillMode::Zero
        } else {
            TensorMapFillMode::OobNan
        };
        let host_address = flags & (1 << 5) != 0;
        let allocation_id = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        let base_byte_offset =
            usize::try_from(u64::from_le_bytes(bytes[8..16].try_into().unwrap()))
                .map_err(|_| OpError::message("TensorMap base byte offset does not fit usize"))?;
        let mut physical_global_shape = [0_usize; 5];
        for (axis, dimension) in physical_global_shape.iter_mut().enumerate() {
            let start = 16 + axis * 4;
            let encoded = u32::from_le_bytes(bytes[start..start + 4].try_into().unwrap());
            let decoded = if encoded == 0 {
                MAX_GLOBAL_DIMENSION
            } else {
                u128::from(encoded)
            };
            *dimension = usize::try_from(decoded).map_err(|_| {
                OpError::message(format!(
                    "TensorMap global dimension {axis} does not fit usize"
                ))
            })?;
        }
        let mut physical_global_strides = [0_usize; 4];
        let stride_mask = (1_u128 << 36) - 1;
        for pair in 0..2 {
            let start = 36 + pair * 9;
            let mut packed_bytes = [0_u8; 16];
            packed_bytes[..9].copy_from_slice(&bytes[start..start + 9]);
            let packed = u128::from_le_bytes(packed_bytes);
            for item in 0..2 {
                let axis = pair * 2 + item;
                let units = (packed >> (item * 36)) & stride_mask;
                physical_global_strides[axis] = usize::try_from(units << 4).map_err(|_| {
                    OpError::message(format!("TensorMap global stride {axis} does not fit usize"))
                })?;
            }
        }
        let mut box_shape = [1_usize; 5];
        for (axis, dimension) in box_shape.iter_mut().enumerate() {
            *dimension = usize::from(bytes[54 + axis]) + 1;
        }
        let im2col = if bytes[63] & 0x10 != 0 {
            if rank < 3
                || bytes[76] & !7 != 0
                || bytes[77..80].iter().any(|b| *b != 0)
                || box_shape[2..].iter().any(|dimension| *dimension != 1)
            {
                return Err(OpError::message("invalid im2col TensorMap extension"));
            }
            box_shape[1] += usize::from(bytes[76] & 3) << 8;
            Some(TensorMapIm2col {
                lower: std::array::from_fn(|i| {
                    i16::from_le_bytes(bytes[64 + i * 2..66 + i * 2].try_into().unwrap())
                }),
                upper: std::array::from_fn(|i| {
                    i16::from_le_bytes(bytes[70 + i * 2..72 + i * 2].try_into().unwrap())
                }),
                wide: bytes[76] & 4 != 0,
            })
        } else {
            None
        };
        let encoded_element_strides = u16::from_le_bytes(bytes[61..63].try_into().unwrap());
        let mut element_strides = [1_usize; 5];
        for (axis, stride) in element_strides.iter_mut().enumerate() {
            *stride = usize::from((encoded_element_strides >> (axis * 3)) & 0b111) + 1;
        }
        Ok(Self {
            allocation_id,
            base_byte_offset,
            host_address,
            rank,
            physical_global_shape,
            physical_global_strides,
            box_shape,
            element_strides,
            element_type,
            interleave_bytes,
            fp4_shared_layout,
            swizzle_bytes,
            swizzle_atomicity: SwizzleAtomicity::from_tag(bytes[63]),
            fill_mode,
            im2col,
        })
    }

    /// Decode the payload prefix of a descriptor (the bytes that the format
    /// tag at byte 63 says belong to the image). The tail is ignored.
    pub fn decode_descriptor(descriptor: &[u8]) -> OpResult<Self> {
        if descriptor.len() < TENSOR_MAP_DESCRIPTOR_BYTES {
            return Err(OpError::message(format!(
                "TensorMap descriptor requires {TENSOR_MAP_DESCRIPTOR_BYTES} bytes, but only {} remain",
                descriptor.len()
            )));
        }
        Self::decode(&descriptor[..tensor_map_payload_bytes(descriptor[63])])
    }

    /// Write the encoded image into the first bytes of a 128-byte descriptor,
    /// leaving the reserved tail untouched.
    pub fn write_descriptor(&self, descriptor: &mut [u8]) -> OpResult<()> {
        if descriptor.len() < TENSOR_MAP_DESCRIPTOR_BYTES {
            return Err(OpError::message(format!(
                "TensorMap descriptor requires {TENSOR_MAP_DESCRIPTOR_BYTES} bytes, but only {} remain",
                descriptor.len()
            )));
        }
        let encoded = self.encode()?;
        descriptor[..encoded.len()].copy_from_slice(&encoded);
        Ok(())
    }

    /// Recognize a descriptor embedded in an otherwise ordinary byte buffer.
    ///
    /// Discovery accepts only the exact canonical representation so arbitrary
    /// buffer contents cannot become TensorMaps by sharing the magic byte and
    /// a few valid-looking fields. Returns `None` for anything else.
    pub fn decode_candidate(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < TENSOR_MAP_DESCRIPTOR_BYTES {
            return None;
        }
        let bytes = &bytes[..TENSOR_MAP_DESCRIPTOR_BYTES];
        let payload_bytes = tensor_map_payload_bytes(bytes[63]);
        if bytes[payload_bytes..].iter().any(|byte| *byte != 0) {
            return None;
        }
        let image = Self::decode(&bytes[..payload_bytes]).ok()?;
        // Decoding and re-encoding alone preserves arbitrary inactive fields,
        // so an ordinary scratch buffer could otherwise look like a descriptor.
        if image.physical_global_shape[image.rank..]
            .iter()
            .any(|dimension| *dimension != 1)
            || image.physical_global_strides[image.rank - 1..]
                .iter()
                .any(|stride| *stride != 0)
            || image.box_shape[image.rank..]
                .iter()
                .any(|dimension| *dimension != 1)
            || image.element_strides[image.rank..]
                .iter()
                .any(|stride| *stride != 1)
            || (image.element_type == TensorMapElementType::Float4E2M1Fn)
                != image.fp4_shared_layout.is_some()
            || (image.interleave_bytes.is_some() && image.rank < 3)
            || (image.swizzle_bytes.is_some_and(|width| width != 128)
                && image.swizzle_atomicity != SwizzleAtomicity::B16)
            || (image.host_address && (image.allocation_id == 0 || image.base_byte_offset != 0))
        {
            return None;
        }
        let canonical = image.encode().ok()?;
        (canonical.as_slice() == &bytes[..payload_bytes]).then_some(image)
    }

    pub fn relocate(&mut self, allocation_id: u64, byte_offset: usize) {
        self.allocation_id = allocation_id;
        self.base_byte_offset = byte_offset;
        self.host_address = false;
    }

    pub fn restore_host_address(&mut self, address: u64) -> OpResult<()> {
        self.allocation_id = address
            .checked_add(
                u64::try_from(self.base_byte_offset)
                    .map_err(|_| OpError::message("TensorMap base byte offset does not fit u64"))?,
            )
            .ok_or_else(|| OpError::message("TensorMap host address overflow"))?;
        self.base_byte_offset = 0;
        self.host_address = true;
        Ok(())
    }

    pub fn replace_global_dimension(&mut self, index: usize, value: usize) -> OpResult<()> {
        if index >= self.physical_global_shape.len() || value as u128 >= MAX_GLOBAL_DIMENSION {
            return Err(OpError::message(format!(
                "TensorMap global dimension field {index}={value} is outside descriptor ranges"
            )));
        }
        self.physical_global_shape[index] = if value == 0 {
            usize::try_from(MAX_GLOBAL_DIMENSION)
                .map_err(|_| OpError::message("TensorMap dimension 2^32 does not fit usize"))?
        } else {
            value
        };
        Ok(())
    }

    pub fn replace_global_stride(&mut self, index: usize, value: usize) -> OpResult<()> {
        if index >= self.physical_global_strides.len()
            || value as u128 >= MAX_GLOBAL_STRIDE
            || (value != 0 && value % 16 != 0)
        {
            return Err(OpError::message(format!(
                "TensorMap global stride field {index}={value} is outside descriptor ranges"
            )));
        }
        self.physical_global_strides[index] = value;
        Ok(())
    }

    /// `tensormap.replace` with PTX field encodings. Cross-field legality is
    /// checked when the updated image is materialized: several replace
    /// instructions may be needed to build a valid new shape.
    pub fn replace_field(&mut self, field: &str, index: Option<usize>, value: usize) -> OpResult<()> {
        match (field, index) {
            ("global_dim", Some(index)) => return self.replace_global_dimension(index, value),
            ("global_stride", Some(index)) => return self.replace_global_stride(index, value),
            ("box_dim" | "element_stride", Some(index)) => {
                let (dimensions, limit) = if field == "box_dim" {
                    (&mut self.box_shape, MAX_BOX_DIMENSION)
                } else {
                    (&mut self.element_strides, MAX_ELEMENT_STRIDE)
                };
                if index >= dimensions.len() || value == 0 || value > limit {
                    return Err(OpError::message(format!(
                        "TensorMap {field}[{index}]={value} is outside descriptor ranges (ord 0..4, value 1..={limit})"
                    )));
                }
                dimensions[index] = value;
            }
            ("rank", None) if value < 5 => self.rank = value + 1,
            ("interleave_layout", None) if value <= 2 => {
                self.interleave_bytes = match value {
                    0 => None,
                    1 => Some(16),
                    _ => Some(32),
                };
            }
            ("elemtype", None) => {
                self.element_type = TensorMapElementType::from_ptx_elemtype(value)?;
                // Packed FP4 layout belongs to the old element type.
                self.fp4_shared_layout = None;
            }
            ("fill_mode", None) if value <= 1 => {
                self.fill_mode = if value == 0 {
                    TensorMapFillMode::Zero
                } else {
                    TensorMapFillMode::OobNan
                };
            }
            ("swizzle_mode", None) if value <= 4 => {
                self.swizzle_bytes = match value {
                    0 => None,
                    4 => Some(96),
                    _ => Some(16 << value),
                };
            }
            _ => {
                return Err(OpError::message(format!(
                    "TensorMap replacement {field}[{index:?}]={value} is not modeled"
                )))
            }
        }
        Ok(())
    }

    /// Instruction-local dimension/stride overrides of
    /// `cp.async.bulk.tensor ... .override_global_dim[_stride]` (the
    /// address part is applied by the caller via [`Self::relocate`] after
    /// [`validate_override_address`]). The source descriptor is not mutated:
    /// callers apply this to a copy.
    pub fn apply_overrides(
        &mut self,
        rank: usize,
        dimensions: &[i64],
        lower_strides: &[i64],
        upper_strides: i64,
        coordinates: &[i64],
    ) -> OpResult<()> {
        if dimensions.is_empty() {
            if !lower_strides.is_empty() || upper_strides != 0 {
                return Err(OpError::message(
                    "TMA stride override requires dimension override",
                ));
            }
            return Ok(());
        }
        if dimensions.len() != rank || lower_strides.len() != rank - 1 || coordinates.len() != rank
        {
            return Err(OpError::message("TMA override rank/operand counts disagree"));
        }
        if coordinates.iter().any(|coordinate| *coordinate != 0) {
            return Err(OpError::message(
                "TMA attribute override requires zero coordinates",
            ));
        }
        for (axis, &dimension) in dimensions.iter().enumerate() {
            if !(1..=255).contains(&dimension) {
                return Err(OpError::message(
                    "TMA override dimension must be a nonzero 8-bit value",
                ));
            }
            self.replace_global_dimension(axis, dimension as usize)?;
        }
        let upper = u16::try_from(upper_strides)
            .map_err(|_| OpError::message("TMA upper strides must fit 16 bits"))?;
        if u32::from(upper) >> (4 * (rank - 1)) != 0 {
            return Err(OpError::message("TMA upper strides have nonzero unused bits"));
        }
        for (axis, &lower) in lower_strides.iter().enumerate() {
            let lower = u32::try_from(lower)
                .map_err(|_| OpError::message("TMA lower stride must fit 32 bits"))?;
            let high = u64::from((upper >> (4 * axis)) & 15);
            let stride = usize::try_from((u64::from(lower) | (high << 32)) << 4)
                .map_err(|_| OpError::message("TMA stride does not fit usize"))?;
            self.replace_global_stride(axis, stride)?;
        }
        Ok(())
    }
}

/// Override-address legality: 16-byte aligned (absolute address) with at
/// least 128 KiB of accessible memory behind it.
pub fn validate_override_address(absolute_address: u64, accessible_bytes: usize) -> OpResult<()> {
    if absolute_address % 16 != 0 {
        return Err(OpError::message(
            "TMA override address must be 16-byte aligned",
        ));
    }
    if accessible_bytes < 128 * 1024 {
        return Err(OpError::message(
            "TMA override address requires 128 KiB of accessible memory",
        ));
    }
    Ok(())
}

/// Descriptor-address legality used by the TensorMap registry and
/// `tensormap.replace`: 128 bytes available and 128-byte alignment.
pub fn validate_descriptor_address(
    allocation_id: u64,
    byte_offset: usize,
    remaining_bytes: usize,
) -> OpResult<()> {
    if remaining_bytes < TENSOR_MAP_DESCRIPTOR_BYTES {
        return Err(OpError::message(format!(
            "TensorMap descriptor requires {TENSOR_MAP_DESCRIPTOR_BYTES} bytes, but only {remaining_bytes} remain"
        )));
    }
    if !byte_offset.is_multiple_of(TENSOR_MAP_DESCRIPTOR_BYTES) {
        return Err(OpError::message(format!(
            "TensorMap descriptor address {allocation_id}:{byte_offset} must be 128-byte aligned"
        )));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RawTmaReductionOp {
    Add,
    Min,
    Max,
    Inc,
    Dec,
    And,
    Or,
    Xor,
}

/// Element reduction applied by `cp.reduce.async.bulk.tensor`; numerics are
/// `crate::atomic::BulkReduction::apply` (legacy `DeferredGlobalReduction`).
pub type TmaReduction = crate::atomic::BulkReduction;

impl RawTmaReductionOp {
    pub fn resolve(self, element_type: TensorMapElementType) -> OpResult<TmaReduction> {
        use RawTmaReductionOp as Op;
        use TensorMapElementType as T;
        use TmaReduction as Reduction;

        let reduction = match (self, element_type) {
            (Op::Add, T::U32) => Reduction::AddU32,
            (Op::Add, T::I32) => Reduction::AddI32,
            (Op::Add, T::U64) => Reduction::AddU64,
            (Op::Add, T::F32 | T::Tf32) => Reduction::AddF32,
            (Op::Add, T::F32Ftz | T::Tf32Ftz) => Reduction::AddF32Ftz,
            (Op::Add, T::F16) => Reduction::AddF16,
            (Op::Add, T::Bf16) => Reduction::AddBf16,
            (Op::Min, T::U32) => Reduction::MinU32,
            (Op::Min, T::I32) => Reduction::MinI32,
            (Op::Min, T::U64) => Reduction::MinU64,
            (Op::Min, T::I64) => Reduction::MinI64,
            (Op::Min, T::F16) => Reduction::MinF16,
            (Op::Min, T::Bf16) => Reduction::MinBf16,
            (Op::Max, T::U32) => Reduction::MaxU32,
            (Op::Max, T::I32) => Reduction::MaxI32,
            (Op::Max, T::U64) => Reduction::MaxU64,
            (Op::Max, T::I64) => Reduction::MaxI64,
            (Op::Max, T::F16) => Reduction::MaxF16,
            (Op::Max, T::Bf16) => Reduction::MaxBf16,
            (Op::Inc, T::U32) => Reduction::IncU32,
            (Op::Dec, T::U32) => Reduction::DecU32,
            (Op::And, element_type) if element_type.bits() == 32 => Reduction::AndB32,
            (Op::And, element_type) if element_type.bits() == 64 => Reduction::AndB64,
            (Op::Or, element_type) if element_type.bits() == 32 => Reduction::OrB32,
            (Op::Or, element_type) if element_type.bits() == 64 => Reduction::OrB64,
            (Op::Xor, element_type) if element_type.bits() == 32 => Reduction::XorB32,
            (Op::Xor, element_type) if element_type.bits() == 64 => Reduction::XorB64,
            _ => {
                return Err(OpError::message(format!(
                    "cp.reduce.async.bulk.tensor operation {self:?} is invalid for TensorMap dtype {element_type}"
                )));
            }
        };
        Ok(reduction)
    }
}

#[cfg(test)]
#[path = "descriptor_tests.rs"]
mod tests;
