//! Pure bit codecs.
//!
//! - Shared-memory address and tcgen runtime-instruction-descriptor codecs
//!   (legacy `engine-rs/src/instruction_codec.rs`).
//! - Storage-only scalar/packed-vector dtype ABI classification (legacy
//!   `numsim/dtype_abi.py`, which the frontend embeds from
//!   `dtype_registry.json`). Membership says nothing about arithmetic.
//! - Little-endian scalar byte transport lives in `crate::scalar::RuntimeScalar`.

use crate::types::Dtype;

pub const fn tcgen_runtime_instruction_descriptor(bits: u32, sf_id: u32) -> u32 {
    (bits & !0x6000_0030_u32) | sf_id.wrapping_shl(29) | sf_id.wrapping_shl(4)
}

pub const fn sm100_tma_2sm_mbarrier_address(address: u64) -> u32 {
    (address as u32) & 0xfeff_ffff_u32
}

// GB200/sm_100a device probes show that cluster shared addresses use the high
// byte for the CTA rank and the low 24 bits for the byte address. NumSim uses
// one fixed virtual generic-shared prefix for its integer address model.
pub const SHARED_CTA_RANK_SHIFT: u32 = 24;
pub const SHARED_BYTE_OFFSET_MASK: u32 = 0x00ff_ffff;
pub const DEFAULT_GENERIC_SHARED_PREFIX: u64 = 0x0000_fffe_0000_0000;
pub const GENERIC_ADDRESS_PREFIX_MASK: u64 = 0xffff_ffff_0000_0000;

pub const fn encode_shared_address(byte_offset: u32, cta_rank: u32) -> Option<u32> {
    if byte_offset > SHARED_BYTE_OFFSET_MASK || cta_rank > u8::MAX as u32 {
        return None;
    }
    Some(byte_offset | (cta_rank << SHARED_CTA_RANK_SHIFT))
}

pub const fn shared_address_byte_offset(address: u32) -> u32 {
    address & SHARED_BYTE_OFFSET_MASK
}

pub const fn shared_address_cta_rank(address: u32) -> u32 {
    address >> SHARED_CTA_RANK_SHIFT
}

pub const fn generic_shared_address(address: u32) -> u64 {
    DEFAULT_GENERIC_SHARED_PREFIX | address as u64
}

pub const fn decode_generic_shared_address(address: u64) -> Option<u32> {
    if address & GENERIC_ADDRESS_PREFIX_MASK == DEFAULT_GENERIC_SHARED_PREFIX {
        Some(address as u32)
    } else {
        None
    }
}

pub const fn replace_shared_address_cta_rank(address: u32, cta_rank: u32) -> Option<u32> {
    encode_shared_address(shared_address_byte_offset(address), cta_rank)
}

pub const fn replace_generic_shared_address_cta_rank(
    address: u64,
    cta_rank: u32,
) -> Option<u64> {
    let shared = match decode_generic_shared_address(address) {
        Some(value) => value,
        None => return None,
    };
    match replace_shared_address_cta_rank(shared, cta_rank) {
        Some(value) => Some(generic_shared_address(value)),
        None => None,
    }
}

pub const fn shared_address(address: u64) -> u32 {
    address as u32
}

/// Packed-vector widths the ordinary vector ABI admits.
pub const PACKED_VECTOR_WIDTHS: [u32; 4] = [16, 32, 64, 128];

/// Fixed-width storage ABI of one ordinary vector dtype such as `float16x2`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VectorDtypeAbi {
    pub element: Dtype,
    pub lanes: u32,
    pub element_bits: u32,
    pub total_bits: u32,
}

impl VectorDtypeAbi {
    pub const fn itemsize(self) -> u32 {
        self.total_bits / 8
    }
}

/// Scalar storage width for byte-addressable dtypes (sub-byte types excluded).
pub fn scalar_dtype_bits(name: &str) -> Option<u32> {
    Dtype::from_name(name)
        .map(Dtype::bits)
        .filter(|bits| *bits >= 8)
}

/// Port of `dtype_abi.vector_dtype_abi`: `<element>x<lanes>` with a byte-sized
/// non-bool element, at least two lanes, and a 16/32/64/128-bit total.
pub fn vector_dtype_abi(dtype: &str) -> Option<VectorDtypeAbi> {
    let split = dtype.rfind('x')?;
    let (element_name, lanes_text) = (&dtype[..split], &dtype[split + 1..]);
    if lanes_text.is_empty()
        || lanes_text.starts_with('0')
        || !lanes_text.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }
    if element_name == "bool" {
        return None;
    }
    let element_bits = scalar_dtype_bits(element_name)?;
    let lanes: u32 = lanes_text.parse().ok()?;
    if lanes <= 1 {
        return None;
    }
    let total_bits = element_bits.checked_mul(lanes)?;
    if !PACKED_VECTOR_WIDTHS.contains(&total_bits) {
        return None;
    }
    Some(VectorDtypeAbi {
        element: Dtype::from_name(element_name)?,
        lanes,
        element_bits,
        total_bits,
    })
}

/// Every vector dtype name [`vector_dtype_abi`] accepts, sorted by name.
pub fn vector_dtype_abis() -> Vec<(String, VectorDtypeAbi)> {
    let mut out = Vec::new();
    for dtype in Dtype::ALL {
        let bits = dtype.bits();
        if bits < 8 {
            continue;
        }
        for lanes in 2..=(128 / bits) {
            let name = format!("{}x{lanes}", dtype.name());
            if let Some(abi) = vector_dtype_abi(&name) {
                out.push((name, abi));
            }
        }
    }
    out.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0));
    out
}

/// Port of `dtype_abi.dtype_itemsize`: byte itemsize of a scalar or vector dtype.
pub fn dtype_itemsize(dtype: &str) -> Option<u32> {
    scalar_dtype_bits(dtype)
        .map(|bits| bits / 8)
        .or_else(|| vector_dtype_abi(dtype).map(VectorDtypeAbi::itemsize))
}

/// Bindings for address/descriptor helper intrinsics.
pub(crate) const BINDINGS: &[crate::registry::Binding] = &[
    crate::registry::Binding { op: "tirx.cuda.runtime_instr_desc", function: "codec::tcgen_runtime_instruction_descriptor" },
    crate::registry::Binding { op: "tirx.cuda.sm100_2sm_leader_smem_addr", function: "codec::sm100_tma_2sm_mbarrier_address" },
    crate::registry::Binding { op: "tirx.cuda.smem_addr_from_uint64", function: "codec::shared_address" },
    crate::registry::Binding { op: "tirx.cuda.cvta_generic_to_shared", function: "codec::decode_generic_shared_address" },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tcgen_runtime_descriptor_replaces_only_the_two_scale_factor_fields() {
        let original = 0xa5a5_5a5a;
        for scale_factor_id in 0..=3 {
            let encoded = tcgen_runtime_instruction_descriptor(original, scale_factor_id);
            assert_eq!(encoded & !0x6000_0030, original & !0x6000_0030);
            assert_eq!((encoded >> 29) & 0x3, scale_factor_id);
            assert_eq!((encoded >> 4) & 0x3, scale_factor_id);
        }
    }

    #[test]
    fn sm100_two_cta_mbarrier_address_clears_the_remote_rank_bit() {
        assert_eq!(sm100_tma_2sm_mbarrier_address(0xffff_ffff), 0xfeff_ffff);
        assert_eq!(sm100_tma_2sm_mbarrier_address(0x0100_0000), 0);
        assert_eq!(sm100_tma_2sm_mbarrier_address(0x1_1234_5678), 0x1234_5678);
    }

    #[test]
    fn shared_address_codec_matches_the_sm100_cluster_layout() {
        let address = encode_shared_address(0x1234, 3).unwrap();
        assert_eq!(address, 0x0300_1234);
        assert_eq!(shared_address_byte_offset(address), 0x1234);
        assert_eq!(shared_address_cta_rank(address), 3);
        assert_eq!(generic_shared_address(address), 0x0000_fffe_0300_1234);
        assert_eq!(decode_generic_shared_address(generic_shared_address(address)), Some(address));
        assert_eq!(decode_generic_shared_address(0x0000_fffd_0300_1234), None);
        assert_eq!(replace_shared_address_cta_rank(address, 0), Some(0x0000_1234));
        assert_eq!(replace_shared_address_cta_rank(address, 7), Some(0x0700_1234));
        assert_eq!(
            replace_generic_shared_address_cta_rank(generic_shared_address(address), 1),
            Some(0x0000_fffe_0100_1234),
        );
        assert_eq!(replace_generic_shared_address_cta_rank(0x1234, 1), None);
    }

    #[test]
    fn cuda_ffs_matches_frontend_formula() {
        assert_eq!(crate::scalar::cuda_ffs_u32(0), 0);
        assert_eq!(crate::scalar::cuda_ffs_u32(1), 1);
        assert_eq!(crate::scalar::cuda_ffs_u32(0x8000_0000), 32);
        assert_eq!(crate::scalar::cuda_ffs_u32(0b1100), 3);
    }

    #[test]
    fn vector_dtype_abi_matches_python_classification() {
        let abi = vector_dtype_abi("float16x2").unwrap();
        assert_eq!((abi.element, abi.lanes, abi.total_bits, abi.itemsize()), (Dtype::F16, 2, 32, 4));
        assert_eq!(vector_dtype_abi("int8x16").unwrap().total_bits, 128);
        assert_eq!(vector_dtype_abi("float32x3"), None);
        assert_eq!(vector_dtype_abi("boolx4"), None);
        assert_eq!(vector_dtype_abi("float4_e2m1fnx2"), None);
        assert_eq!(vector_dtype_abi("float32x1"), None);
        assert_eq!(vector_dtype_abi("float32x02"), None);
        assert_eq!(vector_dtype_abi("float8_e4m3fnx16").unwrap().itemsize(), 16);
        assert_eq!(dtype_itemsize("bool"), Some(1));
        assert_eq!(dtype_itemsize("uint128"), Some(16));
        assert_eq!(dtype_itemsize("float4_e2m1fn"), None);
        assert_eq!(dtype_itemsize("bfloat16x8"), Some(16));
        let all = vector_dtype_abis();
        assert!(all.windows(2).all(|pair| pair[0].0 < pair[1].0));
        assert!(all.iter().all(|(name, abi)| vector_dtype_abi(name) == Some(*abi)));
        // Same 61 entries as Python `vector_dtype_abis()`.
        assert_eq!(all.len(), 61);
        assert_eq!(all[0].0, "bfloat16x2");
        assert_eq!(all.iter().filter(|(_, abi)| abi.element == Dtype::I8).count(), 4);
    }
}
