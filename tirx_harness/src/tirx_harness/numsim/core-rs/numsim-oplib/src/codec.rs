//! Pure bit codecs (legacy `engine-rs/src/instruction_codec.rs`).
//!
//! Only the codecs the v2 contract uses remain. Shared-memory address
//! encoding is the engine's (`numsim_core::arena::addr`), and dtype
//! classification lives in `numsim_types::Dtype`; the legacy duplicates
//! (shared-address helpers, the storage dtype-ABI tables) were removed
//! at step 5.

/// `tirx.cuda.runtime_instr_desc`: the instruction descriptor `bits` with
/// both 2-bit scale-factor ID fields (bits 29..30 and 4..5) set to `sf_id`;
/// every other bit is kept. Pure bit manipulation, no masking of `sf_id`
/// beyond what the shifts drop.
pub const fn tcgen_runtime_instruction_descriptor(bits: u32, sf_id: u32) -> u32 {
    (bits & !0x6000_0030_u32) | sf_id.wrapping_shl(29) | sf_id.wrapping_shl(4)
}

/// `tirx.cuda.sm100_2sm_leader_smem_addr`: the low 32 bits of `address`
/// with the shared::cluster CTA-rank bit 0 (hardware bit 24) cleared, i.e.
/// the same offset in the even CTA of a 2-SM pair.
pub const fn sm100_tma_2sm_mbarrier_address(address: u64) -> u32 {
    (address as u32) & 0xfeff_ffff_u32
}

pub(crate) const BINDINGS: &[crate::registry::Binding] = &[
    crate::registry::Binding {
        op: "tirx.cuda.runtime_instr_desc",
        function: "codec::tcgen_runtime_instruction_descriptor",
    },
    crate::registry::Binding {
        op: "tirx.cuda.sm100_2sm_leader_smem_addr",
        function: "codec::sm100_tma_2sm_mbarrier_address",
    },
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
    fn cuda_ffs_matches_frontend_formula() {
        assert_eq!(crate::scalar::cuda_ffs_u32(0), 0);
        assert_eq!(crate::scalar::cuda_ffs_u32(1), 1);
        assert_eq!(crate::scalar::cuda_ffs_u32(0x8000_0000), 32);
        assert_eq!(crate::scalar::cuda_ffs_u32(0b1100), 3);
    }
}
