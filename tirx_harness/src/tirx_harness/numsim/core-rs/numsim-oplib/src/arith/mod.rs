//! Register arithmetic (`oplib::arith`): pure per-lane kernels of every PTX
//! register instruction except `cvt`.
//!
//! Moved from legacy `engine-rs/src/runtime/instructions/reg.rs` (all but the
//! `cvt` section), `reg_compare.rs`, `prmt.rs`, `packed_fma.rs`,
//! `packed_mul.rs`, `mov_unpack.rs` and `sparse_compress.rs`. The legacy
//! compile-time ABI markers (`variant::*`, `register_variant!`) are replaced
//! by plain functions taking runtime modifier parameters. Warp-wide
//! application, active-mask gating and error site/lane attribution are the
//! engine's job; fallible kernels return [`crate::types::OpResult`].
//!
//! Submodules: [`int`] integer arithmetic, [`bits`] bit manipulation,
//! [`float`] f32/f64, [`half`] f16/bf16 and mixed precision, [`compare`]
//! setp/set/selp/slct/testp, [`mov`] moves and policy, [`sparse`]
//! spcompress/spdecompress.

pub mod bits;
pub mod compare;
pub mod float;
pub mod half;
pub mod int;
pub mod mov;
pub mod sparse;
#[cfg(test)]
mod tests;

pub use bits::*;
pub use compare::*;
pub use float::*;
pub use half::*;
pub use int::*;
pub use mov::*;
pub use sparse::*;

/// Predicate combiner of `set/setp.CmpOp.BoolOp` and `lop3.BoolOp`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoolOp {
    And,
    Or,
    Xor,
}

impl BoolOp {
    /// `value BoolOp predicate`.
    pub fn apply(self, value: bool, predicate: bool) -> bool {
        match self {
            Self::And => value && predicate,
            Self::Or => value || predicate,
            Self::Xor => value ^ predicate,
        }
    }
}

macro_rules! bindings {
    ($($op:literal => [$($function:literal),+ $(,)?]),+ $(,)?) => {
        &[$($(crate::registry::Binding { op: $op, function: $function },)+)+]
    };
}

/// Registry bindings for this family: legacy op name -> OpLib function path.
pub(crate) const BINDINGS: &[crate::registry::Binding] = bindings! {
    "tirx.ptx.add" => ["arith::add_f32", "scalar::add_f64", "cvt::add_f32x2", "arith::add_mixed_f32"],
    "tirx.ptx.sub" => ["arith::sub_f32", "scalar::sub_f64", "cvt::sub_f32x2", "arith::sub_mixed_f32"],
    "tirx.ptx.mul" => ["arith::mul_f32", "scalar::mul_f64", "cvt::mul_f32x2"],
    "tirx.ptx.fma" => ["arith::fma_f32", "scalar::fma_f64", "cvt::fma_f32x2", "arith::fma_mixed_f32"],
    "tirx.ptx.mad_f" => ["arith::fma_f32", "scalar::fma_f64"],
    "tirx.ptx.add_int" => ["arith::add_int", "arith::add_sat_s32", "arith::add_16x2"],
    "tirx.ptx.sub_int" => ["arith::sub_int", "arith::sub_sat_s32"],
    "tirx.ptx.mul_int" => ["arith::mul_lo_int", "arith::mul_hi_int"],
    "tirx.ptx.mad_int" => ["arith::mad_lo_int", "arith::mad_hi_int", "arith::mad_hi_sat_s32"],
    "tirx.ptx.mul_wide" => ["arith::mul_wide_s16", "arith::mul_wide_u16", "arith::mul_wide_s32", "arith::mul_wide_u32"],
    "tirx.ptx.mad_wide" => ["arith::mad_wide_s16", "arith::mad_wide_u16", "arith::mad_wide_s32", "arith::mad_wide_u32"],
    "tirx.ptx.mul24" => ["arith::mul24_s32", "arith::mul24_u32"],
    "tirx.ptx.mad24" => ["arith::mad24_s32", "arith::mad24_u32", "arith::mad24_hi_sat_s32"],
    "tirx.ptx.sad" => ["arith::sad_int"],
    "tirx.ptx.dp2a" => ["arith::dp2a"],
    "tirx.ptx.dp4a" => ["arith::dp4a"],
    "tirx.ptx.div" => ["arith::div_int"],
    "tirx.ptx.rem" => ["arith::rem_int"],
    "tirx.ptx.neg_int" => ["arith::neg_int"],
    "tirx.ptx.abs" => ["arith::abs_int"],
    "tirx.ptx.min" => ["arith::min_int", "arith::minmax_16x2", "arith::min_relu_s32", "arith::min_f32", "arith::minmax_f32", "scalar::cuda_f64_min", "arith::minmax_half", "arith::minmax_half2", "arith::min_f16_widened", "arith::min_bf16_widened", "cvt::hmin2_f16", "cvt::hmin2_bf16"],
    "tirx.ptx.max" => ["arith::max_int", "arith::minmax_16x2", "arith::max_relu_s32", "arith::max_f32", "arith::minmax_f32", "scalar::cuda_f64_max", "arith::minmax_half", "arith::minmax_half2", "arith::max_f16_widened", "arith::max_bf16_widened", "cvt::hmax2_f16", "cvt::hmax2_bf16"],
    "tirx.ptx.min3" => ["arith::minmax_f32"],
    "tirx.ptx.max3" => ["arith::minmax_f32"],
    "tirx.cuda.hmin2" => ["cvt::hmin2_f16", "cvt::hmin2_bf16"],
    "tirx.cuda.hmax2" => ["cvt::hmax2_f16", "cvt::hmax2_bf16"],
    "tirx.ptx.add_half" => ["arith::add_half", "arith::add_half2", "scalar::add_bf16x2_bits_rn"],
    "tirx.ptx.sub_half" => ["arith::sub_half", "arith::sub_half2", "scalar::sub_f16x2_bits_rn", "scalar::sub_bf16x2_bits_rn"],
    "tirx.ptx.mul_half" => ["arith::mul_half", "arith::mul_half2", "arith::mul_f16x2_rn", "arith::mul_bf16x2_rn"],
    "tirx.ptx.fma_half" => ["arith::fma_half", "arith::fma_half2", "arith::fma_f16x2_rn", "arith::fma_bf16x2_rn"],
    "tirx.ptx.add_mixed_vec_up" => ["arith::add_mixed_f32x2"],
    "tirx.ptx.sub_mixed_vec_up" => ["arith::sub_mixed_f32x2"],
    "tirx.ptx.fma_mixed_vec" => ["arith::fma_mixed_f32x2"],
    "tirx.ptx.add_mixed_vec_down_f16" => ["arith::mixed_f32x2_down"],
    "tirx.ptx.add_mixed_vec_down_bf16" => ["arith::mixed_f32x2_down"],
    "tirx.ptx.sub_mixed_vec_down_f16" => ["arith::mixed_f32x2_down"],
    "tirx.ptx.sub_mixed_vec_down_bf16" => ["arith::mixed_f32x2_down"],
    "tirx.ptx.mul_mixed_vec_down_f16" => ["arith::mixed_f32x2_down"],
    "tirx.ptx.mul_mixed_vec_down_bf16" => ["arith::mixed_f32x2_down"],
    "tirx.ptx.mul_mixed_vec_bf16_f16" => ["arith::mul_bf16x2_f16x2"],
    "tirx.ptx.mul_mixed_vec_f16_bf16" => ["arith::mul_f16x2_bf16x2"],
    "tirx.ptx.div_f" => ["arith::div_f32", "scalar::div_f64"],
    "tirx.ptx.rcp" => ["arith::rcp_f32", "arith::rcp_approx_f32", "arith::rcp_f64", "scalar::ptx_rcp_approx_ftz_f64"],
    "tirx.ptx.sqrt" => ["arith::sqrt_f32", "scalar::ptx_sqrt_f64"],
    "tirx.ptx.rsqrt" => ["arith::rsqrt_approx_f32", "arith::rsqrt_f64", "scalar::ptx_rsqrt_approx_ftz_f64"],
    "tirx.ptx.sin" => ["scalar::ptx_sin_approx_f32"],
    "tirx.ptx.cos" => ["scalar::ptx_cos_approx_f32"],
    "tirx.ptx.ex2" => ["arith::ex2_approx_f32"],
    "tirx.ptx.ex2_half" => ["scalar::ptx_exp2_approx_f16", "scalar::ptx_exp2_approx_f16x2", "scalar::ptx_exp2_approx_ftz_bf16", "scalar::ptx_exp2_approx_ftz_bf16x2"],
    "tirx.ptx.lg2" => ["arith::lg2_approx_f32", "arith::lg2_f64"],
    "tirx.ptx.tanh" => ["scalar::ptx_tanh_approx_f32"],
    "tirx.ptx.tanh_half" => ["scalar::ptx_tanh_approx_f16", "scalar::ptx_tanh_approx_f16x2", "scalar::ptx_tanh_approx_bf16", "scalar::ptx_tanh_approx_bf16x2"],
    "tirx.ptx.neg" => ["arith::neg_f32", "arith::neg_f64"],
    "tirx.ptx.neg_half" => ["scalar::ptx_neg_f16_bits", "scalar::ptx_neg_f16x2_bits", "arith::neg_bf16", "arith::neg_bf16x2"],
    "tirx.ptx.abs_f" => ["arith::abs_f32", "arith::abs_f64"],
    "tirx.ptx.abs_half" => ["arith::abs_f16", "arith::abs_f16x2", "arith::abs_bf16", "arith::abs_bf16x2"],
    "tirx.ptx.copysign" => ["arith::copysign_f32", "arith::copysign_f64"],
    "tirx.ptx.clmad" => ["arith::clmad_lo", "arith::clmad_hi"],
    "tirx.ptx.bfe" => ["arith::bfe_u32", "arith::bfe_u64", "arith::bfe_s32", "arith::bfe_s64"],
    "tirx.ptx.bfi" => ["arith::bfi_b32", "arith::bfi_b64"],
    "tirx.ptx.bfind" => ["arith::bfind_u32", "arith::bfind_u64", "arith::bfind_s32", "arith::bfind_s64"],
    "tirx.ptx.bmsk" => ["arith::bmsk_b32"],
    "tirx.ptx.brev" => ["arith::brev_b32", "arith::brev_b64"],
    "tirx.ptx.clz" => ["arith::clz_b32", "arith::clz_b64"],
    "tirx.ptx.cnot" => ["arith::cnot_b16", "arith::cnot_b32", "arith::cnot_b64"],
    "tirx.ptx.popc" => ["arith::popc_b32", "arith::popc_b64"],
    "tirx.ptx.shf" => ["arith::shf_b32"],
    "tirx.ptx.szext" => ["arith::szext_u32", "arith::szext_s32"],
    "tirx.ptx.and" => ["arith::and"],
    "tirx.ptx.or" => ["arith::or"],
    "tirx.ptx.xor" => ["arith::xor"],
    "tirx.ptx.not" => ["arith::not"],
    "tirx.ptx.lop3" => ["arith::lop3_b32"],
    "tirx.ptx.lop3_bool" => ["arith::lop3_bool_b32"],
    "tirx.ptx.lop3_bool_sink" => ["arith::lop3_bool_b32"],
    "tirx.ptx.shl" => ["arith::shl"],
    "tirx.ptx.shr" => ["arith::shr"],
    "tirx.ptx.prmt" => ["arith::prmt_b32"],
    "tirx.ptx.fns" => ["arith::fns_b32"],
    "tirx.ptx.setp" => ["arith::setp_f32", "arith::setp_f64", "arith::setp_signed", "arith::setp_unsigned", "arith::setp_bits"],
    "tirx.ptx.setp_bool" => ["arith::compare_atom", "arith::BoolOp::apply"],
    "tirx.ptx.setp_pq" => ["arith::compare_atom", "arith::predicate_pair"],
    "tirx.ptx.setp_bool_pq" => ["arith::compare_atom", "arith::predicate_pair", "arith::BoolOp::apply"],
    "tirx.ptx.setp_half" => ["arith::setp_f16", "arith::setp_bf16", "arith::compare_mask_f16x2", "arith::compare_mask_bf16x2"],
    "tirx.ptx.setp_half_pq" => ["arith::compare_mask_f16x2", "arith::compare_mask_bf16x2", "arith::predicate_pair"],
    "tirx.ptx.setp_half_bool" => ["arith::setp_f16", "arith::setp_bf16", "arith::BoolOp::apply"],
    "tirx.ptx.setp_half_bool_pq" => ["arith::compare_mask_f16x2", "arith::compare_mask_bf16x2", "arith::combine_mask", "arith::predicate_pair"],
    "tirx.ptx.set" => ["arith::compare_atom", "arith::set_encode"],
    "tirx.ptx.set_bool" => ["arith::compare_atom", "arith::combine_mask", "arith::set_encode"],
    "tirx.ptx.set_half" => ["arith::compare_mask_f16x2", "arith::compare_mask_bf16x2", "arith::set_encode"],
    "tirx.ptx.set_half_bool" => ["arith::compare_mask_f16x2", "arith::compare_mask_bf16x2", "arith::combine_mask", "arith::set_encode"],
    "tirx.ptx.set_packed" => ["arith::set_packed"],
    "tirx.ptx.selp" => ["arith::selp"],
    "tirx.ptx.slct" => ["arith::slct_s32", "arith::slct_f32"],
    "tirx.ptx.testp" => ["arith::testp_f32", "arith::testp_f64"],
    "tirx.ptx.mov" => ["arith::mov"],
    "tirx.ptx.mov_pack_b16x2" => ["arith::mov_pack_b32"],
    "tirx.ptx.mov_unpack_b16x2" => ["arith::mov_unpack_b32"],
    "tirx.ptx.mov_pack_b32x2" => ["arith::mov_pack_b64"],
    "tirx.ptx.mov_unpack_b32x2" => ["arith::mov_unpack_b64"],
    "tirx.ptx.mov_pack_b64x2" => ["arith::mov_pack_b128"],
    "tirx.ptx.mov_unpack_b64x2" => ["cvt::ptx_mov_unpack_b64x2"],
    "tirx.ptx.mov_pack_b16x4" => ["cvt::ptx_mov_pack_b16x4"],
    "tirx.ptx.mov_unpack_b16x4" => ["cvt::ptx_mov_unpack_b16x4"],
    "tirx.ptx.mov_pack_b32x4" => ["cvt::ptx_mov_pack_b32x4"],
    "tirx.ptx.mov_unpack_b32x4" => ["cvt::ptx_mov_unpack_b32x4"],
    "tirx.ptx.createpolicy_fraction" => ["arith::createpolicy_fraction"],
    "tirx.ptx.createpolicy_fractional" => ["arith::createpolicy_fraction"],
    "tirx.ptx.createpolicy_cvt" => ["arith::createpolicy_cvt"],
    "tirx.ptx.createpolicy_range" => ["arith::createpolicy_range"],
    "tirx.cuda.clock64" => ["arith::sreg_clock64"],
    "tirx.cuda.mov_sreg" => ["arith::sreg_laneid", "arith::sreg_clock64"],
    "tirx.ptx.spcompress" => ["arith::spcompress", "arith::compress_sparse_vector"],
    "tirx.ptx.spdecompress" => ["arith::spdecompress"],
};
