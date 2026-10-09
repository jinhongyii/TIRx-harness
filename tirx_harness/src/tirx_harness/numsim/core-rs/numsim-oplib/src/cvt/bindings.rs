//! Registry bindings for the `cvt` family (op names from
//! `engine-rs/SUPPORTED_OPS.md`). Every PTX `cvt` call id lowers to one exact
//! spelling, which `spelling::ptx_cvt` parses and dispatches.

use crate::registry::Binding;

const fn b(op: &'static str, function: &'static str) -> Binding {
    Binding { op, function }
}

pub(crate) const BINDINGS: &[Binding] = &[
    b("tirx.ptx.cvt", "cvt::ptx_cvt"),
    b("tirx.ptx.cvt", "cvt::cvt_int_to_int"),
    b("tirx.ptx.cvt", "cvt::cvt_f32_to_int"),
    b("tirx.ptx.cvt", "cvt::cvt_f64_to_int"),
    b("tirx.ptx.cvt", "cvt::cvt_f16_to_int"),
    b("tirx.ptx.cvt", "cvt::cvt_bf16_to_int"),
    b("tirx.ptx.cvt", "cvt::cvt_int_to_f32"),
    b("tirx.ptx.cvt", "cvt::cvt_int_to_f64"),
    b("tirx.ptx.cvt", "cvt::cvt_int_to_f16"),
    b("tirx.ptx.cvt", "cvt::cvt_int_to_bf16"),
    b("tirx.ptx.cvt", "cvt::cvt_f32_to_f32"),
    b("tirx.ptx.cvt", "cvt::cvt_f64_to_f64"),
    b("tirx.ptx.cvt", "cvt::cvt_f32_to_f64"),
    b("tirx.ptx.cvt", "cvt::cvt_f64_to_f32"),
    b("tirx.ptx.cvt", "cvt::cvt_half_to_half"),
    b("tirx.ptx.cvt", "cvt::cvt_half_to_f32"),
    b("tirx.ptx.cvt", "cvt::cvt_half_to_f64"),
    b("tirx.ptx.cvt", "cvt::cvt_f32_to_half"),
    b("tirx.ptx.cvt", "cvt::cvt_f64_to_half"),
    b("tirx.ptx.cvt", "cvt::cvt_half_cross"),
    b("tirx.ptx.cvt_tf32_f32", "cvt::cvt_f32_to_tf32"),
    b("tirx.ptx.cvt_pzo_scalar_f32", "cvt::cvt_f32_to_half"),
    b("tirx.ptx.cvt_pzo_tf32_f32", "cvt::cvt_f32_to_tf32"),
    b("tirx.ptx.cvt_pzo_fp16x2_f32", "cvt::cvt_f32_pair_to_half2"),
    b("tirx.ptx.cvt_f16x2_f32", "cvt::cvt_f32_pair_to_half2"),
    b("tirx.ptx.cvt_bf16x2_f32", "cvt::cvt_f32_pair_to_half2"),
    b("tirx.ptx.cvt_rs_f16x2_f32", "cvt::cvt_f32_pair_to_half2_rs"),
    b(
        "tirx.ptx.cvt_rs_bf16x2_f32",
        "cvt::cvt_f32_pair_to_half2_rs",
    ),
    b("tirx.ptx.cvt_f8x2_f32", "cvt::cvt_pack_narrow_x2_rn"),
    b("tirx.ptx.cvt_f8x2_fp16x2", "cvt::cvt_pack_narrow_x2_rn"),
    b("tirx.ptx.cvt_f4x2_f32", "cvt::cvt_pack_narrow_x2_rn"),
    b("tirx.ptx.cvt_f4x2_fp16x2", "cvt::cvt_pack_narrow_x2_rn"),
    b("tirx.ptx.cvt_f6x2_f32", "cvt::cvt_pack_narrow_x2_rounded"),
    b(
        "tirx.ptx.cvt_f6x2_fp16x2",
        "cvt::cvt_pack_narrow_x2_rounded",
    ),
    b(
        "tirx.ptx.cvt_94_narrow_f32",
        "cvt::cvt_pack_narrow_x2_rounded",
    ),
    b(
        "tirx.ptx.cvt_94_narrow_fp16x2",
        "cvt::cvt_pack_narrow_x2_rounded",
    ),
    b(
        "tirx.ptx.cvt_ue5m3x2_f32",
        "cvt::cvt_pack_narrow_x2_rounded",
    ),
    b(
        "tirx.ptx.cvt_ue5m3x2_f32",
        "cvt::cvt_pack_ue5m3x2_unsaturated",
    ),
    b(
        "tirx.ptx.cvt_ue5m3x2_f32_scaled",
        "cvt::cvt_pack_narrow_x2_rounded",
    ),
    b(
        "tirx.ptx.cvt_ue5m3x2_f32_scaled",
        "cvt::cvt_pack_ue5m3x2_unsaturated",
    ),
    b(
        "tirx.ptx.cvt_ue5m3x2_fp16x2",
        "cvt::cvt_pack_narrow_x2_rounded",
    ),
    b(
        "tirx.ptx.cvt_ue5m3x2_fp16x2",
        "cvt::cvt_pack_ue5m3x2_unsaturated",
    ),
    b(
        "tirx.ptx.cvt_ue5m3x2_fp16x2_scaled",
        "cvt::cvt_pack_narrow_x2_rounded",
    ),
    b(
        "tirx.ptx.cvt_ue5m3x2_fp16x2_scaled",
        "cvt::cvt_pack_ue5m3x2_unsaturated",
    ),
    b("tirx.ptx.cvt_rs_f8x4_f32", "cvt::cvt_pack_narrow_x4_rs"),
    b("tirx.ptx.cvt_rs_f6x4_f32", "cvt::cvt_pack_narrow_x4_rs"),
    b("tirx.ptx.cvt_rs_f4x4_f32", "cvt::cvt_pack_narrow_x4_rs"),
    b("tirx.ptx.cvt_f16x2_f8x2", "cvt::cvt_unpack_narrow_x2_f16x2"),
    b("tirx.ptx.cvt_f16x2_f6x2", "cvt::cvt_unpack_narrow_x2_f16x2"),
    b("tirx.ptx.cvt_f16x2_f4x2", "cvt::cvt_unpack_narrow_x2_f16x2"),
    b(
        "tirx.ptx.cvt_f16x2_ue5m3x2",
        "cvt::cvt_unpack_narrow_x2_f16x2",
    ),
    b(
        "tirx.ptx.cvt_bf16x2_f8x2",
        "cvt::cvt_unpack_narrow_x2_bf16x2",
    ),
    b(
        "tirx.ptx.cvt_bf16x2_f6x2",
        "cvt::cvt_unpack_narrow_x2_bf16x2",
    ),
    b(
        "tirx.ptx.cvt_bf16x2_f4x2",
        "cvt::cvt_unpack_narrow_x2_bf16x2",
    ),
    b(
        "tirx.ptx.cvt_bf16x2_ue5m3x2",
        "cvt::cvt_unpack_narrow_x2_bf16x2",
    ),
    b("tirx.ptx.cvt_s2f6x2_f32", "cvt::cvt_pack_s2f6x2"),
    b("tirx.ptx.cvt_s2f6x2_bf16x2", "cvt::cvt_pack_s2f6x2"),
    b("tirx.ptx.cvt_bf16x2_s2f6x2", "cvt::cvt_unpack_s2f6x2"),
    b("tirx.ptx.cvt_ue8m0x2_f32", "cvt::cvt_pack_ue8m0x2"),
    b("tirx.ptx.cvt_ue8m0x2_bf16x2", "cvt::cvt_pack_ue8m0x2"),
    b("tirx.ptx.cvt_bf16x2_ue8m0x2", "cvt::cvt_unpack_ue8m0x2"),
    b("tirx.ptx.cvt_pack", "cvt::ptx_cvt"),
    b("tirx.ptx.cvt_pack_c", "cvt::ptx_cvt"),
];
