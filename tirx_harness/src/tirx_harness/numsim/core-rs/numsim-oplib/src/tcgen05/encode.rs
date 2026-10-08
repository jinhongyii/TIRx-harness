//! tcgen05 instruction- and matrix-descriptor ENCODERS (the inverse of
//! `instr_desc` / `smem_desc` decoders).
//!
//! Moved from `frontend-rs/src/emit/tcgen_descriptor.rs:15-451` (lowering-time
//! bit assembly for `tirx.cuda.tcgen05_encode_instr_descriptor{,_block_scaled}`)
//! and the per-lane matrix-descriptor formula emitted by
//! `frontend-rs/src/emit/raw_tcgen.rs:1460-1540`
//! (`tirx.cuda.tcgen05_encode_matrix_descriptor`). Error texts are the legacy
//! `Failure::Unsupported` messages.

use crate::types::{OpError, OpResult};

pub const DESCRIPTOR_CALLS: [&str; 3] = [
    "tirx.cuda.tcgen05_encode_matrix_descriptor",
    "tirx.cuda.tcgen05_encode_instr_descriptor",
    "tirx.cuda.tcgen05_encode_instr_descriptor_block_scaled",
];

// dtype, format bits, dense transpose, block transpose, narrow-N transpose key
#[rustfmt::skip]
const FORMAT_ENCODINGS: &[(&str, i64, bool, bool, bool)] = &[
    ("float16", 0, true, false, false),
    ("bfloat16", 1, true, false, false),
    ("tf32", 2, true, false, false),
    ("float8_e4m3fn", 0, true, true, true),
    ("float8_e4m3fnuz", 0, true, true, true),
    ("float8_e5m2", 1, true, true, true),
    ("float6_e2m3fn", 3, false, false, false),
    ("float6_e3m2fn", 4, false, false, false),
    ("float4_e2m1fn", 5, false, false, false),
    ("uint8", 0, true, false, true),
    ("int8", 1, true, false, true),
    ("float32", 1, false, false, false),
    ("int32", 2, false, false, false),
];

fn format_encoding(dtype: &str) -> &'static (&'static str, i64, bool, bool, bool) {
    FORMAT_ENCODINGS
        .iter()
        .find(|row| row.0 == dtype)
        .expect("mapped descriptor dtype")
}

// Each row maps descriptor operand spellings to the engine family that
// interprets their format bits. These are encoding keys, not inferred dtypes.
#[rustfmt::skip]
const DENSE_ENCODINGS: &[(&str, &str, &str, &str)] = &[
    ("float16", "float16", "float16", "f16"),
    ("float16", "float16", "bfloat16", "f16"),
    ("float16", "bfloat16", "float16", "f16"),
    ("float16", "bfloat16", "bfloat16", "f16"),
    ("float32", "float16", "float16", "f16"),
    ("float32", "float16", "bfloat16", "f16"),
    ("float32", "bfloat16", "float16", "f16"),
    ("float32", "bfloat16", "bfloat16", "f16"),
    ("float32", "tf32", "tf32", "tf32"),
    ("int32", "int8", "int8", "i8"),
    ("int32", "int8", "uint8", "i8"),
    ("int32", "uint8", "int8", "i8"),
    ("int32", "uint8", "uint8", "i8"),
    ("float16", "float8_e4m3fn", "float8_e4m3fn", "f8f6f4"),
    ("float16", "float8_e4m3fn", "float8_e4m3fnuz", "f8f6f4"),
    ("float16", "float8_e4m3fn", "float8_e5m2", "f8f6f4"),
    ("float16", "float8_e4m3fn", "float6_e2m3fn", "f8f6f4"),
    ("float16", "float8_e4m3fn", "float6_e3m2fn", "f8f6f4"),
    ("float16", "float8_e4m3fn", "float4_e2m1fn", "f8f6f4"),
    ("float16", "float8_e4m3fnuz", "float8_e4m3fn", "f8f6f4"),
    ("float16", "float8_e4m3fnuz", "float8_e4m3fnuz", "f8f6f4"),
    ("float16", "float8_e4m3fnuz", "float8_e5m2", "f8f6f4"),
    ("float16", "float8_e4m3fnuz", "float6_e2m3fn", "f8f6f4"),
    ("float16", "float8_e4m3fnuz", "float6_e3m2fn", "f8f6f4"),
    ("float16", "float8_e4m3fnuz", "float4_e2m1fn", "f8f6f4"),
    ("float16", "float8_e5m2", "float8_e4m3fn", "f8f6f4"),
    ("float16", "float8_e5m2", "float8_e4m3fnuz", "f8f6f4"),
    ("float16", "float8_e5m2", "float8_e5m2", "f8f6f4"),
    ("float16", "float8_e5m2", "float6_e2m3fn", "f8f6f4"),
    ("float16", "float8_e5m2", "float6_e3m2fn", "f8f6f4"),
    ("float16", "float8_e5m2", "float4_e2m1fn", "f8f6f4"),
    ("float16", "float6_e2m3fn", "float8_e4m3fn", "f8f6f4"),
    ("float16", "float6_e2m3fn", "float8_e4m3fnuz", "f8f6f4"),
    ("float16", "float6_e2m3fn", "float8_e5m2", "f8f6f4"),
    ("float16", "float6_e2m3fn", "float6_e2m3fn", "f8f6f4"),
    ("float16", "float6_e2m3fn", "float6_e3m2fn", "f8f6f4"),
    ("float16", "float6_e2m3fn", "float4_e2m1fn", "f8f6f4"),
    ("float16", "float6_e3m2fn", "float8_e4m3fn", "f8f6f4"),
    ("float16", "float6_e3m2fn", "float8_e4m3fnuz", "f8f6f4"),
    ("float16", "float6_e3m2fn", "float8_e5m2", "f8f6f4"),
    ("float16", "float6_e3m2fn", "float6_e2m3fn", "f8f6f4"),
    ("float16", "float6_e3m2fn", "float6_e3m2fn", "f8f6f4"),
    ("float16", "float6_e3m2fn", "float4_e2m1fn", "f8f6f4"),
    ("float16", "float4_e2m1fn", "float8_e4m3fn", "f8f6f4"),
    ("float16", "float4_e2m1fn", "float8_e4m3fnuz", "f8f6f4"),
    ("float16", "float4_e2m1fn", "float8_e5m2", "f8f6f4"),
    ("float16", "float4_e2m1fn", "float6_e2m3fn", "f8f6f4"),
    ("float16", "float4_e2m1fn", "float6_e3m2fn", "f8f6f4"),
    ("float16", "float4_e2m1fn", "float4_e2m1fn", "f8f6f4"),
    ("float32", "float8_e4m3fn", "float8_e4m3fn", "f8f6f4"),
    ("float32", "float8_e4m3fn", "float8_e4m3fnuz", "f8f6f4"),
    ("float32", "float8_e4m3fn", "float8_e5m2", "f8f6f4"),
    ("float32", "float8_e4m3fn", "float6_e2m3fn", "f8f6f4"),
    ("float32", "float8_e4m3fn", "float6_e3m2fn", "f8f6f4"),
    ("float32", "float8_e4m3fn", "float4_e2m1fn", "f8f6f4"),
    ("float32", "float8_e4m3fnuz", "float8_e4m3fn", "f8f6f4"),
    ("float32", "float8_e4m3fnuz", "float8_e4m3fnuz", "f8f6f4"),
    ("float32", "float8_e4m3fnuz", "float8_e5m2", "f8f6f4"),
    ("float32", "float8_e4m3fnuz", "float6_e2m3fn", "f8f6f4"),
    ("float32", "float8_e4m3fnuz", "float6_e3m2fn", "f8f6f4"),
    ("float32", "float8_e4m3fnuz", "float4_e2m1fn", "f8f6f4"),
    ("float32", "float8_e5m2", "float8_e4m3fn", "f8f6f4"),
    ("float32", "float8_e5m2", "float8_e4m3fnuz", "f8f6f4"),
    ("float32", "float8_e5m2", "float8_e5m2", "f8f6f4"),
    ("float32", "float8_e5m2", "float6_e2m3fn", "f8f6f4"),
    ("float32", "float8_e5m2", "float6_e3m2fn", "f8f6f4"),
    ("float32", "float8_e5m2", "float4_e2m1fn", "f8f6f4"),
    ("float32", "float6_e2m3fn", "float8_e4m3fn", "f8f6f4"),
    ("float32", "float6_e2m3fn", "float8_e4m3fnuz", "f8f6f4"),
    ("float32", "float6_e2m3fn", "float8_e5m2", "f8f6f4"),
    ("float32", "float6_e2m3fn", "float6_e2m3fn", "f8f6f4"),
    ("float32", "float6_e2m3fn", "float6_e3m2fn", "f8f6f4"),
    ("float32", "float6_e2m3fn", "float4_e2m1fn", "f8f6f4"),
    ("float32", "float6_e3m2fn", "float8_e4m3fn", "f8f6f4"),
    ("float32", "float6_e3m2fn", "float8_e4m3fnuz", "f8f6f4"),
    ("float32", "float6_e3m2fn", "float8_e5m2", "f8f6f4"),
    ("float32", "float6_e3m2fn", "float6_e2m3fn", "f8f6f4"),
    ("float32", "float6_e3m2fn", "float6_e3m2fn", "f8f6f4"),
    ("float32", "float6_e3m2fn", "float4_e2m1fn", "f8f6f4"),
    ("float32", "float4_e2m1fn", "float8_e4m3fn", "f8f6f4"),
    ("float32", "float4_e2m1fn", "float8_e4m3fnuz", "f8f6f4"),
    ("float32", "float4_e2m1fn", "float8_e5m2", "f8f6f4"),
    ("float32", "float4_e2m1fn", "float6_e2m3fn", "f8f6f4"),
    ("float32", "float4_e2m1fn", "float6_e3m2fn", "f8f6f4"),
    ("float32", "float4_e2m1fn", "float4_e2m1fn", "f8f6f4"),
];

// A/B type, scale type, engine kind, A/B format bits, scale format bit.
type BlockEncoding = (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    i64,
    i64,
    i64,
);

#[rustfmt::skip]
const BLOCK_ENCODINGS: &[BlockEncoding] = &[
    ("float8_e4m3fn", "float8_e4m3fn", "float8_e8m0fnu", "mxf8f6f4", 0, 0, 1),
    ("float8_e4m3fn", "float8_e4m3fnuz", "float8_e8m0fnu", "mxf8f6f4", 0, 0, 1),
    ("float8_e4m3fn", "float8_e5m2", "float8_e8m0fnu", "mxf8f6f4", 0, 1, 1),
    ("float8_e4m3fn", "float6_e2m3fn", "float8_e8m0fnu", "mxf8f6f4", 0, 3, 1),
    ("float8_e4m3fn", "float6_e3m2fn", "float8_e8m0fnu", "mxf8f6f4", 0, 4, 1),
    ("float8_e4m3fn", "float4_e2m1fn", "float8_e8m0fnu", "mxf8f6f4", 0, 5, 1),
    ("float8_e4m3fnuz", "float8_e4m3fn", "float8_e8m0fnu", "mxf8f6f4", 0, 0, 1),
    ("float8_e4m3fnuz", "float8_e4m3fnuz", "float8_e8m0fnu", "mxf8f6f4", 0, 0, 1),
    ("float8_e4m3fnuz", "float8_e5m2", "float8_e8m0fnu", "mxf8f6f4", 0, 1, 1),
    ("float8_e4m3fnuz", "float6_e2m3fn", "float8_e8m0fnu", "mxf8f6f4", 0, 3, 1),
    ("float8_e4m3fnuz", "float6_e3m2fn", "float8_e8m0fnu", "mxf8f6f4", 0, 4, 1),
    ("float8_e4m3fnuz", "float4_e2m1fn", "float8_e8m0fnu", "mxf8f6f4", 0, 5, 1),
    ("float8_e5m2", "float8_e4m3fn", "float8_e8m0fnu", "mxf8f6f4", 1, 0, 1),
    ("float8_e5m2", "float8_e4m3fnuz", "float8_e8m0fnu", "mxf8f6f4", 1, 0, 1),
    ("float8_e5m2", "float8_e5m2", "float8_e8m0fnu", "mxf8f6f4", 1, 1, 1),
    ("float8_e5m2", "float6_e2m3fn", "float8_e8m0fnu", "mxf8f6f4", 1, 3, 1),
    ("float8_e5m2", "float6_e3m2fn", "float8_e8m0fnu", "mxf8f6f4", 1, 4, 1),
    ("float8_e5m2", "float4_e2m1fn", "float8_e8m0fnu", "mxf8f6f4", 1, 5, 1),
    ("float6_e2m3fn", "float8_e4m3fn", "float8_e8m0fnu", "mxf8f6f4", 3, 0, 1),
    ("float6_e2m3fn", "float8_e4m3fnuz", "float8_e8m0fnu", "mxf8f6f4", 3, 0, 1),
    ("float6_e2m3fn", "float8_e5m2", "float8_e8m0fnu", "mxf8f6f4", 3, 1, 1),
    ("float6_e2m3fn", "float6_e2m3fn", "float8_e8m0fnu", "mxf8f6f4", 3, 3, 1),
    ("float6_e2m3fn", "float6_e3m2fn", "float8_e8m0fnu", "mxf8f6f4", 3, 4, 1),
    ("float6_e2m3fn", "float4_e2m1fn", "float8_e8m0fnu", "mxf8f6f4", 3, 5, 1),
    ("float6_e3m2fn", "float8_e4m3fn", "float8_e8m0fnu", "mxf8f6f4", 4, 0, 1),
    ("float6_e3m2fn", "float8_e4m3fnuz", "float8_e8m0fnu", "mxf8f6f4", 4, 0, 1),
    ("float6_e3m2fn", "float8_e5m2", "float8_e8m0fnu", "mxf8f6f4", 4, 1, 1),
    ("float6_e3m2fn", "float6_e2m3fn", "float8_e8m0fnu", "mxf8f6f4", 4, 3, 1),
    ("float6_e3m2fn", "float6_e3m2fn", "float8_e8m0fnu", "mxf8f6f4", 4, 4, 1),
    ("float6_e3m2fn", "float4_e2m1fn", "float8_e8m0fnu", "mxf8f6f4", 4, 5, 1),
    ("float4_e2m1fn", "float8_e4m3fn", "float8_e8m0fnu", "mxf8f6f4", 5, 0, 1),
    ("float4_e2m1fn", "float8_e4m3fnuz", "float8_e8m0fnu", "mxf8f6f4", 5, 0, 1),
    ("float4_e2m1fn", "float8_e5m2", "float8_e8m0fnu", "mxf8f6f4", 5, 1, 1),
    ("float4_e2m1fn", "float6_e2m3fn", "float8_e8m0fnu", "mxf8f6f4", 5, 3, 1),
    ("float4_e2m1fn", "float6_e3m2fn", "float8_e8m0fnu", "mxf8f6f4", 5, 4, 1),
    ("float4_e2m1fn", "float4_e2m1fn", "float8_e8m0fnu", "mxf4", 1, 1, 1),
    ("float4_e2m1fn", "float4_e2m1fn", "float8_e4m3fn", "mxf4nvf4", 1, 1, 0),
];

fn dense_kind(d_dtype: &str, a_dtype: &str, b_dtype: &str) -> OpResult<&'static str> {
    match DENSE_ENCODINGS.iter().find(|row| (row.0, row.1, row.2) == (d_dtype, a_dtype, b_dtype)) {
        Some(row) => Ok(row.3),
        None => Err(OpError::message(format!(
            "tcgen05 dense instruction descriptor has invalid dtype combination D={d_dtype}, A={a_dtype}, B={b_dtype}"
        ))),
    }
}

fn block_encoding(
    d_dtype: &str,
    a_dtype: &str,
    b_dtype: &str,
    sfa_dtype: &str,
    sfb_dtype: &str,
) -> OpResult<&'static BlockEncoding> {
    if d_dtype != "float32" {
        return Err(OpError::message("tcgen05 block-scaled instruction descriptor requires float32 D"));
    }
    match BLOCK_ENCODINGS.iter().find(|row| (row.0, row.1, row.2, row.2) == (a_dtype, b_dtype, sfa_dtype, sfb_dtype)) {
        Some(row) => Ok(row),
        None => Err(OpError::message(format!(
            "tcgen05 block-scaled instruction descriptor has invalid dtype combination D={d_dtype}, A={a_dtype}, B={b_dtype}, SFA={sfa_dtype}, SFB={sfb_dtype}"
        ))),
    }
}

// Exact shape keys accepted by the descriptor helpers. N lists include the
// narrow integer encodings and the dense FP4 K96 helper encodings explicitly.
const N8: &[i64] = &[
    8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168,
    176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256,
];
const N16: &[i64] = &[
    16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256,
];
const N32: &[i64] = &[32, 64, 96, 128, 160, 192, 224, 256];
const N_I8_CTA1: &[i64] = &[
    8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256,
];

// kind, CTA group, M, K, sparse, N encodings.
//
// PTX ISA 9.4, 9.7.18.2.1 (tcgen05.mma shape table), as ruled in
// `docs/development/numsim-isa-answers.md` "tcgen05.mma shapes" (delta L4):
// for `.kind::f16`, `.kind::tf32` and `.kind::f8f6f4`,
//   "cta_group::1, M = 64:  N % 8 == 0, 8 <= N <= 256";
//   "cta_group::1, M = 128: N % 16 == 0, 16 <= N <= 256";
//   "cta_group::2, M = 128 / 256: N % 32 == 0, 32 <= N <= 256".
// `.kind::i8`: cta_group::1 M = 64 / 128 take N % 16 plus N in {8, 24};
// cta_group::2 takes N % 32. Block-scaled kinds (`mxf8f6f4`, `mxf4`,
// `mxf4nvf4`): cta_group::1 M = 128 takes N % 8; cta_group::2 M = 128 / 256
// takes N % 16, and a sparse cta_group::2 form needs M = 256. K per kind is
// TVM's `_TCGEN05_MMA_K`, plus dense K = 96 for the two MXF4 kinds at
// (cta_group::1, M = 128) and (cta_group::2, M = 256) (9.7.18.2.1.1). This is
// the same table as TVM's `_TCGEN05_MMA_SHAPE_RULES`. Legacy used N % 8 for
// M = 128 and N % 16 at cta_group::2, which was too permissive.
// The block-scaled scale-factor K extent per instruction ({1, 4, 16}) is a
// property of the tile-level SFA/SFB region and is checked by TVM's dispatch;
// a raw instruction carries no such extent.
#[rustfmt::skip]
const SHAPE_ENCODINGS: &[(&str, i64, i64, i64, bool, &[i64])] = &[
    ("f16", 1, 64, 16, false, N8),
    ("f16", 1, 128, 16, false, N16),
    ("f16", 1, 64, 32, true, N8),
    ("f16", 1, 128, 32, true, N16),
    ("f16", 2, 128, 16, false, N32),
    ("f16", 2, 256, 16, false, N32),
    ("f16", 2, 128, 32, true, N32),
    ("f16", 2, 256, 32, true, N32),
    ("tf32", 1, 64, 8, false, N8),
    ("tf32", 1, 128, 8, false, N16),
    ("tf32", 1, 64, 16, true, N8),
    ("tf32", 1, 128, 16, true, N16),
    ("tf32", 2, 128, 8, false, N32),
    ("tf32", 2, 256, 8, false, N32),
    ("tf32", 2, 128, 16, true, N32),
    ("tf32", 2, 256, 16, true, N32),
    ("f8f6f4", 1, 64, 32, false, N8),
    ("f8f6f4", 1, 128, 32, false, N16),
    ("f8f6f4", 1, 64, 64, true, N8),
    ("f8f6f4", 1, 128, 64, true, N16),
    ("f8f6f4", 2, 128, 32, false, N32),
    ("f8f6f4", 2, 256, 32, false, N32),
    ("f8f6f4", 2, 128, 64, true, N32),
    ("f8f6f4", 2, 256, 64, true, N32),
    ("i8", 1, 64, 32, false, N_I8_CTA1),
    ("i8", 1, 128, 32, false, N_I8_CTA1),
    ("i8", 1, 64, 64, true, N_I8_CTA1),
    ("i8", 1, 128, 64, true, N_I8_CTA1),
    ("i8", 2, 128, 32, false, N32),
    ("i8", 2, 256, 32, false, N32),
    ("i8", 2, 128, 64, true, N32),
    ("i8", 2, 256, 64, true, N32),
    ("mxf8f6f4", 1, 128, 32, false, N8),
    ("mxf8f6f4", 1, 128, 64, true, N8),
    ("mxf8f6f4", 2, 128, 32, false, N16),
    ("mxf8f6f4", 2, 256, 32, false, N16),
    ("mxf8f6f4", 2, 256, 64, true, N16),
    ("mxf4", 1, 128, 64, false, N8),
    ("mxf4", 1, 128, 96, false, N8),
    ("mxf4", 1, 128, 128, true, N8),
    ("mxf4", 2, 128, 64, false, N16),
    ("mxf4", 2, 256, 64, false, N16),
    ("mxf4", 2, 256, 96, false, N16),
    ("mxf4", 2, 256, 128, true, N16),
    ("mxf4nvf4", 1, 128, 64, false, N8),
    ("mxf4nvf4", 1, 128, 96, false, N8),
    ("mxf4nvf4", 1, 128, 128, true, N8),
    ("mxf4nvf4", 2, 128, 64, false, N16),
    ("mxf4nvf4", 2, 256, 64, false, N16),
    ("mxf4nvf4", 2, 256, 96, false, N16),
    ("mxf4nvf4", 2, 256, 128, true, N16),
];

pub fn validate_tcgen05_instruction_shape(
    kind: &str,
    cta_group: i64,
    m: i64,
    n: i64,
    k: i64,
    sparse: bool,
) -> OpResult<()> {
    if SHAPE_ENCODINGS.iter().any(|row| {
        (row.0, row.1, row.2, row.3, row.4) == (kind, cta_group, m, k, sparse) && row.5.contains(&n)
    }) {
        return Ok(());
    }
    // Keep actionable rejection reasons after the single encoding lookup.
    if cta_group != 1 && cta_group != 2 {
        return Err(OpError::message(format!(
            "tcgen05 instruction descriptor cta_group must be 1 or 2, got {cta_group}"
        )));
    }
    if !SHAPE_ENCODINGS.iter().any(|row| row.0 == kind) {
        return Err(OpError::message(format!(
            "unknown tcgen05 instruction descriptor kind {:?}",
            kind
        )));
    }
    if sparse && matches!(kind, "mxf8f6f4" | "mxf4" | "mxf4nvf4") && cta_group == 2 && m != 256 {
        return Err(OpError::message(format!(
            "invalid sparse tcgen05 block-scaled descriptor shape kind={kind}, cta_group={cta_group}, M={m}, N={n}, K={k}; CTA group 2 requires M=256"
        )));
    }
    Err(OpError::message(format!(
        "invalid tcgen05 descriptor shape kind={kind}, cta_group={cta_group}, M={m}, N={n}, K={k}"
    )))
}

fn validate_8bit_transpose_b_shape(
    b_dtype: &str,
    trans_b: bool,
    cta_group: i64,
    n: i64,
) -> OpResult<()> {
    if !trans_b || !format_encoding(b_dtype).4 {
        return Ok(());
    }
    let (step, encodings) = if cta_group == 1 { (16, N16) } else { (32, N32) };
    if !encodings.contains(&n) {
        return Err(OpError::message(format!(
            "tcgen05 8-bit transpose B requires cta_group={cta_group} N in [{step}, 256] with step {step}, got N={n}"
        )));
    }
    Ok(())
}

// Engine kind -> descriptor flag bits it encodes (negate A/B, saturate D).
const DENSE_FLAG_ENCODINGS: &[(&str, i64)] = &[
    ("f16", (1 << 13) | (1 << 14)),
    ("tf32", (1 << 13) | (1 << 14)),
    ("f8f6f4", (1 << 13) | (1 << 14)),
    ("i8", 1 << 3),
];

/// Assemble a dense descriptor using its dtype, shape and flag encoding rows.
#[allow(clippy::too_many_arguments)]
pub fn encode_dense_instr_descriptor_fields(
    d_dtype: &str,
    a_dtype: &str,
    b_dtype: &str,
    m: i64,
    n: i64,
    k: i64,
    trans_a: bool,
    trans_b: bool,
    cta_group: i64,
    neg_a: bool,
    neg_b: bool,
    sat_d: bool,
    sparse: bool,
) -> OpResult<i64> {
    let kind = dense_kind(d_dtype, a_dtype, b_dtype)?;
    validate_tcgen05_instruction_shape(kind, cta_group, m, n, k, sparse)?;
    if trans_a && !format_encoding(a_dtype).2 {
        return Err(OpError::message(format!("tcgen05 transpose A is invalid for {a_dtype}")));
    }
    if trans_b && !format_encoding(b_dtype).2 {
        return Err(OpError::message(format!("tcgen05 transpose B is invalid for {b_dtype}")));
    }
    validate_8bit_transpose_b_shape(b_dtype, trans_b, cta_group, n)?;
    let flag_bits = DENSE_FLAG_ENCODINGS
        .iter()
        .find(|row| row.0 == kind)
        .expect("mapped dense kind")
        .1;
    if (neg_a || neg_b) && flag_bits & ((1 << 13) | (1 << 14)) == 0 {
        return Err(OpError::message(format!("tcgen05 negate is invalid for kind {kind}")));
    }
    if sat_d && flag_bits & (1 << 3) == 0 {
        return Err(OpError::message(format!("tcgen05 saturation is invalid for kind {kind}")));
    }
    let mut value: i64 = i64::from(sparse) << 2;
    value |= i64::from(sat_d) << 3;
    value |= (format_encoding(d_dtype).1 & 0x3) << 4;
    value |= (format_encoding(a_dtype).1 & 0x7) << 7;
    value |= (format_encoding(b_dtype).1 & 0x7) << 10;
    value |= i64::from(neg_a) << 13;
    value |= i64::from(neg_b) << 14;
    value |= i64::from(trans_a) << 15;
    value |= i64::from(trans_b) << 16;
    value |= ((n >> 3) & 0x3F) << 17;
    value |= ((m >> 4) & 0x1F) << 24;
    Ok(value & 0xFFFF_FFFF)
}

/// `encode_block_scaled_instr_descriptor_fields`.
#[allow(clippy::too_many_arguments)]
pub fn encode_block_scaled_instr_descriptor_fields(
    d_dtype: &str,
    a_dtype: &str,
    b_dtype: &str,
    sfa_dtype: &str,
    sfb_dtype: &str,
    m: i64,
    n: i64,
    k: i64,
    trans_a: bool,
    trans_b: bool,
    cta_group: i64,
    neg_a: bool,
    neg_b: bool,
    sparse: bool,
) -> OpResult<i64> {
    let encoding = block_encoding(d_dtype, a_dtype, b_dtype, sfa_dtype, sfb_dtype)?;
    let kind = encoding.3;
    validate_tcgen05_instruction_shape(kind, cta_group, m, n, k, sparse)?;
    if trans_a && !format_encoding(a_dtype).3 {
        return Err(OpError::message(format!(
            "tcgen05 block transpose A is invalid for {a_dtype}"
        )));
    }
    if trans_b && !format_encoding(b_dtype).3 {
        return Err(OpError::message(format!(
            "tcgen05 block transpose B is invalid for {b_dtype}"
        )));
    }
    validate_8bit_transpose_b_shape(b_dtype, trans_b, cta_group, n)?;
    let (a_format, b_format, scale_format) = (encoding.4, encoding.5, encoding.6);
    let mut value: i64 = i64::from(sparse) << 2;
    value |= (a_format & 0x7) << 7;
    value |= (b_format & 0x7) << 10;
    value |= i64::from(neg_a) << 13;
    value |= i64::from(neg_b) << 14;
    value |= i64::from(trans_a) << 15;
    value |= i64::from(trans_b) << 16;
    value |= ((n >> 3) & 0x3F) << 17;
    value |= (scale_format & 0x1) << 23;
    value |= ((m >> 4) & 0x1F) << 24;
    value |= i64::from(k == 96) << 31;
    Ok(value & 0xFFFF_FFFF)
}


/// Per-lane `tcgen05_encode_matrix_descriptor` (TVM `SmemDescriptor` helper):
/// unsigned fields truncate to 14 bits, the version bit 46 is set, and a
/// swizzle outside 1..=4 leaves the zero-initialized layout. `shared_address`
/// is the 32-bit shared-window address (`cvta.to.shared`); the encoder neither
/// reads the allocation nor validates a future TCGEN access.
pub fn encode_matrix_descriptor(shared_address: u32, ldo: i64, sdo: i64, swizzle: i64) -> u64 {
    let layout_type: u64 = match swizzle {
        1 => 6,
        2 => 4,
        3 => 2,
        4 => 1,
        _ => 0,
    };
    u64::from((shared_address >> 4) & 0x3fff)
        | ((ldo as u64 & 0x3fff) << 16)
        | ((sdo as u64 & 0x3fff) << 32)
        | (1_u64 << 46)
        | (layout_type << 61)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// TVM's `_check_tcgen05_mma_matrix_shape` (the PTX ISA table, delta L4),
    /// written out independently of `SHAPE_ENCODINGS`.
    fn isa_shape(kind: &str, cta: i64, m: i64, n: i64, k: i64, sparse: bool) -> bool {
        let block = matches!(kind, "mxf8f6f4" | "mxf4" | "mxf4nvf4");
        let rule: Option<(&[(i64, i64)], &[i64])> = match (kind, cta) {
            ("f16" | "tf32" | "f8f6f4", 1) => Some((&[(64, 8), (128, 16)], &[])),
            ("f16" | "tf32" | "f8f6f4", 2) => Some((&[(128, 32), (256, 32)], &[])),
            ("i8", 1) => Some((&[(64, 16), (128, 16)], &[8, 24])),
            ("i8", 2) => Some((&[(128, 32), (256, 32)], &[])),
            (_, 1) if block => Some((&[(128, 8)], &[])),
            (_, 2) if block => Some((&[(128, 16), (256, 16)], &[])),
            _ => None,
        };
        let Some((steps, extra)) = rule else { return false };
        if block && cta == 2 && sparse && m != 256 {
            return false;
        }
        let Some(&(_, step)) = steps.iter().find(|(mm, _)| *mm == m) else { return false };
        if !extra.contains(&n) && !((step..=256).contains(&n) && n % step == 0) {
            return false;
        }
        let (dense, sparse_k) = match kind {
            "f16" => (16, 32),
            "tf32" => (8, 16),
            "f8f6f4" | "i8" | "mxf8f6f4" => (32, 64),
            _ => (64, 128),
        };
        let k96 = !sparse && matches!(kind, "mxf4" | "mxf4nvf4") && matches!((cta, m), (1, 128) | (2, 256)) && k == 96;
        k == if sparse { sparse_k } else { dense } || k96
    }

    /// Exhaustive grid: every kind, CTA group, M, N, K and density is
    /// accepted exactly when the PTX ISA shape table lists it.
    #[test]
    fn shape_table_matches_the_ptx_isa_grid() {
        let mut accepted = 0;
        for kind in ["f16", "tf32", "f8f6f4", "i8", "mxf8f6f4", "mxf4", "mxf4nvf4"] {
            for cta in [1, 2] {
                for m in (16..=256).step_by(16) {
                    for n in (8..=256).step_by(8) {
                        for k in [8, 16, 32, 64, 96, 128] {
                            for sparse in [false, true] {
                                let ours = validate_tcgen05_instruction_shape(kind, cta, m, n, k, sparse).is_ok();
                                assert_eq!(ours, isa_shape(kind, cta, m, n, k, sparse), "{kind} cta{cta} M{m} N{n} K{k} sparse {sparse}");
                                accepted += usize::from(ours);
                            }
                        }
                    }
                }
            }
        }
        assert!(accepted > 0);
        // The two shapes the ruling names.
        assert!(validate_tcgen05_instruction_shape("f16", 1, 128, 8, 16, false).is_err());
        assert!(validate_tcgen05_instruction_shape("f16", 2, 128, 16, 16, false).is_err());
    }
    use crate::tcgen05::instr_desc::{decode_b16, decode_mxf8f6f4};

    #[test]
    fn dense_bf16_descriptor_bits_and_decoder_round_trip() {
        let bits = encode_dense_instr_descriptor_fields(
            "float32", "bfloat16", "bfloat16", 128, 256, 16, false, true, 1, false, false, false,
            false,
        )
        .unwrap();
        assert_eq!(bits & (0x3 << 4), 1 << 4); // D=f32
        assert_eq!((bits >> 7) & 7, 1); // A=bf16
        assert_eq!((bits >> 10) & 7, 1); // B=bf16
        assert_eq!((bits >> 16) & 1, 1); // transpose B
        assert_eq!((bits >> 17) & 0x3f, 256 >> 3);
        assert_eq!((bits >> 24) & 0x1f, 128 >> 4);
        let decoded = decode_b16(bits as u32, true, true, 1, false, false).unwrap();
        assert_eq!((decoded.m, decoded.n), (128, 256));
    }

    #[test]
    fn dense_encoder_rejects_with_legacy_messages() {
        let err = encode_dense_instr_descriptor_fields(
            "float32", "int8", "bfloat16", 128, 64, 16, false, false, 1, false, false, false, false,
        )
        .unwrap_err();
        assert!(err.0.starts_with("tcgen05 dense instruction descriptor has invalid dtype"));
        let err = validate_tcgen05_instruction_shape("f16", 3, 128, 64, 16, false).unwrap_err();
        assert_eq!(err.0, "tcgen05 instruction descriptor cta_group must be 1 or 2, got 3");
        let err = validate_tcgen05_instruction_shape("f16", 1, 128, 12, 16, false).unwrap_err();
        assert_eq!(
            err.0,
            "invalid tcgen05 descriptor shape kind=f16, cta_group=1, M=128, N=12, K=16"
        );
    }

    #[test]
    fn block_scaled_descriptor_sets_k96_and_scale_format_bits() {
        let bits = encode_block_scaled_instr_descriptor_fields(
            "float32",
            "float4_e2m1fn",
            "float4_e2m1fn",
            "float8_e4m3fn",
            "float8_e4m3fn",
            128,
            128,
            96,
            false,
            false,
            1,
            false,
            false,
            false,
        )
        .unwrap();
        assert_eq!(bits >> 31, 1);
        assert_eq!((bits >> 23) & 1, 0); // nvf4 ue4m3 scales
        let mx = encode_block_scaled_instr_descriptor_fields(
            "float32",
            "float8_e4m3fn",
            "float8_e5m2",
            "float8_e8m0fnu",
            "float8_e8m0fnu",
            128,
            64,
            32,
            false,
            false,
            1,
            false,
            false,
            false,
        )
        .unwrap();
        assert_eq!(((mx >> 7) & 7, (mx >> 10) & 7, (mx >> 23) & 1), (0, 1, 1));
        let decoded = decode_mxf8f6f4(mx as u32, 1, crate::tcgen05::smem_desc::MatrixDescriptorLayout::Sm100);
        assert!(decoded.is_ok(), "{decoded:?}");
    }

    #[test]
    fn matrix_descriptor_truncates_fields_and_maps_swizzle() {
        let value = encode_matrix_descriptor(0x0300_1230, 0x4001, 0x8002, 1);
        assert_eq!(value & 0x3fff, u64::from((0x0300_1230_u32 >> 4) & 0x3fff));
        assert_eq!((value >> 16) & 0x3fff, 1);
        assert_eq!((value >> 32) & 0x3fff, 2);
        assert_eq!((value >> 46) & 1, 1);
        assert_eq!(value >> 61, 6);
        assert_eq!(encode_matrix_descriptor(0, 0, 0, 9) >> 61, 0);
        assert_eq!(encode_matrix_descriptor(0, 0, 0, 4) >> 61, 1);
    }
}
