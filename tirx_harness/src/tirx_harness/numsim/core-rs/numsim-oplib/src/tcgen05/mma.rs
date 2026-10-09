//! Numeric cores of tcgen05.mma: gathered operands (+ input D) -> accumulator.
//!
//! Legacy source: `engine-rs/src/runtime/tcgen_ops.rs` (
//! `raw_tcgen05_sparse_2of4_indices`, `raw_tcgen05_expand_sparse_2of4`,
//! `raw_tcgen05_expand_sparse_mxf4_a`, the numeric closure of
//! `raw_tcgen05_sparse_float_tail` and `RawMmaTail::run{,_with}`). The legacy
//! tile-GEMM compute half (`tile_gemm_bf16_f32_ss_cta1`) was removed with the
//! tile layer: `tirx.tile.gemm{,_async}` lowers through TVM dispatch to
//! tcgen05.mma (decision 6).
//!
//! Every accumulation is the stable logical oracle: each output element is a
//! binary32 FMA chain over increasing K, seeded with `D * scale`. The f32
//! cores (`mma_f32_abt_increasing_k`, banked variant) live in `crate::mma`
//! (moved there by W4-mma from legacy `tcgen_ops.rs:4616-4796`).

use super::instr_desc::FloatKind;
use super::layouts::{DenseTmemLayout, CTA1_PACKED_A_COLUMNS};
use crate::mma::{mma_f32_abt_banked_a_increasing_k, mma_f32_abt_increasing_k};
use crate::types::{OpError, OpResult};

/// The dense tail's numeric choice: banked core when A came from TMEM in a
/// multi-bank layout, else the plain core (legacy `RawMmaTail::run{,_with}`).
/// `shape` is `(m, n, k)`.
pub fn mma_dense_tail(
    (m, n, k): (usize, usize, usize),
    a_values: &[f32],
    b_values: &[f32],
    input_d: Option<(&[f32], f32)>,
    a_in_tmem: bool,
    layout: DenseTmemLayout,
) -> OpResult<Vec<f32>> {
    if a_in_tmem && layout.packed_a_banks() > 1 {
        mma_f32_abt_banked_a_increasing_k(
            m,
            n,
            k,
            a_values,
            b_values,
            input_d,
            layout.packed_a_banks(),
        )
    } else {
        mma_f32_abt_increasing_k(m, n, k, a_values, b_values, input_d)
    }
}

/// The two dense indices (0..4, low code bits first) a 2:4 metadata nibble selects;
/// errors on codes PTX leaves undefined.
pub fn sparse_2of4_indices(code: u8) -> OpResult<[usize; 2]> {
    if !matches!(code, 0x4 | 0x8 | 0xc | 0x9 | 0xd | 0x6 | 0xe) {
        return Err(OpError::message(format!(
            "sparse 2:4 metadata code 0x{code:x} is not a defined index pair"
        )));
    }
    Ok([usize::from(code & 3), usize::from((code >> 2) & 3)])
}

/// Expand packed 2:4 A rows using `metadata(physical_row, chunk)` codes.
pub fn expand_sparse_2of4<Scalar: Copy + Default>(
    packed: &[Scalar],
    rows: usize,
    layout: DenseTmemLayout,
    k: usize,
    metadata: impl Fn(usize, usize) -> OpResult<u8>,
) -> OpResult<Vec<Scalar>> {
    let banks = layout.packed_a_banks();
    let packed_banks = packed.len() / (rows * (k / 2));
    if !matches!(packed_banks, 1) && packed_banks != banks
        || !packed.len().is_multiple_of(rows * (k / 2))
    {
        return Err(OpError::message("sparse 2:4 packed A has the wrong shape"));
    }
    let mut dense = vec![Scalar::default(); banks * rows * k];
    for bank in 0..banks {
        for row in 0..rows {
            let physical_row = layout
                .packed_a_location(bank, row, 0, rows, CTA1_PACKED_A_COLUMNS)?
                .0;
            for chunk in 0..(k / 4) {
                let code = metadata(physical_row, chunk)?;
                let [first, second] = sparse_2of4_indices(code)?;
                let source = ((bank % packed_banks) * rows + row) * (k / 2) + chunk * 2;
                let destination = (bank * rows + row) * k + chunk * 4;
                dense[destination + first] = packed[source];
                dense[destination + second] = packed[source + 1];
            }
        }
    }
    Ok(dense)
}

/// Expand the sparse `mxf4` 128x64 packed A to 128x128 using 4-bit codes
/// `metadata(row, chunk)` (legacy `raw_tcgen05_expand_sparse_mxf4_a`).
pub fn expand_sparse_mxf4_a(
    packed: &[f32],
    mut metadata: impl FnMut(usize, usize) -> OpResult<u8>,
) -> OpResult<Vec<f32>> {
    const M: usize = 128;
    const K: usize = 128;
    if packed.len() != M * (K / 2) {
        return Err(OpError::message("sparse mxf4 packed A has the wrong shape"));
    }
    let mut dense = vec![0.0_f32; M * K];
    for row in 0..M {
        for chunk in 0..(K / 8) {
            let code = metadata(row, chunk)?;
            let first_pair = usize::from(code & 0x3);
            let second_pair = usize::from((code >> 2) & 0x3);
            if first_pair == second_pair {
                return Err(OpError::message(format!(
                    "sparse mxf4 metadata code 0x{code:x} repeats one pair"
                )));
            }
            let packed_base = row * (K / 2) + chunk * 4;
            let dense_base = row * K + chunk * 8;
            dense[dense_base + first_pair * 2] = packed[packed_base];
            dense[dense_base + first_pair * 2 + 1] = packed[packed_base + 1];
            dense[dense_base + second_pair * 2] = packed[packed_base + 2];
            dense[dense_base + second_pair * 2 + 1] = packed[packed_base + 3];
        }
    }
    Ok(dense)
}

/// Sparse floating MMA numeric core (legacy closure of
/// `raw_tcgen05_sparse_float_tail`). Implicit zeros never multiply B: the
/// metadata selects B's K terms and the packed A values feed the FMA core.
///
/// `metadata_code(cta_index, physical_row, chunk)` reads one 4-bit code in
/// the given CTA of the group (`cta_index` in `0..cta_group`).
#[allow(clippy::too_many_arguments)]
pub fn sparse_float_mma(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    d: Option<(&[f32], f32)>,
    kind: FloatKind,
    layout: DenseTmemLayout,
    cta_group: usize,
    mut metadata_code: impl FnMut(usize, usize, usize) -> OpResult<u8>,
) -> OpResult<Vec<f32>> {
    let packed_k = k / 2;
    let rows = m / cta_group;
    let banks = layout.packed_a_banks();
    let bank_columns = n / banks;
    let a_banks = a.len() / (m * packed_k);
    if !a.len().is_multiple_of(m * packed_k) || !(a_banks == 1 || a_banks == banks) {
        return Err(OpError::message(
            "sparse floating MMA packed A has the wrong shape",
        ));
    }
    let mut output = vec![0.0; m * n];
    let mut selected_b = vec![0.0; bank_columns * packed_k];
    for cta in 0..cta_group {
        for local_row in 0..rows {
            let row = cta * rows + local_row;
            for bank in 0..banks {
                let physical_row = layout.packed_a_location(bank, local_row, 0, rows, 8)?.0;
                for inner in 0..packed_k {
                    let code = metadata_code(
                        cta,
                        physical_row,
                        if matches!(kind, FloatKind::Tf32) {
                            inner
                        } else {
                            inner / 2
                        },
                    )?;
                    let selected = kind.sparse_index(code, inner)?;
                    for col in 0..bank_columns {
                        selected_b[col * packed_k + inner] =
                            b[(bank * bank_columns + col) * k + selected];
                    }
                }
                let a_start = ((bank % a_banks) * m + row) * packed_k;
                let start = row * n + bank * bank_columns;
                let end = start + bank_columns;
                let product = mma_f32_abt_increasing_k(
                    1,
                    bank_columns,
                    packed_k,
                    &a[a_start..a_start + packed_k],
                    &selected_b,
                    d.map(|(values, scale)| (&values[start..end], scale)),
                )?;
                output[start..end].copy_from_slice(&product);
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_2of4_expansion_places_pairs() {
        let packed = [1.0_f32, 2.0, 3.0, 4.0];
        // rows=1 is not a legal layout row count; use Layout D with 1 row via F? D accepts any.
        let dense = expand_sparse_2of4(&packed, 1, DenseTmemLayout::D, 8, |_, chunk| {
            Ok(if chunk == 0 { 0x4 } else { 0xe })
        })
        .unwrap();
        assert_eq!(dense, vec![1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0]);
        assert!(sparse_2of4_indices(0x0).is_err());
    }
}
