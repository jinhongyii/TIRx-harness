//! `cta_group::2` forms.

use super::*;

#[test]
fn cta2_bf16_ss_matches_hand_reference() {
    for m in [128_usize, 256] {
        let (n, k) = (32, 16);
        let rows = m / 2;
        let mut machine = Machine::new();
        let mut descriptors = (0, 0);
        for cta in 0..2 {
            machine.on(cta as u32);
            descriptors.0 = place_b16(
                &mut machine,
                0x1000,
                rows,
                k,
                |r, kk| a_val(cta * rows + r, kk),
                true,
            );
            descriptors.1 = place_b16(
                &mut machine,
                0x8000,
                n / 2,
                k,
                |r, kk| b_val(cta * n / 2 + r, kk),
                true,
            );
        }
        let idesc = idesc_cta2(
            "float32", "bfloat16", "bfloat16", m as i64, n as i64, k as i64,
        );
        let p = cta2(payload(
            TcMmaKind::F16,
            TcA::Smem(op()),
            descriptors.0,
            descriptors.1,
            idesc,
            false,
        ));
        machine.run(&p).unwrap();
        assert_eq!(
            read_pair(&mut machine, m, n, f32::from_le_bytes),
            reference(m, n, k, |_, _| 0.0),
            "M={m}"
        );
        // The single-CTA wrapper cannot reach the pair.
        assert_eq!(
            machine.run_single(&p).unwrap_err().kind,
            OpErrorKind::Unsupported
        );
    }
}

#[test]
fn cta2_f8_ts_respects_the_pair_lane_mask() {
    let (m, n, k) = (256, 32, 32);
    let rows = m / 2;
    let mut machine = Machine::new();
    let mut b_desc = 0;
    for cta in 0..2_usize {
        machine.on(cta as u32);
        for r in 0..rows {
            for w in 0..8 {
                let word = (0..4).fold(0_u32, |acc, byte| {
                    acc | u32::from(f32_to_float8_e4m3fn_bits(a_val(
                        cta * rows + r,
                        4 * w + byte,
                    ))) << (8 * byte)
                });
                machine.set(r as u32, A_COL + w as u32, word.to_le_bytes());
            }
        }
        b_desc = machine.place(0x8000, n / 2, k, |r, kk| {
            f32_to_float8_e4m3fn_bits(b_val(cta * n / 2 + r, kk))
        });
    }
    let idesc = idesc_cta2(
        "float32",
        "float8_e4m3fn",
        "float8_e4m3fn",
        m as i64,
        n as i64,
        k as i64,
    );
    let mut p = cta2(payload(
        TcMmaKind::F8f6f4,
        TcA::Tmem(op()),
        u64::from(A_COL),
        b_desc,
        idesc,
        false,
    ));
    // Words 4..8 are CTA 1: disable its lane 5 (row 133).
    p.disable_output_lane = vec![0, 0, 0, 0, 1 << 5, 0, 0, 0];
    p.args.disable_output_lane = vec![op(); 8];
    machine.run(&p).unwrap();
    let expected = reference(m, n, k, |_, _| 0.0);
    for i in 0..m {
        for j in 0..n {
            let (cta, lane, column) = pair_cell(m, n, i, j);
            let cell = machine.on(cta).get(lane, column);
            if i == rows + 5 {
                assert_eq!(cell, None);
            } else {
                assert_eq!(
                    f32::from_le_bytes(cell.unwrap()),
                    expected[i * n + j],
                    "({i},{j})"
                );
            }
        }
    }
    // A mask of the wrong length for the group is invalid.
    p.disable_output_lane.truncate(4);
    assert_eq!(machine.run(&p).unwrap_err().kind, OpErrorKind::Invalid);
}

#[test]
fn cta2_i8_ss_accumulates_exactly() {
    let (m, n, k) = (256, 32, 32);
    let rows = m / 2;
    let a_int = |i: usize, kk: usize| ((i * 37 + kk * 11) % 256) as i32 - 128;
    let b_int = |j: usize, kk: usize| ((j * 13 + kk * 29) % 256) as i32 - 128;
    let mut machine = Machine::new();
    let (mut a_desc, mut b_desc) = (0, 0);
    for cta in 0..2 {
        machine.on(cta as u32);
        a_desc = machine.place(0x1000, rows, k, |r, kk| {
            a_int(cta * rows + r, kk) as i8 as u8
        });
        b_desc = machine.place(0x8000, n / 2, k, |r, kk| {
            b_int(cta * n / 2 + r, kk) as i8 as u8
        });
        for r in 0..rows {
            for j in 0..n {
                machine.set(
                    r as u32,
                    D_COL + j as u32,
                    (r as i32 * 1000 - j as i32).to_le_bytes(),
                );
            }
        }
    }
    let idesc = idesc_cta2("int32", "int8", "int8", m as i64, n as i64, k as i64);
    machine
        .run(&cta2(payload(
            TcMmaKind::I8,
            TcA::Smem(op()),
            a_desc,
            b_desc,
            idesc,
            true,
        )))
        .unwrap();
    let got = read_pair(&mut machine, m, n, i32::from_le_bytes);
    for i in 0..m {
        for j in 0..n {
            let d_in = (i % rows) as i64 * 1000 - j as i64;
            let dot = (0..k)
                .map(|kk| i64::from(a_int(i, kk) * b_int(j, kk)))
                .sum::<i64>();
            assert_eq!(i64::from(got[i * n + j]), d_in + dot);
        }
    }
}

#[test]
fn cta2_mxf8f6f4_uses_per_cta_sfa_and_joint_sfb() {
    let (m, n, k) = (256, 16, 32);
    let rows = m / 2;
    let sa = |row: usize| 126 + (row % 3) as u8;
    let sb = |row: usize| 127 + (row % 2) as u8;
    let mut machine = Machine::new();
    let (mut a_desc, mut b_desc) = (0, 0);
    for cta in 0..2 {
        machine.on(cta as u32);
        a_desc = machine.place(0x1000, rows, k, |r, kk| {
            f32_to_float8_e4m3fn_bits(a_val(cta * rows + r, kk))
        });
        b_desc = machine.place(0x8000, n / 2, k, |r, kk| {
            f32_to_float8_e4m3fn_bits(b_val(cta * n / 2 + r, kk))
        });
        place_replicated_scales(&mut machine, SFA_COL, rows, |r| sa(cta * rows + r));
        place_replicated_scales(&mut machine, SFB_COL, n, sb);
    }
    let idesc = encode_block_scaled_instr_descriptor_fields(
        "float32",
        "float8_e4m3fn",
        "float8_e4m3fn",
        "float8_e8m0fnu",
        "float8_e8m0fnu",
        m as i64,
        n as i64,
        k as i64,
        false,
        false,
        2,
        false,
        false,
        false,
    )
    .unwrap() as u32;
    let mut p = cta2(payload(
        TcMmaKind::MxF8f6f4,
        TcA::Smem(op()),
        a_desc,
        b_desc,
        idesc,
        false,
    ));
    p.args.block_scale = Some((op(), op(), 32));
    p.scale_taddrs = Some((SFA_COL, SFB_COL));
    machine.run(&p).unwrap();
    let value = |bits: u8| 2_f32.powi(i32::from(bits) - 127);
    let mut expected = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            expected[i * n + j] = (0..k)
                .map(|kk| a_val(i, kk) * value(sa(i)) * (b_val(j, kk) * value(sb(j))))
                .sum();
        }
    }
    assert_eq!(read_pair(&mut machine, m, n, f32::from_le_bytes), expected);
    // SFB copies must agree across the pair.
    machine.on(1).set(3, SFB_COL, [120, 0, 0, 0]);
    assert_eq!(machine.run(&p).unwrap_err().kind, OpErrorKind::Invalid);
}

#[test]
fn cta2_mxf4_block_scaled() {
    let (m, n, k) = (256, 16, 64);
    let rows = m / 2;
    let e2m1 = [0.0_f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let value = |code: u8| {
        if code < 8 {
            e2m1[code as usize]
        } else {
            -e2m1[(code - 8) as usize]
        }
    };
    let code_a = |i: usize, kk: usize| ((i + 3 * kk) % 16) as u8;
    let code_b = |j: usize, kk: usize| ((5 * j + kk) % 16) as u8;
    let scale_bits = |row: usize, v: usize| 126 + ((row + v) % 3) as u8;
    let mut machine = Machine::new();
    let (mut a_desc, mut b_desc) = (0, 0);
    for cta in 0..2 {
        machine.on(cta as u32);
        a_desc = machine.place(0x1000, rows, k / 2, |r, byte| {
            code_a(cta * rows + r, 2 * byte) | code_a(cta * rows + r, 2 * byte + 1) << 4
        });
        b_desc = machine.place(0x8000, n / 2, k / 2, |r, byte| {
            code_b(cta * n / 2 + r, 2 * byte) | code_b(cta * n / 2 + r, 2 * byte + 1) << 4
        });
        // SFA: this CTA's rows; SFB: the joint N rows (each CTA reads its half).
        for r in 0..rows {
            let g = cta * rows + r;
            machine.set(
                (r % 32) as u32,
                SFA_COL + (r / 32) as u32,
                [scale_bits(g, 0), scale_bits(g, 1), 0, 0],
            );
        }
        for r in 0..n {
            machine.set(
                r as u32,
                SFB_COL,
                [scale_bits(r, 0), scale_bits(r, 1), 0, 0],
            );
        }
    }
    let idesc = encode_block_scaled_instr_descriptor_fields(
        "float32",
        "float4_e2m1fn",
        "float4_e2m1fn",
        "float8_e8m0fnu",
        "float8_e8m0fnu",
        m as i64,
        n as i64,
        k as i64,
        false,
        false,
        2,
        false,
        false,
        false,
    )
    .unwrap() as u32;
    let mut p = cta2(payload(
        TcMmaKind::MxF4,
        TcA::Smem(op()),
        a_desc,
        b_desc,
        idesc,
        false,
    ));
    p.args.block_scale = Some((op(), op(), 32));
    p.scale_taddrs = Some((SFA_COL, SFB_COL));
    machine.run(&p).unwrap();
    let scale = |bits: u8| 2_f32.powi(i32::from(bits) - 127);
    let mut expected = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            expected[i * n + j] = (0..k).fold(0.0_f32, |acc, kk| {
                let a = value(code_a(i, kk)) * scale(scale_bits(i, kk / 32));
                let b = value(code_b(j, kk)) * scale(scale_bits(j, kk / 32));
                a.mul_add(b, acc)
            });
        }
    }
    assert_eq!(read_pair(&mut machine, m, n, f32::from_le_bytes), expected);
}
