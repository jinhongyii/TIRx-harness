//! CTA1 dense and block-scaled MMA numerics.

use super::*;

// ---------------------------------------------------------------------------
// MMA numerics
// ---------------------------------------------------------------------------

#[test]
fn bf16_ss_matches_hand_reference() {
    for (m, n) in [(128_usize, 16_usize), (64, 8)] {
        let k = 16;
        let mut machine = Machine::new();
        let a_desc = place_b16(&mut machine, 0x1000, m, k, a_val, true);
        let b_desc = place_b16(&mut machine, 0x8000, n, k, b_val, true);
        let idesc = dense_idesc(
            "float32", "bfloat16", "bfloat16", m as i64, n as i64, k as i64,
        );
        machine
            .run(&payload(
                TcMmaKind::F16,
                TcA::Smem(op()),
                a_desc,
                b_desc,
                idesc,
                false,
            ))
            .unwrap();
        assert_eq!(
            read_f32_tile(&machine, m, n),
            reference(m, n, k, |_, _| 0.0),
            "M={m} N={n}"
        );
    }
}

#[test]
fn enable_input_d_accumulates_with_scale() {
    let (m, n, k) = (128, 16, 16);
    let mut machine = Machine::new();
    let a_desc = place_b16(&mut machine, 0x1000, m, k, a_val, false);
    let b_desc = place_b16(&mut machine, 0x8000, n, k, b_val, false);
    let d_in = |i: usize, j: usize| (i as f32) - (j as f32) * 4.0;
    for i in 0..m {
        for j in 0..n {
            machine.set(i as u32, D_COL + j as u32, d_in(i, j).to_le_bytes());
        }
    }
    let idesc = dense_idesc(
        "float32", "float16", "float16", m as i64, n as i64, k as i64,
    );
    let mut p = payload(TcMmaKind::F16, TcA::Smem(op()), a_desc, b_desc, idesc, true);
    machine.run(&p).unwrap();
    let once = reference(m, n, k, d_in);
    assert_eq!(read_f32_tile(&machine, m, n), once);
    // scale-input-d = 1 halves D before the FMA chain.
    p.scale_input_d = Some(1);
    p.args.scale_input_d = Some(op());
    machine.run(&p).unwrap();
    assert_eq!(
        read_f32_tile(&machine, m, n),
        reference(m, n, k, |i, j| once[i * n + j] / 2.0)
    );
}

#[test]
fn f16_ts_with_disabled_output_lane() {
    let (m, n, k) = (64, 8, 16);
    let mut machine = Machine::new();
    // TMEM A (Layout F, M=64): eight packed words per row, two halves each.
    for i in 0..m {
        for w in 0..8 {
            let lo = f32_to_fp16_bits(a_val(i, 2 * w)) as u32;
            let hi = f32_to_fp16_bits(a_val(i, 2 * w + 1)) as u32;
            machine.set(
                layout_f_lane(i).unwrap() as u32,
                A_COL + w as u32,
                (lo | hi << 16).to_le_bytes(),
            );
        }
    }
    let b_desc = place_b16(&mut machine, 0x8000, n, k, b_val, false);
    let idesc = dense_idesc(
        "float32", "float16", "float16", m as i64, n as i64, k as i64,
    );
    let mut p = payload(
        TcMmaKind::F16,
        TcA::Tmem(op()),
        u64::from(A_COL),
        b_desc,
        idesc,
        false,
    );
    // Disable D lane 1 (row 1) and lane 32 (row 16).
    p.disable_output_lane = vec![1 << 1, 1, 0, 0];
    p.args.disable_output_lane = vec![op(); 4];
    machine.run(&p).unwrap();
    let expected = reference(m, n, k, |_, _| 0.0);
    for i in 0..m {
        for j in 0..n {
            let cell = machine.get(d_lane(m, i), D_COL + j as u32);
            if i == 1 || i == 16 {
                assert_eq!(cell, None, "disabled row {i} must not be written");
            } else {
                assert_eq!(f32::from_le_bytes(cell.unwrap()), expected[i * n + j]);
            }
        }
    }
}

#[test]
fn tf32_ss_truncates_storage_bits() {
    let (m, n, k) = (128, 16, 8);
    let mut machine = Machine::new();
    // Low mantissa bits are dropped (tf32 storage), not rounded.
    let a_desc = machine.place(0x1000, m, k * 4, |row, byte| {
        (f32::from_bits(a_val(row, byte / 4).to_bits() | 0x1fff)).to_le_bytes()[byte % 4]
    });
    let b_desc = machine.place(0x8000, n, k * 4, |row, byte| {
        b_val(row, byte / 4).to_le_bytes()[byte % 4]
    });
    let idesc = dense_idesc("float32", "tf32", "tf32", m as i64, n as i64, k as i64);
    machine
        .run(&payload(
            TcMmaKind::Tf32,
            TcA::Smem(op()),
            a_desc,
            b_desc,
            idesc,
            false,
        ))
        .unwrap();
    assert_eq!(
        read_f32_tile(&machine, m, n),
        reference(m, n, k, |_, _| 0.0)
    );
}

#[test]
fn f8_e4m3_ss_and_f16_destination() {
    let (m, n, k) = (128, 16, 32);
    let mut machine = Machine::new();
    let a_desc = machine.place(0x1000, m, k, |row, kk| {
        f32_to_float8_e4m3fn_bits(a_val(row, kk))
    });
    let b_desc = machine.place(0x8000, n, k, |row, kk| {
        f32_to_float8_e4m3fn_bits(b_val(row, kk))
    });
    let idesc = dense_idesc(
        "float32",
        "float8_e4m3fn",
        "float8_e4m3fn",
        m as i64,
        n as i64,
        k as i64,
    );
    machine
        .run(&payload(
            TcMmaKind::F8f6f4,
            TcA::Smem(op()),
            a_desc,
            b_desc,
            idesc,
            false,
        ))
        .unwrap();
    let expected = reference(m, n, k, |_, _| 0.0);
    assert_eq!(read_f32_tile(&machine, m, n), expected);
    // `.f16` D: one RNE conversion on store, high half zero.
    let idesc = dense_idesc(
        "float16",
        "float8_e4m3fn",
        "float8_e4m3fn",
        m as i64,
        n as i64,
        k as i64,
    );
    machine
        .run(&payload(
            TcMmaKind::F8f6f4,
            TcA::Smem(op()),
            a_desc,
            b_desc,
            idesc,
            false,
        ))
        .unwrap();
    let cell = machine.get(5, D_COL + 3).unwrap();
    let [lo, hi] = f32_to_fp16_bits(expected[5 * n + 3]).to_le_bytes();
    assert_eq!(cell, [lo, hi, 0, 0]);
}

#[test]
fn i8_ss_and_ts_exact_with_saturation() {
    let (m, n, k) = (128, 16, 32);
    let a_int = |i: usize, kk: usize| ((i * 37 + kk * 11) % 256) as i32 - 128; // s8
    let b_int = |j: usize, kk: usize| ((j * 13 + kk * 29) % 256) as i32; // u8
    let mut machine = Machine::new();
    let a_desc = machine.place(0x1000, m, k, |row, kk| a_int(row, kk) as i8 as u8);
    let b_desc = machine.place(0x8000, n, k, |row, kk| b_int(row, kk) as u8);
    let idesc = dense_idesc("int32", "int8", "uint8", m as i64, n as i64, k as i64);
    let dot = |i: usize, j: usize| {
        (0..k)
            .map(|kk| i64::from(a_int(i, kk)) * i64::from(b_int(j, kk)))
            .sum::<i64>()
    };
    let read = |machine: &Machine, i: usize, j: usize| {
        i32::from_le_bytes(machine.get(i as u32, D_COL + j as u32).unwrap())
    };
    machine
        .run(&payload(
            TcMmaKind::I8,
            TcA::Smem(op()),
            a_desc,
            b_desc,
            idesc,
            false,
        ))
        .unwrap();
    for i in 0..m {
        for j in 0..n {
            assert_eq!(i64::from(read(&machine, i, j)), dot(i, j));
        }
    }
    // TMEM A, accumulate into D = i32::MAX - 5 with and without .satfinite.
    for i in 0..m {
        for w in 0..8 {
            let word = (0..4).fold(0_u32, |acc, b| {
                acc | u32::from(a_int(i, 4 * w + b) as i8 as u8) << (8 * b)
            });
            machine.set(i as u32, A_COL + w as u32, word.to_le_bytes());
        }
        for j in 0..n {
            machine.set(i as u32, D_COL + j as u32, (i32::MAX - 5).to_le_bytes());
        }
    }
    machine
        .run(&payload(
            TcMmaKind::I8,
            TcA::Tmem(op()),
            u64::from(A_COL),
            b_desc,
            idesc,
            true,
        ))
        .unwrap();
    for i in 0..m {
        for j in 0..n {
            let wide = i64::from(i32::MAX - 5) + dot(i, j);
            assert_eq!(
                read(&machine, i, j),
                wide as i32,
                "wraps without .satfinite"
            );
            machine.set(i as u32, D_COL + j as u32, (i32::MAX - 5).to_le_bytes());
        }
    }
    let sat = encode_dense_instr_descriptor_fields(
        "int32", "int8", "uint8", m as i64, n as i64, k as i64, false, false, 1, false, false,
        true, false,
    )
    .unwrap() as u32;
    machine
        .run(&payload(
            TcMmaKind::I8,
            TcA::Tmem(op()),
            u64::from(A_COL),
            b_desc,
            sat,
            true,
        ))
        .unwrap();
    for i in 0..m {
        for j in 0..n {
            let wide = i64::from(i32::MAX - 5) + dot(i, j);
            assert_eq!(
                i64::from(read(&machine, i, j)),
                wide.clamp(i64::from(i32::MIN), i64::from(i32::MAX))
            );
        }
    }
}

#[test]
fn mxf8f6f4_block_scaled_e4m3() {
    let (m, n, k) = (128, 8, 32);
    let mut machine = Machine::new();
    let a_desc = machine.place(0x1000, m, k, |row, kk| {
        f32_to_float8_e4m3fn_bits(a_val(row, kk))
    });
    let b_desc = machine.place(0x8000, n, k, |row, kk| {
        f32_to_float8_e4m3fn_bits(b_val(row, kk))
    });
    // UE8M0: 127 = 1.0, 128 = 2.0, 126 = 0.5.
    let sa = |row: usize| 126 + (row % 3) as u8;
    let sb = |row: usize| 127 + (row % 2) as u8;
    place_replicated_scales(&mut machine, SFA_COL, m, sa);
    place_replicated_scales(&mut machine, SFB_COL, n, sb);
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
        1,
        false,
        false,
        false,
    )
    .unwrap() as u32;
    let mut p = payload(
        TcMmaKind::MxF8f6f4,
        TcA::Smem(op()),
        a_desc,
        b_desc,
        idesc,
        false,
    );
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
    assert_eq!(read_f32_tile(&machine, m, n), expected);
}

#[test]
fn mxf4_block_scaled_e2m1() {
    let (m, n, k) = (128, 8, 64);
    // E2M1 codes 0..7 = 0, 0.5, 1, 1.5, 2, 3, 4, 6; 8..15 negative.
    let e2m1 = [0.0_f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let code_a = |i: usize, kk: usize| ((i + 3 * kk) % 16) as u8;
    let code_b = |j: usize, kk: usize| ((5 * j + kk) % 16) as u8;
    let value = |code: u8| {
        if code < 8 {
            e2m1[code as usize]
        } else {
            -e2m1[(code - 8) as usize]
        }
    };
    let mut machine = Machine::new();
    let pack = |code: &dyn Fn(usize, usize) -> u8, row: usize, byte: usize| {
        code(row, 2 * byte) | code(row, 2 * byte + 1) << 4
    };
    let a_desc = machine.place(0x1000, m, k / 2, |row, byte| pack(&code_a, row, byte));
    let b_desc = machine.place(0x8000, n, k / 2, |row, byte| pack(&code_b, row, byte));
    // Vec2x UE8M0 (block 32): row r, vector v at lane r % 32, column base + r / 32, byte v.
    let scale_bits = |row: usize, v: usize| 126 + ((row + v) % 3) as u8;
    for (col, rows) in [(SFA_COL, m), (SFB_COL, n)] {
        for row in 0..rows {
            let lane = (row % 32) as u32;
            let column = col + (row / 32) as u32;
            machine.set(lane, column, [scale_bits(row, 0), scale_bits(row, 1), 0, 0]);
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
        1,
        false,
        false,
        false,
    )
    .unwrap() as u32;
    let mut p = payload(
        TcMmaKind::MxF4,
        TcA::Smem(op()),
        a_desc,
        b_desc,
        idesc,
        false,
    );
    p.args.block_scale = Some((op(), op(), 32));
    p.scale_taddrs = Some((SFA_COL, SFB_COL));
    machine.run(&p).unwrap();
    let scale = |bits: u8| 2_f32.powi(i32::from(bits) - 127);
    let mut expected = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for kk in 0..k {
                let a = value(code_a(i, kk)) * scale(scale_bits(i, kk / 32));
                let b = value(code_b(j, kk)) * scale(scale_bits(j, kk / 32));
                acc = a.mul_add(b, acc);
            }
            expected[i * n + j] = acc;
        }
    }
    assert_eq!(read_f32_tile(&machine, m, n), expected);
}
