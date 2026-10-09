//! `.sp` forms (2:4 / narrow metadata, `.ti16`, sparse mxf4).

use super::*;

#[test]
fn sparse_bf16_selects_b_by_2of4_metadata() {
    let (m, n, k) = (128, 16, 32);
    let packed = k / 2;
    let mut machine = Machine::new();
    let a_desc = place_b16(&mut machine, 0x1000, m, packed, a_val, true);
    let b_desc = place_b16(&mut machine, 0x8000, n, k, b_val, true);
    let code = |row: usize, chunk: usize| CODES_2OF4[(row * 3 + chunk) % 7];
    let layout = SparseMetadataLayout::B16 { selector: 0 };
    for row in 0..m {
        for chunk in 0..k / 4 {
            set_code(
                &mut machine,
                sparse_metadata_location(META_COL, layout, row, chunk).unwrap(),
                code(row, chunk).0,
            );
        }
    }
    let idesc = encode_dense_instr_descriptor_fields(
        "float32", "bfloat16", "bfloat16", m as i64, n as i64, k as i64, false, false, 1, false,
        false, false, true,
    )
    .unwrap() as u32;
    machine
        .run(&sparse(payload(
            TcMmaKind::F16,
            TcA::Smem(op()),
            a_desc,
            b_desc,
            idesc,
            false,
        )))
        .unwrap();
    let mut expected = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            expected[i * n + j] = (0..packed)
                .map(|p| {
                    let selected = (p / 2) * 4 + code(i, p / 2).1[p % 2];
                    a_val(i, p) * b_val(j, selected)
                })
                .sum();
        }
    }
    assert_eq!(read_f32_tile(&machine, m, n), expected);
    // An undefined code is an operand error.
    set_code(
        &mut machine,
        sparse_metadata_location(META_COL, layout, 7, 2).unwrap(),
        0x0,
    );
    let p = sparse(payload(
        TcMmaKind::F16,
        TcA::Smem(op()),
        a_desc,
        b_desc,
        idesc,
        false,
    ));
    assert_eq!(machine.run(&p).unwrap_err().kind, OpErrorKind::Invalid);
}

#[test]
fn sparse_f8f6f4_uses_narrow_metadata() {
    let (m, n, k) = (128, 16, 64);
    let packed = k / 2;
    let mut machine = Machine::new();
    let a_desc = machine.place(0x1000, m, packed, |r, kk| {
        f32_to_float8_e4m3fn_bits(a_val(r, kk))
    });
    let b_desc = machine.place(0x8000, n, k, |r, kk| {
        f32_to_float8_e4m3fn_bits(b_val(r, kk))
    });
    let code = |row: usize, chunk: usize| CODES_2OF4[(row + 5 * chunk) % 7];
    let layout = SparseMetadataLayout::Narrow { k };
    for row in 0..m {
        for chunk in 0..k / 4 {
            set_code(
                &mut machine,
                sparse_metadata_location(META_COL, layout, row, chunk).unwrap(),
                code(row, chunk).0,
            );
        }
    }
    let idesc = encode_dense_instr_descriptor_fields(
        "float32",
        "float8_e4m3fn",
        "float8_e4m3fn",
        m as i64,
        n as i64,
        k as i64,
        false,
        false,
        1,
        false,
        false,
        false,
        true,
    )
    .unwrap() as u32;
    machine
        .run(&sparse(payload(
            TcMmaKind::F8f6f4,
            TcA::Smem(op()),
            a_desc,
            b_desc,
            idesc,
            false,
        )))
        .unwrap();
    let mut expected = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            expected[i * n + j] = (0..packed)
                .map(|p| a_val(i, p) * b_val(j, (p / 2) * 4 + code(i, p / 2).1[p % 2]))
                .sum();
        }
    }
    assert_eq!(read_f32_tile(&machine, m, n), expected);
}

#[test]
fn ti16_dense_negate_and_sparse() {
    let (m, n) = (128, 16);
    let ti16 = TcMmaOptions {
        ti16: true,
        ..TcMmaOptions::default()
    };
    let a_int = |i: usize, kk: usize| ((i * 7 + kk * 3) % 41) as i32 - 20;
    let b_int = |j: usize, kk: usize| ((j * 5 + kk * 11) % 2047) as i32 - 1023;
    let read = |machine: &Machine, i: usize, j: usize| {
        i32::from_le_bytes(machine.get(i as u32, D_COL + j as u32).unwrap())
    };
    let base = |m: usize, n: usize| {
        ((2 << 4) | (3 << 7) | (3 << 10) | ((n >> 3) << 17) | ((m >> 4) << 24)) as u32
    };
    // Dense K=16 with negate A.
    let k = 16;
    let mut machine = Machine::new();
    let place = |machine: &mut Machine, start, rows, k, f: &dyn Fn(usize, usize) -> i32| {
        machine.place(start, rows, k * 2, |r, byte| {
            ti16_bits(f(r, byte / 2)).to_le_bytes()[byte % 2]
        })
    };
    let a_desc = place(&mut machine, 0x1000, m, k, &a_int);
    let b_desc = place(&mut machine, 0x8000, n, k, &b_int);
    let idesc = base(m, n) | 1 << 13;
    machine
        .run_with(
            &payload(TcMmaKind::I8, TcA::Smem(op()), a_desc, b_desc, idesc, false),
            &ti16,
        )
        .unwrap();
    for i in 0..m {
        for j in 0..n {
            let dot: i32 = (0..k).map(|kk| -a_int(i, kk) * b_int(j, kk)).sum();
            assert_eq!(read(&machine, i, j), dot);
        }
    }
    // Without `.ti16` the same descriptor is not a kind::i8 descriptor.
    let plain = payload(TcMmaKind::I8, TcA::Smem(op()), a_desc, b_desc, idesc, false);
    assert_eq!(machine.run(&plain).unwrap_err().kind, OpErrorKind::Invalid);
    // Sparse: 16 packed A values select from 32 B values.
    let k = 32;
    let mut machine = Machine::new();
    let a_desc = place(&mut machine, 0x1000, m, k / 2, &a_int);
    let b_desc = place(&mut machine, 0x8000, n, k, &b_int);
    let code = |row: usize, chunk: usize| CODES_2OF4[(2 * row + chunk) % 7];
    let layout = SparseMetadataLayout::B16 { selector: 0 };
    for row in 0..m {
        for chunk in 0..k / 4 {
            set_code(
                &mut machine,
                sparse_metadata_location(META_COL, layout, row, chunk).unwrap(),
                code(row, chunk).0,
            );
        }
    }
    let p = sparse(payload(
        TcMmaKind::I8,
        TcA::Smem(op()),
        a_desc,
        b_desc,
        base(m, n) | 1 << 2,
        false,
    ));
    machine.run_with(&p, &ti16).unwrap();
    for i in 0..m {
        for j in 0..n {
            let dot: i32 = (0..k / 2)
                .map(|p| a_int(i, p) * b_int(j, (p / 2) * 4 + code(i, p / 2).1[p % 2]))
                .sum();
            assert_eq!(read(&machine, i, j), dot);
        }
    }
}

#[test]
fn sparse_mxf4_ss_cta1() {
    let n = 16;
    let e2m1 = [0.0_f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let value = |code: u8| {
        if code < 8 {
            e2m1[code as usize]
        } else {
            -e2m1[(code - 8) as usize]
        }
    };
    let code_a = |i: usize, p: usize| ((i + 3 * p) % 16) as u8;
    let code_b = |j: usize, kk: usize| ((5 * j + kk) % 16) as u8;
    let scale_bits = |row: usize, v: usize| 126 + ((row + v) % 3) as u8;
    let mut machine = Machine::new();
    let a_desc = machine.place(0x1000, 128, 32, |r, byte| {
        code_a(r, 2 * byte) | code_a(r, 2 * byte + 1) << 4
    });
    let b_desc = machine.place(0x8000, n, 64, |r, byte| {
        code_b(r, 2 * byte) | code_b(r, 2 * byte + 1) << 4
    });
    for row in 0..128 {
        machine.set(
            (row % 32) as u32,
            SFA_COL + (row / 32) as u32,
            [scale_bits(row, 0), scale_bits(row, 1), 0, 0],
        );
    }
    for row in 0..n {
        machine.set(
            row as u32,
            SFB_COL,
            [scale_bits(row, 0), scale_bits(row, 1), 0, 0],
        );
    }
    // Pair codes (first, second) per 8-element chunk: distinct pairs of 4.
    let pairs = [(0, 1), (2, 3), (1, 3), (0, 2), (3, 0), (2, 1)];
    let pair = |row: usize, chunk: usize| pairs[(row + chunk) % 6];
    for row in 0..128 {
        for chunk in 0..16 {
            let (first, second) = pair(row, chunk);
            set_code(
                &mut machine,
                (row, (META_COL as usize) + chunk / 8, chunk % 8),
                (first | second << 2) as u8,
            );
        }
    }
    let idesc = (1 << 2) | (1 << 7) | (1 << 10) | ((n as u32 >> 3) << 17) | (1 << 23) | (1 << 27);
    let mut p = sparse(payload(
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
    let mut expected = vec![0.0_f32; 128 * n];
    for i in 0..128 {
        // Dense A: packed element p of chunk c lands in its pair's slot.
        let mut dense = [0.0_f32; 128];
        for chunk in 0..16 {
            let (first, second) = pair(i, chunk);
            for (slot, pair_index) in [first, second].into_iter().enumerate() {
                for half in 0..2 {
                    let p = chunk * 4 + slot * 2 + half;
                    dense[chunk * 8 + pair_index * 2 + half] =
                        value(code_a(i, p)) * scale(scale_bits(i, usize::from(p >= 32)));
                }
            }
        }
        for j in 0..n {
            expected[i * n + j] = (0..128).fold(0.0_f32, |acc, kk| {
                dense[kk].mul_add(
                    value(code_b(j, kk)) * scale(scale_bits(j, usize::from(kk >= 64))),
                    acc,
                )
            });
        }
    }
    assert_eq!(read_f32_tile(&machine, 128, n), expected);
    // Only the legacy SS CTA1 kind::mxf4 sparse form exists.
    let mut tmem_a = p.clone();
    tmem_a.args.a = TcA::Tmem(op());
    assert_eq!(
        machine.run(&tmem_a).unwrap_err().kind,
        OpErrorKind::Unsupported
    );
}
