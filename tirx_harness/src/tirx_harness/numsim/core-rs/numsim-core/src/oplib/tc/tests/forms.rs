//! `.ws`, `.ashift`, `.lut_b`, collectors, per-group decoding and variants.

use super::*;

#[test]
fn ws_zero_column_mask_zeroes_b_rows() {
    let (m, n, k) = (64, 64, 16);
    let mut machine = Machine::new();
    let a_desc = place_b16(&mut machine, 0x1000, m, k, a_val, false);
    let b_desc = place_b16(&mut machine, 0x8000, n, k, b_val, false);
    let idesc = ((1 << 4) | ((n >> 3) << 17) | ((m >> 4) << 24)) as u32;
    let mut p = payload(
        TcMmaKind::F16,
        TcA::Smem(op()),
        a_desc,
        b_desc,
        idesc,
        false,
    );
    p.args.ws = true;
    // Enabled mask, use span 1, zero span 1, start 0: odd B columns are zero.
    let mask = 1_u64 << 39;
    let expected = reference(m, n, k, |_, _| 0.0);
    // Layout E: two 64-lane banks split N.
    let check = |machine: &Machine, zero_odd: bool| {
        for i in 0..m {
            for j in 0..n {
                let lane = (i + 64 * (j / 32)) as u32;
                let got = f32::from_le_bytes(machine.get(lane, D_COL + (j % 32) as u32).unwrap());
                let want = if zero_odd && j % 2 == 1 {
                    0.0
                } else {
                    expected[i * n + j]
                };
                assert_eq!(got, want, "({i},{j})");
            }
        }
    };
    machine.run(&p).unwrap();
    check(&machine, false);
    let options = TcMmaOptions {
        zero_col_mask: Some(mask),
        ..TcMmaOptions::default()
    };
    machine.run_with(&p, &options).unwrap();
    check(&machine, true);
    // The lowering's operand-word fallback gives the same mask.
    p.disable_output_lane = vec![mask as u32, (mask >> 32) as u32];
    machine.tmem.clear();
    machine.run(&p).unwrap();
    check(&machine, true);
    // .ws is CTA1-only.
    assert_eq!(
        machine.run(&cta2(p.clone())).unwrap_err().kind,
        OpErrorKind::Invalid
    );
}

#[test]
fn ashift_shifts_tmem_a_rows_after_the_product() {
    let (m, n, k) = (128, 16, 16);
    let mut machine = Machine::new();
    let word = |i: usize, w: usize| {
        let lo = f32_to_fp16_bits(a_val(i, 2 * w)) as u32;
        let hi = f32_to_fp16_bits(a_val(i, 2 * w + 1)) as u32;
        lo | hi << 16
    };
    for i in 0..m {
        for w in 0..8 {
            machine.set(i as u32, A_COL + w as u32, word(i, w).to_le_bytes());
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
    p.args.ashift = true;
    machine.run(&p).unwrap();
    assert_eq!(
        read_f32_tile(&machine, m, n),
        reference(m, n, k, |_, _| 0.0),
        "product uses unshifted A"
    );
    for i in 0..m {
        let source = if i % 32 == 31 { i } else { i + 1 };
        for w in 0..8 {
            assert_eq!(
                u32::from_le_bytes(machine.get(i as u32, A_COL + w as u32).unwrap()),
                word(source, w),
                "row {i}"
            );
        }
    }
    // .ashift needs TMEM A; sparse CTA-pair M=128 .ashift stays unmodeled.
    let mut smem_a = p.clone();
    smem_a.args.a = TcA::Smem(op());
    assert_eq!(machine.run(&smem_a).unwrap_err().kind, OpErrorKind::Invalid);
    let idesc = encode_dense_instr_descriptor_fields(
        "float32", "float16", "float16", 128, 32, 32, false, false, 2, false, false, false, true,
    )
    .unwrap() as u32;
    let mut sp = cta2(sparse(payload(
        TcMmaKind::F16,
        TcA::Tmem(op()),
        0,
        b_desc,
        idesc,
        false,
    )));
    sp.args.ashift = true;
    let error = machine.run(&sp).unwrap_err();
    assert_eq!(error.kind, OpErrorKind::Unsupported);
    assert!(error.message.contains("tcgen_sparse_m128_ashift_unmodeled"));
}

#[test]
fn lut_b_decodes_three_bit_indices_through_the_table() {
    let (m, n, k) = (128, 16, 64);
    let table = [0.0_f32, 1.0, 2.0, 3.0, -1.0, -2.0, 4.0, 0.5];
    let lut = |group: usize, index: usize| table[(index + group) % 8];
    let index = |j: usize, t: usize| (j * 3 + t * 5) % 8;
    let sm107 = TcMmaOptions {
        arch: TcArch::Sm107,
        lut_b: Some(LUT_COL),
        ..TcMmaOptions::default()
    };
    for segment in 0..2_usize {
        let mut machine = Machine::new();
        let a_desc = machine.place(0x1000, m, k, |r, kk| {
            f32_to_float8_e4m3fn_bits(a_val(r, kk))
        });
        let b_desc = machine.place(0x8000, n, 48, |j, byte| {
            (0..8).fold(0_u8, |acc, bit| {
                let stream = byte * 8 + bit;
                let (t, b) = (stream / 3, stream % 3);
                acc | ((((index(j, t) >> b) & 1) as u8) << bit)
            })
        }) | (segment as u64) << 53;
        for group in 0..n / 8 {
            let bytes: Vec<u8> = (0..8)
                .map(|t| f32_to_float8_e4m3fn_bits(lut(group, t)))
                .collect();
            machine.set(group as u32, LUT_COL, bytes[..4].try_into().unwrap());
            machine.set(group as u32, LUT_COL + 1, bytes[4..].try_into().unwrap());
        }
        let idesc = ((1 << 4) | ((n >> 3) << 17) | ((m >> 4) << 24) | (1 << 29)) as u32;
        let mut p = payload(
            TcMmaKind::F8f6f4,
            TcA::Smem(op()),
            a_desc,
            b_desc,
            idesc,
            false,
        );
        p.args.lut_b = true;
        machine.run_with(&p, &sm107).unwrap();
        let mut expected = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                expected[i * n + j] = (0..k)
                    .map(|kk| a_val(i, kk) * lut(j / 8, index(j, segment * 64 + kk)))
                    .sum();
            }
        }
        assert_eq!(read_f32_tile(&machine, m, n), expected, "segment {segment}");
        // LUT-B needs SM107.
        let sm100 = TcMmaOptions {
            lut_b: Some(LUT_COL),
            ..TcMmaOptions::default()
        };
        assert!(machine.run_with(&p, &sm100).is_err());
        // The form flag without a table address fails closed.
        assert!(machine.run_with(&p, &TcMmaOptions { arch: TcArch::Sm107, ..TcMmaOptions::default() }).is_err());
    }
}

#[test]
fn collectors_track_state_and_keep_numerics() {
    use CollectorOp::*;
    let s = tc_collector_transition(0, Fill, Fill, 2).unwrap();
    assert_eq!(s, 0b01001);
    assert_eq!(tc_collector_transition(s, Use, Use, 2).unwrap(), s);
    assert_eq!(
        tc_collector_transition(s, LastUse, None, 0).unwrap(),
        0b01000
    );
    assert_eq!(tc_collector_transition(s, Discard, Discard, 2).unwrap(), 0);
    let error = tc_collector_transition(0, Use, None, 0).unwrap_err();
    assert!(error.message.contains("requires a valid previous fill"));
    assert!(tc_collector_transition(0, None, Fill, 4).is_err());
    // A collector MMA computes exactly what the plain one does.
    let (m, n, k) = (128, 16, 16);
    let mut machine = Machine::new();
    let a_desc = place_b16(&mut machine, 0x1000, m, k, a_val, true);
    let b_desc = place_b16(&mut machine, 0x8000, n, k, b_val, true);
    let idesc = dense_idesc(
        "float32", "bfloat16", "bfloat16", m as i64, n as i64, k as i64,
    );
    let mut p = payload(
        TcMmaKind::F16,
        TcA::Smem(op()),
        a_desc,
        b_desc,
        idesc,
        false,
    );
    p.args.collector_a = Fill;
    p.args.collector_b = LastUse;
    machine.run(&p).unwrap();
    assert_eq!(
        read_f32_tile(&machine, m, n),
        reference(m, n, k, |_, _| 0.0)
    );
}

#[test]
fn per_group_decoder_and_variant_options() {
    let pair = idesc_cta2("float32", "bfloat16", "bfloat16", 256, 64, 16);
    assert_eq!(
        decode_instr_desc_for(pair, TcMmaKind::F16, 2).unwrap().m,
        256
    );
    assert!(decode_instr_desc_for(pair, TcMmaKind::F16, 1).is_err());
    let ws = ((1 << 4) | ((64 >> 3) << 17) | ((32 >> 4) << 24) | (1 << 30)) as u32;
    let decoded = decode_instr_desc_for(ws, TcMmaKind::F16, 1).unwrap();
    assert_eq!((decoded.m, decoded.n, decoded.max_shift), (32, 64, 1));
    assert!(decode_instr_desc_for(ws, TcMmaKind::F16, 2).is_err());
    assert!(decode_instr_desc_for(ws, TcMmaKind::F16, 3).is_err());
    let ti16 = (2 << 4) | (3 << 7) | (3 << 10) | (2 << 17) | (8 << 24);
    let decoded = decode_instr_desc_for(ti16, TcMmaKind::I8, 1).unwrap();
    assert_eq!(
        (decoded.m, decoded.n, decoded.a, decoded.d),
        (128, 16, None, Some(Dtype::S32))
    );
    assert_eq!(
        TcMmaOptions::parse_variant("ti16.sm_107a").unwrap(),
        TcMmaOptions {
            ti16: true,
            arch: TcArch::Sm107,
            ..TcMmaOptions::default()
        }
    );
    assert_eq!(
        TcMmaOptions::parse_variant("bogus").unwrap_err().kind,
        OpErrorKind::Unsupported
    );
}
