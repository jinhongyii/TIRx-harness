//! Contract-boundary replay of the GPU-recorded cvt goldens: every golden
//! spelling is mapped to EVERY `tirx.ptx.cvt*` table form that spells it, and
//! every row is run through `resolve_ptx` with a hand-built `PtxIo`.

#![allow(clippy::needless_range_loop)]

use super::{spec, spelling, Layout, OpSpec};
use crate::dtype::{Dtype, Ty};
use crate::oplib::{resolve_ptx, OpErrorKind, PtxIo};
use crate::program::OpKey;
use numsim_oplib::cvt::goldens::{self, GoldenForm, Layout as GoldenLayout};
use numsim_oplib::cvt::{CvtSpelling, CvtType};
use numsim_types::{WarpMask, WarpValue, WARP_SIZE};
use std::collections::BTreeMap;

/// (table op, OpKey mods, per-slot tokens).
type TableForm = (&'static OpSpec, Vec<String>, Vec<Option<String>>);

/// Exact-width carrier of a PTX `cvt` type token.
fn carrier(token: &str) -> Ty {
    let parsed = CvtType::parse(token).unwrap_or_else(|| panic!("unknown cvt type {token}"));
    let signed = matches!(parsed, CvtType::Int(kind) if kind.signed());
    Ty::scalar(match (parsed.bits(), signed) {
        (8, false) => Dtype::U8,
        (8, true) => Dtype::S8,
        (16, false) => Dtype::U16,
        (16, true) => Dtype::S16,
        (32, false) => Dtype::U32,
        (32, true) => Dtype::S32,
        (64, false) => Dtype::U64,
        (64, true) => Dtype::S64,
        other => panic!("no carrier for {other:?}"),
    })
}

/// Every table form whose slots spell `text` with a source layout matching
/// the golden layout: (spec, OpKey mods, slot tokens).
fn table_forms(text: &str, layout: GoldenLayout) -> Vec<TableForm> {
    let parts: Vec<&str> = text.split('.').collect();
    assert_eq!(parts[0], "cvt");
    let n = parts.len();
    let (dtype, atype, middle) = (parts[n - 2], parts[n - 1], &parts[1..n - 2]);
    let mut found = Vec::new();
    'spec: for spec in super::table::SPECS {
        let mut tokens: Vec<Option<String>> = vec![None; spec.slots.len()];
        for (slot_name, value) in [("dtype", dtype), ("atype", atype)] {
            match spec.slots.iter().position(|s| s.name == slot_name) {
                Some(i) if spec.slots[i].choices.contains(&value) => {
                    tokens[i] = Some(value.to_string())
                }
                _ => continue 'spec,
            }
        }
        for &token in middle {
            let slot = spec.slots.iter().enumerate().position(|(i, s)| {
                tokens[i].is_none()
                    && s.name != "dtype"
                    && s.name != "atype"
                    && s.choices.contains(&token)
            });
            match slot {
                Some(i) => tokens[i] = Some(token.to_string()),
                None => continue 'spec,
            }
        }
        if spec
            .slots
            .iter()
            .zip(&tokens)
            .any(|(s, t)| !s.optional && t.is_none())
        {
            continue;
        }
        let scaled = spec
            .slots
            .iter()
            .zip(&tokens)
            .any(|(s, t)| s.name == "scaled" && t.is_some());
        let matches = match layout {
            GoldenLayout::Unary => spec.layout == Layout::Unary && !scaled,
            GoldenLayout::Pair => spec.layout == Layout::Pair && !scaled,
            GoldenLayout::Quad => spec.layout == Layout::Quad && !scaled,
            GoldenLayout::Scaled => spec.layout == Layout::Unary && scaled,
        };
        if !matches {
            continue;
        }
        let mods = spec
            .slots
            .iter()
            .zip(&tokens)
            .filter_map(|(s, t)| t.as_ref().map(|t| format!("{}={t}", s.name)))
            .collect();
        found.push((spec, mods, tokens));
    }
    found
}

/// Source carriers for one table form.
fn source_tys(spec: &OpSpec, scaled_n2: Option<bool>, atype: &str) -> Vec<Ty> {
    let primary = carrier(atype);
    let mut tys = match spec.layout {
        Layout::Unary => vec![primary],
        Layout::Pair => vec![primary; 2],
        Layout::PairRbits => vec![primary, primary, Ty::U32],
        Layout::Quad => vec![primary, primary, primary, primary, Ty::U32],
        Layout::Pack => vec![Ty::S32; 2],
        Layout::PackC => vec![Ty::S32, Ty::S32, Ty::U32],
    };
    if let Some(n2) = scaled_n2 {
        tys.push(if n2 { Ty::U16 } else { Ty::U8 });
    }
    tys
}

fn source_file(form: &GoldenForm) -> &'static str {
    if form.name.starts_with("cvt.") {
        "ptx_cvt_scalar_goldens"
    } else if !form.spelling.contains("scaled")
        && !form.spelling.contains("x4.")
        && (form.spelling.contains("e4m3x2") || form.spelling.contains("e5m2x2"))
    {
        "ptx_cvt_fp8_goldens"
    } else {
        "ptx_cvt_narrow_goldens"
    }
}

/// Replay every row of `form` through one resolved table form; returns the
/// mismatch descriptions.
fn replay(
    form: &GoldenForm,
    spec: &OpSpec,
    mods: &[String],
    tokens: &[Option<String>],
) -> Vec<String> {
    let parts: Vec<&str> = form.spelling.split('.').collect();
    let (dtype, atype) = (parts[parts.len() - 2], parts[parts.len() - 1]);
    let scaled = spec
        .slots
        .iter()
        .zip(tokens)
        .find(|(s, t)| s.name == "scaled" && t.is_some())
        .map(|(_, t)| t.as_deref() == Some("scaled::n2::ue8m0"));
    let dst_tys = [carrier(dtype)];
    let src_tys = source_tys(spec, scaled, atype);
    let key = OpKey {
        name: spec.name.into(),
        mods: mods.to_vec(),
    };
    let f = match resolve_ptx(&key, &dst_tys, &src_tys) {
        Ok(f) => f,
        Err(e) => return vec![format!("{} {:?}: resolve failed: {e}", spec.name, mods)],
    };
    let mut mismatches = Vec::new();
    let rows = form.expected.len();
    for base in (0..rows).step_by(WARP_SIZE) {
        let count = (rows - base).min(WARP_SIZE);
        let mut srcs: Vec<WarpValue<u64>> = vec![[0; WARP_SIZE]; src_tys.len()];
        for lane in 0..count {
            let o = goldens::operands(form, base + lane);
            let primaries = match spec.layout {
                Layout::Unary => vec![o.a],
                Layout::Pair => vec![o.a, o.b],
                Layout::Quad => vec![o.a, o.b, o.c, o.d, u64::from(o.rbits)],
                other => panic!("golden layout {other:?}"),
            };
            for (i, v) in primaries.into_iter().enumerate() {
                srcs[i][lane] = v;
            }
            if scaled.is_some() {
                srcs[src_tys.len() - 1][lane] = u64::from(o.scale);
            }
        }
        let mut dsts: Vec<WarpValue<u64>> = vec![[0xdead_beef; WARP_SIZE]; 1];
        let mut io = PtxIo {
            dsts: &mut dsts,
            dst_tys: &dst_tys,
            srcs: &srcs,
            src_tys: &src_tys,
            mask: WarpMask::first_n(count as u32),
        };
        if let Err(e) = f.call(&mut io) {
            mismatches.push(format!("{} rows {base}..: execute failed: {e}", spec.name));
            continue;
        }
        for lane in 0..count {
            let expected = form.expected[base + lane];
            if dsts[0][lane] != expected {
                mismatches.push(format!(
                    "{} {:?} row {}: expected {expected:#x} got {:#x}",
                    spec.name,
                    mods,
                    base + lane,
                    dsts[0][lane]
                ));
            }
        }
    }
    mismatches
}

#[test]
fn every_golden_row_replays_bit_exact_through_resolve_ptx() {
    let mut per_file: BTreeMap<&str, (usize, usize, usize)> = BTreeMap::new();
    let mut unmapped = Vec::new();
    let mut failures = Vec::new();
    for form in goldens::all_forms() {
        let entry = per_file.entry(source_file(form)).or_default();
        entry.1 += form.expected.len();
        let forms = table_forms(form.spelling, form.layout);
        if forms.is_empty() {
            unmapped.push(form.spelling);
            continue;
        }
        let mut ok = true;
        for (spec, mods, tokens) in &forms {
            // The rebuilt spelling must parse to the golden's own form.
            let rebuilt = spelling(spec, tokens).unwrap();
            assert_eq!(
                CvtSpelling::parse(&rebuilt).unwrap(),
                CvtSpelling::parse(form.spelling).unwrap(),
                "{rebuilt} vs {}",
                form.spelling
            );
            let bad = replay(form, spec, mods, tokens);
            ok &= bad.is_empty();
            failures.extend(bad.into_iter().take(4));
            entry.2 += form.expected.len();
        }
        if ok {
            entry.0 += form.expected.len();
        }
    }
    for (file, (ok, total, replays)) in &per_file {
        eprintln!(
            "{file}: {ok}/{total} golden rows bit-exact ({replays} row replays across table forms)"
        );
    }
    assert!(
        unmapped.is_empty(),
        "golden spellings with no table form: {unmapped:?}"
    );
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
    for (_, (ok, total, _)) in per_file {
        assert_eq!(ok, total);
    }
}

fn run1(
    name: &str,
    mods: &[&str],
    dst: Ty,
    srcs: &[(Ty, u64)],
) -> Result<u64, crate::oplib::OpError> {
    let key = OpKey {
        name: name.into(),
        mods: mods.iter().map(|m| m.to_string()).collect(),
    };
    let src_tys: Vec<Ty> = srcs.iter().map(|s| s.0).collect();
    let f = resolve_ptx(&key, &[dst], &src_tys)?;
    let values: Vec<WarpValue<u64>> = srcs.iter().map(|s| [s.1; WARP_SIZE]).collect();
    let mut dsts = vec![[0u64; WARP_SIZE]];
    let mut io = PtxIo {
        dsts: &mut dsts,
        dst_tys: &[dst],
        srcs: &values,
        src_tys: &src_tys,
        mask: WarpMask::lane(3),
    };
    f.call(&mut io)?;
    assert_eq!(dsts[0][0], 0, "inactive lane written");
    Ok(dsts[0][3])
}

#[test]
fn integer_results_sign_extend_into_wider_carriers() {
    let minus_three = (-3.7f32).to_bits() as u64;
    let got = run1(
        "tirx.ptx.cvt",
        &["rnd=rzi", "dtype=s8", "atype=f32"],
        Ty::S32,
        &[(Ty::F32, minus_three)],
    )
    .unwrap();
    assert_eq!(got, 0xffff_fffd);
    let got = run1(
        "tirx.ptx.cvt",
        &["rnd=rzi", "dtype=s8", "atype=f32"],
        Ty::scalar(Dtype::S8),
        &[(Ty::F32, minus_three)],
    )
    .unwrap();
    assert_eq!(got, 0xfd);
    // Bare tokens are matched to slots by choice.
    let got = run1(
        "tirx.ptx.cvt",
        &["rzi", "dtype=u8", "atype=f32"],
        Ty::U32,
        &[(Ty::F32, 300f32.to_bits() as u64)],
    );
    assert_eq!(got.unwrap(), 0xff);
}

#[test]
fn pack_forms_and_pair_order() {
    // cvt.pack.sat.s16.s32 d, a, b: a -> upper half.
    let got = run1(
        "tirx.ptx.cvt_pack",
        &["sat=sat", "convert=s16", "abtype=s32"],
        Ty::U32,
        &[(Ty::S32, (-40000i32) as u32 as u64), (Ty::S32, 7)],
    )
    .unwrap();
    assert_eq!(got, 0x8000_0007);
    // cvt.rn.f16x2.f32 d, a, b: a -> upper half.
    let got = run1(
        "tirx.ptx.cvt_f16x2_f32",
        &["rnd=rn", "dtype=f16x2", "atype=f32"],
        Ty::F16X2,
        &[
            (Ty::F32, 1.0f32.to_bits() as u64),
            (Ty::F32, 2.0f32.to_bits() as u64),
        ],
    )
    .unwrap();
    assert_eq!(got, 0x3c00_4000);
}

#[test]
fn illegal_or_unknown_forms_fail_closed() {
    let unsupported = |r: Result<u64, crate::oplib::OpError>| matches!(r, Err(e) if e.kind == OpErrorKind::Unsupported);
    // Integer-to-integer conversions take no rounding modifier.
    assert!(unsupported(run1(
        "tirx.ptx.cvt",
        &["rnd=rzi", "dtype=s32", "atype=u32"],
        Ty::S32,
        &[(Ty::U32, 1)]
    )));
    // Unknown slot.
    assert!(unsupported(run1(
        "tirx.ptx.cvt",
        &["bogus=x", "dtype=s32", "atype=u32"],
        Ty::S32,
        &[(Ty::U32, 1)]
    )));
    // Token not among the slot's choices.
    assert!(unsupported(run1(
        "tirx.ptx.cvt_tf32_f32",
        &["rnd=rm", "dtype=tf32", "atype=f32"],
        Ty::U32,
        &[(Ty::F32, 1)]
    )));
    // Missing required slot.
    assert!(unsupported(run1(
        "tirx.ptx.cvt_tf32_f32",
        &["dtype=tf32", "atype=f32"],
        Ty::U32,
        &[(Ty::F32, 1)]
    )));
    // Wrong arity (scaled form without its scale register).
    assert!(unsupported(run1(
        "tirx.ptx.cvt_bf16x2_f8x2",
        &[
            "rnd=rn",
            "scaled=scaled::n2::ue8m0",
            "dtype=bf16x2",
            "atype=e4m3x2"
        ],
        Ty::U32,
        &[(Ty::U16, 1)]
    )));
    // Narrow destination carrier.
    assert!(unsupported(run1(
        "tirx.ptx.cvt",
        &["rnd=rn", "dtype=f32", "atype=f64"],
        Ty::U16,
        &[(Ty::F64, 1)]
    )));
    assert!(spec("tirx.ptx.mov").is_none());
}

/// Every table op resolves for at least one modifier combination; prints the
/// per-op count of combinations `CvtSpelling` rejects (fail closed).
#[test]
fn every_table_op_has_resolvable_forms() {
    let mut empty = Vec::new();
    for spec in super::table::SPECS {
        let mut combos: Vec<Vec<Option<&str>>> = vec![Vec::new()];
        for slot in spec.slots {
            let mut next = Vec::new();
            for combo in &combos {
                if slot.optional {
                    let mut c = combo.clone();
                    c.push(None);
                    next.push(c);
                }
                for &choice in slot.choices {
                    let mut c = combo.clone();
                    c.push(Some(choice));
                    next.push(c);
                }
            }
            combos = next;
        }
        let (mut ok, mut rejected) = (0, 0);
        for combo in &combos {
            let tokens: Vec<Option<String>> = combo.iter().map(|t| t.map(str::to_string)).collect();
            let get = |name: &str| {
                spec.slots
                    .iter()
                    .position(|s| s.name == name)
                    .and_then(|i| combo[i])
            };
            let scaled = get("scaled").map(|s| s.contains("n2"));
            let src_ty = match spec.layout {
                Layout::Pack | Layout::PackC => Ty::S32,
                _ => carrier(get("atype").unwrap()),
            };
            let dst_ty = match spec.layout {
                Layout::Pack | Layout::PackC => Ty::U32,
                _ => carrier(get("dtype").unwrap()),
            };
            let mut src_tys = source_tys(spec, scaled, "u8");
            for ty in src_tys.iter_mut().take(match spec.layout {
                Layout::Unary => 1,
                Layout::Pair | Layout::PairRbits => 2,
                Layout::Quad => 4,
                _ => 0,
            }) {
                *ty = src_ty;
            }
            let mods: Vec<String> = spec
                .slots
                .iter()
                .zip(&tokens)
                .filter_map(|(s, t)| t.as_ref().map(|t| format!("{}={t}", s.name)))
                .collect();
            let key = OpKey {
                name: spec.name.into(),
                mods,
            };
            match resolve_ptx(&key, &[dst_ty], &src_tys) {
                Ok(_) => ok += 1,
                Err(e) => {
                    assert_eq!(
                        e.kind,
                        OpErrorKind::Unsupported,
                        "{}: {e}",
                        spelling(spec, &tokens).unwrap()
                    );
                    rejected += 1;
                }
            }
        }
        eprintln!(
            "{}: {ok} resolvable, {rejected} rejected of {} slot combinations",
            spec.name,
            combos.len()
        );
        if ok == 0 {
            empty.push(spec.name);
        }
    }
    assert!(
        empty.is_empty(),
        "table ops with no resolvable form: {empty:?}"
    );
}

/// Integer `cvt.{d}.{a}` into every carrier width 8..=128 (W11-4): the
/// source is read from its carrier's low `a` bits, converted (truncate to
/// `d`, no saturation), and the carrier is filled by extending the `d`-bit
/// result per `d`'s signedness, both 64-bit slots of a 128-bit carrier
/// included (legacy's register semantics).
#[test]
fn integer_cvt_extends_into_every_carrier_width() {
    let ints: [(&str, u32, bool); 8] = [
        ("u8", 8, false), ("s8", 8, true), ("u16", 16, false), ("s16", 16, true),
        ("u32", 32, false), ("s32", 32, true), ("u64", 64, false), ("s64", 64, true),
    ];
    let carriers = [Dtype::U8, Dtype::S8, Dtype::U16, Dtype::S16, Dtype::U32, Dtype::S32, Dtype::U64, Dtype::S64, Dtype::B128];
    let values: [u64; 6] = [0, 1, 0x7f, 0x80, 0xffff_ffff_ffff_ff81, 0x8000_0000_1234_5678];
    let ext = |v: u64, bits: u32, signed: bool| -> u128 {
        let m = if bits == 64 { v } else { v & ((1u64 << bits) - 1) };
        if signed && bits < 128 && (m >> (bits - 1)) & 1 == 1 {
            u128::from(m) | (u128::MAX << bits)
        } else {
            u128::from(m)
        }
    };
    let mut checked = 0;
    for (d, d_bits, d_signed) in ints {
        for (a, a_bits, a_signed) in ints {
            let key = OpKey { name: "tirx.ptx.cvt".into(), mods: vec![format!("dtype={d}"), format!("atype={a}")] };
            for dst_carrier in carriers.iter().filter(|c| Ty::scalar(**c).bits() >= d_bits) {
                for src_carrier in carriers.iter().filter(|c| Ty::scalar(**c).bits() >= a_bits) {
                    let (dt, st) = (Ty::scalar(*dst_carrier), Ty::scalar(*src_carrier));
                    let Ok(f) = resolve_ptx(&key, &[dt], &[st]) else {
                        panic!("cvt.{d}.{a} {dt:?} <- {st:?} did not resolve");
                    };
                    for &v in &values {
                        let mut srcs = vec![[v; WARP_SIZE]];
                        if st.slots() > 1 {
                            srcs.push([0xabcd_ef01_2345_6789; WARP_SIZE]);
                        }
                        let mut out = vec![[0x5555_5555_5555_5555u64; WARP_SIZE]; dt.slots() as usize];
                        let mut io = PtxIo { dsts: &mut out, dst_tys: &[dt], srcs: &srcs, src_tys: &[st], mask: WarpMask::lane(0) };
                        f.call(&mut io).unwrap();
                        // Reference: source value (a bits), then d bits, then the carrier.
                        let source = ext(v, a_bits, a_signed) as u64;
                        let mut want = ext(source, d_bits, d_signed);
                        let carrier_bits = dt.bits();
                        if carrier_bits < 128 {
                            want &= (1u128 << carrier_bits) - 1;
                        }
                        let got = u128::from(out[0][0]) | if dt.slots() > 1 { u128::from(out[1][0]) << 64 } else { 0 };
                        assert_eq!(got, want, "cvt.{d}.{a} {dt:?} <- {st:?} value {v:#x}");
                        checked += 1;
                    }
                }
            }
        }
    }
    assert!(checked > 1000, "{checked}");
}
