//! Contract-boundary tests for `oplib` (lead-owned; per-family bit-exact
//! cases live next to each family).

use super::*;

#[test]
fn scalar_roundtrip() {
    assert_eq!(i32::from_bits((-5i32).to_bits()), -5);
    assert_eq!((-1i8).to_bits(), 0xff);
    assert_eq!(<f32 as Scalar>::from_bits(Scalar::to_bits(1.5f32)), 1.5);
    assert_eq!(E2M1::from_bits(0xff), E2M1(0xf));
    let key = OpKey { name: "tirx.cuda.float_as_uint".into(), mods: vec![] };
    assert!(resolve_ptx(&key, &[Ty::U32], &[Ty::F32]).is_ok());
}

#[test]
fn render_md_reproduces_legacy_supported_ops() {
    let md = render_supported_ops_md(registry());
    assert!(md.contains("| `tirx.ptx.ld` | raw_memory | modeled |"));
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../engine-rs/SUPPORTED_OPS.md");
    if let Ok(legacy) = std::fs::read_to_string(path) {
        assert_eq!(md, legacy);
    }
    assert_eq!(registry().len(), 600);
    let tile = registry().iter().find(|e| e.name == "tirx.tile.copy_async").unwrap();
    assert_eq!(tile.instr, "tile");
    let rejected = registry().iter().find(|e| e.fidelity == Fidelity::Rejected).unwrap();
    assert_eq!(rejected.instr, "");
}

#[test]
fn render_md_small_entries() {
    let e = [
        OpEntry { name: "tirx.ptx.ld", family: "raw_memory", fidelity: Fidelity::Modeled, notes: "", instr: "ld" },
        OpEntry { name: "tirx.tile.add", family: "modeled", fidelity: Fidelity::Modeled, notes: "", instr: "tile" },
    ];
    let md = render_supported_ops_md(&e);
    assert!(md.contains("| `tirx.ptx.ld` | raw_memory | modeled |  |"));
    assert!(md.contains("| `tirx.tile.add` | modeled |  |"));
}

#[test]
fn shfl_and_redux_follow_legacy_lane_rules() {
    let src: WarpValue<u64> = std::array::from_fn(|l| 100 + l as u64);
    let b = [1u64; 32];
    let c = [0x1fu64; 32];
    let (down, valid) = shfl(ShflMode::Down, &src, &b, &c, WarpMask::ALL);
    assert_eq!(down[0], 101);
    assert_eq!(down[31], 131); // out of range: own value, predicate false
    assert!(!valid.contains(31) && valid.contains(30));
    let (bfly, _) = shfl(ShflMode::Bfly, &src, &[1u64; 32], &c, WarpMask::ALL);
    let full = [u64::from(u32::MAX); 32];
    let (checked, ok) = shfl_sync(ShflMode::Down, &src, &b, &c, &full, WarpMask::ALL).unwrap();
    assert_eq!((checked, ok), (down, valid));
    // Lane 1 reads lane 2, which is not active: legacy error.
    let err = shfl_sync(ShflMode::Down, &src, &b, &c, &full, WarpMask(0b11)).unwrap_err();
    assert_eq!(err.kind, OpErrorKind::Invalid);
    assert_eq!((bfly[0], bfly[1]), (101, 100));

    let values: WarpValue<u64> = std::array::from_fn(|l| l as u64);
    assert_eq!(redux(ReduxOp::Add, Ty::U32, &values, WarpMask::ALL).unwrap(), 496);
    assert_eq!(redux(ReduxOp::Max, Ty::U32, &values, WarpMask(0x0000_00f0)).unwrap(), 7);
    let neg: WarpValue<u64> = std::array::from_fn(|l| (-(l as i32)) as u32 as u64);
    assert_eq!(redux(ReduxOp::Min, Ty::S32, &neg, WarpMask::ALL).unwrap(), (-31i32) as u32 as u64);
    let f: WarpValue<u64> = std::array::from_fn(|l| (l as f32 * 0.5).to_bits() as u64);
    assert_eq!(redux(ReduxOp::Max, Ty::F32, &f, WarpMask::ALL).unwrap(), 15.5f32.to_bits() as u64);
    assert!(redux(ReduxOp::Add, Ty::U32, &values, WarpMask::NONE).is_err());
}

/// `resolve_ptx` coverage over W1's inventory of all 566 distinct call ops
/// (`inventory_ops.tsv`). Prints the summary; the floor guards regressions.
#[test]
fn resolve_ptx_inventory_coverage() {
    let known = ptx::known_ops();
    let mut ops = 0usize;
    let mut covered = 0usize;
    let mut occurrences = 0u64;
    let mut covered_occurrences = 0u64;
    let mut ptx_table_ops = 0usize;
    let mut missing_ptx_table = Vec::new();
    let mut other: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for line in include_str!("inventory_ops.tsv").lines().filter(|l| !l.starts_with('#')) {
        let mut fields = line.split('\t');
        let name = fields.next().unwrap();
        let count: u64 = fields.next().unwrap().parse().unwrap();
        ops += 1;
        occurrences += count;
        let hit = known.contains(&name);
        if hit {
            covered += 1;
            covered_occurrences += count;
        }
        if name.starts_with("tirx.ptx.") {
            ptx_table_ops += 1;
        }
        if !hit {
            let family = if name.starts_with("tirx.ptx.") || name.starts_with("tirx.cuda.") || name.starts_with("tirx.tile.") {
                registry::instr_family(name)
            } else {
                "tir/structural"
            };
            *other.entry(family).or_default() += 1;
        }
        if !hit && name.starts_with("tirx.ptx.") && registry::instr_family(name) == "ptx" {
            missing_ptx_table.push(name);
        }
    }
    eprintln!(
        "resolve_ptx covers {covered}/{ops} inventory ops ({covered_occurrences}/{occurrences} occurrences); \
         {ptx_table_ops} inventory ops are tirx.ptx.*; Ptx-family ops not resolved: {missing_ptx_table:?}; \
         the rest by Instr family: {other:?}"
    );
    assert_eq!(ops, 566);
    assert!(missing_ptx_table.is_empty(), "Ptx-family ops without a resolver: {missing_ptx_table:?}");
}

#[test]
fn reserved_operand_bits_are_operand_errors_not_unsupported() {
    use super::{OpError, OpErrorKind};
    let lift = |m: &str| OpError::from(numsim_oplib::types::OpError::message(m)).kind;
    // Legacy text says "unsupported", but legacy reported it as an error.
    assert_eq!(
        lift("raw tcgen05.cp descriptor uses unsupported reserved/base/LBO-mode bits"),
        OpErrorKind::Invalid
    );
    assert_eq!(lift("s1z4m11 operand has nonzero reserved bits"), OpErrorKind::Invalid);
    assert_eq!(lift("ti16_transpose_unmodeled requires an unmodeled analysis contract"), OpErrorKind::Unsupported);
}
