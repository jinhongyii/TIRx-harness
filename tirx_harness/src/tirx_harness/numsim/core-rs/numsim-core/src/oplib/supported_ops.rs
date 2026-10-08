//! The v2 `numsim-oplib/SUPPORTED_OPS.md` generator (step 5: the legacy file
//! moves to numsim-oplib and becomes generated).
//!
//! Inputs, all checked in:
//! * the legacy op table (`numsim_oplib::registry::OPS`, from the legacy
//!   `engine-rs/SUPPORTED_OPS.md`), plus the v2-only ops in [`V2_OPS`];
//! * how v2 lowering handles each op name (`lowering_ops.tsv`, written by
//!   `scripts/numsim-v2/lowering_ops.py`);
//! * which `Instr::Ptx` names `resolve_ptx` implements ([`super::ptx_op_names`]);
//! * the behaviour deltas vs legacy (`docs/development/numsim-behaviour-deltas.md`
//!   row ids, [`DELTAS`]).
//!
//! `cargo run -p numsim-core --example supported_ops` rewrites the file. The
//! generator lives in numsim-core rather than numsim-oplib because the
//! `resolve_ptx` tables do (numsim-core depends on numsim-oplib). The
//! `supported_ops_md_is_current` test fails when the committed file is stale.

use super::{registry, Fidelity, OpEntry};
use std::collections::{BTreeMap, BTreeSet};

/// Ops v2 implements that the legacy table did not list.
pub const V2_OPS: &[OpEntry] = &[
    OpEntry {
        name: "tirx.log1p",
        family: "pure_scalar",
        fidelity: Fidelity::Modeled,
        notes: "f32 host `log1pf` with pinned NaN/special values; f16/bf16 via f32 with RNE back; f64 binary64 (delta D9)",
        instr: "ptx",
    },
    OpEntry {
        name: "tirx.sigmoid",
        family: "pure_scalar",
        fidelity: Fidelity::Modeled,
        notes: "`1/(1+exp(-x))` in binary32 with a pinned NaN; f16/bf16 via f32 with RNE back; f64 binary64 (delta D9)",
        instr: "ptx",
    },
];

/// Behaviour-delta rows (by op name, or `prefix*`) where v2 differs from legacy.
pub const DELTAS: &[(&str, &str)] = &[
    ("tirx.round", "D2"),
    ("tirx.sin", "D3"),
    ("tirx.cos", "D3"),
    ("tirx.tanh", "D3"),
    ("tirx.exp2", "D3"),
    ("tirx.sqrt", "D3"),
    ("tirx.floor", "D3"),
    ("tirx.ceil", "D3"),
    ("tirx.trunc", "D3"),
    ("tirx.ptx.fma", "D8"),
    ("tirx.ptx.add", "D8"),
    ("tirx.ptx.sub", "D8"),
    ("tirx.ptx.mul", "D8"),
    ("tirx.log1p", "D9"),
    ("tirx.sigmoid", "D9"),
    ("tirx.erf", "D10"),
    ("tirx.exp10", "D10"),
    ("tirx.log10", "D10"),
    ("tirx.nearbyint", "D10"),
    ("tirx.cuda.bfloat162float", "P1"),
    ("tirx.cuda.hmin2", "P2"),
    ("tirx.cuda.hmax2", "P2"),
    ("tirx.cuda.sm100_2sm_leader_smem_addr", "P3"),
    ("tirx.cuda.float22half2", "P4"),
    ("tirx.cuda.float8tohalf8", "P4"),
    ("tirx.cuda.half8tofloat8", "P4"),
    ("tirx.cuda.__shfl_sync", "P5"),
    ("tirx.ptx.shfl*", "P5"),
    ("tirx.ptx.prefetch*", "P6"),
    ("tirx.ptx.applypriority*", "P7"),
    ("tirx.tile.sum", "R1"),
    ("tirx.tile.max", "R1, R2"),
    ("tirx.tile.min", "R1, R2"),
    ("tirx.tile.gemm", "T1"),
    ("tirx.ptx.mma*", "T2"),
    ("tirx.ptx.tcgen05_mma*", "T2, T3"),
    ("tirx.ptx.tcgen05_ld*", "T4"),
    ("tirx.ptx.tcgen05_st*", "T4"),
    ("tirx.ptx_legacy.*", "L2"),
];

fn deltas_of(name: &str) -> String {
    let ids: Vec<&str> = DELTAS
        .iter()
        .filter(|(pattern, _)| match pattern.strip_suffix('*') {
            Some(prefix) => name.starts_with(prefix),
            None => name == *pattern,
        })
        .map(|(_, ids)| *ids)
        .collect();
    ids.join(", ")
}

/// How v2 handles one op.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum V2Path {
    /// `Instr::Ptx`, implemented by `resolve_ptx`.
    Oplib,
    /// `Instr::Unary` (`oplib::unary`).
    Unary,
    /// A dedicated engine instruction (the lowering function).
    Engine(&'static str),
    /// `tirx.tile.*`: lowered through TVM's dispatch (contract decision 6).
    TvmDispatch,
    /// Lowered to `Instr::Ptx`, but `resolve_ptx` has no entry: fails closed.
    NoOplib,
    /// The lowering rejects it: fails closed.
    Rejected,
    /// The lowering has no rule for it: fails closed.
    NotLowered,
}

impl V2Path {
    fn describe(self) -> String {
        match self {
            V2Path::Oplib => "`Instr::Ptx` (oplib `resolve_ptx`)".into(),
            V2Path::Unary => "`Instr::Unary` (oplib `unary`)".into(),
            V2Path::Engine(f) => format!("engine instruction (`{f}`)"),
            V2Path::TvmDispatch => "TVM tile dispatch (decision 6)".into(),
            V2Path::NoOplib => "**fails closed**: `Instr::Ptx` without an oplib entry".into(),
            V2Path::Rejected => "fails closed: rejected by lowering".into(),
            V2Path::NotLowered => "**fails closed**: not lowered".into(),
        }
    }
}

fn lowering() -> &'static BTreeMap<&'static str, &'static str> {
    static MAP: std::sync::OnceLock<BTreeMap<&'static str, &'static str>> = std::sync::OnceLock::new();
    MAP.get_or_init(|| {
        include_str!("lowering_ops.tsv")
            .lines()
            .filter(|l| !l.starts_with('#') && !l.is_empty())
            .filter_map(|l| l.split_once('\t'))
            .collect()
    })
}

/// v2's handling of `name` (see [`V2Path`]).
pub fn v2_path(name: &str, oplib_names: &BTreeSet<&'static str>) -> V2Path {
    if name.starts_with("tirx.tile.") {
        return V2Path::TvmDispatch;
    }
    match lowering().get(name).copied() {
        None => V2Path::NotLowered,
        Some("rejected") => V2Path::Rejected,
        Some("unary") => V2Path::Unary,
        Some("ptx_op" | "pure_op") if oplib_names.contains(name) => V2Path::Oplib,
        Some("ptx_op" | "pure_op") => V2Path::NoOplib,
        Some(other) => V2Path::Engine(other.strip_prefix("instr:").unwrap_or(other)),
    }
}

fn cell(value: &str) -> String {
    value.replace('|', "\\|").replace('\n', " ")
}

/// Rows of the generated file: legacy entries plus [`V2_OPS`], sorted.
pub fn rows() -> Vec<(OpEntry, V2Path, String)> {
    let oplib: BTreeSet<&'static str> = super::ptx_op_names().into_iter().collect();
    let mut entries: Vec<OpEntry> = registry::entries().to_vec();
    entries.extend(V2_OPS.iter().copied().filter(|e| !registry::entries().iter().any(|l| l.name == e.name)));
    entries.sort_by_key(|e| e.name);
    entries.into_iter().map(|e| (e, v2_path(e.name, &oplib), deltas_of(e.name))).collect()
}

/// A legacy op v2 does not run: legacy modeled it (not `rejected`), and v2
/// fails closed.
pub fn is_gap(entry: &OpEntry, path: V2Path) -> bool {
    entry.fidelity != Fidelity::Rejected && matches!(path, V2Path::NoOplib | V2Path::NotLowered | V2Path::Rejected)
}

/// The generated `numsim-oplib/SUPPORTED_OPS.md`.
pub fn render() -> String {
    let rows = rows();
    let mut s = String::from("# NumSim Operation Support (v2)\n\n");
    s.push_str(
        "Generated by `cargo run -p numsim-core --example supported_ops` from the legacy op table, the v2 \
lowering table (`numsim-core/src/oplib/lowering_ops.tsv`, written by `scripts/numsim-v2/lowering_ops.py`), \
the oplib `resolve_ptx` tables and the behaviour-delta rows of `docs/development/numsim-behaviour-deltas.md`. \
The `supported_ops_md_is_current` test fails when this file is stale.\n\n",
    );
    s.push_str(
        "A listed operation may still reject modifier, dtype, shape, or layout values outside its exact \
specialization domain; such forms fail closed (`Unsupported`, run verdict `incomplete`).\n\n",
    );
    let gaps: Vec<_> = rows.iter().filter(|(e, p, _)| is_gap(e, *p)).collect();
    s.push_str(&format!(
        "{} operations: {} legacy-documented operations v2 does not run (see [Gaps](#gaps-vs-legacy)).\n\n",
        rows.len(),
        gaps.len()
    ));
    for (title, tile) in [("TIRx CUDA/PTX Ops", false), ("CUDA Tile Primitives", true)] {
        s.push_str(&format!(
            "## {title}\n\n| Operation | Family | Fidelity | v2 | Deltas | Notes |\n| --- | --- | --- | --- | --- | --- |\n"
        ));
        for (e, path, deltas) in rows.iter().filter(|(e, _, _)| e.name.starts_with("tirx.tile.") == tile) {
            s.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} |\n",
                cell(&format!("`{}`", e.name)),
                cell(e.family),
                e.fidelity.name(),
                cell(&path.describe()),
                cell(deltas),
                cell(e.notes)
            ));
        }
        s.push('\n');
    }
    // Op names v2 lowers that neither table lists (TIR math, pure helpers).
    let listed: BTreeSet<&str> = rows.iter().map(|(e, _, _)| e.name).collect();
    let oplib: BTreeSet<&'static str> = super::ptx_op_names().into_iter().collect();
    let extra: Vec<&str> = lowering().keys().copied().filter(|n| !listed.contains(n)).collect();
    s.push_str(&format!(
        "## Other operations v2 lowers\n\nNot in the legacy table: {} op names the v2 lowering accepts (TIR math, \
structural calls, CUDA helpers), with how it runs them.\n\n| Operation | v2 | Deltas |\n| --- | --- | --- |\n",
        extra.len()
    ));
    for name in extra {
        s.push_str(&format!(
            "| `{}` | {} | {} |\n",
            name,
            cell(&v2_path(name, &oplib).describe()),
            cell(&deltas_of(name))
        ));
    }
    s.push('\n');
    s.push_str("## Gaps vs legacy\n\nLegacy documented these as supported; v2 fails closed on them.\n\n");
    s.push_str("| Operation | Legacy fidelity | v2 |\n| --- | --- | --- |\n");
    for (e, path, _) in gaps {
        s.push_str(&format!("| `{}` | {} | {} |\n", e.name, e.fidelity.name(), cell(&path.describe())));
    }
    s
}

#[cfg(test)]
mod tests {
    /// The committed `numsim-oplib/SUPPORTED_OPS.md` equals the generator's
    /// output (regenerate with `cargo run -p numsim-core --example supported_ops`).
    #[test]
    fn supported_ops_md_is_current() {
        let committed = include_str!("../../../numsim-oplib/SUPPORTED_OPS.md");
        assert!(
            committed == super::render(),
            "numsim-oplib/SUPPORTED_OPS.md is stale: run `cargo run -p numsim-core --example supported_ops`"
        );
    }

    #[test]
    fn v2_only_ops_are_documented_and_run() {
        let rows = super::rows();
        for name in ["tirx.log1p", "tirx.sigmoid"] {
            let (_, path, deltas) = rows.iter().find(|(e, _, _)| e.name == name).expect(name);
            assert_eq!(*path, super::V2Path::Oplib, "{name}");
            assert_eq!(deltas, "D9");
        }
    }
}
