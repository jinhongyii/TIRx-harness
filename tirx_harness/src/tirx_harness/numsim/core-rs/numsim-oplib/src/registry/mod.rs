//! Static op table: every registered NumSim operation with its family,
//! fidelity and notes, plus the OpLib functions that implement its numerics.
//!
//! `ops_table.rs` is the hand-maintained source of truth (seeded at step 5 from
//! the deleted legacy `engine-rs/SUPPORTED_OPS.md`). [`render_supported_ops`]
//! renders it in the legacy matrix layout; the published
//! `numsim-oplib/SUPPORTED_OPS.md` is generated from it by
//! `numsim-core`'s `supported_ops` example. Each family module contributes
//! `BINDINGS` that tie op names to OpLib entry points.

mod ops_table;

pub use ops_table::{LEGACY_ABI_VERSION, OPS};

/// One legacy op (IR name or PTX spelling) implemented by an OpLib function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Binding {
    /// Legacy op identifier, e.g. `tirx.ptx.cvt` or `cvt.rn.f16.f32`.
    pub op: &'static str,
    /// Rust path of the OpLib entry point, e.g. `cvt::ptx_cvt_f32_to_f16`.
    pub function: &'static str,
}

/// Which table of `SUPPORTED_OPS.md` the op belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Section {
    /// `tirx.cuda.*` / `tirx.ptx.*` intrinsics.
    CudaPtx,
    /// `tirx.tile.*` primitives.
    TilePrimitive,
}

/// How faithfully NumSim models an op (legacy `support` column).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Fidelity {
    Modeled,
    DeterministicRepresentative,
    OrderingOnly,
    ExactProtocol,
    Rejected,
}

impl Fidelity {
    /// The SUPPORTED_OPS.md spelling of this fidelity.
    pub const fn as_str(self) -> &'static str {
        match self {
            Fidelity::Modeled => "modeled",
            Fidelity::DeterministicRepresentative => "deterministic_representative",
            Fidelity::OrderingOnly => "ordering_only",
            Fidelity::ExactProtocol => "exact_protocol",
            Fidelity::Rejected => "rejected",
        }
    }
}

/// One row of the op table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpInfo {
    pub name: &'static str,
    pub section: Section,
    /// Legacy family (`pure_scalar`, `raw_tma`, ...); `tile_primitive` for tile ops.
    pub family: &'static str,
    pub fidelity: Fidelity,
    pub notes: &'static str,
}

/// Every family module's bindings, in module order.
pub fn bindings() -> impl Iterator<Item = &'static Binding> {
    [
        crate::scalar::BINDINGS,
        crate::codec::BINDINGS,
        crate::arith::BINDINGS,
        crate::atomic::BINDINGS,
        crate::cvt::BINDINGS,
        crate::mma::BINDINGS,
        crate::tcgen05::BINDINGS,
        crate::tma::BINDINGS,
        crate::layout::BINDINGS,
        crate::warp::BINDINGS,
    ]
    .into_iter()
    .flatten()
}

/// Look up one op by its legacy name.
pub fn op(name: &str) -> Option<&'static OpInfo> {
    OPS.iter().find(|info| info.name == name)
}

/// OpLib functions bound to `name`.
pub fn functions_for(name: &str) -> Vec<&'static str> {
    bindings()
        .filter(|binding| binding.op == name)
        .map(|binding| binding.function)
        .collect()
}

fn cell(value: &str) -> String {
    value.replace('|', "\\|").replace('\n', " ")
}

const TILE_PREAMBLE: &str =
    "Every warp-, warpgroup-, or CTA-scoped tile call validates complete dynamic \
participation in its declared execution scope, including register-only \
lowerings with no completion barrier.";

/// Render `SUPPORTED_OPS.md` exactly as `support_matrix.py` does.
pub fn render_supported_ops(abi_version: u32) -> String {
    render(abi_version, false)
}

/// Same table with an extra `OpLib` column listing bound functions.
pub fn render_supported_ops_with_bindings(abi_version: u32) -> String {
    render(abi_version, true)
}

fn render(abi_version: u32, with_bindings: bool) -> String {
    let mut lines: Vec<String> = vec![
        "# NumSim Engine Operation Support".into(),
        String::new(),
        "This file is generated from NumSim's operation registry and is checked by tests.".into(),
        format!("It describes NumSim ABI **v{abi_version}**."),
        "A listed operation may still reject modifier, dtype, shape, or layout values \
outside its exact specialization domain. Handwritten runtime cases cover \
registered operations; GPU parity is tested where the required hardware is available."
            .into(),
        String::new(),
        "## TIRx CUDA/PTX Ops".into(),
        String::new(),
    ];
    let extra = |info: &OpInfo| -> String {
        if with_bindings {
            let functions = functions_for(info.name)
                .iter()
                .map(|function| format!("`{function}`"))
                .collect::<Vec<_>>()
                .join(", ");
            format!(" {} |", cell(&functions))
        } else {
            String::new()
        }
    };
    if with_bindings {
        lines.push("| Operation | Family | Fidelity | Notes | OpLib |".into());
        lines.push("| --- | --- | --- | --- | --- |".into());
    } else {
        lines.push("| Operation | Family | Fidelity | Notes |".into());
        lines.push("| --- | --- | --- | --- |".into());
    }
    for info in OPS.iter().filter(|info| info.section == Section::CudaPtx) {
        lines.push(format!(
            "| {} | {} | {} | {} |{}",
            cell(&format!("`{}`", info.name)),
            cell(info.family),
            info.fidelity.as_str(),
            cell(info.notes),
            extra(info),
        ));
    }
    lines.extend([
        String::new(),
        "## CUDA Tile Primitives".into(),
        String::new(),
        TILE_PREAMBLE.into(),
        String::new(),
    ]);
    if with_bindings {
        lines.push("| Operation | Fidelity | Notes | OpLib |".into());
        lines.push("| --- | --- | --- | --- |".into());
    } else {
        lines.push("| Operation | Fidelity | Notes |".into());
        lines.push("| --- | --- | --- |".into());
    }
    for info in OPS
        .iter()
        .filter(|info| info.section == Section::TilePrimitive)
    {
        lines.push(format!(
            "| {} | {} | {} |{}",
            cell(&format!("`{}`", info.name)),
            info.fidelity.as_str(),
            cell(info.notes),
            extra(info),
        ));
    }
    lines.join("\n") + "\n"
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn op_names_are_unique_and_sections_are_consistent() {
        let mut seen = HashSet::new();
        for info in OPS {
            assert!(seen.insert(info.name), "duplicate op {}", info.name);
            match info.section {
                Section::CudaPtx => assert!(
                    info.name.starts_with("tirx.cuda.") || info.name.starts_with("tirx.ptx.")
                ),
                Section::TilePrimitive => assert!(info.name.starts_with("tirx.tile.")),
            }
        }
        assert!(op("tirx.ptx.cvt").is_some());
    }

    #[test]
    fn bindings_name_registered_ops_or_ptx_spellings() {
        for binding in bindings() {
            assert!(!binding.function.is_empty());
            if binding.op.starts_with("tirx.") {
                assert!(
                    op(binding.op).is_some(),
                    "binding to unknown op {}",
                    binding.op
                );
            }
        }
        let rendered = render_supported_ops_with_bindings(LEGACY_ABI_VERSION);
        assert!(rendered.contains("| OpLib |"));
    }
}
