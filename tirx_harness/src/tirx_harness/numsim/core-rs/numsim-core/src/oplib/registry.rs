//! The op registry, built once from `numsim_oplib::registry::OPS` (generated
//! from the legacy `SUPPORTED_OPS.md`), and the byte-exact renderer.

use super::{Fidelity, OpEntry};
use numsim_oplib::registry as lib;
use std::sync::OnceLock;

/// NumSim ABI version the rendered header names (legacy v38).
pub(super) const ABI_VERSION: u32 = lib::LEGACY_ABI_VERSION;

pub(super) fn entries() -> &'static [OpEntry] {
    static ENTRIES: OnceLock<Vec<OpEntry>> = OnceLock::new();
    ENTRIES.get_or_init(|| {
        lib::OPS
            .iter()
            .map(|info| OpEntry {
                name: info.name,
                family: info.family,
                fidelity: info.fidelity,
                notes: info.notes,
                instr: if info.fidelity == Fidelity::Rejected { "" } else { instr_family(info.name) },
            })
            .collect()
    })
}

/// `Instr::family()` an op lowers to (mirrors W1 `builtins.py`).
pub(in crate::oplib) fn instr_family(name: &str) -> &'static str {
    if name == "tirx.ptx.addr" {
        return "addr_of";
    }
    if let Some(table) = name.strip_prefix("tirx.ptx.") {
        const PREFIXES: &[(&str, &str)] = &[
            ("mbarrier_init", "mbar_init"),
            ("mbarrier_inval", "mbar_inval"),
            ("mbarrier_arrive", "mbar_arrive"),
            ("mbarrier_expect_tx", "mbar_tx"),
            ("mbarrier_complete_tx", "mbar_tx"),
            ("mbarrier_try_wait", "mbar_test_wait"),
            ("mbarrier_test_wait", "mbar_test_wait"),
            ("mbarrier", "mbar_query"),
            ("fence", "fence"),
            ("membar", "fence"),
            ("bar_warp_sync", "warp_sync"),
            ("barrier_cluster_arrive", "cluster_arrive"),
            ("barrier_cluster_wait", "cluster_wait"),
            ("bar_", "barrier"),
            ("barrier", "barrier"),
            ("cp_async_bulk_tensor", "tma"),
            ("cp_reduce_async_bulk_tensor", "tma"),
            ("cp_async_bulk_commit_group", "async_commit"),
            ("cp_async_bulk_wait_group", "async_wait"),
            ("cp_async_commit_group", "async_commit"),
            ("cp_async_wait", "async_wait"),
            ("cp_async_mbarrier_arrive", "cp_async_mbar_arrive"),
            ("cp_async_bulk", "bulk_copy"),
            ("cp_reduce_async_bulk", "bulk_copy"),
            ("cp_async", "cp_async"),
            ("st_async", "st_async"),
            ("red_async", "st_async"),
            ("st_bulk", "st_bulk"),
            ("tcgen05_alloc", "tcgen_alloc"),
            ("tcgen05_dealloc", "tcgen_dealloc"),
            ("tcgen05_relinquish", "tcgen_relinquish"),
            ("tcgen05_commit", "tcgen_commit"),
            ("tcgen05_ld", "tcgen_ld"),
            ("tcgen05_st", "tcgen_st"),
            ("tcgen05_wait", "tcgen_wait"),
            ("tcgen05_cp", "tcgen_cp"),
            ("tcgen05_mma", "tcgen_mma"),
            ("tcgen05_shift", "tcgen_cp"),
            ("tcgen05_fence", "fence"),
            ("shfl", "shfl"),
            ("vote", "vote"),
            ("redux", "redux"),
            ("match", "ptx"),
            ("elect", "elect"),
            ("activemask", "read_special"),
            ("atom", "atom"),
            ("red", "atom"),
            ("mma", "ptx"),
            ("ldmatrix", "ldmatrix"),
            ("stmatrix", "stmatrix"),
            ("movmatrix", "ptx"),
            ("tensormap_cp_fenceproxy", "tensormap_cp_fence"),
            ("tensormap", "tensormap_replace"),
            ("cvta", "cvta"),
            ("mapa", "mapa"),
            ("getctarank", "getctarank"),
            ("isspacep", "isspacep"),
            ("setmaxnreg", "setmaxnreg"),
            ("griddepcontrol", "griddepcontrol"),
            ("clusterlaunchcontrol", "clc_try_cancel"),
            ("nanosleep", "nop"),
            ("trap", "assert"),
            ("pmevent", "nop"),
            ("ld", "load_addr"),
            ("st", "store_addr"),
            ("prefetch", "nop"),
            ("applypriority", "nop"),
            ("discard", "discard"),
        ];
        return PREFIXES
            .iter()
            .find(|(prefix, _)| table.starts_with(prefix))
            .map_or("ptx", |(_, family)| family);
    }
    if name.starts_with("tirx.tile.") {
        return "tile";
    }
    match name {
        "tirx.cuda.mbarrier_wait" | "tirx.cuda.mbarrier_wait_acquire_cluster" => "mbar_wait",
        "tirx.cuda.cta_sync" | "tirx.cuda.warpgroup_sync" | "tirx.cuda.cluster_sync" | "tirx.cuda.cta_reduce" => "barrier",
        "tirx.cuda.warp_sync" => "warp_sync",
        "tirx.cuda.grid_sync" => "grid_sync",
        "tirx.cuda.elect_sync" => "elect",
        "tirx.cuda.__shfl_sync" | "tirx.cuda.__shfl_up_sync" | "tirx.cuda.__shfl_down_sync" | "tirx.cuda.__shfl_xor_sync" => "shfl",
        "tirx.cuda.ballot_sync" | "tirx.cuda.any_sync" => "vote",
        "tirx.cuda.__activemask" | "tirx.cuda.thread_rank" | "tirx.cuda.mov_sreg" | "tirx.cuda.clock64" => "read_special",
        "tirx.cuda.reduce_add_sync_u32" | "tirx.cuda.reduce_min_sync_u32" | "tirx.cuda.warp_reduce" => "redux",
        "tirx.cuda.ldg" => "load_addr",
        "tirx.cuda.atomic_add" | "tirx.cuda.atomic_cas" => "atom",
        "tirx.cuda.cvta_generic_to_shared" | "tirx.cuda.smem_addr_from_uint64" | "tirx.cuda.sm100_2sm_leader_smem_addr" => "cvta",
        "tirx.cuda.nano_sleep" | "tirx.cuda.printf" => "nop",
        name if name.starts_with("tirx.cuda.iket_") => "nop",
        "tirx.cuda.func_call" => "ptx",
        "tirx.cuda.trap_when_assert_failed" => "assert",
        "tirx.cuda.thread_fence" => "fence",
        "tirx.cuda.syncthreads_and" | "tirx.cuda.syncthreads_or" => "barrier",
        "tirx.cuda.wait_until" => "wait_until",
        _ => "ptx",
    }
}

fn cell(value: &str) -> String {
    value.replace('|', "\\|").replace('\n', " ")
}

/// Byte-exact port of `transpiler/support_matrix.py::render_engine_support_matrix`.
pub(super) fn render(entries: &[OpEntry]) -> String {
    let mut ops: Vec<&OpEntry> = entries.iter().filter(|e| !e.name.starts_with("tirx.tile.")).collect();
    ops.sort_by_key(|e| e.name);
    let mut tiles: Vec<&OpEntry> = entries.iter().filter(|e| e.name.starts_with("tirx.tile.")).collect();
    tiles.sort_by_key(|e| e.name);
    let mut s = String::from("# NumSim Engine Operation Support\n\n");
    s.push_str("This file is generated from NumSim's operation registry and is checked by tests.\n");
    s.push_str(&format!("It describes NumSim ABI **v{ABI_VERSION}**.\n"));
    s.push_str(
        "A listed operation may still reject modifier, dtype, shape, or layout values outside its exact \
specialization domain. Handwritten runtime cases cover registered operations; GPU parity is tested where \
the required hardware is available.\n\n",
    );
    s.push_str("## TIRx CUDA/PTX Ops\n\n| Operation | Family | Fidelity | Notes |\n| --- | --- | --- | --- |\n");
    for e in ops {
        s.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            cell(&format!("`{}`", e.name)),
            cell(e.family),
            e.fidelity.as_str(),
            cell(e.notes)
        ));
    }
    s.push_str(
        "\n## CUDA Tile Primitives\n\nEvery warp-, warpgroup-, or CTA-scoped tile call validates complete dynamic \
participation in its declared execution scope, including register-only lowerings with no completion barrier.\n\n\
| Operation | Fidelity | Notes |\n| --- | --- | --- |\n",
    );
    for e in tiles {
        s.push_str(&format!("| {} | {} | {} |\n", cell(&format!("`{}`", e.name)), e.fidelity.as_str(), cell(e.notes)));
    }
    s
}
