"""Builtin call table: TIRx op name -> lowering handler and family.

The table is the single place where an op name is accepted. Anything not in
``HANDLERS`` lowers to ``Unsupported`` (fail closed) with its family named so
coverage reports group gaps by family.

Only a handful of pure scalar builtins have handlers in the skeleton; the
family classification already covers every prefix seen in the corpus
inventory (``docs/development/lowering-inventory.md``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from . import program_builder as pb

if TYPE_CHECKING:
    from .ir_walk import Lowerer


Handler = Callable[["Lowerer", Any, str], pb.Operand]
"""``handler(lowerer, call, result_dtype) -> result operand``."""


# Family classification by op-name prefix, most specific first. The family
# names follow the legacy registry (frontend-rs/src/registry.rs) so coverage
# can be compared one-to-one during migration.
_FAMILY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("tirx.ptx.mbarrier", "sync.mbarrier"),
    ("tirx.ptx.bar", "sync.named_barrier"),
    ("tirx.ptx.barrier", "sync.named_barrier"),
    ("tirx.ptx.fence", "sync.fence"),
    ("tirx.ptx.cp_async_bulk_tensor", "async.tma"),
    ("tirx.ptx.cp_reduce_async_bulk_tensor", "async.tma"),
    ("tirx.ptx.cp_async_bulk", "async.bulk"),
    ("tirx.ptx.cp_reduce_async_bulk", "async.bulk"),
    ("tirx.ptx.cp_async", "async.cp_async"),
    ("tirx.ptx.tcgen05_mma", "tcgen.mma"),
    ("tirx.ptx.tcgen05_ld", "tcgen.ldst"),
    ("tirx.ptx.tcgen05_st", "tcgen.ldst"),
    ("tirx.ptx.tcgen05_cp", "tcgen.copy"),
    ("tirx.ptx.tcgen05", "tcgen.control"),
    ("tirx.cuda.tcgen05_encode", "tcgen.descriptor"),
    ("tirx.ptx.wgmma", "rejected.wgmma"),
    ("tirx.ptx.multimem", "rejected.multimem"),
    ("tirx.ptx.fabric", "rejected.fabric"),
    ("tirx.ptx.mma", "matrix.mma_sync"),
    ("tirx.ptx.ldmatrix", "matrix.ldmatrix"),
    ("tirx.ptx.stmatrix", "matrix.stmatrix"),
    ("tirx.ptx.atom", "memory.atomic"),
    ("tirx.ptx.red", "memory.atomic"),
    ("tirx.ptx.ld", "memory.load"),
    ("tirx.ptx.st", "memory.store"),
    ("tirx.ptx.tensormap", "tensor_map"),
    ("tirx.ptx.elect_sync", "warp.elect"),
    ("tirx.ptx.shfl", "warp.collective"),
    ("tirx.ptx.vote", "warp.collective"),
    ("tirx.ptx.redux", "warp.collective"),
    ("tirx.ptx.match", "warp.collective"),
    ("tirx.ptx.setmaxnreg", "sync.setmaxnreg"),
    ("tirx.ptx.griddepcontrol", "sync.grid_dependency"),
    ("tirx.ptx.clusterlaunchcontrol", "sync.cluster_launch_control"),
    ("tirx.ptx.cvta", "address"),
    ("tirx.ptx.mapa", "address"),
    ("tirx.ptx.getctarank", "address"),
    ("tirx.ptx.", "register"),
    ("tirx.cuda.wait_until", "sync.wait_until"),
    ("tirx.cuda.elect_sync", "warp.elect"),
    ("tirx.cuda.cta_sync", "sync.named_barrier"),
    ("tirx.cuda.warp_sync", "sync.named_barrier"),
    ("tirx.cuda.warpgroup_sync", "sync.named_barrier"),
    ("tirx.cuda.cluster_sync", "sync.cluster_barrier"),
    ("tirx.cuda.grid_sync", "sync.grid_barrier"),
    ("tirx.cuda.func_call", "cuda_helper"),
    ("tirx.cuda.mbarrier", "sync.mbarrier"),
    ("tirx.cuda.", "cuda"),
    ("tirx.tile.", "tile"),
    ("tirx.", "pure"),
)


def family(op_name: str) -> str:
    for prefix, name in _FAMILY_PREFIXES:
        if op_name.startswith(prefix):
            return name
    return "unknown"


# Ops that may return ``Blocked(resource)`` in the engine. The lowering must
# not hoist or duplicate them (they are scheduling points). Seeded from the
# legacy ``suspends`` rows; kept as a prefix test until the contract settles.
_BLOCKING_PREFIXES = (
    "tirx.ptx.mbarrier_try_wait",
    "tirx.ptx.mbarrier_test_wait",
    "tirx.cuda.mbarrier_wait",
    "tirx.ptx.bar_sync",
    "tirx.ptx.barrier_sync",
    "tirx.ptx.barrier_cluster_wait",
    "tirx.cuda.cta_sync",
    "tirx.cuda.warpgroup_sync",
    "tirx.cuda.cluster_sync",
    "tirx.cuda.grid_sync",
    "tirx.ptx.cp_async_wait",
    "tirx.ptx.cp_async_bulk_wait_group",
    "tirx.ptx.tcgen05_wait",
    "tirx.cuda.wait_until",
)


def may_block(op_name: str) -> bool:
    return op_name.startswith(_BLOCKING_PREFIXES)


# --------------------------------------------------------------------------
# Handlers (skeleton subset)
# --------------------------------------------------------------------------


def _unary(op: str) -> Handler:
    def handler(lowerer: "Lowerer", call: Any, dtype: str) -> pb.Operand:
        (arg,) = call.args
        src = lowerer.expr(arg)
        dst = lowerer.builder.reg(dtype, uniform=lowerer.is_uniform(src))
        # CONTRACT: math intrinsics may become their own Instr variant
        # (Instr::Math { op, dtype, ... }) instead of Unary.
        lowerer.builder.emit(pb.Unary(op=op, dtype=dtype, dst=dst, a=src))
        return dst

    return handler


def _if_then_else(lowerer: "Lowerer", call: Any, dtype: str) -> pb.Operand:
    # tirx.if_then_else evaluates both arms in SIMT order only for pure arms;
    # arms containing loads are lowered as Select over both values, which is
    # what CUDA codegen also emits for pure arms. Arms with side effects are
    # rejected by the frontend contract (calls with effects are statements).
    cond, a, b = call.args
    c, x, y = lowerer.expr(cond), lowerer.expr(a), lowerer.expr(b)
    dst = lowerer.builder.reg(dtype, uniform=all(lowerer.is_uniform(v) for v in (c, x, y)))
    lowerer.builder.emit(pb.Select(dtype=dtype, dst=dst, cond=c, a=x, b=y))
    return dst


HANDLERS: dict[str, Handler] = {
    "tirx.if_then_else": _if_then_else,
    "tirx.exp": _unary("Exp"),
    "tirx.exp2": _unary("Exp2"),
    "tirx.log": _unary("Log"),
    "tirx.log2": _unary("Log2"),
    "tirx.sqrt": _unary("Sqrt"),
    "tirx.rsqrt": _unary("Rsqrt"),
    "tirx.fabs": _unary("Abs"),
}


__all__ = ["HANDLERS", "family", "may_block"]
