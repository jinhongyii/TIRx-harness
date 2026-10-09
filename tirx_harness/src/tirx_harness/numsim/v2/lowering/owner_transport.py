"""Element-wise tile ops across register-fragment owners ("owner transport").

A `Tx.wg.copy(dst_view, src_view)` (or cast / unary / binary) between two
register-fragment views whose thread layouts differ moves values between
threads. TVM's dispatch has no form for it (its copy fallback has one thread
write every thread's registers, which is undefined on hardware), and legacy
NumSim transported each element to its destination owner. This lowering does
that with existing Program ops:

1. each source fragment whose owners differ from the executor's is staged
   through a shared scratch buffer: every owner stores its elements, then the
   op's scope synchronizes;
2. the executor (the destination's owner, or the sole fragment source's owner
   when the destination is ordinary memory) computes each element it owns from
   its own registers, the scratch, or ordinary memory;
3. a final barrier makes the scratch reusable.

Functions whose tile ops all are such element-wise ops are lowered entirely
here (TVM dispatch is skipped for them); anything else keeps the TVM path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tvm import tirx

from . import program_builder as pb
from .dtypes import type_key
from .memory import MemRef, _Unsupported, buffer_shape, handle
from .tile_forms.copy import layout_thread_axes as _thread_axes
from .tile_forms.copy import unflatten

if TYPE_CHECKING:
    from .ir_walk import Lowerer

ELEMENTWISE = frozenset(
    {
        "copy",
        "cast",
        "add",
        "sub",
        "mul",
        "fdiv",
        "div",
        "maximum",
        "minimum",
        "sqrt",
        "exp",
        "exp2",
        "log",
        "log2",
        "abs",
        "neg",
        "rsqrt",
        "reciprocal",
    }
)
_BINARY = {
    "add": tirx.Add,
    "sub": tirx.Sub,
    "mul": tirx.Mul,
    "fdiv": tirx.Div,
    "div": tirx.Div,
    "maximum": tirx.Max,
    "minimum": tirx.Min,
}
_UNARY = {
    "sqrt": "sqrt",
    "exp": "exp",
    "exp2": "exp2",
    "log": "log",
    "log2": "log2",
    "abs": "abs",
    "rsqrt": "rsqrt",
}


def _region_parts(arg: Any) -> tuple[Any, Any] | None:
    """(buffer var, layout) of a whole-buffer region operand, else None."""
    source = getattr(arg, "source", None)
    if source is None or not hasattr(arg, "region"):
        return None
    ty = getattr(source, "ty", None)
    if ty is None:
        return None
    try:
        shape = [int(s) for s in ty.shape]
        if [int(r.min) for r in arg.region] != [0] * len(shape) or [
            int(r.extent) for r in arg.region
        ] != shape:
            return None
    except (TypeError, ValueError, AttributeError):
        return None
    return source, getattr(ty, "layout", None)


def is_owner_transport(node: Any) -> bool:
    """Element-wise tile op whose fragment operands do not share one owner layout."""
    op = str(node.op.name).rpartition(".")[2]
    if op not in ELEMENTWISE or not node.args:
        return False
    fragments = []
    for arg in node.args:
        parts = _region_parts(arg)
        if parts is not None and _thread_axes(parts[1]):
            fragments.append(str(parts[1]))
    return len(fragments) >= 2 and len(set(fragments)) > 1


def function_is_owner_transport(func: Any) -> bool:
    """All tile ops are element-wise and at least one moves values across owners."""
    from tvm_ffi import structural_visit

    calls: list[Any] = []
    structural_visit(func.body, [(tirx.TilePrimitiveCall, lambda n, v: calls.append(n))])
    return (
        bool(calls)
        and any(is_owner_transport(c) for c in calls)
        and all(_lowerable(c) for c in calls)
    )


def _lowerable(node: Any) -> bool:
    op = str(node.op.name).rpartition(".")[2]
    if op not in ELEMENTWISE or not node.args or dict(node.config):
        return False
    present = [a for a in node.args[1:] if a is not None]
    if op in _UNARY or op in ("neg", "reciprocal", "copy", "cast"):
        if len(present) != 1:
            return False
    elif len(present) != 2:
        return False
    found = [_region_parts(a) for a in node.args if hasattr(a, "region")]
    regions = [r for r in found if r is not None]
    if not regions or len(regions) != len(found):
        return False
    shapes = {tuple(int(s) for s in r[0].ty.shape) for r in regions}
    return len(shapes) == 1


class OwnerTransportMixin:
    """Lowers element-wise tile ops by explicit owner transport (mixed into ``Lowerer``)."""

    def lower_owner_transport(self: Lowerer, node: Any) -> None:
        op = str(node.op.name).rpartition(".")[2]
        if not _lowerable(node):
            raise _Unsupported(
                node, f"tile op {op}: not an element-wise op over whole buffers of one shape"
            )
        dst_parts = _region_parts(node.args[0])
        assert dst_parts is not None  # _lowerable checked every region operand
        dst_var, dst_layout = dst_parts
        # Unary tile ops carry optional trailing operands (None when unused).
        sources = [a for a in node.args[1:] if a is not None]
        shape = [int(s) for s in dst_var.ty.shape]
        numel = 1
        for extent in shape:
            numel *= extent
        dst_axes = _thread_axes(dst_layout)
        fragments = [p for p in (_region_parts(a) for a in sources) if p is not None]
        if dst_axes:
            exec_layout = dst_layout
        else:
            owners = [p for p in fragments if _thread_axes(p[1])]
            if len({str(p[1]) for p in owners}) > 1:
                raise _Unsupported(
                    node, f"tile op {op}: memory destination fed by differently owned fragments"
                )
            exec_layout = owners[0][1] if owners else None
        if exec_layout is None:
            raise _Unsupported(node, f"tile op {op}: no fragment operand to own the elements")
        flat = tirx.Var("owner_i", "int32")

        def owned(layout: Any) -> Any:
            mapped = {str(k): v for k, v in layout.apply(flat).items()}
            cond = None
            for axis in _thread_axes(layout):
                me = tirx.Var(f"me_{axis}", "int32")
                self.vars[handle(me)] = self.thread_coordinate(axis)
                term = tirx.EQ(tirx.Cast("int32", mapped[axis]), me)
                cond = term if cond is None else tirx.And(cond, term)
            return cond

        scope = str(node.scope)
        staged: dict[int, Any] = {}
        stage_stmts = []
        for position, arg in enumerate(sources):
            parts = _region_parts(arg)
            if parts is None or not _thread_axes(parts[1]) or str(parts[1]) == str(exec_layout):
                continue
            var, layout = parts
            scratch = self.owner_scratch(var, numel)
            staged[position] = scratch
            store = tirx.BufferStore(scratch, var[tuple(unflatten(flat, shape))], [flat])
            stage_stmts.append(
                tirx.For(
                    flat,
                    tirx.IntImm("int32", 0),
                    tirx.IntImm("int32", numel),
                    tirx.ForKind.SERIAL,
                    tirx.IfThenElse(owned(layout), store, None),
                )
            )
        values = []
        for position, arg in enumerate(sources):
            if position in staged:
                values.append(staged[position][flat])
            elif hasattr(arg, "region"):
                values.append(arg.source[tuple(unflatten(flat, shape))])
            else:
                values.append(arg)
        value = self.elementwise_value(node, op, values, dst_var.ty.dtype.dtype)
        compute = tirx.For(
            flat,
            tirx.IntImm("int32", 0),
            tirx.IntImm("int32", numel),
            tirx.ForKind.SERIAL,
            tirx.IfThenElse(
                owned(exec_layout), tirx.BufferStore(dst_var, value, unflatten(flat, shape)), None
            ),
        )
        for stmt in stage_stmts:
            self.stmt(stmt)
        if stage_stmts:
            self.scope_barrier(node, scope)
        self.stmt(compute)
        if stage_stmts:
            self.scope_barrier(node, scope)

    def elementwise_value(self: Lowerer, node: Any, op: str, values: list[Any], dtype: Any) -> Any:
        dtype = str(dtype)

        def as_dtype(v: Any) -> Any:
            if type_key(v) == "ir.FloatImm" or type_key(v) == "ir.IntImm":
                return tirx.const(float(v.value) if "float" in dtype else int(v.value), dtype)
            return v if str(v.ty.dtype) == dtype else tirx.Cast(dtype, v)

        if op in ("copy", "cast"):
            return as_dtype(values[0])
        if op in _BINARY and len(values) == 2:
            return _BINARY[op](as_dtype(values[0]), as_dtype(values[1]))
        if op == "neg":
            return tirx.Sub(tirx.const(0, dtype), as_dtype(values[0]))
        if op == "reciprocal":
            return tirx.Div(tirx.const(1.0, dtype), as_dtype(values[0]))
        if op in _UNARY and len(values) == 1:
            return getattr(tirx, _UNARY[op])(as_dtype(values[0]))
        raise _Unsupported(node, f"tile op {op} with {len(values)} operands")

    def owner_scratch(self: Lowerer, like: Any, numel: int) -> Any:
        dtype = str(like.ty.dtype.dtype)
        var = tirx.decl_buffer((numel,), dtype, scope="shared")
        elem = self.ty(dtype)
        buf = self.builder.buffer(
            pb.BufferDecl(
                name=f"{like.name}.transport",
                space="Shared",
                dtype=elem,
                shape=(pb.DimExpr.const(numel),),
                byte_len=pb.DimExpr.const(numel * elem.bits // 8),
                align=16,
            )
        )
        self.refs[handle(var)] = MemRef(buf=buf, space="Shared", info=buffer_shape(var.ty))
        return var

    def scope_barrier(self: Lowerer, node: Any, scope: str) -> None:
        if "warpgroup" in scope:
            self.barrier(
                node, "Sync", self.const("uint32", 8), self.const("uint32", 128)
            )  # TVM warpgroup_sync(8)
        elif "warp" in scope:
            self.builder.emit("WarpSync", site=self.site(node), membermask=self.full_mask())
        else:
            self.barrier(node, "Sync", self.const("uint32", 0), None)


__all__ = ["OwnerTransportMixin", "function_is_owner_transport", "is_owner_transport"]
