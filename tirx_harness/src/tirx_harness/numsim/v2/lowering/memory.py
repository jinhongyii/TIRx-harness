"""Buffers, addresses and lvalues.

Every TIRx buffer variable resolves to one of three references:

* :class:`RegArray` — a register-promoted ``local`` buffer (one register per
  element). Promotion requires a static shape, a trivial layout, a
  representable dtype, and that the buffer's address never escapes
  (``address_of`` outside a known out-parameter, ``buffer_data``, or a
  ``DeclBuffer`` view). Constant indices resolve to a register; dynamic
  indices use ``LoadRegIndexed`` / ``StoreRegIndexed`` (decision 2).
* :class:`MemRef` — a ``Program.buffers`` entry: parameter buffers, shared
  allocations and their ``DeclBuffer`` views (static ``elem_offset``), and
  non-promoted locals (per-lane ``Local`` memory). Accesses are
  buffer-relative ``Load``/``Store`` with element offsets in the buffer's own
  dtype; ``address_of`` is ``AddrOf``.
* :class:`PtrRef` — a view whose data is a pointer *value* (``reinterpret``,
  a pointer ``Var``, ``handle_add_byte_offset``). Accesses are raw
  ``LoadAddr``/``StoreAddr`` (generic space) on ``base + offset * itemsize``.

Physical element offsets follow the buffer's layout: row-major (or explicit
strides) linearization, then ``Layout.apply`` for non-trivial ``TileLayout``
and ``ComposeLayout`` (swizzles), lowered as ordinary integer arithmetic.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tvm_ffi import structural_visit

from . import builtins
from . import dtypes
from . import program_builder as pb
from .dtypes import type_key
from .uninit import TRACKED_SPACE

if TYPE_CHECKING:
    from .ir_walk import Lowerer


_SPACES = {
    "global": "Global", "shared": "Shared", "shared.dyn": "Shared", "local": "Local",
    "tmem": "Tmem", "param": "Param",
}

_WEAK = {"sem": "Weak", "scope": "Gpu"}


def handle(node: Any) -> int:
    return int(node.__chandle__())


class _Unsupported(Exception):
    def __init__(self, node: Any, reason: str):
        super().__init__(reason)
        self.node = node
        self.reason = reason


@dataclass
class Shape:
    dtype: str                      # TVM dtype string
    shape: tuple[Any, ...]          # PrimExpr nodes
    strides: tuple[Any, ...]        # explicit stride PrimExprs, or ()
    layout: Any | None              # non-trivial TVM layout, else None
    tmem_cols: int | None = None    # TMEM view: physical column span (dense lane x column addressing)
    tmem_per_cell: int = 1          # TMEM view: elements per 32-bit cell (contract item 28)
    tmem_refresh: Any = None        # TMEM view with a runtime base: emits the base_reg update
    tmem_origin: tuple[Any, Any] | None = None  # TMEM layout offset (lane, col) folded into the base
    elem_base: int = 0              # global same-dtype view folded into its root: element offset

    @property
    def static_shape(self) -> tuple[int, ...] | None:
        out = []
        for extent in self.shape:
            if type_key(extent) != "ir.IntImm":
                return None
            out.append(int(extent.value))
        return tuple(out)


@dataclass
class RegArray:
    regs: list[pb.Reg]
    info: Shape


@dataclass
class MemRef:
    buf: int
    space: str                      # AddrSpace of its accesses
    info: Shape


@dataclass
class PtrRef:
    base: pb.Operand                # u64 generic address (or a shared::cluster window address)
    info: Shape
    elem_offset: Any                # PrimExpr
    space: str = "Generic"          # "SharedCluster": base is a mapa.shared::cluster result


BufferRef = RegArray | MemRef | PtrRef


def buffer_shape(ty: Any) -> Shape:
    layout = getattr(ty, "layout", None)
    if layout is not None:
        trivial = getattr(layout, "is_trivial", None)
        if trivial is not None and trivial():
            layout = None
    return Shape(dtype=str(ty.dtype.dtype), shape=tuple(ty.shape),
                 strides=tuple(getattr(ty, "strides", ()) or ()), layout=layout)


def escaped_locals(body: Any) -> set[int]:
    """Handles of local buffers whose address escapes (they live in memory)."""

    import tvm
    from tvm import tirx

    escaped: set[int] = set()

    def addressed(node: Any) -> int | None:
        if type_key(node) == "ir.TensorLoad":
            return handle(node.source)
        if type_key(node) == "ir.Var":
            return handle(node)
        return None

    def visit_call(call: Any, visitor: Any) -> None:
        name = str(getattr(call.op, "name", ""))
        if name in ("tirx.address_of", "tirx.buffer_data"):
            target = addressed(call.args[0]) if call.args else None
            if target is not None:
                escaped.add(target)
            visitor.default_visit(call)
            return
        helper = builtins.HELPERS.get(name)
        if helper is not None and any(r in "ox" for r in helper.roles.rstrip("*")):
            for position, arg in enumerate(call.args):
                role = builtins.role(helper.roles, position)
                if (
                    role in "ox"
                    and type_key(arg) == "ir.Call"
                    and str(getattr(arg.op, "name", "")) == "tirx.address_of"
                    and type_key(arg.args[0]) == "ir.TensorLoad"
                ):
                    for index in arg.args[0].indices:
                        visitor.visit(index)
                    continue
                visitor.visit(arg)
            return
        visitor.default_visit(call)

    def visit_decl(node: Any, visitor: Any) -> None:
        target = addressed(node.data)
        if target is None and type_key(node.data) == "ir.Call" and \
                str(getattr(node.data.op, "name", "")) == "tirx.buffer_data":
            target = addressed(node.data.args[0])
        if target is not None:
            escaped.add(target)
        visitor.default_visit(node)

    structural_visit(body, [(tvm.ir.Call, visit_call), (tirx.DeclBuffer, visit_decl)])
    return escaped


def sum_bases(buffers: list[pb.BufferDecl], index: int) -> int:
    """Byte offset of ``buffers[index]`` from the start of its view_of chain root."""
    total = 0
    while buffers[index].view_of is not None:
        total += buffers[index].base
        index = buffers[index].view_of
    return total


def promotable_locals(body: Any, escaped: set[int]) -> dict[int, tuple[int, ...]]:
    """Local allocations eligible for register promotion: handle -> static shape."""
    from tvm import tirx

    found: dict[int, tuple[int, ...]] = {}

    def on_alloc(node: Any, visitor: Any) -> None:
        var = node.buffer
        info = buffer_shape(var.ty)
        static = info.static_shape
        if str(var.ty.storage_scope) == "local" and static is not None and info.layout is None \
                and not info.strides and handle(var) not in escaped:
            found[handle(var)] = tuple(static)
        visitor.default_visit(node)

    structural_visit(body, [(tirx.AllocBuffer, on_alloc)])
    return found


class MemoryMixin:
    """Buffer resolution and access lowering (mixed into ``Lowerer``)."""

    # -- declarations ------------------------------------------------------
    def declare_alloc(self: "Lowerer", node: Any) -> None:
        var = node.buffer
        ty = var.ty
        scope = str(ty.storage_scope)
        info = buffer_shape(ty)
        static = info.static_shape
        name = str(var.name)
        elem_ty = self.ty(info.dtype, node)
        if scope == "local" and static is not None and info.layout is None and not info.strides \
                and handle(var) not in self.escaped and handle(var) not in self.uninit_locals:
            count = 1
            for extent in static:
                count *= extent
            regs = [self.builder.reg(elem_ty, name=name) for _ in range(count)]
            self.refs[handle(var)] = RegArray(regs=regs, info=info)
            return
        space = _SPACES.get(scope)
        if handle(var) in self.uninit_locals:
            # May be read before written (V2C-19/20): keep it in tracked memory.
            space = TRACKED_SPACE
        if space not in ("Local", "Shared", "Reg") or static is None:
            raise _Unsupported(node, f"allocation in scope {scope!r} with shape {[str(s) for s in info.shape]}")
        numel = _numel(static)
        buf = self.builder.buffer(
            pb.BufferDecl(
                name=name, space=space, dtype=elem_ty,
                shape=tuple(pb.DimExpr.const(e) for e in static),
                byte_len=pb.DimExpr.const((_layout_span(info.layout, numel) * dtypes.bits(info.dtype) + 7) // 8),
                align=max(16, int(ty.data_alignment), _swizzle_period_bytes(info.layout, info.dtype)),
            )
        )
        if scope == "shared.dyn":
            self.dyn_pools.add(buf)
        self.refs[handle(var)] = MemRef(buf=buf, space=space, info=info)

    def declare_view(self: "Lowerer", node: Any) -> None:
        var = node.buffer
        ty = var.ty
        scope = str(ty.storage_scope)
        info = buffer_shape(ty)
        name = str(var.name)
        data = node.data
        offset = ty.elem_offset
        elem_ty = self.ty(info.dtype, node)
        backing = None
        if type_key(data) == "ir.Call" and str(data.op.name) == "tirx.buffer_data":
            backing = self.refs.get(handle(data.args[0]))
        elif type_key(data) == "ir.Var":
            backing = self.refs.get(handle(data))
        if scope == "tmem":
            self.declare_tmem_view(node, var, ty, info, elem_ty)
            return
        if isinstance(backing, MemRef) and type_key(offset) == "ir.IntImm":
            parent = self.builder.program.buffers[backing.buf]
            if parent.space == "Global" and parent.view_of is None and parent.dtype == elem_ty \
                    and info.layout is None and not info.strides:
                # A same-dtype view of a global buffer is an offset into it, like a
                # C pointer: indices may legally reach before or past the view
                # (W2-20, mega_moe's signed delta to a sibling plane), so it is
                # addressed in the root, whose logical identity it shares anyway.
                info.elem_base = int(offset.value)
                self.refs[handle(var)] = MemRef(buf=backing.buf, space=backing.space, info=info)
                return
            static = info.static_shape
            shape = tuple(pb.DimExpr.const(e) for e in static) if static is not None else \
                tuple(self.dim_expr(e) for e in info.shape)
            span = _strided_span(static, info.strides) if static is not None else None
            if span is not None and info.layout is not None:
                span = _layout_span(info.layout, span)
            byte_len = pb.DimExpr.const((span * dtypes.bits(info.dtype) + 7) // 8) if span is not None else None
            # `elem_offset` counts from the backing's *data pointer*, which a view
            # shares with its own view_of chain root; BufferDecl.base is relative
            # to view_of, so subtract the parent's offset within that chain.
            base = (int(offset.value) * dtypes.bits(info.dtype)) // 8
            view_of = backing.buf
            buffers = self.builder.program.buffers
            chain = view_of
            while buffers[chain].view_of is not None:
                base -= buffers[chain].base
                chain = buffers[chain].view_of
            if base < 0:  # starts before the parent: hang it off the chain root
                base += sum_bases(buffers, view_of)
                view_of = chain
            buf = self.builder.buffer(
                pb.BufferDecl(
                    name=name, space=parent.space, dtype=elem_ty, shape=shape,
                    byte_len=byte_len, align=int(ty.data_alignment), view_of=view_of, base=base,
                )
            )
            self.refs[handle(var)] = MemRef(buf=buf, space=backing.space, info=info)
            return
        if isinstance(backing, RegArray):
            raise _Unsupported(node, "view of a register-promoted local (escape analysis gap)")
        if isinstance(backing, PtrRef):
            # `elem_offset` counts from the shared *data pointer* (the backing's
            # base), not from the backing view's first element: adding the
            # backing's own offset again would apply it twice (W2,
            # sparse_flashmla_decode_head64 `o_ptr.view("uint64")`).
            self.refs[handle(var)] = PtrRef(base=backing.base, info=info, elem_offset=offset, space=backing.space)
            return
        value = self.expr(data)
        space = "Generic"
        if scope in ("shared", "shared.dyn") and self.is_cluster_window(value):
            # A shared-scope buffer over a `mapa.shared::cluster` result: the base
            # is a shared::cluster window address, not a generic pointer (W2).
            space = "SharedCluster"
        base = self.as_address(value)
        self.refs[handle(var)] = PtrRef(base=base, info=info, elem_offset=offset, space=space)

    def is_cluster_window(self: "Lowerer", value: pb.Operand) -> bool:
        """Is ``value`` (through Mov/Cast copies) only ever a mapa.shared::cluster result?"""
        seen: set[int] = set()
        pending = [value]
        while pending:
            operand = pending.pop()
            if not isinstance(operand, pb.Reg) or operand.index in seen:
                if not isinstance(operand, pb.Reg):
                    return False
                continue
            seen.add(operand.index)
            writers = [i for i in self.builder.code if operand in i.writes()]
            if not writers:
                return False
            for instr in writers:
                if instr.variant == "Mapa":
                    if instr.space != "SharedCluster":
                        return False
                elif instr.variant in ("Mov", "Cast"):
                    pending.append(instr.fields["src"])
                else:
                    return False
        return True

    def declare_tmem_view(self: "Lowerer", node: Any, var: Any, ty: Any, info: Shape, elem_ty: pb.Ty) -> None:
        """TMEM ``DeclBuffer`` -> ``Buf`` in ``Space::Tmem`` (coordinator ruling, phase 3).

        CONTRACT convention: the Buf is the physical rectangle the view covers,
        ``shape = [lane_span, col_span]`` (32-bit columns), ``base`` = the
        static taddr (``lane << 16 | col``) of its first cell, and a
        ``Load/Store`` offset is ``lane * col_span + col`` within it. Only
        layouts that map every logical element to one (TLane, TCol) cell are
        representable; replicated layouts and runtime bases fail closed when
        accessed directly.
        """
        addrs = list(getattr(ty, "allocated_addr", ()) or ())
        if len(addrs) != 1:
            raise _Unsupported(node, f"TMEM view with {len(addrs)} allocated addresses")
        if type_key(ty.elem_offset) != "ir.IntImm" or int(ty.elem_offset.value) != 0:
            raise _Unsupported(node, "TMEM view with an element offset")
        layout = getattr(ty, "layout", None)
        spans = _tmem_spans(layout)
        if spans is None:
            raise _Unsupported(node, f"TMEM layout {str(layout)[:80]} is not a dense lane x column map")
        lane_span, col_span, replicated = spans
        # Contract item 28: 8/16-bit views pack `32 / bits` elements per 32-bit
        # cell. TIR counts their TCol in elements, so `col_span` and the column
        # origin are element columns; the Buf's column extent is in cells.
        bits = dtypes.bits(info.dtype)
        per_cell = 32 // bits if bits in (8, 16) else 1
        cells = -(-col_span // per_cell)
        # The layout's (TLane, TCol) offset moves the view's origin: fold it into the
        # base taddr (``lane << 16 | col``) and subtract it from each access.
        offsets = {str(axis.name): expr for axis, expr in (layout.offset.items() if layout.offset else ())}
        lane_off, col_off = offsets.get("TLane"), offsets.get("TCol")
        start = addrs[0]
        static = type_key(start) == "ir.IntImm" and all(
            o is None or type_key(o) == "ir.IntImm" for o in (lane_off, col_off))
        base, base_reg = 0, None
        if static and col_off is not None and int(col_off.value) % per_cell:
            raise _Unsupported(node, f"TMEM view column origin {int(col_off.value)} splits a 32-bit cell")
        if static:
            base = int(start.value) + (int(lane_off.value) << 16 if lane_off is not None else 0) + (
                int(col_off.value) // per_cell if col_off is not None else 0)
        else:
            # Contract item 22: runtime taddr in a register, read at each access;
            # base 0. The address expression is evaluated right before every
            # access, not at the declaration: the view is often declared before
            # `tcgen05.alloc` writes the address (W2-20, mxf8_cta2).
            base_reg = self.builder.reg(pb.Ty("U32"), name=f"{var.name}.taddr")

            def refresh(start=start, lane_off=lane_off, col_off=col_off, per_cell=per_cell,
                        base_reg=base_reg) -> None:
                value = self.cast_to(self.expr(start), pb.Ty("U32"))
                if lane_off is not None:
                    lane_bits = self.binary("Shl", pb.Ty("U32"), self.cast_to(self.expr(lane_off), pb.Ty("U32")),
                                            self.const("uint32", 16))
                    value = self.binary("Add", pb.Ty("U32"), value, lane_bits)
                if col_off is not None:
                    cell_off = self.cast_to(self.expr(col_off), pb.Ty("U32"))
                    if per_cell > 1:
                        cell_off = self.binary("FloorDiv", pb.Ty("U32"), cell_off, self.const("uint32", per_cell))
                    value = self.binary("Add", pb.Ty("U32"), value, cell_off)
                self.builder.emit("Mov", dst=base_reg, src=value)

            info.tmem_refresh = refresh
            if type_key(start) != "ir.IntImm":
                # The address itself comes from a register (an alloc result): it
                # needs a live lease. A static address with a runtime layout
                # offset stays implicit-TMEM eligible.
                self.tmem_runtime_views = True
        if lane_off is not None or col_off is not None:
            info.tmem_origin = (lane_off, col_off)
        buf = self.builder.buffer(
            pb.BufferDecl(
                name=str(var.name), space="Tmem", dtype=elem_ty,
                shape=(pb.DimExpr.const(lane_span), pb.DimExpr.const(cells)),
                base=base, byte_len=pb.DimExpr.const(lane_span * cells * 4), align=4, base_reg=base_reg,
            )
        )
        info.layout = layout
        info.tmem_cols = None if replicated else cells
        info.tmem_per_cell = per_cell
        self.tmem_views = True
        self.refs[handle(var)] = MemRef(buf=buf, space="Tmem", info=info)

    def finish_shared(self: "Lowerer") -> int:
        """Size dynamic pools from their views, assign CTA shared bases; returns static bytes."""
        program = self.builder.program
        ends: dict[int, int] = {}
        for decl in program.buffers:
            if decl.view_of is not None and decl.byte_len is not None and decl.byte_len.is_const:
                end = decl.base + decl.byte_len.value
                ends[decl.view_of] = max(ends.get(decl.view_of, 0), end)
        cursor = 0
        for index, decl in enumerate(program.buffers):
            if decl.space != "Shared" or decl.view_of is not None:
                continue
            size = decl.byte_len.value if decl.byte_len is not None else 0
            if index in self.dyn_pools and self.dyn_smem_bytes is not None and len(self.dyn_pools) == 1:
                # The committed `tirx.dyn_smem_bytes` is the CTA's dynamic shared memory.
                size = max(size, self.dyn_smem_bytes)
            elif size == 0:
                # A zero-extent pool owner is sized by its views' static spans
                # (legacy `test_zero_extent_pool_owner_uses_only_concrete_view_span_evidence`).
                size = ends.get(index, 0)
            # Otherwise a pool is its allocation (what CUDA codegen launches with): a
            # `decl_buffer` view whose declared extent overruns it does not grow it,
            # and its accesses past the pool fail as out_of_bounds at run time.
            align = max(16, decl.align)
            cursor = (cursor + align - 1) // align * align
            program.buffers[index] = dataclasses.replace(
                decl, base=cursor, byte_len=pb.DimExpr.const(size), align=align)
            cursor += size
        return cursor

    # -- offsets -----------------------------------------------------------
    def index_dtype(self: "Lowerer", indices: Any) -> str:
        for index in indices:
            if dtypes.dtype_of(index) in ("int64", "uint64"):
                return "int64"
        return "int32"

    def flat_offset(self: "Lowerer", ref: BufferRef, indices: Any) -> tuple[pb.Operand, int]:
        """Physical element offset of ``indices`` and the access lane count."""
        info = ref.info
        lanes = 1
        idx_dtype = self.index_dtype(indices)
        lowered = []
        for index in indices:
            if type_key(index) == "prim.Ramp":
                stride = index.stride
                if type_key(stride) != "ir.IntImm" or int(stride.value) != 1:
                    raise _Unsupported(index, "non-unit Ramp index")
                lanes = int(index.lanes)
                index = index.base
            lowered.append(self.cast_to(self.expr(index), idx_dtype))
        if info.strides:
            terms = [self.mul_extent(v, s, idx_dtype) for v, s in zip(lowered, info.strides)]
            flat = terms[0] if terms else self.const(idx_dtype, 0)
            for term in terms[1:]:
                flat = self.binary("Add", idx_dtype, flat, term)
        else:
            flat = None
            for axis, value in enumerate(lowered):
                if flat is None:
                    flat = value
                else:
                    flat = self.mul_extent(flat, info.shape[axis], idx_dtype)
                    flat = self.binary("Add", idx_dtype, flat, value)
            if flat is None:
                flat = self.const(idx_dtype, 0)
        if isinstance(ref, MemRef) and ref.space == "Tmem":
            if info.tmem_cols is None:
                # Contract item 29.
                raise _Unsupported(None, f"tmem_replicated_view: {self.builder.program.buffers[ref.buf].name}")
            if info.tmem_refresh is not None:
                info.tmem_refresh()
            width = dtypes.bits(info.dtype) * lanes
            if width not in (8, 16, 32) or (width < 32 and info.tmem_per_cell == 1):
                raise _Unsupported(None, f"direct TMEM access of {info.dtype}x{lanes} (32-bit, 16-bit or 8-bit cells only)")
            lane, col = self.apply_tmem_layout(info.layout, flat, idx_dtype)
            if info.tmem_origin is not None:
                lane_off, col_off = info.tmem_origin
                if lane_off is not None:
                    lane = self.binary("Sub", idx_dtype, lane, self.cast_to(self.expr(lane_off), idx_dtype))
                if col_off is not None:
                    col = self.binary("Sub", idx_dtype, col, self.cast_to(self.expr(col_off), idx_dtype))
            # Element offset: cell (lane * cols + col / per_cell) * per_cell + col % per_cell.
            row = self.mul_extent(lane, _imm(info.tmem_cols * info.tmem_per_cell), idx_dtype)
            return self.binary("Add", idx_dtype, row, col), lanes
        if info.layout is not None:
            flat = self.apply_layout(info.layout, flat, idx_dtype)
        if isinstance(ref, MemRef) and info.elem_base:
            if idx_dtype == "int32" and not -(1 << 31) <= info.elem_base < (1 << 31):
                idx_dtype = "int64"
                flat = self.cast_to(flat, idx_dtype)
            flat = self.binary("Add", idx_dtype, flat, self.const(idx_dtype, info.elem_base))
        if isinstance(ref, MemRef):
            # Buffer-relative offsets count scalar elements of the buffer's
            # `dtype.elem`, not whole vector elements (V2C-11): a `uint32x4`
            # buffer's element i starts at scalar element 4*i.
            vector = _vector_lanes(info.dtype)
            if vector > 1:
                flat = self.mul_extent(flat, _imm(vector), idx_dtype)
        return flat, lanes

    def mul_extent(self: "Lowerer", value: pb.Operand, factor: Any, dtype: str) -> pb.Operand:
        if type_key(factor) == "ir.IntImm":
            amount = int(factor.value)
            if amount == 1:
                return value
            if isinstance(value, pb.Const):
                return self.const(dtype, self.const_int(value) * amount)
            return self.binary("Mul", dtype, value, self.const(dtype, amount))
        return self.binary("Mul", dtype, value, self.cast_to(self.expr(factor), dtype))

    def apply_layout(self: "Lowerer", layout: Any, flat: pb.Operand, dtype: str) -> pb.Operand:
        key = handle(layout)
        cached = self.layout_exprs.get(key)
        if cached is None:
            from tvm import tirx

            var = tirx.Var("flat", dtype)
            mapped = layout.apply(var)
            axes = {str(k): v for k, v in mapped.items()}
            threads = {name: v for name, v in axes.items() if name != "m"}
            unknown = sorted(set(threads) - set(_THREAD_AXES))
            if unknown:
                raise _Unsupported(None, f"layout maps to non-memory axes {sorted(axes)}")
            cached = (var, axes.get("m"), threads)
            self.layout_exprs[key] = cached
        var, expr, threads = cached
        self.vars[handle(var)] = flat
        try:
            # Register (fragment) layouts: element -> (owning thread coordinates,
            # register m). The storage is per thread, so the offset is `m`; an
            # access to an element another thread owns is a runtime Assert.
            for name, coordinate in threads.items():
                owner = self.cast_to(self.expr(coordinate), "int32")
                own = self.thread_coordinate(name)
                ok = self.builder.reg(pb.Ty("Pred"))
                self.builder.emit("Compare", op="Eq", ty=pb.Ty("S32"), dst=ok, a=owner, b=own)
                # Anchor the check at the tile call that produced the access
                # (W11-5), else at the access itself.
                anchor = self.tile_ops[-1] if getattr(self, "tile_ops", None) else getattr(self, "access_node", None)
                site = self.site(anchor) if anchor is not None else None
                self.builder.emit("Assert", site=site, cond=ok, msg=self.builder.string(
                    f"register-layout element owned by another thread ({name})"))
            if expr is None:
                return self.const(dtype, 0)
            return self.cast_to(self.expr(expr), dtype)
        finally:
            del self.vars[handle(var)]

    def thread_coordinate(self: "Lowerer", axis: str) -> pb.Operand:
        sreg, modulus = _THREAD_AXES[axis]
        reg = self.builder.reg(pb.Ty("S32"), name=axis)
        self.builder.emit("ReadSpecial", dst=reg, sreg=sreg)
        if modulus is None:
            return reg
        return self.binary("FloorMod", "int32", reg, self.const("int32", modulus))

    def apply_tmem_layout(self: "Lowerer", layout: Any, flat: pb.Operand, dtype: str) -> tuple[pb.Operand, pb.Operand]:
        from tvm import tirx

        var = tirx.Var("flat", dtype)
        axes = {str(k): v for k, v in layout.apply(var).items()}
        if set(axes) != {"TLane", "TCol"}:
            raise _Unsupported(None, f"TMEM layout maps to axes {sorted(axes)}")
        self.vars[handle(var)] = flat
        try:
            return (self.cast_to(self.expr(axes["TLane"]), dtype), self.cast_to(self.expr(axes["TCol"]), dtype))
        finally:
            del self.vars[handle(var)]

    def element_bytes(self: "Lowerer", dtype: str, offset: pb.Operand) -> pb.Operand:
        """Byte offset (u64) of ``offset`` elements of TVM ``dtype``."""
        width = dtypes.bits(dtype)
        wide = self.cast_to(offset, "int64")
        if width % 8 == 0:
            scaled = wide if width == 8 else self.binary("Mul", "int64", wide, self.const("int64", width // 8))
        else:
            scaled = self.binary("Shr", "int64", self.binary("Mul", "int64", wide, self.const("int64", width)),
                                 self.const("int64", 3))
        return self.reinterpret(scaled, pb.Ty("U64"))

    def as_address(self: "Lowerer", value: pb.Operand) -> pb.Operand:
        """A 64-bit generic address value."""
        ty = self.operand_ty(value)
        if ty == pb.Ty("U64"):
            return value
        if ty.bits == 64:
            return self.reinterpret(value, pb.Ty("U64"))
        return self.cast_to(value, pb.Ty("U64"))

    # -- accesses ----------------------------------------------------------
    def ref_of(self: "Lowerer", var: Any) -> BufferRef | None:
        return self.refs.get(handle(var))

    def access_ty(self: "Lowerer", ref: BufferRef, lanes: int, node: Any) -> pb.Ty:
        ty = self.ty(ref.info.dtype, node)
        return ty if lanes == 1 else ty.with_lanes(ty.lanes * lanes)

    def load(self: "Lowerer", node: Any) -> pb.Operand:
        ref = self.ref_of(node.source)
        if ref is None:
            raise _Unsupported(node, f"load from unknown buffer {node.source.name}")
        if isinstance(ref, RegArray):
            return self.reg_array_read(node, ref, node.indices)
        self.access_node = node
        offset, lanes = self.flat_offset(ref, node.indices)
        ty = self.access_ty(ref, lanes, node)
        dst = self.builder.reg(ty)
        if isinstance(ref, MemRef):
            site = self.site(node, buffer=ref.buf)
            self.builder.emit("Load", site=site, ty=ty, dst=dst, buf=ref.buf, offset=offset,
                              mods=pb.mem_mods(), **_WEAK)
        else:
            addr = self.ptr_address(ref, offset)
            space = ref.space
            if space == "SharedCluster":
                addr = self.cast_to(addr, pb.Ty("U32"))  # 32-bit shared::cluster window address
            self.builder.emit("LoadAddr", site=self.site(node), ty=ty, dst=dst, addr=addr, space=space,
                              mods=pb.mem_mods(), **_WEAK)
        return dst

    def store(self: "Lowerer", node: Any, var: Any, indices: Any, value: pb.Operand) -> None:
        ref = self.ref_of(var)
        if ref is None:
            raise _Unsupported(node, f"store to unknown buffer {var.name}")
        if isinstance(ref, RegArray):
            self.reg_array_write(node, ref, indices, value)
            return
        self.access_node = node
        offset, lanes = self.flat_offset(ref, indices)
        ty = self.access_ty(ref, lanes, node)
        value = self.cast_to(value, ty)
        if isinstance(ref, MemRef):
            site = self.site(node, buffer=ref.buf)
            self.builder.emit("Store", site=site, ty=ty, buf=ref.buf, offset=offset, value=value,
                              mods=pb.mem_mods(), **_WEAK)
        else:
            addr = self.ptr_address(ref, offset)
            space = ref.space
            if space == "SharedCluster":
                addr = self.cast_to(addr, pb.Ty("U32"))  # 32-bit shared::cluster window address
            self.builder.emit("StoreAddr", site=self.site(node), ty=ty, addr=addr, space=space,
                              value=value, mods=pb.mem_mods(), **_WEAK)

    def ptr_address(self: "Lowerer", ref: PtrRef, offset: pb.Operand) -> pb.Operand:
        total = offset
        if not (type_key(ref.elem_offset) == "ir.IntImm" and int(ref.elem_offset.value) == 0):
            ty = self.operand_ty(offset)
            total = self.binary("Add", ty, offset, self.cast_to(self.expr(ref.elem_offset), ty))
        return self.binary("Add", pb.Ty("U64"), ref.base, self.element_bytes(ref.info.dtype, total))

    def address_of(self: "Lowerer", node: Any) -> pb.Operand:
        """``address_of(x)``: generic 64-bit address."""
        target = node.args[0]
        kind = type_key(target)
        if kind == "ir.Var":
            ref = self.refs.get(handle(target))
            if isinstance(ref, MemRef):
                return self.addr_of(ref.buf, self.const("int32", 0))
            raise _Unsupported(node, f"address of variable {target.name}")
        if kind != "ir.TensorLoad":
            raise _Unsupported(node, f"address of {kind}")
        ref = self.ref_of(target.source)
        if ref is None:
            raise _Unsupported(node, f"address of unknown buffer {target.source.name}")
        if isinstance(ref, RegArray):
            raise _Unsupported(node, "address of a register-promoted local")
        offset, _ = self.flat_offset(ref, target.indices)
        if isinstance(ref, MemRef):
            return self.addr_of(ref.buf, offset)
        return self.ptr_address(ref, offset)

    def addr_of(self: "Lowerer", buf: int, offset: pb.Operand) -> pb.Reg:
        dst = self.builder.reg(pb.Ty("U64"))
        self.builder.emit("AddrOf", dst=dst, buf=buf, offset=offset)
        return dst

    def buffer_data(self: "Lowerer", node: Any) -> pb.Operand:
        var = node.args[0]
        ref = self.refs.get(handle(var))
        if isinstance(ref, MemRef):
            return self.addr_of(ref.buf, self.const("int32", 0))
        if isinstance(ref, PtrRef):
            return self.ptr_address(ref, self.const("int32", 0))
        raise _Unsupported(node, f"buffer_data of {var.name}")

    def buffer_target(self: "Lowerer", addr_node: Any) -> tuple[int, pb.Operand] | None:
        """``(buf, element offset)`` if ``addr_node`` is ``address_of(buf[...])`` of a MemRef."""
        if type_key(addr_node) != "ir.Call" or str(getattr(addr_node.op, "name", "")) != "tirx.address_of":
            return None
        target = addr_node.args[0]
        if type_key(target) != "ir.TensorLoad":
            return None
        ref = self.ref_of(target.source)
        if not isinstance(ref, MemRef):
            return None
        offset, lanes = self.flat_offset(ref, target.indices)
        return ref.buf, offset

    # -- register arrays ---------------------------------------------------
    def reg_array_slot(self: "Lowerer", ref: RegArray, indices: Any) -> int | pb.Operand:
        static = ref.info.static_shape
        assert static is not None
        if all(type_key(i) == "ir.IntImm" for i in indices):
            flat = 0
            for extent, index in zip(static, indices):
                flat = flat * extent + int(index.value)
            if not 0 <= flat < len(ref.regs):
                # Out of bounds: keep it a run-time access so the engine reports it
                # (LoadRegIndexed/StoreRegIndexed OOB = error finding).
                return self.const("int32", flat)
            return flat
        offset, lanes = self.flat_offset(ref, indices)
        if lanes != 1:
            raise _Unsupported(None, "vector access to a register-promoted local")
        return offset

    def reg_array_read(self: "Lowerer", node: Any, ref: RegArray, indices: Any) -> pb.Operand:
        if any(type_key(i) == "prim.Ramp" for i in indices):
            raise _Unsupported(node, "vector (Ramp) read of a register-promoted local")
        slot = self.reg_array_slot(ref, indices)
        if isinstance(slot, int):
            reg = ref.regs[slot]
            return self.pred_args.get(reg.index, reg)
        dst = self.builder.reg(self.builder.reg_ty(ref.regs[0]))
        self.builder.emit("LoadRegIndexed", site=self.site(node), dst=dst, base=ref.regs[0],
                          len=len(ref.regs), idx=slot)
        return dst

    def reg_array_write(self: "Lowerer", node: Any, ref: RegArray, indices: Any, value: pb.Operand) -> None:
        if any(type_key(i) == "prim.Ramp" for i in indices):
            raise _Unsupported(node, "vector (Ramp) write of a register-promoted local")
        elem = self.builder.reg_ty(ref.regs[0])
        value = self.cast_to(value, elem)
        slot = self.reg_array_slot(ref, indices)
        if isinstance(slot, int):
            reg = ref.regs[slot]
            self.builder.set_uniform(reg, False)
            self.builder.emit("Mov", dst=reg, src=value)
            return
        self.builder.emit("StoreRegIndexed", site=self.site(node), base=ref.regs[0], len=len(ref.regs),
                          idx=slot, value=value)

    # -- lvalues (destination operands) -----------------------------------
    def lvalue_target(self: "Lowerer", node: Any, ty_hint: pb.Ty | None = None) -> tuple[pb.Reg, Any]:
        """A register to write for lvalue ``node`` and a write-back thunk (or None)."""
        kind = type_key(node)
        if kind == "ir.Call" and str(node.op.name) == "tirx.address_of":
            node = node.args[0]
            kind = type_key(node)
        if kind != "ir.TensorLoad":
            raise _Unsupported(node, f"destination operand is a {kind}")
        ref = self.ref_of(node.source)
        if isinstance(ref, RegArray) and not any(type_key(i) == "prim.Ramp" for i in node.indices):
            slot = self.reg_array_slot(ref, node.indices)
            if isinstance(slot, int):
                reg = ref.regs[slot]
                self.builder.set_uniform(reg, False)
                return reg, None
        dtype = dtypes.dtype_of(node)
        temp = self.builder.reg(self.ty(dtype, node) if dtype else ty_hint)
        source, indices = node.source, node.indices

        def write_back() -> None:
            self.store(node, source, indices, temp)

        return temp, write_back


# Thread axes of register (fragment) layouts -> (special register, modulus).
_THREAD_AXES = {
    "laneid": ("LaneId", None),
    "tid_in_wg": ("ThreadInCta", 128),
    "tid_in_cta": ("ThreadInCta", None),
    "wid_in_wg": ("WarpInCta", 4),
    "warpid": ("WarpInCta", None),
}


def _vector_lanes(dtype: str) -> int:
    try:
        return pb.Ty.from_tvm(str(dtype)).lanes
    except pb.UnrepresentableType:
        return 1


def _imm(value: int) -> Any:
    from tvm import tirx

    return tirx.IntImm("int32", value)


def _tmem_spans(layout: Any) -> tuple[int, int, bool] | None:
    """(lane span, column span, replicated) of a TMEM TileLayout, or None."""
    if layout is None or type_key(layout) != "tirx.TileLayout":
        return None
    if any(str(axis.name) not in ("TLane", "TCol") for axis in (layout.offset or {}).keys()):
        return None
    spans = {"TLane": 0, "TCol": 0}
    for iterator in list(layout.shard) + list(layout.replica):
        axis = str(iterator.axis.name)
        if axis not in spans or type_key(iterator.extent) != "ir.IntImm" or type_key(iterator.stride) != "ir.IntImm":
            return None
        spans[axis] += (int(iterator.extent.value) - 1) * int(iterator.stride.value)
    return spans["TLane"] + 1, spans["TCol"] + 1, len(layout.replica) > 0


def _swizzle_period_bytes(layout: Any, dtype: str) -> int:
    """Byte period of a swizzled (``ComposeLayout``) allocation, else 0.

    TIRx swizzles relative to the buffer start, while tcgen05/wgmma matrix
    descriptors and TMA swizzle the absolute shared address (PTX matrix
    descriptor "base offset" 0): the two agree only when the buffer starts on a
    multiple of the pattern's repeat, 2**(per_element + atom_len + swizzle_len)
    elements (256B for SWIZZLE_32B, 512B for 64B, 1024B for 128B). TIRx's own
    pool allocator places these operands at ``align=1024`` for the same reason.
    """
    if layout is None or type_key(layout) != "tirx.ComposeLayout":
        return 0
    try:
        bits = int(layout.per_element) + int(layout.atom_len) + int(layout.swizzle_len)
    except (AttributeError, TypeError, ValueError):
        return 0
    return max(16, ((1 << bits) * dtypes.bits(dtype) + 7) // 8)


def _layout_span(layout: Any, numel: int) -> int:
    """Physical elements a memory layout covers (its largest ``m`` + 1): a padded,
    swizzled or offset layout may reach past ``numel`` (ComposeLayout padding,
    ``layout.storage()`` offsets). Non-memory or symbolic layouts keep ``numel``."""
    if layout is None or numel <= 0 or numel > (1 << 16):
        return numel
    from tvm import tirx

    top = numel - 1
    try:
        for index in range(numel):
            mapped = {str(k): v for k, v in layout.apply(tirx.IntImm("int32", index)).items()}
            value = mapped.get("m")
            if set(mapped) - {"m"} or value is None or type_key(value) != "ir.IntImm":
                return numel
            top = max(top, int(value.value))
    except Exception:  # noqa: BLE001 - layouts TVM cannot evaluate keep the logical size
        return numel
    return top + 1


def _strided_span(static: tuple[int, ...], strides: tuple[Any, ...]) -> int | None:
    """Elements a view covers: numel when dense, else ``sum((e - 1) * s) + 1``."""
    if not strides:
        return _numel(static)
    if any(type_key(s) != "ir.IntImm" for s in strides) or len(strides) != len(static):
        return None
    if any(e == 0 for e in static):
        return 0
    return sum((e - 1) * int(s.value) for e, s in zip(static, strides)) + 1


def _numel(shape: tuple[int, ...] | None) -> int:
    total = 1
    for extent in shape or ():
        total *= extent
    return total


__all__ = ["BufferRef", "MemRef", "MemoryMixin", "PtrRef", "RegArray", "escaped_locals", "handle"]
