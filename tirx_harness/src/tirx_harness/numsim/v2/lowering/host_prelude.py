"""Parameters, the host TensorMap prelude, and ``DimExpr`` (decisions 3 and 8).

Host ABI slots are created in parameter order:

* a global ``T.Buffer`` -> ``Buffer`` slot (+ a global ``BufferDecl``); every
  dynamic shape variable that is not a parameter becomes an implicit
  ``Scalar`` slot bound from that buffer's runtime shape;
* a primitive scalar -> ``Scalar`` slot (read once at entry with ``ReadParam``);
* ``PointerType(TensorMapType)`` -> ``TensorMap`` slot (a 128-byte ``Param``
  buffer whose address ``address_of(var)`` yields);
* ``PointerType(PrimType)`` -> ``Pointer`` slot (a u64 register).

Statements before the single top-level ``tirx.device_entry`` are host code.
Accepted forms (everything else fails closed):

* ``v = T.tvm_stack_alloca("tensormap", 1)`` then either
  ``T.tensormap_encode_tiled(v, base, dims, strides, box, elem_strides)``
  with ``TensorMapEncodeTiledAttr`` (current TVM, which the legacy
  normalizer rejects) or
  ``T.tvm_call_packed("runtime.cuTensorMapEncodeTiled", v, dtype, rank, base, ...)``
  (older form) -> an implicit ``TensorMap`` slot with a ``TensorMapSpec``;
* integer ``Bind`` (host scalars, inlined where referenced);
* ``AssertStmt`` (checked at entry) and ``DeclBuffer`` aliases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import dtypes
from . import program_builder as pb
from .dtypes import type_key
from .memory import MemRef, Shape, _Unsupported, handle

if TYPE_CHECKING:
    from .ir_walk import Lowerer


_DIM_OPS = {
    "prim.Add": "Add", "prim.Sub": "Sub", "prim.Mul": "Mul", "prim.FloorDiv": "FloorDiv",
    "prim.Min": "Min", "prim.Max": "Max",
    # Extents are non-negative, where truncating and flooring division agree.
    "prim.Div": "FloorDiv",
}


class PreludeMixin:
    """Mixed into ``Lowerer``."""

    # -- parameters ---------------------------------------------------------
    def bind_params(self: "Lowerer") -> None:
        program = self.builder.program
        buffers: list[tuple[int, Any]] = []
        for index, param in enumerate(self.func.params):
            ty = param.ty
            kind = type_key(ty)
            name = str(param.name)
            slot = len(program.host_abi)
            if kind == "tirx.BufferType":
                program.host_abi.append(pb.ParamSlot(name=name, kind="Buffer", param_index=index))
                buffers.append((slot, param))
            elif kind == "ir.PrimType" and dtypes.known(str(ty.dtype)) and dtypes.bits(str(ty.dtype)) > 256:
                # Coordinator ruling: register values are capped at 256 bits; a wider
                # by-value parameter (e.g. boolx128) is a u8[N] Param-space buffer.
                nbytes = (dtypes.bits(str(ty.dtype)) + 7) // 8
                buf = self.builder.buffer(
                    pb.BufferDecl(name=name, space="Param", dtype=pb.Ty("U8"), shape=(pb.DimExpr.const(nbytes),),
                                  param_slot=slot, byte_len=pb.DimExpr.const(nbytes), align=16)
                )
                program.host_abi.append(pb.ParamSlot(name=name, kind="Buffer", param_index=index, dtype=pb.Ty("U8"),
                                                     shape=(pb.DimExpr.const(nbytes),), buf=buf))
                self.wide_params.add(handle(param))
                self.refs[handle(param)] = MemRef(
                    buf=buf, space="Param", info=Shape(dtype="uint8", shape=(), strides=(), layout=None))
            elif kind == "ir.PrimType":
                program.host_abi.append(pb.ParamSlot(name=name, kind="Scalar", param_index=index,
                                                     dtype=self.ty(str(ty.dtype), param)))
                self.scalar_slots[handle(param)] = slot
            elif kind == "ir.PointerType" and type_key(ty.element_type) == "tirx.TensorMapType":
                program.host_abi.append(pb.ParamSlot(name=name, kind="TensorMap", param_index=index))
                self.tensor_map_buffer(param, name, slot)
            elif kind == "ir.PointerType":
                element = getattr(ty.element_type, "dtype", None)
                element_ty = None
                if element is not None and str(element) not in ("", "void"):
                    element_ty = self.ty(str(element), param)
                program.host_abi.append(pb.ParamSlot(name=name, kind="Pointer", param_index=index,
                                                     dtype=element_ty))
                self.scalar_slots[handle(param)] = slot
            else:
                raise _Unsupported(param, f"parameter {name} of type {kind}")
        for slot, param in buffers:
            self.bind_buffer_param(slot, param)

    def bind_buffer_param(self: "Lowerer", slot: int, param: Any) -> None:
        program = self.builder.program
        ty = param.ty
        name = str(param.name)
        scope = str(ty.storage_scope)
        info = Shape(dtype=str(ty.dtype.dtype), shape=tuple(ty.shape),
                     strides=tuple(getattr(ty, "strides", ()) or ()), layout=None)
        layout = getattr(ty, "layout", None)
        if layout is not None and not layout.is_trivial():
            info.layout = layout
        if scope != "global":
            raise _Unsupported(param, f"buffer parameter {name} in scope {scope}")
        if type_key(ty.elem_offset) != "ir.IntImm" or int(ty.elem_offset.value) != 0:
            raise _Unsupported(param, f"buffer parameter {name} with an element offset")
        index = program.host_abi[slot].param_index
        # Implicit shape variables become ``ParamKind::ImplicitShape{buffer, axis}``
        # slots (contract item 16); the binder takes their value from the bound array.
        for axis, extent in enumerate(ty.shape):
            if type_key(extent) == "ir.Var" and handle(extent) not in self.scalar_slots:
                shape_slot = len(program.host_abi)
                program.host_abi.append(
                    pb.ParamSlot(name=f"{name}.shape{axis}", local_name=str(extent.name), kind="ImplicitShape",
                                 dtype=self.ty(dtypes.dtype_of(extent), extent), shape_of=(slot, axis))
                )
                self.scalar_slots[handle(extent)] = shape_slot
        shape = tuple(self.dim_expr(extent) for extent in ty.shape)
        strides = tuple(self.dim_expr(s) for s in info.strides)
        elem_ty = self.ty(info.dtype, param)
        buf = self.builder.buffer(
            pb.BufferDecl(name=name, space="Global", dtype=elem_ty, shape=shape, strides=strides,
                          param_slot=slot, byte_len=None, align=int(ty.data_alignment))
        )
        program.host_abi[slot] = pb.ParamSlot(name=name, kind="Buffer", param_index=index, dtype=elem_ty,
                                              shape=shape, buf=buf)
        self.refs[handle(param)] = MemRef(buf=buf, space="Global", info=info)

    def tensor_map_buffer(self: "Lowerer", var: Any, name: str, slot: int) -> None:
        buf = self.builder.buffer(
            pb.BufferDecl(name=name, space="Param", dtype=pb.Ty("U8"), shape=(pb.DimExpr.const(128),),
                          param_slot=slot, byte_len=pb.DimExpr.const(128), align=64)
        )
        self.builder.program.host_abi[slot].buf = buf
        self.refs[handle(var)] = MemRef(
            buf=buf, space="Param", info=Shape(dtype="uint8", shape=(), strides=(), layout=None)
        )

    def read_params(self: "Lowerer") -> None:
        """``ReadParam`` every scalar/pointer/shape slot once at entry."""
        for var_handle, slot in self.scalar_slots.items():
            decl = self.builder.program.host_abi[slot]
            ty = pb.Ty("U64") if decl.kind == "Pointer" else decl.dtype
            reg = self.builder.reg(ty, name=decl.local_name or decl.name, uniform=True)
            self.builder.emit("ReadParam", dst=reg, slot=slot)
            self.vars[var_handle] = reg

    # -- DimExpr ------------------------------------------------------------
    def dim_expr(self: "Lowerer", node: Any) -> pb.DimExpr:
        kind = type_key(node)
        if kind == "ir.IntImm":
            return pb.DimExpr.const(int(node.value))
        if kind == "ir.Var":
            slot = self.scalar_slots.get(handle(node))
            if slot is not None:
                return pb.DimExpr.param(slot)
            bound = self.host_binds.get(handle(node))
            if bound is not None:
                return self.dim_expr(bound)
            raise _Unsupported(node, f"extent references non-parameter variable {node.name}")
        if kind == "prim.Cast":
            return self.dim_expr(node.value)
        op = _DIM_OPS.get(kind)
        if op is None:
            raise _Unsupported(node, f"extent expression {kind} is not a DimExpr")
        a, b = self.dim_expr(node.a), self.dim_expr(node.b)
        if a.is_const and b.is_const:
            folded = _fold(op, a.value, b.value)
            if folded is not None:
                return pb.DimExpr.const(folded)
        return pb.DimExpr(op, args=(a, b))

    # -- host prelude -------------------------------------------------------
    def split_prelude(self: "Lowerer", body: Any) -> list[Any]:
        """Consume host statements; return the device statements to lower."""
        statements = list(body.seq) if type_key(body) == "tirx.SeqStmt" else [body]
        entries = [
            i for i, s in enumerate(statements)
            if type_key(s) == "tirx.AttrStmt" and str(s.attr_key) in ("tirx.device_entry", "thread_extent")
        ]
        if not entries:
            return statements
        if len(entries) != 1:
            raise _Unsupported(body, "more than one top-level device entry")
        entry = entries[0]
        if entry != len(statements) - 1:
            raise _Unsupported(statements[entry + 1], "host statements after tirx.device_entry")
        allocations: dict[int, Any] = {}
        encoded: set[int] = set()
        device_prefix: list[Any] = []
        for statement in statements[:entry]:
            kind = type_key(statement)
            if kind == "tirx.Bind":
                value = statement.value
                if type_key(value) == "ir.Call" and str(value.op.name) == "tirx.tvm_stack_alloca":
                    allocations[handle(statement.var)] = statement.var
                    continue
                dtype = dtypes.dtype_of(value)
                if not dtype.startswith(("int", "uint")):
                    raise _Unsupported(statement, f"host binding of dtype {dtype}")
                self.host_binds[handle(statement.var)] = value
                continue
            if kind == "tirx.Evaluate" and type_key(statement.value) == "ir.Call":
                call = statement.value
                name = str(call.op.name)
                if name == "tirx.tensormap_encode_tiled":
                    self.encode_tensor_map(call, allocations, encoded, legacy=False)
                    continue
                if name == "tirx.tvm_call_packed" and call.args and \
                        str(getattr(call.args[0], "value", "")) == "runtime.cuTensorMapEncodeTiled":
                    self.encode_tensor_map(call, allocations, encoded, legacy=True)
                    continue
            if kind in ("tirx.AssertStmt", "tirx.DeclBuffer"):
                device_prefix.append(statement)
                continue
            raise _Unsupported(statement, f"host statement before tirx.device_entry: {kind}")
        missing = [str(v.name) for h, v in allocations.items() if h not in encoded]
        if missing:
            raise _Unsupported(body, f"host TensorMaps are not encoded: {missing}")
        return device_prefix + [statements[entry]]

    def encode_tensor_map(self: "Lowerer", call: Any, allocations: dict[int, Any], encoded: set[int],
                          *, legacy: bool) -> None:
        args = list(call.args)
        if legacy:
            target, dtype_node, rank_node, base = args[1], args[2], args[3], args[4]
            dtype = str(dtype_node.value)
            rank = int(rank_node.value)
            rest = args[5:]
            values = rest[:4 * rank - 1]
            fixed = rest[4 * rank - 1:]
            interleave, swizzle, l2, fill = (int(v.value) for v in fixed[:4])
            force = int(fixed[4].value) if len(fixed) > 4 else -1
        else:
            target, base = args[0], args[1]
            attrs = call.attrs
            dtype = str(attrs.descriptor_dtype)
            rank = int(attrs.rank)
            values = args[2:]
            interleave, swizzle, l2, fill = (int(attrs.interleave), int(attrs.swizzle),
                                             int(attrs.l2_promotion), int(attrs.oob_fill))
            force = int(attrs.force_cu_dtype)
        if type_key(target) != "ir.Var" or handle(target) not in allocations:
            raise _Unsupported(call, "TensorMap encode targets no preceding tvm_stack_alloca")
        if handle(target) in encoded:
            raise _Unsupported(call, f"TensorMap {target.name} is encoded more than once")
        if not 1 <= rank <= 5 or len(values) != 4 * rank - 1:
            raise _Unsupported(call, f"rank-{rank} TensorMap encode has {len(values)} extent arguments")
        base_slot, base_offset = self.tensor_map_base(base)
        dims = [self.dim_expr(v) for v in values]
        box = dims[2 * rank - 1:3 * rank - 1]
        elem = dims[3 * rank - 1:]
        if not all(d.is_const for d in box + elem):
            raise _Unsupported(call, "TensorMap box/element strides must be static")
        spec = pb.TensorMapSpec(
            dtype=self.ty(dtype, call).elem, rank=rank, global_dim=tuple(dims[:rank]),
            global_stride=tuple(dims[rank:2 * rank - 1]), box_dim=tuple(d.value for d in box),
            element_stride=tuple(d.value for d in elem), interleave=interleave, swizzle=swizzle,
            l2_promotion=l2, oob_fill=fill, base_offset=base_offset,
            # Raw CUtensorMapDataType override (contract item 15): the engine decides.
            force_cu_dtype=None if force == -1 else force,
        )
        program = self.builder.program
        slot = len(program.host_abi)
        name = str(target.name)
        program.host_abi.append(pb.ParamSlot(name=name, kind="TensorMap", tensor_map=spec,
                                             implicit_base=base_slot))
        self.tensor_map_buffer(target, name, slot)
        encoded.add(handle(target))

    def tensor_map_base(self: "Lowerer", node: Any) -> tuple[int, pb.DimExpr]:
        offset = pb.DimExpr.const(0)
        while type_key(node) == "ir.Call" and str(node.op.name) == "tirx.handle_add_byte_offset":
            extra = self.dim_expr(node.args[1])
            offset = extra if offset.is_const and offset.value == 0 else pb.DimExpr("Add", args=(offset, extra))
            node = node.args[0]
        if type_key(node) == "ir.Call" and str(node.op.name) == "tirx.buffer_data":
            node = node.args[0]
        if type_key(node) == "ir.Var":
            ref = self.refs.get(handle(node))
            if isinstance(ref, MemRef):
                slot = self.builder.program.buffers[ref.buf].param_slot
                if slot is not None:
                    return slot, offset
            slot = self.scalar_slots.get(handle(node))
            if slot is not None and self.builder.program.host_abi[slot].kind == "Pointer":
                return slot, offset
        raise _Unsupported(node, "TensorMap base must be a buffer or pointer parameter")


def _fold(op: str, a: int, b: int) -> int | None:
    if op == "Add":
        return a + b
    if op == "Sub":
        return a - b
    if op == "Mul":
        return a * b
    if op == "FloorDiv":
        return a // b if b else None
    if op == "Min":
        return min(a, b)
    if op == "Max":
        return max(a, b)
    return None


__all__ = ["PreludeMixin"]
