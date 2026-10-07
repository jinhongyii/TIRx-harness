"""TIRx ``PrimFunc`` -> provisional ``Program`` lowering (skeleton).

Scope of the skeleton (see ``docs/development/lowering-inventory.md`` for the
full design):

* parameters: global ``T.Buffer`` and primitive scalars;
* ``ScopeIdDefStmt`` for every scope binding with static extents;
* ``AllocBuffer`` of ``local`` scalars and small static local arrays (register
  promoted), ``shared``/``shared.dyn`` with trivial layouts;
* ``BufferStore`` / ``TensorLoad`` with trivial (row-major ``TileLayout``)
  addressing;
* integer/float/bool arithmetic, comparisons, casts, ``Select``;
* ``IfThenElse``, serial/unrolled ``For``, ``While``;
* the pure builtins in ``builtins.HANDLERS``.

Everything else lowers to ``Unsupported`` and is collected in
``Program.unsupported``; ``lower(..., strict=True)`` (the default) then
raises ``LoweringUnsupported`` listing every gap with its site, mirroring the
legacy collect-then-reject behaviour.

The walker dispatches on the FFI ``type_key`` rather than Python classes so it
is insensitive to Python wrapper renames across TVM versions.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any

from . import builtins
from . import program_builder as pb


class LoweringUnsupported(Exception):
    """The PrimFunc uses constructs the lowering does not model (fail closed)."""

    def __init__(self, program: pb.Program):
        self.program = program
        self.reasons = list(program.unsupported)
        head = "; ".join(self.reasons[:8])
        more = f" (+{len(self.reasons) - 8} more)" if len(self.reasons) > 8 else ""
        super().__init__(f"{program.name}: unsupported TIRx: {head}{more}")


def type_key(node: Any) -> str:
    info = getattr(type(node), "__tvm_ffi_type_info__", None)
    return info.type_key if info is not None else type(node).__name__


def _dtype_of(node: Any) -> str:
    ty = getattr(node, "ty", None)
    dtype = getattr(ty, "dtype", None)
    return str(dtype) if dtype is not None else ""


def _handle(node: Any) -> int:
    """Identity key for FFI objects (Var / buffer Var)."""
    return int(node.__chandle__())


_ITEMSIZE = {
    "bool": 1, "int8": 1, "uint8": 1, "int16": 2, "uint16": 2, "float16": 2,
    "bfloat16": 2, "int32": 4, "uint32": 4, "float32": 4, "int64": 8,
    "uint64": 8, "float64": 8, "float8_e4m3fn": 1, "float8_e5m2": 1,
    "float8_e8m0fnu": 1,
}

_INT_BITS = {
    "bool": 1, "int8": 8, "uint8": 8, "int16": 16, "uint16": 16, "int32": 32,
    "uint32": 32, "int64": 64, "uint64": 64,
}

_BINARY = {
    "prim.Add": "Add", "prim.Sub": "Sub", "prim.Mul": "Mul", "prim.Div": "Div",
    "prim.Mod": "Mod", "prim.FloorDiv": "FloorDiv", "prim.FloorMod": "FloorMod",
    "prim.Min": "Min", "prim.Max": "Max", "prim.BitwiseAnd": "And",
    "prim.BitwiseOr": "Or", "prim.BitwiseXor": "Xor", "prim.LShift": "Shl",
    "prim.RShift": "Shr", "prim.And": "And", "prim.Or": "Or",
}

_COMPARE = {
    "prim.EQ": "Eq", "prim.NE": "Ne", "prim.LT": "Lt", "prim.LE": "Le",
    "prim.GT": "Gt", "prim.GE": "Ge",
}

# ScopeBinding (tvm/tirx/exec_scope.h) -> how the flat id is computed.
_SCOPE_BINDINGS = {
    0: ("kernel", "cluster"), 1: ("kernel", "cta"), 2: ("cluster", "cta"),
    3: ("cta", "warpgroup"), 4: ("cta", "warp"), 5: ("warpgroup", "warp"),
    6: ("warp", "thread"), 7: ("cta", "thread"), 8: ("warpgroup", "thread"),
    9: ("cluster", "cta_pair"),
}

# AttrStmt keys that carry no numerical semantics for NumSim.
_IGNORED_ATTRS = frozenset(
    {
        "tirx.device_entry",
        "tirx.launch_bounds_min_blocks_per_sm",
        "tirx.required_block_size",
        "tirx.max_registers",
        "tirx.dyn_smem_bytes",   # CONTRACT: becomes a Launch field once sized
        "tirx.pool_max_bytes",
    }
)


@dataclass
class _LocalArray:
    """A register-promoted local buffer: one register per element."""

    regs: list[pb.Reg]
    dtype: str


@dataclass
class _Scope:
    vars: dict[int, pb.Reg] = field(default_factory=dict)


class Lowerer:
    def __init__(self, func: Any, name: str):
        self.func = func
        self.builder = pb.ProgramBuilder(name)
        self.vars: dict[int, pb.Operand] = {}
        self.local_arrays: dict[int, _LocalArray] = {}
        self.buffers: dict[int, int] = {}          # buffer var handle -> Program.buffers idx
        self.scope_extents: dict[tuple[str, str], int] = {}

    # ------------------------------------------------------------------ util
    def unsupported(self, node: Any, reason: str) -> None:
        site = self.site(node)
        message = f"site#{site} {type_key(node)}: {reason}"
        self.builder.program.unsupported.append(message)
        # CONTRACT: Instr::Unsupported { reason, site }
        self.builder.emit(pb.Unsupported(reason=reason, site=site))

    def site(self, node: Any, op_name: str | None = None, buffer: int | None = None) -> int:
        spans = _spans(getattr(node, "span", None))
        dtype = _dtype_of(node) or None
        return self.builder.site(
            pb.SiteInfo(
                kind=type_key(node), spans=spans, op_name=op_name, dtype=dtype, buffer=buffer
            )
        )

    def is_uniform(self, operand: pb.Operand) -> bool:
        if isinstance(operand, pb.Const):
            return True
        return self.builder.reg_uniform(operand)

    def operand_dtype(self, operand: pb.Operand) -> str:
        if isinstance(operand, pb.Const):
            return self.builder.const_dtype(operand)
        return self.builder.reg_dtype(operand)

    def const(self, dtype: str, value: int | float) -> pb.Const:
        try:
            return self.builder.const(dtype, _scalar_bits(dtype, value))
        except ValueError:
            # Placeholders for already-reported gaps may name vector/narrow
            # dtypes the skeleton cannot encode; the gap itself fails closed.
            if not self.builder.program.unsupported:
                raise
            return self.builder.const("int32", 0)

    # ---------------------------------------------------------------- entry
    def lower(self) -> pb.Program:
        program = self.builder.program
        attrs = self.func.attrs
        if attrs is not None and "tirx.cuda_arch" in attrs:
            program.arch = str(attrs["tirx.cuda_arch"])
        for index, param in enumerate(self.func.params):
            self.bind_param(index, param)
        self.collect_topology(self.func.body)
        self.stmt(self.func.body)
        return self.builder.finish()

    def bind_param(self, index: int, param: Any) -> None:
        ty = param.ty
        kind = type_key(ty)
        name = str(param.name)
        program = self.builder.program
        if kind == "tirx.BufferType":
            shape, strides = _static_shape_strides(ty)
            dtype = str(ty.dtype.dtype)
            if shape is None or str(ty.storage_scope) != "global":
                program.unsupported.append(f"param {name}: dynamic or non-global buffer")
                return
            slot = len(program.host_abi)
            # CONTRACT: ParamSlot
            program.host_abi.append(
                pb.ParamSlot(name=name, kind="buffer", param_index=index, dtype=dtype, shape=shape)
            )
            buf = self.builder.buffer(
                pb.BufferDecl(
                    name=name, space=pb.Space.GLOBAL, dtype=dtype, shape=shape,
                    strides=strides, param_slot=slot,
                    byte_size=_numel(shape) * _ITEMSIZE.get(dtype, 0),
                )
            )
            self.buffers[_handle(param)] = buf
        elif kind == "ir.PrimType":
            dtype = str(ty.dtype)
            slot = len(program.host_abi)
            program.host_abi.append(
                pb.ParamSlot(name=name, kind="scalar", param_index=index, dtype=dtype)
            )
            # CONTRACT: scalar params need an Instr::ReadParam { dst, slot } (or
            # a pre-initialized uniform register). Until then the skeleton
            # records the slot and fails closed if the scalar is used.
            self.vars[_handle(param)] = None  # type: ignore[assignment]
        else:
            program.unsupported.append(f"param {name}: {kind} not supported by the skeleton")

    # ------------------------------------------------------------- topology
    def collect_topology(self, body: Any) -> None:
        """Static launch topology from the ScopeIdDef extents (pre-pass)."""

        stack = [body]
        while stack:
            node = stack.pop()
            kind = type_key(node)
            if kind == "tirx.ScopeIdDefStmt":
                definition = getattr(node, "def")
                binding = _SCOPE_BINDINGS.get(int(definition.scope))
                extents = definition.extents
                if binding is None or extents is None:
                    continue
                total = 1
                for extent in extents:
                    if type_key(extent) != "ir.IntImm":
                        total = None
                        break
                    total *= int(extent.value)
                if total is not None:
                    self.scope_extents[binding] = total
            elif kind == "tirx.SeqStmt":
                stack.extend(node.seq)
            elif kind in ("tirx.AttrStmt",):
                stack.append(node.body)
        ext = self.scope_extents
        threads = ext.get(("cta", "thread"))
        if threads is None and ("cta", "warp") in ext:
            threads = ext[("cta", "warp")] * 32
        ctas_per_cluster = ext.get(("cluster", "cta"), 1)
        clusters = ext.get(("kernel", "cluster"))
        if clusters is None and ("kernel", "cta") in ext:
            clusters = max(1, ext[("kernel", "cta")] // ctas_per_cluster)
        if threads is None or clusters is None:
            self.builder.program.unsupported.append(
                "topology: launch extents are deferred or dynamic"
            )
            return
        self.builder.program.topology = pb.Launch(
            clusters=clusters, ctas_per_cluster=ctas_per_cluster, threads_per_cta=threads
        )

    # ----------------------------------------------------------- statements
    def stmt(self, node: Any) -> None:
        kind = type_key(node)
        method = getattr(self, "stmt_" + kind.replace(".", "_"), None)
        if method is None:
            self.unsupported(node, "statement kind not lowered")
            return
        method(node)

    def stmt_tirx_SeqStmt(self, node: Any) -> None:
        for child in node.seq:
            self.stmt(child)

    def stmt_tirx_AttrStmt(self, node: Any) -> None:
        key = str(node.attr_key)
        if key not in _IGNORED_ATTRS:
            self.unsupported(node, f"attribute {key!r}")
        self.stmt(node.body)

    def stmt_tirx_ScopeIdDefStmt(self, node: Any) -> None:
        definition = getattr(node, "def")
        scope = int(definition.scope)
        variables = list(definition.def_ids)
        extents = definition.extents
        flat = self.scope_flat_id(node, scope)
        if flat is None:
            return
        if len(variables) == 1:
            self.vars[_handle(variables[0])] = self.cast_to(flat, _dtype_of(variables[0]))
            return
        if extents is None or any(type_key(e) != "ir.IntImm" for e in extents):
            self.unsupported(node, "multi-coordinate scope ids need static extents")
            return
        # The first coordinate is fastest-varying (legacy emit/stmt.rs:1200).
        divisor = 1
        for var, extent in zip(variables, extents):
            value = int(extent.value)
            quotient = self.binary("FloorDiv", "int32", flat, self.const("int32", divisor))
            coordinate = self.binary("FloorMod", "int32", quotient, self.const("int32", value))
            self.vars[_handle(var)] = self.cast_to(coordinate, _dtype_of(var))
            divisor *= value

    def scope_flat_id(self, node: Any, scope: int) -> pb.Operand | None:
        b = self.builder
        wpg = 4

        def special(kind: pb.SpecialReg, uniform: bool) -> pb.Reg:
            reg = b.reg("int32", name=kind.value, uniform=uniform)
            # CONTRACT: Instr::ReadSpecial { dst, special }
            b.emit(pb.ReadSpecial(dst=reg, special=kind))
            return reg

        if scope == 0:
            return special(pb.SpecialReg.CLUSTER_LINEAR, True)
        if scope == 1:
            return special(pb.SpecialReg.CTA_LINEAR, True)
        if scope == 2:
            return special(pb.SpecialReg.CTA_IN_CLUSTER, True)
        if scope == 9:
            return self.binary(
                "FloorMod", "int32", special(pb.SpecialReg.CTA_IN_CLUSTER, True),
                self.const("int32", 2),
            )
        if scope == 4:
            return special(pb.SpecialReg.WARP_IN_CTA, True)
        if scope == 3:
            return self.binary(
                "FloorDiv", "int32", special(pb.SpecialReg.WARP_IN_CTA, True),
                self.const("int32", wpg),
            )
        if scope == 5:
            return self.binary(
                "FloorMod", "int32", special(pb.SpecialReg.WARP_IN_CTA, True),
                self.const("int32", wpg),
            )
        if scope == 6:
            return special(pb.SpecialReg.LANE, False)
        if scope == 7:
            return special(pb.SpecialReg.THREAD_IN_CTA, False)
        if scope == 8:
            return self.binary(
                "FloorMod", "int32", special(pb.SpecialReg.THREAD_IN_CTA, False),
                self.const("int32", wpg * 32),
            )
        self.unsupported(node, f"scope binding {scope}")
        return None

    def stmt_tirx_AllocBuffer(self, node: Any) -> None:
        var = node.buffer
        ty = var.ty
        scope = str(ty.storage_scope)
        dtype = str(ty.dtype.dtype)
        shape, strides = _static_shape_strides(ty)
        if shape is None:
            self.unsupported(node, f"{scope} buffer with dynamic shape or non-trivial layout")
            return
        if scope == "local":
            count = _numel(shape)
            # CONTRACT: register arrays. Dynamic indices into a promoted array
            # need Instr::{LoadRegIndexed, StoreRegIndexed} (see design doc).
            regs = [self.builder.reg(dtype, name=str(var.name)) for _ in range(count)]
            self.local_arrays[_handle(var)] = _LocalArray(regs=regs, dtype=dtype)
            return
        if scope in ("shared", "shared.dyn"):
            buf = self.builder.buffer(
                pb.BufferDecl(
                    name=str(var.name), space=pb.Space.SHARED, dtype=dtype, shape=shape,
                    strides=strides, byte_size=_numel(shape) * _ITEMSIZE.get(dtype, 0),
                    alignment=max(16, int(ty.data_alignment)),
                )
            )
            self.buffers[_handle(var)] = buf
            return
        self.unsupported(node, f"allocation scope {scope!r}")

    def stmt_tirx_BufferStore(self, node: Any) -> None:
        var = node.buffer
        handle = _handle(var)
        value = self.expr(node.value)
        array = self.local_arrays.get(handle)
        if array is not None:
            slot = self.static_flat_index(var, node.indices)
            if slot is None:
                self.unsupported(node, "dynamic index into a register-promoted local array")
                return
            dst = array.regs[slot]
            self.mark_uniform(dst, self.is_uniform(value))
            self.builder.emit(pb.Mov(dst=dst, src=self.cast_to(value, array.dtype)))
            return
        buf = self.buffers.get(handle)
        if buf is None:
            self.unsupported(node, f"store to unknown buffer {var.name}")
            return
        decl = self.builder.program.buffers[buf]
        offset = self.flat_offset(var, node.indices)
        site = self.site(node, buffer=buf)
        # CONTRACT: Instr::Store { dtype, space, buf, offset, value, site }
        self.builder.emit(
            pb.Store(
                dtype=decl.dtype, space=decl.space, buf=buf, offset=offset,
                value=self.cast_to(value, decl.dtype), site=site,
            )
        )

    def stmt_tirx_Bind(self, node: Any) -> None:
        value = self.expr(node.value)
        dtype = _dtype_of(node.var)
        reg = self.builder.reg(dtype, name=str(node.var.name), uniform=self.is_uniform(value))
        self.builder.emit(pb.Mov(dst=reg, src=self.cast_to(value, dtype)))
        self.vars[_handle(node.var)] = reg

    def stmt_tirx_IfThenElse(self, node: Any) -> None:
        cond = self.expr(node.condition)
        b = self.builder
        site = self.site(node)
        if_pc = b.emit(pb.If(cond=cond, else_pc=-1, end_pc=-1, site=site))
        self.stmt(node.then_case)
        else_pc = -1
        if node.else_case is not None:
            else_pc = b.emit(pb.Else(end_pc=-1))
            self.stmt(node.else_case)
        end_pc = b.emit(pb.EndIf())
        # CONTRACT: If.else_pc == end_pc when there is no Else.
        b.patch(if_pc, pb.If(cond=cond, else_pc=else_pc if else_pc >= 0 else end_pc,
                             end_pc=end_pc, site=site))
        if else_pc >= 0:
            b.patch(else_pc, pb.Else(end_pc=end_pc))

    def stmt_tirx_For(self, node: Any) -> None:
        kind = int(node.kind)
        if kind not in (0, 3):  # SERIAL, UNROLLED
            self.unsupported(node, f"for-loop kind {kind}")
            return
        if node.thread_binding is not None:
            self.unsupported(node, "thread-bound loop")
            return
        b = self.builder
        dtype = _dtype_of(node.loop_var)
        start = self.cast_to(self.expr(node.min), dtype)
        extent = self.cast_to(self.expr(node.extent), dtype)
        step = self.cast_to(self.expr(node.step), dtype) if node.step is not None else self.const(dtype, 1)
        uniform = all(self.is_uniform(v) for v in (start, extent, step))
        var = b.reg(dtype, name=str(node.loop_var.name), uniform=uniform)
        b.emit(pb.Mov(dst=var, src=start))
        stop = self.binary("Add", dtype, start, extent)
        self.vars[_handle(node.loop_var)] = var
        site = self.site(node)
        begin_pc = b.emit(pb.LoopBegin(end_pc=-1, site=site))
        head_pc = b.pc
        cond = b.reg("bool", uniform=uniform)
        b.emit(pb.Compare(op="Lt", dtype=dtype, dst=cond, a=var, b=stop))
        loopif_pc = b.emit(pb.LoopIf(cond=cond, end_pc=-1))
        self.stmt(node.body)
        b.emit(pb.Binary(op="Add", dtype=dtype, dst=var, a=var, b=step))
        end_pc = b.emit(pb.LoopEnd(head_pc=head_pc))
        b.patch(begin_pc, pb.LoopBegin(end_pc=end_pc, site=site))
        b.patch(loopif_pc, pb.LoopIf(cond=cond, end_pc=end_pc))

    def stmt_tirx_While(self, node: Any) -> None:
        b = self.builder
        site = self.site(node)
        begin_pc = b.emit(pb.LoopBegin(end_pc=-1, site=site))
        head_pc = b.pc
        cond = self.expr(node.condition)
        loopif_pc = b.emit(pb.LoopIf(cond=cond, end_pc=-1))
        self.stmt(node.body)
        end_pc = b.emit(pb.LoopEnd(head_pc=head_pc))
        b.patch(begin_pc, pb.LoopBegin(end_pc=end_pc, site=site))
        b.patch(loopif_pc, pb.LoopIf(cond=cond, end_pc=end_pc))

    def stmt_tirx_Evaluate(self, node: Any) -> None:
        value = node.value
        if type_key(value) == "ir.Call":
            self.call(value, statement=True)
            return
        if type_key(value) == "ir.IntImm":
            return  # Evaluate(0): no-op
        self.unsupported(node, f"evaluate of {type_key(value)}")

    # ---------------------------------------------------------- expressions
    def expr(self, node: Any) -> pb.Operand:
        kind = type_key(node)
        dtype = _dtype_of(node)
        if kind in ("ir.IntImm", "ir.FloatImm"):
            value = int(node.value) if kind == "ir.IntImm" else float(node.value)
            try:
                return self.const(dtype, value)
            except ValueError as error:
                self.unsupported(node, str(error))
                return self.const("int32", 0)
        if kind == "ir.Var":
            operand = self.vars.get(_handle(node))
            if operand is None:
                self.unsupported(node, f"unbound or unsupported variable {node.name}")
                return self.const(dtype or "int32", 0)
            return operand
        if kind == "ir.TensorLoad":
            return self.load(node)
        if kind in _BINARY:
            a, b = self.expr(node.a), self.expr(node.b)
            return self.binary(_BINARY[kind], dtype, a, b)
        if kind in _COMPARE:
            a, b = self.expr(node.a), self.expr(node.b)
            operand_dtype = _dtype_of(node.a)
            dst = self.builder.reg("bool", uniform=self.is_uniform(a) and self.is_uniform(b))
            self.builder.emit(pb.Compare(op=_COMPARE[kind], dtype=operand_dtype, dst=dst, a=a, b=b))
            return dst
        if kind in ("prim.Not", "prim.BitwiseNot"):
            a = self.expr(node.a)
            dst = self.builder.reg(dtype, uniform=self.is_uniform(a))
            self.builder.emit(pb.Unary(op="Not" if kind == "prim.Not" else "BitNot", dtype=dtype, dst=dst, a=a))
            return dst
        if kind == "prim.Cast":
            return self.cast_to(self.expr(node.value), dtype)
        if kind == "prim.Select":
            c = self.expr(node.condition)
            x, y = self.expr(node.true_value), self.expr(node.false_value)
            dst = self.builder.reg(dtype, uniform=all(self.is_uniform(v) for v in (c, x, y)))
            self.builder.emit(pb.Select(dtype=dtype, dst=dst, cond=c, a=x, b=y))
            return dst
        if kind == "ir.Call":
            return self.call(node, statement=False)
        self.unsupported(node, "expression kind not lowered")
        return self.const(dtype or "int32", 0)

    def binary(self, op: str, dtype: str, a: pb.Operand, b: pb.Operand) -> pb.Operand:
        dst = self.builder.reg(dtype, uniform=self.is_uniform(a) and self.is_uniform(b))
        # CONTRACT: Instr::Binary { op, dtype, dst, a, b }
        self.builder.emit(pb.Binary(op=op, dtype=dtype, dst=dst, a=a, b=b))
        return dst

    def cast_to(self, value: pb.Operand, dtype: str) -> pb.Operand:
        source = self.operand_dtype(value)
        if not dtype or source == dtype:
            return value
        dst = self.builder.reg(dtype, uniform=self.is_uniform(value))
        self.builder.emit(pb.Cast(src_dtype=source, dst_dtype=dtype, dst=dst, a=value))
        return dst

    def mark_uniform(self, reg: pb.Reg, uniform: bool) -> None:
        regs = self.builder.program.regs
        decl = regs[reg.index]
        # A register written more than once is uniform only if every write is.
        # The first write decides optimistically only for fresh registers.
        regs[reg.index] = pb.RegDecl(dtype=decl.dtype, name=decl.name, uniform=decl.uniform and uniform)

    def load(self, node: Any) -> pb.Operand:
        var = node.source
        handle = _handle(var)
        array = self.local_arrays.get(handle)
        if array is not None:
            slot = self.static_flat_index(var, node.indices)
            if slot is None:
                self.unsupported(node, "dynamic index into a register-promoted local array")
                return self.const(array.dtype, 0)
            return array.regs[slot]
        buf = self.buffers.get(handle)
        if buf is None:
            self.unsupported(node, f"load from unknown buffer {var.name}")
            return self.const(_dtype_of(node) or "int32", 0)
        decl = self.builder.program.buffers[buf]
        offset = self.flat_offset(var, node.indices)
        dst = self.builder.reg(decl.dtype)
        site = self.site(node, buffer=buf)
        # CONTRACT: Instr::Load { dtype, space, dst, buf, offset, site }
        self.builder.emit(
            pb.Load(dtype=decl.dtype, space=decl.space, dst=dst, buf=buf, offset=offset, site=site)
        )
        return dst

    def static_flat_index(self, var: Any, indices: Any) -> int | None:
        _, strides = _static_shape_strides(var.ty)
        if strides is None:
            return None
        total = 0
        for index, stride in zip(indices, strides):
            if type_key(index) != "ir.IntImm":
                return None
            total += int(index.value) * stride
        return total

    def flat_offset(self, var: Any, indices: Any) -> pb.Operand:
        _, strides = _static_shape_strides(var.ty)
        assert strides is not None
        offset: pb.Operand | None = None
        for index, stride in zip(indices, strides):
            if type_key(index) == "prim.Ramp":
                self.unsupported(index, "vector (Ramp) buffer index")
            term = self.expr(index)
            dtype = self.operand_dtype(term)
            if stride != 1:
                term = self.binary("Mul", dtype, term, self.const(dtype, stride))
            offset = term if offset is None else self.binary("Add", dtype, offset, term)
        return offset if offset is not None else self.const("int32", 0)

    def call(self, node: Any, *, statement: bool) -> pb.Operand:
        op = node.op
        name = str(getattr(op, "name", ""))
        dtype = _dtype_of(node)
        handler = builtins.HANDLERS.get(name)
        if handler is None:
            self.unsupported(node, f"builtin {name} (family {builtins.family(name)})")
            return self.const(dtype or "int32", 0)
        return handler(self, node, dtype)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _spans(span: Any) -> tuple[pb.SourceSpan, ...]:
    if span is None:
        return ()
    nested = getattr(span, "spans", None)
    if nested is not None:
        out: list[pb.SourceSpan] = []
        for item in nested:
            out.extend(_spans(item))
        return tuple(out)
    source = getattr(getattr(span, "source_name", None), "name", None)
    if source is None:
        return ()
    return (
        pb.SourceSpan(
            file=str(source), line=int(span.line), column=int(span.column),
            end_line=int(span.end_line), end_column=int(span.end_column),
        ),
    )


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for extent in shape:
        total *= extent
    return total


def _static_shape_strides(ty: Any) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None]:
    """Row-major element strides for a static buffer with a trivial layout."""

    shape: list[int] = []
    for extent in ty.shape:
        if type_key(extent) != "ir.IntImm":
            return None, None
        shape.append(int(extent.value))
    layout = getattr(ty, "layout", None)
    if layout is not None:
        if type_key(layout) != "tirx.TileLayout":
            return None, None
        is_trivial = getattr(layout, "is_trivial", None)
        if is_trivial is not None and not is_trivial():
            return None, None
    explicit = list(getattr(ty, "strides", ()) or ())
    if explicit:
        if any(type_key(s) != "ir.IntImm" for s in explicit):
            return None, None
        return tuple(shape), tuple(int(s.value) for s in explicit)
    strides = [1] * len(shape)
    for axis in range(len(shape) - 2, -1, -1):
        strides[axis] = strides[axis + 1] * shape[axis + 1]
    return tuple(shape), tuple(strides)


def _scalar_bits(dtype: str, value: int | float) -> int:
    if dtype in _INT_BITS:
        bits = _INT_BITS[dtype]
        return int(value) & ((1 << bits) - 1) if bits > 1 else int(bool(value))
    if dtype == "float32":
        return struct.unpack("<I", struct.pack("<f", float(value)))[0]
    if dtype == "float64":
        return struct.unpack("<Q", struct.pack("<d", float(value)))[0]
    if dtype == "float16":
        return struct.unpack("<H", struct.pack("<e", float(value)))[0]
    if dtype == "bfloat16":
        # Round-to-nearest-even from float32 bits.
        f32 = struct.unpack("<I", struct.pack("<f", float(value)))[0]
        return ((f32 + 0x7FFF + ((f32 >> 16) & 1)) >> 16) & 0xFFFF
    raise ValueError(f"constant of dtype {dtype} is not supported by the skeleton")


def lower(func: Any, *, name: str | None = None, strict: bool = True) -> pb.Program:
    """Lower one TIRx ``PrimFunc`` to a provisional ``Program``.

    With ``strict=False`` unsupported constructs stay in the program as
    ``Unsupported`` instructions (run-time ``incomplete`` if reached).
    """

    if name is None:
        attrs = func.attrs
        name = str(attrs["global_symbol"]) if attrs is not None and "global_symbol" in attrs else "kernel"
    program = Lowerer(func, name).lower()
    if strict and program.unsupported:
        raise LoweringUnsupported(program)
    return program


__all__ = ["LoweringUnsupported", "Lowerer", "lower", "type_key"]
