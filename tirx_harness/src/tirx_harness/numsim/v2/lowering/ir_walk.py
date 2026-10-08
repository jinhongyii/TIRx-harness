"""TIRx ``PrimFunc`` -> ``numsim_core::Program`` lowering: the core walker.

Statements and expressions are lowered here; buffers/addresses live in
``memory.py``, calls in ``calls.py`` and ``ptx_lower.py``, parameters and the
host prelude in ``host_prelude.py``. Dispatch is on the FFI ``type_key`` so a
new node kind lands in ``Program.unsupported`` instead of being skipped.

Fail-closed model (doc §B.11): every construct the lowering cannot model is
recorded in ``Program.unsupported`` and an ``Unsupported`` instruction is
emitted in its place; ``lower(..., strict=True)`` then raises
``LoweringUnsupported`` listing all of them.
"""

from __future__ import annotations

from typing import Any

import tvm
from tvm import tirx
from tvm_ffi import structural_visit

from . import dtypes
from . import program_builder as pb
from .calls import CallsMixin
from .dtypes import dtype_of, type_key
from .host_prelude import PreludeMixin
from .memory import MemoryMixin, MemRef, _Unsupported, escaped_locals, handle


class LoweringUnsupported(Exception):
    """The PrimFunc uses constructs the lowering does not model (fail closed)."""

    def __init__(self, program: pb.Program):
        self.program = program
        self.reasons = list(program.unsupported)
        head = "; ".join(self.reasons[:8])
        more = f" (+{len(self.reasons) - 8} more)" if len(self.reasons) > 8 else ""
        super().__init__(f"{program.name}: unsupported TIRx: {head}{more}")


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

# ScopeBinding (tvm/tirx/exec_scope.h) -> (parent, child).
_SCOPE_BINDINGS = {
    0: ("kernel", "cluster"), 1: ("kernel", "cta"), 2: ("cluster", "cta"),
    3: ("cta", "warpgroup"), 4: ("cta", "warp"), 5: ("warpgroup", "warp"),
    6: ("warp", "thread"), 7: ("cta", "thread"), 8: ("warpgroup", "thread"),
    9: ("cluster", "cta_pair"),
}

# AttrStmt keys without numerical semantics (recorded where useful).
_IGNORED_ATTRS = frozenset(
    {
        "tirx.device_entry", "tirx.launch_bounds_min_blocks_per_sm",
        "tirx.required_block_size", "tirx.max_registers", "tirx.dyn_smem_bytes",
        "tirx.pool_max_bytes",
    }
)

_THREAD_TAGS = {"threadIdx": "Tid", "blockIdx": "CtaId", "clusterCtaIdx": "ClusterCtaId", "clusterIdx": "ClusterId"}

_ELECT_OPS = frozenset({"tirx.cuda.elect_sync", "tirx.ptx.elect_sync"})

WARPS_PER_WARPGROUP = 4
U64 = pb.Ty("U64")


class Lowerer(MemoryMixin, CallsMixin, PreludeMixin):
    def __init__(self, func: Any, name: str):
        self.func = func
        self.builder = pb.ProgramBuilder(name)
        self.vars: dict[int, pb.Operand] = {}
        self.refs: dict[int, Any] = {}
        self.scalar_slots: dict[int, int] = {}
        self.host_binds: dict[int, Any] = {}
        self.layout_exprs: dict[int, Any] = {}
        self.dyn_pools: set[int] = set()
        self.dyn_smem_bytes: int | None = None
        self.min_blocks_per_sm: int | None = None
        self.pred_args: dict[int, pb.Reg] = {}
        self.pending_preds: list[tuple[list[pb.Instr], list[int], pb.Reg, pb.Reg, bool]] = []
        self.elect_buffers: set[int] = set()
        self.escaped: set[int] = set()

    # ------------------------------------------------------------------ util
    def unsupported(self, node: Any, reason: str) -> None:
        site = self.site(node) if node is not None else self.builder.site(pb.SiteInfo(kind="", spans=()))
        kind = type_key(node) if node is not None else "?"
        self.builder.program.unsupported.append(f"site#{site} {kind}: {reason}")
        self.builder.emit("Unsupported", site=site, reason=self.builder.string(reason))

    def site(self, node: Any, op_name: str | None = None, buffer: int | None = None) -> int:
        spans = _spans(getattr(node, "span", None))
        key = (handle(node), op_name, buffer) if hasattr(node, "__chandle__") else None
        buffer_name = self.builder.program.buffers[buffer].name if buffer is not None else None
        return self.builder.site(
            pb.SiteInfo(kind=type_key(node), spans=spans, op_name=op_name or "",
                        dtype=dtype_of(node) or None, buffer=buffer_name),
            key=key,
        )

    def ty(self, dtype: str | pb.Ty, node: Any = None) -> pb.Ty:
        if isinstance(dtype, pb.Ty):
            return dtype
        try:
            return pb.Ty.from_tvm(dtype)
        except pb.UnrepresentableType as error:
            raise _Unsupported(node, str(error)) from error

    def is_uniform(self, operand: pb.Operand) -> bool:
        if isinstance(operand, pb.Const):
            return True
        return self.builder.reg_uniform(operand)

    def operand_ty(self, operand: pb.Operand) -> pb.Ty:
        if isinstance(operand, pb.Const):
            return self.builder.const_ty(operand)
        return self.builder.reg_ty(operand)

    def const(self, dtype: str | pb.Ty, value: int | float) -> pb.Const:
        if isinstance(dtype, pb.Ty):
            if dtype.elem in ("U64", "S64", "U32", "S32", "U16", "S16", "U8", "S8", "Pred", "B128"):
                bits = int(value) & ((1 << dtype.bits) - 1)
                return self.builder.const(dtype, bits)
            raise ValueError(f"float constant of Ty {dtype} needs a TVM dtype")
        if dtype == "handle":
            return self.builder.const(U64, int(value) & ((1 << 64) - 1))
        return self.builder.const(self.ty(dtype), dtypes.encode(dtype, value))

    def const_int(self, operand: pb.Const) -> int:
        return self.builder.program.consts[operand.index][1]

    # ---------------------------------------------------------------- entry
    def lower(self) -> pb.Program:
        program = self.builder.program
        attrs = self.func.attrs
        if attrs is not None and "tirx.cuda_arch" in attrs:
            program.arch = str(attrs["tirx.cuda_arch"])
        try:
            self.bind_params()
            statements = self.split_prelude(self.func.body)
        except _Unsupported as error:
            self.unsupported(error.node if error.node is not None else self.func.body, error.reason)
            return self.finish()
        self.escaped = escaped_locals(self.func.body)
        self.collect_topology(statements)
        self.read_params()
        for statement in statements:
            self.stmt(statement)
        return self.finish()

    def finish(self) -> pb.Program:
        static_smem = self.finish_shared()
        program = self.builder.finish()
        # wait_until predicate sub-programs live after the main body.
        for code, sites, arg, result, reads_memory in self.pending_preds:
            start = len(program.code)
            program.code.extend(code)
            program.code_sites.extend(sites)
            program.preds.append(pb.PredProgram(arg=arg, start=start, end=len(program.code), result=result,
                                                reads_memory=reads_memory))
        if program.topology is None:
            program.topology = pb.Launch(grid=(pb.DimExpr.const(1),) * 3)
        program.topology.static_smem_bytes = static_smem
        program.topology.dyn_smem_bytes = pb.DimExpr.const(0)
        program.topology.min_blocks_per_sm = self.min_blocks_per_sm
        return program

    # ------------------------------------------------------------- topology
    def collect_topology(self, statements: list[Any]) -> None:
        extents: dict[tuple[str, str], pb.DimExpr] = {}
        launch_threads: dict[str, pb.DimExpr] = {}

        def visit(node: Any, visitor: Any) -> None:
            definition = getattr(node, "def")
            binding = _SCOPE_BINDINGS.get(int(definition.scope))
            if binding is not None and definition.extents is not None:
                total = pb.DimExpr.const(1)
                for extent in definition.extents:
                    try:
                        total = _mul(total, self.dim_expr(extent))
                    except _Unsupported:
                        return
                extents.setdefault(binding, total)

        def on_attr(node: Any, visitor: Any) -> None:
            key = str(node.attr_key)
            if key == "thread_extent":
                tag = str(node.node.thread_tag)
                try:
                    launch_threads.setdefault(tag, self.dim_expr(node.value))
                except _Unsupported:
                    pass
            if key == "tirx.dyn_smem_bytes" and type_key(node.value) == "ir.IntImm":
                self.dyn_smem_bytes = int(node.value.value)
            elif key == "tirx.launch_bounds_min_blocks_per_sm" and type_key(node.value) == "ir.IntImm":
                self.min_blocks_per_sm = int(node.value.value)
            visitor.default_visit(node)

        for statement in statements:
            structural_visit(statement, [(tirx.ScopeIdDefStmt, visit), (tirx.AttrStmt, on_attr)])

        if launch_threads:
            self.thread_extent_topology(launch_threads)
            return

        def static(key: tuple[str, str]) -> int | None:
            value = extents.get(key)
            return value.value if value is not None and value.is_const else None

        threads = static(("cta", "thread"))
        warps = static(("cta", "warp"))
        if warps is None and static(("cta", "warpgroup")) is not None:
            warps = static(("cta", "warpgroup")) * (static(("warpgroup", "warp")) or WARPS_PER_WARPGROUP)
        if warps is None and static(("warpgroup", "warp")) is not None:
            warps = static(("warpgroup", "warp"))
        if threads is None and warps is not None:
            threads = warps * 32
        if threads is None and static(("warpgroup", "thread")) is not None:
            threads = static(("warpgroup", "thread"))
        if threads is None and any(k[1] in ("warpgroup",) or k[0] == "warpgroup" for k in extents):
            threads = WARPS_PER_WARPGROUP * 32       # implicit single warpgroup (legacy rule)
        if threads is None:
            threads = 32                             # no CTA-level extent: one warp (legacy rule)
        lane = static(("warp", "thread"))
        if lane is not None and lane != 32:
            self.builder.program.unsupported.append(f"topology: warp>thread extent {lane} != 32")
        ctas_per_cluster = static(("cluster", "cta")) or (2 if ("cluster", "cta_pair") in extents else 1)
        grid = extents.get(("kernel", "cta"))
        if grid is None and ("kernel", "cluster") in extents:
            grid = _mul(extents[("kernel", "cluster")], pb.DimExpr.const(ctas_per_cluster))
        if grid is None:
            grid = pb.DimExpr.const(ctas_per_cluster)
        if threads is None:
            self.builder.program.unsupported.append("topology: threads per CTA are deferred or unknown")
            return
        self.builder.program.topology = pb.Launch(
            grid=(grid, pb.DimExpr.const(1), pb.DimExpr.const(1)), cluster=(ctas_per_cluster, 1, 1),
            block=(threads, 1, 1),
        )

    def thread_extent_topology(self, tags: dict[str, pb.DimExpr]) -> None:
        """Topology of a launch_thread-style kernel (TVM dispatch output, legacy kernels)."""
        def dim(prefix: str, axis: str) -> pb.DimExpr:
            return tags.get(f"{prefix}.{axis}", pb.DimExpr.const(1))

        block = [dim("threadIdx", a) for a in "xyz"]
        cluster = [dim("clusterCtaIdx", a) for a in "xyz"]
        if not all(d.is_const for d in block + cluster):
            self.builder.program.unsupported.append("topology: dynamic block or cluster extent")
            return
        # CUDA semantics: gridDim counts CTAs; the cluster shape divides it.
        grid = tuple(dim("blockIdx", a) for a in "xyz")
        self.builder.program.topology = pb.Launch(
            grid=grid, cluster=tuple(d.value for d in cluster), block=tuple(d.value for d in block),
        )

    # ----------------------------------------------------------- statements
    def stmt(self, node: Any) -> None:
        kind = type_key(node)
        method = getattr(self, "stmt_" + kind.replace(".", "_"), None)
        if method is None:
            self.unsupported(node, "statement kind not lowered")
            return
        try:
            method(node)
        except _Unsupported as error:
            self.unsupported(error.node if error.node is not None else node, error.reason)
        except pb.UnrepresentableType as error:
            self.unsupported(node, str(error))

    def stmt_tirx_SeqStmt(self, node: Any) -> None:
        for child in node.seq:
            self.stmt(child)

    def stmt_tirx_AttrStmt(self, node: Any) -> None:
        key = str(node.attr_key)
        if key == "thread_extent":
            iter_var = node.node
            tag = str(iter_var.thread_tag)
            base, _, axis = tag.partition(".")
            sreg = _THREAD_TAGS.get(base)
            if sreg is None or axis not in ("x", "y", "z"):
                raise _Unsupported(node, f"thread_extent tag {tag!r}")
            reg = self.builder.reg(pb.Ty("S32"), name=tag, uniform=base != "threadIdx")
            self.builder.emit("ReadSpecial", dst=reg, sreg={sreg: axis.upper()})
            self.vars[handle(iter_var.var)] = self.cast_to(reg, dtype_of(iter_var.var))
            self.stmt(node.body)
            return
        if key not in _IGNORED_ATTRS:
            raise _Unsupported(node, f"attribute {key!r}")
        self.stmt(node.body)

    def stmt_tirx_ScopeIdDefStmt(self, node: Any) -> None:
        definition = getattr(node, "def")
        scope = int(definition.scope)
        variables = list(definition.def_ids)
        flat = self.scope_flat_id(node, scope)
        if len(variables) == 1:
            self.vars[handle(variables[0])] = self.cast_to(flat, dtype_of(variables[0]))
            return
        extents = definition.extents
        if extents is None or any(type_key(e) != "ir.IntImm" for e in extents):
            raise _Unsupported(node, "multi-coordinate scope ids need static extents")
        # The first coordinate is fastest-varying (legacy emit/stmt.rs:1200).
        divisor = 1
        for var, extent in zip(variables, extents):
            value = int(extent.value)
            quotient = self.binary("FloorDiv", "int32", flat, self.const("int32", divisor))
            coordinate = self.binary("FloorMod", "int32", quotient, self.const("int32", value))
            self.vars[handle(var)] = self.cast_to(coordinate, dtype_of(var))
            divisor *= value

    def scope_flat_id(self, node: Any, scope: int) -> pb.Operand:
        def special(sreg: str, uniform: bool) -> pb.Reg:
            reg = self.builder.reg(pb.Ty("S32"), name=sreg, uniform=uniform)
            self.builder.emit("ReadSpecial", dst=reg, sreg=sreg)
            return reg

        wpg = self.const("int32", WARPS_PER_WARPGROUP)
        if scope == 0:
            return special("ClusterLinear", True)
        if scope == 1:
            return special("CtaLinear", True)
        if scope == 2:
            return special("ClusterCtaRank", True)
        if scope == 9:
            return self.binary("FloorMod", "int32", special("ClusterCtaRank", True), self.const("int32", 2))
        if scope == 4:
            return special("WarpInCta", True)
        if scope == 3:
            return special("WarpgroupInCta", True)
        if scope == 5:
            return self.binary("FloorMod", "int32", special("WarpInCta", True), wpg)
        if scope == 6:
            return special("LaneId", False)
        if scope == 7:
            return special("ThreadInCta", False)
        if scope == 8:
            return self.binary("FloorMod", "int32", special("ThreadInCta", False),
                               self.const("int32", WARPS_PER_WARPGROUP * 32))
        raise _Unsupported(node, f"scope binding {scope}")

    def stmt_tirx_AllocBuffer(self, node: Any) -> None:
        self.declare_alloc(node)

    def stmt_tirx_DeclBuffer(self, node: Any) -> None:
        self.declare_view(node)

    def stmt_tirx_BufferStore(self, node: Any) -> None:
        value = self.expr(node.value)
        self.store(node, node.buffer, node.indices, value)

    def stmt_tirx_Bind(self, node: Any) -> None:
        value = self.expr(node.value)
        ty = self.ty(dtype_of(node.var), node)
        reg = self.builder.reg(ty, name=str(node.var.name), uniform=self.is_uniform(value))
        self.builder.emit("Mov", dst=reg, src=self.cast_to(value, ty))
        self.vars[handle(node.var)] = reg

    def stmt_tirx_IfThenElse(self, node: Any) -> None:
        cond = self.cast_to(self.expr(node.condition), pb.Ty("Pred"))
        elect = self.is_elect(node.condition)
        b = self.builder
        site = self.site(node)
        if_pc = b.emit("If", site=site, cond=cond, else_pc=0, end_pc=0, elect=elect)
        self.stmt(node.then_case)
        else_pc = -1
        if node.else_case is not None:
            else_pc = b.emit("Else", end_pc=0)
            self.stmt(node.else_case)
        end_pc = b.emit("EndIf")
        b.patch(if_pc, "If", cond=cond, else_pc=else_pc if else_pc >= 0 else end_pc, end_pc=end_pc, elect=elect)
        if else_pc >= 0:
            b.patch(else_pc, "Else", end_pc=end_pc)

    def stmt_tirx_For(self, node: Any) -> None:
        kind = int(node.kind)
        if kind not in (0, 3):  # SERIAL, UNROLLED (kept as loops: decision 2)
            raise _Unsupported(node, f"for-loop kind {kind}")
        if node.thread_binding is not None:
            raise _Unsupported(node, "thread-bound loop")
        b = self.builder
        ty = self.ty(dtype_of(node.loop_var), node)
        start = self.cast_to(self.expr(node.min), ty)
        extent = self.cast_to(self.expr(node.extent), ty)
        step = self.cast_to(self.expr(node.step), ty) if node.step is not None else self.const(ty, 1)
        uniform = all(self.is_uniform(v) for v in (start, extent, step))
        var = b.reg(ty, name=str(node.loop_var.name), uniform=uniform)
        b.emit("Mov", dst=var, src=start)
        stop = self.binary("Add", ty, start, extent)
        self.vars[handle(node.loop_var)] = var
        site = self.site(node)
        begin_pc = b.emit("LoopBegin", site=site, end_pc=0)
        head_pc = b.pc
        cond = b.reg(pb.Ty("Pred"), uniform=uniform)
        b.emit("Compare", op="Lt", ty=ty, dst=cond, a=var, b=stop)
        loopif_pc = b.emit("LoopIf", cond=cond, end_pc=0)
        self.stmt(node.body)
        b.emit("Binary", op="Add", ty=ty, dst=var, a=var, b=step)
        end_pc = b.emit("LoopEnd", head_pc=head_pc)
        b.patch(begin_pc, "LoopBegin", end_pc=end_pc)
        b.patch(loopif_pc, "LoopIf", cond=cond, end_pc=end_pc)

    def stmt_tirx_While(self, node: Any) -> None:
        b = self.builder
        site = self.site(node)
        begin_pc = b.emit("LoopBegin", site=site, end_pc=0)
        head_pc = b.pc
        try:
            cond = self.cast_to(self.expr(node.condition), pb.Ty("Pred"))
        except _Unsupported:
            del b.code[begin_pc:]
            del b.code_sites[begin_pc:]
            raise
        loopif_pc = b.emit("LoopIf", cond=cond, end_pc=0)
        self.stmt(node.body)
        end_pc = b.emit("LoopEnd", head_pc=head_pc)
        b.patch(begin_pc, "LoopBegin", end_pc=end_pc)
        b.patch(loopif_pc, "LoopIf", cond=cond, end_pc=end_pc)

    def stmt_tirx_Break(self, node: Any) -> None:
        self.builder.emit("Break")

    def stmt_tirx_Continue(self, node: Any) -> None:
        self.builder.emit("Continue")

    def stmt_tirx_Return(self, node: Any) -> None:
        self.builder.emit("Exit")

    def stmt_tirx_AssertStmt(self, node: Any) -> None:
        cond = self.cast_to(self.expr(node.condition), pb.Ty("Pred"))
        parts = [str(getattr(p, "value", "")) for p in (node.message_parts or ())]
        self.builder.emit("Assert", site=self.site(node), cond=cond, msg=self.builder.string("".join(parts)))

    def stmt_tirx_Evaluate(self, node: Any) -> None:
        value = node.value
        kind = type_key(value)
        if kind == "ir.Call":
            self.call(value, statement=True)
            return
        if kind == "ir.IntImm":
            return  # Evaluate(0): no-op
        raise _Unsupported(node, f"evaluate of {kind}")

    def is_elect(self, cond: Any) -> bool:
        found = False

        def on_call(node: Any, visitor: Any) -> None:
            nonlocal found
            if str(getattr(node.op, "name", "")) in _ELECT_OPS:
                found = True
            visitor.default_visit(node)

        def on_load(node: Any, visitor: Any) -> None:
            nonlocal found
            if handle(node.source) in self.elect_buffers:
                found = True
            visitor.default_visit(node)

        structural_visit(cond, [(tvm.ir.Call, on_call), (tvm.ir.TensorLoad, on_load)])
        return found

    # ---------------------------------------------------------- expressions
    def expr(self, node: Any) -> pb.Operand:
        kind = type_key(node)
        dtype = dtype_of(node)
        if kind in ("ir.IntImm", "ir.FloatImm"):
            value = int(node.value) if kind == "ir.IntImm" else float(node.value)
            try:
                return self.const(dtype, value)
            except (ValueError, pb.UnrepresentableType) as error:
                raise _Unsupported(node, str(error)) from error
        if kind == "ir.Var":
            return self.var(node)
        if kind == "ir.TensorLoad":
            return self.load(node)
        if kind in _BINARY:
            ty = self.ty(dtype, node)
            a, b = self.cast_to(self.expr(node.a), ty), self.cast_to(self.expr(node.b), ty)
            return self.binary(_BINARY[kind], ty, a, b)
        if kind in _COMPARE:
            operand_ty = self.ty(dtype_of(node.a), node)
            a, b = self.cast_to(self.expr(node.a), operand_ty), self.cast_to(self.expr(node.b), operand_ty)
            dst = self.builder.reg(self.ty(dtype, node), uniform=self.is_uniform(a) and self.is_uniform(b))
            self.builder.emit("Compare", op=_COMPARE[kind], ty=operand_ty, dst=dst, a=a, b=b)
            return dst
        if kind in ("prim.Not", "prim.BitwiseNot"):
            ty = self.ty(dtype, node)
            a = self.cast_to(self.expr(node.a), ty)
            dst = self.builder.reg(ty, uniform=self.is_uniform(a))
            self.builder.emit("Unary", op="Not" if kind == "prim.Not" else "BitNot", ty=ty, dst=dst, a=a)
            return dst
        if kind == "prim.Cast":
            return self.cast_to(self.expr(node.value), dtype)
        if kind == "prim.Select":
            ty = self.ty(dtype, node)
            c = self.cast_to(self.expr(node.condition), pb.Ty("Pred"))
            x, y = self.cast_to(self.expr(node.true_value), ty), self.cast_to(self.expr(node.false_value), ty)
            dst = self.builder.reg(ty, uniform=all(self.is_uniform(v) for v in (c, x, y)))
            self.builder.emit("Select", ty=ty, dst=dst, cond=c, a=x, b=y)
            return dst
        if kind == "prim.Broadcast":
            ty = self.ty(dtype, node)
            a = self.expr(node.value)
            return self.pack([a] * ty.lanes, ty)
        if kind == "prim.Shuffle":
            if any(type_key(i) != "ir.IntImm" for i in node.indices):
                raise _Unsupported(node, "Shuffle with dynamic indices")
            lanes: list[pb.Operand] = []
            for vector in node.vectors:
                value = self.expr(vector)
                count = self.operand_ty(value).lanes
                lanes.extend([value] if count == 1 else self.unpack(value, count))
            if any(not 0 <= int(i.value) < len(lanes) for i in node.indices):
                raise _Unsupported(node, "Shuffle index outside the concatenated lanes")
            picked = [lanes[int(i.value)] for i in node.indices]
            ty = self.ty(dtype, node)
            return picked[0] if len(picked) == 1 else self.pack(picked, ty)
        if kind == "prim.Let":
            self.vars[handle(node.var)] = self.cast_to(self.expr(node.value), dtype_of(node.var))
            return self.expr(node.body)
        if kind == "ir.Call":
            result = self.call(node, statement=False)
            if result is None:
                raise _Unsupported(node, "void call used as a value")
            return result
        raise _Unsupported(node, "expression kind not lowered")

    def var(self, node: Any) -> pb.Operand:
        key = handle(node)
        operand = self.vars.get(key)
        if operand is not None:
            return operand
        bound = self.host_binds.get(key)
        if bound is not None:
            return self.expr(bound)
        ref = self.refs.get(key)
        if isinstance(ref, MemRef):
            # A TensorMap (or buffer) variable used as a value is its address.
            return self.addr_of(ref.buf, self.const("int32", 0))
        raise _Unsupported(node, f"unbound variable {node.name}")

    def binary(self, op: str, ty: str | pb.Ty, a: pb.Operand, b: pb.Operand) -> pb.Operand:
        ty = self.ty(ty)
        dst = self.builder.reg(ty, uniform=self.is_uniform(a) and self.is_uniform(b))
        self.builder.emit("Binary", op=op, ty=ty, dst=dst, a=a, b=b)
        return dst

    def cast_to(self, value: pb.Operand, target: str | pb.Ty) -> pb.Operand:
        if not target:
            return value
        ty = self.ty(target)
        source = self.operand_ty(value)
        if source == ty:
            return value
        if isinstance(value, pb.Const) and _is_int(source) and _is_int(ty):
            raw = self.const_int(value)
            if source.elem.startswith("S") and raw >> (source.bits - 1):
                raw -= 1 << source.bits
            if ty.elem == "Pred":
                raw = int(raw != 0)
            return self.const(ty, raw)
        dst = self.builder.reg(ty, uniform=self.is_uniform(value))
        self.builder.emit("Cast", dst=dst, src=value, rnd="Default", sat=False, **{"from": source, "to": ty})
        return dst

    def reinterpret(self, value: pb.Operand, ty: pb.Ty) -> pb.Operand:
        """Same bits, new type (``Mov`` between types; contract: reinterpret uses Mov)."""
        source = self.operand_ty(value)
        if source == ty:
            return value
        if source.bits != ty.bits:
            return self.cast_to(value, ty)
        if isinstance(value, pb.Const):
            return self.builder.const(ty, self.const_int(value))
        dst = self.builder.reg(ty, uniform=self.is_uniform(value))
        self.builder.emit("Mov", dst=dst, src=value)
        return dst

    def convert(self, value: pb.Operand, ty: pb.Ty) -> pb.Operand:
        """Carrier -> PTX type: reinterpret when widths match, else convert."""
        source = self.operand_ty(value)
        if source.bits == ty.bits:
            return self.reinterpret(value, ty)
        return self.cast_to(value, ty)

    def assign(self, dst: pb.Reg, value: pb.Operand) -> None:
        self.builder.emit("Mov", dst=dst, src=self.convert(value, self.builder.reg_ty(dst)))

    def pack(self, values: list[pb.Operand], ty: pb.Ty) -> pb.Reg:
        dst = self.builder.reg(ty)
        op = self.builder.op(pb.OpKey("numsim.pack", (f"ty={ty.elem}x{ty.lanes}",)))
        self.builder.emit("Ptx", op=op, dsts=[dst], srcs=list(values), pred=None, keep_dst=False)
        return dst

    def unpack(self, value: pb.Operand, lanes: int) -> list[pb.Reg]:
        ty = self.operand_ty(value)
        elem = pb.Ty(ty.elem, ty.lanes // lanes)
        dsts = [self.builder.reg(elem) for _ in range(lanes)]
        op = self.builder.op(pb.OpKey("numsim.unpack", (f"ty={ty.elem}x{ty.lanes}",)))
        self.builder.emit("Ptx", op=op, dsts=dsts, srcs=[value], pred=None, keep_dst=False)
        return dsts

    def address_in(self, value: pb.Operand, space: str) -> pb.Operand:
        """Coerce an address value to ``space``'s encoding (generic pointers -> cvta)."""
        bits = self.operand_ty(value).bits
        if space in ("Shared", "SharedCluster") and bits == 64:
            dst = self.builder.reg(pb.Ty("U32"))
            self.builder.emit("Cvta", dst=dst, src=value, space=space, to_generic=False)
            return dst
        if space in ("Generic", "Global", "Local", "Param", "Const") and bits != 64:
            return self.cast_to(value, U64)
        return value

    def buffer_access(self, addr_node: Any, space_token: str) -> tuple[int, pb.Operand] | None:
        """Buffer form for a PTX memory operand: ``address_of(buf[i])`` (maybe via cvta)."""
        node = addr_node
        if type_key(node) == "ir.Call" and str(getattr(node.op, "name", "")) == "tirx.cuda.cvta_generic_to_shared":
            node = node.args[0]
            if space_token not in ("shared", "shared::cta"):
                return None
        target = self.buffer_target(node)
        if target is None:
            return None
        buf, _ = target
        decl = self.builder.program.buffers[buf]
        root = decl
        while root.view_of is not None:
            root = self.builder.program.buffers[root.view_of]
        expected = {"": None, "global": "Global", "shared": "Shared", "shared::cta": "Shared",
                    "local": "Local"}.get(space_token, "?")
        if expected == "?" or (expected is not None and expected != root.space):
            return None
        return target


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _is_int(ty: pb.Ty) -> bool:
    return ty.lanes == 1 and ty.elem in ("Pred", "U8", "S8", "U16", "S16", "U32", "S32", "U64", "S64")


def _mul(a: pb.DimExpr, b: pb.DimExpr) -> pb.DimExpr:
    if a.is_const and b.is_const:
        return pb.DimExpr.const(a.value * b.value)
    if a.is_const and a.value == 1:
        return b
    if b.is_const and b.value == 1:
        return a
    return pb.DimExpr("Mul", args=(a, b))


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
        pb.SourceSpan(file=str(source), line=max(0, int(span.line)), column=max(0, int(span.column)),
                      end_line=max(0, int(span.end_line)), end_column=max(0, int(span.end_column))),
    )


def lower(func: Any, *, name: str | None = None, strict: bool = True) -> pb.Program:
    """Lower one TIRx ``PrimFunc`` to a ``numsim_core::Program``.

    With ``strict=False`` unsupported constructs stay in the program as
    ``Unsupported`` instructions (run-time ``incomplete`` if reached).
    """

    if name is None:
        attrs = func.attrs
        name = str(attrs["global_symbol"]) if attrs is not None and "global_symbol" in attrs else "kernel"
    func = dispatch_tile_primitives(func)
    program = Lowerer(func, name).lower()
    if strict and program.unsupported:
        raise LoweringUnsupported(program)
    return program


def dispatch_tile_primitives(func: Any) -> Any:
    """Decision 6: lower ``tirx.tile.*`` through TVM's own dispatch to PTX-level IR.

    The simulated semantics of a tile op are then, by construction, those of
    the code TVM emits for the GPU. Functions without tile ops are returned
    unchanged; a dispatch failure leaves the tile op in place (fail closed).
    """
    found = False

    def on_tile(node: Any, visitor: Any) -> None:
        nonlocal found
        found = True

    structural_visit(func.body, [(tirx.TilePrimitiveCall, on_tile)])
    if not found:
        return func
    attrs = func.attrs
    arch = str(attrs["tirx.cuda_arch"]) if attrs is not None and "tirx.cuda_arch" in attrs else "sm_100a"
    try:
        with tvm.target.Target({"kind": "cuda", "arch": arch}):
            module = tirx.transform.TilePrimitiveDispatch()(tvm.IRModule({"main": func}))
    except Exception:  # noqa: BLE001 - dispatch rejection: keep the tile op, fail closed later
        return func
    return module["main"]


def lower_module(funcs: Any, *, strict: bool = True) -> pb.Module:
    """Lower one PrimFunc or a sequence (kernels launched in order)."""
    items = list(funcs) if isinstance(funcs, (list, tuple)) else [funcs]
    return pb.Module(kernels=[lower(f, strict=strict) for f in items])


__all__ = ["LoweringUnsupported", "Lowerer", "lower", "lower_module"]
