"""Call lowering: dispatch, CUDA helpers, pointer plumbing and ``wait_until``.

``tirx.ptx.*`` table ops go to ``ptx_lower``. Pure CUDA helpers become
``Instr::Ptx`` with ``OpKey{name, mods}`` (oplib implements them);
effectful helpers lower to the dedicated variants (``MbarWait``, ``Barrier``,
``Shfl``, ``Atom``, ``LoadAddr``...); annotations (printf/iket/nanosleep)
become ``Nop``. Destination operands passed as ``address_of(lvalue)`` bind
to the lvalue's register (or a temporary stored back).
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING, Any

from . import builtins, dtypes, ptx_decode
from . import program_builder as pb
from .dtypes import type_key
from .memory import AccessPtr, RegArray, _Unsupported, access_facts
from .ptx_lower import lower_ptx

if TYPE_CHECKING:
    from .ir_walk import Lowerer


def _string(node: Any) -> str | None:
    return str(node.value) if type_key(node) == "ir.StringImm" else None


def _int32(value: int) -> Any:
    from tvm import tirx

    return tirx.IntImm("int32", value)


def _op_name(call: Any) -> str:
    return str(getattr(call.op, "name", ""))


_SREGS = {
    "laneid": "LaneId",
    "warpid": "WarpInCta",
    "smid": "SmId",
    "nsmid": "NSmId",
    "gridid": "GridId",
    "clock": "Clock",
    "clock64": "Clock64",
    "globaltimer": "GlobalTimer",
    "lanemask_eq": "LaneMaskEq",
    "lanemask_lt": "LaneMaskLt",
    "lanemask_le": "LaneMaskLe",
    "lanemask_gt": "LaneMaskGt",
    "lanemask_ge": "LaneMaskGe",
    "cluster_ctarank": "ClusterCtaRank",
    "cluster_nctarank": "ClusterNCtaRank",
    "dynamic_smem_size": "DynamicSmemSize",
    "total_smem_size": "TotalSmemSize",
    "nwarpid": "NWarpId",
}
_SREG_AXIS = {
    "tid": "Tid",
    "ntid": "NTid",
    "ctaid": "CtaId",
    "nctaid": "NCtaId",
    "clusterid": "ClusterId",
    "nclusterid": "NClusterId",
    "cluster_ctaid": "ClusterCtaId",
    "cluster_nctaid": "ClusterNCtaId",
}

_ASM_REPLACE = re.compile(
    r'"tensormap\.replace\.tile\.(global_address|global_dim|global_stride)\.global\.b1024\.b(32|64) '
    r'\[%0\], (?:(\d+), )?%1;"'
)
_ASM_RELEASE = re.compile(r'"fence\.proxy\.tensormap::generic\.release\.(cta|cluster|gpu|sys);')
_ASM_ACQUIRE = re.compile(
    r'"fence\.proxy\.tensormap::generic\.acquire\.(cta|cluster|gpu|sys) \[%0\], 128;'
)
_SCOPES = {"cta": "Cta", "cluster": "Cluster", "gpu": "Gpu", "sys": "Sys"}

_PURE_STRUCTURAL = frozenset({"tirx.reinterpret", "prim.if_then_else", "prim.likely"})


def _cuda_ldg_rejection(node: Any, dtype: str) -> str | None:
    """CUDA's ``__ldg`` overload set (legacy ``is_cuda_ldg_dtype``/``cuda_ldg_parts``).

    Scalars: integers, float16, bfloat16, float32, float64. Vectors: integer,
    float32 and float64 lanes; float16/bfloat16 x2 or x8; float8 only as a
    64- or 128-bit packet. bool and packed bool have no overload (boolx2/x4
    have no packed-bool ABI). An ``address_of(buf[i])`` pointer must point at
    the loaded dtype.
    """
    result = dtypes.dtype_of(node)
    if result != dtype:
        return f"cuda.ldg dtype attribute {dtype!r} does not match the result dtype {result!r}"
    base, _, lanes_text = dtype.rpartition("x")
    if not (base and lanes_text.isdigit()):
        base, lanes = dtype, 1
    else:
        lanes = int(lanes_text)
    integer = base.startswith(("int", "uint")) and base[-1].isdigit()
    if lanes == 1:
        ok = integer or base in ("float16", "bfloat16", "float32", "float64")
    elif base.startswith("float8_"):
        ok = 8 * lanes in (64, 128)
    elif base in ("float16", "bfloat16"):
        ok = lanes in (2, 8)
    else:
        ok = integer or base in ("float32", "float64")
    if not ok:
        return f"cuda.ldg has no __ldg overload for {dtype!r}"
    pointer = node.args[0]
    if (
        type_key(pointer) == "ir.Call"
        and _op_name(pointer) == "tirx.address_of"
        and len(pointer.args) == 1
    ):
        pointee = dtypes.dtype_of(pointer.args[0])
        if pointee and pointee != dtype:
            return f"cuda.ldg pointer to {pointee!r} does not match the loaded dtype {dtype!r}"
    return None


class CallsMixin:
    """Mixed into ``Lowerer``."""

    def call(self: Lowerer, node: Any, *, statement: bool) -> pb.Operand | None:
        name = _op_name(node)
        if name.startswith("tirx.ptx.") and ptx_decode.is_table_op(name):
            lower_ptx(self, node)
            return None
        unary = builtins.UNARY_OPS.get(name)
        if unary is not None:
            ty = self.ty(dtypes.dtype_of(node), node)
            value = self.cast_to(self.expr(node.args[0]), ty)
            dst = self.builder.reg(ty, uniform=self.is_uniform(value))
            self.builder.emit("Unary", op=unary, ty=ty, dst=dst, a=value)
            return dst
        structural = getattr(self, "call_" + name.replace(".", "_"), None)
        helper = builtins.HELPERS.get(name)
        if structural is not None and (helper is None or helper.kind == "special"):
            result: pb.Operand | None = structural(node)
            return result
        if helper is not None and helper.kind == "pure":
            return self.pure_helper(node, name, helper.roles)
        if name.startswith("tirx.tile."):
            raise _Unsupported(
                node, f"tile op {name} was lowered by neither TVM dispatch nor a v2 tile form"
            )
        raise _Unsupported(node, f"builtin {name}")

    # -- pure helpers -> Ptx ------------------------------------------------
    def pure_helper(
        self: Lowerer,
        node: Any,
        name: str,
        roles: str,
        args: list[Any] | None = None,
        extra_mods: tuple[str, ...] = (),
    ) -> pb.Operand | None:
        dsts: list[pb.Reg] = []
        srcs: list[pb.Operand] = []
        mods: list[str] = []
        write_backs = []
        result = None
        result_dtype = dtypes.dtype_of(node)
        if result_dtype:
            result = self.builder.reg(self.ty(result_dtype, node))
            dsts.append(result)
        for position, arg in enumerate(node.args if args is None else args):
            role = builtins.role(roles, position)
            text = _string(arg)
            if role == "?":
                raise _Unsupported(node, f"{name}: unexpected argument {position}")
            if role == "s" or (role == "a" and text is not None):
                if text is None:
                    raise _Unsupported(
                        node, f"{name}: argument {position} must be a string literal"
                    )
                mods.append(f"arg{position}={text}")
                continue
            if role in "ox" and not self.is_lvalue_ref(arg):
                # A pointer value: read/write through memory around the op.
                element = getattr(getattr(arg, "ty", None), "element_type", None)
                elem_dtype = str(getattr(element, "dtype", "")) if element is not None else ""
                if not elem_dtype or elem_dtype == "void":
                    raise _Unsupported(
                        node, f"{name}: out-parameter {position} has no element type"
                    )
                ty = self.ty(elem_dtype, node)
                addr = self.as_address(self.expr(arg))
                reg = self.builder.reg(ty)
                site = self.site(node, op_name=name)
                if role == "x":
                    self.builder.emit(
                        "LoadAddr",
                        site=site,
                        ty=ty,
                        dst=reg,
                        addr=addr,
                        space="Generic",
                        sem="Weak",
                        scope="Gpu",
                        mods=pb.mem_mods(),
                    )
                    srcs.append(reg)
                dsts.append(reg)

                def store_through(
                    reg: pb.Reg = reg, addr: pb.Operand = addr, ty: pb.Ty = ty, site: int = site
                ) -> None:
                    self.builder.emit(
                        "StoreAddr",
                        site=site,
                        ty=ty,
                        addr=addr,
                        space="Generic",
                        value=reg,
                        sem="Weak",
                        scope="Gpu",
                        mods=pb.mem_mods(),
                    )

                write_backs.append(store_through)
                continue
            if role in "ox":
                reg, write_back = self.lvalue_target(arg)
                if role == "x":
                    current = self.expr(arg.args[0])
                    if write_back is not None:
                        self.builder.emit("Mov", dst=reg, src=current)
                    srcs.append(reg)
                dsts.append(reg)
                if write_back is not None:
                    write_backs.append(write_back)
                continue
            srcs.append(self.expr(arg))
        op = self.builder.op(pb.OpKey(name, tuple(mods) + extra_mods))
        self.builder.emit(
            "Ptx",
            site=self.site(node, op_name=name),
            op=op,
            dsts=dsts,
            srcs=srcs,
            pred=None,
            keep_dst=False,
        )
        for write_back in write_backs:
            write_back()
        return result

    def is_lvalue_ref(self: Lowerer, arg: Any) -> bool:
        return (
            type_key(arg) == "ir.Call"
            and _op_name(arg) == "tirx.address_of"
            and type_key(arg.args[0]) == "ir.TensorLoad"
        )

    def result_from(self: Lowerer, node: Any, value: pb.Operand) -> pb.Operand:
        dtype = dtypes.dtype_of(node)
        return self.cast_to(value, dtype) if dtype else value

    def full_mask(self: Lowerer) -> pb.Const:
        return self.const("uint32", 0xFFFF_FFFF)

    # -- structural tirx.* ops ---------------------------------------------
    def call_tirx_address_of(self: Lowerer, node: Any) -> pb.Operand:
        return self.address_of(node)

    def call_tirx_buffer_data(self: Lowerer, node: Any) -> pb.Operand:
        return self.buffer_data(node)

    def call_tirx_type_annotation(self: Lowerer, node: Any) -> pb.Operand:
        return self.const("int32", 0)

    def call_tirx_reinterpret(self: Lowerer, node: Any) -> pb.Operand:
        dtype = dtypes.dtype_of(node)
        reason = _reinterpret_rejection(dtypes.dtype_of(node.args[0]) or "", dtype or "")
        if reason is not None:
            raise _Unsupported(node, f"tirx.reinterpret: {reason}")
        value = self.expr(node.args[0])
        if dtype == "handle":
            return self.as_address(value)
        return self.reinterpret(value, self.ty(dtype, node))

    def call_tirx_ptr_byte_offset(self: Lowerer, node: Any) -> pb.Operand:
        base = self.as_address(self.expr(node.args[0]))
        offset = self.reinterpret(self.cast_to(self.expr(node.args[1]), "int64"), pb.Ty("U64"))
        return self.binary("Add", pb.Ty("U64"), base, offset)

    def call_tirx_handle_add_byte_offset(self: Lowerer, node: Any) -> pb.Operand:
        return self.call_tirx_ptr_byte_offset(node)

    def call_tirx_tvm_access_ptr(self: Lowerer, node: Any) -> pb.Operand:
        """``tvm_access_ptr(type, data, offset, extent, rw_mask)`` = ``data + offset`` elements.

        ``extent`` and ``rw_mask`` are access-pattern hints with no semantics in the
        engine (the Arena checks the actual accesses through the pointer).
        """
        type_node, data, offset = node.args[0], node.args[1], node.args[2]
        elem = dtypes.dtype_of(type_node)
        if not elem or elem in ("void", "handle"):
            raise _Unsupported(node, "tvm_access_ptr without an element type")
        data_value = self.expr(data)
        base = self.as_address(data_value)
        result = self.binary("Add", pb.Ty("U64"), base, self.element_bytes(elem, self.expr(offset)))
        # The access contract (extent, rw_mask) is a TIR fact the lowering checks
        # statically where the pointer is used (finish(), DeclBuffer views).
        mask = (
            int(node.args[4].value)
            if len(node.args) > 4 and type_key(node.args[4]) == "ir.IntImm"
            else 3
        )
        extent = (
            int(node.args[3].value)
            if len(node.args) > 3 and type_key(node.args[3]) == "ir.IntImm"
            else None
        )
        parent = access_facts(self, data_value)
        if parent is not None and mask & ~parent.mask:
            added = "write" if mask & ~parent.mask & 2 else "read"
            raise _Unsupported(
                node,
                f"tvm_access_ptr cannot add {added} access to a pointer created with "
                f"access mask {parent.mask}",
            )
        if isinstance(result, pb.Reg):
            self.access_ptrs[result.index] = AccessPtr(mask=mask, elem=elem, extent=extent)
        return result

    def call_tirx_isnullptr(self: Lowerer, node: Any) -> pb.Operand:
        value = self.as_address(self.expr(node.args[0]))
        dst = self.builder.reg(pb.Ty("Pred"), uniform=self.is_uniform(value))
        self.builder.emit(
            "Compare", op="Eq", ty=pb.Ty("U64"), dst=dst, a=value, b=self.const(pb.Ty("U64"), 0)
        )
        return dst

    def call_prim_likely(self: Lowerer, node: Any) -> pb.Operand:
        return self.expr(node.args[0])

    def call_tirx_fma(self: Lowerer, node: Any) -> pb.Operand:
        ty = self.ty(dtypes.dtype_of(node), node)
        a, b, c = (self.cast_to(self.expr(x), ty) for x in node.args)
        dst = self.builder.reg(ty, uniform=all(self.is_uniform(v) for v in (a, b, c)))
        self.builder.emit("Ternary", op="Fma", ty=ty, dst=dst, a=a, b=b, c=c)
        return dst

    def call_prim_if_then_else(self: Lowerer, node: Any) -> pb.Operand:
        cond_node, a_node, b_node = node.args
        ty = self.ty(dtypes.dtype_of(node), node)
        cond = self.cast_to(self.expr(cond_node), pb.Ty("Pred"))
        if self.is_pure(a_node) and self.is_pure(b_node):
            a, b = self.cast_to(self.expr(a_node), ty), self.cast_to(self.expr(b_node), ty)
            dst = self.builder.reg(ty, uniform=all(self.is_uniform(v) for v in (cond, a, b)))
            self.builder.emit("Select", ty=ty, dst=dst, cond=cond, a=a, b=b)
            return dst
        # Only the taken arm executes (it can guard an out-of-bounds load).
        return self.choose_lazily(
            node,
            cond,
            ty,
            lambda: self.cast_to(self.expr(a_node), ty),
            lambda: self.cast_to(self.expr(b_node), ty),
        )

    def call_tirx_break_loop(self: Lowerer, node: Any) -> None:
        self.builder.emit("Break")
        return None

    def call_tirx_continue_loop(self: Lowerer, node: Any) -> None:
        self.builder.emit("Continue")
        return None

    # -- special registers --------------------------------------------------
    def read_special(self: Lowerer, node: Any, sreg: Any, ty: pb.Ty) -> pb.Operand:
        dst = self.builder.reg(ty)
        self.builder.emit("ReadSpecial", dst=dst, sreg=sreg)
        return self.result_from(node, dst)

    def call_tirx_cuda_thread_rank(self: Lowerer, node: Any) -> pb.Operand:
        return self.read_special(node, "ThreadInCta", pb.Ty("S32"))

    def call_tirx_cuda_clock64(self: Lowerer, node: Any) -> pb.Operand:
        return self.read_special(node, "Clock64", pb.Ty("U64"))

    def call_tirx_cuda___activemask(self: Lowerer, node: Any) -> pb.Operand:
        return self.read_special(node, "ActiveMask", pb.Ty("U32"))

    call_tirx_tvm_warp_activemask = call_tirx_cuda___activemask

    def call_tirx_cuda_mov_sreg(self: Lowerer, node: Any) -> pb.Operand:
        name = _string(node.args[1])
        if name is None:
            raise _Unsupported(node, "mov_sreg register name must be a literal")
        name = name.removeprefix("%")  # legacy accepted the PTX spelling `%laneid` too (W11-6)
        bits = int(node.args[0].value)
        ty = pb.Ty("U64") if bits == 64 else pb.Ty("U32")
        if name in _SREGS:
            return self.read_special(node, _SREGS[name], ty)
        if name in ("clock_hi", "globaltimer_hi", "globaltimer_lo"):
            clock = self.builder.reg(pb.Ty("U64"))
            self.builder.emit(
                "ReadSpecial",
                dst=clock,
                sreg="Clock64" if name.startswith("clock") else "GlobalTimer",
            )
            value: pb.Operand = clock
            if name.endswith("_hi"):
                value = self.binary("Shr", pb.Ty("U64"), clock, self.const("uint64", 32))
            return self.cast_to(value, dtypes.dtype_of(node) or "uint32")
        base, _, axis = name.partition(".")
        if base in _SREG_AXIS and axis in ("x", "y", "z"):
            return self.read_special(node, {_SREG_AXIS[base]: axis.upper()}, ty)
        raise _Unsupported(node, f"special register %{name}")

    # -- annotations ----------------------------------------------------------
    def nop(self: Lowerer, node: Any) -> pb.Operand | None:
        self.builder.emit("Nop", site=self.site(node, op_name=_op_name(node)))
        dtype = dtypes.dtype_of(node)
        if dtype:
            return self.const(dtype, 0)
        return None

    call_tirx_cuda_printf = nop
    # Profiler flush: writes only the host-side trace, numerically a no-op (legacy).
    call_tirx_timer_finalize_cuda = nop
    call_tirx_cuda_iket_mark = nop
    call_tirx_cuda_iket_range_start = nop
    call_tirx_cuda_iket_range_end = nop
    call_tirx_cuda_iket_range_push = nop
    call_tirx_cuda_iket_range_pop = nop
    call_tirx_cuda_iket_sentinel_token = nop
    call_tirx_cuda_iket_official_event = nop
    call_tirx_cuda_nano_sleep = nop

    def call_tirx_cuda_trap_when_assert_failed(self: Lowerer, node: Any) -> None:
        cond = self.cast_to(self.expr(node.args[0]), pb.Ty("Pred"))
        self.builder.emit(
            "Assert",
            site=self.site(node),
            cond=cond,
            msg=self.builder.string("trap_when_assert_failed"),
        )
        return None

    # -- synchronization ------------------------------------------------------
    def mbar_wait(self: Lowerer, node: Any, scope: str) -> None:
        addr = self.expr(node.args[0])
        space = "Shared" if self.operand_ty(addr).bits == 32 else "Generic"
        phase = self.cast_to(self.expr(node.args[1]), pb.Ty("U32"))
        self.builder.emit(
            "MbarWait",
            site=self.site(node, op_name=_op_name(node)),
            mbar=addr,
            space=space,
            phase=pb.phase_parity(phase),
            sem="Acquire",
            scope=scope,
        )

    def call_tirx_cuda_mbarrier_wait(self: Lowerer, node: Any) -> None:
        self.mbar_wait(node, "Cta")

    def call_tirx_cuda_mbarrier_wait_acquire_cluster(self: Lowerer, node: Any) -> None:
        self.mbar_wait(node, "Cluster")

    def barrier(
        self: Lowerer, node: Any, kind: Any, ident: pb.Operand, count: pb.Operand | None
    ) -> None:
        self.builder.emit(
            "Barrier",
            site=self.site(node, op_name=_op_name(node)),
            kind=kind,
            id=ident,
            count=count,
            aligned=True,
        )

    def call_tirx_cuda_cta_sync(self: Lowerer, node: Any) -> None:
        self.barrier(node, "Sync", self.const("uint32", 0), None)

    def call_tirx_tvm_storage_sync(self: Lowerer, node: Any) -> None:
        scope = _string(node.args[0]) if node.args else "shared"
        if scope not in ("shared", "shared.dyn"):
            raise _Unsupported(node, f"tvm_storage_sync({scope!r})")
        self.barrier(node, "Sync", self.const("uint32", 0), None)

    def call_tirx_cuda_warpgroup_sync(self: Lowerer, node: Any) -> None:
        ident = self.cast_to(self.expr(node.args[0]), pb.Ty("U32"))
        self.barrier(node, "Sync", ident, self.const("uint32", 128))

    def call_tirx_cuda_warp_sync(self: Lowerer, node: Any) -> None:
        mask = (
            self.cast_to(self.expr(node.args[0]), pb.Ty("U32")) if node.args else self.full_mask()
        )
        self.builder.emit("WarpSync", site=self.site(node, op_name=_op_name(node)), membermask=mask)

    def call_tirx_cuda_cluster_sync(self: Lowerer, node: Any) -> None:
        site = self.site(node, op_name=_op_name(node))
        self.builder.emit("ClusterArrive", site=site, sem="Release", aligned=True)
        self.builder.emit("ClusterWait", site=site, acquire=True, aligned=True)

    def call_tirx_cuda_grid_sync(self: Lowerer, node: Any) -> None:
        self.builder.emit("GridSync", site=self.site(node, op_name=_op_name(node)))

    def syncthreads_red(self: Lowerer, node: Any, op: str) -> pb.Operand:
        pred = self.cast_to(self.expr(node.args[0]), pb.Ty("Pred"))
        dst = self.builder.reg(pb.Ty("Pred"))
        self.barrier(
            node,
            {"Red": {"op": op, "pred": pb.opnd(pred), "dst": dst}},
            self.const("uint32", 0),
            None,
        )
        return self.result_from(node, dst)

    def call_tirx_cuda_syncthreads_and(self: Lowerer, node: Any) -> pb.Operand:
        return self.syncthreads_red(node, "And")

    def call_tirx_cuda_syncthreads_or(self: Lowerer, node: Any) -> pb.Operand:
        return self.syncthreads_red(node, "Or")

    def call_tirx_cuda_thread_fence(self: Lowerer, node: Any) -> None:
        self.builder.emit(
            "Fence",
            site=self.site(node, op_name=_op_name(node)),
            kind="Thread",
            sem="Sc",
            scope="Gpu",
        )

    # -- warp collectives -----------------------------------------------------
    def call_tirx_cuda_elect_sync(self: Lowerer, node: Any) -> pb.Operand:
        mask = (
            self.cast_to(self.expr(node.args[0]), pb.Ty("U32")) if node.args else self.full_mask()
        )
        pred = self.builder.reg(pb.Ty("Pred"))
        self.builder.emit(
            "Elect",
            site=self.site(node, op_name=_op_name(node)),
            dst_pred=pred,
            dst_lane=None,
            membermask=mask,
        )
        return self.result_from(node, pred)

    def shfl(
        self: Lowerer, node: Any, mode: str, mask: Any, value: Any, lane: Any, width: Any
    ) -> pb.Operand:
        ty = self.ty(dtypes.dtype_of(node), node)
        src = self.cast_to(self.expr(value), ty)
        lane_op = self.cast_to(self.expr(lane), pb.Ty("U32"))
        width_op = self.cast_to(self.expr(width), pb.Ty("U32"))
        self.check_shfl_width(node, width_op)
        # CUDA: c = ((32 - width) << 8) | (up ? 0 : 0x1f)
        segment = self.binary(
            "Shl",
            pb.Ty("U32"),
            self.binary("Sub", pb.Ty("U32"), self.const("uint32", 32), width_op),
            self.const("uint32", 8),
        )
        clamp = (
            segment
            if mode == "Up"
            else self.binary("Or", pb.Ty("U32"), segment, self.const("uint32", 0x1F))
        )
        dst = self.builder.reg(ty)
        self.builder.emit(
            "Shfl",
            site=self.site(node, op_name=_op_name(node)),
            mode=mode,
            ty=ty,
            dst=dst,
            dst_pred=None,
            src=src,
            lane=lane_op,
            clamp=clamp,
            membermask=self.cast_to(self.expr(mask), pb.Ty("U32")),
        )
        return dst

    def check_shfl_width(self: Lowerer, node: Any, width: pb.Operand) -> None:
        """CUDA ``__shfl*_sync`` width must be a power of two in [1, 32] (legacy runtime check)."""
        message = "invalid warp shuffle selector/width (width must be a power of two in [1, 32])"
        if isinstance(width, pb.Const):
            value = self.builder.program.const_value(width)
            if not (1 <= value <= 32 and value & (value - 1) == 0):
                raise _Unsupported(node, message)
            return
        # ok <=> ((w & (w - 1)) | ((w - 1) >> 5)) == 0  (w - 1 wraps for w = 0)
        u32 = pb.Ty("U32")
        minus_one = self.binary("Sub", u32, width, self.const("uint32", 1))
        bad = self.binary(
            "Or",
            u32,
            self.binary("And", u32, width, minus_one),
            self.binary("Shr", u32, minus_one, self.const("uint32", 5)),
        )
        ok = self.builder.reg(pb.Ty("Pred"))
        self.builder.emit("Compare", op="Eq", ty=u32, dst=ok, a=bad, b=self.const("uint32", 0))
        self.builder.emit(
            "Assert",
            site=self.site(node, op_name=_op_name(node)),
            cond=ok,
            msg=self.builder.string(message),
        )

    def call_tirx_cuda___shfl_sync(self: Lowerer, node: Any) -> pb.Operand:
        return self.shfl(node, "Idx", *node.args)

    def call_tirx_cuda___shfl_up_sync(self: Lowerer, node: Any) -> pb.Operand:
        return self.shfl(node, "Up", *node.args)

    def call_tirx_cuda___shfl_down_sync(self: Lowerer, node: Any) -> pb.Operand:
        return self.shfl(node, "Down", *node.args)

    def call_tirx_cuda___shfl_xor_sync(self: Lowerer, node: Any) -> pb.Operand:
        return self.shfl(node, "Bfly", *node.args)

    def call_tirx_tvm_warp_shuffle(self: Lowerer, node: Any) -> pb.Operand:
        mask, value, lane, width, _warp_size = node.args
        return self.shfl(node, "Idx", mask, value, lane, width)

    def call_tirx_tvm_warp_shuffle_up(self: Lowerer, node: Any) -> pb.Operand:
        mask, value, lane, width, _warp_size = node.args
        return self.shfl(node, "Up", mask, value, lane, width)

    def call_tirx_tvm_warp_shuffle_down(self: Lowerer, node: Any) -> pb.Operand:
        mask, value, lane, width, _warp_size = node.args
        return self.shfl(node, "Down", mask, value, lane, width)

    def call_tirx_tvm_warp_shuffle_xor(self: Lowerer, node: Any) -> pb.Operand:
        mask, value, lane, width, _warp_size = node.args
        return self.shfl(node, "Bfly", mask, value, lane, width)

    def vote(self: Lowerer, node: Any, mode: str, dst_ty: pb.Ty) -> pb.Operand:
        mask, pred = node.args
        dst = self.builder.reg(dst_ty)
        self.builder.emit(
            "Vote",
            site=self.site(node, op_name=_op_name(node)),
            mode=mode,
            dst=dst,
            pred=self.cast_to(self.expr(pred), pb.Ty("Pred")),
            membermask=self.cast_to(self.expr(mask), pb.Ty("U32")),
        )
        return self.result_from(node, dst)

    def call_tirx_cuda_ballot_sync(self: Lowerer, node: Any) -> pb.Operand:
        return self.vote(node, "Ballot", pb.Ty("U32"))

    def call_tirx_cuda_any_sync(self: Lowerer, node: Any) -> pb.Operand:
        return self.vote(node, "Any", pb.Ty("Pred"))

    def redux(self: Lowerer, node: Any, op: str) -> pb.Operand:
        mask, value = node.args
        dst = self.builder.reg(pb.Ty("U32"))
        self.builder.emit(
            "Redux",
            site=self.site(node, op_name=_op_name(node)),
            op=op,
            ty=pb.Ty("U32"),
            dst=dst,
            src=self.cast_to(self.expr(value), pb.Ty("U32")),
            membermask=self.cast_to(self.expr(mask), pb.Ty("U32")),
        )
        return self.result_from(node, dst)

    def call_tirx_cuda_reduce_add_sync_u32(self: Lowerer, node: Any) -> pb.Operand:
        return self.redux(node, "Add")

    def call_tirx_cuda_reduce_min_sync_u32(self: Lowerer, node: Any) -> pb.Operand:
        return self.redux(node, "Min")

    # -- reductions (TVM's templated butterfly helpers, expanded) -------------
    def butterfly(self: Lowerer, node: Any, value: pb.Operand, op: str, width: int) -> pb.Operand:
        """``tvm_builtin_cuda_warp_reduce_<op>_<width>``: log2(width) shfl.bfly steps."""
        ty = self.operand_ty(value)
        step = {"sum": "Add", "max": "Max", "min": "Min"}[op]
        site = self.site(node, op_name=_op_name(node))
        mask = width >> 1
        while mask > 0:
            shuffled = self.builder.reg(ty)
            self.builder.emit(
                "Shfl",
                site=site,
                mode="Bfly",
                ty=ty,
                dst=shuffled,
                dst_pred=None,
                src=value,
                lane=self.const("uint32", mask),
                clamp=self.const("uint32", 0x1F),
                membermask=self.full_mask(),
            )
            value = self.binary(step, ty, value, shuffled)
            mask >>= 1
        return value

    def reduce_args(
        self: Lowerer, node: Any, op_arg: Any, count_arg: Any, what: str
    ) -> tuple[str, int]:
        op = _string(op_arg)
        if op not in ("sum", "max", "min"):
            raise _Unsupported(node, f"{what} op {op!r}")
        if type_key(count_arg) != "ir.IntImm":
            raise _Unsupported(node, f"{what} width must be a constant")
        count = int(count_arg.value)
        if count < 1 or count > 32 or count & (count - 1):
            raise _Unsupported(node, f"{what} width {count} is not a power of two in [1, 32]")
        return op, count

    def call_tirx_cuda_warp_reduce(self: Lowerer, node: Any) -> pb.Operand:
        value_arg, op_arg = node.args[0], node.args[1]
        width_arg = node.args[2] if len(node.args) > 2 else None
        op, width = self.reduce_args(
            node, op_arg, width_arg if width_arg is not None else _int32(32), "warp_reduce"
        )
        ty = self.ty(dtypes.dtype_of(node), node)
        return self.butterfly(node, self.cast_to(self.expr(value_arg), ty), op, width)

    def call_tirx_cuda_cta_reduce(self: Lowerer, node: Any) -> pb.Operand:
        """``tvm_builtin_cuda_cta_reduce_<op>_<nw>(val, scratch)`` expanded (TVM cpp/builtins.py)."""
        value_arg, op_arg, warps_arg, scratch_arg = node.args
        op, num_warps = self.reduce_args(node, op_arg, warps_arg, "cta_reduce")
        dtype = dtypes.dtype_of(node)
        ty = self.ty(dtype, node)
        b = self.builder
        site = self.site(node, op_name=_op_name(node))
        scratch = self.as_address(self.expr(scratch_arg))
        value = self.butterfly(node, self.cast_to(self.expr(value_arg), ty), op, 32)
        tid = b.reg(pb.Ty("S32"))
        b.emit("ReadSpecial", dst=tid, sreg="ThreadInCta")
        warp = self.binary("FloorDiv", "int32", tid, self.const("int32", 32))
        lane = self.binary("FloorMod", "int32", tid, self.const("int32", 32))

        def slot(index: pb.Operand) -> pb.Operand:
            return self.binary("Add", pb.Ty("U64"), scratch, self.element_bytes(dtype, index))

        def store(index: pb.Operand, data: pb.Operand) -> None:
            b.emit(
                "StoreAddr",
                site=site,
                ty=ty,
                addr=slot(index),
                space="Generic",
                value=data,
                sem="Weak",
                scope="Gpu",
                mods=pb.mem_mods(),
            )

        def load(index: pb.Operand, dst: pb.Reg) -> None:
            b.emit(
                "LoadAddr",
                site=site,
                ty=ty,
                dst=dst,
                addr=slot(index),
                space="Generic",
                sem="Weak",
                scope="Gpu",
                mods=pb.mem_mods(),
            )

        def guarded(cond: pb.Operand, then: Any, otherwise: Any = None) -> None:
            if_pc = b.emit("If", site=site, cond=cond, else_pc=0, end_pc=0, elect=False)
            then()
            else_pc = -1
            if otherwise is not None:
                else_pc = b.emit("Else", end_pc=0)
                otherwise()
            end = b.emit("EndIf")
            b.patch(
                if_pc,
                "If",
                cond=cond,
                else_pc=else_pc if else_pc >= 0 else end,
                end_pc=end,
                elect=False,
            )
            if else_pc >= 0:
                b.patch(else_pc, "Else", end_pc=end)

        def is_zero(x: pb.Operand) -> pb.Reg:
            dst = b.reg(pb.Ty("Pred"))
            b.emit("Compare", op="Eq", ty=pb.Ty("S32"), dst=dst, a=x, b=self.const("int32", 0))
            return dst

        guarded(is_zero(lane), lambda: store(warp, value))
        b.emit(
            "Barrier", site=site, kind="Sync", id=self.const("uint32", 0), count=None, aligned=True
        )
        partial = b.reg(ty)

        def leader_warp() -> None:
            in_range = b.reg(pb.Ty("Pred"))
            b.emit(
                "Compare",
                op="Lt",
                ty=pb.Ty("S32"),
                dst=in_range,
                a=lane,
                b=self.const("int32", num_warps),
            )
            guarded(
                in_range,
                lambda: load(lane, partial),
                lambda: b.emit("Mov", dst=partial, src=self.reduce_identity(dtype, op)),
            )
            reduced = self.butterfly(node, partial, op, 32)
            guarded(is_zero(lane), lambda: store(self.const("int32", 0), reduced))

        guarded(is_zero(warp), leader_warp)
        b.emit(
            "Barrier", site=site, kind="Sync", id=self.const("uint32", 0), count=None, aligned=True
        )
        result = b.reg(ty)
        load(self.const("int32", 0), result)
        return result

    def reduce_identity(self: Lowerer, dtype: str, op: str) -> pb.Const:
        if op == "sum":
            return self.const(dtype, 0)
        if dtypes.is_float(dtype):
            return self.const(dtype, float("-inf") if op == "max" else float("inf"))
        # CUDA converts +-INFINITY to the integer type's extreme.
        width = dtypes.bits(dtype)
        if dtypes.is_signed(dtype):
            return self.const(dtype, -(1 << (width - 1)) if op == "max" else (1 << (width - 1)) - 1)
        return self.const(dtype, 0 if op == "max" else (1 << width) - 1)

    # -- memory helpers -------------------------------------------------------
    def call_tirx_cuda_ldg(self: Lowerer, node: Any) -> pb.Operand:
        addr = self.as_address(self.expr(node.args[0]))
        text = _string(node.args[1]) if len(node.args) > 1 else None
        if text is None or len(node.args) != 2:
            raise _Unsupported(node, "cuda.ldg vector/destination form")
        rejection = _cuda_ldg_rejection(node, text)
        if rejection is not None:
            raise _Unsupported(node, rejection)
        ty = self.ty(text, node)
        dst = self.builder.reg(ty)
        self.builder.emit(
            "LoadAddr",
            site=self.site(node, op_name=_op_name(node)),
            ty=ty,
            dst=dst,
            addr=addr,
            space="Global",
            sem="Weak",
            scope="Gpu",
            mods=pb.mem_mods(nc=True),
        )
        self.builder.program.requirements.readonly_proxy = True
        return self.result_from(node, dst)

    def call_tirx_s_tir_ldg32(self: Lowerer, node: Any) -> None:
        """``s_tir.ldg32(reg.data, guard, src[i], k)``: TVM's CUDA codegen emits
        ``setp.ne.b32 p, guard, 0; @!p mov.b32 reg[k], 0; @p ld.global.nc.f32
        reg[k], [&src[i]]``: the guarded lanes load, the others write 0.0."""
        from tvm import tirx
        from tvm.script import tirx as T

        if len(node.args) != 4:
            raise _Unsupported(node, "s_tir.ldg32 requires (reg, guard, source load, reg offset)")
        reg, guard, load, offset = node.args
        if type_key(reg) != "ir.Call" or str(getattr(reg.op, "name", "")) != "tirx.buffer_data":
            raise _Unsupported(node, "s_tir.ldg32 destination must be a local buffer's data")
        var = reg.args[0]
        if str(var.ty.storage_scope) != "local" or str(var.ty.dtype.dtype) != "float32":
            raise _Unsupported(node, "s_tir.ldg32 destination must be a float32 local buffer")
        if (
            type_key(load) != "ir.TensorLoad"
            or str(load.source.ty.dtype.dtype) != "float32"
            or str(load.source.ty.storage_scope) != "global"
        ):
            raise _Unsupported(node, "s_tir.ldg32 source must be a float32 global buffer element")
        value = T.cuda.ldg(tirx.address_of(load), "float32")
        cond = tirx.NE(
            tirx.Cast("int32", guard) if str(guard.ty.dtype) != "int32" else guard,
            tirx.IntImm("int32", 0),
        )
        self.stmt(
            tirx.IfThenElse(
                cond,
                tirx.BufferStore(var, value, [offset], span=node.span),
                tirx.BufferStore(var, tirx.FloatImm("float32", 0.0), [offset], span=node.span),
                span=node.span,
            )
        )
        return None

    def atomic(self: Lowerer, node: Any, op: str) -> pb.Operand:
        args = list(node.args)
        addr = self.as_address(self.expr(args[0]))
        ty = self.ty(dtypes.dtype_of(node), node)
        cmp = self.cast_to(self.expr(args[1]), ty) if op == "Cas" else None
        value = self.cast_to(self.expr(args[-1]), ty)
        dst = self.builder.reg(ty)
        self.builder.emit(
            "Atom",
            site=self.site(node, op_name=_op_name(node)),
            op=op,
            ty=ty,
            dst=dst,
            addr=addr,
            space="Generic",
            value=value,
            cmp=cmp,
            sem="Relaxed",
            scope="Gpu",
            ftz=False,
        )
        return dst

    def call_tirx_cuda_atomic_add(self: Lowerer, node: Any) -> pb.Operand:
        return self.atomic(node, "Add")

    def call_tirx_cuda_atomic_cas(self: Lowerer, node: Any) -> pb.Operand:
        return self.atomic(node, "Cas")

    def call_tirx_cuda_cvta_generic_to_shared(self: Lowerer, node: Any) -> pb.Operand:
        src = self.as_address(self.expr(node.args[0]))
        dst = self.builder.reg(self.ty(dtypes.dtype_of(node) or "uint32", node))
        self.builder.emit("Cvta", dst=dst, src=src, space="Shared", to_generic=False)
        return dst

    call_tirx_cuda_smem_addr_from_uint64 = call_tirx_cuda_cvta_generic_to_shared

    def convert_through_memory(
        self: Lowerer, node: Any, src_ty: pb.Ty, dst_ty: pb.Ty, src_first: bool = False
    ) -> None:
        # TVM's signatures differ: `cuda_float22half2(void* dst, void* src)` but
        # `cuda_{half8tofloat8,float8tohalf8}(void* src_addr, void* dst_addr)`
        # (legacy cuda_helper.rs DPS_HELPERS: destination index 0 vs 1).
        first, second = (self.as_address(self.expr(a)) for a in node.args)
        src_ptr, dst_ptr = (first, second) if src_first else (second, first)
        site = self.site(node, op_name=_op_name(node))
        loaded = self.builder.reg(src_ty)
        self.builder.emit(
            "LoadAddr",
            site=site,
            ty=src_ty,
            dst=loaded,
            addr=src_ptr,
            space="Generic",
            sem="Weak",
            scope="Gpu",
            mods=pb.mem_mods(),
        )
        converted = self.builder.reg(dst_ty)
        op = self.builder.op(pb.OpKey(_op_name(node) + ".value"))
        self.builder.emit(
            "Ptx", site=site, op=op, dsts=[converted], srcs=[loaded], pred=None, keep_dst=False
        )
        self.builder.emit(
            "StoreAddr",
            site=site,
            ty=dst_ty,
            addr=dst_ptr,
            space="Generic",
            value=converted,
            sem="Weak",
            scope="Gpu",
            mods=pb.mem_mods(),
        )

    # -- legacy warp-MMA fragment helpers (tirx.mma_fill / tirx.mma_store) ---
    _MMA_ACCUMULATORS = ("float16", "float32", "float64", "int32")

    def call_tirx_mma_fill(self: Lowerer, node: Any) -> None:
        """``mma_fill(local_size, ptr, offset)``: zero ``local_size`` accumulator
        elements at element ``offset`` of ``ptr`` (legacy emit/matrix.rs)."""
        dtype = dtypes.dtype_of(node) or ""
        if dtype not in self._MMA_ACCUMULATORS or len(node.args) != 3:
            raise _Unsupported(
                node, f"tirx.mma_fill accumulator dtype must be one of {self._MMA_ACCUMULATORS}"
            )
        local_size = node.args[0]
        if type_key(local_size) != "ir.IntImm" or int(local_size.value) <= 0:
            raise _Unsupported(node, "tirx.mma_fill.local_size must be a positive integer constant")
        ty = self.ty(dtype, node)
        site = self.site(node, op_name=_op_name(node))
        base = self.binary(
            "Add",
            pb.Ty("U64"),
            self.as_address(self.expr(node.args[1])),
            self.element_bytes(dtype, self.expr(node.args[2])),
        )
        zero = self.const(dtype, 0)
        for slot in range(int(local_size.value)):
            addr = self.binary(
                "Add", pb.Ty("U64"), base, self.const("uint64", slot * dtypes.bits(dtype) // 8)
            )
            self.builder.emit(
                "StoreAddr",
                site=site,
                ty=ty,
                addr=addr,
                space="Generic",
                value=zero,
                sem="Weak",
                scope="Gpu",
                mods=pb.mem_mods(),
            )

    call_tirx_mma_fill_legacy = call_tirx_mma_fill

    def call_tirx_mma_store(self: Lowerer, node: Any) -> None:
        """``mma_store(m, n, dst, src, src_offset, dst_stride)`` for the 16x16
        accumulator fragment: lane-owned element ``local_id`` (0..8) goes to
        ``row = 8*((id%4)/2) + lane/4``, ``col = 8*(id/4) + 2*(lane%4) + id%2``
        of ``dst`` (row stride ``dst_stride``), as legacy emit/matrix.rs."""
        dtype = dtypes.dtype_of(node) or ""
        if dtype not in self._MMA_ACCUMULATORS or len(node.args) != 6:
            raise _Unsupported(
                node, f"tirx.mma_store accumulator dtype must be one of {self._MMA_ACCUMULATORS}"
            )
        m, n = node.args[0], node.args[1]
        if any(type_key(x) != "ir.IntImm" or int(x.value) != 16 for x in (m, n)):
            raise _Unsupported(node, "tirx.mma_store supports the 16x16 accumulator fragment only")
        ty = self.ty(dtype, node)
        i64, u64 = pb.Ty("S64"), pb.Ty("U64")
        site = self.site(node, op_name=_op_name(node))
        dst = self.as_address(self.expr(node.args[2]))
        src = self.as_address(self.expr(node.args[3]))
        src_offset = self.cast_to(self.expr(node.args[4]), i64)
        stride = self.cast_to(self.expr(node.args[5]), i64)
        lane = self.cast_to(self.thread_coordinate("laneid"), i64)
        quad_row = self.binary("FloorDiv", i64, lane, self.const(i64, 4))
        pair_col = self.binary(
            "Mul", i64, self.binary("FloorMod", i64, lane, self.const(i64, 4)), self.const(i64, 2)
        )
        for local_id in range(8):
            source_elem = self.binary("Add", i64, src_offset, self.const(i64, local_id))
            row = self.binary("Add", i64, quad_row, self.const(i64, 8 * ((local_id % 4) // 2)))
            col = self.binary(
                "Add", i64, pair_col, self.const(i64, 8 * (local_id // 4) + local_id % 2)
            )
            dest_elem = self.binary("Add", i64, self.binary("Mul", i64, row, stride), col)
            value = self.builder.reg(ty)
            self.builder.emit(
                "LoadAddr",
                site=site,
                ty=ty,
                dst=value,
                space="Generic",
                addr=self.binary("Add", u64, src, self.element_bytes(dtype, source_elem)),
                sem="Weak",
                scope="Gpu",
                mods=pb.mem_mods(),
            )
            self.builder.emit(
                "StoreAddr",
                site=site,
                ty=ty,
                space="Generic",
                value=value,
                addr=self.binary("Add", u64, dst, self.element_bytes(dtype, dest_elem)),
                sem="Weak",
                scope="Gpu",
                mods=pb.mem_mods(),
            )

    call_tirx_mma_store_legacy = call_tirx_mma_store

    def call_tirx_cuda_float22half2(self: Lowerer, node: Any) -> None:
        self.convert_through_memory(node, pb.Ty("F32", 2), pb.Ty("F16", 2))

    def call_tirx_cuda_float8tohalf8(self: Lowerer, node: Any) -> None:
        self.convert_through_memory(node, pb.Ty("F32", 8), pb.Ty("F16", 8), src_first=True)

    def call_tirx_cuda_half8tofloat8(self: Lowerer, node: Any) -> None:
        self.convert_through_memory(node, pb.Ty("F16", 8), pb.Ty("F32", 8), src_first=True)

    # -- reviewed CUDA helpers ------------------------------------------------
    def call_tirx_cuda_func_call(self: Lowerer, node: Any) -> pb.Operand | None:
        args = list(node.args)
        name = _string(args[0]) if args else None
        if name == "tvm_builtin_pointer_offset":
            return self.pointer_offset(node, args[1], args[2])
        args = args[1:]
        source = ""
        if args and _string(args[-1]) is not None:
            source = _string(args[-1]) or ""
            args = args[:-1]
        if (
            name is not None
            and name.startswith("tvm_builtin_cast_")
            and self.pair_cast_helper(node, name, source, args)
        ):
            return None
        if name == "smem_desc_make_lo_uniform":
            self.smem_desc_make_lo_uniform(node, source, args)
            return None
        if name is not None and name not in builtins.PURE_FUNC_CALLS:
            if self.single_asm_helper(node, source, args):
                return None
        if name is None or name not in builtins.PURE_FUNC_CALLS:
            raise _Unsupported(node, f"cuda.func_call of unreviewed or effectful helper {name!r}")
        digest = hashlib.sha256("".join(source.split()).encode()).hexdigest()[:16]
        self.check_reviewed_helper(node, name, digest, args)
        return self.pure_helper(
            node,
            f"tirx.cuda.func_call.{name}",
            "v*",
            args=args,
            extra_mods=(f"source_sha256={digest}",),
        )

    def pair_cast_helper(self: Lowerer, node: Any, name: str, source: str, args: list[Any]) -> bool:
        """TVM's ``tvm_builtin_cast_<s>x2_<d>x2(void* dst, void* src)`` pair converts (W4-14).

        The body is TVM's generated ``((d2*)dst)[0] = __float22half2_rn(((s2*)src)[0])``
        family; it must match TVM's generator text exactly. Lowered as two element
        loads, round-to-nearest ``Cast``s and two element stores.
        """
        match = re.fullmatch(r"tvm_builtin_cast_(\w+?)x2_(\w+?)x2", name)
        if match is None or len(args) != 2:
            return False
        src_dtype, dst_dtype = match.groups()
        try:
            from tvm.backend.cuda.tile_primitive.elementwise.vec_emit import cast_vec2
        except ImportError:
            return False
        if (src_dtype, dst_dtype) not in cast_vec2._VEC2_CAST_INTRINSICS:
            return False
        expected = cast_vec2._intrinsic_source(src_dtype, dst_dtype)
        if "".join(source.split()) != "".join(expected.split()):
            raise _Unsupported(
                node,
                f"tirx.cuda.func_call helper {name!r} body does not match the validated "
                f"TVM pair-cast implementation",
            )
        src_ty, dst_ty = self.ty(src_dtype, node), self.ty(dst_dtype, node)
        site = self.site(node, op_name=f"tirx.cuda.func_call.{name}")
        values = [self.load_element(args[1], src_dtype, i, src_ty, site) for i in range(2)]
        converted = []
        for value in values:
            dst = self.builder.reg(dst_ty)
            self.builder.emit_cast(dst, value, src_ty, dst_ty, rnd="Rn")
            converted.append(dst)
        for i, value in enumerate(converted):
            self.element_at(args[0], dst_dtype, i, dst_ty, site, store=value)
        return True

    def smem_desc_make_lo_uniform(self: Lowerer, node: Any, source: str, args: list[Any]) -> None:
        """Reviewed ``smem_desc_make_lo_uniform(uint64_t* desc)`` (W4-14): the descriptor's
        low word becomes lane 0's, ``d->lo = __shfl_sync(0xffffffff, d->lo, 0)``."""
        name = "smem_desc_make_lo_uniform"
        digest = hashlib.sha256("".join(source.split()).encode()).hexdigest()[:16]
        if digest != "344b73c0cc918023":
            raise _Unsupported(
                node,
                f"tirx.cuda.func_call helper {name!r} body does not match the validated "
                f"lane-zero low-32-bit descriptor broadcast",
            )
        if len(args) != 1 or _helper_dtype(args[0]) != "handle":
            raise _Unsupported(
                node, f"tirx.cuda.func_call helper {name!r} requires ['handle'] -> void"
            )
        u32, u64 = pb.Ty("U32"), pb.Ty("U64")
        site = self.site(node, op_name=f"tirx.cuda.func_call.{name}")
        desc = self.load_element(args[0], "uint64", 0, u64, site)
        low = self.builder.reg(u32)
        self.builder.emit_cast(low, desc, u64, u32)
        shuffled = self.builder.reg(u32)
        self.builder.emit(
            "Shfl",
            site=site,
            mode="Idx",
            ty=u32,
            dst=shuffled,
            dst_pred=None,
            src=low,
            lane=self.const("uint32", 0),
            clamp=self.const("uint32", 0x1F),
            membermask=self.const("uint32", 0xFFFFFFFF),
        )
        high = self.binary("And", u64, desc, self.const("uint64", 0xFFFFFFFF00000000))
        wide = self.builder.reg(u64)
        self.builder.emit_cast(wide, shuffled, u32, u64)
        self.element_at(args[0], "uint64", 0, u64, site, store=self.binary("Or", u64, high, wide))

    def load_element(
        self: Lowerer, pointer: Any, dtype: str, index: int, ty: pb.Ty, site: int
    ) -> pb.Operand:
        """Load element ``index`` past ``pointer`` (``element_at`` without ``store``)."""
        value = self.element_at(pointer, dtype, index, ty, site)
        assert value is not None  # element_at returns the value when it loads
        return value

    def element_at(
        self: Lowerer,
        pointer: Any,
        dtype: str,
        index: int,
        ty: pb.Ty,
        site: int,
        store: pb.Operand | None = None,
    ) -> pb.Operand | None:
        """Load (or store) element ``index`` past pointer ``pointer`` (``address_of(buf[i])`` or raw)."""
        target = self.buffer_target(pointer)
        if target is not None:
            buf, offset = target
            offset = self.binary(
                "Add", self.operand_ty(offset), offset, self.const(self.operand_ty(offset), index)
            )
            if store is not None:
                self.builder.emit(
                    "Store",
                    site=site,
                    ty=ty,
                    buf=buf,
                    offset=offset,
                    value=store,
                    mods=pb.mem_mods(),
                    sem="Weak",
                    scope="Gpu",
                )
                return None
            dst = self.builder.reg(ty)
            self.builder.emit(
                "Load",
                site=site,
                ty=ty,
                dst=dst,
                buf=buf,
                offset=offset,
                mods=pb.mem_mods(),
                sem="Weak",
                scope="Gpu",
            )
            return dst
        addr = self.as_address(self.expr(pointer))
        if index:
            addr = self.binary(
                "Add", pb.Ty("U64"), addr, self.const("uint64", index * dtypes.bits(dtype) // 8)
            )
        if store is not None:
            self.builder.emit(
                "StoreAddr",
                site=site,
                ty=ty,
                addr=addr,
                space="Generic",
                value=store,
                mods=pb.mem_mods(),
                sem="Weak",
                scope="Gpu",
            )
            return None
        dst = self.builder.reg(ty)
        self.builder.emit(
            "LoadAddr",
            site=site,
            ty=ty,
            dst=dst,
            addr=addr,
            space="Generic",
            mods=pb.mem_mods(),
            sem="Weak",
            scope="Gpu",
        )
        return dst

    def check_reviewed_helper(
        self: Lowerer, node: Any, name: str, digest: str, args: list[Any]
    ) -> None:
        """Fail closed unless ``name`` is the reviewed helper (signature, then body)."""
        reviewed = builtins.REVIEWED_HELPERS.get(name)
        if reviewed is None:
            return
        expected_digest, arg_dtypes, result_dtype, description = reviewed
        if arg_dtypes or result_dtype:
            actual = [_helper_dtype(a) for a in args]
            if actual != list(arg_dtypes) or (dtypes.dtype_of(node) or "") != result_dtype:
                raise _Unsupported(
                    node,
                    f"tirx.cuda.func_call helper {name!r} requires {list(arg_dtypes)} -> "
                    f"{result_dtype or 'void'}",
                )
        if digest != expected_digest:
            raise _Unsupported(
                node, f"tirx.cuda.func_call helper {name!r} body does not match the {description}"
            )

    def single_asm_helper(self: Lowerer, node: Any, source: str, args: list[Any]) -> bool:
        """Lower an effectful helper whose body is exactly one reviewed PTX statement.

        The semantics come from the helper's own asm text (tensormap replace and
        tensormap proxy fences), so a modified body cannot be accepted by name.
        """
        if source.count("asm") != 1:
            return False
        site = self.site(node, op_name="tirx.cuda.func_call")
        match = _ASM_REPLACE.search(source)
        if match and len(args) == 2:
            field = {
                "global_address": "GlobalAddress",
                "global_dim": "GlobalDim",
                "global_stride": "GlobalStride",
            }[match.group(1)]
            value_ty = pb.Ty("U64") if match.group(2) == "64" else pb.Ty("U32")
            ordinal = int(match.group(3)) if match.group(3) is not None else None
            tmap = self.as_address(self.expr(args[0]))
            value = self.convert(self.expr(args[1]), value_ty)
            self.builder.emit(
                "TensorMapReplace",
                site=site,
                tmap=tmap,
                space="Global",
                field=field,
                ord=ordinal,
                value=value,
            )
            return True
        match = _ASM_RELEASE.search(source)
        if match and not args:
            self.builder.emit(
                "Fence",
                site=site,
                kind="TensormapRelease",
                sem="Release",
                scope=_SCOPES[match.group(1)],
            )
            return True
        match = _ASM_ACQUIRE.search(source)
        if match and len(args) == 1:
            addr = self.as_address(self.expr(args[0]))
            self.builder.emit(
                "Fence",
                site=site,
                kind={"TensormapAcquire": {"addr": pb.opnd(addr), "space": "Generic"}},
                sem="Acquire",
                scope=_SCOPES[match.group(1)],
            )
            return True
        return False

    def call_tirx_ptx_addr(self: Lowerer, node: Any) -> pb.Operand:
        """``T.ptx.addr(base, byte_offset)``: an address operand ``base + byte_offset``."""
        base = self.as_address(self.expr(node.args[0]))
        offset = self.reinterpret(self.cast_to(self.expr(node.args[1]), "int64"), pb.Ty("U64"))
        return self.binary("Add", pb.Ty("U64"), base, offset)

    def pointer_offset(self: Lowerer, node: Any, pointer: Any, offset: Any) -> pb.Operand:
        """TVM's ``tvm_builtin_pointer_offset(T* ptr, int offset)`` = ``ptr + offset`` elements."""
        if (
            type_key(pointer) == "ir.Call"
            and _op_name(pointer) == "tirx.address_of"
            and type_key(pointer.args[0]) == "ir.TensorLoad"
        ):
            elem = dtypes.dtype_of(pointer.args[0])
        else:
            element = getattr(getattr(pointer, "ty", None), "element_type", None)
            elem = str(getattr(element, "dtype", "")) if element is not None else ""
        if not elem or elem == "void":
            raise _Unsupported(node, "tvm_builtin_pointer_offset on an untyped pointer")
        base = self.as_address(self.expr(pointer))
        return self.binary("Add", pb.Ty("U64"), base, self.element_bytes(elem, self.expr(offset)))

    # -- wait_until -----------------------------------------------------------
    def call_tirx_cuda_wait_until(self: Lowerer, node: Any) -> None:
        args = list(node.args)
        if len(args) < 3:
            raise _Unsupported(node, "wait_until needs (dst, ptr, predicate, ...)")
        dst_node, ptr_node, pred_node = args[:3]
        strings = [_string(a) for a in args[3:]]
        scope = strings[0] if len(strings) > 0 and strings[0] else "gpu"
        space_name = strings[1] if len(strings) > 1 and strings[1] else "global"
        width = strings[2] if len(strings) > 2 and strings[2] else ""
        if type_key(dst_node) != "ir.TensorLoad" or not isinstance(
            self.ref_of(dst_node.source), RegArray
        ):
            raise _Unsupported(node, "wait_until destination must be a promoted local scalar")
        dst, write_back = self.lvalue_target(dst_node)
        if write_back is not None:
            raise _Unsupported(node, "wait_until destination must have a constant index")
        addr = self.as_address(self.expr(ptr_node))
        ty = pb.Ty.from_ptx(width) if width else self.builder.reg_ty(dst)
        pred_index, captures = self.lower_predicate(pred_node, dst)
        space = {"global": "Global", "shared": "Shared"}.get(space_name, "Generic")
        self.mark_sync_words(ptr_node)
        self.builder.emit(
            "WaitUntil",
            site=self.site(node, op_name="tirx.cuda.wait_until"),
            dst=dst,
            addr=addr,
            ty=ty,
            space=space,
            sem="Acquire",
            scope={"cta": "Cta", "cluster": "Cluster", "gpu": "Gpu", "sys": "Sys"}[scope],
            pred=pred_index,
            captures=captures,
        )
        return None

    def mark_sync_words(self: Lowerer, ptr_node: Any) -> None:
        """``BufferDecl::sync_words`` hint for the buffer the wait polls."""
        if type_key(ptr_node) == "ir.Call" and _op_name(ptr_node) == "tirx.address_of":
            target = ptr_node.args[0]
            var = target.source if type_key(target) == "ir.TensorLoad" else target
            ref = self.refs.get(int(var.__chandle__()))
            buf = getattr(ref, "buf", None)
            if buf is not None:
                # W5-7: the polled buffer itself (a shared/cluster view or a global
                # parameter), never its backing pool.
                decl = self.builder.program.buffers[buf]
                from dataclasses import replace

                self.builder.program.buffers[buf] = replace(decl, sync_words=True)

    def lower_predicate(self: Lowerer, pred_node: Any, dst: pb.Reg) -> tuple[int, list[pb.Reg]]:
        """Lower ``pred_node`` out of line with ``dst`` bound to a fresh argument register."""
        arg = self.builder.reg(self.builder.reg_ty(dst), name="wait_value")
        code: list[pb.Instr] = []
        sites: list[int] = []
        first_reg = len(self.builder.program.regs)
        saved = self.builder.redirect(code, sites)
        saved_args = self.pred_args
        self.pred_args = {dst.index: arg}
        try:
            value = self.cast_to(self.expr(pred_node), pb.Ty("Pred"))
            if isinstance(value, pb.Const):
                result = self.builder.reg(pb.Ty("Pred"))
                self.builder.emit("Mov", dst=result, src=value)
            else:
                result = value
        finally:
            self.builder.redirect(*saved)
            self.pred_args = saved_args
        reads_memory = False
        captures: list[pb.Reg] = []
        defined = {arg.index}
        for instr in code:
            if instr.variant not in pb.PRED_ALLOWED:
                raise _Unsupported(pred_node, f"wait_until predicate contains {instr.variant}")
            if instr.variant in ("Load", "LoadAddr", "LoadRegIndexed"):
                reads_memory = True
            for value in instr.operands():
                if (
                    isinstance(value, pb.Reg)
                    and value.index < first_reg
                    and value.index not in defined
                    and value not in captures
                ):
                    captures.append(value)
            for reg in instr.writes():
                defined.add(reg.index)
        self.pending_preds.append((code, sites, arg, result, reads_memory))
        return len(self.pending_preds) - 1, captures

    def is_pure(self: Lowerer, node: Any) -> bool:
        """No memory reads of non-register buffers and no calls with effects."""
        from tvm_ffi import structural_visit

        import tvm

        pure = True

        def on_load(sub: Any, visitor: Any) -> None:
            nonlocal pure
            if not isinstance(self.ref_of(sub.source), RegArray):
                pure = False
            visitor.default_visit(sub)

        def on_call(sub: Any, visitor: Any) -> None:
            nonlocal pure
            name = _op_name(sub)
            helper = builtins.HELPERS.get(name)
            if not (
                name in builtins.UNARY_OPS
                or name in _PURE_STRUCTURAL
                or (helper is not None and helper.kind == "pure")
            ):
                pure = False
            visitor.default_visit(sub)

        structural_visit(node, [(tvm.ir.TensorLoad, on_load), (tvm.ir.Call, on_call)])
        return pure


__all__ = ["CallsMixin"]


def _helper_dtype(node: Any) -> str:
    """Legacy helper argument typing: "handle" for pointers, else the dtype."""
    ty = getattr(node, "ty", None)
    if ty is not None and type_key(ty) == "ir.PointerType":
        return "handle"
    return dtypes.dtype_of(node) or ""


# Legacy reinterpret validation (frontend-rs emit/pure.rs `validate_reinterpret`
# over dtype_registry.json capability classes). Registers hold bits, so v2 could
# model more, but scalar low-precision/storage-only payload reinterprets stay
# rejected exactly as before.
_SCALAR_CLASSES = {"integer", "boolean", "scalar", "scalar_raw"}
_RAW_CLASSES = {"integer", "scalar_raw"}


def _dtype_classes() -> dict[str, str]:
    global _DTYPE_CLASSES
    if _DTYPE_CLASSES is None:
        import json
        import pathlib

        path = pathlib.Path(__file__).resolve().parents[2] / "dtype_registry.json"
        types = json.loads(path.read_text())["types"]
        _DTYPE_CLASSES = {
            k: (v.get("class") if isinstance(v, dict) else v) for k, v in types.items()
        }
    return _DTYPE_CLASSES


_DTYPE_CLASSES: dict[str, str] | None = None


def _reinterpret_rejection(source: str, target: str) -> str | None:
    if {source, target} <= {"handle", "uint64"} and "handle" in (source, target):
        return None
    if not (dtypes.known(source) and dtypes.known(target)):
        return None  # pointer forms and the like are checked elsewhere
    if dtypes.bits(source) != dtypes.bits(target):
        return "source and result must have identical bit widths"
    if (source, target) in {
        ("uint16", "float16"),
        ("float16", "uint16"),
        ("uint16", "bfloat16"),
        ("bfloat16", "uint16"),
    }:
        return None
    classes = _dtype_classes()
    source_vector = dtypes.split(source)[1] > 1
    target_vector = dtypes.split(target)[1] > 1
    if source == target and (classes.get(source) in _SCALAR_CLASSES or source_vector):
        return None
    if (classes.get(source) in _RAW_CLASSES or source_vector) and (
        classes.get(target) in _RAW_CLASSES or target_vector
    ):
        return None
    return (
        "raw payload reinterpret is not modeled for scalar low-precision/storage-only dtype; "
        "use an explicitly supported packed storage dtype"
    )
