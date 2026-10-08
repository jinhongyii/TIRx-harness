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

from . import builtins
from . import dtypes
from . import program_builder as pb
from . import ptx_decode
from .dtypes import type_key
from .memory import RegArray, _Unsupported
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
    "laneid": "LaneId", "warpid": "WarpInCta", "smid": "SmId", "nsmid": "NSmId", "gridid": "GridId",
    "clock": "Clock", "clock64": "Clock64", "globaltimer": "GlobalTimer", "lanemask_eq": "LaneMaskEq",
    "lanemask_lt": "LaneMaskLt", "lanemask_le": "LaneMaskLe", "lanemask_gt": "LaneMaskGt",
    "lanemask_ge": "LaneMaskGe", "cluster_ctarank": "ClusterCtaRank", "cluster_nctarank": "ClusterNCtaRank",
    "dynamic_smem_size": "DynamicSmemSize", "total_smem_size": "TotalSmemSize", "nwarpid": "NWarpId",
}
_SREG_AXIS = {"tid": "Tid", "ntid": "NTid", "ctaid": "CtaId", "nctaid": "NCtaId", "clusterid": "ClusterId",
              "nclusterid": "NClusterId", "cluster_ctaid": "ClusterCtaId", "cluster_nctaid": "ClusterNCtaId"}

_ASM_REPLACE = re.compile(
    r'"tensormap\.replace\.tile\.(global_address|global_dim|global_stride)\.global\.b1024\.b(32|64) '
    r'\[%0\], (?:(\d+), )?%1;"'
)
_ASM_RELEASE = re.compile(r'"fence\.proxy\.tensormap::generic\.release\.(cta|cluster|gpu|sys);')
_ASM_ACQUIRE = re.compile(r'"fence\.proxy\.tensormap::generic\.acquire\.(cta|cluster|gpu|sys) \[%0\], 128;')
_SCOPES = {"cta": "Cta", "cluster": "Cluster", "gpu": "Gpu", "sys": "Sys"}

_PURE_STRUCTURAL = frozenset({"tirx.reinterpret", "tirx.if_then_else", "prim.if_then_else", "tirx.likely"})


class CallsMixin:
    """Mixed into ``Lowerer``."""

    def call(self: "Lowerer", node: Any, *, statement: bool) -> pb.Operand | None:
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
            return structural(node)
        if helper is not None and helper.kind == "pure":
            return self.pure_helper(node, name, helper.roles)
        if name.startswith("tirx.tile."):
            raise _Unsupported(node, f"tile op {name} (decision 6: lower via TVM dispatch; not done yet)")
        raise _Unsupported(node, f"builtin {name}")

    # -- pure helpers -> Ptx ------------------------------------------------
    def pure_helper(self: "Lowerer", node: Any, name: str, roles: str, args: list[Any] | None = None,
                    extra_mods: tuple[str, ...] = ()) -> pb.Operand | None:
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
                    raise _Unsupported(node, f"{name}: argument {position} must be a string literal")
                mods.append(f"arg{position}={text}")
                continue
            if role in "ox" and not self.is_lvalue_ref(arg):
                # A pointer value: read/write through memory around the op.
                element = getattr(getattr(arg, "ty", None), "element_type", None)
                elem_dtype = str(getattr(element, "dtype", "")) if element is not None else ""
                if not elem_dtype or elem_dtype == "void":
                    raise _Unsupported(node, f"{name}: out-parameter {position} has no element type")
                ty = self.ty(elem_dtype, node)
                addr = self.as_address(self.expr(arg))
                reg = self.builder.reg(ty)
                site = self.site(node, op_name=name)
                if role == "x":
                    self.builder.emit("LoadAddr", site=site, ty=ty, dst=reg, addr=addr, space="Generic",
                                      sem="Weak", scope="Gpu", mods=pb.mem_mods())
                    srcs.append(reg)
                dsts.append(reg)

                def store_through(reg: pb.Reg = reg, addr: pb.Operand = addr, ty: pb.Ty = ty, site: int = site) -> None:
                    self.builder.emit("StoreAddr", site=site, ty=ty, addr=addr, space="Generic", value=reg,
                                      sem="Weak", scope="Gpu", mods=pb.mem_mods())

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
        self.builder.emit("Ptx", site=self.site(node, op_name=name), op=op, dsts=dsts, srcs=srcs, pred=None,
                          keep_dst=False)
        for write_back in write_backs:
            write_back()
        return result

    def is_lvalue_ref(self: "Lowerer", arg: Any) -> bool:
        return type_key(arg) == "ir.Call" and _op_name(arg) == "tirx.address_of" and \
            type_key(arg.args[0]) == "ir.TensorLoad"

    def result_from(self: "Lowerer", node: Any, value: pb.Operand) -> pb.Operand:
        dtype = dtypes.dtype_of(node)
        return self.cast_to(value, dtype) if dtype else value

    def full_mask(self: "Lowerer") -> pb.Const:
        return self.const("uint32", 0xFFFF_FFFF)

    # -- structural tirx.* ops ---------------------------------------------
    def call_tirx_address_of(self: "Lowerer", node: Any) -> pb.Operand:
        return self.address_of(node)

    def call_tirx_buffer_data(self: "Lowerer", node: Any) -> pb.Operand:
        return self.buffer_data(node)

    def call_tirx_type_annotation(self: "Lowerer", node: Any) -> pb.Operand:
        return self.const("int32", 0)

    def call_tirx_reinterpret(self: "Lowerer", node: Any) -> pb.Operand:
        value = self.expr(node.args[0])
        dtype = dtypes.dtype_of(node)
        if dtype == "handle":
            return self.as_address(value)
        return self.reinterpret(value, self.ty(dtype, node))

    def call_tirx_ptr_byte_offset(self: "Lowerer", node: Any) -> pb.Operand:
        base = self.as_address(self.expr(node.args[0]))
        offset = self.reinterpret(self.cast_to(self.expr(node.args[1]), "int64"), pb.Ty("U64"))
        return self.binary("Add", pb.Ty("U64"), base, offset)

    def call_tirx_handle_add_byte_offset(self: "Lowerer", node: Any) -> pb.Operand:
        return self.call_tirx_ptr_byte_offset(node)

    def call_tirx_tvm_access_ptr(self: "Lowerer", node: Any) -> pb.Operand:
        """``tvm_access_ptr(type, data, offset, extent, rw_mask)`` = ``data + offset`` elements.

        ``extent`` and ``rw_mask`` are access-pattern hints with no semantics in the
        engine (the Arena checks the actual accesses through the pointer).
        """
        type_node, data, offset = node.args[0], node.args[1], node.args[2]
        elem = dtypes.dtype_of(type_node)
        if not elem or elem in ("void", "handle"):
            raise _Unsupported(node, "tvm_access_ptr without an element type")
        base = self.as_address(self.expr(data))
        return self.binary("Add", pb.Ty("U64"), base, self.element_bytes(elem, self.expr(offset)))

    def call_tirx_isnullptr(self: "Lowerer", node: Any) -> pb.Operand:
        value = self.as_address(self.expr(node.args[0]))
        dst = self.builder.reg(pb.Ty("Pred"), uniform=self.is_uniform(value))
        self.builder.emit("Compare", op="Eq", ty=pb.Ty("U64"), dst=dst, a=value, b=self.const(pb.Ty("U64"), 0))
        return dst

    def call_tirx_likely(self: "Lowerer", node: Any) -> pb.Operand:
        return self.expr(node.args[0])

    def call_tirx_fma(self: "Lowerer", node: Any) -> pb.Operand:
        ty = self.ty(dtypes.dtype_of(node), node)
        a, b, c = (self.cast_to(self.expr(x), ty) for x in node.args)
        dst = self.builder.reg(ty, uniform=all(self.is_uniform(v) for v in (a, b, c)))
        self.builder.emit("Ternary", op="Fma", ty=ty, dst=dst, a=a, b=b, c=c)
        return dst

    def call_tirx_if_then_else(self: "Lowerer", node: Any) -> pb.Operand:
        cond_node, a_node, b_node = node.args
        ty = self.ty(dtypes.dtype_of(node), node)
        cond = self.cast_to(self.expr(cond_node), pb.Ty("Pred"))
        if self.is_pure(a_node) and self.is_pure(b_node):
            a, b = self.cast_to(self.expr(a_node), ty), self.cast_to(self.expr(b_node), ty)
            dst = self.builder.reg(ty, uniform=all(self.is_uniform(v) for v in (cond, a, b)))
            self.builder.emit("Select", ty=ty, dst=dst, cond=cond, a=a, b=b)
            return dst
        # Only the taken arm executes (it can guard an out-of-bounds load).
        dst = self.builder.reg(ty)
        b = self.builder
        site = self.site(node)
        if_pc = b.emit("If", site=site, cond=cond, else_pc=0, end_pc=0, elect=False)
        b.emit("Mov", dst=dst, src=self.cast_to(self.expr(a_node), ty))
        else_pc = b.emit("Else", end_pc=0)
        b.emit("Mov", dst=dst, src=self.cast_to(self.expr(b_node), ty))
        end_pc = b.emit("EndIf")
        b.patch(if_pc, "If", cond=cond, else_pc=else_pc, end_pc=end_pc, elect=False)
        b.patch(else_pc, "Else", end_pc=end_pc)
        return dst

    call_prim_if_then_else = call_tirx_if_then_else

    def call_tirx_break_loop(self: "Lowerer", node: Any) -> None:
        self.builder.emit("Break")
        return None

    def call_tirx_continue_loop(self: "Lowerer", node: Any) -> None:
        self.builder.emit("Continue")
        return None

    # -- special registers --------------------------------------------------
    def read_special(self: "Lowerer", node: Any, sreg: Any, ty: pb.Ty) -> pb.Operand:
        dst = self.builder.reg(ty)
        self.builder.emit("ReadSpecial", dst=dst, sreg=sreg)
        return self.result_from(node, dst)

    def call_tirx_cuda_thread_rank(self: "Lowerer", node: Any) -> pb.Operand:
        return self.read_special(node, "ThreadInCta", pb.Ty("S32"))

    def call_tirx_cuda_clock64(self: "Lowerer", node: Any) -> pb.Operand:
        return self.read_special(node, "Clock64", pb.Ty("U64"))

    def call_tirx_cuda___activemask(self: "Lowerer", node: Any) -> pb.Operand:
        return self.read_special(node, "ActiveMask", pb.Ty("U32"))

    call_tirx_tvm_warp_activemask = call_tirx_cuda___activemask

    def call_tirx_cuda_mov_sreg(self: "Lowerer", node: Any) -> pb.Operand:
        name = _string(node.args[1])
        if name is None:
            raise _Unsupported(node, "mov_sreg register name must be a literal")
        bits = int(node.args[0].value)
        ty = pb.Ty("U64") if bits == 64 else pb.Ty("U32")
        if name in _SREGS:
            return self.read_special(node, _SREGS[name], ty)
        if name in ("clock_hi", "globaltimer_hi", "globaltimer_lo"):
            source = self.builder.reg(pb.Ty("U64"))
            self.builder.emit("ReadSpecial", dst=source,
                              sreg="Clock64" if name.startswith("clock") else "GlobalTimer")
            if name.endswith("_hi"):
                source = self.binary("Shr", pb.Ty("U64"), source, self.const("uint64", 32))
            return self.cast_to(source, dtypes.dtype_of(node) or "uint32")
        base, _, axis = name.partition(".")
        if base in _SREG_AXIS and axis in ("x", "y", "z"):
            return self.read_special(node, {_SREG_AXIS[base]: axis.upper()}, ty)
        raise _Unsupported(node, f"special register %{name}")

    # -- annotations ----------------------------------------------------------
    def nop(self: "Lowerer", node: Any) -> pb.Operand | None:
        self.builder.emit("Nop", site=self.site(node, op_name=_op_name(node)))
        dtype = dtypes.dtype_of(node)
        if dtype:
            return self.const(dtype, 0)
        return None

    call_tirx_cuda_printf = nop
    call_tirx_cuda_iket_mark = nop
    call_tirx_cuda_iket_range_start = nop
    call_tirx_cuda_iket_range_end = nop
    call_tirx_cuda_iket_range_push = nop
    call_tirx_cuda_iket_range_pop = nop
    call_tirx_cuda_iket_sentinel_token = nop
    call_tirx_cuda_iket_official_event = nop
    call_tirx_cuda_nano_sleep = nop

    def call_tirx_cuda_trap_when_assert_failed(self: "Lowerer", node: Any) -> None:
        cond = self.cast_to(self.expr(node.args[0]), pb.Ty("Pred"))
        self.builder.emit("Assert", site=self.site(node), cond=cond,
                          msg=self.builder.string("trap_when_assert_failed"))
        return None

    # -- synchronization ------------------------------------------------------
    def mbar_wait(self: "Lowerer", node: Any, scope: str) -> None:
        addr = self.expr(node.args[0])
        space = "Shared" if self.operand_ty(addr).bits == 32 else "Generic"
        phase = self.cast_to(self.expr(node.args[1]), pb.Ty("U32"))
        self.builder.emit("MbarWait", site=self.site(node, op_name=_op_name(node)), mbar=addr, space=space,
                          phase=pb.phase_parity(phase), sem="Acquire", scope=scope)

    def call_tirx_cuda_mbarrier_wait(self: "Lowerer", node: Any) -> None:
        self.mbar_wait(node, "Cta")

    def call_tirx_cuda_mbarrier_wait_acquire_cluster(self: "Lowerer", node: Any) -> None:
        self.mbar_wait(node, "Cluster")

    def barrier(self: "Lowerer", node: Any, kind: Any, ident: pb.Operand, count: pb.Operand | None) -> None:
        self.builder.emit("Barrier", site=self.site(node, op_name=_op_name(node)), kind=kind, id=ident,
                          count=count, aligned=True)

    def call_tirx_cuda_cta_sync(self: "Lowerer", node: Any) -> None:
        self.barrier(node, "Sync", self.const("uint32", 0), None)

    def call_tirx_tvm_storage_sync(self: "Lowerer", node: Any) -> None:
        scope = _string(node.args[0]) if node.args else "shared"
        if scope not in ("shared", "shared.dyn"):
            raise _Unsupported(node, f"tvm_storage_sync({scope!r})")
        self.barrier(node, "Sync", self.const("uint32", 0), None)

    def call_tirx_cuda_warpgroup_sync(self: "Lowerer", node: Any) -> None:
        ident = self.cast_to(self.expr(node.args[0]), pb.Ty("U32"))
        self.barrier(node, "Sync", ident, self.const("uint32", 128))

    def call_tirx_cuda_warp_sync(self: "Lowerer", node: Any) -> None:
        mask = self.cast_to(self.expr(node.args[0]), pb.Ty("U32")) if node.args else self.full_mask()
        self.builder.emit("WarpSync", site=self.site(node, op_name=_op_name(node)), membermask=mask)

    def call_tirx_cuda_cluster_sync(self: "Lowerer", node: Any) -> None:
        site = self.site(node, op_name=_op_name(node))
        self.builder.emit("ClusterArrive", site=site, sem="Release", aligned=True)
        self.builder.emit("ClusterWait", site=site, acquire=True, aligned=True)

    def call_tirx_cuda_grid_sync(self: "Lowerer", node: Any) -> None:
        self.builder.emit("GridSync", site=self.site(node, op_name=_op_name(node)))

    def syncthreads_red(self: "Lowerer", node: Any, op: str) -> pb.Operand:
        pred = self.cast_to(self.expr(node.args[0]), pb.Ty("Pred"))
        dst = self.builder.reg(pb.Ty("Pred"))
        self.barrier(node, {"Red": {"op": op, "pred": pb.opnd(pred), "dst": dst}}, self.const("uint32", 0), None)
        return self.result_from(node, dst)

    def call_tirx_cuda_syncthreads_and(self: "Lowerer", node: Any) -> pb.Operand:
        return self.syncthreads_red(node, "And")

    def call_tirx_cuda_syncthreads_or(self: "Lowerer", node: Any) -> pb.Operand:
        return self.syncthreads_red(node, "Or")

    def call_tirx_cuda_thread_fence(self: "Lowerer", node: Any) -> None:
        self.builder.emit("Fence", site=self.site(node, op_name=_op_name(node)), kind="Thread", sem="Sc",
                          scope="Gpu")

    # -- warp collectives -----------------------------------------------------
    def call_tirx_cuda_elect_sync(self: "Lowerer", node: Any) -> pb.Operand:
        mask = self.cast_to(self.expr(node.args[0]), pb.Ty("U32")) if node.args else self.full_mask()
        pred = self.builder.reg(pb.Ty("Pred"))
        self.builder.emit("Elect", site=self.site(node, op_name=_op_name(node)), dst_pred=pred, dst_lane=None,
                          membermask=mask)
        return self.result_from(node, pred)

    def shfl(self: "Lowerer", node: Any, mode: str, mask: Any, value: Any, lane: Any, width: Any) -> pb.Operand:
        ty = self.ty(dtypes.dtype_of(node), node)
        src = self.cast_to(self.expr(value), ty)
        lane_op = self.cast_to(self.expr(lane), pb.Ty("U32"))
        width_op = self.cast_to(self.expr(width), pb.Ty("U32"))
        # CUDA: c = ((32 - width) << 8) | (up ? 0 : 0x1f)
        segment = self.binary("Shl", pb.Ty("U32"), self.binary("Sub", pb.Ty("U32"), self.const("uint32", 32), width_op),
                              self.const("uint32", 8))
        clamp = segment if mode == "Up" else self.binary("Or", pb.Ty("U32"), segment, self.const("uint32", 0x1F))
        dst = self.builder.reg(ty)
        self.builder.emit("Shfl", site=self.site(node, op_name=_op_name(node)), mode=mode, ty=ty, dst=dst,
                          dst_pred=None, src=src, lane=lane_op, clamp=clamp,
                          membermask=self.cast_to(self.expr(mask), pb.Ty("U32")))
        return dst

    def call_tirx_cuda___shfl_sync(self: "Lowerer", node: Any) -> pb.Operand:
        return self.shfl(node, "Idx", *node.args)

    def call_tirx_cuda___shfl_up_sync(self: "Lowerer", node: Any) -> pb.Operand:
        return self.shfl(node, "Up", *node.args)

    def call_tirx_cuda___shfl_down_sync(self: "Lowerer", node: Any) -> pb.Operand:
        return self.shfl(node, "Down", *node.args)

    def call_tirx_cuda___shfl_xor_sync(self: "Lowerer", node: Any) -> pb.Operand:
        return self.shfl(node, "Bfly", *node.args)

    def call_tirx_tvm_warp_shuffle(self: "Lowerer", node: Any) -> pb.Operand:
        mask, value, lane, width, _warp_size = node.args
        return self.shfl(node, "Idx", mask, value, lane, width)

    def call_tirx_tvm_warp_shuffle_up(self: "Lowerer", node: Any) -> pb.Operand:
        mask, value, lane, width, _warp_size = node.args
        return self.shfl(node, "Up", mask, value, lane, width)

    def call_tirx_tvm_warp_shuffle_down(self: "Lowerer", node: Any) -> pb.Operand:
        mask, value, lane, width, _warp_size = node.args
        return self.shfl(node, "Down", mask, value, lane, width)

    def call_tirx_tvm_warp_shuffle_xor(self: "Lowerer", node: Any) -> pb.Operand:
        mask, value, lane, width, _warp_size = node.args
        return self.shfl(node, "Bfly", mask, value, lane, width)

    def vote(self: "Lowerer", node: Any, mode: str, dst_ty: pb.Ty) -> pb.Operand:
        mask, pred = node.args
        dst = self.builder.reg(dst_ty)
        self.builder.emit("Vote", site=self.site(node, op_name=_op_name(node)), mode=mode, dst=dst,
                          pred=self.cast_to(self.expr(pred), pb.Ty("Pred")),
                          membermask=self.cast_to(self.expr(mask), pb.Ty("U32")))
        return self.result_from(node, dst)

    def call_tirx_cuda_ballot_sync(self: "Lowerer", node: Any) -> pb.Operand:
        return self.vote(node, "Ballot", pb.Ty("U32"))

    def call_tirx_cuda_any_sync(self: "Lowerer", node: Any) -> pb.Operand:
        return self.vote(node, "Any", pb.Ty("Pred"))

    def redux(self: "Lowerer", node: Any, op: str) -> pb.Operand:
        mask, value = node.args
        dst = self.builder.reg(pb.Ty("U32"))
        self.builder.emit("Redux", site=self.site(node, op_name=_op_name(node)), op=op, ty=pb.Ty("U32"), dst=dst,
                          src=self.cast_to(self.expr(value), pb.Ty("U32")),
                          membermask=self.cast_to(self.expr(mask), pb.Ty("U32")))
        return self.result_from(node, dst)

    def call_tirx_cuda_reduce_add_sync_u32(self: "Lowerer", node: Any) -> pb.Operand:
        return self.redux(node, "Add")

    def call_tirx_cuda_reduce_min_sync_u32(self: "Lowerer", node: Any) -> pb.Operand:
        return self.redux(node, "Min")

    # -- reductions (TVM's templated butterfly helpers, expanded) -------------
    def butterfly(self: "Lowerer", node: Any, value: pb.Operand, op: str, width: int) -> pb.Operand:
        """``tvm_builtin_cuda_warp_reduce_<op>_<width>``: log2(width) shfl.bfly steps."""
        ty = self.operand_ty(value)
        step = {"sum": "Add", "max": "Max", "min": "Min"}[op]
        site = self.site(node, op_name=_op_name(node))
        mask = width >> 1
        while mask > 0:
            shuffled = self.builder.reg(ty)
            self.builder.emit("Shfl", site=site, mode="Bfly", ty=ty, dst=shuffled, dst_pred=None, src=value,
                              lane=self.const("uint32", mask), clamp=self.const("uint32", 0x1F),
                              membermask=self.full_mask())
            value = self.binary(step, ty, value, shuffled)
            mask >>= 1
        return value

    def reduce_args(self: "Lowerer", node: Any, op_arg: Any, count_arg: Any, what: str) -> tuple[str, int]:
        op = _string(op_arg)
        if op not in ("sum", "max", "min"):
            raise _Unsupported(node, f"{what} op {op!r}")
        if type_key(count_arg) != "ir.IntImm":
            raise _Unsupported(node, f"{what} width must be a constant")
        count = int(count_arg.value)
        if count < 1 or count > 32 or count & (count - 1):
            raise _Unsupported(node, f"{what} width {count} is not a power of two in [1, 32]")
        return op, count

    def call_tirx_cuda_warp_reduce(self: "Lowerer", node: Any) -> pb.Operand:
        value_arg, op_arg = node.args[0], node.args[1]
        width_arg = node.args[2] if len(node.args) > 2 else None
        op, width = self.reduce_args(node, op_arg, width_arg if width_arg is not None else _int32(32), "warp_reduce")
        ty = self.ty(dtypes.dtype_of(node), node)
        return self.butterfly(node, self.cast_to(self.expr(value_arg), ty), op, width)

    def call_tirx_cuda_cta_reduce(self: "Lowerer", node: Any) -> pb.Operand:
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
            b.emit("StoreAddr", site=site, ty=ty, addr=slot(index), space="Generic", value=data, sem="Weak",
                   scope="Gpu", mods=pb.mem_mods())

        def load(index: pb.Operand, dst: pb.Reg) -> None:
            b.emit("LoadAddr", site=site, ty=ty, dst=dst, addr=slot(index), space="Generic", sem="Weak",
                   scope="Gpu", mods=pb.mem_mods())

        def guarded(cond: pb.Operand, then: Any, otherwise: Any = None) -> None:
            if_pc = b.emit("If", site=site, cond=cond, else_pc=0, end_pc=0, elect=False)
            then()
            else_pc = -1
            if otherwise is not None:
                else_pc = b.emit("Else", end_pc=0)
                otherwise()
            end = b.emit("EndIf")
            b.patch(if_pc, "If", cond=cond, else_pc=else_pc if else_pc >= 0 else end, end_pc=end, elect=False)
            if else_pc >= 0:
                b.patch(else_pc, "Else", end_pc=end)

        def is_zero(x: pb.Operand) -> pb.Reg:
            dst = b.reg(pb.Ty("Pred"))
            b.emit("Compare", op="Eq", ty=pb.Ty("S32"), dst=dst, a=x, b=self.const("int32", 0))
            return dst

        guarded(is_zero(lane), lambda: store(warp, value))
        b.emit("Barrier", site=site, kind="Sync", id=self.const("uint32", 0), count=None, aligned=True)
        partial = b.reg(ty)

        def leader_warp() -> None:
            in_range = b.reg(pb.Ty("Pred"))
            b.emit("Compare", op="Lt", ty=pb.Ty("S32"), dst=in_range, a=lane, b=self.const("int32", num_warps))
            guarded(in_range, lambda: load(lane, partial),
                    lambda: b.emit("Mov", dst=partial, src=self.reduce_identity(dtype, op)))
            reduced = self.butterfly(node, partial, op, 32)
            guarded(is_zero(lane), lambda: store(self.const("int32", 0), reduced))

        guarded(is_zero(warp), leader_warp)
        b.emit("Barrier", site=site, kind="Sync", id=self.const("uint32", 0), count=None, aligned=True)
        result = b.reg(ty)
        load(self.const("int32", 0), result)
        return result

    def reduce_identity(self: "Lowerer", dtype: str, op: str) -> pb.Const:
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
    def call_tirx_cuda_ldg(self: "Lowerer", node: Any) -> pb.Operand:
        addr = self.as_address(self.expr(node.args[0]))
        text = _string(node.args[1]) if len(node.args) > 1 else None
        if text is None or len(node.args) != 2:
            raise _Unsupported(node, "cuda.ldg vector/destination form")
        ty = self.ty(text, node)
        dst = self.builder.reg(ty)
        self.builder.emit("LoadAddr", site=self.site(node, op_name=_op_name(node)), ty=ty, dst=dst, addr=addr,
                          space="Global", sem="Weak", scope="Gpu", mods=pb.mem_mods(nc=True))
        self.builder.program.requirements.readonly_proxy = True
        return self.result_from(node, dst)

    def atomic(self: "Lowerer", node: Any, op: str) -> pb.Operand:
        args = list(node.args)
        addr = self.as_address(self.expr(args[0]))
        ty = self.ty(dtypes.dtype_of(node), node)
        cmp = self.cast_to(self.expr(args[1]), ty) if op == "Cas" else None
        value = self.cast_to(self.expr(args[-1]), ty)
        dst = self.builder.reg(ty)
        self.builder.emit("Atom", site=self.site(node, op_name=_op_name(node)), op=op, ty=ty, dst=dst, addr=addr,
                          space="Generic", value=value, cmp=cmp, sem="Relaxed", scope="Gpu", ftz=False)
        return dst

    def call_tirx_cuda_atomic_add(self: "Lowerer", node: Any) -> pb.Operand:
        return self.atomic(node, "Add")

    def call_tirx_cuda_atomic_cas(self: "Lowerer", node: Any) -> pb.Operand:
        return self.atomic(node, "Cas")

    def call_tirx_cuda_cvta_generic_to_shared(self: "Lowerer", node: Any) -> pb.Operand:
        src = self.as_address(self.expr(node.args[0]))
        dst = self.builder.reg(self.ty(dtypes.dtype_of(node) or "uint32", node))
        self.builder.emit("Cvta", dst=dst, src=src, space="Shared", to_generic=False)
        return dst

    call_tirx_cuda_smem_addr_from_uint64 = call_tirx_cuda_cvta_generic_to_shared

    def convert_through_memory(self: "Lowerer", node: Any, src_ty: pb.Ty, dst_ty: pb.Ty) -> None:
        dst_ptr, src_ptr = (self.as_address(self.expr(a)) for a in node.args)
        site = self.site(node, op_name=_op_name(node))
        loaded = self.builder.reg(src_ty)
        self.builder.emit("LoadAddr", site=site, ty=src_ty, dst=loaded, addr=src_ptr, space="Generic",
                          sem="Weak", scope="Gpu", mods=pb.mem_mods())
        converted = self.builder.reg(dst_ty)
        op = self.builder.op(pb.OpKey(_op_name(node) + ".value"))
        self.builder.emit("Ptx", site=site, op=op, dsts=[converted], srcs=[loaded], pred=None, keep_dst=False)
        self.builder.emit("StoreAddr", site=site, ty=dst_ty, addr=dst_ptr, space="Generic", value=converted,
                          sem="Weak", scope="Gpu", mods=pb.mem_mods())

    def call_tirx_cuda_float22half2(self: "Lowerer", node: Any) -> None:
        self.convert_through_memory(node, pb.Ty("F32", 2), pb.Ty("F16", 2))

    def call_tirx_cuda_float8tohalf8(self: "Lowerer", node: Any) -> None:
        self.convert_through_memory(node, pb.Ty("F32", 8), pb.Ty("F16", 8))

    def call_tirx_cuda_half8tofloat8(self: "Lowerer", node: Any) -> None:
        self.convert_through_memory(node, pb.Ty("F16", 8), pb.Ty("F32", 8))

    # -- reviewed CUDA helpers ------------------------------------------------
    def call_tirx_cuda_func_call(self: "Lowerer", node: Any) -> pb.Operand | None:
        args = list(node.args)
        name = _string(args[0]) if args else None
        if name == "tvm_builtin_pointer_offset":
            return self.pointer_offset(node, args[1], args[2])
        args = args[1:]
        source = ""
        if args and _string(args[-1]) is not None:
            source = _string(args[-1]) or ""
            args = args[:-1]
        if name is not None and name not in builtins.PURE_FUNC_CALLS:
            if self.single_asm_helper(node, source, args):
                return None
        if name is None or name not in builtins.PURE_FUNC_CALLS:
            raise _Unsupported(node, f"cuda.func_call of unreviewed or effectful helper {name!r}")
        digest = hashlib.sha256("".join(source.split()).encode()).hexdigest()[:16]
        return self.pure_helper(node, f"tirx.cuda.func_call.{name}", "v*", args=args,
                                extra_mods=(f"source_sha256={digest}",))

    def single_asm_helper(self: "Lowerer", node: Any, source: str, args: list[Any]) -> bool:
        """Lower an effectful helper whose body is exactly one reviewed PTX statement.

        The semantics come from the helper's own asm text (tensormap replace and
        tensormap proxy fences), so a modified body cannot be accepted by name.
        """
        if source.count("asm") != 1:
            return False
        site = self.site(node, op_name="tirx.cuda.func_call")
        match = _ASM_REPLACE.search(source)
        if match and len(args) == 2:
            field = {"global_address": "GlobalAddress", "global_dim": "GlobalDim",
                     "global_stride": "GlobalStride"}[match.group(1)]
            value_ty = pb.Ty("U64") if match.group(2) == "64" else pb.Ty("U32")
            ordinal = int(match.group(3)) if match.group(3) is not None else None
            tmap = self.as_address(self.expr(args[0]))
            value = self.convert(self.expr(args[1]), value_ty)
            self.builder.emit("TensorMapReplace", site=site, tmap=tmap, space="Global", field=field, ord=ordinal,
                              value=value)
            return True
        match = _ASM_RELEASE.search(source)
        if match and not args:
            self.builder.emit("Fence", site=site, kind="TensormapRelease", sem="Release",
                              scope=_SCOPES[match.group(1)])
            return True
        match = _ASM_ACQUIRE.search(source)
        if match and len(args) == 1:
            addr = self.as_address(self.expr(args[0]))
            self.builder.emit("Fence", site=site, kind={"TensormapAcquire": {"addr": pb.opnd(addr), "space": "Generic"}},
                              sem="Acquire", scope=_SCOPES[match.group(1)])
            return True
        return False

    def call_tirx_ptx_addr(self: "Lowerer", node: Any) -> pb.Operand:
        """``T.ptx.addr(base, byte_offset)``: an address operand ``base + byte_offset``."""
        base = self.as_address(self.expr(node.args[0]))
        offset = self.reinterpret(self.cast_to(self.expr(node.args[1]), "int64"), pb.Ty("U64"))
        return self.binary("Add", pb.Ty("U64"), base, offset)

    def pointer_offset(self: "Lowerer", node: Any, pointer: Any, offset: Any) -> pb.Operand:
        """TVM's ``tvm_builtin_pointer_offset(T* ptr, int offset)`` = ``ptr + offset`` elements."""
        if type_key(pointer) == "ir.Call" and _op_name(pointer) == "tirx.address_of" and \
                type_key(pointer.args[0]) == "ir.TensorLoad":
            elem = dtypes.dtype_of(pointer.args[0])
        else:
            element = getattr(getattr(pointer, "ty", None), "element_type", None)
            elem = str(getattr(element, "dtype", "")) if element is not None else ""
        if not elem or elem == "void":
            raise _Unsupported(node, "tvm_builtin_pointer_offset on an untyped pointer")
        base = self.as_address(self.expr(pointer))
        return self.binary("Add", pb.Ty("U64"), base, self.element_bytes(elem, self.expr(offset)))

    # -- wait_until -----------------------------------------------------------
    def call_tirx_cuda_wait_until(self: "Lowerer", node: Any) -> None:
        args = list(node.args)
        if len(args) < 3:
            raise _Unsupported(node, "wait_until needs (dst, ptr, predicate, ...)")
        dst_node, ptr_node, pred_node = args[:3]
        strings = [_string(a) for a in args[3:]]
        scope = strings[0] if len(strings) > 0 and strings[0] else "gpu"
        space_name = strings[1] if len(strings) > 1 and strings[1] else "global"
        width = strings[2] if len(strings) > 2 and strings[2] else ""
        if type_key(dst_node) != "ir.TensorLoad" or not isinstance(self.ref_of(dst_node.source), RegArray):
            raise _Unsupported(node, "wait_until destination must be a promoted local scalar")
        dst, write_back = self.lvalue_target(dst_node)
        if write_back is not None:
            raise _Unsupported(node, "wait_until destination must have a constant index")
        addr = self.as_address(self.expr(ptr_node))
        ty = pb.Ty.from_ptx(width) if width else self.builder.reg_ty(dst)
        pred_index, captures = self.lower_predicate(pred_node, dst)
        space = {"global": "Global", "shared": "Shared"}.get(space_name, "Generic")
        self.mark_sync_words(ptr_node)
        self.builder.emit("WaitUntil", site=self.site(node, op_name="tirx.cuda.wait_until"), dst=dst, addr=addr,
                          ty=ty, space=space, sem="Acquire", scope={"cta": "Cta", "cluster": "Cluster",
                                                                     "gpu": "Gpu", "sys": "Sys"}[scope],
                          pred=pred_index, captures=captures)
        return None

    def mark_sync_words(self: "Lowerer", ptr_node: Any) -> None:
        """``BufferDecl::sync_words`` hint for the buffer the wait polls."""
        if type_key(ptr_node) == "ir.Call" and _op_name(ptr_node) == "tirx.address_of":
            target = ptr_node.args[0]
            var = target.source if type_key(target) == "ir.TensorLoad" else target
            ref = self.refs.get(int(var.__chandle__()))
            buf = getattr(ref, "buf", None)
            if buf is not None:
                decl = self.builder.program.buffers[buf]
                while decl.view_of is not None:
                    buf = decl.view_of
                    decl = self.builder.program.buffers[buf]
                from dataclasses import replace

                self.builder.program.buffers[buf] = replace(decl, sync_words=True)

    def lower_predicate(self: "Lowerer", pred_node: Any, dst: pb.Reg) -> tuple[int, list[pb.Reg]]:
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
                if isinstance(value, pb.Reg) and value.index < first_reg and value.index not in defined \
                        and value not in captures:
                    captures.append(value)
            for reg in instr.writes():
                defined.add(reg.index)
        self.pending_preds.append((code, sites, arg, result, reads_memory))
        return len(self.pending_preds) - 1, captures

    def is_pure(self: "Lowerer", node: Any) -> bool:
        """No memory reads of non-register buffers and no calls with effects."""
        import tvm
        from tvm_ffi import structural_visit

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
            if not (name in builtins.UNARY_OPS or name in _PURE_STRUCTURAL
                    or (helper is not None and helper.kind == "pure")):
                pure = False
            visitor.default_visit(sub)

        structural_visit(node, [(tvm.ir.TensorLoad, on_load), (tvm.ir.Call, on_call)])
        return pure


__all__ = ["CallsMixin"]
