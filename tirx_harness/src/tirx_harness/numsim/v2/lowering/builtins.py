"""Non-table builtin facts shared by the escape analysis and call lowering.

``HELPERS`` lists every accepted ``tirx.cuda.*`` / ``tirx.*`` helper with its
operand roles (one letter per argument; a trailing ``*`` repeats the
previous letter):

* ``v`` value operand
* ``o`` out-parameter written through ``address_of(lvalue)``
* ``x`` in/out parameter through ``address_of(lvalue)``
* ``s`` string literal (becomes an ``OpKey`` modifier)
* ``a`` auto: string literal -> modifier, else value

and its lowering ``kind``:

* ``pure`` -> ``Instr::Ptx`` with ``OpKey{name, mods}`` (oplib implements it);
* ``special`` -> a dedicated lowering in ``calls.py`` (``call_<name>``).

Anything not listed (and not a TVM PTX table op, a ``UNARY_OPS`` math op or a
structural op) fails closed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Helper:
    roles: str
    kind: str = "pure"          # pure | special


def _p(roles: str = "a*") -> Helper:
    return Helper(roles, "pure")


def _s(roles: str = "a*") -> Helper:
    return Helper(roles, "special")


HELPERS: dict[str, Helper] = {
    # synchronization (special: dedicated Instr variants)
    "tirx.cuda.mbarrier_wait": _s("vv"),
    "tirx.cuda.mbarrier_wait_acquire_cluster": _s("vv"),
    "tirx.cuda.cta_sync": _s(""),
    "tirx.cuda.warp_sync": _s("a*"),
    "tirx.cuda.warpgroup_sync": _s("v"),
    "tirx.cuda.cluster_sync": _s(""),
    "tirx.cuda.grid_sync": _s(""),
    "tirx.cuda.syncthreads_and": _s("v"),
    "tirx.cuda.syncthreads_or": _s("v"),
    "tirx.tvm_storage_sync": _s("a*"),
    "tirx.cuda.thread_fence": _s(""),
    # warp collectives
    "tirx.cuda.elect_sync": _s("a*"),
    "tirx.cuda.__shfl_sync": _s("vvvv"),
    "tirx.cuda.__shfl_up_sync": _s("vvvv"),
    "tirx.cuda.__shfl_down_sync": _s("vvvv"),
    "tirx.cuda.__shfl_xor_sync": _s("vvvv"),
    "tirx.tvm_warp_shuffle": _s("vvvvv"),
    "tirx.tvm_warp_shuffle_up": _s("vvvvv"),
    "tirx.tvm_warp_shuffle_down": _s("vvvvv"),
    "tirx.tvm_warp_shuffle_xor": _s("vvvvv"),
    "tirx.cuda.ballot_sync": _s("vv"),
    "tirx.cuda.any_sync": _s("vv"),
    "tirx.cuda.__activemask": _s(""),
    "tirx.tvm_warp_activemask": _s(""),
    "tirx.cuda.reduce_add_sync_u32": _s("vv"),
    "tirx.cuda.reduce_min_sync_u32": _s("vv"),
    # memory / atomics / addresses
    "tirx.cuda.ldg": _s("va"),
    "tirx.cuda.atomic_add": _s("vv"),
    "tirx.cuda.atomic_cas": _s("vvv"),
    "tirx.cuda.cvta_generic_to_shared": _s("v"),
    "tirx.cuda.smem_addr_from_uint64": _s("v"),
    "tirx.cuda.float22half2": _s("vv"),
    "tirx.cuda.float8tohalf8": _s("vv"),
    "tirx.cuda.half8tofloat8": _s("vv"),
    # special registers / control / annotations
    "tirx.cuda.thread_rank": _s(""),
    "tirx.cuda.clock64": _s(""),
    "tirx.cuda.mov_sreg": _s("va"),
    "tirx.cuda.nano_sleep": _s("v"),
    "tirx.cuda.printf": _s("a*"),
    "tirx.cuda.iket_mark": _s("a*"),
    "tirx.cuda.iket_range_start": _s("a*"),
    "tirx.cuda.iket_range_end": _s("a*"),
    "tirx.cuda.iket_range_push": _s("a*"),
    "tirx.cuda.iket_range_pop": _s(""),
    "tirx.cuda.iket_sentinel_token": _s("a*"),
    "tirx.cuda.iket_official_event": _s("a*"),
    "tirx.cuda.trap_when_assert_failed": _s("v"),
    "tirx.cuda.wait_until": _s("a*"),
    "tirx.cuda.func_call": _s("a*"),
    # pure value helpers -> Instr::Ptx
    "tirx.cuda.get_tmem_addr": _p("vvv"),
    "tirx.cuda.sm100_2sm_leader_smem_addr": _p("v"),
    "tirx.cuda.tcgen05_encode_matrix_descriptor": _p("oa*"),
    "tirx.cuda.tcgen05_encode_instr_descriptor": _p("oa*"),
    "tirx.cuda.tcgen05_encode_instr_descriptor_block_scaled": _p("oa*"),
    "tirx.cuda.runtime_instr_desc": _p("xv"),
    "tirx.cuda.make_float2": _p("vv"),
    "tirx.cuda.float2_x": _p("v"),
    "tirx.cuda.float2_y": _p("v"),
    "tirx.cuda.uint_as_float": _p("v"),
    "tirx.cuda.float_as_uint": _p("v"),
    "tirx.cuda.ffs_u32": _p("v"),
    "tirx.cuda.float22bfloat162_rn": _p("vv"),
    "tirx.cuda.float22bfloat162_rn_from_float2": _p("v"),
    "tirx.cuda.bfloat1622float2": _p("v"),
    "tirx.cuda.hmin2": _p("vv"),
    "tirx.cuda.hmax2": _p("vv"),
    "tirx.cuda.fmul2_rn": _p("vv"),
    "tirx.cuda.fadd2_rn": _p("vv"),
    "tirx.cuda.fdividef": _p("vv"),
    "tirx.cuda.fp8x4_e4m3_from_float4": _p("vvvv"),
    "tirx.cuda.half2float": _p("v"),
    "tirx.cuda.bfloat162float": _p("v"),
    "tirx.log1p": _p("v"),
    "tirx.sigmoid": _p("v"),
    "tirx.exp10": _p("v"),
    "tirx.log10": _p("v"),
    "tirx.erf": _p("v"),
    "tirx.nearbyint": _p("v"),
}

# tirx.* math ops lowered to ``Instr::Unary`` (``UnOp``).
UNARY_OPS: dict[str, str] = {
    "tirx.exp": "Exp", "tirx.exp2": "Exp2", "tirx.log": "Log", "tirx.log2": "Log2", "prim.log2": "Log2",
    "tirx.sqrt": "Sqrt", "tirx.rsqrt": "Rsqrt", "tirx.fabs": "Abs", "tirx.sin": "Sin", "tirx.cos": "Cos",
    "tirx.tanh": "Tanh", "tirx.popcount": "Popcount", "tirx.clz": "Clz", "tirx.floor": "Floor",
    "tirx.ceil": "Ceil", "tirx.round": "Round", "tirx.trunc": "Trunc", "tirx.isnan": "IsNan",
    "tirx.isinf": "IsInf", "tirx.isfinite": "IsFinite",
}

# CUDA helpers recognized by name and normalized-source hash (legacy
# emit/cuda_helper.rs). Pure ones become ``Ptx`` ops; effectful ones must be
# lowered to primitives (not done yet: they fail closed).
PURE_FUNC_CALLS = frozenset(
    {
        "gdn_lg2_approx_ftz", "flashkda_rsqrtf", "flashkda_tanh_approx", "flashkda_fmaf_rn",
        "tvm_builtin_fma_scale_sub_f32x2", "combine_int_frac_ex2", "shl_u32_clamp",
        "smem_desc_add_16B_offset",
    }
)


def role(roles: str, position: int) -> str:
    if roles.endswith("*"):
        fixed = roles[:-2]
        return fixed[position] if position < len(fixed) else roles[-2]
    return roles[position] if position < len(roles) else "?"


__all__ = ["HELPERS", "Helper", "PURE_FUNC_CALLS", "UNARY_OPS", "role"]
