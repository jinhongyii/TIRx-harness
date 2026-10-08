"""Kernels the v2 lowering tests use, copied verbatim from legacy test modules.

The legacy modules are cut or deleted at step 5 (test migration); these copies
keep tests/numsim/v2/test_lowering_*.py free of them. Each block names its source.
"""

# ruff: noqa: E501

from tests.numsim.support.kernels import _TEST_WARP_GEMM_B_FRAG, _TEST_WARP_GEMM_D_FRAG
from tvm.ir.type import PointerType, PrimType
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import S, TileLayout, laneid, tid_in_wg, wg_local_layout
import tirx_kernels.tirx_lite as txl
import tvm


# From numsim/runtime/test_tile_unary_codegen.py
@T.prim_func
def tile_unary_unknown_config(output: T.Buffer((32,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    value: T.f32[1]
    result: T.f32[1]
    value[0] = T.float32(1)
    Tx.exp(result, value, undocumented_mode=True)
    output[lane] = result[0]


# From numsim/integration/test_warp_gemm_artifact.py
_WRONG_WARP_GEMM_A_FRAG = TileLayout(S[(2, 8, 2, 4, 2) : (4, 4 @ laneid, 2, 1 @ laneid, 1)])


# From numsim/integration/test_warp_gemm_artifact.py
@T.prim_func
def _warp_gemm_wrong_a_fragment_layout():
    T.device_entry()
    _warp = T.warp_id([1])
    _lane = T.lane_id([32])
    left = T.alloc_buffer((16, 16), "bfloat16", scope="local", layout=_WRONG_WARP_GEMM_A_FRAG)
    right = T.alloc_buffer((16, 8), "bfloat16", scope="local", layout=_TEST_WARP_GEMM_B_FRAG)
    accumulator = T.alloc_buffer((16, 8), "float32", scope="local", layout=_TEST_WARP_GEMM_D_FRAG)
    destination = T.alloc_buffer((16, 8), "float32", scope="local", layout=_TEST_WARP_GEMM_D_FRAG)
    Tx.warp.gemm(
        destination,
        left,
        right,
        accumulator,
        transpose_A=False,
        transpose_B=False,
        alpha=1.0,
        beta=0.0,
    )


# From numsim/runtime/test_tile_general_semantics.py
@T.prim_func
def right_aligned_elementwise_broadcast(
    source: T.Buffer((32, 2, 4), "float32"),
    column: T.Buffer((32, 4), "float32"),
    row: T.Buffer((32, 2), "float32"),
    half_column: T.Buffer((32, 4), "float16"),
    output: T.Buffer((32, 4, 2, 4), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    source_local = T.alloc_buffer((2, 4), "float32", scope="local")
    column_local = T.alloc_buffer((4,), "float32", scope="local")
    row_local = T.alloc_buffer((2, 1), "float32", scope="local")
    half_column_local = T.alloc_buffer((4,), "float16", scope="local")
    add_result = T.alloc_buffer((2, 4), "float32", scope="local")
    sub_result = T.alloc_buffer((2, 4), "float32", scope="local")
    mul_result = T.alloc_buffer((2, 4), "float32", scope="local")
    cast_result = T.alloc_buffer((2, 4), "float32", scope="local")
    for row_index in T.serial(2):
        row_local[row_index, 0] = row[lane, row_index]
        for column_index in T.serial(4):
            source_local[row_index, column_index] = source[lane, row_index, column_index]
    for column_index in T.serial(4):
        column_local[column_index] = column[lane, column_index]
        half_column_local[column_index] = half_column[lane, column_index]

    Tx.add(add_result[:, :], source_local[:, :], column_local[:])
    Tx.sub(sub_result[:, :], source_local[:, :], row_local[:, :])
    Tx.mul(mul_result[:, :], source_local[:, :], row_local[:, :])
    Tx.cast(cast_result[:, :], half_column_local[:])

    for row_index in T.serial(2):
        for column_index in T.serial(4):
            output[lane, 0, row_index, column_index] = add_result[row_index, column_index]
            output[lane, 1, row_index, column_index] = sub_result[row_index, column_index]
            output[lane, 2, row_index, column_index] = mul_result[row_index, column_index]
            output[lane, 3, row_index, column_index] = cast_result[row_index, column_index]


# From numsim/integration/test_opaque_helper_artifact.py
_SMEM_DESC_MAKE_LO_UNIFORM_SOURCE = r"""
__forceinline__ __device__ void smem_desc_make_lo_uniform(uint64_t* desc) {
    SmemDescriptor* d = reinterpret_cast<SmemDescriptor*>(desc);
    d->lo = __shfl_sync(0xffffffff, d->lo, 0);
}
"""


# From numsim/integration/test_opaque_helper_artifact.py
@T.prim_func
def smem_descriptor_make_lo_uniform_helper(
    descriptors: T.Buffer((32,), "uint64"), output: T.Buffer((32,), "uint64")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    descriptor = T.alloc_local((1,), "uint64")
    descriptor[0] = descriptors[lane]
    T.evaluate(
        T.cuda.func_call(
            "smem_desc_make_lo_uniform",
            T.address_of(descriptor[0]),
            source_code=_SMEM_DESC_MAKE_LO_UNIFORM_SOURCE,
            return_type="void",
        )
    )
    output[lane] = descriptor[0]


# From numsim/runtime/test_ptx_spdecompress.py
@T.prim_func
def ptx_spdecompress_b8_b4_2_4_x2(
    metadata: T.Buffer((32,), "uint32"),
    compressed: T.Buffer((32,), "uint32"),
    output: T.Buffer((2, 32), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx["spdecompress.b8.b4.sp::2:4.x2"](
        output[0, lane], output[1, lane], metadata[lane], compressed[lane]
    )


# From numsim/integration/test_host_prelude.py
@T.prim_func
def host_encoded_dynamic_integer_tensor_map(
    source: T.Buffer((32,), "uint8"), output: T.Buffer((16,), "uint8"), delta: T.int32
):
    tensor_map: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
    T.call_packed(
        "runtime.cuTensorMapEncodeTiled",
        tensor_map,
        "uint8",
        1,
        source.data,
        T.truncdiv(delta, T.int32(2)) + T.int32(35),
        T.floordiv(delta, T.int32(2)) + T.int32(20),
        1,
        0,
        0,
        0,
        0,
    )
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((16,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.evaluate(
            T.ptx[
                "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
            ](T.address_of(shared[0]), T.address_of(tensor_map), 16, T.address_of(barrier[0]))
        )
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barrier[0]), 16)
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.cuda.warp_sync()
    if lane < 16:
        output[lane] = shared[lane]


# From numsim/runtime/test_atomic_f32_noftz.py
def atomic_kernel(kind, width, space, *, noftz=True, offset=0, race=False):
    mnemonic = "red" if kind == "red" else "atom"
    tokens = [mnemonic, "relaxed", "cta", space, "add", "noftz" if noftz else ""]
    tokens += [f"v{width}" if width > 1 else "", "f32"]
    spelling = ".".join(token for token in tokens if token)
    pointer = "shared" if space.startswith("shared") else "destination"
    arguments = [f"{pointer}.ptr_to([lane * {width} + {offset}])"]
    values = [f"value[lane * {width} + {i}]" for i in range(width)]
    if kind == "atom":
        arguments.insert(0, ", ".join(f"old[{i}]" for i in range(width)))
    arguments.extend(values)
    if width > 1:
        arguments.append("pred=lane % 2 == 0")
        if kind == "atom":
            arguments.append("preserve_dst=True")
    return tvm.script.from_source(
        f"""
@T.prim_func
def atomic(destination: T.Buffer((128,), "float32"), value: T.Buffer((128,), "float32"),
           returned: T.Buffer((128,), "float32")):
    T.device_entry()
    warp = T.warp_id([{2 if race else 1}])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((128,), "float32", scope="shared", align=16)
    old = T.alloc_local(({width},), "float32")
    if warp == 0:
        for i in T.serial({width}):
            shared[lane * {width} + i] = destination[lane * {width} + i]
            old[i] = T.float32(-1)
        T.ptx["{spelling}"]({", ".join(arguments)})
        for i in T.serial({width}):
            returned[lane * {width} + i] = old[i]
            {"destination[lane * " + str(width) + " + i] = shared[lane * " + str(width) + " + i]" if pointer == "shared" else "T.evaluate(0)"}
    else:
        destination[lane * {width}] = T.float32(3)
""",
        {"T": T},
    )


# From analysis_tools/synccheck/test_reported_tool_regressions.py
@txl.kernel(warps=1, arch="sm_100a", grid=1)
def packed_bf16_vector_reduction(out: txl.gptr[txl.bf16]):
    with txl.If(txl.thread_id() == 0), txl.Then():
        txl.ptx["red.global.v2.bf16x2.add.noftz"](
            out.ptr_to([0]), txl.uint32(0x3F803F80), txl.uint32(0x3F803F80)
        )


# From numsim/runtime/test_mbarrier_report.py
def report_kernel(pattern, *, cluster=False, layout=1, tensor=False):
    family = ".tensor.1d" if tensor else ""
    queries = []
    for parity in (False, True):
        for action, hint in (("test_wait", False), ("try_wait", False), ("try_wait", True)):
            for value in (False, True):
                slot = len(queries)
                operands = "ready[0], report[0], " + ("value[0], " if value else "")
                operands += "barrier.ptr_to([0]), " + ("T.uint32(phase)" if parity else "state[0]")
                operands += ", T.uint32(1)" if hint else ""
                suffix = ".parity" if parity else ""
                queries.append(
                    f'T.ptx["mbarrier.{action}{suffix}.phase_type::primary.shared.b64"]({operands})\n'
                    f"out[phase, {slot}] = ready[0] + 2 * report[0]"
                    + (" + T.Cast('uint32', value[0]) * 4" if value else "")
                )
    query_source = "\n            ".join(q.replace("\n", "\n            ") for q in queries)
    report = "disabled" if pattern == "disabled" else f"validity::{pattern}"
    parameter = "input_map: T.TensorMap()" if tensor else 'source: T.Buffer((32,), "uint8")'
    source_operands = (
        "T.address_of(input_map), phase * 16"
        if tensor
        else "source.ptr_to([phase * 16]), T.uint32(16)"
    )
    return tvm.script.from_source(
        f"""
@T.prim_func
def kernel({parameter}, out: T.Buffer((2, 12), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    shared = T.alloc_buffer((16,), "uint8", scope="shared", align=128)
    state = T.alloc_local((1,), "uint64")
    ready = T.alloc_local((1,), "uint32")
    report = T.alloc_local((1,), "uint32")
    value = T.alloc_local((1,), "uint8")
    if lane == 0:
        T.ptx["mbarrier.init.layout::v{layout}.shared.b64"](barrier.ptr_to([0]), 1)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    for phase in T.serial(2):
        if lane == 0:
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(state[0], barrier.ptr_to([0]), 16)
            T.ptx["cp.async.bulk{family}.shared::{"cluster" if cluster else "cta"}.global.mbarrier::complete_tx::bytes.mbarrier::report::{report}"](
                shared.ptr_to([0]), {source_operands}, barrier.ptr_to([0]))
            T.cuda.mbarrier_wait(barrier.ptr_to([0]), phase)
            {query_source}
""",
        {"T": T},
    )


# From numsim/runtime/test_scalar_control.py
@T.prim_func
def pointer_conversions_and_descriptor(
    source: T.Buffer((32, 8), "float32"),
    half: T.Buffer((32, 8), "float16"),
    roundtrip: T.Buffer((32, 8), "float32"),
    descriptor: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.cuda.float8tohalf8(T.address_of(source[lane, 0]), T.address_of(half[lane, 0]))
    T.cuda.half8tofloat8(T.address_of(half[lane, 0]), T.address_of(roundtrip[lane, 0]))
    T.cuda.float22half2(T.address_of(half[lane, 0]), T.address_of(source[lane, 0]))
    T.cuda.runtime_instr_desc(T.address_of(descriptor[lane]), lane % 4)


# From numsim/runtime/test_scalar_control.py
@T.prim_func
def canonical_mapa(output: T.Buffer((32,), "int32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    _cta = T.cta_id_in_cluster([1])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32,), "int32", scope="shared")
    shared[lane] = lane + 37
    T.cuda.cta_sync()
    mapped = T.local_scalar("uint64")
    T.ptx.mapa.shared__cluster.u64(mapped, T.address_of(shared[0]), T.uint32(0))
    mapped_ptr: T.let[
        T.Var(name="canonical_mapped_ptr", ty=PointerType(PrimType("int32"), "shared"))
    ] = T.reinterpret(PointerType(PrimType("int32"), "shared"), mapped)
    mapped_buffer = T.decl_buffer((1,), "int32", scope="shared", data=mapped_ptr)
    output[lane] = mapped_buffer[0]


# From numsim/runtime/test_tile_owner_transport.py
def _transposed_warpgroup_layout() -> TileLayout:
    return TileLayout(S[(32, 4, 2) : (1 @ tid_in_wg, 32 @ tid_in_wg, 1)])


# From numsim/runtime/test_tile_owner_transport.py
@T.prim_func
def _copy_cross_warp_owner_remap(
    source: T.Buffer((128, 2), "float32"), output: T.Buffer((128, 2), "float32")
):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    _lane = T.lane_id([32])
    source_storage = T.alloc_buffer((2,), "float32", scope="local")
    destination_storage = T.alloc_buffer((2,), "float32", scope="local")
    source_view = source_storage.view(128, 2, layout=wg_local_layout(2))
    destination_view = destination_storage.view(128, 2, layout=_transposed_warpgroup_layout())
    Tx.wg.copy(source_view[:, :], source[:, :])
    Tx.wg.copy(destination_view[:, :], source_view[:, :])
    Tx.wg.copy(output[:, :], destination_view[:, :])


# From numsim/runtime/test_matrix_instruction_codegen.py
@T.prim_func
def mma_fragment_fill_and_store(
    filled: T.Buffer((2, 32, 8), "float32"), stored: T.Buffer((2, 16, 16), "float32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    current_fragment = T.alloc_local((8,), "float32")
    legacy_fragment = T.alloc_local((8,), "float32")
    current_fill = T.alloc_local((8,), "float32")
    legacy_fill = T.alloc_local((8,), "float32")
    for local_id in T.serial(8):
        current_fragment[local_id] = T.cast(lane * 100 + local_id, "float32")
        legacy_fragment[local_id] = T.cast(10000 + lane * 100 + local_id, "float32")
        current_fill[local_id] = T.float32(1)
        legacy_fill[local_id] = T.float32(2)

    T.evaluate(T.cuda.mma_fill(8, current_fill.data, 0, dtype="float32"))
    T.evaluate(T.cuda.mma_fill_legacy(8, legacy_fill.data, 0, dtype="float32"))
    T.evaluate(
        T.cuda.mma_store(
            16, 16, stored.ptr_to([0, 0, 0]), current_fragment.data, 0, 16, dtype="float32"
        )
    )
    T.evaluate(
        T.cuda.mma_store_legacy(
            16, 16, stored.ptr_to([1, 0, 0]), legacy_fragment.data, 0, 16, dtype="float32"
        )
    )
    for local_id in T.serial(8):
        filled[0, lane, local_id] = current_fill[local_id]
        filled[1, lane, local_id] = legacy_fill[local_id]


# From numsim/integration/test_topology_artifact.py
@T.prim_func
def too_many_warps(output: T.Buffer((1,), "int32")):
    T.device_entry()
    warp = T.warp_id([33])
    lane = T.lane_id([32])
    if (warp == 0) and (lane == 0):
        output[0] = 1


# From numsim/integration/test_topology_artifact.py
@T.prim_func
def too_many_cluster_ctas(output: T.Buffer((1,), "int32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([65])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if (cta == 0) and (lane == 0):
        output[0] = 1
