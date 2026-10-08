"""Checker verdicts that only the interpreter can decide (``needs_kernel`` rows).

Replaces the eight ``needs_kernel`` rows of
``scripts/numsim-v2/coverage/other_b.tsv``: their contract cannot be written
as contract events because the verdict depends on interpreter behaviour
(``griddepcontrol`` assumption, multicast ``ctaMask`` resolution, the
readonly-bytes overlap check, ``mapa``/u32 address arithmetic, tcgen05 MMA
read footprints and their ``zero_mask`` elision, and the TMA
``override::global_address`` destination). Each test runs the legacy kernel
through the public v2 surface.
"""

from __future__ import annotations

import numpy as np
import pytest
import tvm
from tvm import tirx
from tvm.ir import Call, Expr
from tvm.script import tirx as T
from tvm_ffi import structural_map, structural_walk

from tirx_harness import numsim
from tirx_harness.numsim import v2

from ._runnable import assert_clean, no_spec, race_access_pairs, requires_v2_engine
from ._tcgen_kernels import SPARSE_B16_CASES, lut_b_case, sparse_float_case

pytestmark = requires_v2_engine

CHECKERS = ("synccheck", "racecheck")


def _check(checker, kernel, inputs):
    return getattr(v2, checker)(kernel, inputs)


def _no_spec_unlisted(what: str):
    """Like :func:`no_spec`, for a semantic that no new spec mentions and that
    test-migration.md does not list yet as a numbered no-spec item."""

    return pytest.mark.xfail(
        strict=False,
        reason=f"no spec: {what} (not yet a test-migration.md no-spec item); ruling needed",
    )


# test-migration.md no-spec item 11: CTA 0's qualifier-less
# ``mbarrier.arrive.expect_tx.shared::cluster`` on CTA 1's barrier defaults to
# ``.release.cta``; the new racecheck reports ``scope_mismatch`` against CTA 1's
# ``.cta`` acquire where legacy was clean.
_REMOTE_ARRIVE_DEFAULT_CTA_SCOPE = no_spec(
    11, "default .cta scope of a qualifier-less remote mbarrier arrive (racecheck scope_mismatch)"
)


# -- griddepcontrol.wait --------------------------------------------------------


@T.prim_func
def native_public_griddep_wait_kernel(output: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.griddepcontrol.wait()
    if lane == 0:
        output[0] = 1


def test_public_synccheck_assumes_external_grid_dependency_is_satisfied():
    """Replaces ``tests/analysis_tools/racecheck/test_native_public_api.py::test_public_synccheck_assumes_external_grid_dependency_is_satisfied``.

    ``griddepcontrol.wait`` with no prerequisite grid in the invocation is
    assumed satisfied: Synccheck is clean with no findings. The assumption is
    stated only in lowering-inventory.md (test-migration.md no-spec item 19);
    the v2 interpreter keeps it (``griddepcontrol`` is a no-op).
    """

    report = v2.synccheck(native_public_griddep_wait_kernel, {"output": np.zeros((1,), dtype=np.int32)})
    assert_clean(report)


# -- mbarrier multicast ctaMask outside the cluster -------------------------------


def _multicast_outside_cluster_kernel(ctas=20, mask=1 << 20):
    """``multicast_barrier_kernel("arrive", mask=1 << 20)`` from the legacy test."""

    initial = 2
    instruction = "mbarrier.arrive.shared::cluster.multicast::cluster::32b.b64"
    args = ", ".join(["barrier.ptr_to([0])", "T.uint32(1)", f"T.uint32({mask})"])
    return tvm.script.from_source(
        f"""
@T.prim_func
def multicast_barrier(out: T.Buffer(({ctas}, 2), "uint32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([{ctas}])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    state = T.alloc_local((1,), "uint64")
    ready = T.alloc_local((1,), "uint32")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), {initial})
        T.evaluate(0)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if cta == 0 and lane < 1:
        T.ptx["{instruction}"]({args})
    T.cuda.cluster_sync()
    if lane == 0:
        if cta == 0 or cta == {ctas - 1}:
            T.evaluate(0)
            T.ptx.mbarrier.arrive.shared.b64(state[0], barrier.ptr_to([0]), T.uint32(1))
        else:
            T.evaluate(0)
            T.ptx.mbarrier.arrive.shared.b64(state[0], barrier.ptr_to([0]), T.uint32({initial}))
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
        T.ptx.mbarrier.test_wait.shared.b64(ready[0], barrier.ptr_to([0]), state[0])
        out[cta, 0] = ready[0]
        T.ptx.mbarrier.arrive.shared.b64(state[0], barrier.ptr_to([0]),
            T.uint32({initial}) if cta == 0 or cta == {ctas - 1} else T.uint32({initial}))
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 1)
        T.ptx.mbarrier.test_wait.shared.b64(ready[0], barrier.ptr_to([0]), state[0])
        out[cta, 1] = ready[0]
    T.cuda.cluster_sync()
""",
        {"T": T},
    )


@_no_spec_unlisted(
    "a multicast ctaMask bit naming a rank outside the cluster; the v2 interpreter drops it "
    "(ranks_of) and the never-completed barrier ends incomplete divergent_block"
)
@pytest.mark.parametrize("checker", CHECKERS)
def test_mbarrier_multicast_outside_cluster(checker):
    """Replaces ``tests/numsim/runtime/test_mbarrier_multicast.py::test_mbarrier_multicast_outside_cluster`` (its checker loop as a param).

    A 32-bit ``ctaMask`` selecting rank 20 of a 20-CTA cluster: verdict
    ``error`` and the report says the target is "outside the cluster".
    No new spec says how an out-of-cluster ``ctaMask`` bit is handled
    (sync-semantics.md only says "multicast to the ``ctaMask`` CTAs"), and no
    numbered no-spec item covers it yet.
    """

    report = _check(checker, _multicast_outside_cluster_kernel(), {"out": np.zeros((20, 2), np.uint32)})
    assert report.verdict == "error", report.format()
    assert "outside the cluster" in report.format(), report.format()


# -- cp_mask bulk store over readonly-read bytes ---------------------------------


def _cp_mask_readonly_kernel(before):
    @T.prim_func
    def kernel(
        destination: T.Buffer((16,), "uint8"),
        byte_mask: T.uint32,
        output: T.Buffer((1,), "uint32"),
    ):
        T.device_entry()
        _warp = T.warp_id([1])
        lane = T.lane_id([32])
        shared = T.alloc_shared((16,), "uint8", align=16)
        value = T.alloc_local((1,), "uint32")
        if lane < 16:
            shared[lane] = T.uint8(0x35)
        T.cuda.warp_sync()
        T.ptx.fence.proxy.async_.shared__cta()
        if lane == 0:
            if before:
                T.ptx["ld.global.u32.proxy::readonly"](value[0], destination.ptr_to([0]))
            T.ptx["cp.async.bulk.global.shared::cta.bulk_group.cp_mask"](
                destination.ptr_to([0]),
                shared.ptr_to([0]),
                T.uint32(16),
                T.cast(byte_mask, "uint16"),
            )
            T.ptx.cp.async_.bulk.commit_group()
            T.ptx.cp.async_.bulk.wait_group(0)
            if not before:
                T.ptx["ld.global.u32.proxy::readonly"](value[0], destination.ptr_to([0]))
            output[0] = value[0]

    return kernel


_READONLY_OVERLAP = no_spec(20, "write overlapping readonly-read bytes")


@pytest.mark.parametrize(
    "mask",
    [
        pytest.param(0, id="mask0"),
        pytest.param(0xF0, id="mask0xf0"),
        pytest.param(1, id="mask1", marks=_READONLY_OVERLAP),
    ],
)
@pytest.mark.parametrize("before", [False, True], ids=["read_after", "read_before"])
def test_cp_mask_readonly_overlap_uses_selected_bytes(before, mask):
    """Replaces ``tests/numsim/runtime/test_non_tensor_bulk_forms.py::test_cp_mask_readonly_overlap_uses_selected_bytes`` (its ``before``/``mask`` loops as params).

    A ``.cp_mask`` bulk store and a ``ld.global...proxy::readonly`` of word 0:
    only the mask-selected bytes are written. Masks 0 and 0xF0 select none of
    the read bytes, so both checkers are clean and NumSim writes exactly the
    selected bytes. Mask 1 writes a read byte: both checkers report ``error``
    and NumSim raises, all with "write overlaps readonly bytes". No checker
    owns that check in the new specs (test-migration.md no-spec item 20;
    racecheck treats ``Proxy::ReadOnly`` as generic), so mask 1 is ``no_spec``.
    """

    kernel = _cp_mask_readonly_kernel(before)

    def args():
        return {
            "destination": np.full(16, 0xD3, np.uint8),
            "byte_mask": mask,
            "output": np.zeros(1, np.uint32),
        }

    for checker in CHECKERS:
        report = _check(checker, kernel, args())
        if mask == 1:
            assert report.verdict == "error", report.format()
            assert "write overlaps readonly bytes" in str(report.to_dict()), report.format()
        else:
            report.require_clean()
    module = v2.transpile(kernel)
    if mask == 1:
        with pytest.raises(v2.ExecutionError, match="write overlaps readonly bytes"):
            v2.Engine().run(module, args())
    else:
        result = v2.Engine().run(module, args())
        assert result.verdict == "clean"
        np.testing.assert_array_equal(result.outputs["output"], [0xD3D3D3D3])
        np.testing.assert_array_equal(
            result.outputs["destination"], np.where((mask >> np.arange(16)) & 1, 0x35, 0xD3)
        )


# -- st.async through mapa-mapped u32 addresses ---------------------------------


@T.prim_func
def raw_st_async_uses_mapped_u32_addresses(output: T.Buffer((8,), "uint32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    destination = T.alloc_buffer((8,), "uint32", scope="shared", align=16)
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if (cta == 1) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
    T.ptx.fence.proxy.async_.shared__cluster()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if (cta == 0) and (lane == 0):
        remote_barrier = T.alloc_local((1,), "uint32")
        T.ptx.mapa.shared__cluster.u32(
            remote_barrier[0],
            T.cuda.cvta_generic_to_shared(barrier.ptr_to([0])),
            T.uint32(1),
        )
        remote_destination = T.alloc_local((1,), "uint32")
        remote_destination_hi = T.alloc_local((1,), "uint32")
        T.ptx.mapa.shared__cluster.u32(
            remote_destination[0],
            T.cuda.cvta_generic_to_shared(destination.ptr_to([0])),
            T.uint32(1),
        )
        T.ptx.mbarrier.arrive.expect_tx.shared__cluster.b64(
            remote_barrier[0], T.uint32(32), pred=True
        )
        T.ptx.st_async.shared__cluster.mbarrier__complete_tx__bytes.v4.u32(
            remote_destination[0],
            T.uint32(0x10203040),
            T.uint32(0x50607080),
            T.uint32(0x90A0B0C0),
            T.uint32(0xD0E0F001),
            remote_barrier[0],
        )
        T.ptx.add.u32(remote_destination_hi[0], remote_destination[0], T.uint32(16))
        T.ptx.st_async.shared__cluster.mbarrier__complete_tx__bytes.v4.u32(
            remote_destination_hi[0],
            T.uint32(0x12345678),
            T.uint32(0x9ABCDEF0),
            T.uint32(0x0BADF00D),
            T.uint32(0xCAFEBABE),
            remote_barrier[0],
        )
    if (cta == 1) and (lane == 0):
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
    T.cuda.cluster_sync()
    if (cta == 1) and (lane < 8):
        output[lane] = destination[lane]


@T.prim_func
def raw_st_async_uses_mapped_u32_expression_offset(output: T.Buffer((8,), "uint32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    destination = T.alloc_buffer((8,), "uint32", scope="shared", align=16)
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if (cta == 1) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
    T.ptx.fence.proxy.async_.shared__cluster()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if (cta == 0) and (lane == 0):
        remote_barrier = T.alloc_local((1,), "uint32")
        T.ptx.mapa.shared__cluster.u32(
            remote_barrier[0],
            T.cuda.cvta_generic_to_shared(barrier.ptr_to([0])),
            T.uint32(1),
        )
        remote_destination = T.alloc_local((1,), "uint32")
        T.ptx.mapa.shared__cluster.u32(
            remote_destination[0],
            T.cuda.cvta_generic_to_shared(destination.ptr_to([0])),
            T.uint32(1),
        )
        T.ptx.mbarrier.arrive.expect_tx.shared__cluster.b64(
            remote_barrier[0], T.uint32(32), pred=True
        )
        T.ptx.st_async.shared__cluster.mbarrier__complete_tx__bytes.v4.u32(
            remote_destination[0],
            T.uint32(0x10203040),
            T.uint32(0x50607080),
            T.uint32(0x90A0B0C0),
            T.uint32(0xD0E0F001),
            remote_barrier[0],
        )
        T.ptx.st_async.shared__cluster.mbarrier__complete_tx__bytes.v4.u32(
            remote_destination[0] + T.uint32(16),
            T.uint32(0x12345678),
            T.uint32(0x9ABCDEF0),
            T.uint32(0x0BADF00D),
            T.uint32(0xCAFEBABE),
            remote_barrier[0],
        )
    if (cta == 1) and (lane == 0):
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
    T.cuda.cluster_sync()
    if (cta == 1) and (lane < 8):
        output[lane] = destination[lane]


def mapped_st_async_case(offset_form):
    """Use the same remote-store protocol for instruction and expression offsets."""

    kernel = raw_st_async_uses_mapped_u32_addresses
    if offset_form == "instruction":
        return kernel
    loads = {}
    variables = {}

    def record(node):
        if type(node).__name__ == "TensorLoad":
            loads[node.source.name] = node
        elif type(node).__name__ == "Var":
            variables[node.name] = node

    structural_walk(kernel.body, (Expr, record))
    base = loads["remote_destination"]
    dynamic_offset = tirx.Cast("uint32", variables["lane"] + 16)
    expressions = {
        "constant": base + tirx.const(16, "uint32"),
        "dynamic": base + dynamic_offset,
        "wrapped": (base + tirx.const(0xFFFFFFFF, "uint32")) + tirx.const(17, "uint32"),
        "reversed": dynamic_offset + base,
        "subtracted": (base + tirx.const(32, "uint32")) - dynamic_offset,
    }

    def replace(call):
        if call.op.name == "tirx.ptx.st_async_vec" and call.args[0].same_as(
            loads["remote_destination_hi"]
        ):
            return Call(
                call.op,
                [expressions[offset_form], *call.args[1:]],
                attrs=call.attrs,
                ty_args=call.ty_args,
                span=call.span,
                ret_ty=call.ty,
            )
        return call

    return kernel.with_body(structural_map(kernel.body, (Call, replace)))


_ST_ASYNC_EXPECTED = np.array(
    [0x10203040, 0x50607080, 0x90A0B0C0, 0xD0E0F001, 0x12345678, 0x9ABCDEF0, 0x0BADF00D, 0xCAFEBABE],
    dtype=np.uint32,
)


@pytest.mark.parametrize(
    "checker",
    ["synccheck", pytest.param("racecheck", marks=_REMOTE_ARRIVE_DEFAULT_CTA_SCOPE)],
)
@pytest.mark.parametrize(
    "offset_form", ["instruction", "constant", "dynamic", "reversed", "subtracted", "wrapped"]
)
def test_raw_st_async_preserves_mapped_remote_cta_ownership(offset_form, checker):
    """Replaces ``tests/numsim/runtime/test_non_tensor_bulk_forms.py::test_raw_st_async_preserves_mapped_remote_cta_ownership[offset_form]`` (its checker loop as a param).

    CTA 0 ``st.async``-es two disjoint 16-byte halves into CTA 1 through a
    ``mapa``-ed u32 address, the second half offset by an ``add.u32`` or by
    one of five equivalent u32 expressions. Both resolve to CTA 1's
    ``destination``: the checker is clean, and NumSim reads back both halves.
    The racecheck param is ``no_spec`` item 11 (see
    ``_REMOTE_ARRIVE_DEFAULT_CTA_SCOPE``).
    """

    kernel = mapped_st_async_case(offset_form)
    _check(checker, kernel, {"output": np.zeros(8, dtype=np.uint32)}).require_clean()
    result = v2.Engine().run(v2.transpile(kernel), {"output": np.zeros(8, dtype=np.uint32)})
    np.testing.assert_array_equal(result.outputs["output"], _ST_ASYNC_EXPECTED)


@pytest.mark.parametrize(
    "checker",
    ["synccheck", pytest.param("racecheck", marks=_REMOTE_ARRIVE_DEFAULT_CTA_SCOPE)],
)
def test_raw_st_async_mapped_expression_offsets_are_disjoint(checker):
    """Replaces ``tests/numsim/runtime/test_non_tensor_bulk_forms.py::test_raw_st_async_mapped_expression_offsets_are_disjoint[checker]``.

    The second remote ``st.async`` addresses ``remote_destination[0] + 16``
    inline: its footprint is disjoint from the first, so the checker is clean
    with no findings. The racecheck param is ``no_spec`` item 11.
    """

    report = _check(checker, raw_st_async_uses_mapped_u32_expression_offset, {"output": np.zeros(8, dtype=np.uint32)})
    assert_clean(report)


# -- tcgen05 sparse-B16 / LUT-B async read lifetimes -----------------------------

# Only the MMA-enabled params depend on it: the disabled MMA reads nothing.
_A_ONLY_RESTRICTED_COMMIT = _no_spec_unlisted(
    "tcgen05.commit .sync_restrict::shared::read::mma::a retiring only the MMA's shared-A reads "
    "(not in the TcgenCommit contract per lowering-inventory.md; the v2 interpreter treats it as a "
    "full commit, so the B/metadata/LUT reuse is ordered and racecheck is clean)"
)
_DISABLED = pytest.mark.parametrize(
    "disabled",
    [
        pytest.param(True, id="mma_disabled"),
        pytest.param(False, id="mma_enabled", marks=_A_ONLY_RESTRICTED_COMMIT),
    ],
)


@_DISABLED
@pytest.mark.parametrize("resource", ["b", "lookup"])
def test_sparse_b16_metadata_and_b_lifetimes(resource, disabled):
    """Replaces ``tests/numsim/runtime/test_tcgen05_sparse_b16.py::test_sparse_b16_metadata_and_b_lifetimes`` (its ``resource``/``disabled`` loops as params).

    After the A-only restricted commit (``sync_restrict::shared::read::mma::a``)
    completes, the kernel overwrites shared B or the TMEM sparse metadata. The
    MMA still reads them, so racecheck reports ``error``; when ``zero_mask``
    bit 63 predicates the MMA off there is no read and racecheck is clean.
    The A-only retire set of the restricted commit is in no new spec, so the
    MMA-enabled params are ``no_spec`` (unnumbered).
    """

    kernel, args, _ = sparse_float_case(*SPARSE_B16_CASES[0], early_reuse=resource)
    args["zero_mask"][0] = np.uint64(1 << 63 if disabled else 0)
    report = v2.racecheck(kernel, args)
    assert report.verdict == ("clean" if disabled else "error"), report.format()


@_DISABLED
@pytest.mark.parametrize("block", [False, True], ids=["dense", "block_scale"])
@pytest.mark.parametrize("resource", ["b", "lookup"])
def test_lut_b_async_read_lifetimes(resource, block, disabled):
    """Replaces ``tests/numsim/runtime/test_tcgen_lut_b.py::test_lut_b_async_read_lifetimes[resource]`` (its ``block``/``disabled`` loops as params).

    ``.decompress::lut::b`` MMA (segment 1): the shared write lands in B's
    unused first 24-byte half and the TMEM write overlaps the lookup table.
    Neither is retired by the A-only restricted commit, so racecheck reports
    ``error``; with the MMA predicated off by ``zero_mask`` it is clean.
    The MMA-enabled params are ``no_spec`` (unnumbered) as for sparse B16.
    """

    kernel, args, _ = lut_b_case(segment=1, block=block, early_reuse=resource)
    args["zero_mask"][0] = np.uint64(1 << 63 if disabled else 0)
    report = v2.racecheck(kernel, args)
    assert report.verdict == ("clean" if disabled else "error"), report.format()


# -- TMA override::global_address store destination ------------------------------


_TMA_OVERRIDE_S2G_READ_EARLY = """
@T.prim_func
def kernel(input_map: T.TensorMap(), replacement: T.Buffer((32772,), "float32"),
           output: T.Buffer((4,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "float32", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane < 4:
        shared[lane] = T.cast(lane + 1, "float32")
    if lane == 0:
        T.ptx["mbarrier.init.shared.b64"](barrier.ptr_to([0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx["cp.async.bulk.tensor.1d.global.shared::cta.bulk_group.override::global_address.override::global_dim"](T.address_of(input_map), T.reinterpret("uint64", replacement.ptr_to([4])), T.cast(4, "uint16"), T.int32(0), shared.ptr_to([0]), pred=T.bool(True))
        T.ptx.cp.async_.bulk.commit_group()
    T.cuda.cta_sync()
    if lane < 4:
        output[lane] = replacement[4 + lane]
    if lane == 0:
        T.ptx.cp.async_.bulk.wait_group(0)
"""


def test_tma_override_keeps_async_destination_race():
    """Replaces ``tests/numsim/runtime/test_tma_overrides.py::test_tma_override_keeps_async_destination_race``.

    ``override_kernel("s2g", "dim_b16", read_early=True)`` (expanded source):
    a TMA store whose ``override::global_address`` points into ``replacement``
    is committed but not waited before the CTA reads ``replacement[4:8]``. The
    store's footprint is the overridden destination (``replacement`` bytes
    [16, 32)), so racecheck reports ``error`` with a ``data_race`` on it.
    Legacy asserted ``access_pair == write_read`` (the store's write placed at
    issue). The new semantics land bulk-write bytes at completion
    (racecheck-semantics.md, write-history "Fallback" bullet), here the
    ``wait_group`` after the read, so v2 reports the same race as
    ``read_write``; either orientation is accepted.
    """

    original = np.full(64, -10, np.float32)
    descriptor = numsim.TensorMap(
        base=original, global_shape=(8,), global_strides=(), box_shape=(4,), element_strides=(1,)
    ).numpy()
    inputs = {
        "input_map": descriptor,
        "replacement": np.arange(32772, dtype=np.float32),
        "output": np.zeros(4, np.float32),
    }
    kernel = tvm.script.from_source(_TMA_OVERRIDE_S2G_READ_EARLY, {"T": T})
    report = v2.racecheck(kernel, inputs)
    assert report.verdict == "error", report.format()
    assert race_access_pairs(report) & {"write_read", "read_write"}, report.format()
    races = [f for f in report.findings if f.status == "error" and f.kind == "data_race"]
    assert any("[16..32)" in f.message for f in races), report.format()
