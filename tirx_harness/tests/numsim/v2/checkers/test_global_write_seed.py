"""Host-input aliasing and raw pointer provenance into global memory.

Replaces the seven ``gap_unportable`` rows of
``tests/analysis_tools/racecheck/test_native_global_write_seed.py``. Two kernel
parameters bound to overlapping host arrays must be checked as ONE allocation,
including through raw pointer arithmetic, ``selp``/loop-carried pointers,
TensorMap base replacement and ``discard.global.L2``. CTA 0 reads, CTA 1 writes
(no ordering between CTAs), so an aliased binding is a cross-CTA read/write race.

The legacy write-seed optimisation (``global_write_seed_replays``) has no
analogue in the new core and is not asserted, nor is the legacy
``inspect_accesses`` parametrisation (payload shape, C).

Host-input aliasing has no spec sentence (test-migration.md, "Semantics in
legacy tests that no new spec mentions" item 2), and v2 fails closed on it
today (``run.py::_reject_aliased_buffers`` raises ``NotImplementedError``,
CONTRACT_REQUESTS W8-6). The aliased parametrisations are therefore
``xfail(strict=False)``; the non-aliased controls must be clean.
"""

from __future__ import annotations

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx

from tirx_harness import numsim
from tirx_harness.numsim import v2

from ._runnable import (
    RACE,
    assert_clean,
    assert_error_kind,
    assert_no_incomplete,
    no_spec,
    race_access_pairs,
    requires_v2_engine,
    v2_gap,
)

pytestmark = requires_v2_engine

_HOST_ALIAS = no_spec(2, "host-input aliasing; v2 fails closed (NotImplementedError, CONTRACT_REQUESTS W8-6)")
_HOST_POINTER = no_spec(
    2,
    "host-input aliasing; the legacy kernel is fed HOST addresses (ndarray.ctypes.data) as pointer "
    "data, and the v2 API has no way to obtain an engine address of a binding",
)


def _assert_alias_race(report) -> None:
    assert_no_incomplete(report)
    assert_error_kind(report, RACE)
    assert race_access_pairs(report) & {"read_write", "write_read"}, report.format()


def _alias(flag: bool):
    return pytest.param(flag, id="aliased" if flag else "distinct", marks=_HOST_ALIAS if flag else ())


# -- kernels (copied from the legacy test) ---------------------------------------


@T.prim_func
def structured_write(source: T.Buffer((1,), "int32"), target: T.Buffer((1,), "int32"),
                     output: T.Buffer((1,), "int32")):
    T.device_entry()
    cta = T.cta_id([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    alias = T.decl_buffer((1,), "int32", data=target.data)
    if lane == 0:
        if cta == 0:
            output[0] = source[0]
        else:
            alias[0] = 7


@T.prim_func
def raw_write(source: T.Buffer((1,), "int32"), target: T.Buffer((1,), "int32"),
              output: T.Buffer((1,), "int32")):
    T.device_entry()
    cta = T.cta_id([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        if cta == 0:
            output[0] = source[0]
        else:
            T.ptx.st.global_.s32(target.ptr_to([0]), T.int32(7))


@T.prim_func
def register_result_to_global(source: T.Buffer((1,), "float32"),
                              output: T.Buffer((2,), "float32")):
    T.device_entry()
    cta = T.cta_id([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        T.ptx.ex2.approx.ftz.f32(output[cta], source[0])


@T.prim_func
def selected_pointer_write(source: T.Buffer((3,), "int32"),
                           a: T.Buffer((3,), "int32"), b: T.Buffer((3,), "int32"),
                           output: T.Buffer((1,), "int32"), select_b: T.int32):
    T.device_entry()
    cta = T.cta_id([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    pointer = T.alloc_buffer((1,), "uint64", scope="local")
    if lane == 0:
        if cta == 0:
            output[0] = source[2]
        else:
            T.ptx.mov.b64(pointer[0], T.reinterpret("uint64", a.ptr_to([0])))
            T.ptx.selp.u64(pointer[0], T.reinterpret("uint64", b.ptr_to([0])),
                          pointer[0], T.ptx.pred(select_b != 0))
            for _ in T.serial(2):
                pointer[0] = pointer[0] + T.uint64(4)
            T.ptx.st.global_.s32(T.reinterpret("handle", pointer[0]), 7)


@T.prim_func
def clobbered_pointer_write(source: T.Buffer((1,), "int32"), target: T.Buffer((1,), "int32"),
                            pointer_bits: T.Buffer((1,), "uint64"),
                            output: T.Buffer((1,), "int32")):
    T.device_entry()
    cta = T.cta_id([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    pointer = T.alloc_buffer((1,), "uint64", scope="local")
    if lane == 0:
        if cta == 0:
            output[0] = source[0]
        else:
            pointer[0] = T.reinterpret("uint64", target.ptr_to([0]))
            T.ptx.xor.b64(pointer[0], pointer_bits[0], T.uint64(0))
            T.ptx.st.global_.s32(T.reinterpret("handle", pointer[0]), 7)


@T.prim_func
def copied_pointer_write(source: T.Buffer((1,), "int32"), target: T.Buffer((1,), "int32"),
                         pointer_bits: T.Buffer((1,), "uint64"),
                         output: T.Buffer((1,), "int32")):
    T.device_entry()
    cta = T.cta_id([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    pointer = T.alloc_buffer((1,), "uint64", scope="local")
    if cta == 0:
        if lane == 0:
            output[0] = source[0]
    else:
        pointer[0] = T.reinterpret("uint64", target.ptr_to([0]))
        Tx.copy(pointer[:], pointer_bits[:])
        if lane == 0:
            T.ptx.st.global_.s32(T.reinterpret("handle", pointer[0]), 7)


def tensor_map_writer(replace):
    # Typed TensorMap parameters are grid-constant. Mutate a global descriptor
    # buffer instead, just as the raw descriptor runtime cases do.
    map_type = 'T.Buffer((128,), "uint8")' if replace else "T.TensorMap()"
    descriptor = "target_map.ptr_to([0])" if replace else "T.address_of(target_map)"
    return tvm.script.from_source(f'''
@T.prim_func
def kernel(source: T.Buffer((32,), "int32"), replacement: T.Buffer((32,), "int32"),
           output: T.Buffer((1,), "int32"), target_map: {map_type}):
    T.device_entry()
    cta = T.cta_id([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32,), "int32", scope="shared")
    if cta == 0:
        if lane == 0:
            output[0] = source[0]
    else:
        shared[lane] = 7
        T.cuda.warp_sync()
        T.ptx.fence.proxy.async_.shared__cta()
        if lane == 0:
            if {replace}:
                T.ptx.tensormap_replace.tile.global_address.global_.b1024.b64(
                    {descriptor}, T.reinterpret("uint64", replacement.ptr_to([0])))
                T.ptx.fence.proxy.tensormap__generic.release.gpu()
                T.ptx.fence.proxy.tensormap__generic.acquire.gpu({descriptor})
            T.ptx["cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group"](
                {descriptor}, 0, shared.ptr_to([0]))
            T.ptx.cp.async_.bulk.commit_group()
            T.ptx.cp.async_.bulk.wait_group(0)
''', {"T": T})


@T.prim_func
def discard_after_read(source: T.Buffer((128,), "uint8"), target: T.Buffer((128,), "uint8"),
                       output: T.Buffer((1,), "uint8")):
    T.device_entry()
    cta = T.cta_id([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        if cta == 0:
            output[0] = source[0]
        else:
            T.ptx.discard.global_.L2(target.ptr_to([0]))


@T.prim_func
def offset_pointer_write(source: T.Buffer((1,), "int32"), target: T.Buffer((1,), "int32"),
                         redirected: T.Buffer((1,), "int32"), offset: T.Buffer((1,), "uint64"),
                         output: T.Buffer((1,), "int32")):
    T.device_entry()
    cta = T.cta_id([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        if cta == 0:
            output[0] = source[0]
        else:
            if target[0] == 0:
                T.ptx.st.global_.s32(T.reinterpret("handle",
                    T.reinterpret("uint64", target.ptr_to([0])) + offset[0]), 7)
                target[0] = 1


# -- tests --------------------------------------------------------------------------


@pytest.mark.parametrize("func", [structured_write, raw_write], ids=["structured", "raw"])
@pytest.mark.parametrize("alias_inputs", [_alias(False), _alias(True)])
def test_read_before_write_is_still_checked_for_host_aliases(func, alias_inputs):
    """Replaces ``tests/analysis_tools/racecheck/test_native_global_write_seed.py::test_read_before_write_is_still_checked_for_host_aliases`` (all params; ``inspect_accesses`` dropped).

    ``source`` and ``target`` bound to the same host array: CTA 0's read and
    CTA 1's write race (error, read/write pair); distinct arrays are clean.
    """

    source = np.zeros(1, dtype=np.int32)
    target = source if alias_inputs else np.zeros(1, dtype=np.int32)
    report = v2.racecheck(func, {"source": source, "target": target, "output": np.zeros(1, dtype=np.int32)})
    if alias_inputs:
        _assert_alias_race(report)
    else:
        assert_clean(report)


@pytest.mark.parametrize("alias_inputs", [_alias(False), _alias(True)])
def test_register_result_writes_preserve_host_alias_races(alias_inputs):
    """Replaces ``tests/analysis_tools/racecheck/test_native_global_write_seed.py::test_register_result_writes_preserve_host_alias_races`` (all params; ``inspect_accesses`` dropped).

    ``ex2`` writes its register result to ``output[cta]``; ``source`` is a view
    of ``output[:1]`` when aliased: race; otherwise clean.
    """

    output = np.zeros(2, dtype=np.float32)
    source = output[:1] if alias_inputs else np.zeros(1, dtype=np.float32)
    report = v2.racecheck(register_result_to_global, {"source": source, "output": output})
    if alias_inputs:
        _assert_alias_race(report)
    else:
        assert_clean(report)


@pytest.mark.parametrize("select_b", [0, 1])
@pytest.mark.parametrize(
    "alias",
    [
        pytest.param("none"),
        pytest.param("selected", marks=_HOST_ALIAS),
        pytest.param("unselected", marks=_HOST_ALIAS),
    ],
)
def test_loop_carried_selected_pointer_preserves_compact_alias_races(select_b, alias):
    """Replaces ``tests/analysis_tools/racecheck/test_native_global_write_seed.py::test_loop_carried_selected_pointer_preserves_compact_alias_races`` (all params).

    The store address is ``selp(a, b) + 8`` built through ``mov``/``selp`` and a
    loop; ``source[2]`` races only when ``source`` aliases the SELECTED array.
    """

    a, b = np.zeros(3, dtype=np.int32), np.zeros(3, dtype=np.int32)
    if alias == "selected":
        source = b if select_b else a
    elif alias == "unselected":
        source = a if select_b else b
    else:
        source = np.zeros(3, dtype=np.int32)
    report = v2.racecheck(
        selected_pointer_write,
        {"source": source, "a": a, "b": b, "output": np.zeros(1, dtype=np.int32), "select_b": np.int32(select_b)},
    )
    if alias == "selected":
        _assert_alias_race(report)
    else:
        assert_clean(report)


@_HOST_POINTER
@pytest.mark.parametrize("func", [clobbered_pointer_write, copied_pointer_write], ids=["xor", "tile-copy"])
def test_unknown_register_overwrite_keeps_read_before_write_race(func):
    """Replaces ``tests/analysis_tools/racecheck/test_native_global_write_seed.py::test_unknown_register_overwrite_keeps_read_before_write_race`` (all params).

    The store pointer is overwritten from data (``xor`` / ``Tx.copy``) with the
    address of ``source``: CTA 1's store races CTA 0's read of ``source``.
    """

    source = np.zeros(1, dtype=np.int32)
    report = v2.racecheck(func, {
        "source": source, "target": np.zeros(1, dtype=np.int32),
        "pointer_bits": np.array([source.ctypes.data], dtype=np.uint64),
        "output": np.zeros(1, dtype=np.int32),
    })
    _assert_alias_race(report)


@pytest.mark.parametrize(
    "replace",
    [
        pytest.param(False, id="initial-base", marks=v2_gap(
            "TMA store through a TensorMap bound over a host array (tensor_map_of): "
            "bad_address 'Global address ... is not mapped'")),
        pytest.param(True, id="replaced-base", marks=v2_gap(
            "tensormap.replace + fence.proxy.tensormap::generic release/acquire still reports a "
            "missing_proxy_bridge data_race on the descriptor bytes")),
    ],
)
@pytest.mark.parametrize("alias_inputs", [_alias(False), _alias(True)])
def test_tensor_map_initial_and_replaced_bases_keep_compact_alias_races(replace, alias_inputs):
    """Replaces ``tests/analysis_tools/racecheck/test_native_global_write_seed.py::test_tensor_map_initial_and_replaced_bases_keep_compact_alias_races`` (all params).

    A TMA store through a TensorMap whose (initial or ``tensormap.replace``d)
    global base aliases ``source``: race; distinct arrays: clean.
    """

    initial, replacement = np.zeros(32, dtype=np.int32), np.zeros(32, dtype=np.int32)
    source = (replacement if replace else initial) if alias_inputs else np.zeros(32, dtype=np.int32)
    descriptor = numsim.TensorMap(
        initial, global_shape=(32,), global_strides=(), box_shape=(32,), element_strides=(1,),
    )
    report = v2.racecheck(tensor_map_writer(replace), {
        "source": source, "replacement": replacement, "output": np.zeros(1, dtype=np.int32),
        "target_map": descriptor.numpy(),
    })
    if alias_inputs:
        _assert_alias_race(report)
    else:
        assert_clean(report)


@pytest.mark.parametrize("alias_inputs", [_alias(False), _alias(True)])
def test_discard_is_a_write_for_compact_alias_races(alias_inputs):
    """Replaces ``tests/analysis_tools/racecheck/test_native_global_write_seed.py::test_discard_is_a_write_for_compact_alias_races`` (all params).

    ``discard.global.L2`` invalidates its 128 bytes, i.e. a write
    (``arena.rs`` treats discard as invalidating): it races CTA 0's read of an
    aliased ``source``; distinct arrays are clean.
    """

    storage = np.zeros(256, dtype=np.uint8)
    offset = -storage.ctypes.data % 128
    target = storage[offset:offset + 128]
    source = target if alias_inputs else np.zeros(128, dtype=np.uint8)
    report = v2.racecheck(discard_after_read, {"source": source, "target": target, "output": np.zeros(1, dtype=np.uint8)})
    if alias_inputs:
        _assert_alias_race(report)
    else:
        assert_clean(report)


@_HOST_POINTER
@pytest.mark.parametrize("alias_inputs", [False, True], ids=["distinct", "aliased"])
def test_raw_offset_escaping_seed_restarts_from_original_inputs(alias_inputs):
    """Replaces ``tests/analysis_tools/racecheck/test_native_global_write_seed.py::test_raw_offset_escaping_seed_restarts_from_original_inputs`` (all params; ``inspect_accesses`` dropped).

    ``target.ptr + offset`` escapes into ``redirected``: when ``source`` is
    ``redirected`` the store races CTA 0's read; otherwise clean. The caller's
    arrays are never modified. The seed replay count is C and not asserted.
    """

    target, redirected = sorted((np.zeros(1, np.int32), np.zeros(1, np.int32)), key=lambda value: value.ctypes.data)
    source = redirected if alias_inputs else np.zeros(1, np.int32)
    inputs = dict(source=source, target=target, redirected=redirected,
                  offset=np.array([redirected.ctypes.data - target.ctypes.data], np.uint64),
                  output=np.zeros(1, np.int32))
    report = v2.racecheck(offset_pointer_write, inputs)
    if alias_inputs:
        _assert_alias_race(report)
    else:
        assert_clean(report)
    for name in ("source", "target", "redirected", "output"):
        np.testing.assert_array_equal(inputs[name], np.zeros(1, np.int32))
