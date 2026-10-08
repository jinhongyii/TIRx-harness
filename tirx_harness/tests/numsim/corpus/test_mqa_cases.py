from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tests.numsim.corpus.kernels.deepgemm import (
    FP4_MQA_CONFIGS,
    FP8_MQA_CONFIGS,
    _dense_mqa_reference,
    _e2m1_bits_to_float32,
    _pack_e2m1,
    _pack_e8m0_words,
    _unpack_e2m1,
    _unpack_e8m0_words,
    prepare_fp4_mqa_case,
    prepare_fp8_mqa_case,
)
from tests.numsim.support._tirx_kernels import config_params, load_tirx_kernel
from tests.numsim.support.host_bindings import decode_tensor_maps, host_layout, tensor_map_base_array
from tests.numsim.support.kernel_facts import launch_topology
from tirx_harness.numsim import v2

_DENSE_KERNELS = (
    "deepgemm_sm100_fp4_mqa_logits",
    "deepgemm_sm100_fp8_mqa_logits",
)
_CORPUS = {
    _DENSE_KERNELS[0]: SimpleNamespace(configs=FP4_MQA_CONFIGS, prepare=prepare_fp4_mqa_case),
    _DENSE_KERNELS[1]: SimpleNamespace(configs=FP8_MQA_CONFIGS, prepare=prepare_fp8_mqa_case),
}


def _module(name: str):
    return load_tirx_kernel(name)


def _case_family(module_name: str):
    _module(module_name)
    return _CORPUS[module_name]


def _host_identity(args: dict) -> tuple:
    """Address-free identity of the host arguments: how arrays alias, their
    bytes (TensorMap images without their host pointer), the tensor each image
    addresses, and every scalar."""

    arrays = {name: value for name, value in args.items() if isinstance(value, np.ndarray)}
    contents = {}
    for name, value in arrays.items():
        raw = np.ascontiguousarray(value).view(np.uint8).reshape(-1).copy()
        bases = []
        for descriptor in decode_tensor_maps(value):
            raw[descriptor.byte_offset : descriptor.byte_offset + 8] = 0
            start = descriptor.byte_offset
            bases.append(tensor_map_base_array(value.view(np.uint8).reshape(-1)[start : start + 128]).tobytes())
        contents[name] = (value.dtype.str, value.shape, raw.tobytes(), tuple(bases))
    scalars = {name: value for name, value in args.items() if name not in arrays}
    return (
        host_layout(arrays).buffers,
        contents,
        sorted((name, type(value).__name__, repr(value)) for name, value in scalars.items()),
    )


def _case(module_name: str, index: int):
    family = _case_family(module_name)
    return family.prepare(**config_params(family.configs[index]))


def test_dense_mqa_numpy_reference_matches_independent_scalar_reference():
    q = np.array(
        [[[1.0, -2.0, 0.5], [0.0, 1.0, 2.0]], [[-1.0, 0.5, 1.5], [2.0, -1.0, 0.0]]],
        dtype=np.float32,
    )
    kv = np.array(
        [[1.0, 0.0, 1.0], [0.5, -1.0, 2.0], [-1.0, 2.0, 0.5], [2.0, 1.0, -1.0]], dtype=np.float32
    )
    weights = np.array([[1.0, -0.25], [0.5, 2.0]], dtype=np.float32)
    starts = np.array([0, 1], dtype=np.int32)
    ends = np.array([3, 4], dtype=np.int32)
    expected = np.full((2, 4), -np.inf, dtype=np.float32)
    for row in range(2):
        for token in range(int(starts[row]), int(ends[row])):
            value = 0.0
            for head in range(2):
                score = sum(float(q[row, head, dim] * kv[token, dim]) for dim in range(3))
                value += max(score, 0.0) * float(weights[row, head])
            expected[row, token] = value

    np.testing.assert_allclose(
        _dense_mqa_reference(q, kv, weights, starts, ends), expected, rtol=0.0, atol=0.0
    )


def test_mqa_low_precision_packing_round_trips_physical_codes():
    codes = np.arange(16, dtype=np.uint8).reshape(2, 8)
    packed = _pack_e2m1(codes)

    np.testing.assert_array_equal(_unpack_e2m1(packed), codes)
    np.testing.assert_array_equal(
        _e2m1_bits_to_float32(np.array([0x2, 0xA], dtype=np.uint8)),
        np.array([1.0, -1.0], dtype=np.float32),
    )
    exponents = np.array([[-2, -1, 0, 1], [1, 0, -1, -2]], dtype=np.int16)
    np.testing.assert_array_equal(
        _unpack_e8m0_words(_pack_e8m0_words(exponents)).astype(np.int16) - 127, exponents
    )


