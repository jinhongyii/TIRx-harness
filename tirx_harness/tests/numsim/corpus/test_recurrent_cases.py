from __future__ import annotations

import numpy as np

from tests.numsim.corpus.kernels.recurrent import (
    gdn_numpy_reference,
    prepare_native_kda_fixed_case,
    prepare_native_kda_forward_case,
    prepare_gdn_prefill_case,
)
from tirx_harness.numsim import run_case
from tests.numsim.support.host_bindings import decode_tensor_maps, tensor_map_base_array
from tests.numsim.support.kernel_facts import launch_topology


def _bfloat16_bits(value: float) -> np.uint16:
    bits = np.asarray([value], dtype=np.float32).view(np.uint32)[0]
    return np.uint16(bits >> np.uint32(16))


def _descriptor_metadata(array: np.ndarray) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            descriptor.byte_offset,
            descriptor.required_byte_len,
            descriptor.global_shape,
            descriptor.global_strides,
            descriptor.box_shape,
            descriptor.element_strides,
            descriptor.dtype,
            descriptor.fp4_shared_layout,
            descriptor.swizzle,
            descriptor.fill_mode,
        )
        for descriptor in decode_tensor_maps(array)
    )


def _assert_same_arguments(actual: dict[str, object], expected: dict[str, object]) -> None:
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        repeated = expected[name]
        if not isinstance(value, np.ndarray):
            assert value == repeated
            continue
        assert isinstance(repeated, np.ndarray)
        descriptors = decode_tensor_maps(value)
        if not descriptors:
            np.testing.assert_array_equal(value, repeated)
            continue
        assert _descriptor_metadata(value) == _descriptor_metadata(repeated)
        for descriptor in descriptors:
            start = descriptor.byte_offset
            np.testing.assert_array_equal(
                tensor_map_base_array(value.view(np.uint8).reshape(-1)[start : start + 128]),
                tensor_map_base_array(repeated.view(np.uint8).reshape(-1)[start : start + 128]),
            )


def test_gdn_oracle_detects_value_corruption() -> None:
    case = prepare_gdn_prefill_case()
    q = case.args["q"].reshape(1, 2, 128)
    k = case.args["k"].reshape(1, 2, 128)
    v = case.args["v"].reshape(1, 8, 128)
    gate = case.args["gate"].reshape(1, 8)
    beta = case.args["beta"].reshape(1, 8)
    initial_state = case.args["initial_state"].reshape(1, 8, 128, 128)
    expected, expected_state = gdn_numpy_reference(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        seq_lens=(1,),
        scale=case.args["scale"],
    )

    corrupted_v = v.copy()
    corrupted_v[0, 0, 0] = np.float16(corrupted_v[0, 0, 0] + np.float16(1.0))
    corrupted, corrupted_state = gdn_numpy_reference(
        q,
        k,
        corrupted_v,
        gate,
        beta,
        initial_state,
        seq_lens=(1,),
        scale=case.args["scale"],
    )

    assert not np.array_equal(corrupted, expected)
    assert not np.array_equal(corrupted_state, expected_state)
