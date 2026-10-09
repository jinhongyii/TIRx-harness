from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
import pytest
from threadpoolctl import threadpool_info

from tirx_harness import numsim


def _blas_pool_threads():
    return {
        pool["filepath"]: pool["num_threads"]
        for pool in threadpool_info()
        if pool["user_api"] == "blas"
    }


def test_result_assert_close_reports_first_mismatch():
    result = numsim.NumSimResult({"out": np.array([1.0, 3.0], dtype=np.float32)})
    with pytest.raises(AssertionError, match="first mismatch"):
        result.assert_close({"out": np.array([1.0, 2.0], dtype=np.float32)})


@pytest.mark.parametrize("dtype", [np.int32, np.uint64])
def test_integer_comparison_is_exact_by_default(dtype):
    result = numsim.NumSimResult({"out": np.array([100_000], dtype=dtype)})

    report = numsim.compare(result, {"out": np.array([100_001], dtype=dtype)})

    assert not report.ok
    assert report.mismatches[0].index == (0,)


def test_comparison_decodes_declared_bfloat16_backing():
    values = np.array([1.0, -2.5], dtype=np.float32)
    bits = values.view(np.uint32)
    encoded = (bits >> np.uint32(16)).astype(np.uint16)

    report = numsim.compare(
        numsim.NumSimResult({"out": encoded}),
        {"out": values},
        tolerances={"out": numsim.ComparisonSpec(rtol=0.0, atol=0.0, actual_encoding="bfloat16")},
    )

    assert report.ok


def test_comparison_decodes_declared_bfloat16_byte_carrier():
    values = np.array([1.0, -2.5], dtype=np.float32)
    bits = values.view(np.uint32)
    encoded = (bits >> np.uint32(16)).astype(np.uint16).view(np.uint8)

    report = numsim.compare(
        numsim.NumSimResult({"out": encoded}),
        {"out": values},
        tolerances={
            "out": numsim.ComparisonSpec(
                rtol=0.0,
                atol=0.0,
                actual_encoding="bfloat16",
            )
        },
    )

    assert report.ok


def test_comparison_rejects_odd_bfloat16_byte_carrier():
    with pytest.raises(numsim.NumSimExecutionError, match="even byte count"):
        numsim.compare(
            numsim.NumSimResult({"out": np.zeros(3, dtype=np.uint8)}),
            {"out": np.zeros(1, dtype=np.float32)},
            tolerances={"out": numsim.ComparisonSpec(actual_encoding="bfloat16")},
        )


def test_comparison_regions_map_physical_storage_to_logical_reference():
    actual = np.full((2, 6), -99, dtype=np.int32)
    expected = np.full((2, 6), 77, dtype=np.int32)
    actual[0, :2] = [3, 4]
    expected[0, 3:5] = [3, 4]
    spec = numsim.ComparisonSpec(
        rtol=0.0,
        atol=0.0,
        regions=(numsim.ComparisonRegion(actual=(0, slice(0, 2)), expected=(0, slice(3, 5))),),
    )

    assert numsim.compare(
        numsim.NumSimResult({"out": actual}), {"out": expected}, tolerances={"out": spec}
    ).ok

    actual[0, 1] = 5
    report = numsim.compare(
        numsim.NumSimResult({"out": actual}), {"out": expected}, tolerances={"out": spec}
    )
    assert not report.ok
    assert report.mismatches[0].index == (0, 1)


def test_compare_rejects_empty_expected_outputs():
    with pytest.raises(numsim.NumSimExecutionError, match="must not be empty"):
        numsim.compare(numsim.NumSimResult({}), {})


def test_compare_rejects_unknown_comparison_specs():
    with pytest.raises(numsim.NumSimExecutionError, match="unknown expected outputs"):
        numsim.compare(
            numsim.NumSimResult({"output": np.zeros(1, dtype=np.float32)}),
            {"output": np.zeros(1, dtype=np.float32)},
            tolerances={"typo": numsim.ComparisonSpec()},
        )


def test_compare_rejects_zero_element_default_region():
    with pytest.raises(numsim.NumSimExecutionError, match="selects zero elements"):
        numsim.compare(
            numsim.NumSimResult({"output": np.zeros(0, dtype=np.float32)}),
            {"output": np.zeros(0, dtype=np.float32)},
        )


def test_compare_rejects_zero_element_explicit_region():
    with pytest.raises(numsim.NumSimExecutionError, match="selects zero elements"):
        numsim.compare(
            numsim.NumSimResult({"output": np.zeros(4, dtype=np.float32)}),
            {"output": np.zeros(4, dtype=np.float32)},
            tolerances={
                "output": numsim.ComparisonSpec(
                    regions=(numsim.ComparisonRegion(actual=(slice(0, 0),)),)
                )
            },
        )


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"rtol": -1.0}, ValueError, "finite non-negative"),
        ({"atol": float("inf")}, ValueError, "finite non-negative"),
        ({"equal_nan": "false"}, TypeError, "equal_nan must be a bool"),
        ({"regions": []}, TypeError, "regions must be a tuple"),
    ],
)
def test_comparison_spec_rejects_permissive_field_values(kwargs, error, message):
    with pytest.raises(error, match=message):
        numsim.ComparisonSpec(**kwargs)


def test_comparison_region_normalizes_integer_like_indices_and_rejects_fractional_indices():
    region = numsim.ComparisonRegion(actual=(np.int32(1), slice(np.int64(2), None, 1)))

    assert region.actual == (1, slice(2, None, 1))
    with pytest.raises(TypeError, match="index must be an integer"):
        numsim.ComparisonRegion(actual=(1.5,))


