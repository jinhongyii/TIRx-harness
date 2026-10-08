"""Output comparison against expected arrays, moved from the legacy ``numsim/api.py``
(redesign step 5). Behaviour unchanged."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tirx_harness.numsim.cases import ComparisonRegion, ComparisonSpec
from tirx_harness.numsim.errors import NumSimExecutionError
from tirx_harness.numsim.report import Mismatch, NumSimReport


@dataclass
class NumSimResult:
    outputs: dict[str, Any]
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        if any(item.get("status") == "review" for item in self.diagnostics):
            return "review"
        return "clean"

    def assert_close(
        self, expected: dict[str, Any], tolerances: dict[str, ComparisonSpec] | None = None
    ) -> None:
        compare(self, expected, tolerances=tolerances).require_ok()


def compare(
    result: NumSimResult,
    expected: dict[str, Any],
    *,
    tolerances: dict[str, ComparisonSpec] | None = None,
) -> NumSimReport:
    if not expected:
        raise NumSimExecutionError("NumSim expected outputs must not be empty")
    tolerances = {} if tolerances is None else tolerances
    unknown_tolerances = tolerances.keys() - expected.keys()
    if unknown_tolerances:
        raise NumSimExecutionError(
            f"NumSim comparison specs refer to unknown expected outputs: "
            f"{sorted(unknown_tolerances)}"
        )
    mismatches: list[Mismatch] = []
    for name, reference in expected.items():
        if name not in result.outputs:
            mismatches.append(Mismatch(name, (), "<missing>", "<present>"))
            continue
        actual_array = _decode_comparison_array(
            np.asarray(result.outputs[name]), tolerances.get(name, ComparisonSpec()).actual_encoding
        )
        expected_array = np.asarray(reference)
        explicit_tolerance = name in tolerances
        spec = tolerances.get(name, ComparisonSpec())
        regions = spec.regions or (
            ComparisonRegion(
                tuple(slice(None) for _ in actual_array.shape),
                tuple(slice(None) for _ in expected_array.shape),
            ),
        )
        for region in regions:
            actual_selector = _normalize_comparison_selector(
                region.actual, actual_array.ndim, field=f"{name}.actual"
            )
            expected_selector = _normalize_comparison_selector(
                region.expected if region.expected is not None else region.actual,
                expected_array.ndim,
                field=f"{name}.expected",
            )
            actual_view = np.asarray(actual_array[actual_selector])
            expected_view = np.asarray(expected_array[expected_selector])
            if actual_view.size == 0 or expected_view.size == 0:
                raise NumSimExecutionError(
                    f"NumSim comparison region for {name!r} selects zero elements"
                )
            if actual_view.shape != expected_view.shape:
                mismatches.append(
                    Mismatch(
                        name,
                        _comparison_index(actual_selector, (), actual_array.shape),
                        actual_view.shape,
                        expected_view.shape,
                    )
                )
                break
            exact_dtypes = {"b", "i", "u"}
            actual_numeric = _comparison_numeric_view(actual_view)
            expected_numeric = _comparison_numeric_view(expected_view)
            if (
                not explicit_tolerance
                and actual_view.dtype.kind in exact_dtypes
                and expected_view.dtype.kind in exact_dtypes
            ):
                close = np.equal(actual_view, expected_view)
            elif actual_numeric is not None and expected_numeric is not None:
                close = np.isclose(
                    actual_numeric,
                    expected_numeric,
                    rtol=spec.rtol,
                    atol=spec.atol,
                    equal_nan=spec.equal_nan,
                )
            else:
                close = np.equal(actual_view, expected_view)
            if np.all(close):
                continue
            first = tuple(int(value) for value in np.argwhere(~close)[0])
            mismatches.append(
                Mismatch(
                    name,
                    _comparison_index(actual_selector, first, actual_array.shape),
                    actual_view[first].item(),
                    expected_view[first].item(),
                )
            )
            break
    return NumSimReport(ok=not mismatches, mismatches=mismatches, diagnostics=result.diagnostics)


def _comparison_numeric_view(array: np.ndarray) -> np.ndarray | None:
    if array.dtype.kind in "biufc":
        return array
    if array.dtype.name == "bfloat16":
        return array.astype(np.float32)
    return None


def _decode_comparison_array(array: np.ndarray, encoding: str | None) -> np.ndarray:
    if encoding is None:
        return array
    if encoding == "bfloat16":
        if array.dtype.kind not in "iu":
            raise NumSimExecutionError(
                f"bfloat16 comparison requires an integer backing, got {array.dtype}"
            )
        if array.dtype.itemsize == 2:
            encoded = array.astype(np.uint16, copy=False)
        elif array.dtype.itemsize == 1:
            if array.ndim != 1 or array.size % 2:
                raise NumSimExecutionError(
                    "bfloat16 byte-carrier comparison requires a one-dimensional even byte count"
                )
            encoded = np.ascontiguousarray(array).view(np.uint8).view(np.uint16)
        else:
            raise NumSimExecutionError(
                "bfloat16 comparison requires a 16-bit integer or 8-bit byte-carrier "
                f"backing, got {array.dtype}"
            )
        bits = encoded.astype(np.uint32) << np.uint32(16)
        return bits.view(np.float32)
    raise AssertionError(f"unvalidated NumSim comparison encoding {encoding!r}")


def _normalize_comparison_selector(
    selector: tuple[int | slice, ...], rank: int, *, field: str
) -> tuple[int | slice, ...]:
    if len(selector) > rank:
        raise NumSimExecutionError(
            f"NumSim comparison selector {field} has rank {len(selector)}, exceeding array rank {rank}"
        )
    normalized: list[int | slice] = []
    for value in selector:
        if isinstance(value, bool) or not isinstance(value, (int, slice)):
            raise NumSimExecutionError(
                f"NumSim comparison selector {field} contains unsupported index {value!r}"
            )
        normalized.append(value)
    normalized.extend(slice(None) for _ in range(rank - len(normalized)))
    return tuple(normalized)


def _comparison_index(
    selector: tuple[int | slice, ...], local: tuple[int, ...], shape: tuple[int, ...]
) -> tuple[int, ...]:
    result: list[int] = []
    local_axis = 0
    for axis, value in enumerate(selector):
        if isinstance(value, int):
            result.append(value + shape[axis] if value < 0 else value)
            continue
        start, stop, step = value.indices(shape[axis])
        if local_axis >= len(local):
            result.append(start)
        else:
            coordinate = start + local[local_axis] * step
            if coordinate < 0 or coordinate >= shape[axis] or coordinate == stop:
                raise NumSimExecutionError("NumSim comparison mismatch index escaped its region")
            result.append(coordinate)
        local_axis += 1
    return tuple(result)
