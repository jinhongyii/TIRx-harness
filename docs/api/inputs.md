# Simulation inputs

These types describe NumSim inputs and numerical comparisons. Unless otherwise
noted, import them from `tirx_harness.numsim`. Fields and defaults below are
read from the Python source. These are data classes; construct them with the
listed field names as keyword arguments.

## Reusable cases and comparisons

```{eval-rst}
.. autoapiclass:: tirx_harness.numsim.NumSimCase
   :members:
   :undoc-members:
```

`kernel` is the specialized TIRx function; `args` binds its parameters,
including output storage. `outputs` selects the result buffers. `reference`
is a zero-argument callable returning a dictionary of expected outputs;
`comparisons` maps output names to comparison specifications. Pass the case to
{py:func}`~tirx_harness.numsim.v2.api.run_case`.

```{eval-rst}
.. autoapiclass:: tirx_harness.numsim.ComparisonSpec
   :members:
   :undoc-members:
```

`rtol` and `atol` are finite, nonnegative relative and absolute tolerances.
`equal_nan` controls whether matching not-a-number values compare equal.
`actual_encoding="bfloat16"` decodes a simulated output's integer backing
storage before comparison. `regions` optionally restricts the compared areas.

```{eval-rst}
.. autoapiclass:: tirx_harness.numsim.ComparisonRegion
   :members:
   :undoc-members:
```

`actual` and `expected` are tuples of integer indices or slices. If `expected`
is omitted, the same selection is used for both arrays. Selected regions must
be nonempty and have matching shapes.

## Tensor maps

A tensor map describes the storage and tile shape for a tensor-memory transfer.
Use `TensorMap(...).numpy()` to create a simulator descriptor, then bind that
array under the kernel's tensor-map parameter name. This descriptor is for
CPU simulation. The descriptor addresses its base array; select the tensor
map as an output to receive that array back in the map's logical layout.

```{eval-rst}
.. autoapiclass:: tirx_harness.numsim.TensorMap
   :members:
   :undoc-members:
```

Shapes and element strides follow the descriptor's dimension order, with the
innermost dimension first; global strides are in bytes. Tensor Memory
Accelerator (TMA) dtype overrides include TensorFloat-32 (`tf32`),
flush-to-zero (`ftz`) floating-point modes, and packed six-bit unsigned values
(`uint6`). `fp4_shared_layout` selects a four-bit floating-point storage layout.

```{eval-rst}
.. autoapiclass:: tirx_harness.numsim.Im2col
   :members:
   :undoc-members:
```

`Im2col` describes an image-to-column transfer for convolution. Its spatial
coordinates use width, height, and depth order (W/H/D).

## Launch selection

Most callers execute the full launch. To select complete clusters, import
`ExecutionSubset` from `tirx_harness.numsim.v2.api` and pass it as `subset` to
`Engine.run` or a checker phase; only the selected clusters run. A cooperative
thread array (CTA) is a thread block. `cluster_ids` are linear cluster ids
(x fastest). `cta_ids` are flattened global CTA ids and must cover whole
clusters; they need a static grid, and a partial cluster raises `InputError`.
With both, the run uses their intersection. A run that skips part of the
launch cannot certify it, so checker verdicts on a subset are at least
`incomplete`: the payload's `analysis_scope` is `{"kind": "subset", ...}` with
the selected and total warp counts.

```{eval-rst}
.. autoapiclass:: tirx_harness.numsim.v2.api.ExecutionSubset
   :members: cluster_ids, cta_ids
   :undoc-members:
```

```{eval-rst}
.. autoapidata:: tirx_harness.numsim.v2.api.ExecutionSubsetSelection
```

This type alias accepts an `ExecutionSubset` or a mapping from integer phase
indices to `ExecutionSubset` objects. One engine run serves every launch of a
module, so the mapped subsets must be equal.

```{eval-rst}
.. autoapiclass:: tirx_harness.numsim.ExecutionAssumptions
   :members: external_grid_dependencies_satisfied
   :undoc-members:
```

The redesigned engine accepts `ExecutionAssumptions` and ignores it: grid
dependencies on an earlier launch are satisfied at the launch boundary
(pending: remove the type when the legacy engine is deleted).
