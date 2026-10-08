# Checkers and reports

Import the synchronization checker (`synccheck`) and memory-race checker
(`racecheck`) from `tirx_harness`. Both execute on the CPU for a concrete kernel
invocation, on the same engine run that NumSim uses. Their guarantees and
limitations are described in [Compiler analysis](../components/tools.md).

## Entry points

Pass a specialized TIRx function as `kernel`; for a TIRx-lite kernel, use
`kernel.func`. Bind all runtime parameters by name in `inputs`, including
output buffers. Buffer bindings are CPU NumPy arrays; scalar parameters take
concrete scalar values. Only parameterless kernels may omit the bindings.
Tensor-map parameters use {py:meth}`tirx_harness.numsim.TensorMap.numpy`.

```{eval-rst}
.. autoapifunction:: tirx_harness.synccheck
```

Returns a {py:class}`~tirx_harness.numsim.v2.report.SyncCheckReport`.

```{eval-rst}
.. autoapifunction:: tirx_harness.racecheck
```

Returns a {py:class}`~tirx_harness.numsim.v2.report.RaceReport`.

The signatures and report classes on this page are those of the redesigned
engine (pending: the root entry points switch from the legacy engine to
`tirx_harness.numsim.v2` when the migration completes).

### Synccheck budgets

The root `synccheck` uses the default exploration budgets. To set them, call
the engine-level entry point with a
{py:class}`~tirx_harness.numsim.v2.api.ResourceLimits`:

```{eval-rst}
.. autoapifunction:: tirx_harness.numsim.v2.api.synccheck
```

```{eval-rst}
.. autoapiclass:: tirx_harness.numsim.v2.api.ResourceLimits
   :members:
   :undoc-members:
```

All fields are required non-negative integers. The explorer enforces
`max_backtrack_nodes` (explored states), `max_loop_steps` (transitions), and
`max_wall_time_ms`; the other fields are recorded in the report. Exhausting a
budget gives `incomplete` with reason `resource_limit`, never `clean`.

```{eval-rst}
.. autoapiclass:: tirx_harness.numsim.v2.api.CoverageBounds
   :members:
   :undoc-members:
```

`CoverageBounds` is accepted for compatibility and ignored: the explorer
covers every interleaving the synchronization dependencies allow, with no
preemption or completion-deviation bound.

## Reports

Inspect `report.verdict` and `report.findings`, save `report.to_dict()`, or call
`report.print()` for readable evidence. `report.require_clean()` raises unless
the verdict is `clean` (pending: the redesigned reports raise
`AssertionError`; the legacy reports raised `CheckFailed`, a `RuntimeError`
subclass).

| Verdict | Meaning |
| --- | --- |
| `clean` | The invocation passed within the checker's supported model and coverage. |
| `review` | An advisory needs investigation. |
| `incomplete` | Unsupported behavior, an input the checker could not certify, or a coverage limit prevented a complete check. |
| `error` | The checker detected a violation. |

Verdict precedence is `error > incomplete > review > clean`. Each finding has
`id`, `status`, `kind`, `message`, and `details` fields, plus `to_dict()` for
serialization. Details carry available source and witness evidence. A clean
report applies to the supplied specialization, launch, and inputs; see the
[checker scope](../components/tools.md#checker-api).

`to_dict()` returns `schema_version` 5. It holds the overall `verdict`, the
flattened `findings`, and one payload per kernel launch under `phases`. Each
phase payload separates `findings` (errors), `advisories` (`review` items),
`incomplete` reasons, and `execution_error` (the engine stopped the launch).
{py:func}`tirx_harness.numsim.v2.report.payload_json_schema` returns the JSON
schema of a phase payload. Each payload also carries `timing` in wall-clock
milliseconds; one engine run serves every launch of an invocation, so its
`run` and `check` times repeat on each phase.

The following classes are return types; obtain them from the entry points above.

```{eval-rst}
.. autoapiclass:: tirx_harness.numsim.v2.report.SyncCheckReport
   :members: verdict, findings, to_dict, format, print, require_clean
   :undoc-members:
```

```{eval-rst}
.. autoapiclass:: tirx_harness.numsim.v2.report.RaceReport
   :members: verdict, findings, to_dict, format, print, require_clean
   :undoc-members:
```

```{eval-rst}
.. autoapifunction:: tirx_harness.numsim.v2.report.payload_json_schema
```
