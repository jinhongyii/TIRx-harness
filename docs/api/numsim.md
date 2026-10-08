# Numerical simulation

NumSim is the CPU numerical simulator exposed by `from tirx_harness import
numsim`. It lowers a specialized TIRx function to a cached `Program` module,
executes it with concrete inputs on the Rust engine, and lets you compare its
outputs with an independent reference. See the [runnable example and
supported-model limitations](../components/tools.md#numsim).

The signatures below are those of the redesigned engine, which lives in
`tirx_harness.numsim.v2` until the migration completes (pending: the public
`tirx_harness.numsim` names switch to these objects when the legacy engine is
deleted). Import them through `tirx_harness.numsim` as shown in the examples.

## Compile and execute

```{eval-rst}
.. autoapifunction:: tirx_harness.numsim.v2.compile.transpile
```

Pass a TIRx function, such as a TIRx-lite kernel's `.func`, or a sequence of
functions for a multi-kernel launch. `cache_dir` selects the module-cache
root; modules are stored under `<cache_dir>/v2-modules/`. Without it, the
root is `NUMSIM_CACHE_DIR` (default `~/.cache/tirx-harness/numsim`). Keyword
parameters beginning with `_` are accepted for compatibility and ignored:
one module serves NumSim, Synccheck, and Racecheck.

```{eval-rst}
.. autoapiclass:: tirx_harness.numsim.v2.compile.CompiledModule
   :members: cache_key, cache_hit, lower_ms
   :undoc-members:
```

Obtain a compiled module from `transpile` and pass it to `Engine.run`. A module
is validated JSON bytecode; it contains no native code. `cache_hit` reports
whether it was loaded from the module cache, and `lower_ms` how long lowering
or loading took.

```{eval-rst}
.. autoapiclass:: tirx_harness.numsim.v2.run.Engine
   :members: run, address_of
   :undoc-members:
```

| Parameter | Meaning |
| --- | --- |
| `max_workers=1` | Threads that run independent clusters in parallel. Results and checker findings do not depend on it. `"auto"` currently selects one thread (pending: `"auto"` should use the detected CPU count). |
| `seed=None` | Scheduler seed for warp rotation and asynchronous-completion timing. `None` reads `NUMSIM_V2_SEED`, default `0`. A fixed module, inputs, and seed always give the same result. |
| `native_loop_iteration_budget=None` | Maximum iterations of one loop instance per warp before the run stops as `incomplete`. `None` uses the engine default, 2^24. |
| `native_loop_reschedule_quantum=None` | Maximum instructions a warp runs before the scheduler moves to the next warp. `None` uses the engine default, 256. |

For `run`, `inputs` maps kernel parameter names to concrete scalars and CPU
NumPy buffers, including output storage. Shape parameters that a bound
buffer's shape determines are filled from that array. Buffers bound to
overlapping host memory share one simulated allocation, so writes through one
name are visible through the other. Sub-byte data (FP4, FP6, 4-bit integers)
is bound packed as a contiguous `uint8` array. `outputs` selects buffer names
or maps result names to buffer names; `None` selects the bound output
buffers. The [input types](inputs.md) describe optional launch subsets.

`run` raises `InputError` for an unknown, missing, or inconsistent binding,
and `ExecutionError`, with the engine's diagnostics, when the launch does not
run to completion.

`address_of(module, inputs, name)` returns the simulated global address at
which a buffer argument will be bound for these inputs, for kernels that read
raw pointers from memory.

```{eval-rst}
.. autoapiclass:: tirx_harness.numsim.v2.report.NumSimResult
   :members: outputs, diagnostics, stats, timing, verdict, assert_close
   :undoc-members:
```

`outputs` contains the selected arrays, `diagnostics` contains simulator
messages, and `stats` contains execution statistics. The result's `verdict`
reflects simulator advisories, such as an `uninitialized_read` `review` for
bytes the kernel read before anything wrote them. Compare against a
reference to check numerical correctness. `assert_close` performs that
comparison and raises on a mismatch.

`timing` holds wall-clock milliseconds for each stage: `lower` (transpile or
cache load), `bind` (input canonicalization), `run` (engine execution),
`check` (0 for NumSim), and `report`. Simulation time does not measure GPU latency.

## Compare outputs

```{eval-rst}
.. autoapifunction:: tirx_harness.numsim.v2.report.compare
```

`expected` must be a nonempty mapping from output names to reference arrays.
Use `tolerances` to supply a {py:class}`~tirx_harness.numsim.ComparisonSpec` per
output. Without an explicit specification, integer and boolean arrays use
exact equality; floating-point comparisons use the specification's defaults.

```{eval-rst}
.. autoapifunction:: tirx_harness.numsim.v2.api.run_case
```

Compile, execute, and compare a {py:class}`~tirx_harness.numsim.NumSimCase` in
one call. Supply an existing engine to reuse its execution configuration.

```{eval-rst}
.. autoapiclass:: tirx_harness.numsim.v2.report.NumSimReport
   :members: ok, mismatches, diagnostics, verdict, require_ok
   :undoc-members:
```

`require_ok()` raises `AssertionError` when a numerical comparison fails.
An advisory can leave `ok=True` while `verdict` is `review`; keep the diagnostics
alongside the numerical result.

```{eval-rst}
.. autoapiclass:: tirx_harness.numsim.report.Mismatch
   :members: output, index, actual, expected, render
   :undoc-members:
```

## Environment variables

The engine reads its environment in one place. Explicit arguments take
precedence.

| Variable | Meaning | Default |
| --- | --- | --- |
| `NUMSIM_CACHE_DIR` | Cache root. Lowered modules are stored under `v2-modules/`. Delete the directory to force re-lowering. | `~/.cache/tirx-harness/numsim` |
| `NUMSIM_V2_SEED` | Default scheduler seed. | `0` |
| `NUMSIM_V2_NO_CACHE` | `1` disables the module cache. | unset |

(pending: the `NUMSIM_V2_` prefix may be shortened when the legacy engine is
deleted.)

## Exceptions

These exceptions describe lowering and execution failures. Checker entry
points instead return reports for the failures they handle; inspect their
verdicts as described in [Checkers and reports](checkers.md).

```{eval-rst}
.. autoapiexception:: tirx_harness.numsim.NumSimError
   :show-inheritance:
```

```{eval-rst}
.. autoapiexception:: tirx_harness.numsim.UnsupportedTIRxError
   :show-inheritance:
```

Lowering raises `UnsupportedTIRxError` for a TIRx form the engine does not
model; its `unsupported` attribute lists the reasons.

```{eval-rst}
.. autoapiexception:: tirx_harness.numsim.NumSimExecutionError
   :show-inheritance:
```

`InputError` (a binding problem, also a `ValueError`) and `ExecutionError` (the
launch stopped) both derive from `NumSimExecutionError`.
