# Compiler analysis

```{container} component-subtitle
Domain-specific compiler analysis
```

```{container} lead
Inspect numerical behavior, synchronization, and memory races directly from TIRx.
```

## Key idea

TIRx Harness provides compiler analysis through the
`tirx_harness` Python package. Three CPU tools share the same TIRx execution model:
**NumSim** enables numerical iteration without GPU access, **Synccheck** checks
synchronization before GPU execution, and **Racecheck** checks memory-access
ordering. Each consumes a
kernel and concrete inputs, so an agent can investigate a candidate before
running it on a GPU. The package also exports generated-code inspection.

NumSim, Synccheck, and Racecheck run on the CPU. See the
[installation guide](../installation.md#install-python-packages) to get started.
The [Python API reference](../api/index.md) provides generated signatures,
defaults, and return types for these tools.

All three tools share one pipeline. Python lowers the TIRx function to a
`Program`, a compact bytecode of warp-wide instructions. One Rust engine
executes that program, and the two checkers observe the execution without
changing it. A kernel that runs under NumSim therefore runs the same
instructions, values, and control flow under Synccheck and Racecheck. The
[architecture overview](../development/architecture.md) describes the
components for contributors.

```{note}
This page describes the redesigned engine. Until the migration completes,
`tirx_harness.numsim`, `tirx_harness.synccheck`, and
`tirx_harness.racecheck` still run the legacy engine, and the redesigned
engine is importable under the same names from `tirx_harness.numsim.v2`.
```

## NumSim

**When to use it.** Use NumSim when getting a real GPU run is costly or
inconvenient:

- **Shared or scarce GPUs:** keep iterating while a job waits in the queue or
  the target GPU is occupied.
- **Multi-GPU workloads:** investigate a supported individual kernel's
  computation before reserving the devices and launching the full distributed
  job.
- **CPU-only development and CI:** run supported numerical regression cases
  on a development machine or CI worker without assigning a GPU to every edit.
- **Many agent-generated candidates:** screen small cases on CPU before
  submitting candidates to a remote or metered GPU service.

NumSim moves numerical debugging into the local development loop. With small,
representative inputs and an independent reference, it can identify output
mismatches by index and value before you spend a GPU allocation on the
candidate. Resolve synchronization and race findings first; use the target
GPU to validate selected candidates and measure performance.

**Mechanism.** NumSim works in two stages:

1. **Lower.** Python walks the specialized TIRx function and emits a
   validated `Program`: warp-wide instructions over 32-lane registers,
   structured control flow (`if`/`else` and loops with an active-lane mask),
   a constant pool, and a table of source sites. Tile operations are lowered
   through TVM's own tile dispatch, so NumSim runs the instructions the GPU
   would run. The lowered module is cached on disk, keyed by the TIRx source
   and the lowering version.
2. **Execute.** The Rust engine runs the program. Its scheduler visits every
   resident thread block (CTA) in rounds and gives each runnable warp a
   bounded slice of instructions in a seeded order. Blocking instructions,
   such as barrier waits, are retried on later rounds. Asynchronous copies
   land after a seeded delay. Every synchronization protocol (mbarrier, named
   and cluster barriers, async groups, tcgen05, `setmaxnreg`) is one state
   machine shared by all three tools. Memory keeps one validity bit per byte,
   so reading bytes that nothing wrote is reported. Numerical results come
   from one operation library.

NumSim returns output arrays and diagnostics. Comparing them with an
independent reference identifies numerical mismatches.

The engine interprets the program directly, so no Rust compiler is involved
at run time. The interpreter is the only executor.

**Runnable example.** Save this kernel as `vector_add.py`:

```python
import tirx_kernels.tirx_lite as txl


@txl.kernel(warps=1, arch="sm_100a", grid=1)
def kernel(a: txl.gptr(txl.f32), b: txl.gptr(txl.f32), out: txl.gptr(txl.f32)):
    lane = txl.lane_id()
    x = txl.local_scalar("float32")
    y = txl.local_scalar("float32")
    txl.ptx.ld.global_.f32(x, a.ptr_to([lane]))
    txl.ptx.ld.global_.f32(y, b.ptr_to([lane]))
    txl.ptx.st.global_.f32(out.ptr_to([lane]), x + y)
```

**API.** Import `numsim` from `tirx_harness`. For a TIRx-lite vector-add kernel
with three 32-element `float32` buffers named `a`, `b`, and `out`, bind concrete
NumPy arrays and compare the result with a NumPy reference:

```python
import numpy as np
from tirx_harness import numsim
from vector_add import kernel

a = np.arange(32, dtype=np.float32)
b = np.ones(32, dtype=np.float32)
expected = a + b

module = numsim.transpile(kernel.func)
result = numsim.Engine().run(
    module,
    inputs={"a": a, "b": b, "out": np.zeros_like(a)},
    outputs=("out",),
)
np.testing.assert_allclose(
    result.outputs["out"], expected, rtol=1e-5, atol=1e-8, equal_nan=False
)
```

Save this as `check_numsim.py` beside `vector_add.py` and run
`python check_numsim.py`.

Dictionary keys must match kernel parameter names; include output buffers
alongside input arrays and any scalar parameters. Shape parameters that a
buffer's shape determines are filled from the bound array. Choose
tolerances according to the workload; use `np.testing.assert_array_equal`
when exact equality is required. Decode buffers carrying encoded values
before comparing their numerical contents.

| Interface | Main parameters | Result |
| --- | --- | --- |
| `numsim.transpile(func, *, cache_dir=None)` | `func`: specialized TIRx function, or a sequence of them for a multi-kernel launch. `cache_dir`: optional module-cache root; the default is `NUMSIM_CACHE_DIR`. | `CompiledModule` |
| `numsim.Engine(max_workers=1, *, seed=None)` | `max_workers`: threads that run independent clusters in parallel; results do not depend on it. `seed`: scheduler seed. | Execution engine |
| `engine.run(module, inputs, *, outputs=None)` | `module`: transpiled module. `inputs`: concrete binding dictionary. `outputs`: buffer names or a mapping from result names to buffer names; `None` selects bound output buffers. | `NumSimResult` |

`NumSimResult` exposes `.outputs`, `.diagnostics`, `.stats`, and `.timing`
(wall-clock milliseconds for lowering, binding, build, run, and report). Keep
simulator diagnostics alongside the workload's numerical comparison result.

See the [NumSim API reference](../api/numsim.md) for full signatures.

**Limitations.**

- Only modeled TIRx operations and their supported dtype, shape, and modifier
  combinations can execute. Lowering rejects any other form with
  `UnsupportedTIRxError` instead of guessing. Opaque CUDA bodies are
  unsupported, and so is a tile operation that TVM's tile dispatch cannot
  lower. Consult the
  {repo}`operation coverage table <tirx_harness/src/tirx_harness/numsim/engine-rs/SUPPORTED_OPS.md>`.
- Hardware timing and some instruction results use deterministic
  representatives. Simulation time is not GPU latency. Transcendental math
  that the legacy simulator did not model (for example `sin`, `cos`, `tanh`,
  `exp2`, `pow`) uses the host math library, not CUDA's device library, so it
  can differ in the last bits. Matrix multiply-accumulate operations sum each
  output as one increasing-K chain of fused multiply-adds.
- One run follows one seeded schedule. Change the seed (`Engine(seed=...)` or
  `NUMSIM_SEED`) to see other asynchronous-completion timings; the
  checkers below cover the other orders.
- Clusters run in parallel only when the launch has no launch-wide state. A
  launch that uses grid synchronization or a cooperative launch, polls memory
  with `wait_until`, or mixes tcgen05 `cta_group` sizes runs on one thread.
- A warp that blocks inside one branch of a divergent `if` can resume the
  other branch only when that `if` has an `else`. An `if` without an `else`
  whose skipped lanes would release the blocked lanes, or a loop whose exit
  differs between lanes while some lanes wait, stops the run as `incomplete`
  (`divergent_block`), never as a deadlock.
- Loop-iteration and scheduler-round budgets stop a run as `incomplete`.
  Launch subsets select whole clusters; thread-block subsets are not
  supported.
- NumSim is not a numerical oracle or a race/synchronization verifier. Use
  independent reference outputs, the two checkers below, and device tests.

## Synccheck

**When to use it.** Run Synccheck as a CPU precheck before submitting a
candidate to the GPU. Repeat it after kernel changes, especially changes to
barrier arrivals and waits, pipeline stage reuse, or warp roles. Its findings
help you fix synchronization protocol errors before spending GPU time on a
launch that could hang.

**Guarantee.** Within its supported model, Synccheck detects:

- Barrier use without initialization ordered before it, and reuse of an
  mbarrier phase before the previous phase was consumed by a successful wait.
- Invalid barrier participants, arrival counts, or transaction counts. Every
  non-exited thread of a warp must take part in a named or cluster barrier;
  a single elected lane does not count for its warp.
- Tensor-memory (TMEM) allocation errors: deallocation that does not match
  its allocation, and inconsistent `cta_group` use. An allocation that cannot
  be satisfied yet waits, as on hardware, instead of failing.
- `setmaxnreg` errors: wrong direction, missing warpgroup synchronization,
  and register-pool deadlocks.
- Synchronization deadlocks, schedules that end in different final protocol
  states, and inconsistent final protocol states.

The check covers **all interleavings allowed by program order and
synchronization dependencies**, including when each asynchronous completion
arrives and the order of register-pool grants. Each warp's executed path and
values are held fixed. It therefore detects synchronization errors that
another warp/completion order can expose, even when the CPU simulation
happens to finish successfully.

Exits follow the PTX rules. Exited threads leave cluster barriers and named
barriers that use the default thread count, which can complete a pending
phase. A `bar.arrive` with no matching completion at exit, or `cp.async`
copies never committed to a group, is a `review` advisory, not an error.
Bulk asynchronous copies left uncommitted at exit are committed implicitly.

**Mechanism.** The algorithm has four steps:

1. **Run the kernel once.** Execute supported computations, memory accesses,
   and data-dependent branches and loops on the CPU, and record each warp's
   synchronization commands with their resolved barriers, counts, byte totals,
   and parities. Each asynchronous operation is recorded with the barriers it
   will signal. A protocol error in this run is reported directly.
2. **Build a reference schedule.** Replay one complete order of the recorded
   commands to number each barrier phase and record which commands are
   ordered before which.
3. **Split and certify.** Check each synchronization resource (a barrier, an
   async group, a register pool) separately, keeping the ordering
   constraints the other resources impose. For common patterns, such as a
   pipeline ring of mbarriers, a certificate proves that every phase pairs
   the same arrivals, completions, and waits in every schedule; a certified
   resource needs no search. Resources with an identical command pattern
   are checked once.
4. **Explore the remaining patterns.** Keep each warp's next command, the
   resource state, and the pending completions. Try every enabled command or
   completion, including orders different from the CPU run, and merge
   equivalent states and orders that cannot affect each other. Report
   protocol violations with a witness schedule, deadlocks where no command
   can progress, and schedules that end in different states.

For example, warp A initializes a barrier and warp B uses it. The checker
requires a dependency chain guaranteeing that A's initialization precedes
B's use. A merely running first in the CPU simulation does not establish
that guarantee.

The check fails closed. It reports `incomplete` when a state, transition, or
wall-time budget runs out, or when some schedule would let a wait pass on a
different barrier phase than the reference schedule. A wall-time cut depends
on host speed, but it can only produce `incomplete`.

**API:** `synccheck(kernel, inputs=None)` returns a `SyncCheckReport`.

Bind NumPy arrays by kernel parameter name, including output buffers. For
example, for three 32-element `float32` buffers named `a`, `b`, and `out`:

```python
import numpy as np
from tirx_harness import synccheck
from vector_add import kernel

report = synccheck(
    kernel.func,
    inputs={
        "a": np.arange(32, dtype=np.float32),
        "b": np.ones(32, dtype=np.float32),
        "out": np.zeros(32, dtype=np.float32),
    },
)
report.print()
report.require_clean()
```

Save this as `check_synccheck.py` beside `vector_add.py` and run
`python check_synccheck.py`.

Both arguments and the report interface are described in
[Checker API](#checker-api).

**Limitations.** Data-dependent execution is supported, but the verification
covers the synchronization program selected by this invocation. Alternate
inputs, ordinary-memory values, atomic return orders, and the different
control-flow paths they might select are not enumerated. The results of
polls that succeeded and the TMEM addresses returned by allocations are also
fixed by the run; a schedule in which an allocation would return a different
address is reported as an error. A named barrier with an explicit thread
count that waits on warps that already exited is reported as `incomplete`.

## Racecheck

**When to use it.** Run Racecheck as a CPU precheck before submitting a
candidate to the GPU. Repeat it after changing memory-access patterns,
buffer reuse, or asynchronous transfers. It helps catch missing synchronization
between memory accesses before you spend GPU time on results that may depend
on execution order.

**Guarantee.** Within its supported model, Racecheck detects:

- Read/write and write/write conflicts in global memory, shared memory
  (including another CTA's shared memory in the cluster), or TMEM that lack
  the required ordering, including accesses through aliases of the same
  storage and host arrays bound to overlapping memory.
- Missing memory ordering, even when the execution order is established:
  - a release/acquire pair whose scopes do not include both threads. An
    mbarrier arrive or wait without a scope qualifier is `.cta`, so it does not
    order a waiter in another CTA;
  - an mbarrier wait with `.relaxed` semantics, which orders nothing until a
    later acquire fence in the same thread;
  - a missing proxy fence between ordinary accesses and asynchronous-proxy or
    tensor-map accesses.
- Asynchronous-completion errors. A `cp.async` or bulk group wait orders only
  the waiting thread's own copies. `cp.async.bulk.wait_group.read` makes only
  the source safe to reuse, not the destination safe to read. tcgen05 work
  is complete only after its wait or after its commit is observed through an
  mbarrier.
- Accesses outside a buffer view or its backing allocation.
- Reuse or release of memory still accessed by an unfinished asynchronous
  operation.
- Plain accesses that race on a word polled with `wait_until`.

Racecheck reports every race it finds; it does not stop at the first one.
Two strong accesses that the PTX memory model makes morally strong (same
scope coverage, same proxy, complete overlap), such as two atomics of the
same scope, are not a race.

For the accesses selected by this invocation, the check requires both
**execution ordering and the necessary memory-ordering dependencies**. It can
detect a race even when the CPU simulation produces the expected output.

**Mechanism.** The algorithm has three steps:

1. **Run the kernel once.** Execute supported computations, memory accesses,
   and data-dependent branches and loops on the CPU. Racecheck observes the
   run as two streams: each active lane's physical byte ranges with access
   kind, strength, scope, and proxy, and the synchronization events (barrier
   arrivals and waits with their scopes, fences, asynchronous issues and
   completions).
2. **Track ordering dependencies.** Keep vector clocks, compact records of
   which earlier events each actor is ordered after. Actors are threads and
   asynchronous operations; an asynchronous operation has separate
   read-complete and write-complete milestones. Each happens-before rule is
   taken from the PTX memory model: program order, barriers, release/acquire
   through the value read and through fences, asynchronous completions, and
   proxy and tcgen05 fences. For a `wait_until` poll, the ordering comes from
   the earliest write that satisfies the predicate, whichever write the run
   happened to observe.
3. **Check overlapping accesses.** Each allocation keeps a shadow of byte
   ranges holding the last write and the reads since it, in the style of the
   FastTrack race detector. A new access is checked against the overlapping
   ranges. A race is reported when two accesses overlap, at least one writes,
   neither happens before the other, and they are not morally strong.
   Physical bytes are compared, so different buffer names cannot hide an
   overlap.

For example, warp A writes a shared-memory value and warp B reads it. The
checker requires a dependency chain ordering the write before the read.
A merely writing first in the CPU simulation does not establish that
guarantee.

Racecheck fails closed: an event it cannot interpret, such as a barrier
whose memory-ordering qualifiers were lost, or a `wait_until` exit no
recorded write explains, is `incomplete`.

**API:** `racecheck(kernel, inputs=None)` returns a `RaceReport`.

Bind NumPy arrays by kernel parameter name, including output buffers. For
example, for three 32-element `float32` buffers named `a`, `b`, and `out`:

```python
import numpy as np
from tirx_harness import racecheck
from vector_add import kernel

report = racecheck(
    kernel.func,
    inputs={
        "a": np.arange(32, dtype=np.float32),
        "b": np.ones(32, dtype=np.float32),
        "out": np.zeros(32, dtype=np.float32),
    },
)
report.print()
report.require_clean()
```

Save this as `check_racecheck.py` beside `vector_add.py` and run
`python check_racecheck.py`.

Both arguments and the report interface are described in
[Checker API](#checker-api).

**Limitations.** Data-dependent execution is supported, but the check covers
the accesses selected by this invocation. Alternate inputs, ordinary-memory
values, atomic return orders, and the different control-flow paths or addresses
they might select are not enumerated. When clusters run in parallel, the
checker sees the accesses in an order consistent with what each cluster read;
if no such order exists, the result is `incomplete`. A bulk copy still in
flight when its CTA exits is treated as drained, because the PTX ISA does not
specify this case; a race between that copy and another CTA's later access is
still reported.

## Checker API

Synccheck and Racecheck share these public parameters:

| Parameter | Meaning |
| --- | --- |
| `kernel` | A TIRx `PrimFunc`; pass `.func` from a TIRx-lite `Kernel`. |
| `inputs=None` | Dictionary from parameter names to concrete scalars and CPU NumPy buffers, including output storage. Parameterless kernels may omit it; otherwise supply all runtime bindings. |

For tensor-map bindings, `tirx_harness.numsim.TensorMap(...).numpy()` constructs
the simulator's descriptor array. The
{repo}`binding types <tirx_harness/src/tirx_harness/numsim/cases.py>` define its
shape, strides, dtype, and swizzle parameters.

| Report interface | Meaning |
| --- | --- |
| `.verdict` | `clean`, `review`, `incomplete`, or `error`. |
| `.findings` | Structured findings with status, kind, message, and source/witness evidence. |
| `.to_dict()` | JSON-safe report for an agent or artifact store (`schema_version` 5). |
| `.print()` | Human-readable findings and available source context. |
| `.require_clean()` | Raise unless the verdict is `clean`. |

A verdict covers the supplied specialization, launch, and inputs. A
multi-kernel invocation is checked launch by launch, and the report's verdict
is the worst launch verdict. Unsupported effects and coverage limits produce
`incomplete`; `review` indicates an advisory, and `error` indicates a
detected violation. A clean report does not establish correctness for other
inputs or replace an independent GPU correctness test. The
{repo}`checker entry points <tirx_harness/src/tirx_harness/numsim/checkers.py>`
own these signatures.

Missing bindings, and forms that lowering rejects, also produce `incomplete`
in the legacy entry points.

## Inspecting generated code

Use generated-code inspection when an edit changes performance unexpectedly,
or to confirm the instructions and register, spill, and shared-memory usage
produced by compilation.

`tirx_harness.dump_kernel.dump_module` extracts CUDA, PTX, SASS, and compiler
resource information from a compiled module:

```python
from tirx_harness.dump_kernel import dump_module


def inspect_candidate(compiled_module, artifact_dir):
    result = dump_module(
        compiled_module, ptx=True, outdir=artifact_dir, name="candidate"
    )
    if not result.ok:
        raise RuntimeError(result.errors)
    return result.paths
```

Save both code blocks in `inspect_vector_add.py` beside `vector_add.py`,
then run `python inspect_vector_add.py`:

```python
from vector_add import kernel
from tvm.target import Target

target = Target({"kind": "cuda", "arch": "sm_100a"})
with target:
    compiled = kernel.compile(target=target)
print(inspect_candidate(compiled, "artifacts/vector-add"))
```

`ptx` and `sass` select artifact stages; CUDA source is always returned.
`outdir` and `name` control saved files. Requested stages
need their CUDA-toolkit executables and can fail independently. See the
{repo}`source-dump API <skills/tirx-profile-kernel/references/dump-source.md>`.

## External tools

TIRx Harness uses these external tools in its agent loop for GPU-side
debugging and profiling. The [debugging and profiling skills](../installation.md#install-agent-skills)
guide the agent to select a tool, capture its reports or traces, and use that
evidence to decide how to revise the kernel.

| Tool | Role in the agent loop |
| --- | --- |
| {repo}`Nsight Compute <skills/tirx-profile-kernel/references/ncu.md>` | The agent uses hardware-counter reports to identify stalls, utilization limits, and memory bottlenecks before choosing an optimization. |
| {repo}`IKET <skills/tirx-profile-kernel/references/iket.md>` | The agent uses annotated timelines to locate pipeline bubbles, missing overlap, and load imbalance. |
| {repo}`Compute Sanitizer <skills/tirx-debug-kernel/references/compute-sanitizer.md>` | The agent uses device-side error reports to investigate memory, initialization, race, or synchronization failures in the launched binary. |

The agent runs these tools on a local GPU or through the harness's
[kcoral adapters](kcoral.md#framework-adapters) for remote execution and
artifact retrieval. After revising the kernel, it reruns correctness checks
and the ordinary benchmark. Profiler replay and instrumentation timings
serve diagnosis; benchmark results determine performance improvements.

## Replace analysis checks

Add or replace checks for the properties you need, and return their results to
the agent. Keep the workload's independent numerical reference.
