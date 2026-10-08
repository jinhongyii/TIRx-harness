# Python API

This reference covers the `tirx_harness` Python API.

Start with [Installation](../installation.md) and the runnable
[compiler analysis examples](../components/tools.md). Use the reference when
integrating checks, simulation, or generated-code inspection into your own loop.

| Task | Reference |
| --- | --- |
| Check synchronization and memory-access ordering | [Checkers and reports](checkers.md) |
| Execute a kernel on the CPU and compare outputs | [Numerical simulation](numsim.md) |
| Describe inputs, tensor maps, and comparisons | [Simulation inputs](inputs.md) |
| Extract generated code and compiler resource information | [Generated-code inspection](inspection.md) |

The reference focuses on callable tools and the objects their callers supply
or receive. NumSim and the checkers are documented for the redesigned engine,
whose implementation lives in `tirx_harness.numsim.v2` until the migration
completes (pending: the public names switch to v2 when the legacy engine is
deleted). Generated pages link to that module path.

```{toctree}
:maxdepth: 1

checkers
numsim
inputs
inspection
```
