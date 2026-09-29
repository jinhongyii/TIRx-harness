# Generated-code inspection

Import these helpers from `tirx_harness.dump_kernel`. They accept a compiled
TVM executable or runtime module and expose CUDA source, Parallel Thread
Execution (PTX) assembly, native GPU assembly (SASS), and compiler resource
information. See the [compilation example](../components/tools.md#inspecting-generated-code).

## Extract code

```{eval-rst}
.. autoapifunction:: tirx_harness.dump_kernel.dump_module
```

The requested assembly stages require CUDA toolkit executables. Set `arch`
explicitly when inspecting code for a GPU other than the local device.
`sass=True` by default; use `ptx=False, sass=False` to extract only CUDA source,
or call `dump_cuda` directly.

```{eval-rst}
.. autoapifunction:: tirx_harness.dump_kernel.dump_cuda
```

## Results

```{eval-rst}
.. autoapiclass:: tirx_harness.dump_kernel.DumpResult
   :members:
   :undoc-members:
```

`ptxas` is the CUDA assembler's resource report. Its `smem` entry describes
static shared memory; an absent entry does not imply zero dynamic shared-memory
use. Check `ok` and `errors` before consuming a requested artifact. A failure
to extract the initial CUDA source raises `ValueError` directly.

## Inspect existing text

```{eval-rst}
.. autoapifunction:: tirx_harness.dump_kernel.extract_symbols
```

```{eval-rst}
.. autoapifunction:: tirx_harness.dump_kernel.parse_ptxas
```
