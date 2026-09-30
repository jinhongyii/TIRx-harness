# Export a kernel

Export a kernel as CUDA C++ source with its host launch wrapper, then build and
run it with `tvm-ffi`. Consumers need no TVM, `tirx-kernels`, or `tirx-harness`
installation. You can also distribute a compiled shared library so consumers
can skip compilation.

This workflow uses TVM's
[`export_cuda_host`](https://github.com/apache/tvm/blob/ea0cfa320fcdea0567e2f4c19d363625c66630d1/python/tvm/backend/cuda/host.py)
API.

## Export in the development environment

Exporting requires a compatible CUDA
toolchain because `tvm.compile` also compiles the device code.

Save this as `export_zero.py`:

```python
from pathlib import Path

import tvm
from tvm.backend.cuda import export_cuda_host

import tirx_kernels.tirx_lite as txl


@txl.kernel(warps=1, arch="sm_100a", grid=1)
def zero(out: txl.gptr(txl.f32)):
    txl.ptx.st.global_.f32(out.ptr_to([txl.lane_id()]), txl.float32(0))


target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"}, host="cuda_host")
compiled = zero.compile(target=target)
Path("zero.cu").write_text(export_cuda_host(compiled.mod), encoding="utf-8")
```

```bash
python export_zero.py
```

For an existing TIRx-lite kernel, replace `zero` with your `Kernel` object.
For a native TIRx `PrimFunc` or `IRModule`, use
`tvm.compile(kernel, target=target, tir_pipeline="tirx")` and pass the result's
`.mod` to `export_cuda_host` in the same way.

The `.cu` file contains both device code and the host wrapper, including FFI
entry points and CUDA launches. Set `host="cuda_host"` before compiling;
`export_cuda_host` expects that module, with its retained device source.
`kernel.source()` and `tirx_harness.dump_cuda()` extract device code for
inspection and do not provide this complete launch wrapper.

## Build with tvm-ffi

Copy `zero.cu` to the consumer's project. In a separate Python environment,
install the FFI package with its build dependencies:

```bash
python -m pip install "apache-tvm-ffi[cpp]>=0.1.14.post0,<0.2"
```

This range starts at the tested baseline and allows updates within the `0.1.x`
ABI, following [TVM-FFI's versioning policy](https://github.com/apache/tvm-ffi#status-and-release-versioning).

Building requires a C++ compiler and a CUDA Toolkit with `nvcc` that supports
the kernel's architecture. Set `CUDA_HOME` if the toolkit is outside the
default location. Save this as `build_zero.py` beside `zero.cu`:

```python
from pathlib import Path

import tvm_ffi.cpp


library = tvm_ffi.cpp.build(
    name="zero",
    sources=["zero.cu"],
    build_directory=str(Path("build").resolve()),
    backend="cuda",
)
print(library)
```

Compile for the architecture used when exporting:

```bash
TVM_FFI_CUDA_ARCH_LIST=10.0a python build_zero.py
```

This produces `build/zero.so`. `tvm-ffi` supplies its own headers, DLPack
headers, and link options. The generated source already exports the kernel's
FFI entry point.

## Load and call the kernel

The application needs `apache-tvm-ffi`, a compatible NVIDIA driver and CUDA
runtime libraries, and a tensor library for its GPU inputs. This example uses
an existing CUDA-enabled PyTorch installation:

```python
from pathlib import Path

import torch
import tvm_ffi


module = tvm_ffi.load_module(str(Path("build/zero.so").resolve()))
out = torch.ones(32, dtype=torch.float32, device="cuda")
module["zero"](out)
torch.cuda.synchronize()
torch.testing.assert_close(out, torch.zeros_like(out))
```

The exported name comes from the TIRx function's `global_symbol` (`zero` here).
Pass tensor and scalar arguments in the original kernel's order. Preserve its
shape, dtype, layout, and device requirements. Launches use the CUDA stream
provided through `tvm-ffi`; synchronize before reading results on the CPU.

To ship a binary, distribute `zero.so` and use only the load-and-call step.
Install `apache-tvm-ffi>=0.1.14.post0,<0.2` without the `[cpp]` extra on machines
that only load the library. They need compatible GPU, CUDA runtime, host
platform, and FFI versions; they do not need `nvcc` or a C++ compiler.
Document those requirements and the kernel's argument contract alongside the
artifact. Shipping `.cu` lets consumers rebuild for their deployment environment.

## Kernels with additional requirements

- TMA tensor-map encoding must use `tirx.tensormap_encode_tiled`. The older
  `call_packed("runtime.cuTensorMapEncodeTiled", ...)` form is rejected with
  `cuda_host requires tensormap_encode_tiled instead of a packed tensor-map encoder`.
  TIRx-lite provides `txl.cu_tensor_map_encode_tiled(...)` for this purpose;
  `tirx-kernels` 0.1.2.post1 includes the migration for FP16/BF16 GEMM and
  DeepSeek-V4 MLA. If the generated wrapper encodes tensor maps, add
  `extra_ldflags=["-lcuda"]` to `tvm_ffi.cpp.build` to link the CUDA driver
  library. Any tensor-map arguments prepared outside the exported function
  still need equivalent preparation in the consumer application.
- Preserve any extra CUDA headers and compiler options required by your
  kernel. The export helper concatenates source; it does not package external
  headers or Python input preparation.
- CUDA-host code generation rejects unsupported launch modes and TVM runtime
  services, including runtime workspace allocation. Validate the exported
  artifact with the same inputs and correctness checks as the original kernel
  before distributing it.
