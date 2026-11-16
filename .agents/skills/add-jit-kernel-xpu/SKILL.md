---
name: add-jit-kernel-xpu
description: Add or port a JIT kernel for Intel XPU (SYCL, built with icpx and loaded through tvm-ffi) in sglang.kernels -- the XPU build path, which JIT headers compile under icpx, the SYCL kernel and launcher pattern, Python dispatch next to the CUDA path, registry capabilities, and XPU tests. Use when writing a SYCL kernel for sglang.kernels.jit, porting a CUDA JIT kernel to XPU, or debugging an XPU JIT build.
---

# Adding a JIT Kernel for Intel XPU (SYCL)

This is the XPU companion of [add-jit-kernel](../add-jit-kernel/SKILL.md). Read
that skill first: its conventions (tuning policy in Python, checks in the C++
launcher, thin per-call entry points, Doxygen, ASCII-only sources) apply
unchanged. Apply [kernel-organization](../kernel-organization/SKILL.md) for
placement and registry metadata, and
[write-sglang-test](../write-sglang-test/SKILL.md) for test mechanics. This file
covers only what differs on XPU.

The reference implementation is the XPU RMSNorm, a port of the CUDA rmsnorm family
(plain, fused residual add, fp8 quant, Hugging Face rounding):

| Piece | File |
|---|---|
| SYCL kernel + launcher (port of `rmsnorm.cuh`) | `python/sglang/kernels/jit/csrc/elementwise/rmsnorm_xpu.hpp` |
| SYCL helpers | `python/sglang/kernels/jit/include/sgl_kernel/xpu/utils.hpp` |
| Python dispatch, shared with CUDA | `python/sglang/kernels/ops/layernorm/norm.py` (`_schedule_rmsnorm`, `_jit_rmsnorm_module`) |
| Mode enum shared with CUDA | `python/sglang/kernels/jit/include/sgl_kernel/impl/rmsnorm_mode.h` |
| Registry entry | `python/sglang/kernels/ops/layernorm/__init__.py` (`CUDA_ONLY \| _XPU`) |
| Shared tests (CUDA, ROCm, XPU) | `test/registered/kernels/ops/layernorm/test_rmsnorm.py` |
| XPU-only tests | `test/registered/kernels/ops/layernorm/test_rmsnorm_xpu.py` |

---

## How an XPU JIT build works

`load_jit` is the same entry point as on CUDA. On an XPU runtime
(`is_xpu_runtime()`, i.e. `torch.version.xpu` is set):

- **Sources:** pass `sycl_files=` / `sycl_wrappers=`. They take the slot that
  `cuda_files` / `cuda_wrappers` fill on CUDA. Passing `cuda_files` on XPU, or
  `sycl_files` anywhere else, raises `ValueError`.
- **Compiler:** every unit, host and device, is compiled and linked with
  `icpx -fsycl -fsycl-targets=<targets> -std=c++20 -O3 -DSGL_XPU=1`. icpx is found
  through `CMPLR_ROOT`, then `PATH`, then `ONEAPI_ROOT/compiler/latest/bin`.
- **Generated units:** the device wrapper is `sycl.cpp`, not `cuda.cu`, because
  icpx compiles a `.cu` file as CUDA. Extra `cuda_units` become
  `sycl_unit{i}.cpp`.
- **Exports:** the generated wrapper uses `TVM_FFI_DLL_EXPORT_TYPED_FUNC` inside
  `namespace sglang`, exactly as on CUDA, so the kernel side looks the same.
- **Target:** `SGLANG_JIT_SYCL_TARGETS` sets `-fsycl-targets`. Unset means
  `spir64`: portable SPIR-V that the driver finalizes for the device at first
  load. Set it to an `intel_gpu_*` name your icpx accepts (for example
  `intel_gpu_bmg_g21`) to compile ahead of time. `spir64_gen` alone does not
  work, because it also needs `-Xs "-device <name>"`, which the env var cannot
  pass.
- **Cache:** builds land under `<cache root>/xpu_<targets>/`. The key also
  fingerprints the device name and driver version, so a driver update rebuilds.
- **Line info:** `SGLANG_JIT_ENABLE_LINEINFO` adds `-g` on XPU (icpx keeps no
  line tables alone for SPIR-V device code), the device debug info VTune and
  gdb-oneapi map source lines from.
- **Not on XPU:** `SGLANG_JIT_LOG_RESOURCE_USAGE` is ignored, `SGLANG_JIT_DEBUG_TRACE`
  has nothing to trace (the hooks live in `debug/trace.cuh`), and there is no PDL
  (`is_arch_support_pdl()` is False).

## Headers: what compiles under icpx

`.cuh` headers are CUDA/HIP only; do not include them. Under icpx you have:

- **`sgl_kernel/xpu/utils.hpp`**, the SYCL counterpart of `utils.cuh`:
  - `fp32_t` / `fp16_t` / `bf16_t` bound to `float` / `sycl::half` /
    `sycl::ext::oneapi::bfloat16`. `make_cpp_args(torch.float16)` still renders
    `fp16_t`, so the Python side does not change.
  - `fp8_e4m3_t`, one `torch.float8_e4m3fn` value as storage, and
    `device::cast_fp8_e4m3(float)`: nearest-even, saturating to +-448, bit-identical
    to torch's `clamp(-448, 448).to(torch.float8_e4m3fn)`. It is software, so it
    needs no fp8 device aspect (the `sycl::ext::oneapi::experimental::fp8_e4m3` type
    does, and converts only from `half` / `bfloat16`).
  - `device::kSubGroupSize` (16), `device::AlignedVector<T, N>`, and
    `device::sub_group::reduce_sum<kLanes>(sg, value)`, the xor butterfly of
    `warp::reduce_sum<kLanes>` in the same order.
  - `host::LaunchKernel(num_groups, group_size, device)(functor)`, which submits
    an `nd_range<1>` to the current queue and rethrows SYCL errors as tvm-ffi
    `RuntimeError`s.
  - `host::get_current_sycl_queue(device)` and `SGL_CHECK_SYCL(stmt)` for
    anything `LaunchKernel` does not cover.
- **Host-only headers that also compile under icpx:** `sgl_kernel/utils.h`
  (`CHECK_HOST`, `div_ceil`), `sgl_kernel/bits.h` (`is_pow2`; it takes unsigned
  types, so cast an `int` template parameter), and `sgl_kernel/tensor.h`.
  `tensor.h` knows the SYCL fp16/bf16 types and `fp8_e4m3_t`, so `TensorMatcher`
  works unchanged with `.with_device(device)` after
  `device.set_options<kDLOneAPI>()`.

## Streams

Never create a `sycl::queue`. When a module is called with `xpu` tensors, tvm-ffi
sets the stream of the `kDLOneAPI` device to the current `torch.xpu` stream's
`sycl::queue*`. That holds inside `torch.xpu.stream(s)` and
`tvm_ffi.use_torch_stream(torch.xpu.stream(s))` too, and was verified with
torch 2.15. `LaunchKernel` reads that queue through `get_current_sycl_queue`,
which raises `No SYCL queue is set` if none was passed. If you support an older
torch, recheck this: tvm-ffi's own torch DLPack add-on only reports CUDA and ROCm
streams.

---

## Step 1: The SYCL kernel in `kernels/jit/csrc/`

Name the file after its CUDA sibling with an `_xpu` suffix and a `.hpp`
extension, e.g. `csrc/elementwise/<name>_xpu.hpp`, and put everything in
`namespace sglang`.

**When a CUDA kernel exists, port it one-to-one:** the same tile and schedule
parameters, the same parameter struct, the same launcher name, template parameters
and call signature, and the same order of floating-point operations, line for
line. `rmsnorm_xpu.hpp` mirrors `rmsnorm.cuh` this way. The translation:

| CUDA (`*.cuh`) | SYCL (`*_xpu.hpp`) |
|---|---|
| `__global__` kernel function, `__launch_bounds__`, `__grid_constant__` params | a functor with `void operator()(sycl::nd_item<1>) const`, holding the params by value |
| `threadIdx.x`, `blockIdx.x` | `item.get_local_id(0)`, `item.get_group(0)` |
| warp (32 lanes), `warp::reduce_sum<kLanes>(v)` | sub-group (16 lanes), `device::sub_group::reduce_sum<kLanes>(item.get_sub_group(), v)` |
| `__shared__ float s[R][W];` | `auto& s = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[R][W]>(item.get_group());` |
| `__syncthreads()` | `sycl::group_barrier(item.get_group())` |
| `fmaf`, `math::rsqrt` | `sycl::fma`, `sycl::rsqrt` |
| packed pairs (`T2`, `AlignedVector<T2, kPacks>`) | `AlignedVector<T, kVecElems>` of scalars; process the elements in the same order |
| `cast<fp8x2_e4m3_t>` (saturating) | `device::cast_fp8_e4m3` |
| `LaunchKernel(grid, block, dev).enable_pdl(kUsePDL)(kernel, params)` | `LaunchKernel(num_groups, group_size, dev)(Kernel{params})`; keep `kUsePDL` in the template and `static_assert(!kUsePDL)` |
| `device_sym.set_options<kDLGPU>()` | `device_sym.set_options<kDLOneAPI>()` |
| Blackwell / arch-specific paths (`kMixedFMA`, 32 B vectors) | drop; the generic path is the order to follow |

Where the hardware forces a difference, keep it as small as possible and state it
in a comment. In rmsnorm the CUDA warp is 32 lanes and the XPU sub-group 16, so a
32-thread row reduces in two sub-groups plus one local-memory step: the same
operations in the same order inside each lane, a different reduction tree across
lanes. 16-lane sub-groups are the native choice on Intel GPUs, and a 32-lane
sub-group halves the registers each lane has for the row it holds.

**Without a CUDA sibling,** follow the same layout anyway: a params struct of
plain data (the functor is copied to the device: no references, no `TensorView`),
a device functor with `[[sycl::reqd_sub_group_size(device::kSubGroupSize)]]`, and
a host struct whose `static void run(...)` validates with `TensorMatcher` and
launches with `LaunchKernel`.

Rules that apply either way:

- **Every knob is a template parameter**, chosen in Python, and the C++ side only
  `static_assert`s that the values are consistent.
- **Accumulate in fp32** and convert with `static_cast<float>` /
  `static_cast<T>`; `sycl::half` and `bfloat16` convert both ways.
- **Share mode enums with the CUDA kernel through a host-only header.** A `.cuh`
  cannot be included under icpx, so `RMSNormMode` lives in
  `sgl_kernel/impl/rmsnorm_mode.h`, included by both kernels.
- **Check the dtype in C++** with a `static_assert` whose message names the
  dtype, as the CUDA template does. The failing build surfaces in Python as a
  `RuntimeError` carrying that message.

## Step 2: Python dispatch in `kernels/ops/`

With a one-to-one port, the Python side is the CUDA side's. The public functions,
the schedule (`_schedule_rmsnorm`), the module factory (`_jit_rmsnorm_module`),
`precompile_rmsnorm` and `RMSNormDispatch` all serve both platforms. XPU branches
only where it must:

- **The source file**, in the module factory:

  ```python
  wrappers = [("rmsnorm", f"RMSNormKernel<{args}>::run")]
  if is_xpu_runtime():
      load_jit_kwargs = dict(sycl_files=["elementwise/rmsnorm_xpu.hpp"], sycl_wrappers=wrappers)
  else:
      load_jit_kwargs = dict(cuda_files=["elementwise/rmsnorm.cuh"], cuda_wrappers=wrappers)
  return load_jit("rmsnorm", *args, **load_jit_kwargs)
  ```

  One `load_jit` call keeps the module name and template arguments identical
  across devices; only the source keyword differs.

- **A schedule input the hardware decides differently**, inside the schedule
  function. rmsnorm has two:
  - the vector width: CUDA always uses 16 B (32 B on Blackwell), while XPU takes the
    widest power of two that divides the row, up to 16 B, so hidden sizes that are
    not a multiple of 16 B still run;
  - rows of those narrower vectors can need more vectors per work-item than CUDA's
    thread caps allow, so XPU widens such a row to keep each work-item at 8 vectors
    or fewer, up to 512 work-items. 512 is this kernel's own work-group limit,
    below the device's 1024, because it holds rows in registers. The XPU tile
    bounds a work-item by bytes (256, CUDA's 16 vectors of 16 B) rather than by
    vector count, so the widest odd rows (32 vectors of 2 B) still fit.

Everything else in the schedule (threads per row, rows per work-group, the
decode/prefill threshold, per-size overrides) applies unchanged: every fp16 / bf16
schedule with 16 B vectors equals CUDA's. Only fp32, which the CUDA kernel does not
take, can be widened at the same hidden size (from 8196 up, and 3072).

Without a CUDA factory to share, add a `@cache_once` XPU module factory and branch
once at the top of the public function, with one `is_xpu_runtime()` check. A mode
or option the XPU kernel does not implement then raises `NotImplementedError`
there, before compiling. Do not silently ignore it: it changes the numbers.

## Step 3: Registry capabilities

Declare XPU on the entries that the XPU path actually implements, and only
those:

```python
_XPU = frozenset({CapabilityRequirement.XPU})
...
capabilities=(
    CUDA_ONLY | _XPU
    if _fn in ("rmsnorm", "fused_add_rmsnorm")
    else CUDA_ONLY
),
```

On XPU, `PlatformInfo.detect()` reports `xpu`, so `select_kernel(op)` finds the
entry. Check it with
`registry.get_backend(op, KernelBackend.JIT).is_available(PlatformInfo.detect())`.

## Step 4: Tests

Run the operator's existing tests on XPU too, and put only what they cannot cover
in an XPU-only file. Both reference implementations then stay checked against the
same spec.

**The shared test file** (`test/registered/kernels/ops/<group>/test_<name>.py`):

- Register XPU next to CUDA and ROCm, and take the device from the platform:

  ```python
  from sglang.srt.utils import get_device, is_xpu
  from sglang.test.ci.ci_register import register_xpu_ci

  register_xpu_ci(est_time=600, suite="stage-b-test-1-gpu-xpu")
  DEVICE = get_device()
  ```

  Keep `est_time` and `suite` literal: CI parses them from the AST.
- Skip, on XPU, the modes the XPU kernel does not implement, with
  `pytest.mark.skipif(is_xpu(), reason=...)`. If one test mixes a supported and an
  unsupported mode, split it so the supported half still runs.
- If the file prebuilds modules for CUDA, build the XPU factory's modules on XPU
  instead, with the same `parallel_compile`. A CUDA prebuild fails on XPU, and
  an autouse fixture would then error every test.

**The XPU-only file** (`test/registered/kernels/ops/<group>/test_<name>_xpu.py`):

```python
from sglang.srt.utils import is_xpu
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=120, suite="stage-a-test-1-gpu-xpu")
pytestmark = pytest.mark.skipif(not is_xpu(), reason="XPU required")
```

Put here what only the XPU kernel has:

- **Its own dtypes and shapes.** Cover what the shared file's shapes cannot reach
  on XPU: XPU-only dtypes, rows that run narrower vectors than CUDA's, short rows
  packed several to a work-group, and threads whose last vector falls past the
  row.
- **Its negative branches**: misaligned row strides (expect `aligned`) and
  unsupported dtypes.
- **Rounding modes, separated with exact matches.** Two modes that differ by one
  ulp pass the same ulp bound. Assert that most elements, for example at least
  99%, equal the requested mode's reference exactly, and that the output is not
  identical to the other mode's reference. Measure the match rates on the device
  first: in RMSNorm each mode matches its own reference 100% and the other about
  74%.

In both files, check 16-bit outputs with an ulp bound (`_assert_close_16` in
`test_rmsnorm.py`), not just a loose `assert_close`. In XPU-only tests, compute
the reference and compare on CPU (`out.cpu()`), so the reference does not depend
on the device under test.

## Step 5: Benchmark with XPU graphs

`marker.do_bench` is CUDA-only today (`torch.cuda.CUDAGraph`, `torch.cuda.Event`).
On XPU, time the kernel inside an XPU graph directly; tvm-ffi JIT kernels capture
like any torch op, because they launch on the current stream's queue. Follow
[`rules/kernel-benchmark.md`](../../rules/kernel-benchmark.md): rotate enough input
copies to exceed the last-level cache several times over, so each call runs with a
cold cache.

```python
import torch

from sglang.kernels.ops.layernorm.norm import rmsnorm


def bench_xpu_us(fn, make_args, *, replays: int = 20) -> float:
    """Average time of one `fn(*args)` call, from an XPU graph over rotated inputs."""
    args = make_args()
    fn(*args)  # JIT-compile and warm up outside the capture
    llc = torch.xpu.get_device_properties().last_level_cache_size
    nbytes = sum(a.numel() * a.element_size() for a in args if isinstance(a, torch.Tensor))
    copies = [args] + [make_args() for _ in range(max(1, 4 * llc // nbytes))]
    torch.xpu.synchronize()

    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        for copy in copies:
            fn(*copy)
    graph.replay()  # warm up the graph

    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    torch.xpu.synchronize()
    return start.elapsed_time(end) * 1000 / (replays * len(copies))


def make_args(num_tokens=4096, hidden_size=4096, dtype=torch.bfloat16):
    x = torch.randn(num_tokens, hidden_size, device="xpu", dtype=dtype)
    w = torch.randn(hidden_size, device="xpu", dtype=dtype)
    return x, w, torch.empty_like(x), 1e-6


print(f"rmsnorm: {bench_xpu_us(rmsnorm, make_args):.2f} us")
```

Report bandwidth from the bytes the kernel touches (here read `x` and `w`, write
`out`).

## Running locally

With a oneAPI environment (icpx on `PATH`) and an XPU build of torch:

```bash
python -m pytest -v test/registered/kernels/ops/<group>/test_<name>_xpu.py
SGLANG_JIT_FORCE_RECOMPILE=1 python -m pytest ...   # rebuild instead of using the cache
```

After changing tvm-ffi's C++ code, reinstall it (`pip install --no-build-isolation
--no-deps -e .`). An editable install reads the Python files live, but the
compiled `libtvm_ffi.so` is a copy made at install time. Add `--no-index` if your
pip index is unreachable; this install needs nothing from it.

## Pitfalls

| Symptom | Cause / fix |
|---|---|
| `CUDA sources cannot be built for XPU, pass sycl_files.` | Pass `sycl_files` / `sycl_wrappers` on XPU. |
| `No SYCL queue is set for oneapi:<n>` | The call did not carry XPU tensors from torch, or torch is too old to report its XPU stream (see Streams). |
| `Tensor stride for dimension 0 is not aligned to N bytes` (or `Tensor data pointer is not aligned ...`) | The rows do not fit the kernel's vector width; this is the `ensure_alignment` contract working. |
| `SYCL error (4): Total number of work-items in a work-group cannot exceed N for this kernel` | The compiled kernel's work-group limit is below the device's `max_work_group_size`, typically because it holds data in registers. Cap the schedule at that limit. |
| `no matching function for call to 'is_pow2'` | `bits.h` takes unsigned types; cast the template parameter. |
| Wrong values only when the item count is one more than a multiple of the work-group size, in a loop `for (i = tid; i < n; i += kConstStride)` | icpx 2026.3 (spir64) miscompiles that loop shape. Stride by `item.get_local_range(0)`, or use the CUDA kernels' fixed-trip loop with an `in_bound` guard, which is not affected. Revisit on a compiler upgrade. |
| Every first call is slow | `spir64` is finalized by the driver on first load; set `SGLANG_JIT_SYCL_TARGETS` to an `intel_gpu_*` target to compile ahead of time. |
