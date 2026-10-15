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

The reference implementation is the XPU RMSNorm:

| Piece | File |
|---|---|
| SYCL kernel + launcher | `python/sglang/kernels/jit/csrc/elementwise/rmsnorm_xpu.hpp` |
| SYCL helpers | `python/sglang/kernels/jit/include/sgl_kernel/xpu/utils.hpp` |
| Python dispatch | `python/sglang/kernels/ops/layernorm/norm.py` (`_jit_rmsnorm_xpu_module`, `precompile_rmsnorm`) |
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
- **Not on XPU:** `SGLANG_JIT_LOG_RESOURCE_USAGE` is ignored, and there is no
  PDL (`is_arch_support_pdl()` is False).

## Headers: what compiles under icpx

`.cuh` headers are CUDA/HIP only; do not include them. Under icpx you have:

- **`sgl_kernel/xpu/utils.hpp`**, the SYCL counterpart of `utils.cuh`:
  - `fp32_t` / `fp16_t` / `bf16_t` bound to `float` / `sycl::half` /
    `sycl::ext::oneapi::bfloat16`. `make_cpp_args(torch.float16)` still renders
    `fp16_t`, so the Python side does not change.
  - `device::kSubGroupSize` (16) and `device::AlignedVector<T, N>`.
  - `host::LaunchKernel(num_groups, group_size, device)(functor)`, which submits
    an `nd_range<1>` to the current queue and rethrows SYCL errors as tvm-ffi
    `RuntimeError`s.
  - `host::get_current_sycl_queue(device)` and `SGL_CHECK_SYCL(stmt)` for
    anything `LaunchKernel` does not cover.
- **Host-only headers that also compile under icpx:** `sgl_kernel/utils.h`
  (`CHECK_HOST`, `div_ceil`), `sgl_kernel/bits.h` (`is_pow2`; it takes unsigned
  types, so cast an `int` template parameter), and `sgl_kernel/tensor.h`.
  `tensor.h` knows the SYCL fp16/bf16 types, so `TensorMatcher` works unchanged
  with `.with_device(device)` after `device.set_options<kDLOneAPI>()`.

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
extension, e.g. `csrc/elementwise/<name>_xpu.hpp`. Put everything in
`namespace sglang`. The pattern, from `rmsnorm_xpu.hpp`:

```cpp
#include <sgl_kernel/bits.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/xpu/utils.hpp>

#include <tvm/ffi/container/tensor.h>

#include <sycl/sycl.hpp>

#include <cstdint>

namespace sglang {

// Plain data only: the functor is copied to the device, so no references,
// no TensorView, no host pointers to host objects.
struct MyParams {
  const void* input;
  void* output;
  int64_t input_stride;
  uint32_t num_rows;
};

/// \brief One work-group per row.
/// \tparam kVecSize   Elements per vectorized access, chosen in Python
/// \tparam kGroupSize Work-items per work-group, chosen in Python
template <int64_t kDim, int kVecSize, uint32_t kGroupSize, typename Float>
struct MyKernel {
  MyParams params;

  [[sycl::reqd_sub_group_size(device::kSubGroupSize)]] void operator()(::sycl::nd_item<1> item) const {
    using Vec = device::AlignedVector<Float, kVecSize>;
    const int64_t row = item.get_group(0);
    const int64_t tid = item.get_local_id(0);
    // Stride by the runtime local range, not kGroupSize (see Pitfalls).
    const int64_t num_threads = item.get_local_range(0);
    // ... fp32 accumulation, ::sycl::reduce_over_group(item.get_group(), x, ::sycl::plus<float>()) ...
  }
};

/// \brief Validate the tensors and launch `MyKernel`.
template <int64_t kDim, int kVecSize, uint32_t kGroupSize, typename DType>
struct MyOp {
  static_assert(kGroupSize % device::kSubGroupSize == 0);

  static void run(const tvm::ffi::TensorView input, const tvm::ffi::TensorView output) {
    using namespace host;
    auto N = SymbolicSize{"num_rows"};
    auto D = SymbolicSize{"hidden_size"};
    auto SI = SymbolicSize{"input_stride"};
    auto device = SymbolicDevice{};
    D.set_value(kDim);
    device.set_options<kDLOneAPI>();
    TensorMatcher({N, D})
        .with_strides({SI, 1})
        .with_dtype<DType>()
        .with_device(device)
        .ensure_alignment(kVecSize * sizeof(DType))  // the width the kernel vectorizes at
        .verify(input);
    // ... verify output, early-return on zero rows ...
    const auto params = MyParams{/* ... */};
    LaunchKernel(static_cast<uint32_t>(N.unwrap()), kGroupSize, device.unwrap())(
        MyKernel<kDim, kVecSize, kGroupSize, DType>{params});
  }
};

}  // namespace sglang
```

Rules that differ from CUDA:

- **Every knob is a template parameter set from Python**, including the
  work-group size and vector width. The launcher only `static_assert`s that they
  are consistent (power-of-two vector width, group size a multiple of 16).
- **Accumulate in fp32** and convert with `static_cast<float>` /
  `static_cast<Float>`; `sycl::half` and `bfloat16` convert both ways.
- **Use SYCL group algorithms** (`reduce_over_group`, `group_broadcast`) instead
  of hand-written shared-memory reductions.
- **Give the host launcher the CUDA launcher's name, template parameters and call
  signature**, so the Python side can treat the two modules alike. The XPU
  `RMSNormKernel<DType, kDim, kMode, kCastXBeforeOutMul, kVecSize, kGroupSize>::run`
  takes the same `(input, weight, output?, residual?, scale?, eps)` as the CUDA
  one, with `tvm::ffi::Optional<TensorView>` for the optional tensors, and checks
  per mode which of them must be present.
- **Share mode enums with the CUDA kernel through a host-only header.** A `.cuh`
  cannot be included under icpx, so `RMSNormMode` lives in
  `sgl_kernel/impl/rmsnorm_mode.h`, included by both kernels. Reject the modes XPU
  does not implement with a `static_assert` (rmsnorm: the fp8 quant modes).
- **Inside the device kernel, a mode is a `bool` template parameter**, with the
  arithmetic of the default mode left untouched: `kFusedAdd` adds the residual
  first, and `kCastXBeforeOutMul` rounds `x * r` to the activation dtype before
  the weight multiply, matching the CUDA kernel's flag of the same name.
- **A fused in-place pass may write a buffer and read it back later in the same
  work-item without a barrier**, as long as each work-item reads only what it
  wrote: the fused residual add stores `residual` in the first loop and reloads it
  in the second, with the work-group reduction in between.

## Step 2: Python dispatch in `kernels/ops/`

Keep the public functions as they are, and branch where the CUDA side builds its
module, so every caller (including code that precompiles before graph capture)
gets the XPU module through the same path. Put the dtype and mode checks and the
tuning heuristics in a `@cache_once` XPU factory:

```python
@cache_once
def _jit_my_op_xpu_module(dim: int, dtype: torch.dtype, mode: MyOpMode) -> Module:
    if mode not in _MY_OP_XPU_MODES:
        raise NotImplementedError(f"jit my_op (xpu): {mode.name} is not supported.")
    if dtype not in _MY_OP_XPU_DTYPES:
        raise RuntimeError(f"jit my_op (xpu): unsupported dtype {dtype}.")
    vec_size = ...  # heuristic lives here, not in C++
    group_size = ...
    args = make_cpp_args(dtype, dim, mode.cpp_name, vec_size, group_size)
    return load_jit(
        "my_op_xpu",
        *args,
        sycl_files=["elementwise/my_op_xpu.hpp"],
        sycl_wrappers=[("my_op", f"MyOpKernel<{args}>::run")],
    )
```

In rmsnorm that branch is the top of `precompile_rmsnorm`. CUDA builds a decode and
a prefill schedule there; XPU has one schedule, so the same module serves both
token-count ranges and the `RMSNormDispatch` object is reused unchanged:

```python
if is_xpu_runtime():
    module = _jit_rmsnorm_xpu_module(dim, dtype, mode, cast_x_before_out_mul)
    return RMSNormDispatch(module, module, 0)
```

If the CUDA side has no such factory, branch once at the top of the public
function instead, with one `is_xpu_runtime()` check.

A mode or option the XPU kernel does not implement raises `NotImplementedError` on
XPU, before compiling. Do not silently ignore it: it changes the numbers.

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

- **Its own dtypes and shapes.** Cover each vectorization path, rows not a
  multiple of the group size, and the case where the number of vectors per row is
  one more than a multiple of the group size (e.g. hidden size 1028 with 4-wide
  vectors and 256 work-items).
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
| `no matching function for call to 'is_pow2'` | `bits.h` takes unsigned types; cast the template parameter. |
| Wrong values only when vectors per row is one more than a multiple of the group size | icpx 2026.3 (spir64) miscompiles a work-item loop strided by a compile-time constant. Stride by `item.get_local_range(0)` instead. Revisit on a compiler upgrade. |
| Every first call is slow | `spir64` is finalized by the driver on first load; set `SGLANG_JIT_SYCL_TARGETS` to an `intel_gpu_*` target to compile ahead of time. |
