// The rmsnorm family for Intel XPU (SYCL), ported from `rmsnorm.cuh`: the same tile, schedule
// parameters, launcher signature and order of floating-point operations. The fp8 e4m3 output
// rounds in software (`device::cast_fp8_e4m3`).

#include <sgl_kernel/bits.h>
#include <sgl_kernel/impl/rmsnorm_mode.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/xpu/utils.hpp>

#include <sycl/ext/oneapi/group_local_memory.hpp>
#include <sycl/sycl.hpp>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace sglang {

// One row per `kThreadsPerRow` work-items (whole sub-groups, or a power-of-two slice of one
// sub-group for short rows), `kRowsPerCTA` rows per work-group, `kVecBytes` per access. The
// tile may over-cover the row; `in_bound` idles the surplus vectors.
template <typename T_, int64_t kDim_, uint32_t kVecBytes_, uint32_t kThreadsPerRow_, uint32_t kRowsPerCTA_>
struct RMSNormTile {
  using T = T_;
  static constexpr int64_t kDim = kDim_;
  static constexpr uint32_t kVecBytes = kVecBytes_;
  static_assert(host::is_pow2(kVecBytes) && kVecBytes >= sizeof(T) && kVecBytes <= 32);
  static constexpr uint32_t kVecElems = kVecBytes / sizeof(T);
  static_assert(kDim % kVecElems == 0, "hidden size must be a whole number of vectors");
  static constexpr uint32_t kNumVecs = kDim / kVecElems;

  static constexpr uint32_t kThreadsPerRow = kThreadsPerRow_;
  static_assert(
      kThreadsPerRow % device::kSubGroupSize == 0 ||
          (device::kSubGroupSize % kThreadsPerRow == 0 && host::is_pow2(kThreadsPerRow)),
      "a row takes whole sub-groups or a power-of-two slice of one sub-group");
  static constexpr uint32_t kSubGroupsPerRow = host::div_ceil(kThreadsPerRow, device::kSubGroupSize);
  static constexpr uint32_t kRowsPerCTA = kRowsPerCTA_;
  static constexpr uint32_t kNumThreads = kThreadsPerRow * kRowsPerCTA;
  static_assert(kNumThreads % device::kSubGroupSize == 0 && kNumThreads <= 1024, "whole sub-groups per work-group");

  static constexpr uint32_t kVecsPerThread = host::div_ceil(kNumVecs, kThreadsPerRow);
  static constexpr bool kAligned = (kVecsPerThread * kThreadsPerRow == kNumVecs);
  // CUDA's 16 vectors of 16 B, as bytes: rows of narrower vectors may take more of them.
  static_assert(kVecsPerThread * kVecBytes <= 256, "too many bytes per thread; use more threads per row");

  static constexpr bool in_bound(uint32_t i, uint32_t tid) {
    if constexpr (kAligned) return true;
    // only the last step can be partial
    if constexpr ((kVecsPerThread - 1) * kThreadsPerRow <= kNumVecs) {
      if (i + 1 < kVecsPerThread) return true;
    }
    return i * kThreadsPerRow + tid < kNumVecs;
  }
};

struct RMSNormParams {
  const void* __restrict__ input;
  const void* __restrict__ weight;
  void* __restrict__ output;        // PLAIN / QUANT only; the fused modes rewrite `input`
  void* __restrict__ residual;      // fused modes: residual in, x + residual out
  const float* __restrict__ scale;  // quant modes: per-tensor fp8 scale, device scalar
  int64_t input_stride;
  int64_t output_stride;
  int64_t residual_stride;
  uint32_t num_tokens;
  float eps;
};

// One row tile per work-group. The row stays in registers between its two uses, and rows
// spanning several sub-groups reduce through local memory with one barrier. Rows past
// `num_tokens` (the last packed work-group's surplus) load zeros, store nothing and still join
// the barrier.
template <typename Tile, RMSNormMode kMode, bool kCastXBeforeOutMul>
struct RMSNormDeviceKernel {
  RMSNormParams params;

  [[sycl::reqd_sub_group_size(device::kSubGroupSize)]] void operator()(::sycl::nd_item<1> item) const {
    using T = typename Tile::T;
    using vec_t = device::AlignedVector<T, Tile::kVecElems>;
    constexpr uint32_t kVecElems = Tile::kVecElems;
    constexpr uint32_t kVecsPerThread = Tile::kVecsPerThread;
    constexpr uint32_t kThreadsPerRow = Tile::kThreadsPerRow;
    constexpr uint32_t kSubGroupsPerRow = Tile::kSubGroupsPerRow;
    constexpr bool kFusedAdd = (kMode == RMSNormMode::FUSED_ADD || kMode == RMSNormMode::FUSED_ADD_QUANT);
    constexpr bool kQuant = (kMode == RMSNormMode::QUANT || kMode == RMSNormMode::FUSED_ADD_QUANT);
    constexpr float kInvDim = 1.0f / static_cast<float>(Tile::kDim);

    const auto local_id = static_cast<uint32_t>(item.get_local_id(0));
    const auto tid = local_id % kThreadsPerRow;
    const auto row_in_cta = local_id / kThreadsPerRow;
    const auto sub_group_in_row = tid / device::kSubGroupSize;
    const auto lane_id = tid % device::kSubGroupSize;
    const auto row = static_cast<uint32_t>(item.get_group(0)) * Tile::kRowsPerCTA + row_in_cta;
    // with one row per work-group the grid is exactly num_tokens; only packed groups have surplus rows
    const bool valid = Tile::kRowsPerCTA == 1 || row < params.num_tokens;

    const auto zero = [] {
      vec_t v;
#pragma unroll
      for (uint32_t j = 0; j < kVecElems; ++j) {
        v[j] = static_cast<T>(0.0f);
      }
      return v;
    };

    const auto weight_ptr = static_cast<const vec_t*>(params.weight);
    vec_t weight[kVecsPerThread];
#pragma unroll
    for (uint32_t i = 0; i < kVecsPerThread; ++i) {
      weight[i] = Tile::in_bound(i, tid) ? weight_ptr[i * kThreadsPerRow + tid] : zero();
    }
    [[maybe_unused]] float scale_inv = 1.0f;
    if constexpr (kQuant) {
      scale_inv = 1.0f / params.scale[0];
    }

    const auto input_ptr =
        reinterpret_cast<const vec_t*>(static_cast<const T*>(params.input) + row * params.input_stride);
    vec_t x[kVecsPerThread];
#pragma unroll
    for (uint32_t i = 0; i < kVecsPerThread; ++i) {
      x[i] = (valid && Tile::in_bound(i, tid)) ? input_ptr[i * kThreadsPerRow + tid] : zero();
    }

    if constexpr (kFusedAdd) {
      const auto residual_ptr =
          reinterpret_cast<vec_t*>(static_cast<T*>(params.residual) + row * params.residual_stride);
      vec_t res[kVecsPerThread];
#pragma unroll
      for (uint32_t i = 0; i < kVecsPerThread; ++i) {
        res[i] = (valid && Tile::in_bound(i, tid)) ? residual_ptr[i * kThreadsPerRow + tid] : zero();
      }
      // the exact sum is rounded once to T, and the rounded value is what gets normalized
#pragma unroll
      for (uint32_t i = 0; i < kVecsPerThread; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < kVecElems; ++j) {
          x[i][j] = static_cast<T>(static_cast<float>(x[i][j]) + static_cast<float>(res[i][j]));
        }
        if (valid && Tile::in_bound(i, tid)) {
          residual_ptr[i * kThreadsPerRow + tid] = x[i];
        }
      }
    }

    // two accumulation chains, over the even and the odd elements
    float acc0 = 0.0f, acc1 = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < kVecsPerThread; ++i) {
#pragma unroll
      for (uint32_t j = 0; j < kVecElems; ++j) {
        const float v = static_cast<float>(x[i][j]);
        if (j % 2 == 0) {
          acc0 = ::sycl::fma(v, v, acc0);
        } else {
          acc1 = ::sycl::fma(v, v, acc1);
        }
      }
    }
    // rows shorter than a sub-group reduce over their own lane slice
    constexpr uint32_t kReduceLanes = std::min(kThreadsPerRow, device::kSubGroupSize);
    float sum = device::sub_group::reduce_sum<kReduceLanes>(item.get_sub_group(), acc0 + acc1);
    if constexpr (kSubGroupsPerRow > 1) {
      auto& s_partial =
          *::sycl::ext::oneapi::group_local_memory_for_overwrite<float[Tile::kRowsPerCTA][kSubGroupsPerRow]>(
              item.get_group());
      if (lane_id == 0) s_partial[row_in_cta][sub_group_in_row] = sum;
      ::sycl::group_barrier(item.get_group());
      sum = 0.0f;
#pragma unroll
      for (uint32_t w = 0; w < kSubGroupsPerRow; ++w) {
        sum += s_partial[row_in_cta][w];
      }
    }
    const float r = ::sycl::rsqrt(sum * kInvDim + params.eps);

    // `factor` multiplies the normalized product: r for T output, r / scale for fp8. The HF form
    // must round x * r to T before the weight multiply.
    const float factor = kQuant ? r * scale_inv : r;
    [[maybe_unused]] const float hf_post = kQuant ? scale_inv : 1.0f;
    const auto scaled = [&](uint32_t i, uint32_t j) -> float {
      const float v = static_cast<float>(x[i][j]);
      const float w = static_cast<float>(weight[i][j]);
      if constexpr (kCastXBeforeOutMul) {
        return static_cast<float>(static_cast<T>(v * r)) * w * hf_post;
      } else {
        return v * w * factor;
      }
    };

    if constexpr (kQuant) {
      using out_vec_t = device::AlignedVector<fp8_e4m3_t, kVecElems>;
      const auto output_ptr =
          reinterpret_cast<out_vec_t*>(static_cast<fp8_e4m3_t*>(params.output) + row * params.output_stride);
#pragma unroll
      for (uint32_t i = 0; i < kVecsPerThread; ++i) {
        out_vec_t out;
#pragma unroll
        for (uint32_t j = 0; j < kVecElems; ++j) {
          out[j] = device::cast_fp8_e4m3(scaled(i, j));
        }
        if (valid && Tile::in_bound(i, tid)) {
          output_ptr[i * kThreadsPerRow + tid] = out;
        }
      }
    } else {
      const auto out_base = kFusedAdd ? const_cast<void*>(params.input) : params.output;
      const auto out_stride = kFusedAdd ? params.input_stride : params.output_stride;
      const auto output_ptr = reinterpret_cast<vec_t*>(static_cast<T*>(out_base) + row * out_stride);
#pragma unroll
      for (uint32_t i = 0; i < kVecsPerThread; ++i) {
        vec_t out;
#pragma unroll
        for (uint32_t j = 0; j < kVecElems; ++j) {
          out[j] = static_cast<T>(scaled(i, j));
        }
        if (valid && Tile::in_bound(i, tid)) {
          output_ptr[i * kThreadsPerRow + tid] = out;
        }
      }
    }
  }
};

// `kVecBytes`, `kThreadsPerRow` and `kRowsPerCTA` are the schedule chosen in Python
// (`sglang.kernels.ops.layernorm.norm`), one JIT module per schedule. The fused modes rewrite
// `input` in place and ignore `output`; `scale` is the fp32 device scalar of the quant modes.
// The same template parameters and call signature as `RMSNormKernel` in `rmsnorm.cuh`.
template <
    typename T,
    int64_t kDim,
    RMSNormMode kMode,
    bool kCastXBeforeOutMul,
    bool kUsePDL,
    uint32_t kVecBytes,
    uint32_t kThreadsPerRow,
    uint32_t kRowsPerCTA>
struct RMSNormKernel {
  static constexpr bool kFusedAdd = (kMode == RMSNormMode::FUSED_ADD || kMode == RMSNormMode::FUSED_ADD_QUANT);
  static constexpr bool kQuant = (kMode == RMSNormMode::QUANT || kMode == RMSNormMode::FUSED_ADD_QUANT);
  static_assert(!kUsePDL, "XPU has no programmatic dependent launch");
  static_assert(
      std::is_same_v<T, fp32_t> || std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>,
      "rmsnorm (xpu): unsupported dtype; fp32 / fp16 / bf16 only");
  using Tile = RMSNormTile<T, kDim, kVecBytes, kThreadsPerRow, kRowsPerCTA>;
  using Kernel = RMSNormDeviceKernel<Tile, kMode, kCastXBeforeOutMul>;

  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::Optional<tvm::ffi::TensorView> output,
      const tvm::ffi::Optional<tvm::ffi::TensorView> residual,
      const tvm::ffi::Optional<tvm::ffi::TensorView> scale,
      const float eps) {
    using namespace host;
    constexpr int64_t D = kDim;
    auto N = SymbolicSize{"num_tokens"};
    auto device_sym = SymbolicDevice{};
    device_sym.set_options<kDLOneAPI>();

    // the vector accesses need the base pointers and the row strides aligned
    TensorMatcher({N, D})  // input
        .with_strides({-1, 1})
        .with_dtype<T>()
        .with_device(device_sym)
        .ensure_alignment(kVecBytes)
        .verify(input);
    TensorMatcher({D})  // weight
        .with_dtype<T>()
        .with_device(device_sym)
        .ensure_alignment(kVecBytes)
        .verify(weight);

    RMSNormParams params{};
    params.input = input.data_ptr();
    params.weight = weight.data_ptr();
    params.input_stride = input.stride(0);
    params.eps = eps;
    if constexpr (kQuant) {
      CHECK_HOST(output.has_value() && scale.has_value()) << "quant modes need `output` and `scale`";
      TensorMatcher({N, D})  // fp8 output
          .with_strides({-1, 1})
          .with_dtype<fp8_e4m3_t>()
          .with_device(device_sym)
          .ensure_alignment(Tile::kVecElems * sizeof(fp8_e4m3_t))
          .verify(output.value());
      // Static FP8 loaders produce the per-tensor scale as either [1] or 0-D.
      const auto& scale_value = scale.value();
      CHECK_HOST(scale_value.numel() == 1 && is_type<fp32_t>(scale_value.dtype()))
          << "`scale` must hold a single fp32 element";
      CHECK_HOST(
          scale_value.device().device_type == input.device().device_type &&
          scale_value.device().device_id == input.device().device_id)
          << "`scale` must be on the same device as `input`";
      params.output = output.value().data_ptr();
      params.output_stride = output.value().stride(0);
      params.scale = static_cast<const float*>(scale_value.data_ptr());
    } else if constexpr (!kFusedAdd) {
      CHECK_HOST(output.has_value()) << "rmsnorm needs `output`";
      TensorMatcher({N, D})  // output
          .with_strides({-1, 1})
          .with_dtype<T>()
          .with_device(device_sym)
          .ensure_alignment(kVecBytes)
          .verify(output.value());
      params.output = output.value().data_ptr();
      params.output_stride = output.value().stride(0);
    }
    if constexpr (kFusedAdd) {
      CHECK_HOST(residual.has_value()) << "fused modes need `residual`";
      TensorMatcher({N, D})  // residual
          .with_strides({-1, 1})
          .with_dtype<T>()
          .with_device(device_sym)
          .ensure_alignment(kVecBytes)
          .verify(residual.value());
      params.residual = residual.value().data_ptr();
      params.residual_stride = residual.value().stride(0);
    }

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    if (num_tokens == 0) return;
    params.num_tokens = num_tokens;
    const auto num_blocks = div_ceil(num_tokens, kRowsPerCTA);
    LaunchKernel(num_blocks, Tile::kNumThreads, device_sym.unwrap())(Kernel{params});
  }
};

}  // namespace sglang
