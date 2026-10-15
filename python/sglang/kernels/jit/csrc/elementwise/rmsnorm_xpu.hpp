// RMSNorm for Intel XPU (SYCL): y = x / sqrt(mean(x^2) + eps) * weight, plain or with
// a fused residual add. One work-group per token row, fp32 accumulation, work-group reduction.
// Ported from deepklox jit_kernel/xe_3/elementwise/rmsnorm.hpp.

#include <sgl_kernel/bits.h>
#include <sgl_kernel/impl/rmsnorm_mode.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/xpu/utils.hpp>

#include <sycl/sycl.hpp>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include <cstdint>

namespace sglang {

struct RMSNormParams {
  const void* input;
  const void* __restrict__ weight;
  void* output;
  void* residual;  // fused residual add only
  int64_t input_stride;
  int64_t output_stride;
  int64_t residual_stride;
  uint32_t num_tokens;
  float eps;
};

/**
 * \brief Work-group RMSNorm: work-group `g` normalizes token row `g`.
 *
 * With `kFusedAdd`, the row first becomes `residual = Float(input + residual)`, written
 * back to `residual`, and that rounded sum is what gets normalized.
 *
 * \tparam kDim               Hidden size
 * \tparam kVecSize           Elements per vectorized load/store
 * \tparam kGroupSize         Work-items per work-group
 * \tparam kFusedAdd          Add `residual` to the input before normalizing
 * \tparam kCastXBeforeOutMul Round x * r to `Float` before the weight multiply (Hugging Face)
 * \tparam Float              Element type: fp32_t | fp16_t | bf16_t
 */
template <int64_t kDim, int kVecSize, uint32_t kGroupSize, bool kFusedAdd, bool kCastXBeforeOutMul, typename Float>
struct RMSNormWorkGroupKernel {
  static_assert(kDim > 0 && kDim % kVecSize == 0, "Hidden size must be a multiple of the vector size");
  static_assert(
      host::is_pow2(static_cast<uint32_t>(kVecSize)) && kVecSize <= 8, "Vector size must be a power of two <= 8");
  static_assert(kGroupSize % device::kSubGroupSize == 0, "Work-group size must be a multiple of the sub-group size");

  RMSNormParams params;

  [[sycl::reqd_sub_group_size(device::kSubGroupSize)]] void operator()(::sycl::nd_item<1> item) const {
    using Vec = device::AlignedVector<Float, kVecSize>;
    constexpr int64_t kNumVecs = kDim / kVecSize;

    const auto& [input, weight_ptr, output, residual, input_stride, output_stride, residual_stride, num_tokens, eps] =
        params;
    const int64_t token = item.get_group(0);
    const int64_t tid = item.get_local_id(0);
    // Equal to kGroupSize, but read at runtime: with the constant stride, icpx 2026.3
    // (spir64, CRI) miscompiles kNumVecs % kGroupSize == 1, e.g. hidden size 1028.
    // Revisit on a compiler upgrade; test_rmsnorm_xpu.py covers 1028.
    const int64_t num_threads = item.get_local_range(0);

    const auto input_vec = reinterpret_cast<const Vec*>(static_cast<const Float*>(input) + token * input_stride);
    const auto weight_vec = static_cast<const Vec*>(weight_ptr);
    const auto output_vec = reinterpret_cast<Vec*>(static_cast<Float*>(output) + token * output_stride);
    Vec* residual_vec = nullptr;
    if constexpr (kFusedAdd) {
      residual_vec = reinterpret_cast<Vec*>(static_cast<Float*>(residual) + token * residual_stride);
    }

    float sum_of_squares = 0.0f;
#pragma unroll
    for (int64_t i = tid; i < kNumVecs; i += num_threads) {
      Vec v = input_vec[i];
      if constexpr (kFusedAdd) {
        const Vec r = residual_vec[i];
#pragma unroll
        for (int j = 0; j < kVecSize; ++j) {
          v[j] = static_cast<Float>(static_cast<float>(v[j]) + static_cast<float>(r[j]));
        }
        residual_vec[i] = v;
      }
#pragma unroll
      for (int j = 0; j < kVecSize; ++j) {
        const float x = static_cast<float>(v[j]);
        sum_of_squares += x * x;
      }
    }

    sum_of_squares = ::sycl::reduce_over_group(item.get_group(), sum_of_squares, ::sycl::plus<float>());
    const float norm_factor = ::sycl::rsqrt(sum_of_squares / static_cast<float>(kDim) + eps);

#pragma unroll
    for (int64_t i = tid; i < kNumVecs; i += num_threads) {
      // Each work-item reads back only the residual sums it wrote itself.
      const Vec v = kFusedAdd ? residual_vec[i] : input_vec[i];
      const Vec w = weight_vec[i];
      Vec out;
#pragma unroll
      for (int j = 0; j < kVecSize; ++j) {
        float normed = static_cast<float>(v[j]) * norm_factor;
        if constexpr (kCastXBeforeOutMul) {
          normed = static_cast<float>(static_cast<Float>(normed));
        }
        out[j] = static_cast<Float>(normed * static_cast<float>(w[j]));
      }
      output_vec[i] = out;
    }
  }
};

/**
 * \brief Validate the tensors and launch `RMSNormWorkGroupKernel`.
 *
 * The XPU counterpart of `RMSNormKernel` in `rmsnorm.cuh`: the same call signature, so
 * one Python dispatch serves both. The vector width and work-group size are chosen in
 * Python (`_jit_rmsnorm_xpu_module`); this only checks they are consistent.
 *
 * \tparam DType              Element type: fp32_t | fp16_t | bf16_t
 * \tparam kDim               Hidden size
 * \tparam kMode              `PLAIN` or `FUSED_ADD`; the fp8 quant modes are CUDA-only
 * \tparam kCastXBeforeOutMul Round x * r to `DType` before the weight multiply (Hugging Face)
 * \tparam kVecSize           Elements per vectorized load/store; rows must be aligned to it
 * \tparam kGroupSize         Work-items per work-group, a multiple of the sub-group size
 */
template <typename DType, int64_t kDim, RMSNormMode kMode, bool kCastXBeforeOutMul, int kVecSize, uint32_t kGroupSize>
struct RMSNormKernel {
  static_assert(kMode == PLAIN || kMode == FUSED_ADD, "fp8 quant modes are CUDA-only");
  static constexpr bool kFusedAdd = kMode == FUSED_ADD;
  using Kernel = RMSNormWorkGroupKernel<kDim, kVecSize, kGroupSize, kFusedAdd, kCastXBeforeOutMul, DType>;

  /**
   * \param input    [num_tokens, kDim], unit stride in the last dimension; with
   *                 `FUSED_ADD` it also receives the output
   * \param weight   [kDim], contiguous
   * \param output   `PLAIN`: [num_tokens, kDim], may alias `input`. `FUSED_ADD`: unused
   * \param residual `FUSED_ADD`: [num_tokens, kDim], receives the sum. `PLAIN`: unused
   * \param scale    Unused; must be empty (the fp8 quant modes are CUDA-only)
   * \param eps      Epsilon added to the mean square
   */
  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::Optional<tvm::ffi::TensorView> output,
      const tvm::ffi::Optional<tvm::ffi::TensorView> residual,
      const tvm::ffi::Optional<tvm::ffi::TensorView> scale,
      const float eps) {
    using namespace host;
    auto N = SymbolicSize{"num_tokens"};
    auto D = SymbolicSize{"hidden_size"};
    auto device = SymbolicDevice{};
    D.set_value(kDim);
    device.set_options<kDLOneAPI>();
    constexpr int64_t kAlignment = kVecSize * sizeof(DType);

    TensorMatcher({N, D})  // input
        .with_strides({-1, 1})
        .with_dtype<DType>()
        .with_device(device)
        .ensure_alignment(kAlignment)
        .verify(input);
    TensorMatcher({D})  // weight
        .with_dtype<DType>()
        .with_device(device)
        .ensure_alignment(kAlignment)
        .verify(weight);
    CHECK_HOST(!scale.has_value()) << "rmsnorm (xpu): `scale` is only used by the fp8 quant modes";

    RMSNormParams params{};
    params.input = input.data_ptr();
    params.weight = weight.data_ptr();
    params.input_stride = input.stride(0);
    params.eps = eps;
    if constexpr (kFusedAdd) {
      CHECK_HOST(residual.has_value()) << "fused modes need `residual`";
      TensorMatcher({N, D})  // residual
          .with_strides({-1, 1})
          .with_dtype<DType>()
          .with_device(device)
          .ensure_alignment(kAlignment)
          .verify(residual.value());
      params.residual = residual.value().data_ptr();
      params.residual_stride = residual.value().stride(0);
      params.output = input.data_ptr();
      params.output_stride = input.stride(0);
    } else {
      CHECK_HOST(output.has_value()) << "rmsnorm needs `output`";
      TensorMatcher({N, D})  // output
          .with_strides({-1, 1})
          .with_dtype<DType>()
          .with_device(device)
          .ensure_alignment(kAlignment)
          .verify(output.value());
      params.output = output.value().data_ptr();
      params.output_stride = output.value().stride(0);
    }

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    if (num_tokens == 0) return;
    params.num_tokens = num_tokens;

    LaunchKernel(num_tokens, kGroupSize, device.unwrap())(Kernel{params});
  }
};

}  // namespace sglang
