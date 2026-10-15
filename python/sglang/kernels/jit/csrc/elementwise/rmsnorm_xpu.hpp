// RMSNorm for Intel XPU (SYCL): y = x / sqrt(mean(x^2) + eps) * weight.
// One work-group per token row, fp32 accumulation, work-group reduction.
// Ported from deepklox jit_kernel/xe_3/elementwise/rmsnorm.hpp.

#include <sgl_kernel/bits.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/xpu/utils.hpp>

#include <sycl/sycl.hpp>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

struct RMSNormParams {
  const void* input;
  const void* __restrict__ weight;
  void* output;
  int64_t input_stride;
  int64_t output_stride;
  uint32_t num_tokens;
  float eps;
};

/**
 * \brief Work-group RMSNorm: work-group `g` normalizes token row `g`.
 *
 * \tparam kDim       Hidden size
 * \tparam kVecSize   Elements per vectorized load/store
 * \tparam kGroupSize Work-items per work-group
 * \tparam Float      Element type: fp32_t | fp16_t | bf16_t
 */
template <int64_t kDim, int kVecSize, uint32_t kGroupSize, typename Float>
struct RMSNormWorkGroupKernel {
  RMSNormParams params;

  [[sycl::reqd_sub_group_size(device::kSubGroupSize)]] void operator()(::sycl::nd_item<1> item) const {
    using Vec = device::AlignedVector<Float, kVecSize>;
    constexpr int64_t kNumVecs = kDim / kVecSize;

    const auto& [input, weight_ptr, output, input_stride, output_stride, num_tokens, eps] = params;
    const int64_t token = item.get_group(0);
    const int64_t tid = item.get_local_id(0);
    // Equal to kGroupSize, but read at runtime: with the constant stride, icpx 2026.3
    // (spir64, CRI) miscompiles kNumVecs % kGroupSize == 1, e.g. hidden size 1028.
    // Revisit on a compiler upgrade; test_rmsnorm_xpu.py covers 1028.
    const int64_t num_threads = item.get_local_range(0);

    const auto input_vec = reinterpret_cast<const Vec*>(static_cast<const Float*>(input) + token * input_stride);
    const auto weight_vec = static_cast<const Vec*>(weight_ptr);
    const auto output_vec = reinterpret_cast<Vec*>(static_cast<Float*>(output) + token * output_stride);

    float sum_of_squares = 0.0f;
#pragma unroll
    for (int64_t i = tid; i < kNumVecs; i += num_threads) {
      const Vec v = input_vec[i];
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
      const Vec v = input_vec[i];
      const Vec w = weight_vec[i];
      Vec out;
#pragma unroll
      for (int j = 0; j < kVecSize; ++j) {
        out[j] = static_cast<Float>(static_cast<float>(v[j]) * norm_factor * static_cast<float>(w[j]));
      }
      output_vec[i] = out;
    }
  }
};

/**
 * \brief Validate the tensors and launch `RMSNormWorkGroupKernel`.
 *
 * The vector width and work-group size are chosen in Python
 * (`_jit_rmsnorm_xpu_module`); this only checks they are consistent.
 *
 * \tparam kDim       Hidden size
 * \tparam kVecSize   Elements per vectorized load/store; rows must be aligned to it
 * \tparam kGroupSize Work-items per work-group, a multiple of the sub-group size
 * \tparam DType      Element type: fp32_t | fp16_t | bf16_t
 */
template <int64_t kDim, int kVecSize, uint32_t kGroupSize, typename DType>
struct RMSNormKernel {
  static_assert(kDim > 0 && kDim % kVecSize == 0, "Hidden size must be a multiple of the vector size");
  static_assert(
      host::is_pow2(static_cast<uint32_t>(kVecSize)) && kVecSize <= 8, "Vector size must be a power of two <= 8");
  static_assert(kGroupSize % device::kSubGroupSize == 0, "Work-group size must be a multiple of the sub-group size");
  using Kernel = RMSNormWorkGroupKernel<kDim, kVecSize, kGroupSize, DType>;

  /**
   * \param input  [num_tokens, kDim], unit stride in the last dimension
   * \param weight [kDim], contiguous
   * \param output [num_tokens, kDim], may alias `input`
   * \param eps    Epsilon added to the mean square
   */
  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView output,
      float eps) {
    using namespace host;
    auto N = SymbolicSize{"num_tokens"};
    auto D = SymbolicSize{"hidden_size"};
    auto SI = SymbolicSize{"input_stride"};
    auto SO = SymbolicSize{"output_stride"};
    auto device = SymbolicDevice{};
    D.set_value(kDim);
    device.set_options<kDLOneAPI>();
    constexpr int64_t kAlignment = kVecSize * sizeof(DType);

    TensorMatcher({N, D})  // input
        .with_strides({SI, 1})
        .with_dtype<DType>()
        .with_device(device)
        .ensure_alignment(kAlignment)
        .verify(input);
    TensorMatcher({D})  // weight
        .with_dtype<DType>()
        .with_device(device)
        .ensure_alignment(kAlignment)
        .verify(weight);
    TensorMatcher({N, D})  // output
        .with_strides({SO, 1})
        .with_dtype<DType>()
        .with_device(device)
        .ensure_alignment(kAlignment)
        .verify(output);

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    if (num_tokens == 0) return;
    const auto params = RMSNormParams{
        .input = input.data_ptr(),
        .weight = weight.data_ptr(),
        .output = output.data_ptr(),
        .input_stride = SI.unwrap(),
        .output_stride = SO.unwrap(),
        .num_tokens = num_tokens,
        .eps = eps,
    };

    LaunchKernel(num_tokens, kGroupSize, device.unwrap())(Kernel{params});
  }
};

}  // namespace sglang
