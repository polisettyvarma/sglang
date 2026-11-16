/// \file xpu/utils.hpp
/// \brief SYCL counterpart of `utils.cuh` for Intel XPU JIT kernels.
///
/// Provides:
/// - Element type aliases (`fp16_t`, `bf16_t`, `fp32_t`) bound to SYCL types, and
///   `fp8_e4m3_t` storage with a float conversion (`device::cast_fp8_e4m3`).
/// - `device::AlignedVector` for vectorized loads/stores.
/// - `host::LaunchKernel`, which submits an `nd_range` kernel to the current
///   SYCL queue of the tensor's device (the current `torch.xpu` stream).
///
/// Requires a SYCL 2020 compiler in C++20 mode (`icpx -fsycl -std=c++20`).

#pragma once

#include <sgl_kernel/utils.h>

#include <dlpack/dlpack.h>
#include <sycl/sycl.hpp>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <bit>
#include <cstddef>
#include <cstdint>

namespace sglang {

using fp32_t = float;
using fp16_t = ::sycl::half;
using bf16_t = ::sycl::ext::oneapi::bfloat16;

/// \brief One OCP fp8 e4m3fn value (`torch.float8_e4m3fn`): storage only, built with
///        `device::cast_fp8_e4m3`. A struct rather than `uint8_t`, so tensor checks see an
///        fp8 dtype, not an integer one.
struct fp8_e4m3_t {
  uint8_t bits;
};

namespace device {

/// \brief Sub-group size the XPU JIT kernels are compiled for (Xe2 / Xe3).
inline constexpr uint32_t kSubGroupSize = 16;

/**
 * \brief `N` elements of `T`, aligned so a whole-vector load/store lowers to
 *        one vector memory operation.
 * \tparam T Element type
 * \tparam N Number of elements, a power of two
 */
template <typename T, int N>
struct alignas(N * sizeof(T)) AlignedVector {
  static_assert(N > 0 && (N & (N - 1)) == 0, "AlignedVector size must be a power of two");
  T data[N];

  T& operator[](int i) {
    return data[i];
  }
  const T& operator[](int i) const {
    return data[i];
  }
};

/**
 * \brief Round `value` to fp8 e4m3fn: nearest-even, saturating to +-448, NaN kept as NaN.
 *
 * Bit-identical to torch's `value.clamp(-448, 448).to(torch.float8_e4m3fn)`. Done in
 * software so it needs no fp8 device aspect.
 */
inline fp8_e4m3_t cast_fp8_e4m3(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  const auto sign = static_cast<uint8_t>((bits >> 24) & 0x80u);
  const uint32_t magnitude = bits & 0x7FFFFFFFu;
  if (magnitude > 0x7F800000u) return {static_cast<uint8_t>(sign | 0x7Fu)};   // NaN
  if (magnitude >= 0x43E00000u) return {static_cast<uint8_t>(sign | 0x7Eu)};  // >= 448: max finite
  if (magnitude < 0x3C800000u) {
    // Below 2^-6, the smallest normal: count units of 2^-9 (exact power-of-two scaling) and
    // round to nearest even. A result of 8 is 0x08, the smallest normal, so it carries over.
    const float units = ::sycl::rint(std::bit_cast<float>(magnitude) * 512.0f);
    return {static_cast<uint8_t>(sign | static_cast<uint32_t>(units))};
  }
  // Normal: rebias the exponent (127 -> 7) and round the mantissa from 23 to 3 bits.
  const uint32_t exponent = (magnitude >> 23) - 120u;
  const uint32_t mantissa = magnitude & 0x7FFFFFu;
  uint32_t code = (exponent << 3) | (mantissa >> 20);
  const uint32_t rest = mantissa & 0xFFFFFu;
  if (rest > 0x80000u || (rest == 0x80000u && (code & 1u))) ++code;  // a carry bumps the exponent
  return {static_cast<uint8_t>(sign | (code < 0x7Eu ? code : 0x7Eu))};
}

namespace sub_group {

/**
 * \brief Sum `value` over each group of `kLanes` consecutive lanes of the sub-group, by an
 *        xor butterfly from `kLanes / 2` down to 1, the order of `warp::reduce_sum` in
 *        `warp.cuh`. Every lane of a group ends with the group's sum.
 * \tparam kLanes Lanes per group: a power of two, at most `kSubGroupSize`
 */
template <uint32_t kLanes>
inline float reduce_sum(::sycl::sub_group sg, float value) {
  static_assert(kLanes > 0 && (kLanes & (kLanes - 1)) == 0 && kLanes <= kSubGroupSize);
#pragma unroll
  for (uint32_t mask = kLanes / 2; mask >= 1; mask >>= 1) {
    value += ::sycl::permute_group_by_xor(sg, value, mask);
  }
  return value;
}

}  // namespace sub_group

}  // namespace device

namespace host {

/// \brief Rethrow a `sycl::exception` raised by `STMT` as a tvm-ffi `RuntimeError`.
#define SGL_CHECK_SYCL(STMT)                                                         \
  do {                                                                               \
    try {                                                                            \
      STMT;                                                                          \
    } catch (const ::sycl::exception& sgl_sycl_error) {                              \
      TVM_FFI_THROW(RuntimeError) << "SYCL error (" << sgl_sycl_error.code().value() \
                                  << "): " << sgl_sycl_error.what();                 \
    }                                                                                \
  } while (0)

/**
 * \brief The SYCL queue set as the current tvm-ffi stream of a `kDLOneAPI` device.
 *
 * tvm-ffi defines the stream handle of `kDLOneAPI` as a non-owning `sycl::queue*`,
 * valid for the duration of the call (see `TVMFFIStreamHandle` in `c_env_api.h`).
 */
inline ::sycl::queue& get_current_sycl_queue(DLDevice device) {
  CHECK_HOST(device.device_type == kDLOneAPI)
      << "Expected a kDLOneAPI device, got device type " << static_cast<int>(device.device_type);
  void* stream = TVMFFIEnvGetStream(device.device_type, device.device_id);
  CHECK_HOST(stream != nullptr) << "No SYCL queue is set for oneapi:" << device.device_id;
  return *static_cast<::sycl::queue*>(stream);
}

/**
 * \brief Submit an `nd_range<1>` kernel to the current SYCL queue of a device.
 *
 * Usage: `LaunchKernel(num_groups, group_size, device)(kernel_functor);`.
 * SYCL errors raised at submission are rethrown as tvm-ffi `RuntimeError`s.
 */
class LaunchKernel {
 public:
  /**
   * \param num_groups Number of work-groups
   * \param group_size Work-items per work-group
   * \param device     A `kDLOneAPI` device; its current FFI stream is the queue
   */
  LaunchKernel(uint32_t num_groups, uint32_t group_size, DLDevice device)
      : m_queue(get_current_sycl_queue(device)),
        m_range(::sycl::range<1>(static_cast<size_t>(num_groups) * group_size), ::sycl::range<1>(group_size)) {}

  /// \brief Launch `kernel`, a device-copyable functor taking `sycl::nd_item<1>`.
  template <typename Kernel>
  void operator()(const Kernel& kernel) const {
    SGL_CHECK_SYCL(m_queue.parallel_for(m_range, kernel));
  }

 private:
  ::sycl::queue& m_queue;
  ::sycl::nd_range<1> m_range;
};

}  // namespace host

}  // namespace sglang
