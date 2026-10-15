/// \file xpu/utils.hpp
/// \brief SYCL counterpart of `utils.cuh` for Intel XPU JIT kernels.
///
/// Provides:
/// - Element type aliases (`fp16_t`, `bf16_t`, `fp32_t`) bound to SYCL types.
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

#include <cstddef>
#include <cstdint>

namespace sglang {

using fp32_t = float;
using fp16_t = ::sycl::half;
using bf16_t = ::sycl::ext::oneapi::bfloat16;

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
