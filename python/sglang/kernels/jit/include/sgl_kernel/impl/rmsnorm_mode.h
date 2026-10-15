/// \file impl/rmsnorm_mode.h
/// \brief The rmsnorm family's modes, shared by the CUDA and the XPU (SYCL) kernels.

#pragma once

namespace sglang {

/// \brief What one rmsnorm launch computes. Python passes the enumerator name as the
///        template argument (`RMSNormMode.cpp_name`), resolved by the `using enum` below.
enum class RMSNormMode {
  PLAIN,            // out = norm(x) * w
  FUSED_ADD,        // res = T(x + res); x = norm(res) * w
  QUANT,            // out8 = fp8(norm(x) * w / scale)
  FUSED_ADD_QUANT,  // both
};
using enum RMSNormMode;

}  // namespace sglang
