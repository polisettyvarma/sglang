# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the SGLang project

"""Compressed-tensors MXFP8 (OCP microscaling) W8A8 scheme for Intel XPU / CRI.

Model format (compressed-tensors ``mxfp8-quantized``):
  - weights:            float8_e4m3fn, per-32-K-block scale, static, symmetric.
  - input_activations:  float8_e4m3fn, per-32-K-block scale, dynamic, symmetric.
  - scale storage:      UE8M0 (uint8, exponent + 127), group_size = 32.

Weights are loaded already quantized, so this scheme never quantizes them -- the
only post-load fixup is reinterpreting the UE8M0 scale bytes as
``torch.float8_e8m0fnu``. Activations are quantized per forward with the fused
``sgl_per_token_group_quant_8bit_v2`` kernel (rather than a custom SYCL kernel),
and the matmul runs through ``torch._scaled_mm``, which dispatches to oneDNN's
native MXFP8 (BlockWise1x32) GEMM on CRI (Xe35).

Why ``torch._scaled_mm`` and not ``torch.nn.functional.scaled_mm``: the v2 op
(``_scaled_mm_v2``) shares a single structured meta function across backends,
and that meta calls ``at::native::scaled::validate_scaled_mm_v2_inputs`` from
libtorch_cpu. For BlockWise1x32 that validator enforces the *NVIDIA* padded +
swizzled scale layout (``round_up(M,128) x round_up(K/32,4)`` elements plus
``SWIZZLE_32_4_4``) on every non-ROCm device, while the XPU implementation
requires unpadded ``[M, K/32]`` scales and explicitly rejects any swizzle
("XPU does not support swizzle yet."). Those two requirements are mutually
exclusive, so no argument combination reaches the XPU MXFP8 kernel through v2.
The v1 op has no such shared meta: ``_scaled_mm_out_xpu`` runs its own
``get_joint_scaling`` probe, which accepts BlockWise1x32 with unpadded e8m0
scales. Revisit if the upstream validator becomes device-aware.
"""

from typing import Callable, Optional

import torch
from compressed_tensors.quantization import QuantizationArgs

from sglang.srt.layers.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsLinearScheme,
)

__all__ = ["CompressedTensorsW8A8MXFp8"]

# OCP-MX block size: one scale per 32 contiguous K elements.
MXFP8_GROUP_SIZE = 32


class CompressedTensorsW8A8MXFp8(CompressedTensorsLinearScheme):
    """MXFP8 W8A8 dense linear via torch._scaled_mm (oneDNN MXFP8 GEMM on XPU)."""

    def __init__(self, weight_quant: QuantizationArgs, input_quant: QuantizationArgs):
        self.group_size = weight_quant.group_size or MXFP8_GROUP_SIZE
        assert (
            self.group_size == MXFP8_GROUP_SIZE
        ), f"MXFP8 requires group_size=={MXFP8_GROUP_SIZE}, got {self.group_size}"
        # apply_weights always quantizes activations per forward, so the recipe's
        # dynamic-activation assumption has to hold (routing only checks that the
        # *weights* are static). No input_scale parameter is created or consumed.
        assert (
            input_quant is None or input_quant.dynamic
        ), "MXFP8 W8A8 requires dynamically quantized activations"

    @classmethod
    def get_min_capability(cls) -> int:
        # Intel CRI (Xe35) provides native MXFP8 via oneDNN. Capability gating is
        # handled by the platform check in compressed_tensors.py; return 0 so the
        # generic (CUDA SM) gate does not exclude XPU.
        return 0

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        del input_size, output_size  # unused; shapes come from *_per_partition
        output_size_per_partition = sum(output_partition_sizes)
        # Consumed by apply_weights as the scaled_mm output dtype.
        layer.orig_dtype = params_dtype

        assert input_size_per_partition % self.group_size == 0, (
            f"K={input_size_per_partition} must be a multiple of "
            f"group_size={self.group_size} for MXFP8"
        )
        scale_k = input_size_per_partition // self.group_size

        # WEIGHT: [N, K] float8_e4m3fn.
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE: [N, K/32] UE8M0 stored as uint8, per-group along K.
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                scale_k,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Weights arrive already MXFP8-quantized, so there is nothing to requantize
        # here. The only fixup needed is reinterpreting the loaded uint8 UE8M0 bytes
        # as the native float8_e8m0fnu dtype the scaled_mm recipe consumes.
        #
        # The weight itself stays [N, K] and is transposed at apply time, which keeps
        # the loaded scales aligned to the N (output) rows.
        weight_scale_e8m0 = layer.weight_scale.data.contiguous().view(
            torch.float8_e8m0fnu
        )
        layer.weight_scale = torch.nn.Parameter(weight_scale_e8m0, requires_grad=False)

    # dtypes the fused kernel is instantiated for. We call it with enable_v2=True,
    # and v2 dispatches over BFloat16/Half only (SYCL_DISPATCH_ONLY_FLOATING16_TYPES
    # in per_token_group_quant_8bit_v2.cpp), so fp32 input raises
    #   'sgl_per_token_group_quant_8bit_v2' not implemented for 'Float'
    # (the v1 kernel does have an fp32 instantiation, but v2 is the faster path).
    _FUSED_QUANT_DTYPES = (torch.bfloat16, torch.float16)

    def _quantize_activation(self, x_2d, M, K, scale_k):
        """MXFP8-quantize a [M, K] activation to (E4M3 elements, UE8M0 block-32
        scales viewed as float8_e8m0fnu).

        Uses the fused ``sgl_per_token_group_quant_8bit`` kernel, casting the input
        to a dtype that kernel supports if needed. Falls back to a pure-torch
        equivalent only when sgl_kernel is not installed.
        """
        try:
            from sgl_kernel import sgl_per_token_group_quant_8bit
        except ImportError:
            return self._quantize_activation_torch(x_2d, M, K, scale_k)

        # Feed the kernel a dtype it is actually instantiated for. Casting to
        # bfloat16 keeps the fused path (and its coalesced stores) instead of
        # dropping to the pure-torch quantizer; the extra rounding is immaterial
        # here because the destination E4M3 carries only 3 mantissa bits and the
        # UE8M0 scale carries none, both far coarser than bf16's 8.
        if x_2d.dtype not in self._FUSED_QUANT_DTYPES:
            x_2d = x_2d.to(torch.bfloat16)

        fp8_info = torch.finfo(torch.float8_e4m3fn)
        xq = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=x_2d.device)

        # A contiguous (row-major) [M, K/32] UE8M0 buffer is exactly the ``scale_a``
        # layout that _scaled_mm's is_blockwise_1x32_scaling probe expects, and the
        # SYCL kernel writes that layout directly: its row-major branch stores one
        # UE8M0 byte at the group index when instantiated with scale_packed_t=uint8_t
        # (see sgl-kernel-xpu/src/sycl/per_token_group_quant_8bit_v2.cpp -- the
        # launcher selects that instantiation for a row-major output_s with
        # scale_ue8m0=True). No pack/transpose/unpack round-trip is needed.
        x_scale_u8 = torch.empty(M, scale_k, dtype=torch.uint8, device=x_2d.device)

        # Guard the kernel's dtype contract at the call site: the cast above should
        # have already guaranteed this, so a failure here means the dispatch table
        # and _FUSED_QUANT_DTYPES have drifted apart. Assert rather than let the
        # kernel raise "not implemented for '<dtype>'" from inside SYCL.
        assert x_2d.dtype in self._FUSED_QUANT_DTYPES, (
            f"sgl_per_token_group_quant_8bit requires input dtype in "
            f"{self._FUSED_QUANT_DTYPES}, got {x_2d.dtype}"
        )
        sgl_per_token_group_quant_8bit(
            input=x_2d,
            output_q=xq,
            output_s=x_scale_u8.view(torch.float8_e8m0fnu),
            group_size=self.group_size,
            eps=1e-10,
            fp8_min=fp8_info.min,
            fp8_max=fp8_info.max,
            scale_ue8m0=True,
            enable_v2=True,
        )
        return xq, x_scale_u8.view(torch.float8_e8m0fnu)

    def _quantize_activation_torch(self, x_2d, M, K, scale_k):
        """Pure-torch MXFP8 activation quant, matching the SYCL kernel exactly:
        ``exp = ceil(log2(amax / E4M3_MAX))``, scale = ``2**exp``, stored as
        UE8M0 (``exp + 127``)."""
        fp8_info = torch.finfo(torch.float8_e4m3fn)
        xf = x_2d.to(torch.float32).view(M, scale_k, self.group_size)
        amax = xf.abs().amax(dim=-1).clamp_min(1e-10)
        exp = torch.ceil(torch.log2(amax / fp8_info.max)).clamp(-127, 127)
        scale = torch.exp2(exp)
        xq = (
            torch.clamp(xf / scale.unsqueeze(-1), fp8_info.min, fp8_info.max)
            .to(torch.float8_e4m3fn)
            .view(M, K)
        )
        x_scale = (exp.to(torch.int32) + 127).to(torch.uint8).view(torch.float8_e8m0fnu)
        return xq, x_scale

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
        M, K = x_2d.shape
        scale_k = K // self.group_size

        # Dynamically quantize activations to MXFP8 (E4M3 + UE8M0 block-32 scales).
        # Prefer the fused sgl_kernel op; fall back to a pure-torch quantizer when
        # sgl_kernel is unavailable, so the path works on any XPU torch build.
        xq, x_scale = self._quantize_activation(x_2d, M, K, scale_k)

        # B for scaled_mm must be [K, N]; weight is [N, K]. Transpose (view) both
        # the weight and its per-group scale so column j of B has scale row j.
        w = layer.weight  # [N, K]
        w_scale = layer.weight_scale  # [N, K/32] e8m0

        # v1 ``torch._scaled_mm``: BlockWise1x32 is selected by ``get_joint_scaling``
        # in _scaled_mm_out_xpu, which probes is_blockwise_1x32_scaling(a, scale_a)
        # and is_blockwise_1x32_scaling(b.t(), scale_b.t()). So scale_a is the
        # unpadded [M, K/32] and scale_b is [K/32, N] (i.e. w_scale.t(), whose .t()
        # is the [N, K/32] the probe wants). See the module docstring for why the
        # v2 F.scaled_mm entry point cannot be used on XPU.
        out = torch._scaled_mm(
            xq,  # [M, K] e4m3
            w.t(),  # [K, N] e4m3
            x_scale,  # [M, K/32] e8m0
            w_scale.t(),  # [K/32, N] e8m0
            bias=bias,
            out_dtype=layer.orig_dtype,
        )
        return out.reshape(*orig_shape[:-1], out.shape[-1])
