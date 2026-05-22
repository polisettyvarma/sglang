from __future__ import annotations

import functools
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import torch
import triton
import triton.language as tl

from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
from sglang.srt.layers.moe.utils import get_moe_padding_size
from sglang.srt.layers.quantization.fp8_kernel import (
    per_token_group_quant_fp8,
    scaled_fp8_quant,
    sglang_per_token_group_quant_fp8,
)
from sglang.srt.layers.quantization.int8_kernel import (
    per_token_group_quant_int8,
    per_token_quant_int8,
    sglang_per_token_group_quant_int8,
)
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_cuda,
    is_hip,
    is_sm90_supported,
)

try:
    from triton.tools.tensor_descriptor import TensorDescriptor

    _support_tensor_descriptor = True
except:
    _support_tensor_descriptor = False

_is_hip = is_hip()
_is_cuda = is_cuda()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _is_cuda:
    pass
elif _is_cpu and _is_cpu_amx_available:
    pass
elif _is_hip:
    pass

padding_size = get_moe_padding_size(_use_aiter)


def support_tensor_descriptor():
    return _support_tensor_descriptor


# swap_ab benefits SM90 GPUs (H20, H100, H200, etc.) for certain block shapes.
@functools.lru_cache(maxsize=8)
def should_enable_swap_ab(
    BLOCK_SIZE_M: int,
    BLOCK_SIZE_N: int,
) -> bool:
    if not _is_cuda or is_batch_invariant_mode_enabled():
        return False

    return is_sm90_supported() and BLOCK_SIZE_M < 64 and BLOCK_SIZE_N >= 64


@triton.jit
def write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    N,
    offs_token,
    token_mask,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    compute_type,
):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def _e2m1_to_dtype(nibble, dtype: tl.constexpr):
    # Decode a 4-bit E2M1 value (sign | 2-bit exp | 1-bit mantissa) directly
    # into `dtype`. The 8 unsigned magnitudes {0, 0.5, 1, 1.5, 2, 3, 4, 6} are
    # all exactly representable in bf16 and fp16 — no fp32 round-trip needed.
    sign = (nibble >> 3) & 0x1
    exp = (nibble >> 1) & 0x3
    mant = nibble & 0x1
    mant_d = mant.to(dtype)
    # exp == 0 (subnormal): mag = mant * 0.5            -> {0.0, 0.5}
    # exp >= 1  (normal):   mag = (1 + mant*0.5) * 2^(exp-1)
    # Use ((1 << exp) >> 1) so exp=0 yields 0 (safe — masked out by tl.where).
    pow2 = ((1 << exp) >> 1).to(dtype)
    sub_mag = mant_d * tl.full([1], 0.5, dtype)
    norm_mag = (tl.full([1], 1.0, dtype) + mant_d * tl.full([1], 0.5, dtype)) * pow2
    mag = tl.where(exp != 0, norm_mag, sub_mag)
    return tl.where(sign != 0, -mag, mag)


@triton.jit
def fused_moe_kernel_gptq_awq(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    b_zp_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N: tl.constexpr,
    K: tl.constexpr,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_bze,
    stride_bzk,
    stride_bzn,
    group_size: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    has_zp: tl.constexpr,
    use_int4_w4a16: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    even_Ks: tl.constexpr,
    filter_expert: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.
    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if filter_expert and off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )

    if use_int4_w4a16:
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + (offs_k[:, None] // 2) * stride_bk
            + offs_bn[None, :] * stride_bn
        )
        b_shifter = (offs_k[:, None] % 2) * 4
    elif use_int8_w8a16:
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + offs_k[:, None] * stride_bk
            + offs_bn[None, :] * stride_bn
        )

    if not has_zp and use_int4_w4a16:
        b_zp_num = 8
    if not has_zp and use_int8_w8a16:
        b_zp_num = 128
    elif has_zp and use_int4_w4a16:
        b_zp_shifter = (offs_bn[None, :] % 2) * 4

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.

        if not even_Ks:
            k_mask = offs_k[:, None] < K - k * BLOCK_SIZE_K
            k_other = 0.0
        else:
            k_mask = None
            k_other = None

        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(b_ptrs)
        if use_int4_w4a16:
            b = (b >> b_shifter) & 0xF

        b_scale_ptrs = (
            b_scale_ptr
            + off_experts * stride_bse
            + offs_bn[None, :] * stride_bsn
            + ((offs_k[:, None] + BLOCK_SIZE_K * k) // group_size) * stride_bsk
        )
        b_scale = tl.load(b_scale_ptrs, mask=k_mask, other=k_other)
        b_scale = b_scale.to(tl.float32)

        if has_zp and use_int4_w4a16:
            offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
            b_zp_ptrs = (
                b_zp_ptr
                + off_experts * stride_bze
                + (offs_bn[None, :] // 2) * stride_bzn
                + offs_k_true * stride_bzk
            )
            b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
            b_zp = (b_zp >> b_zp_shifter) & 0xF
            b_zp = b_zp.to(tl.float32)
        elif has_zp and use_int8_w8a16:
            offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
            b_zp_ptrs = (
                b_zp_ptr
                + off_experts * stride_bze
                + offs_bn[None, :] * stride_bzn
                + offs_k_true * stride_bzk
            )
            b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
            b_zp = b_zp.to(tl.float32)

        # We accumulate along the K dimension.
        if has_zp:
            b = ((b.to(tl.float32) - b_zp) * b_scale).to(compute_type)
        else:
            b = ((b.to(tl.float32) - b_zp_num) * b_scale).to(compute_type)
        accumulator = tl.dot(a, b, acc=accumulator)

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        if use_int4_w4a16:
            b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
        else:
            b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def fused_moe_kernel_mxfp4(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    bias_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,  # logical K (number of FP4 elements along the contracting dim)
    EM,
    num_valid_tokens,
    # Strides for A: (M, K) in `compute_type` (e.g. bf16/fp16).
    stride_am,
    stride_ak,
    # Strides for B: packed FP4 weights of shape (E, N, K // 2), uint8.
    # `stride_bk` is the stride along the (packed) K axis (i.e. per byte = 2 fp4 vals).
    stride_be,
    stride_bk,
    stride_bn,
    # Strides for C: (M, topk, N) in `compute_type`.
    stride_cm,
    stride_cn,
    # Strides for B_scale: E8M0 scales of shape (E, N, K // MX_GROUP_SIZE), uint8.
    stride_bse,
    stride_bsk,
    stride_bsn,
    # Strides for bias: (E, N).
    stride_bias_e,
    stride_bias_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # logical K-tile (must be a multiple of MX_GROUP_SIZE)
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    c_sorted: tl.constexpr,
    filter_expert: tl.constexpr,
    MX_GROUP_SIZE: tl.constexpr,  # microscaling group size, must be 32 for MXFP4
    EVEN_K: tl.constexpr,  # True when K % BLOCK_SIZE_K == 0 (skip K-edge masks)
):
    """
    Fused MoE GEMM where weights are MXFP4 (packed E2M1 values, two per byte)
    with one E8M0 (uint8) scale per group of `MX_GROUP_SIZE` (=32) elements
    along K. The per-group scales are consumed directly by `tl.dot_scaled`
    (no dequant of B is performed in software).

    Activations A are passed in `compute_type` (bf16/fp16); the LHS side of
    `tl.dot_scaled` uses an implicit unit scale (`lhs_scale=None`).

    Layout assumptions:
      - A:        (M, K)            in compute_type
      - B:        (E, N, K // 2)    uint8, two FP4 nibbles packed per byte;
                                    low nibble = even-K element, high = odd-K
      - B_scale:  (E, N, K // 32)   uint8 (E8M0)
      - C:        (M, topk, N)      in compute_type
    """
    # -----------------------------------------------------------
    # Pid -> (pid_m, pid_n) with super-grouped ordering for L2 reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Token / expert resolution.
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if filter_expert and off_experts == -1:
        write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    # ----------------------------------------------------------
    # Pointer setup.
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N

    # A is contiguous in K. We load a [BLOCK_SIZE_M, BLOCK_SIZE_K] tile once per
    # K-step and then split it into even/odd K halves for the two FP4 dots.
    a_row_base = offs_token[:, None] // top_k * stride_am
    offs_k_a = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + a_row_base + offs_k_a[None, :] * stride_ak

    # B is packed: 2 FP4 elements per byte along K. We load a
    # [PACKED_BLOCK_K, BLOCK_SIZE_N] uint8 tile per K-iteration.
    PACKED_BLOCK_K: tl.constexpr = BLOCK_SIZE_K // 2
    offs_bk_packed = tl.arange(0, PACKED_BLOCK_K)
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_bk_packed[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    # B scales: one E8M0 byte per MX_GROUP_SIZE elements along K. Both nibbles
    # in one packed byte cover K=[2i, 2i+1]; since MX_GROUP_SIZE=32 is even and
    # byte-aligned, they share the same scale group:
    #   scale_idx(byte i) = (2*i) // MX_GROUP_SIZE = i // (MX_GROUP_SIZE // 2)
    PACKED_PER_SCALE: tl.constexpr = MX_GROUP_SIZE // 2  # bytes per scale
    SCALES_PER_TILE: tl.constexpr = BLOCK_SIZE_K // MX_GROUP_SIZE
    scale_idx_in_tile = offs_bk_packed // PACKED_PER_SCALE  # [PACKED_BLOCK_K]
    bs_ptrs = (
        b_scale_ptr
        + off_experts * stride_bse
        + scale_idx_in_tile[:, None] * stride_bsk
        + offs_bn[None, :] * stride_bsn
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    K_PACKED = K // 2
    num_k_iters = tl.cdiv(K, BLOCK_SIZE_K)
    for k_iter in range(0, num_k_iters):
        k_start = k_iter * BLOCK_SIZE_K

        # ---- A: single load per K-tile ----------------------------------
        if EVEN_K:
            a_tile = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        else:
            k_lane = k_start + offs_k_a
            a_tile = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (k_lane[None, :] < K),
                other=0.0,
            )

        # ---- B: packed FP4 [PACKED_BLOCK_K, BLOCK_SIZE_N] uint8 ----------
        if EVEN_K:
            b_packed = tl.load(b_ptrs)
        else:
            packed_k_remaining = K_PACKED - (k_start // 2)
            b_packed = tl.load(
                b_ptrs,
                mask=offs_bk_packed[:, None] < packed_k_remaining,
                other=0,
            )

        # ---- B_scale gather ---------------------------------------------
        if EVEN_K:
            scale_raw = tl.load(bs_ptrs)
        else:
            scale_k_offset = k_start // MX_GROUP_SIZE
            scales_k_total = tl.cdiv(K, MX_GROUP_SIZE)
            scale_mask = (scale_idx_in_tile[:, None] + scale_k_offset) < scales_k_total
            scale_raw = tl.load(bs_ptrs, mask=scale_mask, other=0)

        # Decode E8M0 (uint8 exponent) -> 2^(e-127), then cast to compute_type.
        if scale_raw.dtype == tl.uint8:
            scale_dq = tl.math.exp2(scale_raw.to(tl.float32) - 127.0).to(compute_type)
        else:
            scale_dq = scale_raw.to(compute_type)

        # Decode E2M1 nibbles directly into compute_type. FP4 magnitudes
        # {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6} are exact in bf16/fp16.
        n_low = b_packed & 0xF
        n_high = (b_packed >> 4) & 0xF
        b_low_dq = _e2m1_to_dtype(n_low, compute_type) * scale_dq
        b_high_dq = _e2m1_to_dtype(n_high, compute_type) * scale_dq

        # Split A's K-axis into even / odd halves in-register (no extra loads).
        # A is contiguous in K, so reshape to (M, PACKED_BLOCK_K, 2) and split
        # along the trailing 2-axis: index 0 = even-K, index 1 = odd-K.
        a_even, a_odd = tl.split(a_tile.reshape(BLOCK_SIZE_M, PACKED_BLOCK_K, 2))

        accumulator += tl.dot(a_even, b_low_dq)
        accumulator += tl.dot(a_odd, b_high_dq)

        # Advance pointers. Both nibbles of one byte share a scale group, so the
        # scale pointer advances by SCALES_PER_TILE every iteration.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += PACKED_BLOCK_K * stride_bk
        bs_ptrs += SCALES_PER_TILE * stride_bsk

    if bias_ptr is not None:
        bias = tl.load(
            bias_ptr + off_experts * stride_bias_e + offs_bn[None, :] * stride_bias_n
        ).to(tl.float32)
        accumulator = accumulator + bias

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)

    # -----------------------------------------------------------
    # Store the output tile.
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if c_sorted:
        c_ptrs = (
            c_ptr + stride_cm * offs_token_id[:, None] + stride_cn * offs_cn[None, :]
        )
    else:
        c_ptrs = (
            c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        )
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# MXFP4 microscaling group size (fixed by the OCP MX spec).
_MXFP4_GROUP_SIZE = 32


@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    a_desc,
    b_ptr,
    b_desc,
    bias_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    add_mask_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_bias_e,
    stride_bias_n,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
    even_Ks: tl.constexpr,
    c_sorted: tl.constexpr,
    filter_expert: tl.constexpr,
    swap_ab: tl.constexpr,
    FUSE_ADD_TO_OUTPUT: tl.constexpr,
    FUSE_SUM_ALL_REDUCE: tl.constexpr,
    ROUTER_TOPK: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.

    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts_i32 = tl.load(expert_ids_ptr + pid_m)
    off_experts = off_experts_i32.to(tl.int64)

    if filter_expert and off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        if not FUSE_ADD_TO_OUTPUT:
            # skip the zero-write to preserve existing values.
            write_zeros_to_output(
                c_ptr,
                stride_cm,
                stride_cn,
                pid_n,
                N,
                offs_token,
                token_mask,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                compute_type,
            )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    if a_desc is not None:
        assert use_fp8_w8a8 and group_n > 0 and group_k > 0
        start_offs_m = pid_m * BLOCK_SIZE_M
    else:
        a_ptrs = a_ptr + (
            offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
        )

    if b_desc is not None:
        start_offs_n = pid_n * BLOCK_SIZE_N
    else:
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        )

    if bias_ptr is not None:
        bias = tl.load(
            bias_ptr + off_experts * stride_bias_e + offs_bn[None, :] * stride_bias_n
        )
    if use_int8_w8a16:
        b_scale_ptrs = (
            b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
        )
        b_scale = tl.load(b_scale_ptrs)

    if use_fp8_w8a8 or use_int8_w8a8:
        # block-wise
        if group_k > 0 and group_n > 0:
            if a_desc is not None:
                a_scale_ptrs = a_scale_ptr + offs_token_id * stride_asm
            else:
                a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            if BLOCK_SIZE_N > group_n:
                offs_bsn = offs_bn // group_n
            else:
                offs_bsn = pid_n * BLOCK_SIZE_N // group_n
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn
            )
        # channel-wise
        elif per_channel_quant:
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
            )
            b_scale = tl.load(b_scale_ptrs)
            # Load per-token scale for activations
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:, None]
        # tensor-wise
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    if swap_ab:
        accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_SIZE_K):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        if a_desc is not None:
            a = a_desc.load([start_offs_m, k_start])
        elif even_Ks:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None],
                other=0.0,
            )
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k_start),
                other=0.0,
            )

        if b_desc is not None:
            b = (
                b_desc.load([off_experts_i32, start_offs_n, k_start])
                .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K)
                .T
            )
        elif even_Ks:
            b = tl.load(b_ptrs)
        else:
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_start, other=0.0)

        # We accumulate along the K dimension.
        if use_int8_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:
                offs_ks = k_start // group_k
                a_scale = tl.load(
                    a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
                )
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
                if swap_ab:
                    a, b = tl.trans(b, (1, 0)), tl.trans(a, (1, 0))
                    a_scale, b_scale = b_scale, a_scale
                if BLOCK_SIZE_N > group_n:
                    accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
                else:
                    accumulator += tl.dot(a, b) * (a_scale[:, None] * b_scale)
            else:
                if use_fp8_w8a8:
                    if swap_ab:
                        a, b = tl.trans(b, (1, 0)), tl.trans(a, (1, 0))
                    accumulator = tl.dot(a, b, acc=accumulator)
                else:
                    accumulator += tl.dot(a, b)
        else:
            accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        if a_desc is None:
            a_ptrs += BLOCK_SIZE_K * stride_ak
        if b_desc is None:
            b_ptrs += BLOCK_SIZE_K * stride_bk

    if swap_ab:
        accumulator = tl.trans(accumulator, (1, 0))

    if use_int8_w8a16:
        accumulator *= b_scale
    elif use_fp8_w8a8 or use_int8_w8a8:
        if group_k == 0 or group_n == 0:
            accumulator *= a_scale * b_scale

    if bias_ptr is not None:
        accumulator += bias

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator *= moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    if FUSE_ADD_TO_OUTPUT:
        # Accumulate into existing output with per-token mask.
        offs_token_out = offs_token // ROUTER_TOPK
        add_mask = tl.load(add_mask_ptr + offs_token_out, mask=token_mask, other=False)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & add_mask[:, None] & (offs_cn[None, :] < N)
        existing = tl.load(c_ptrs, mask=c_mask, other=0.0)
        tl.store(c_ptrs, existing + accumulator, mask=c_mask)
    elif FUSE_SUM_ALL_REDUCE:
        offs_token_out = offs_token // ROUTER_TOPK
        c_ptrs = (
            c_ptr + stride_cm * offs_token_out[:, None] + stride_cn * offs_cn[None, :]
        )
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.atomic_add(c_ptrs, accumulator, mask=c_mask)
    else:
        if c_sorted:
            c_ptrs = (
                c_ptr
                + stride_cm * offs_token_id[:, None]
                + stride_cn * offs_cn[None, :]
            )
        else:
            c_ptrs = (
                c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
            )
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)


# -----------------------------------------------------------------------------
# TMA allocator: set once per process (avoid per-call triton.set_allocator)
# -----------------------------------------------------------------------------
_TMA_ALLOCATOR_SET = False


def _set_triton_tma_allocator():
    """TMA descriptors require a global allocator; set it once to avoid per-call overhead."""
    global _TMA_ALLOCATOR_SET
    if _TMA_ALLOCATOR_SET:
        return

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        # NOTE: keep this allocation on CUDA device
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)
    _TMA_ALLOCATOR_SET = True


# --- B TensorDescriptor cache (LRU) ---
_B_DESC_CACHE_MAX = 64
_B_DESC_CACHE: "OrderedDict[tuple, TensorDescriptor]" = OrderedDict()


def _get_b_tma_desc_cached(B: torch.Tensor, block_n: int, block_k: int):
    """
    Cache TensorDescriptor for constant weight B.
    Keyed by storage ptr + shape/stride/dtype + tile shape.
    """
    key = (
        int(B.data_ptr()),
        tuple(B.shape),
        tuple(B.stride()),
        str(B.dtype),
        int(block_n),
        int(block_k),
    )

    desc = _B_DESC_CACHE.get(key, None)
    if desc is not None:
        _B_DESC_CACHE.move_to_end(key)
        return desc

    # Create outside lock to reduce lock hold time (ok if duplicated rarely)
    desc = TensorDescriptor(
        B,
        B.shape,
        B.stride(),
        [1, block_n, block_k],
    )

    _B_DESC_CACHE[key] = desc
    _B_DESC_CACHE.move_to_end(key)
    if len(_B_DESC_CACHE) > _B_DESC_CACHE_MAX:
        _B_DESC_CACHE.popitem(last=False)

    return desc


def invoke_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    bias: Optional[torch.Tensor],
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    B_zp: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    compute_type: tl.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    block_shape: Optional[List[int]] = None,
    no_combine: bool = False,
    a_use_tma: bool = False,
    b_use_tma: bool = False,
    c_sorted: bool = False,
    filter_expert: bool = True,
    fuse_sum_all_reduce: bool = False,
    router_topk: int = 1,
    fuse_add_to_output: bool = False,
    add_output_mask: Optional[torch.Tensor] = None,
) -> None:
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    # TODO: Add use_mxfp4_w4a16 to thsis functions signature and
    # use values passed from framework
    # Temporary hard-coding
    use_mxfp4_w4a16 = True
    use_fp8_w8a8 = False

    if use_mxfp4_w4a16:
        assert B_scale is not None, "MXFP4 requires scales"
        assert B_scale.dtype in (torch.uint8, torch.float32), (
            f"MXFP4 expects uint8 (E8M0) or float32 scales, got {B_scale.dtype}"
        )
        assert B_zp is None and A_scale is None
        assert not (a_use_tma or b_use_tma), "TMA is not supported for MXFP4"
        assert not fuse_sum_all_reduce, "fuse_sum_all_reduce is not supported for MXFP4"

        N = B.shape[1]
        K = B.shape[2] * 2  # logical K (FP4 elements)
        assert A.shape[-1] == K, f"A K={A.shape[-1]} mismatches packed B K={K}"
        assert B_scale.shape == (B.shape[0], N, K // _MXFP4_GROUP_SIZE), (
            f"unexpected B_scale shape {tuple(B_scale.shape)}; "
            f"expected ({B.shape[0]}, {N}, {K // _MXFP4_GROUP_SIZE})"
        )
        # MXFP4 manual unpacking requires adjustments to meet tl.dot constraints:
        # - BLOCK_SIZE_K: multiple of MX_GROUP_SIZE (32), >= 64 (so PACKED_BLOCK_K >= 32)
        # - BLOCK_SIZE_N: >= 16 (tl.dot N dimension minimum)
        # - BLOCK_SIZE_M: >= 1 (tl.dot M dimension minimum, usually satisfied)
        min_block_k = max(64, _MXFP4_GROUP_SIZE)
        min_block_n = 16

        needs_adjustment = (
            config["BLOCK_SIZE_K"] < min_block_k
            or config["BLOCK_SIZE_K"] % _MXFP4_GROUP_SIZE != 0
            or config["BLOCK_SIZE_N"] < min_block_n
        )

        if needs_adjustment:
            old_config = dict(config)
            config = dict(config)  # Make a mutable copy

            # Adjust BLOCK_SIZE_K: round up to multiple of 32, minimum 64
            if config["BLOCK_SIZE_K"] < min_block_k or config["BLOCK_SIZE_K"] % _MXFP4_GROUP_SIZE != 0:
                config["BLOCK_SIZE_K"] = max(
                    min_block_k,
                    ((config["BLOCK_SIZE_K"] + _MXFP4_GROUP_SIZE - 1) // _MXFP4_GROUP_SIZE) * _MXFP4_GROUP_SIZE
                )

            # Adjust BLOCK_SIZE_N: minimum 16
            if config["BLOCK_SIZE_N"] < min_block_n:
                config["BLOCK_SIZE_N"] = min_block_n

        # tl.dot_scaled with e2m1 format requires uint8 tensors
        if B.dtype == torch.int8:
            B = B.view(torch.uint8)
        if B_scale.dtype == torch.int8:
            B_scale = B_scale.view(torch.uint8)

        grid = lambda META: (
            triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

        fused_moe_kernel_mxfp4[grid](
            A,
            B,
            C,
            B_scale,
            bias,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            N,
            K,
            sorted_token_ids.shape[0],
            topk_ids.numel(),
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            C.stride(-2),
            C.stride(-1),
            B_scale.stride(0),
            B_scale.stride(2),
            B_scale.stride(1),
            bias.stride(0) if bias is not None else 0,
            bias.stride(1) if bias is not None else 0,
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            top_k=top_k,
            compute_type=compute_type,
            c_sorted=c_sorted,
            filter_expert=filter_expert,
            MX_GROUP_SIZE=_MXFP4_GROUP_SIZE,
            EVEN_K=(K % config["BLOCK_SIZE_K"] == 0),
            **config,
        )
        return

    if use_fp8_w8a8:
        swap_ab = should_enable_swap_ab(config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"])
    else:
        swap_ab = False

    padded_size = 0
    if use_fp8_w8a8:
        assert B_scale is not None
        if block_shape is None:
            # activation tensor-wise fp8 quantization, dynamic or static
            padded_size = padding_size
            # activations apply per-token quantization when weights apply per-channel quantization by default
            A, A_scale = scaled_fp8_quant(
                A, A_scale, use_per_token_if_dynamic=per_channel_quant
            )
        else:
            # activation block-wise fp8 quantization
            assert len(block_shape) == 2
            block_n, block_k = block_shape[0], block_shape[1]
            if _is_cuda:
                A, A_scale = sglang_per_token_group_quant_fp8(A, block_k)
            else:
                A, A_scale = per_token_group_quant_fp8(A, block_k)
            assert triton.cdiv(A.shape[-1], block_k) == A_scale.shape[-1]
            assert triton.cdiv(B.shape[-2], block_n) == B_scale.shape[-2]
            assert triton.cdiv(B.shape[-1], block_k) == B_scale.shape[-1]
    elif use_int8_w8a8:
        assert B_scale is not None
        if block_shape is None:
            # activation channel-wise int8 quantization
            assert (
                per_channel_quant
            ), "int8 quantization only supports channel-wise quantization except for block-wise quantization"
            A, A_scale = per_token_quant_int8(A)
        else:
            # activation block-wise int8 quantization
            assert len(block_shape) == 2
            block_n, block_k = block_shape[0], block_shape[1]
            if _is_cuda:
                A, A_scale = sglang_per_token_group_quant_int8(A, block_k)
            else:
                A, A_scale = per_token_group_quant_int8(A, block_k)
            assert triton.cdiv(A.shape[-1], block_k) == A_scale.shape[-1]
            assert triton.cdiv(B.shape[-2], block_n) == B_scale.shape[-2]
            assert triton.cdiv(B.shape[-1], block_k) == B_scale.shape[-1]
    elif use_int8_w8a16 or use_int4_w4a16:
        assert B_scale is not None
        assert block_shape is None or block_shape[0] == 0
    else:
        assert A_scale is None
        assert B_scale is None

    grid = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )

    K = B.shape[2] - padded_size
    if K % config["BLOCK_SIZE_K"] == 0:
        even_Ks = True
    else:
        even_Ks = False

    if fuse_sum_all_reduce:
        assert not c_sorted, "fuse_sum_all_reduce only supports c_sorted=False"
    if fuse_add_to_output:
        assert (
            not fuse_sum_all_reduce
        ), "fuse_add_to_output and fuse_sum_all_reduce are mutually exclusive"
        assert (
            add_output_mask is not None
        ), "add_output_mask required when fuse_add_to_output=True"

    if (
        (use_int8_w8a16 or use_int4_w4a16)
        and block_shape is not None
        and block_shape[1] > 0
    ):
        assert (
            not fuse_sum_all_reduce
        ), "fuse_sum_all_reduce is not supported for GPTQ/AWQ kernels"
        assert B_scale is not None and B_scale.ndim == 3
        assert B_zp is None or B_zp.ndim == 3
        assert bias is None
        fused_moe_kernel_gptq_awq[grid](
            A,
            B,
            C,
            B_scale,
            B_zp,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            B.shape[1],
            A.shape[1],
            sorted_token_ids.shape[0],
            topk_ids.numel(),
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            C.stride(-2),
            C.stride(-1),
            B_scale.stride(0),
            B_scale.stride(2),
            B_scale.stride(1),
            B_zp.stride(0) if B_zp is not None else 0,
            B_zp.stride(2) if B_zp is not None else 0,
            B_zp.stride(1) if B_zp is not None else 0,
            group_size=block_shape[1],
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            top_k=top_k,
            compute_type=compute_type,
            has_zp=B_zp is not None,
            use_int4_w4a16=use_int4_w4a16,
            use_int8_w8a16=use_int8_w8a16,
            even_Ks=even_Ks,
            filter_expert=filter_expert,
            **config,
        )

    else:
        if a_use_tma or b_use_tma:
            _set_triton_tma_allocator()

        if a_use_tma:
            a_desc = TensorDescriptor(
                A, A.shape, A.stride(), [config["BLOCK_SIZE_M"], config["BLOCK_SIZE_K"]]
            )
        else:
            a_desc = None
        if b_use_tma:
            # B is constant weights -> cache descriptor
            b_desc = _get_b_tma_desc_cached(
                B,
                config["BLOCK_SIZE_N"],
                config["BLOCK_SIZE_K"],
            )
        else:
            b_desc = None

        fused_moe_kernel[grid](
            A,
            a_desc,
            B,
            b_desc,
            bias,
            C,
            A_scale,
            B_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            add_output_mask,
            B.shape[1],
            B.shape[2] - padded_size,
            sorted_token_ids.shape[0],
            topk_ids.numel(),
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            bias.stride(0) if bias is not None else 0,
            bias.stride(1) if bias is not None else 0,
            C.stride(-2),
            C.stride(-1),
            A_scale.stride(0) if A_scale is not None and A_scale.ndim == 2 else 0,
            A_scale.stride(1) if A_scale is not None and A_scale.ndim == 2 else 0,
            B_scale.stride(0) if B_scale is not None and B_scale.ndim >= 2 else 0,
            B_scale.stride(2) if B_scale is not None and B_scale.ndim == 3 else 0,
            B_scale.stride(1) if B_scale is not None and B_scale.ndim >= 2 else 0,
            0 if block_shape is None else block_shape[0],
            0 if block_shape is None else block_shape[1],
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            top_k=top_k,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            per_channel_quant=per_channel_quant,
            even_Ks=even_Ks,
            c_sorted=c_sorted,
            filter_expert=filter_expert,
            swap_ab=swap_ab,
            FUSE_ADD_TO_OUTPUT=fuse_add_to_output,
            FUSE_SUM_ALL_REDUCE=fuse_sum_all_reduce,
            ROUTER_TOPK=router_topk,
            **config,
        )


@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _apply_activation(x, ACTIVATION_TYPE: tl.constexpr):
    """
    Apply activation function based on compile-time constant.

    Args:
        x: Input tensor (converted to float32 inside)
        ACTIVATION_TYPE: Compile-time constant string ("silu" or "gelu")

    Returns:
        Activated output in the same dtype as input
    """
    x = x.to(tl.float32)
    if ACTIVATION_TYPE == "silu":
        return x * tl.sigmoid(x)
    elif ACTIVATION_TYPE == "gelu":
        kAlpha = 0.7978845608028654
        return 0.5 * x * (1 + tanh(kAlpha * (x + 0.044715 * x * x * x)))
    else:
        raise ValueError(f"Unsupported activation: {ACTIVATION_TYPE}")


@triton.jit
def act_and_mul_kernel(
    gateup_output,
    down_input,
    hidden_size,
    expert_ids_ptr,
    expert_step: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ACTIVATION_TYPE: tl.constexpr,
    SWIGLU_LIMIT: tl.constexpr = 0.0,
    HAS_SWIGLU_LIMIT: tl.constexpr = False,
):
    """
    Unified activation and multiply kernel that handles both sorted and unsorted routing,
    and both SiLU and GELU activations using compile-time constants.
    """
    InDtype = gateup_output.dtype.element_ty
    OutDtype = down_input.dtype.element_ty

    half_hidden_size = hidden_size // 2
    pid = tl.program_id(0)

    expert_id = tl.load(expert_ids_ptr + pid // expert_step)

    if expert_id == -1:
        return

    gateup_output_ptr = gateup_output + pid * hidden_size
    down_input_ptr = down_input + pid * half_hidden_size
    gate_output_ptr = gateup_output_ptr
    up_output_ptr = gateup_output_ptr + half_hidden_size

    for start_offset in tl.range(0, half_hidden_size, BLOCK_SIZE):
        offset = start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offset < half_hidden_size

        gate_output = tl.load(gate_output_ptr + offset, mask=mask)
        up_output = tl.load(up_output_ptr + offset, mask=mask)

        if HAS_SWIGLU_LIMIT:
            gate_output = tl.minimum(gate_output, SWIGLU_LIMIT)
            up_output = tl.maximum(tl.minimum(up_output, SWIGLU_LIMIT), -SWIGLU_LIMIT)

        gate_output_activated = _apply_activation(gate_output, ACTIVATION_TYPE)
        gate_output_activated = gate_output_activated.to(InDtype)

        act_mul_output = gate_output_activated * up_output
        act_mul_output = act_mul_output.to(OutDtype)
        tl.store(down_input_ptr + offset, act_mul_output, mask=mask)


def act_and_mul_triton(
    gateup_output: torch.Tensor,
    down_input: torch.Tensor,
    config: Dict[str, Any],
    topk_ids: Optional[torch.Tensor] = None,
    expert_ids: Optional[torch.Tensor] = None,
    down_moe_use_tma: bool = False,
    activation: str = "silu",
    swiglu_limit: Optional[float] = None,
) -> None:
    """
    Args:
        gateup_output: Input tensor containing gate and up outputs concatenated
        down_input: Output tensor for the result
        config: Configuration dictionary with BLOCK_SIZE_M and BLOCK_SIZE_N
        topk_ids: Expert IDs for unsorted routing (used when down_moe_use_tma=False)
        expert_ids: Expert IDs for sorted routing (used when down_moe_use_tma=True)
        down_moe_use_tma: Whether to use sorted routing layout
        activation: Activation type ("silu" or "gelu")
        swiglu_limit: if not None, clamp gate to [-inf, L] and up to [-L, L] before activation
                      (compiles a separate kernel variant via tl.constexpr).
    """
    grid = (down_input.shape[0],)
    hidden_size = gateup_output.shape[1]
    expert_ids_row = topk_ids.view(-1) if not down_moe_use_tma else expert_ids
    expert_step = 1 if not down_moe_use_tma else config["BLOCK_SIZE_M"]
    has_swiglu_limit = swiglu_limit is not None
    act_and_mul_kernel[grid](
        gateup_output,
        down_input,
        hidden_size,
        expert_ids_row,
        expert_step,
        BLOCK_SIZE=512,
        ACTIVATION_TYPE=activation,
        SWIGLU_LIMIT=float(swiglu_limit) if has_swiglu_limit else 0.0,
        HAS_SWIGLU_LIMIT=has_swiglu_limit,
    )


# _moe_sum_reduce_kernel kernel modified from https://github.com/ModelTC/lightllm/blob/main/lightllm/common/fused_moe/moe_sum_reduce.py
@triton.jit
def _moe_sum_reduce_kernel(
    input_ptr,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    output_ptr,
    output_stride_0,
    output_stride_1,
    token_num: int,
    topk_num: int,
    hidden_dim: int,
    routed_scaling_factor: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    NUM_STAGE: tl.constexpr,
):
    input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
    input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
    output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)

    token_block_id = tl.program_id(0)
    dim_block_id = tl.program_id(1)

    offs_token = token_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_dim = dim_block_id * BLOCK_DIM + tl.arange(0, BLOCK_DIM)

    mask_token = offs_token < token_num
    mask_dim = offs_dim < hidden_dim

    base_ptrs = input_ptr + offs_token[:, None] * input_stride_0 + offs_dim[None, :]

    accumulator = tl.zeros((BLOCK_M, BLOCK_DIM), dtype=tl.float32)

    for i in tl.range(0, topk_num, num_stages=NUM_STAGE):
        tile = tl.load(
            base_ptrs + i * input_stride_1,
            mask=mask_token[:, None] & mask_dim[None, :],
            other=0.0,
        )
        accumulator += tile.to(tl.float32)
    accumulator *= routed_scaling_factor

    # -------- Write back --------
    store_ptrs = output_ptr + offs_token[:, None] * output_stride_0 + offs_dim[None, :]
    tl.store(
        store_ptrs,
        accumulator.to(input_ptr.dtype.element_ty),
        mask=mask_token[:, None] & mask_dim[None, :],
    )


def moe_sum_reduce_triton(
    input: torch.Tensor, output: torch.Tensor, routed_scaling_factor: float
):
    assert input.is_contiguous()
    assert output.is_contiguous()

    token_num, topk_num, hidden_dim = input.shape
    assert output.shape[0] == token_num and output.shape[1] == hidden_dim

    BLOCK_M = 1
    BLOCK_DIM = 2048
    NUM_STAGE = 1
    num_warps = 16

    grid = (
        triton.cdiv(token_num, BLOCK_M),
        triton.cdiv(hidden_dim, BLOCK_DIM),
    )

    _moe_sum_reduce_kernel[grid](
        input,
        *input.stride(),
        output,
        *output.stride(),
        token_num=token_num,
        topk_num=topk_num,
        hidden_dim=hidden_dim,
        routed_scaling_factor=routed_scaling_factor,
        BLOCK_M=BLOCK_M,
        BLOCK_DIM=BLOCK_DIM,
        NUM_STAGE=NUM_STAGE,
        num_warps=num_warps,
    )
    return


@triton.jit
def _fused_append_shared_experts_kernel(
    topk_ids_ptr,
    topk_weights_ptr,
    out_ids_ptr,
    out_weights_ptr,
    N_BASE,  # runtime scalar
    scale_factor,  # runtime scalar
    K: tl.constexpr,
    S: tl.constexpr,
):
    """
    for m in range(M):
        for n in range(K):
            fused_ids[m, n] = topk_ids[m, n]
            fused_weights[m, n] = topk_weights[m, n]
        for s in range(S):
            fused_ids[m, K + s] = N + s
            fused_weights[m, K + s] = scale_factor
    """
    pid = tl.program_id(0)

    ids_row_ptr = pid * K
    w_row_ptr = pid * K
    out_ids_row_ptr = pid * (K + S)
    out_w_row_ptr = pid * (K + S)

    offs_k = tl.arange(0, K)
    ids = tl.load(topk_ids_ptr + ids_row_ptr + offs_k)
    ws = tl.load(topk_weights_ptr + w_row_ptr + offs_k)

    tl.store(out_ids_ptr + out_ids_row_ptr + offs_k, ids)
    tl.store(out_weights_ptr + out_w_row_ptr + offs_k, ws)

    offs_s = tl.arange(0, S)

    shared_ids = tl.cast(N_BASE + offs_s, ids.dtype)
    shared_ws = tl.full([S], scale_factor, dtype=ws.dtype)

    tl.store(out_ids_ptr + out_ids_row_ptr + K + offs_s, shared_ids)
    tl.store(out_weights_ptr + out_w_row_ptr + K + offs_s, shared_ws)


def fused_append_shared_experts(
    topk_ids, topk_weights, num_fused_shared_experts, scale_factor, N=None
):
    assert N is not None, "N (shared expert base id) must be provided"
    m, k = topk_ids.shape
    s = int(num_fused_shared_experts)
    if s <= 0:
        return topk_ids, topk_weights

    out_ids = torch.empty((m, k + s), dtype=topk_ids.dtype, device=topk_ids.device)
    out_weights = torch.empty(
        (m, k + s), dtype=topk_weights.dtype, device=topk_weights.device
    )

    _fused_append_shared_experts_kernel[(m,)](
        topk_ids,
        topk_weights,
        out_ids,
        out_weights,
        N_BASE=N,
        scale_factor=scale_factor,
        K=k,
        S=s,
        num_warps=1,
    )
    return out_ids, out_weights


@triton.jit
def _fused_append_shared_experts_with_weights_kernel(
    topk_ids_ptr,
    topk_weights_ptr,
    shared_weights_ptr,
    out_ids_ptr,
    out_weights_ptr,
    N_BASE,
    K: tl.constexpr,
    S: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)

    ids_row_ptr = pid * K
    out_row_ptr = pid * (K + S)

    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    ids = tl.load(topk_ids_ptr + ids_row_ptr + offs_k, mask=mask_k)
    ws = tl.load(topk_weights_ptr + ids_row_ptr + offs_k, mask=mask_k)

    tl.store(out_ids_ptr + out_row_ptr + offs_k, ids, mask=mask_k)
    tl.store(out_weights_ptr + out_row_ptr + offs_k, ws, mask=mask_k)

    offs_s = tl.arange(0, BLOCK_S)
    mask_s = offs_s < S
    shared_ids = tl.cast(N_BASE + offs_s, ids.dtype)
    shared_ws = tl.load(shared_weights_ptr + pid * S + offs_s, mask=mask_s)

    tl.store(out_ids_ptr + out_row_ptr + K + offs_s, shared_ids, mask=mask_s)
    tl.store(out_weights_ptr + out_row_ptr + K + offs_s, shared_ws, mask=mask_s)


def fused_append_shared_experts_with_weights(
    topk_ids, topk_weights, shared_weights, num_fused_shared_experts, N=None
):
    """Like fused_append_shared_experts but accepts per-token shared weights tensor."""
    assert N is not None, "N (shared expert base id) must be provided"
    m, k = topk_ids.shape
    s = int(num_fused_shared_experts)
    if s <= 0:
        return topk_ids, topk_weights

    shared_weights_2d = shared_weights.to(topk_weights.dtype)
    if shared_weights_2d.ndim == 1:
        shared_weights_2d = shared_weights_2d.unsqueeze(-1)
    if shared_weights_2d.shape[1] < s:
        shared_weights_2d = shared_weights_2d.expand(m, s)
    shared_weights_2d = shared_weights_2d.contiguous()

    out_ids = torch.empty((m, k + s), dtype=topk_ids.dtype, device=topk_ids.device)
    out_weights = torch.empty(
        (m, k + s), dtype=topk_weights.dtype, device=topk_weights.device
    )

    block_k = triton.next_power_of_2(k)
    block_s = triton.next_power_of_2(s)

    _fused_append_shared_experts_with_weights_kernel[(m,)](
        topk_ids,
        topk_weights,
        shared_weights_2d,
        out_ids,
        out_weights,
        N_BASE=N,
        K=k,
        S=s,
        BLOCK_K=block_k,
        BLOCK_S=block_s,
        num_warps=1,
    )
    return out_ids, out_weights
