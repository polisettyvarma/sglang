"""XPU-only cases of the JIT rmsnorm; the shared cases run from test_rmsnorm.py."""

import itertools
import sys

import pytest
import torch

from sglang.kernels.jit.utils import get_ci_test_range
from sglang.srt.utils import is_xpu
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=120, suite="stage-a-test-1-gpu-xpu")

pytestmark = pytest.mark.skipif(not is_xpu(), reason="XPU required")

EPS = 1e-6
DEVICE = "xpu"
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
# Rows CUDA's 16 B schedule cannot take (100, 1028 and 1036 run narrower vectors in
# 16-bit dtypes), short rows packed several per work-group (100), and threads whose
# last vector falls past the row (1000, 1028, 1036); fp32 is XPU-only throughout.
BS_LIST = [1, 7, 64, 257]
HIDDEN_SIZE_LIST = [100, 1000, 1028, 1036, 2048]
CASES = get_ci_test_range(
    list(itertools.product(BS_LIST, HIDDEN_SIZE_LIST)),
    [(7, 100), (3, 1000), (64, 1028), (257, 2048)],
)


def sglang_jit_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
    eps: float = EPS,
    cast_x_before_out_mul: bool = False,
) -> None:
    from sglang.kernels.ops.layernorm.norm import rmsnorm

    rmsnorm(
        input,
        weight,
        out=output,
        eps=eps,
        cast_x_before_out_mul=cast_x_before_out_mul,
    )


def sglang_jit_fused_add_rmsnorm(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
) -> None:
    from sglang.kernels.ops.layernorm.norm import fused_add_rmsnorm

    fused_add_rmsnorm(input, residual, weight, EPS)


def torch_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = EPS,
    cast_x_before_out_mul: bool = False,
) -> torch.Tensor:
    """CPU reference, the same spec as `_reference` in test_rmsnorm.py.

    With `cast_x_before_out_mul`, x * r is rounded to the input dtype before the
    weight multiply, as Hugging Face computes it.
    """
    input, weight = input.cpu(), weight.cpu()
    x = input.float()
    normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    if cast_x_before_out_mul:
        normed = normed.to(input.dtype).float()
    return (normed * weight.float()).to(input.dtype)


def _assert_close(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Within one ulp for 16-bit dtypes, as `_assert_close_16` in test_rmsnorm.py."""
    out = out.cpu()
    if out.dtype == torch.float32:
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
        return
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
    diff = (out.view(torch.int16).int() - ref.view(torch.int16).int()).abs()
    diff = torch.where((out == 0) & (ref == 0), torch.zeros_like(diff), diff)  # +-0
    assert diff.max().item() <= 1, f"max ulp distance {diff.max().item()}"


@pytest.mark.parametrize("batch_size,hidden_size", CASES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("in_place", [False, True])
def test_rmsnorm(
    batch_size: int, hidden_size: int, dtype: torch.dtype, in_place: bool
) -> None:
    input = torch.randn(batch_size, hidden_size, device=DEVICE, dtype=dtype)
    weight = torch.randn(hidden_size, device=DEVICE, dtype=dtype)
    expected = torch_rmsnorm(input, weight)

    if in_place:
        output = input.clone()
        sglang_jit_rmsnorm(output, weight, output=output)
    else:
        output = torch.empty_like(input)
        sglang_jit_rmsnorm(input, weight, output=output)

    _assert_close(output, expected)


@pytest.mark.parametrize("batch_size,hidden_size", CASES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_add_rmsnorm(
    batch_size: int, hidden_size: int, dtype: torch.dtype
) -> None:
    input = torch.randn(batch_size, hidden_size, device=DEVICE, dtype=dtype)
    residual = torch.randn(batch_size, hidden_size, device=DEVICE, dtype=dtype)
    weight = torch.randn(hidden_size, device=DEVICE, dtype=dtype)
    # The sum is rounded once to the activation dtype, and that is what gets normalized.
    expected_residual = (input.cpu().float() + residual.cpu().float()).to(dtype)
    expected = torch_rmsnorm(expected_residual, weight)

    sglang_jit_fused_add_rmsnorm(input, residual, weight)

    assert torch.equal(residual.cpu(), expected_residual)
    _assert_close(input, expected)


@pytest.mark.parametrize("fused", [False, True])
def test_rmsnorm_wide_odd_row(fused: bool) -> None:
    """The widest odd row runs: 1-element vectors at the kernel's 512 work-items.

    Hidden size 16383 in bf16 needs 32 vectors per work-item even at 512, so it
    takes the tile's byte budget (32 x 2 B) rather than a 16-vector count; a
    1024-item work-group would exceed this kernel's launch limit.
    """
    hidden_size, dtype = 16383, torch.bfloat16
    input = torch.randn(3, hidden_size, device=DEVICE, dtype=dtype)
    weight = torch.randn(hidden_size, device=DEVICE, dtype=dtype)
    if fused:
        residual = torch.randn_like(input)
        expected_residual = (input.cpu().float() + residual.cpu().float()).to(dtype)
        expected = torch_rmsnorm(expected_residual, weight)
        sglang_jit_fused_add_rmsnorm(input, residual, weight)
        assert torch.equal(residual.cpu(), expected_residual)
        _assert_close(input, expected)
    else:
        expected = torch_rmsnorm(input, weight)
        output = torch.empty_like(input)
        sglang_jit_rmsnorm(input, weight, output=output)
        _assert_close(output, expected)


@pytest.mark.parametrize("batch_size,hidden_size", [(7, 100), (64, 2048)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("cast_x_before_out_mul", [False, True])
def test_rmsnorm_cast_x_before_out_mul(
    batch_size: int, hidden_size: int, dtype: torch.dtype, cast_x_before_out_mul: bool
) -> None:
    """Each rounding mode matches its own reference, not the other one.

    The two modes differ by one ulp in about a quarter of the elements, which the
    ulp bounds in test_rmsnorm.py cannot see, so this compares exact matches.
    """
    input = torch.randn(batch_size, hidden_size, device=DEVICE, dtype=dtype)
    weight = torch.randn(hidden_size, device=DEVICE, dtype=dtype)
    output = torch.empty_like(input)

    sglang_jit_rmsnorm(
        input, weight, output=output, cast_x_before_out_mul=cast_x_before_out_mul
    )

    expected = torch_rmsnorm(input, weight, cast_x_before_out_mul=cast_x_before_out_mul)
    other = torch_rmsnorm(
        input, weight, cast_x_before_out_mul=not cast_x_before_out_mul
    )
    exact = (output.cpu() == expected).float().mean().item()
    assert exact >= 0.99, f"only {exact:.2%} of elements match the requested mode"
    assert not torch.equal(output.cpu(), other)


def test_rmsnorm_rejects_misaligned_rows() -> None:
    # 2048 runs 8-wide (16B for bf16); a 1-element row padding breaks that.
    padded = torch.randn(4, 2049, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(2048, device=DEVICE, dtype=torch.bfloat16)
    with pytest.raises(Exception, match="aligned"):
        sglang_jit_rmsnorm(
            padded[:, :2048], weight, output=torch.empty_like(padded[:, :2048])
        )


def test_fused_add_rmsnorm_rejects_misaligned_residual() -> None:
    input = torch.randn(4, 2048, device=DEVICE, dtype=torch.bfloat16)
    padded = torch.randn(4, 2049, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(2048, device=DEVICE, dtype=torch.bfloat16)
    with pytest.raises(Exception, match="aligned"):
        sglang_jit_fused_add_rmsnorm(input, padded[:, :2048], weight)


def _quant_reference(
    x: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor | None, scale: float
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """As `_reference` in test_rmsnorm.py: the rounded sum is normalized; fp8 saturates."""
    x, weight = x.cpu(), weight.cpu()
    s = (x.float() + residual.cpu().float()).to(x.dtype) if residual is not None else x
    sf = s.float()
    r = torch.rsqrt(sf.pow(2).mean(-1, keepdim=True) + EPS)
    y = (sf * r * weight.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return y, (s if residual is not None else None)


@pytest.mark.parametrize("batch_size,hidden_size", CASES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fused", [False, True])
def test_rmsnorm_quant(
    batch_size: int, hidden_size: int, dtype: torch.dtype, fused: bool
) -> None:
    from sglang.kernels.ops.layernorm.norm import fused_add_rmsnorm_quant, rmsnorm_quant

    input = torch.randn(batch_size, hidden_size, device=DEVICE, dtype=dtype)
    weight = torch.randn(hidden_size, device=DEVICE, dtype=dtype)
    residual = torch.randn_like(input) if fused else None
    scale = torch.full((1,), 0.05, device=DEVICE, dtype=torch.float32)
    expected, expected_residual = _quant_reference(
        input, weight, residual, scale.item()
    )
    out = torch.empty(batch_size, hidden_size, device=DEVICE, dtype=torch.float8_e4m3fn)

    if fused:
        fused_add_rmsnorm_quant(input, residual, weight, scale, out, EPS)
        assert torch.equal(residual.cpu(), expected_residual)
    else:
        rmsnorm_quant(input, weight, scale, out, EPS)

    # Rounding-boundary flips are one fp8 ulp at the element's magnitude.
    out = out.cpu()
    mismatch = (
        (out.view(torch.uint8) != expected.view(torch.uint8)).float().mean().item()
    )
    assert mismatch < 2e-3, f"fp8 bit mismatch fraction {mismatch}"
    torch.testing.assert_close(out.float(), expected.float(), atol=0.0, rtol=0.13)


def test_rmsnorm_quant_saturates() -> None:
    """Values past the fp8 range clamp to +-448; without saturation they would become NaN."""
    from sglang.kernels.ops.layernorm.norm import rmsnorm_quant

    input = torch.randn(8, 2048, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.ones(2048, device=DEVICE, dtype=torch.bfloat16)
    scale = torch.full((1,), 1e-3, device=DEVICE, dtype=torch.float32)
    out = torch.empty(8, 2048, device=DEVICE, dtype=torch.float8_e4m3fn)

    rmsnorm_quant(input, weight, scale, out, EPS)

    expected, _ = _quant_reference(input, weight, None, scale.item())
    out = out.cpu().float()
    assert not out.isnan().any()
    assert (out.abs() == 448.0).float().mean().item() > 0.5  # most elements saturate
    torch.testing.assert_close(out, expected.float(), atol=0.0, rtol=0.13)


def test_rmsnorm_rejects_unsupported_dtype() -> None:
    input = torch.randn(4, 128, device=DEVICE, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="dtype"):
        sglang_jit_rmsnorm(input, torch.randn(128, device=DEVICE, dtype=torch.float64))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
