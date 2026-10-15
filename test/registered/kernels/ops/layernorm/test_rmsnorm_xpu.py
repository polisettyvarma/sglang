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
TOLERANCE = {
    torch.float32: dict(atol=1e-5, rtol=1e-5),
    torch.float16: dict(atol=1e-2, rtol=1e-2),
    torch.bfloat16: dict(atol=2e-2, rtol=2e-2),
}


def sglang_jit_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
    eps: float = EPS,
) -> None:
    from sglang.kernels.ops.layernorm.norm import rmsnorm

    rmsnorm(input, weight, out=output, eps=eps)


def torch_rmsnorm(
    input: torch.Tensor, weight: torch.Tensor, *, eps: float = EPS
) -> torch.Tensor:
    """CPU reference, so it does not depend on the device under test."""
    input, weight = input.cpu(), weight.cpu()
    x = input.float()
    normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (normed * weight.float()).to(input.dtype)


BS_LIST = [1, 2, 7, 64, 257, 1024]
# Covers the scalar path (< 512 or not a multiple of 4), the 4- and 8-wide
# vector paths, and rows that are not a multiple of the work-group size.
HIDDEN_SIZE_LIST = [64, 100, 128, 256, 1000, 1028, 1536, 2048, 2304, 4096, 8192]
RMSNORM_CASES = get_ci_test_range(
    list(itertools.product(BS_LIST, HIDDEN_SIZE_LIST)),
    [(1, 64), (7, 100), (64, 1028), (257, 2048), (2, 4096)],
)


@pytest.mark.parametrize("batch_size,hidden_size", RMSNORM_CASES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("specify_out", [True, False])
def test_rmsnorm(
    batch_size: int, hidden_size: int, dtype: torch.dtype, specify_out: bool
) -> None:
    input = torch.randn(batch_size, hidden_size, device=DEVICE, dtype=dtype)
    weight = torch.randn(hidden_size, device=DEVICE, dtype=dtype)
    expected = torch_rmsnorm(input, weight)

    if specify_out:
        output = torch.empty_like(input)
        sglang_jit_rmsnorm(input, weight, output=output)
    else:
        output = input.clone()
        sglang_jit_rmsnorm(output, weight, output=output)

    torch.testing.assert_close(output.cpu(), expected, **TOLERANCE[dtype])


@pytest.mark.parametrize("hidden_size", [128, 2048])
def test_rmsnorm_strided_rows(hidden_size: int) -> None:
    dtype = torch.bfloat16
    padded = torch.randn(16, hidden_size + 64, device=DEVICE, dtype=dtype)
    input = padded[:, :hidden_size]
    weight = torch.randn(hidden_size, device=DEVICE, dtype=dtype)
    output = torch.empty(16, hidden_size + 128, device=DEVICE, dtype=dtype)

    sglang_jit_rmsnorm(input, weight, output=output[:, :hidden_size])

    torch.testing.assert_close(
        output[:, :hidden_size].cpu(), torch_rmsnorm(input, weight), **TOLERANCE[dtype]
    )


def test_rmsnorm_rejects_misaligned_rows() -> None:
    # 2048 runs 8-wide (16B for bf16); a 1-element row padding breaks that.
    padded = torch.randn(4, 2049, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(2048, device=DEVICE, dtype=torch.bfloat16)
    with pytest.raises(Exception, match="aligned"):
        sglang_jit_rmsnorm(
            padded[:, :2048], weight, output=torch.empty_like(padded[:, :2048])
        )


def test_rmsnorm_rejects_unsupported_dtype() -> None:
    input = torch.randn(4, 128, device=DEVICE, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="dtype"):
        sglang_jit_rmsnorm(input, torch.randn(128, device=DEVICE, dtype=torch.float64))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
