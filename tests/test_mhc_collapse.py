from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MHC = ROOT / "vllm_musa" / "deepseek_v4_mhc.py"

try:
    import torch
except ImportError:  # the source check still runs
    torch = None

requires_musa = pytest.mark.skipif(
    torch is None or not hasattr(torch, "musa") or not torch.musa.is_available(),
    reason="requires a MUSA device",
)


def test_prefill_mhc_collapses_use_the_kernel() -> None:
    source = MHC.read_text(encoding="utf-8")
    assert "mhc_collapse(residual, pre_mix)" in source
    assert "aux = mhc_collapse(residual_cur)" in source
    assert "(pre_mix.unsqueeze(-1) * residual.float())" not in source


@requires_musa
@pytest.mark.parametrize("tokens", [1, 257, 1030])
def test_collapse_matches_the_fp32_torch_expression(tokens) -> None:
    from vllm_musa.kernels.mhc_collapse import mhc_collapse

    torch.manual_seed(tokens)
    residual = torch.randn(tokens, 4, 5120).to(torch.bfloat16).to("musa")
    mix = (torch.rand(tokens, 4) + 0.5).to("musa")
    products = mix.unsqueeze(-1) * residual.float()
    for out, ref, scale in (
        (mhc_collapse(residual, mix), products.sum(dim=1), products.abs().sum(dim=1)),
        (
            mhc_collapse(residual),
            residual.float().mean(dim=1),
            residual.float().abs().mean(dim=1),
        ),
    ):
        assert out.dtype == torch.bfloat16 and out.shape == (tokens, 5120)
        # BF16 rounding of the FP32 sum, plus FP32 summation-order error.
        assert ((out.float() - ref).abs() <= ref.abs() * 2**-8 + scale * 2**-20).all()
        assert (out == ref.to(torch.bfloat16)).float().mean().item() > 0.99
