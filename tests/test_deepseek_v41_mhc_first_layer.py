# SPDX-License-Identifier: Apache-2.0
"""First-layer mHC pre without a carried pre-mix (the DSpark draft's first layer)."""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MHC = REPO_ROOT / "vllm_musa" / "deepseek_v4_mhc.py"


def _function(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"{name} not found in {path}")


def test_first_layer_runs_the_decode_fuse_with_the_identity_pre_mix() -> None:
    entry = _function(MHC, "mhc_pre_delayed_musa")
    first = entry.index("if x is None and pre_mix is None and residual.dim() == 3:")
    assert entry.index("_identity_pre_mix(residual)") > first
    assert first < entry.index("_mhc_pre_delayed_big_fuse(")
    identity = _function(MHC, "_identity_pre_mix")
    assert "identity[:, 0] = 1.0" in identity
    assert "if _stream_is_capturing():" in identity
    assert "_identity_pre_mix(residual)" in _function(
        MHC, "_mhc_pre_delayed_broadcast_inputs"
    )


@pytest.mark.parametrize("num_tokens", [1, 5, 8])
def test_first_layer_fused_pre_matches_the_reference(
    num_tokens: int, monkeypatch
) -> None:
    torch = pytest.importorskip("torch")
    if getattr(torch.version, "musa", None) is None:
        pytest.skip("needs a MUSA device")
    import torchada  # noqa: F401
    from vllm.model_executor.kernels.mhc.torch import mhc_pre_delayed_torch

    import vllm_musa.deepseek_v4_mhc as mhc

    def _no_fallback(*args, **kwargs):
        raise AssertionError("first-layer pre fell back to the unfused path")

    monkeypatch.setattr(mhc, "_mhc_pre_delayed_big_fuse", _no_fallback)
    gen = torch.Generator().manual_seed(num_tokens)
    hc, hidden = 4, 5120
    residual = (
        torch.randn(num_tokens, hc, hidden, generator=gen).to(torch.bfloat16).cuda()
    )
    fn = (torch.randn(hc * (2 + hc), hc * hidden, generator=gen) * 0.02).cuda()
    hc_scale = (torch.rand(3, generator=gen) + 0.5).cuda()
    hc_base = (torch.randn(hc * (2 + hc), generator=gen) * 0.1).cuda()
    norm_weight = (torch.rand(hidden, generator=gen) + 0.5).to(torch.bfloat16).cuda()
    args = (residual, fn, hc_scale, hc_base, 1e-6, 1e-6, 1e-6, 2.0, 20)

    post, comb, layer_input, pre = mhc.mhc_pre_delayed_musa(
        *args, norm_weight=norm_weight, norm_eps=1e-6
    )
    ref_post, ref_comb, ref_input, ref_pre = mhc_pre_delayed_torch(*args)
    x = ref_input.float()
    ref_input = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)).to(
        torch.bfloat16
    ) * norm_weight

    torch.testing.assert_close(post.view_as(ref_post), ref_post, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(comb.view_as(ref_comb), ref_comb, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(pre, ref_pre, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(layer_input, ref_input, rtol=1.6e-2, atol=1e-2)
