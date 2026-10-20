from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CUSTOM_OPS = ROOT / "vllm_musa" / "_custom_ops.py"

try:
    import torch
except ImportError:  # the source check still runs
    torch = None

requires_mp31 = pytest.mark.skipif(
    torch is None
    or not hasattr(torch, "musa")
    or not torch.musa.is_available()
    or tuple(torch.musa.get_device_capability()) != (3, 1),
    reason="requires an mp31 MUSA device",
)


def _function(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    node = next(
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    return ast.get_source_segment(source, node)


def test_fp32_scale_calls_take_the_triton_kernel_before_the_native_op() -> None:
    body = _function(CUSTOM_OPS, "deepseek_v4_fused_inv_rope_fp8_quant")
    assert (
        body.index("inv_rope_fp8_quant_supported(")
        < body.index("return inv_rope_fp8_quant(")
        < body.index("torch.ops._C_musa_ops.deepseek_v4_fused_inv_rope_fp8_quant(")
    )


def _rope_cache(max_pos: int) -> torch.Tensor:
    inv = 1.0 / (10000 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    angles = torch.arange(max_pos, dtype=torch.float32)[:, None] * inv[None, :]
    return torch.cat([angles.cos(), angles.sin()], dim=1).to("musa")


def _reference(o, positions, cache):
    """FP32 inverse RoPE of each head and its power-of-two per-128 scales."""
    x = o.float().clone()
    cos, sin = cache[positions, :32][:, None, :], cache[positions, 32:][:, None, :]
    even, odd = x[..., 448::2].clone(), x[..., 449::2].clone()
    x[..., 448::2] = even * cos + odd * sin
    x[..., 449::2] = odd * cos - even * sin
    x = x.view(*x.shape[:-1], 4, 128)
    scale = (x.abs().amax(-1) / 448.0).clamp_min(1e-10)
    return x, torch.exp2(torch.ceil(torch.log2(scale)))


@requires_mp31
@pytest.mark.parametrize("tokens", [1, 5, 40, 161, 1030])
def test_triton_kernel_matches_the_native_op(tokens) -> None:
    import vllm_musa._custom_ops  # noqa: F401
    from vllm_musa.kernels import deepseek_v4_inv_rope_fp8_quant as ir

    torch.manual_seed(tokens)
    heads, groups = 8, 1
    o = (torch.randn(tokens, heads, 512) * 4).to(torch.bfloat16).to("musa")
    positions = torch.randint(0, 8192, (tokens,), dtype=torch.int64).to("musa")
    cache = _rope_cache(8192)
    assert ir.inv_rope_fp8_quant_supported(o, positions, cache, 448, 64, 128, False)
    assert not ir.inv_rope_fp8_quant_supported(o, positions, cache, 448, 64, 128, True)
    q, scales = ir.inv_rope_fp8_quant(o, positions, cache, groups, heads)
    native = torch.ops._C_musa_ops.deepseek_v4_fused_inv_rope_fp8_quant
    native_q, native_scales = native(
        o, positions, cache, groups, heads, 448, 64, 128, False
    )
    assert q.shape == native_q.shape and q.stride() == native_q.stride()
    assert scales.shape == native_scales.shape
    assert scales.stride() == native_scales.stride()
    assert torch.equal(scales, native_scales)
    codes, native_codes = q.view(torch.uint8), native_q.view(torch.uint8)
    assert (codes != native_codes).float().mean().item() < 1e-4

    x, ref_scales = _reference(o, positions, cache)
    assert torch.equal(scales[:, 0], ref_scales.view(tokens, heads * 4))
    # Within half an FP8 step of the rotated value: 2^-4 relative, 2^-10 scales below.
    s = scales.view(tokens, heads, 4)[..., None]
    deq = q.float().view(tokens, heads, 4, 128) * s
    assert ((deq - x).abs() <= x.abs() * 2**-4 + s * 2**-10).all()

    aligned = (tokens + 3) // 4 * 4
    padded = scales.transpose(0, 1).as_strided(
        (groups, aligned, heads * 4), (heads * 4 * aligned, heads * 4, 1)
    )
    assert torch.count_nonzero(padded[:, tokens:]) == 0
