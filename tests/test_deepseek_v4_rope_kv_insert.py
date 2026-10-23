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


def test_supported_calls_take_the_triton_kernel_before_the_native_op() -> None:
    body = _function(CUSTOM_OPS, "deepseek_v4_qnorm_rope_kv_insert")
    assert (
        body.index("slot_mapping[: q.shape[0]]")
        < body.index("rope_kv_insert_supported(")
        < body.index("return rope_kv_insert(")
        < body.index("torch.ops._C_musa_ops.deepseek_v4_qnorm_rope_kv_insert(")
    )


def _rope_cache(max_pos: int) -> torch.Tensor:
    inv = 1.0 / (10000 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    angles = torch.arange(max_pos, dtype=torch.float32)[:, None] * inv[None, :]
    return torch.cat([angles.cos(), angles.sin()], dim=1).to("musa")


@requires_mp31
@pytest.mark.parametrize("tokens", [1, 5, 40, 161, 1030, 4096])
@pytest.mark.parametrize("padded", [64, 0])
@pytest.mark.parametrize("q_norm", [False, True])
def test_triton_kernel_matches_the_native_op(tokens, padded, q_norm) -> None:
    import vllm_musa._custom_ops  # noqa: F401
    from vllm_musa.kernels import deepseek_v4_rope_kv_insert as rk

    torch.manual_seed(tokens + padded)
    heads, block_size, num_blocks = 8, 64, 80
    q = (torch.randn(tokens, heads, 512) * 4).to(torch.bfloat16)
    q[1::5] = 0  # a zero query, as the DSpark context insert passes
    q = q.to("musa")
    kv = (torch.randn(tokens, 512) * 4).to(torch.bfloat16)
    kv[::3, :64] *= 1e-6  # groups under the 1e-4 amax floor
    kv = kv.to("musa")
    pos_dtype = torch.int32 if tokens % 2 == 0 else torch.int64
    positions = torch.randint(0, 8192, (tokens,), dtype=pos_dtype).to("musa")
    slots = torch.randperm(num_blocks * block_size)[:tokens]
    slots[::7] = -1  # padding slots skip the insert
    num_slots = tokens - tokens // 4  # fewer slots than rows
    slot_mapping = slots[:num_slots].to(torch.int32 if tokens % 2 else torch.int64)
    slot_mapping = slot_mapping.to("musa")
    cache = _rope_cache(8192)
    kv_cache = torch.randint(0, 256, (num_blocks, block_size, 584), dtype=torch.uint8)
    kv_cache = kv_cache.to("musa")
    assert rk.rope_kv_insert_supported(
        q, kv, kv_cache, slot_mapping, positions, cache, block_size, padded
    )

    native_q, native_cache = q.clone(), kv_cache.clone()
    out = rk.rope_kv_insert(
        q,
        kv,
        kv_cache,
        slot_mapping,
        positions,
        cache,
        1e-6,
        block_size,
        padded,
        q_norm,
    )
    ref = torch.ops._C_musa_ops.deepseek_v4_qnorm_rope_kv_insert(
        native_q,
        kv,
        native_cache,
        slot_mapping,
        positions,
        cache,
        1e-6,
        block_size,
        padded,
        q_norm,
    )
    assert out.shape == ref.shape
    assert torch.equal(out, ref)
    assert torch.equal(kv_cache, native_cache)


@requires_mp31
def test_a_strided_rope_cache_keeps_the_native_op() -> None:
    from vllm_musa.kernels import deepseek_v4_rope_kv_insert as rk

    q = torch.zeros(4, 8, 512, dtype=torch.bfloat16, device="musa")
    kv = torch.zeros(4, 512, dtype=torch.bfloat16, device="musa")
    kv_cache = torch.zeros(2, 64, 584, dtype=torch.uint8, device="musa")
    slots = torch.arange(4, device="musa")
    positions = torch.arange(4, device="musa")
    cache = _rope_cache(16)
    assert rk.rope_kv_insert_supported(q, kv, kv_cache, slots, positions, cache, 64, 64)
    # The native op reads the cache rows at a stride of 64.
    wide = torch.zeros(16, 128, device="musa")
    wide[:, :64] = cache
    assert not rk.rope_kv_insert_supported(
        q, kv, kv_cache, slots, positions, wide[:, :64], 64, 64
    )
