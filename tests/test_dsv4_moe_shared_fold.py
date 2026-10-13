from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOPK_KERNEL = (
    ROOT / "vllm_musa" / "jit_kernel" / "csrc" / "topk" / "sqrtsoftplus_topk.mu"
)
ROUTER = (
    ROOT
    / "vllm_musa"
    / "model_executor"
    / "layers"
    / "fused_moe"
    / "router"
    / "fused_topk_bias_router.py"
)
FOLD = (
    ROOT
    / "vllm_musa"
    / "model_executor"
    / "layers"
    / "fused_moe"
    / "shared_expert_fold.py"
)
MOE_MMA = ROOT / "vllm_musa" / "kernels" / "fp8_moe_mma.py"


def _function(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    node = next(
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    return ast.get_source_segment(source, node)


def test_routing_kernel_uses_the_reference_arithmetic() -> None:
    kernel = TOPK_KERNEL.read_text(encoding="utf-8")
    # softplus threshold, fast intrinsics, NaN clamp and the cleared value of
    # topkGatingSoftplusSqrt; the picked weight is the biased score minus the bias.
    assert "xv[j] > 20.0f ? xv[j] : __logf(1.0f + __expf(xv[j]))" in kernel
    assert "sqrtf(s)" in kernel and "isnan(s)" in kernel
    assert "kCleared = -10000.0f" in kernel
    assert "best - bv[j]" in kernel
    assert "other == best && other_arg < arg" in kernel
    assert "scale /= denom" in kernel


def test_router_override_keeps_hash_routing_upstream() -> None:
    gate = _function(ROUTER, "_musa_routing_supported")
    for cond in (
        'router.scoring_func == "sqrtsoftplus"',
        "router._hash_indices_table is None",
        "router.num_fused_shared_experts <= 1",
        "indices_type in (None, torch.int32)",
        "router_logits.dtype == torch.float32",
        "input_ids is not None",
    ):
        assert cond in gate
    body = _function(ROUTER, "_compute_routing")
    assert "_ORIGINAL_COMPUTE_ROUTING" in body
    assert "self.global_num_experts if self.num_fused_shared_experts else -1" in body
    assert "_get_padding_mask" in body
    assert "image_sentinel_lo=self.image_sentinel_lo" in body
    assert "topk_ids.musa_route_map = route_map" in body


def test_image_tokens_select_with_bias_vl_like_the_reference() -> None:
    kernel = TOPK_KERNEL.read_text(encoding="utf-8")
    assert "token >= rb.image_lo && token < rb.image_lo + 5" in kernel
    assert "rb.bias_vl != nullptr && rb.image_lo > 0" in kernel


def test_router_route_map_matches_the_moe_layout() -> None:
    wrapper = (
        ROOT / "vllm_musa" / "jit_kernel" / "csrc" / "sqrtsoftplus_topk.py"
    ).read_text(encoding="utf-8")
    kernel = TOPK_KERNEL.read_text(encoding="utf-8")
    mma = MOE_MMA.read_text(encoding="utf-8")
    assert "MAP_SLOTS = 64" in wrapper and "kMapSlots = 64" in kernel
    assert "_MAX_ROUTED = 64" in mma
    assert "MAP_WIDTH = 18" in wrapper and "kMapWidth = 2 + kMapRows" in kernel
    assert (
        "kMapRows = 16" in kernel and "_expert_rows(map_ptr, u, TOPK, MAX_R, 16)" in mma
    )
    assert 'getattr(topk_ids, "musa_route_map", None)' in _function(
        MOE_MMA, "fp8_moe_mma"
    )


def test_moe_dot_path_fits_a_folded_top7_at_verify_batch() -> None:
    tree = ast.parse(MOE_MMA.read_text(encoding="utf-8"))
    limit = next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_MAX_ROUTED" for t in n.targets)
    )
    assert limit >= 7 * 9


def _moe(**overrides):
    proj = lambda rows, cols: SimpleNamespace(  # noqa: E731
        quant_method=SimpleNamespace(block_quant=True, weight_block_size=[32, 32]),
        weight=SimpleNamespace(shape=(rows, cols)),
    )
    moe = SimpleNamespace(
        shared_experts=SimpleNamespace(
            gate_up_proj=proj(576, 5120), down_proj=proj(5120, 288)
        ),
        use_sequence_parallel=False,
        scoring_func="sqrtsoftplus",
        gate=SimpleNamespace(tid2eid=None, bias_vl=None),
    )
    for key, value in overrides.items():
        setattr(moe, key, value)
    return moe


def _config(**overrides):
    parallel = SimpleNamespace(
        enable_expert_parallel=False,
        enable_eplb=False,
        data_parallel_size=1,
        tensor_parallel_size=8,
    )
    for key, value in overrides.items():
        setattr(parallel, key, value)
    return SimpleNamespace(parallel_config=parallel)


def test_shared_fold_eligibility(monkeypatch) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from vllm_musa.model_executor.layers.fused_moe import shared_expert_fold as fold

    monkeypatch.setattr(fold, "_ENABLED", True)
    hf = SimpleNamespace(
        n_shared_experts=1, moe_intermediate_size=2304, hidden_size=5120
    )
    quant = SimpleNamespace(expert_dtype="fp8", expert_weight_block_size=[32, 32])
    assert fold.deepseek_v4_shared_expert_foldable(_config(), _moe(), hf, quant)
    # 128x128 checkpoint expert blocks load refined to 32x32 for a 288-wide shard.
    coarse = SimpleNamespace(expert_dtype="fp8", expert_weight_block_size=[128, 128])
    assert fold.deepseek_v4_shared_expert_foldable(_config(), _moe(), hf, coarse)
    wide = SimpleNamespace(
        n_shared_experts=1, moe_intermediate_size=4608, hidden_size=5120
    )
    assert not fold.deepseek_v4_shared_expert_foldable(_config(), _moe(), wide, coarse)
    assert not fold.deepseek_v4_shared_expert_foldable(
        _config(enable_expert_parallel=True), _moe(), hf, quant
    )
    assert not fold.deepseek_v4_shared_expert_foldable(
        _config(), _moe(gate=SimpleNamespace(tid2eid=object(), bias_vl=None)), hf, quant
    )
    assert not fold.deepseek_v4_shared_expert_foldable(
        _config(), _moe(), hf, SimpleNamespace(expert_dtype="fp4")
    )
    assert not fold.deepseek_v4_shared_expert_foldable(
        _config(tensor_parallel_size=4), _moe(), hf, quant
    )
    monkeypatch.setattr(fold, "_ENABLED", False)
    assert not fold.deepseek_v4_shared_expert_foldable(_config(), _moe(), hf, quant)


try:
    import torch
except ImportError:  # source checks above still run
    torch = None

requires_mp31 = pytest.mark.skipif(
    torch is None
    or not hasattr(torch, "musa")
    or not torch.musa.is_available()
    or tuple(torch.musa.get_device_capability()) != (3, 1),
    reason="requires an mp31 MUSA device",
)


@requires_mp31
@pytest.mark.parametrize("experts,top_k", [(384, 6), (128, 3)])
@pytest.mark.parametrize("rows", [1, 5, 37, 512])
def test_routing_matches_topk_softplus_sqrt_bitwise(experts, top_k, rows) -> None:
    from vllm import _custom_ops as ops

    import vllm_musa._custom_ops  # noqa: F401
    from vllm_musa.jit_kernel.csrc.sqrtsoftplus_topk import sqrtsoftplus_topk

    torch.manual_seed(rows + experts)
    logits = torch.randn(rows, experts, device="musa") * 3.0
    logits[:, ::7] += 25.0
    bias = torch.randn(experts, device="musa") * 0.3
    is_padding = torch.zeros(rows, dtype=torch.bool, device="musa")
    is_padding[rows // 2 :] = rows > 1
    ref_w = torch.empty(rows, top_k, device="musa")
    ref_i = torch.empty(rows, top_k, dtype=torch.int32, device="musa")
    tei = torch.empty(rows, top_k, dtype=torch.int32, device="musa")
    ops.topk_hash_softplus_sqrt(
        ref_w, ref_i, tei, logits, True, 1.5, bias, None, None, is_padding=is_padding
    )
    x = torch.randn(rows, 5120, device="musa").to(torch.bfloat16)
    w, i, route_map, x_quant = sqrtsoftplus_topk(
        logits,
        bias,
        top_k,
        True,
        1.5,
        is_padding=is_padding,
        shared_id=experts,
        with_route_map=True,
        x=x,
    )
    real = ~is_padding
    assert torch.equal(i[real, :top_k], ref_i[real])
    assert torch.equal(w[real, :top_k].view(torch.int32), ref_w[real].view(torch.int32))
    assert bool((i[real, top_k] == experts).all()) and bool(
        (w[real, top_k] == 1.0).all()
    )
    assert bool((i[~real] == -1).all())
    if route_map is not None:
        from vllm_musa.kernels import fp8_moe_mma

        flat = i.reshape(-1).contiguous()
        expect = torch.zeros_like(route_map)
        fp8_moe_mma._moe_route_map_kernel[(1,)](
            flat, expect, flat.numel(), experts + 1, MAX_R=64, BLOCK_M=16, num_warps=4
        )
        distinct = int(expect[64 * 18])
        assert int(route_map[64 * 18]) == distinct
        for u in range(distinct):
            got, ref = route_map[u * 18 : u * 18 + 18], expect[u * 18 : u * 18 + 18]
            assert torch.equal(got[: 2 + int(ref[1])], ref[: 2 + int(ref[1])])
        from vllm_musa.jit_kernel.csrc import fp8_block_gemv as native

        xq, xs = native.act_quant(x)
        assert torch.equal(x_quant[0].view(torch.uint8), xq.view(torch.uint8))
        assert torch.equal(x_quant[1], xs)


@requires_mp31
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64] if torch else [None])
def test_image_token_routing_matches_reference_bitwise(ids_dtype) -> None:
    from vllm import _custom_ops as ops

    import vllm_musa._custom_ops  # noqa: F401
    from vllm_musa.jit_kernel.csrc.sqrtsoftplus_topk import sqrtsoftplus_topk

    torch.manual_seed(5)
    rows, experts, top_k, lo = 24, 384, 6, 129270
    logits = torch.randn(rows, experts, device="musa") * 3.0
    bias = torch.randn(experts, device="musa") * 0.3
    bias_vl = torch.randn(experts, device="musa") * 0.5
    input_ids = torch.randint(0, 129280, (rows,), device="musa")
    input_ids[::3] = lo + torch.arange(0, rows, 3, device="musa") % 6
    input_ids = input_ids.to(ids_dtype)
    ref_w = torch.empty(rows, top_k, device="musa")
    ref_i = torch.empty(rows, top_k, dtype=torch.int32, device="musa")
    tei = torch.empty(rows, top_k, dtype=torch.int32, device="musa")
    ops.topk_hash_softplus_sqrt(
        ref_w,
        ref_i,
        tei,
        logits,
        True,
        1.5,
        bias,
        input_ids,
        None,
        bias_vl=bias_vl,
        image_sentinel_lo=lo,
    )
    w, i, _, _ = sqrtsoftplus_topk(
        logits,
        bias,
        top_k,
        True,
        1.5,
        bias_vl=bias_vl,
        input_ids=input_ids,
        image_sentinel_lo=lo,
    )
    assert torch.equal(i, ref_i)
    assert torch.equal(w.view(torch.int32), ref_w.view(torch.int32))


@requires_mp31
@pytest.mark.parametrize("tokens", [4, 5, 9])
def test_folded_shared_slot_equals_routed_plus_shared(tokens) -> None:
    from vllm_musa.kernels import fp8_moe_mma
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    from .test_fp8_block_gemv_native import _block_fp8_weight

    torch.manual_seed(41 + tokens)
    experts, top_k, hidden, inter = 64, 6, 5120, 288
    w1, w1s = _block_fp8_weight(2 * inter, hidden, experts + 1)
    w2, w2s = _block_fp8_weight(hidden, inter, experts + 1)
    x = torch.randn(tokens, hidden).to(torch.bfloat16).to("musa")
    ids = torch.stack([torch.randperm(experts)[:top_k] for _ in range(tokens)])
    ids = (
        torch.cat([ids, torch.full((tokens, 1), experts)], 1).to(torch.int32).to("musa")
    )
    weights = torch.cat(
        [torch.softmax(torch.randn(tokens, top_k), -1) * 1.5, torch.ones(tokens, 1)], 1
    ).to("musa")
    assert fp8_moe_mma.fp8_moe_mma_supported(x, w1, w2, w1s, w2s, ids)
    got = fp8_moe_mma.fp8_moe_mma(x, w1, w2, weights, ids, w1s, w2s, 10.0)
    ref = torch.zeros(tokens, hidden, device="musa")
    for cols in (slice(0, top_k), slice(top_k, top_k + 1)):
        for t in range(tokens):
            ref[t] += fused_moe._musa_fp8_block_moe_gemv(
                x[t : t + 1],
                w1,
                w2,
                weights[t : t + 1, cols].contiguous(),
                ids[t : t + 1, cols].contiguous(),
                w1s,
                w2s,
                [32, 32],
                10.0,
            )[0].float()
    torch.testing.assert_close(got.float(), ref, rtol=3e-2, atol=3e-2)
