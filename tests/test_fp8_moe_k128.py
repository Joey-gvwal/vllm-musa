"""FP8 block MoE whose gate/up projection steps K by 128 (kernels/fp8_moe_k128.py)."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
FUSED_MOE = ROOT / "vllm_musa/model_executor/layers/fused_moe/fused_moe.py"
K128 = ROOT / "vllm_musa/kernels/fp8_moe_k128.py"

requires_mp31 = pytest.mark.skipif(
    torch is None
    or not hasattr(torch, "musa")
    or not torch.musa.is_available()
    or tuple(torch.musa.get_device_capability()) != (3, 1),
    reason="requires an mp31 MUSA device",
)

# Folded layout in small: routed experts whose 32x32 scales repeat over 128-wide K
# groups, and a shared expert in the last slot with its own 32x32 scales.
ROUTED, TOP_K, HIDDEN, INTER = 32, 4, 1024, 96
EXPERTS = ROUTED + 1
LIMIT = 10.0
CONFIG = {
    "BLOCK_SIZE_M": 32,
    "BLOCK_SIZE_N": 32,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 1,
}


def _weights(seed: int = 0) -> list[torch.Tensor]:
    """w1 (E, 2I, H), w2 (E, H, I) in FP8 and their power-of-two 32x32 scales."""
    g = torch.Generator().manual_seed(seed)
    n = 2 * INTER
    w1 = (torch.randn(EXPERTS, n, HIDDEN, generator=g) * 0.5).to(torch.float8_e4m3fn)
    w2 = (torch.randn(EXPERTS, HIDDEN, INTER, generator=g) * 0.5).to(
        torch.float8_e4m3fn
    )
    exp = torch.randint(-7, -3, (EXPERTS, n // 32, HIDDEN // 128), generator=g)
    s1 = torch.exp2(exp.float()).repeat_interleave(4, dim=2)
    exp = torch.randint(-7, -3, (n // 32, HIDDEN // 32), generator=g)
    s1[ROUTED] = torch.exp2(exp.float())
    exp = torch.randint(-7, -3, (EXPERTS, HIDDEN // 32, INTER // 32), generator=g)
    s2 = torch.exp2(exp.float())
    return [t.contiguous().to("musa") for t in (w1, w2, s1, s2)]


def _batch(tokens: int, seed: int = 0) -> list[torch.Tensor]:
    """Tokens, each routed to TOP_K - 1 routed experts and to the shared slot."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(tokens, HIDDEN, generator=g).to(torch.bfloat16)
    ids = [torch.randperm(ROUTED, generator=g)[: TOP_K - 1] for _ in range(tokens)]
    ids = torch.cat([torch.stack(ids), torch.full((tokens, 1), ROUTED)], dim=1)
    w = torch.softmax(torch.randn(tokens, TOP_K - 1, generator=g), dim=-1) * 1.5
    w = torch.cat([w, torch.ones(tokens, 1)], dim=1)
    return [x.to("musa"), ids.to(torch.int32).to("musa"), w.to("musa")]


def _dequant(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """FP8 weights times their 32x32 block scales, in FP64 on CPU."""
    s = s.cpu().double()
    return w.cpu().double() * s.repeat_interleave(32, -2).repeat_interleave(32, -1)


def _reference(x, ids, w, w1, w2, s1, s2) -> torch.Tensor:
    """The MoE with the clamped SwiGLU on the dequantized experts, in FP64 on CPU."""
    x, ids, w = x.cpu().double(), ids.cpu().long(), w.cpu().double()
    out = torch.zeros(x.shape, dtype=torch.float64)
    for e in ids.unique().tolist():
        tok, slot = (ids == e).nonzero(as_tuple=True)
        h = x[tok] @ _dequant(w1[e], s1[e]).T
        gate, up = h[:, :INTER].clamp(max=LIMIT), h[:, INTER:].clamp(-LIMIT, LIMIT)
        act = torch.nn.functional.silu(gate) * up
        out.index_add_(0, tok, (act @ _dequant(w2[e], s2[e]).T) * w[tok, slot, None])
    return out


def _k128(x, ids, w, w1, w2, s1, s2) -> torch.Tensor:
    from vllm.model_executor.layers.fused_moe import override_config

    from vllm_musa.kernels.fp8_moe_k128 import fp8_moe_k128_experts

    with override_config(CONFIG):
        return fp8_moe_k128_experts(x, w1, w2, w, ids, s1, s2, [32, 32], -1, LIMIT)


def _fused_experts(x, ids, w, w1, w2, s1, s2) -> torch.Tensor:
    """The serving entry: the registered op that reaches the MUSA dispatcher."""
    from vllm.model_executor.layers.fused_moe import override_config

    import vllm_musa.model_executor.layers.fused_moe.fused_moe  # noqa: F401

    with override_config(CONFIG):
        return torch.ops.vllm.fused_experts(
            hidden_states=x,
            w1=w1,
            w2=w2,
            topk_weights=w,
            topk_ids=ids,
            activation="silu",
            use_fp8_w8a8=True,
            w1_scale=s1,
            w2_scale=s2,
            block_shape=[32, 32],
            gemm1_clamp_limit=LIMIT,
        )


def _rel(out: torch.Tensor, ref: torch.Tensor) -> float:
    return ((out.cpu().double() - ref).norm() / ref.norm()).item()


def test_dispatcher_tries_k128_in_the_config_scope_before_upstream() -> None:
    source = FUSED_MOE.read_text(encoding="utf-8")
    dispatch = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_musa_fused_experts_impl_dispatch"
    )
    scope = next(
        node
        for node in ast.walk(dispatch)
        if isinstance(node, ast.With)
        and "_upstream_triton_config_scope" in ast.unparse(node.items[0])
    )
    gate, upstream = scope.body
    assert isinstance(gate, ast.If)
    assert ast.unparse(gate.test.func) == "_musa_fp8_moe_k128_eligible"
    assert "fp8_moe_k128_experts(" in ast.unparse(gate.body[-1])
    assert "_musa_original_fused_experts_impl(" in ast.unparse(upstream)


def test_the_switch_reads_the_environment(monkeypatch) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("vllm")
    for value, enabled in (("0", False), ("1", True), (None, True)):
        if value is None:
            monkeypatch.delenv("VLLM_MUSA_FP8_MOE_W1_K128", raising=False)
        else:
            monkeypatch.setenv("VLLM_MUSA_FP8_MOE_W1_K128", value)
        spec = importlib.util.spec_from_file_location("fp8_moe_k128_switch", K128)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module._ENABLED is enabled


@requires_mp31
def test_fine_experts_flag_scales_that_change_inside_a_128_group() -> None:
    from vllm_musa.kernels import fp8_moe_k128 as k128

    scale = torch.ones(5, 3, 8, device="musa")
    scale[1, 2, 5] = 2.0  # changes inside the second 128-wide group
    scale[3, :, 4:] = 4.0  # changes only between groups
    flags, count = k128._fine_experts(scale)
    assert flags.dtype == torch.int32 and flags.tolist() == [0, 1, 0, 0, 0]
    assert count == 1
    assert k128._fine_experts(scale)[0] is flags


@requires_mp31
def test_supported_needs_32x32_blocks_aligned_k_and_few_fine_experts(
    monkeypatch,
) -> None:
    from vllm_musa.kernels import fp8_moe_k128 as k128

    w1, _, s1, _ = _weights()
    x = _batch(40)[0]
    assert k128.fp8_moe_k128_supported(x, w1, s1, [32, 32])
    assert not k128.fp8_moe_k128_supported(x, w1, s1, [128, 128])
    assert not k128.fp8_moe_k128_supported(x, w1, None, [32, 32])
    assert not k128.fp8_moe_k128_supported(x.half(), w1, s1, [32, 32])
    assert not k128.fp8_moe_k128_supported(x, w1, s1[:, :, :16], [32, 32])
    fine = s1.clone()
    fine[:, :, 1::4] *= 2  # every expert changes inside its 128-wide groups
    assert not k128.fp8_moe_k128_supported(x, w1, fine, [32, 32])
    narrow = torch.zeros(EXPERTS, 2 * INTER, HIDDEN - 32, device="musa")
    narrow = narrow.to(torch.float8_e4m3fn)
    narrow_scale = torch.ones(EXPERTS, 2 * INTER // 32, HIDDEN // 32 - 1, device="musa")
    assert not k128.fp8_moe_k128_supported(
        x[:, 32:].contiguous(), narrow, narrow_scale, [32, 32]
    )
    monkeypatch.setattr(k128, "_ENABLED", False)
    assert not k128.fp8_moe_k128_supported(x, w1, s1, [32, 32])


@requires_mp31
@pytest.mark.parametrize("tokens", [1, 10, 40, 161])
def test_w1_matches_the_exact_product_of_its_quantized_inputs(tokens) -> None:
    from vllm.model_executor.layers.fused_moe import fused_moe as fm
    from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input

    import vllm_musa.model_executor.layers.fused_moe.fused_moe  # noqa: F401
    from vllm_musa.kernels import fp8_moe_k128 as k128

    w1, _, s1, _ = _weights()
    x, ids, _ = _batch(tokens, seed=tokens)
    xq, xs = moe_kernel_quantize_input(
        A=x,
        A_scale=None,
        quant_dtype=torch.float8_e4m3fn,
        per_act_token_quant=False,
        block_shape=[32, 128],
    )
    assert xs.shape == (tokens, HIDDEN // 128)
    sorted_ids, expert_ids, post_padded = fm._prepare_expert_assignment(
        ids,
        CONFIG,
        tokens,
        TOP_K,
        EXPERTS,
        None,
        block_shape=[32, 32],
        ignore_invalid_experts=True,
    )
    out = torch.empty(tokens, TOP_K, 2 * INTER, dtype=torch.bfloat16, device="musa")
    k128.moe_w1_k128(xq, xs, w1, s1, out, sorted_ids, expert_ids, post_padded, CONFIG)
    a = xq.cpu().double() * xs.cpu().double().repeat_interleave(128, dim=1)
    ids = ids.cpu().long()
    ref = torch.empty(out.shape, dtype=torch.float64)
    mag = torch.empty_like(ref)
    for e in ids.unique().tolist():
        tok, slot = (ids == e).nonzero(as_tuple=True)
        weight = _dequant(w1[e], s1[e])
        ref[tok, slot] = a[tok] @ weight.T
        mag[tok, slot] = a[tok].abs() @ weight.abs().T
    # BF16 rounding of the sum, plus the FP32 accumulation error of the FP8 dots.
    excess = (out.cpu().double() - ref).abs() - ref.abs() * 2**-8 - mag * 2**-12
    assert excess.max() <= 0, f"max excess {excess.max().item():.3e}"


@requires_mp31
@pytest.mark.parametrize("tokens", [10, 161, 1030])
def test_moe_matches_the_fp32_reference_like_the_32_wide_kernel(tokens) -> None:
    from vllm.model_executor.layers.fused_moe import fused_moe as fm
    from vllm.model_executor.layers.fused_moe import override_config

    import vllm_musa.model_executor.layers.fused_moe.fused_moe  # noqa: F401

    weights = _weights()
    x, ids, w = _batch(tokens, seed=tokens)
    ref = _reference(x, ids, w, *weights)
    out = _k128(x, ids, w, *weights)
    w1, w2, s1, s2 = weights
    with override_config(CONFIG):
        base = fm._musa_original_fused_experts_impl(
            x,
            w1,
            w2,
            w,
            ids,
            activation="silu",
            use_fp8_w8a8=True,
            w1_scale=s1,
            w2_scale=s2,
            block_shape=[32, 32],
            gemm1_clamp_limit=LIMIT,
        )
    assert out.dtype == torch.bfloat16 and out.shape == x.shape
    assert torch.isfinite(out).all()
    # 128-wide activation groups round like the 32-wide ones of the upstream kernel.
    err, base_err = _rel(out, ref), _rel(base, ref)
    assert err < 0.06 and err < 1.5 * base_err + 0.005, (err, base_err)


@requires_mp31
def test_tokens_get_the_same_output_in_any_batch() -> None:
    weights = _weights()
    x, ids, w = _batch(161, seed=3)
    full = _k128(x, ids, w, *weights)
    for rows in (slice(0, 1), slice(5, 6), slice(0, 40), slice(100, 161)):
        part = _k128(x[rows], ids[rows].contiguous(), w[rows].contiguous(), *weights)
        assert torch.equal(part, full[rows])


@requires_mp31
def test_moe_replays_in_a_graph() -> None:
    weights = _weights()
    x, ids, w = _batch(40)
    _k128(x, ids, w, *weights)  # flags the fine experts outside capture
    side = torch.musa.Stream()
    side.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(side):
        _k128(x, ids, w, *weights)
    torch.musa.current_stream().wait_stream(side)
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        out = _k128(x, ids, w, *weights)
    for seed in (1, 2):
        new_x, new_ids, new_w = _batch(40, seed)
        x.copy_(new_x)
        ids.copy_(new_ids)
        w.copy_(new_w)
        graph.replay()
        assert torch.equal(out, _k128(new_x, new_ids, new_w, *weights))


@requires_mp31
def test_flags_are_kept_on_the_scale_and_not_computed_in_capture() -> None:
    from vllm_musa.kernels import fp8_moe_k128 as k128

    w1, _, s1, _ = _weights()
    x = _batch(40)[0]
    y = torch.empty_like(x)
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        captured = k128.fp8_moe_k128_supported(x, w1, s1, [32, 32])
        y.copy_(x)
    assert not captured and getattr(s1, k128._FINE_ATTR, None) is None
    assert k128.fp8_moe_k128_supported(x, w1, s1, [32, 32])
    assert getattr(s1, k128._FINE_ATTR)[1] == 1


@requires_mp31
@pytest.mark.parametrize("enabled", [True, False])
def test_the_registered_op_takes_the_k128_path_unless_switched_off(
    enabled, monkeypatch
) -> None:
    from vllm_musa.kernels import fp8_moe_k128 as k128

    calls = []
    real = k128.fp8_moe_k128_experts

    def spy(*args, **kwargs):
        calls.append(args[0].shape[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(k128, "fp8_moe_k128_experts", spy)
    monkeypatch.setattr(k128, "_ENABLED", enabled)
    weights = _weights()
    x, ids, w = _batch(40)
    out = _fused_experts(x, ids, w, *weights)
    assert calls == ([40] if enabled else [])
    if enabled:
        # The flags ride on the caller's scale tensor through the op boundary.
        assert getattr(weights[2], k128._FINE_ATTR)[1] == 1
        assert torch.equal(out, _k128(x, ids, w, *weights))
    err = _rel(out, _reference(x, ids, w, *weights))
    assert err < 0.06, err


@requires_mp31
def test_eligibility_keeps_other_contracts_on_the_upstream_path() -> None:
    from vllm_musa.model_executor.layers.fused_moe import fused_moe as musa_fm

    w1, _, s1, _ = _weights()
    x = _batch(40)[0]
    one = torch.ones(1, device="musa")
    base = dict(
        hidden_states=x,
        w1=w1,
        w1_scale=s1,
        activation="silu",
        apply_router_weight_on_input=False,
        use_fp8_w8a8=True,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        ocp_mx_scheme=None,
        per_channel_quant=False,
        expert_map=None,
        w1_zp=None,
        w2_zp=None,
        a1_scale=None,
        a2_scale=None,
        block_shape=[32, 32],
        w1_bias=None,
        w2_bias=None,
    )
    assert musa_fm._musa_fp8_moe_k128_eligible(**base)
    for key, value in (
        ("use_fp8_w8a8", False),
        ("use_int8_w8a8", True),
        ("activation", "gelu"),
        ("apply_router_weight_on_input", True),
        ("per_channel_quant", True),
        ("expert_map", torch.arange(EXPERTS, device="musa")),
        ("a1_scale", one),
        ("a2_scale", one),
        ("w1_bias", one),
        ("block_shape", [128, 128]),
    ):
        assert not musa_fm._musa_fp8_moe_k128_eligible(**{**base, key: value}), key
