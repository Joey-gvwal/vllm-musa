from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GEMV = ROOT / "vllm_musa" / "kernels" / "fp8_block_gemv.py"
NATIVE = ROOT / "vllm_musa" / "jit_kernel" / "csrc" / "fp8_block_gemv.py"
KERNEL = ROOT / "vllm_musa" / "jit_kernel" / "csrc" / "gemv" / "fp8_block_gemv.mu"
LINEAR = (
    ROOT
    / "vllm_musa"
    / "model_executor"
    / "kernels"
    / "linear"
    / "scaled_mm"
    / "block_gemv.py"
)
FUSED_MOE = (
    ROOT / "vllm_musa" / "model_executor" / "layers" / "fused_moe" / "fused_moe.py"
)


def _function(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    node = next(
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    return ast.get_source_segment(source, node)


def _literal(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found")


def test_native_tables_cover_eight_rows_with_known_configs() -> None:
    kernel = KERNEL.read_text(encoding="utf-8")
    known = {int(c) for c in re.findall(r"case (\d+): launch_dense", kernel)}
    for name in ("_NATIVE_BF16_CFG", "_NATIVE_FP8_CFG"):
        table = _literal(GEMV, name)
        assert (1792, 5120) in table
        for cfgs in table.values():
            assert len(cfgs) == 8
            assert {c for c in cfgs if c is not None} <= known


def test_native_gemv_needs_32x32_fp32_scales_and_has_a_kill_switch() -> None:
    source = _function(GEMV, "_native_cfg")
    assert "_NATIVE_GEMV" in source
    assert "_weight_groups(weight, weight_scale) != (32, 32)" in source
    assert "torch.float32" in source
    assert 'os.environ.get("VLLM_MUSA_FP8_NATIVE_GEMV", "1")' in GEMV.read_text(
        encoding="utf-8"
    )


def test_quantizing_gemv_keeps_swiglu_on_the_triton_kernel() -> None:
    source = _function(GEMV, "fp8_block_gemv_quant")
    assert "swiglu_limit is None and group_size == 32" in source
    assert "native.act_quant(x)" in source


def test_native_moe_gate_matches_the_instantiated_down_shape() -> None:
    gate = _function(FUSED_MOE, "_native_fp8_block_moe_gemv_supported")
    assert "topk_ids.shape[1] in (3, 6)" in gate
    assert "w2.shape[2] == 288" in gate
    assert "_NATIVE_FP8_BLOCK_MOE_GEMV_MIN_TOKENS" in gate
    assert "gemm1_clamp_limit is not None" in gate
    assert "list(block_shape) == [32, 32]" in gate
    kernel = KERNEL.read_text(encoding="utf-8")
    assert "MOE_W2_CASE(6, 288)" in kernel and "MOE_W2_CASE(3, 288)" in kernel
    body = _function(FUSED_MOE, "_musa_fp8_block_moe_gemv")
    assert body.index("_native_fp8_block_moe_gemv_supported") < body.index(
        "\n    fp8_block_moe_gemv("
    )
    assert "expert_ids.to(torch.int32)" in body
    assert "to(torch.float32)" in body


def test_swiglu_down_projection_past_gemv_rows_quantizes_once() -> None:
    source = _function(LINEAR, "apply_swiglu_clamp_weights")
    assert "rows > _gemv_max_m(params.weight)" in source
    helper = _function(LINEAR, "_swiglu_clamp_block_mm")
    assert "native.act_quant(gate_up.view(rows, -1), swiglu_limit)" in helper
    assert "fp8_block_scaled_mm(" in helper


def test_activation_quant_divides_like_per_token_group_quant() -> None:
    kernel = KERNEL.read_text(encoding="utf-8")
    assert "#pragma float_control(precise, on, push)" in kernel
    assert "div_rn(v[i * 4], s)" in kernel
    assert "__musa_cvt_float4_to_fp8x4(f4, __MT_SATFINITE, __MT_E4M3)" in kernel


def test_custom_ops_mutate_only_their_outputs() -> None:
    source = NATIVE.read_text(encoding="utf-8")
    for op, mutated in (
        ("musa_fp8_gemv_act_quant", '["xq", "xs"]'),
        ("musa_fp8_dense_gemv", '["out"]'),
        ("musa_fp8_moe_w1_gemv", '["out"]'),
        ("musa_fp8_moe_w2_gemv", '["out"]'),
    ):
        block = source[source.index(f'op_name="{op}"') :]
        assert f"mutates_args={mutated}" in block[: block.index(")")]


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


def _block_fp8_weight(n: int, k: int, experts: int | None = None):
    shape = (n, k) if experts is None else (experts, n, k)
    w = torch.randn(*shape) * 0.02
    blocks = w.view(*shape[:-2], n // 32, 32, k // 32, 32)
    scale = torch.exp2(torch.ceil(torch.log2(blocks.abs().amax(dim=(-3, -1)) / 448.0)))
    full = scale.repeat_interleave(32, -2).repeat_interleave(32, -1)
    weight = (w / full).clamp(-448, 448).to(torch.float8_e4m3fn)
    return weight.to("musa"), scale.float().contiguous().to("musa")


def _dequant(weight, scale):
    return weight.float() * scale.repeat_interleave(32, -2).repeat_interleave(32, -1)


@requires_mp31
@pytest.mark.parametrize("shape", [(1792, 5120), (1024, 4096), (576, 5120)])
@pytest.mark.parametrize("rows", [1, 3, 5, 8])
def test_native_dense_gemv_matches_triton(shape, rows) -> None:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )

    import vllm_musa.kernels.fp8_block_gemv as gemv

    torch.manual_seed(rows)
    n, k = shape
    weight, scale = _block_fp8_weight(n, k)
    x = torch.randn(rows, k).to(torch.bfloat16).to("musa")
    xq, xs = per_token_group_quant_fp8(x, 32, use_ue8m0=False)
    ref = (xq.float() * xs.repeat_interleave(32, -1)) @ _dequant(weight, scale).T
    for fn, args in (
        (gemv.fp8_block_gemv_quant, (x, weight, scale)),
        (gemv.fp8_block_gemv, (xq, xs, weight, scale)),
    ):
        outs = []
        for native in (False, True):
            gemv._NATIVE_GEMV = native
            out = torch.empty((rows, n), dtype=torch.bfloat16, device="musa")
            outs.append(
                fn(*args, out, 32)
                if fn is gemv.fp8_block_gemv_quant
                else fn(*args, out)
            )
        gemv._NATIVE_GEMV = True
        torch.testing.assert_close(
            outs[1].float(), outs[0].float(), rtol=2e-2, atol=2e-2
        )
        torch.testing.assert_close(outs[1].float(), ref, rtol=3e-2, atol=3e-2)


@requires_mp31
def test_native_activation_quant_is_bit_exact() -> None:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )

    from vllm_musa.jit_kernel.csrc import fp8_block_gemv as native

    torch.manual_seed(7)
    x = (torch.randn(6, 5120) * 4).to(torch.bfloat16).to("musa")
    xq, xs = native.act_quant(x)
    rq, rs = per_token_group_quant_fp8(x, 32, use_ue8m0=False)
    assert torch.equal(xq.view(torch.uint8), rq.view(torch.uint8))
    assert torch.equal(xs, rs)

    gate_up = (torch.randn(5, 576) * 3).to(torch.bfloat16).to("musa")
    aq, a_scale = native.act_quant(gate_up, 10.0)
    gate = gate_up.float()[:, :288].clamp(max=10.0)
    up = gate_up.float()[:, 288:].clamp(-10.0, 10.0)
    silu = (gate / (1 + torch.exp(-gate))).to(torch.bfloat16).float()
    rq, rs = per_token_group_quant_fp8(
        (silu * up).to(torch.bfloat16), 32, use_ue8m0=False
    )
    assert torch.equal(aq.view(torch.uint8), rq.view(torch.uint8))
    assert torch.equal(a_scale, rs)


@requires_mp31
@pytest.mark.parametrize("tokens", [1, 2, 5])
def test_native_moe_gemv_matches_triton(tokens) -> None:
    from vllm_musa.kernels import fp8_moe_mma
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    torch.manual_seed(11 + tokens)
    experts, top_k, hidden, inter = 64, 6, 5120, 288
    w1, w1s = _block_fp8_weight(2 * inter, hidden, experts)
    w2, w2s = _block_fp8_weight(hidden, inter, experts)
    x = torch.randn(tokens, hidden).to(torch.bfloat16).to("musa")
    ids = torch.stack([torch.randperm(experts)[:top_k] for _ in range(tokens)])
    ids = ids.to(torch.int32).to("musa")
    weights = torch.softmax(torch.randn(tokens, top_k), -1).to("musa")
    outs = []
    fp8_moe_mma._ENABLED = False
    try:
        for native in (False, True):
            fused_moe._NATIVE_FP8_BLOCK_MOE_GEMV = native
            outs.append(
                fused_moe._musa_fp8_block_moe_gemv(
                    x, w1, w2, weights, ids, w1s, w2s, [32, 32], 10.0
                )
            )
    finally:
        fused_moe._NATIVE_FP8_BLOCK_MOE_GEMV = True
        fp8_moe_mma._ENABLED = True
    torch.testing.assert_close(outs[1].float(), outs[0].float(), rtol=1e-2, atol=1e-2)


SPLITK = ROOT / "vllm_musa" / "kernels" / "fp8_block_splitk.py"
MOE_MMA = ROOT / "vllm_musa" / "kernels" / "fp8_moe_mma.py"


def test_splitk_configs_tile_k_and_keep_one_row_on_the_gemv() -> None:
    configs = _literal(SPLITK, "_CONFIGS")
    for (n, k), (split, block_n, warps, unroll, min_rows) in configs.items():
        assert k % (split * 32 * unroll) == 0
        assert min_rows >= 2
    source = _function(SPLITK, "fp8_block_splitk_config")
    assert "rows < config[4]" in source
    assert "2 <= rows <= MAX_ROWS" in source


def test_splitk_reduction_sums_partials_in_a_fixed_order() -> None:
    source = SPLITK.read_text(encoding="utf-8")
    reduce = source[source.index("def _fp8_block_splitk_reduce_kernel") :]
    assert "for s in tl.static_range(SPLIT)" in reduce
    assert "atomic" not in source


def test_dense_entry_points_try_splitk_first() -> None:
    for name in ("fp8_block_gemv", "fp8_block_gemv_quant"):
        body = _function(GEMV, name)
        assert body.index("fp8_block_splitk_config") < body.index("_native_cfg")
    assert "_small_m_path(params.weight, params.block_scale, rows)" in LINEAR.read_text(
        encoding="utf-8"
    )


def test_moe_dot_path_reads_each_expert_once_and_bounds_rows() -> None:
    source = MOE_MMA.read_text(encoding="utf-8")
    assert "_moe_route_map_kernel" in source
    supported = _function(MOE_MMA, "fp8_moe_mma_supported")
    assert "tokens * top_k <= _MAX_ROUTED" in supported
    assert "tokens <= 16" in supported
    body = _function(FUSED_MOE, "_musa_fp8_block_moe_gemv")
    assert body.index("_fp8_moe_mma_eligible") < body.index("native.act_quant")
    gate = _function(FUSED_MOE, "_fp8_moe_mma_eligible")
    assert "fp8_moe_mma_supported" in gate
    assert "gemm1_clamp_limit is None" in gate
    assert "MIN_TOKENS" in gate
    assert "_fp8_moe_mma_eligible" in _function(
        FUSED_MOE, "_can_use_musa_fp8_block_moe_gemv"
    )


@requires_mp31
@pytest.mark.parametrize("tokens", [4, 5])
def test_moe_dot_path_matches_triton(tokens) -> None:
    from vllm_musa.kernels import fp8_moe_mma
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    torch.manual_seed(23 + tokens)
    experts, top_k, hidden, inter = 64, 6, 5120, 288
    w1, w1s = _block_fp8_weight(2 * inter, hidden, experts)
    w2, w2s = _block_fp8_weight(hidden, inter, experts)
    x = torch.randn(tokens, hidden).to(torch.bfloat16).to("musa")
    ids = torch.stack([torch.randperm(experts)[:top_k] for _ in range(tokens)])
    ids[1:, :3] = ids[:-1, :3]
    ids = ids.to(torch.int32).to("musa")
    weights = torch.softmax(torch.randn(tokens, top_k), -1).to("musa")
    assert fp8_moe_mma.fp8_moe_mma_supported(x, w1, w2, w1s, w2s, ids)
    fused_moe._NATIVE_FP8_BLOCK_MOE_GEMV = False
    fp8_moe_mma._ENABLED = False
    try:
        ref = fused_moe._musa_fp8_block_moe_gemv(
            x, w1, w2, weights, ids, w1s, w2s, [32, 32], 10.0
        )
    finally:
        fused_moe._NATIVE_FP8_BLOCK_MOE_GEMV = True
        fp8_moe_mma._ENABLED = True
    got = fp8_moe_mma.fp8_moe_mma(x, w1, w2, weights, ids, w1s, w2s, 10.0)
    torch.testing.assert_close(got.float(), ref.float(), rtol=3e-2, atol=3e-2)
