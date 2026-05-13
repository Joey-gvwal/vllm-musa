# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch


def test_mxfp4_scale_to_float_handles_e8m0_dtype():
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is None:
        pytest.skip("torch.float8_e8m0fnu is unavailable")

    from vllm_musa.model_executor.layers.fused_moe.fused_moe import (
        _musa_mxfp4_scale_to_float,
    )

    scale_bytes = torch.tensor([120, 121, 127, 128], dtype=torch.uint8)
    scales = scale_bytes.view(e8m0_dtype)

    expected = (scale_bytes.to(torch.int32) << 23).view(torch.float32)
    torch.testing.assert_close(_musa_mxfp4_scale_to_float(scales), expected)
    torch.testing.assert_close(_musa_mxfp4_scale_to_float(scale_bytes), expected)


def test_mxfp4_moe_fallback_dequants_selected_experts_only(monkeypatch):
    from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import (
        OCP_MX_Scheme,
    )
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    monkeypatch.setattr(fused_moe.current_platform, "is_musa", lambda: True)
    calls = []
    original_dequant = fused_moe._dequant_mxfp4_musa

    def wrapped_dequant(x, scale, float_dtype):
        calls.append(tuple(x.shape))
        return original_dequant(x, scale, float_dtype)

    monkeypatch.setattr(fused_moe, "_dequant_mxfp4_musa", wrapped_dequant)

    hidden_states = torch.zeros((2, 32), dtype=torch.float32)
    topk_weights = torch.ones((2, 1), dtype=torch.float32)
    topk_ids = torch.full((2, 1), 1, dtype=torch.int64)
    w1 = torch.full((3, 64, 16), 0x22, dtype=torch.uint8)
    w2 = torch.full((3, 32, 16), 0x22, dtype=torch.uint8)
    w1_scale = torch.full((3, 64, 1), 127, dtype=torch.uint8)
    w2_scale = torch.full((3, 32, 1), 127, dtype=torch.uint8)

    out = fused_moe._musa_torch_fused_moe_fallback(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=type("FakeSiluActivation", (), {"value": "silu"})(),
        apply_router_weight_on_input=False,
        expert_map=None,
        w1_bias=None,
        w2_bias=None,
        ocp_mx_scheme=OCP_MX_Scheme.w_mxfp4,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )

    assert out.shape == hidden_states.shape
    assert calls == [(64, 16), (32, 16)]


def test_mxfp4_moe_fallback_applies_swiglu_limit():
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    hidden_states = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    topk_weights = torch.ones((1, 1), dtype=torch.float32)
    topk_ids = torch.zeros((1, 1), dtype=torch.int64)
    w1 = torch.tensor(
        [[[20.0, 0.0], [5.0, 0.0], [20.0, 0.0], [-20.0, 0.0]]],
        dtype=torch.float32,
    )
    w2 = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)

    out = fused_moe._musa_torch_fused_moe_fallback(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation="silu",
        apply_router_weight_on_input=False,
        expert_map=None,
        w1_bias=None,
        w2_bias=None,
        swiglu_limit=10.0,
    )

    gate = torch.tensor([[10.0, 5.0]], dtype=torch.float32)
    up = torch.tensor([[10.0, -10.0]], dtype=torch.float32)
    expected = torch.nn.functional.silu(gate) * up
    torch.testing.assert_close(out, expected)


def test_mxfp4_moe_fallback_supports_swigluoai_activation():
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    hidden_states = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    topk_weights = torch.ones((1, 1), dtype=torch.float32)
    topk_ids = torch.zeros((1, 1), dtype=torch.int64)
    w1 = torch.tensor(
        [[[20.0, 0.0], [5.0, 0.0], [-20.0, 0.0], [-5.0, 0.0]]],
        dtype=torch.float32,
    )
    w2 = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)

    out = fused_moe._musa_torch_fused_moe_fallback(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SWIGLUOAI,
        apply_router_weight_on_input=False,
        expert_map=None,
        w1_bias=None,
        w2_bias=None,
        swiglu_limit=10.0,
        swiglu_alpha=1.702,
        swiglu_beta=1.0,
    )

    gate = torch.tensor([[10.0, -20.0]], dtype=torch.float32)
    up = torch.tensor([[5.0, -5.0]], dtype=torch.float32)
    expected = (up + 1.0) * (gate * torch.sigmoid(gate * 1.702))
    torch.testing.assert_close(out, expected)


def test_deepgemm_post_process_upcasts_e8m0_scales_when_disabled():
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is None:
        pytest.skip("torch.float8_e8m0fnu is unavailable")

    from vllm_musa.model_executor.layers.quantization.utils.fp8_utils import (
        deepgemm_post_process_fp8_weight_block,
    )

    weight = torch.empty((2, 2), dtype=torch.uint8)
    scale_bytes = torch.tensor([126, 127, 128, 129], dtype=torch.uint8)
    scales = scale_bytes.view(e8m0_dtype).reshape(2, 2)

    out_weight, out_scales = deepgemm_post_process_fp8_weight_block(
        weight,
        scales,
        quant_block_shape=(128, 128),
        use_e8m0=False,
    )

    assert out_weight is weight
    expected = (scale_bytes.to(torch.int32) << 23).view(torch.float32).reshape(2, 2)
    assert out_scales.dtype == torch.float32
    torch.testing.assert_close(out_scales, expected)


def test_deepgemm_post_process_keeps_e8m0_scales_when_enabled():
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is None:
        pytest.skip("torch.float8_e8m0fnu is unavailable")

    from vllm_musa.model_executor.layers.quantization.utils.fp8_utils import (
        deepgemm_post_process_fp8_weight_block,
    )

    weight = torch.empty((2, 2), dtype=torch.uint8)
    scales = torch.tensor([126, 127, 128, 129], dtype=torch.uint8).view(
        e8m0_dtype
    )

    out_weight, out_scales = deepgemm_post_process_fp8_weight_block(
        weight,
        scales,
        quant_block_shape=(128, 128),
        use_e8m0=True,
    )

    assert out_weight is weight
    assert out_scales is scales
