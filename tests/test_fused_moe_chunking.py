# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for MUSA fused MoE chunked execution."""

import torch


def _patch_fake_moe_kernels(monkeypatch, fused_moe, top_k):
    def fake_musa_fused_gemv_moe(
        input_tensor,
        weight,
        output,
        _bias,
        _scale,
        _topk_weights,
        current_topk_ids,
        *_args,
        use_swigelu,
        **_kwargs,
    ):
        tokens = current_topk_ids.shape[0]
        if use_swigelu:
            output[: tokens * top_k].fill_(1)
        else:
            output[:tokens].fill_(2)

    def fake_moe_sum(intermediate_cache3, output):
        output.copy_(intermediate_cache3.sum(dim=1))

    monkeypatch.setattr(
        fused_moe.musa_ops, "musa_fused_gemv_moe", fake_musa_fused_gemv_moe
    )
    monkeypatch.setattr(fused_moe.ops, "moe_sum", fake_moe_sum)


def test_musa_fused_experts_preserves_output_shape_across_chunks(monkeypatch):
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    chunk_size = 16384
    num_tokens = chunk_size + 3
    hidden_size = 4
    intermediate_size = 8
    num_experts = 2
    top_k = 1

    hidden_states = torch.zeros(num_tokens, hidden_size, dtype=torch.float32)
    w1 = torch.zeros(num_experts, intermediate_size, hidden_size)
    w2 = torch.zeros(num_experts, hidden_size, intermediate_size // 2)
    topk_weights = torch.ones(num_tokens, top_k, dtype=torch.float32)
    topk_ids = torch.zeros(num_tokens, top_k, dtype=torch.int64)

    second_gemm_calls = 0

    def fake_musa_fused_gemv_moe(
        input_tensor,
        weight,
        output,
        _bias,
        _scale,
        _topk_weights,
        current_topk_ids,
        *_args,
        use_swigelu,
        **_kwargs,
    ):
        nonlocal second_gemm_calls
        tokens = current_topk_ids.shape[0]
        if use_swigelu:
            output[: tokens * top_k].fill_(1)
        else:
            fill_value = 11.0 if second_gemm_calls == 0 else 23.0
            output[:tokens].fill_(fill_value)
            second_gemm_calls += 1

    def fake_moe_sum(intermediate_cache3, output):
        required_shape = (
            intermediate_cache3.shape[0],
            intermediate_cache3.shape[-1],
        )
        if tuple(output.shape) != required_shape:
            output.resize_(required_shape)
        output.copy_(intermediate_cache3.sum(dim=1))

    monkeypatch.setattr(
        fused_moe.musa_ops, "musa_fused_gemv_moe", fake_musa_fused_gemv_moe
    )
    monkeypatch.setattr(fused_moe.ops, "moe_sum", fake_moe_sum)

    result = fused_moe.fused_experts_impl(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )

    assert result.shape == (num_tokens, hidden_size)
    assert torch.all(result[:chunk_size] == 11.0)
    assert torch.all(result[chunk_size:] == 23.0)


def _deepgemm_gate_kwargs(torch_module):
    return {
        "hidden_states": torch_module.empty(
            (4100, 4096), device="meta", dtype=torch_module.bfloat16
        ),
        "w1": torch_module.empty(
            (256, 512, 4096),
            device="meta",
            dtype=torch_module.float8_e4m3fn,
        ),
        "w2": torch_module.empty(
            (256, 4096, 256),
            device="meta",
            dtype=torch_module.float8_e4m3fn,
        ),
        "topk_ids": torch_module.empty(
            (4100, 6), device="meta", dtype=torch_module.int32
        ),
        "activation": "silu",
        "apply_router_weight_on_input": False,
        "use_fp8_w8a8": True,
        "use_int8_w8a8": False,
        "use_int8_w8a16": False,
        "use_int4_w4a16": False,
        "ocp_mx_scheme": None,
        "per_channel_quant": False,
        "expert_map": None,
        "w1_scale": torch_module.empty(
            (256, 4, 32), device="meta", dtype=torch_module.float32
        ),
        "w2_scale": torch_module.empty(
            (256, 32, 2), device="meta", dtype=torch_module.float32
        ),
        "a1_scale": None,
        "a2_scale": None,
        "block_shape": [128, 128],
        "w1_bias": None,
        "w2_bias": None,
    }


def test_deepseek_v4_deepgemm_prefill_gate_is_shape_gated():
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    kwargs = _deepgemm_gate_kwargs(torch)

    assert fused_moe._can_use_deepseek_v4_moe_deepgemm_prefill(**kwargs)


def test_deepseek_v4_deepgemm_prefill_gate_rejects_nonmatching_shape():
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    kwargs = _deepgemm_gate_kwargs(torch)
    kwargs["topk_ids"] = torch.empty((4100, 2), device="meta", dtype=torch.int32)

    assert not fused_moe._can_use_deepseek_v4_moe_deepgemm_prefill(**kwargs)


def test_deepseek_v4_deepgemm_prefill_gate_rejects_nonmatching_dtype():
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    kwargs = _deepgemm_gate_kwargs(torch)
    kwargs["w1_scale"] = torch.empty((256, 4, 32), device="meta", dtype=torch.bfloat16)
    assert not fused_moe._can_use_deepseek_v4_moe_deepgemm_prefill(**kwargs)

    kwargs = _deepgemm_gate_kwargs(torch)
    kwargs["w2_scale"] = torch.empty((256, 32, 2), device="meta", dtype=torch.bfloat16)
    assert not fused_moe._can_use_deepseek_v4_moe_deepgemm_prefill(**kwargs)

    kwargs = _deepgemm_gate_kwargs(torch)
    kwargs["topk_ids"] = torch.empty((4100, 6), device="meta", dtype=torch.int64)
    assert not fused_moe._can_use_deepseek_v4_moe_deepgemm_prefill(**kwargs)
