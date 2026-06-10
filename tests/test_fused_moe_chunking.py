# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for MUSA fused MoE chunked execution."""

import json

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


def test_musa_fused_moe_shape_inventory_is_disabled_by_default(monkeypatch, tmp_path):
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    top_k = 1
    output_path = tmp_path / "inventory.jsonl"

    monkeypatch.delenv("VLLM_MUSA_DEEPSEEK_V4_MOE_SHAPE_INVENTORY", raising=False)
    monkeypatch.setenv(
        "VLLM_MUSA_DEEPSEEK_V4_MOE_SHAPE_INVENTORY_PATH", str(output_path)
    )
    monkeypatch.setenv("VLLM_MUSA_DEEPSEEK_V4_MOE_SHAPE_INVENTORY_MIN_TOKENS", "1")
    monkeypatch.setattr(fused_moe, "_MOE_SHAPE_INVENTORY_RECORDS", 0)
    _patch_fake_moe_kernels(monkeypatch, fused_moe, top_k)

    fused_moe.fused_experts_impl(
        hidden_states=torch.zeros(2, 4),
        w1=torch.zeros(2, 8, 4),
        w2=torch.zeros(2, 4, 4),
        topk_weights=torch.ones(2, top_k),
        topk_ids=torch.zeros(2, top_k, dtype=torch.int64),
    )

    assert not output_path.exists()


def test_musa_fused_moe_shape_inventory_records_bridge_contract(monkeypatch, tmp_path):
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    top_k = 2
    output_path = tmp_path / "inventory.jsonl"

    monkeypatch.setenv("VLLM_MUSA_DEEPSEEK_V4_MOE_SHAPE_INVENTORY", "1")
    monkeypatch.setenv(
        "VLLM_MUSA_DEEPSEEK_V4_MOE_SHAPE_INVENTORY_PATH", str(output_path)
    )
    monkeypatch.setenv("VLLM_MUSA_DEEPSEEK_V4_MOE_SHAPE_INVENTORY_MIN_TOKENS", "1")
    monkeypatch.setattr(fused_moe, "_MOE_SHAPE_INVENTORY_RECORDS", 0)
    _patch_fake_moe_kernels(monkeypatch, fused_moe, top_k)

    result = fused_moe.fused_experts_impl(
        hidden_states=torch.zeros(3, 4),
        w1=torch.zeros(3, 8, 4),
        w2=torch.zeros(3, 4, 4),
        topk_weights=torch.ones(3, top_k),
        topk_ids=torch.tensor([[0, 2], [1, 2], [2, 0]], dtype=torch.int64),
        w1_scale=torch.ones(3),
        w2_scale=torch.ones(3),
        block_shape=[128, 128],
        use_fp8_w8a8=True,
        apply_router_weight_on_input=True,
        global_num_experts=3,
    )

    assert result.shape == (3, 4)
    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(records) == 1

    record = records[0]
    assert record["event"] == "deepseek_v4_moe_shape_inventory"
    assert record["num_tokens"] == 3
    assert record["top_k"] == 2
    assert record["num_local_experts"] == 3
    assert record["global_num_experts"] == 3
    assert record["w1"]["shape"] == [3, 8, 4]
    assert record["w2"]["shape"] == [3, 4, 4]
    assert record["topk_ids"]["dtype"] == "torch.int64"
    assert record["w1_scale"]["shape"] == [3]
    assert record["block_shape"] == [128, 128]
    assert record["apply_router_weight_on_input"] is True
    assert record["use_fp8_w8a8"] is True
    assert record["routed_token_stats"]["histogram"] == [2, 1, 3]
    assert record["routed_token_stats"]["slot_histograms"] == [
        [1, 1, 1],
        [1, 0, 2],
    ]


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


def test_deepseek_v4_deepgemm_prefill_gate_is_env_gated(monkeypatch):
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    kwargs = _deepgemm_gate_kwargs(torch)
    monkeypatch.delenv("VLLM_MUSA_DEEPSEEK_V4_MOE_DEEPGEMM_PREFILL", raising=False)

    assert not fused_moe._can_use_deepseek_v4_moe_deepgemm_prefill(**kwargs)

    monkeypatch.setenv("VLLM_MUSA_DEEPSEEK_V4_MOE_DEEPGEMM_PREFILL", "1")

    assert fused_moe._can_use_deepseek_v4_moe_deepgemm_prefill(**kwargs)


def test_deepseek_v4_deepgemm_prefill_gate_rejects_nonmatching_shape(
    monkeypatch,
):
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    kwargs = _deepgemm_gate_kwargs(torch)
    kwargs["topk_ids"] = torch.empty((4100, 2), device="meta", dtype=torch.int32)
    monkeypatch.setenv("VLLM_MUSA_DEEPSEEK_V4_MOE_DEEPGEMM_PREFILL", "1")

    assert not fused_moe._can_use_deepseek_v4_moe_deepgemm_prefill(**kwargs)


def test_deepseek_v4_deepgemm_prefill_gate_rejects_nonmatching_dtype(
    monkeypatch,
):
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    monkeypatch.setenv("VLLM_MUSA_DEEPSEEK_V4_MOE_DEEPGEMM_PREFILL", "1")

    kwargs = _deepgemm_gate_kwargs(torch)
    kwargs["w1_scale"] = torch.empty((256, 4, 32), device="meta", dtype=torch.bfloat16)
    assert not fused_moe._can_use_deepseek_v4_moe_deepgemm_prefill(**kwargs)

    kwargs = _deepgemm_gate_kwargs(torch)
    kwargs["w2_scale"] = torch.empty((256, 32, 2), device="meta", dtype=torch.bfloat16)
    assert not fused_moe._can_use_deepseek_v4_moe_deepgemm_prefill(**kwargs)

    kwargs = _deepgemm_gate_kwargs(torch)
    kwargs["topk_ids"] = torch.empty((4100, 6), device="meta", dtype=torch.int64)
    assert not fused_moe._can_use_deepseek_v4_moe_deepgemm_prefill(**kwargs)
