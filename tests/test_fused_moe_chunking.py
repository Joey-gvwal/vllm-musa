# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for MUSA fused MoE chunked execution."""

import json

import torch

from vllm_musa.model_executor.layers.fused_moe import moe_shape_inventory


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
    monkeypatch.setattr(moe_shape_inventory, "_RECORDS", 0)
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
    monkeypatch.setattr(moe_shape_inventory, "_RECORDS", 0)
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


def _qwen_deepgemm_gate_kwargs(torch_module, tokens=66000):
    return {
        "hidden_states": torch_module.empty(
            (tokens, 2048), device="meta", dtype=torch_module.bfloat16
        ),
        "w1": torch_module.empty(
            (256, 1024, 2048),
            device="meta",
            dtype=torch_module.float8_e4m3fn,
        ),
        "w2": torch_module.empty(
            (256, 2048, 512),
            device="meta",
            dtype=torch_module.float8_e4m3fn,
        ),
        "topk_ids": torch_module.empty(
            (tokens, 8), device="meta", dtype=torch_module.int32
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
            (256, 8, 16), device="meta", dtype=torch_module.float32
        ),
        "w2_scale": torch_module.empty(
            (256, 16, 4), device="meta", dtype=torch_module.float32
        ),
        "a1_scale": None,
        "a2_scale": None,
        "block_shape": [128, 128],
        "w1_bias": None,
        "w2_bias": None,
    }


def _isolate_dispatch(monkeypatch):
    """Clear dispatch envs and point the tuned cache at a missing path."""
    from vllm_musa.model_executor.layers.fused_moe import moe_dispatch

    for name in (
        "VLLM_MUSA_MOE_DEEPGEMM",
        "VLLM_MUSA_MOE_DEEPGEMM_MIN_TOKENS",
        "VLLM_MUSA_DEEPSEEK_V4_MOE_DEEPGEMM_PREFILL",
        "VLLM_MUSA_DEEPSEEK_V4_MOE_DEEPGEMM_PREFILL_MIN_TOKENS",
        "VLLM_MUSA_DEEPSEEK_V4_FUSED_MOE_GEMV",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VLLM_MUSA_MOE_DEEPGEMM_TUNED_CACHE", "/nonexistent/tuned.json")
    moe_dispatch.reset_tuned_cache()
    return moe_dispatch


def test_deepseek_signature_eligible_but_env_gated(monkeypatch):
    md = _isolate_dispatch(monkeypatch)
    inp = md.build_dispatch_input(**_deepgemm_gate_kwargs(torch))

    assert md.grouped_deepgemm_contiguous_eligible(inp)
    # Opt-in: disabled until the profile env is set; crossover 4096.
    assert md.resolve_deepgemm_policy(inp) == (False, 4096)
    assert md.select_backend(inp) is md.MoEBackend.NATIVE_GEMV

    monkeypatch.setenv("VLLM_MUSA_DEEPSEEK_V4_MOE_DEEPGEMM_PREFILL", "1")
    assert md.resolve_deepgemm_policy(inp) == (True, 4096)
    # Fixture has 4100 tokens (>= 4096) -> grouped DeepGEMM.
    assert md.select_backend(inp) is md.MoEBackend.GROUPED_DEEPGEMM_CONTIGUOUS


def test_deepseek_legacy_min_tokens_env(monkeypatch):
    md = _isolate_dispatch(monkeypatch)
    monkeypatch.setenv("VLLM_MUSA_DEEPSEEK_V4_MOE_DEEPGEMM_PREFILL", "1")
    monkeypatch.setenv("VLLM_MUSA_DEEPSEEK_V4_MOE_DEEPGEMM_PREFILL_MIN_TOKENS", "8192")
    inp = md.build_dispatch_input(**_deepgemm_gate_kwargs(torch))
    assert md.resolve_deepgemm_policy(inp) == (True, 8192)
    # 4100 tokens now below the raised threshold -> native GEMV.
    assert md.select_backend(inp) is md.MoEBackend.NATIVE_GEMV


def test_qwen_signature_on_by_default_large_prefill(monkeypatch):
    md = _isolate_dispatch(monkeypatch)
    small = md.build_dispatch_input(**_qwen_deepgemm_gate_kwargs(torch, tokens=20000))
    large = md.build_dispatch_input(**_qwen_deepgemm_gate_kwargs(torch, tokens=66000))

    assert md.grouped_deepgemm_contiguous_eligible(small)
    assert md.resolve_deepgemm_policy(small) == (True, 65536)
    assert md.select_backend(small) is md.MoEBackend.NATIVE_GEMV
    assert md.select_backend(large) is md.MoEBackend.GROUPED_DEEPGEMM_CONTIGUOUS


def test_signed_topk_ids_accepted_for_any_signature(monkeypatch):
    md = _isolate_dispatch(monkeypatch)

    qwen = _qwen_deepgemm_gate_kwargs(torch, tokens=66000)
    qwen["topk_ids"] = torch.empty((66000, 8), device="meta", dtype=torch.int64)
    assert md.grouped_deepgemm_contiguous_eligible(md.build_dispatch_input(**qwen))

    # The unified contract accepts int64 topk for the DeepSeek shape too; the
    # old gate rejected it with a per-model int32-only rule.
    ds = _deepgemm_gate_kwargs(torch)
    ds["topk_ids"] = torch.empty((4100, 6), device="meta", dtype=torch.int64)
    assert md.grouped_deepgemm_contiguous_eligible(md.build_dispatch_input(**ds))


def test_non_block128_shape_not_eligible(monkeypatch):
    md = _isolate_dispatch(monkeypatch)
    kwargs = _qwen_deepgemm_gate_kwargs(torch, tokens=66000)
    kwargs["w1"] = torch.empty((256, 640, 2048), device="meta", dtype=torch.float8_e4m3fn)
    kwargs["w2"] = torch.empty((256, 2048, 320), device="meta", dtype=torch.float8_e4m3fn)
    kwargs["w1_scale"] = torch.empty((256, 5, 16), device="meta", dtype=torch.float32)
    kwargs["w2_scale"] = torch.empty((256, 16, 3), device="meta", dtype=torch.float32)
    inp = md.build_dispatch_input(**kwargs)
    assert not md.grouped_deepgemm_contiguous_eligible(inp)
    assert md.select_backend(inp) is md.MoEBackend.NATIVE_GEMV


def test_nonmatching_scale_or_hidden_dtype_not_eligible(monkeypatch):
    md = _isolate_dispatch(monkeypatch)

    kwargs = _deepgemm_gate_kwargs(torch)
    kwargs["w1_scale"] = torch.empty((256, 4, 32), device="meta", dtype=torch.bfloat16)
    assert not md.grouped_deepgemm_contiguous_eligible(md.build_dispatch_input(**kwargs))

    kwargs = _deepgemm_gate_kwargs(torch)
    kwargs["hidden_states"] = torch.empty(
        (4100, 4096), device="meta", dtype=torch.float16
    )
    assert not md.grouped_deepgemm_contiguous_eligible(md.build_dispatch_input(**kwargs))


def test_unknown_eligible_shape_defaults_to_native(monkeypatch):
    md = _isolate_dispatch(monkeypatch)
    # Block-128 consistent but an untuned signature (128 experts).
    kwargs = _qwen_deepgemm_gate_kwargs(torch, tokens=100000)
    kwargs["w1"] = torch.empty((128, 1024, 2048), device="meta", dtype=torch.float8_e4m3fn)
    kwargs["w2"] = torch.empty((128, 2048, 512), device="meta", dtype=torch.float8_e4m3fn)
    kwargs["w1_scale"] = torch.empty((128, 8, 16), device="meta", dtype=torch.float32)
    kwargs["w2_scale"] = torch.empty((128, 16, 4), device="meta", dtype=torch.float32)
    kwargs["topk_ids"] = torch.empty((100000, 8), device="meta", dtype=torch.int32)
    inp = md.build_dispatch_input(**kwargs)

    assert md.grouped_deepgemm_contiguous_eligible(inp)
    assert md.resolve_deepgemm_policy(inp)[0] is False
    assert md.select_backend(inp) is md.MoEBackend.NATIVE_GEMV

    monkeypatch.setenv("VLLM_MUSA_MOE_DEEPGEMM", "1")
    assert md.select_backend(inp) is md.MoEBackend.GROUPED_DEEPGEMM_CONTIGUOUS


def test_global_min_tokens_override(monkeypatch):
    md = _isolate_dispatch(monkeypatch)
    monkeypatch.setenv("VLLM_MUSA_MOE_DEEPGEMM_MIN_TOKENS", "1000")
    inp = md.build_dispatch_input(**_qwen_deepgemm_gate_kwargs(torch, tokens=2000))
    assert md.resolve_deepgemm_policy(inp) == (True, 1000)
    assert md.select_backend(inp) is md.MoEBackend.GROUPED_DEEPGEMM_CONTIGUOUS


def test_expert_parallel_not_eligible_or_routed(monkeypatch):
    md = _isolate_dispatch(monkeypatch)
    kwargs = _qwen_deepgemm_gate_kwargs(torch, tokens=66000)
    kwargs["expert_map"] = torch.empty((256,), device="meta", dtype=torch.int32)
    inp = md.build_dispatch_input(**kwargs)
    assert not md.grouped_deepgemm_contiguous_eligible(inp)
    assert md.should_route_to_musa_impl(inp) is False


def test_grouped_eligible_shape_routes_to_musa_impl(monkeypatch):
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    _isolate_dispatch(monkeypatch)
    kwargs = _qwen_deepgemm_gate_kwargs(torch, tokens=20000)
    topk_weights = torch.empty_like(kwargs["topk_ids"], dtype=torch.bfloat16)
    marker = object()

    monkeypatch.setattr(fused_moe, "fused_experts_impl", lambda *a, **kw: marker)
    monkeypatch.setattr(
        fused_moe._upstream_fused_moe,
        "_musa_original_fused_experts_impl",
        lambda *a, **kw: object(),
    )

    result = fused_moe._musa_fused_experts_impl_dispatch(
        hidden_states=kwargs["hidden_states"],
        w1=kwargs["w1"],
        w2=kwargs["w2"],
        topk_weights=topk_weights,
        topk_ids=kwargs["topk_ids"],
        activation=kwargs["activation"],
        apply_router_weight_on_input=kwargs["apply_router_weight_on_input"],
        use_fp8_w8a8=kwargs["use_fp8_w8a8"],
        use_int8_w8a8=kwargs["use_int8_w8a8"],
        use_int8_w8a16=kwargs["use_int8_w8a16"],
        use_int4_w4a16=kwargs["use_int4_w4a16"],
        ocp_mx_scheme=kwargs["ocp_mx_scheme"],
        per_channel_quant=kwargs["per_channel_quant"],
        expert_map=kwargs["expert_map"],
        w1_scale=kwargs["w1_scale"],
        w2_scale=kwargs["w2_scale"],
        a1_scale=kwargs["a1_scale"],
        a2_scale=kwargs["a2_scale"],
        block_shape=kwargs["block_shape"],
        w1_bias=kwargs["w1_bias"],
        w2_bias=kwargs["w2_bias"],
    )
    assert result is marker


def test_non_musa_shape_falls_through_to_upstream(monkeypatch):
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    _isolate_dispatch(monkeypatch)
    marker = object()
    monkeypatch.setattr(fused_moe, "fused_experts_impl", lambda *a, **kw: object())
    monkeypatch.setattr(
        fused_moe._upstream_fused_moe,
        "_musa_original_fused_experts_impl",
        lambda *a, **kw: marker,
    )
    # bf16 (non-FP8) weights are not grouped-eligible -> upstream path.
    result = fused_moe._musa_fused_experts_impl_dispatch(
        hidden_states=torch.zeros(8, 16, dtype=torch.bfloat16),
        w1=torch.zeros(2, 32, 16, dtype=torch.bfloat16),
        w2=torch.zeros(2, 16, 16, dtype=torch.bfloat16),
        topk_weights=torch.ones(8, 2, dtype=torch.bfloat16),
        topk_ids=torch.zeros(8, 2, dtype=torch.int32),
    )
    assert result is marker


def test_qwen_tp4_sharded_shape_engages_grouped(monkeypatch):
    md = _isolate_dispatch(monkeypatch)
    # TP4 shards the MoE intermediate: per-rank w1 N = 2*(512/4) = 256, so the
    # policy must key on (hidden, experts, top_k), not the per-rank width.
    kwargs = _qwen_deepgemm_gate_kwargs(torch, tokens=66000)
    kwargs["w1"] = torch.empty((256, 256, 2048), device="meta", dtype=torch.float8_e4m3fn)
    kwargs["w2"] = torch.empty((256, 2048, 128), device="meta", dtype=torch.float8_e4m3fn)
    kwargs["w1_scale"] = torch.empty((256, 2, 16), device="meta", dtype=torch.float32)
    kwargs["w2_scale"] = torch.empty((256, 16, 1), device="meta", dtype=torch.float32)
    inp = md.build_dispatch_input(**kwargs)
    assert md.grouped_deepgemm_contiguous_eligible(inp)
    assert md.resolve_deepgemm_policy(inp) == (True, 65536)
    assert md.select_backend(inp) is md.MoEBackend.GROUPED_DEEPGEMM_CONTIGUOUS

    small = dict(kwargs)
    small["hidden_states"] = torch.empty((20000, 2048), device="meta", dtype=torch.bfloat16)
    small["topk_ids"] = torch.empty((20000, 8), device="meta", dtype=torch.int32)
    assert md.select_backend(md.build_dispatch_input(**small)) is md.MoEBackend.NATIVE_GEMV
