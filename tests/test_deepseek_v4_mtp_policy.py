from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_musa.deepseek_v4_policy import (
    deepseek_v4_mtp_async_prefill_queue_fence_enabled,
    deepseek_v4_mtp_car_graph_guard_enabled,
    deepseek_v4_mtp_car_graph_staging_plan,
    deepseek_v4_mtp_graph_registered_inputs_enabled,
    deepseek_v4_mtp_prefill_step_requires_sync,
    deepseek_v4_mtp_sparse_direct_out_enabled,
    deepseek_v4_mtp_sparse_prefill_headroom_bytes,
)


def _config(
    *,
    tp: int = 8,
    max_num_seqs: int = 64,
    speculative_tokens: int = 4,
    mtp_draft: bool = False,
) -> SimpleNamespace:
    architectures = (
        ("DeepSeekV4MTPModel",)
        if mtp_draft
        else ("DeepseekV4ForCausalLM",)
    )
    model_type = "deepseek_mtp" if mtp_draft else "deepseek_v4"
    text_config = SimpleNamespace(
        architectures=architectures,
        model_type=model_type,
        hidden_size=4096,
        num_hidden_layers=43,
        num_attention_heads=64,
        num_key_value_heads=1,
        head_dim=512,
        vocab_size=129280,
        n_routed_experts=256,
        num_experts_per_tok=6,
        n_shared_experts=1,
        moe_intermediate_size=2048,
        expert_dtype="fp8",
        hidden_act="silu",
        swiglu_limit=10.0,
        index_topk=512,
        quantization_config={
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        },
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            architectures=architectures,
            model_type=model_type,
            hf_config=text_config,
            hf_text_config=text_config,
            dtype="bfloat16",
            quantization="deepseek_v4_fp8",
            use_mla=True,
            is_hybrid=False,
            is_moe=True,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
            disable_custom_all_reduce=False,
        ),
        cache_config=SimpleNamespace(cache_dtype="fp8", block_size=64),
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=8195,
            async_scheduling=True,
        ),
        attention_config=SimpleNamespace(backend="FLASHMLA"),
        compilation_config=SimpleNamespace(
            mode="NONE",
            cudagraph_mode="FULL_DECODE_ONLY",
        ),
        speculative_config=(
            None
            if mtp_draft
            else SimpleNamespace(
                method="mtp",
                num_speculative_tokens=speculative_tokens,
            )
        ),
        quant_config=SimpleNamespace(weight_block_size=[128, 128]),
    )


def test_multibatch_target_enables_mtp_policies() -> None:
    config = _config()

    assert deepseek_v4_mtp_car_graph_guard_enabled(config)
    assert not deepseek_v4_mtp_graph_registered_inputs_enabled(config)
    assert deepseek_v4_mtp_sparse_direct_out_enabled(config)
    assert deepseek_v4_mtp_async_prefill_queue_fence_enabled(config)
    assert deepseek_v4_mtp_sparse_prefill_headroom_bytes(config) > 512 * 1024 * 1024

    plan = deepseek_v4_mtp_car_graph_staging_plan(config)
    assert plan is not None
    assert plan.capture_descriptors == frozenset(
        {(5, 1), (10, 2), (20, 4), (40, 8), (80, 16)}
    )
    assert plan.car_ops_per_descriptor == 87
    assert plan.communicator_buffer_bytes <= 512 * 1024 * 1024


def test_bs1_uses_registered_inputs_but_not_prefill_fast_paths() -> None:
    config = _config(max_num_seqs=1)

    assert deepseek_v4_mtp_car_graph_guard_enabled(config)
    assert deepseek_v4_mtp_graph_registered_inputs_enabled(config)
    assert not deepseek_v4_mtp_sparse_direct_out_enabled(config)
    assert not deepseek_v4_mtp_async_prefill_queue_fence_enabled(config)
    assert deepseek_v4_mtp_sparse_prefill_headroom_bytes(config) == 0


def test_mtp_draft_uses_sparse_prefill_policy_without_outer_spec_config() -> None:
    config = _config(mtp_draft=True)

    assert deepseek_v4_mtp_sparse_direct_out_enabled(
        config,
        attention_backend_hint="flashmla",
    )
    assert not deepseek_v4_mtp_car_graph_guard_enabled(config)


@pytest.mark.parametrize(
    ("mutation", "expected_guard"),
    [
        (lambda config: setattr(config.parallel_config, "tensor_parallel_size", 4), False),
        (lambda config: setattr(config.model_config.hf_text_config, "hidden_size", 7168), False),
        (lambda config: setattr(config.cache_config, "cache_dtype", "auto"), False),
        (lambda config: setattr(config.attention_config, "backend", "FLASH_ATTN"), False),
    ],
)
def test_policy_fails_closed_outside_validated_shape(mutation, expected_guard) -> None:
    config = _config()
    mutation(config)

    assert deepseek_v4_mtp_car_graph_guard_enabled(config) is expected_guard
    assert not deepseek_v4_mtp_sparse_direct_out_enabled(config)
    assert deepseek_v4_mtp_sparse_prefill_headroom_bytes(config) == 0


def test_non_mtp4_keeps_guard_but_disables_staging_plan() -> None:
    config = _config(speculative_tokens=3)

    assert deepseek_v4_mtp_car_graph_guard_enabled(config)
    assert deepseek_v4_mtp_car_graph_staging_plan(config) is None


def test_prefill_queue_fence_uses_scheduler_context_facts() -> None:
    assert deepseek_v4_mtp_prefill_step_requires_sync(
        SimpleNamespace(scheduled_new_reqs=(object(),))
    )
    assert deepseek_v4_mtp_prefill_step_requires_sync(
        SimpleNamespace(
            scheduled_new_reqs=(),
            scheduled_cached_reqs=SimpleNamespace(num_output_tokens=(1, 0, 1)),
        )
    )
    assert not deepseek_v4_mtp_prefill_step_requires_sync(
        SimpleNamespace(
            scheduled_new_reqs=(),
            scheduled_cached_reqs=SimpleNamespace(num_output_tokens=(1, 1)),
        )
    )
