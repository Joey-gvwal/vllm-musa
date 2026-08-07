from types import SimpleNamespace

from vllm_musa.optimization_contract import (
    OptimizationFeature,
    resolve_optimization_contract,
)


def _qwen_config(*, is_moe: bool) -> SimpleNamespace:
    architecture = (
        "Qwen3_5MoeForConditionalGeneration"
        if is_moe
        else "Qwen3_5ForConditionalGeneration"
    )
    text_config = SimpleNamespace(
        architectures=[architecture],
        model_type="qwen3_5_moe_text" if is_moe else "qwen3_5_text",
        hidden_size=2048,
        num_experts=256 if is_moe else None,
        num_experts_per_tok=8 if is_moe else None,
        moe_intermediate_size=512 if is_moe else None,
        linear_conv_kernel_dim=4,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=192,
    )
    model_config = SimpleNamespace(
        architectures=[architecture],
        model_type=text_config.model_type,
        hf_text_config=text_config,
        hf_config=SimpleNamespace(
            architectures=[architecture],
            model_type=text_config.model_type,
            text_config=text_config,
        ),
        dtype="bfloat16",
        quantization=None,
        enforce_eager=False,
        is_model_moe=lambda: is_moe,
        is_hybrid=lambda: True,
    )
    return SimpleNamespace(
        model_config=model_config,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=64),
        scheduler_config=SimpleNamespace(max_num_seqs=64),
        speculative_config=None,
        quant_config=None,
        attention_config=SimpleNamespace(backend="FLASH_ATTN"),
        compilation_config=SimpleNamespace(
            mode="NONE",
            cudagraph_mode="FULL_DECODE_ONLY",
        ),
    )


def test_qwen36_moe_contract_restores_pr171_features() -> None:
    contract = resolve_optimization_contract(_qwen_config(is_moe=True))

    assert contract.prefers(OptimizationFeature.HYBRID_SEPARATE_MAMBA_POOL)
    assert contract.prefers(OptimizationFeature.QWEN35_SHARED_EXPERT_FOLD)
    assert not contract.prefers(OptimizationFeature.QWEN35_GDN_WIDTH4_PREFILL)


def test_qwen36_dense_contract_keeps_width4_prefill_gate() -> None:
    contract = resolve_optimization_contract(_qwen_config(is_moe=False))

    assert contract.prefers(OptimizationFeature.HYBRID_SEPARATE_MAMBA_POOL)
    assert contract.prefers(OptimizationFeature.QWEN35_GDN_WIDTH4_PREFILL)
    assert not contract.prefers(OptimizationFeature.QWEN35_SHARED_EXPERT_FOLD)
