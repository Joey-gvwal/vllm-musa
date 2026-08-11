# SPDX-License-Identifier: Apache-2.0
"""Narrow runtime policy for validated DeepSeek-V4 MTP fast paths.

The v0.26 branch intentionally does not carry the model-wide optimization
contract introduced on v0.24.  Keep the PR #174 MTP policy local to its
consumers and fail closed unless the complete DeepSeek-V4-Flash-Base TP8
execution shape is present.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_DEEPSEEK_V4_ARCHITECTURES = ("DeepseekV4ForCausalLM",)
_DEEPSEEK_V4_MTP_ARCHITECTURES = ("DeepSeekV4MTPModel",)
_DEEPSEEK_V4_QUANTIZATION = frozenset({"fp8", "deepseek_v4_fp8"})
_DEEPSEEK_V4_SPARSE_PADDED_HEADS = 64
_DEEPSEEK_V4_SPARSE_HEAD_DIM = 512
_DEEPSEEK_V4_SPARSE_DTYPE_BYTES = 2
_DEEPSEEK_V4_CUSTOM_AR_ALLOCATOR_MARGIN_BYTES = 512 * 1024 * 1024
_DEEPSEEK_V4_MTP4_TOKENS_PER_REQUEST = 5
_DEEPSEEK_V4_MTP_CAR_GRAPH_REQUEST_SIZES = (1, 2, 4, 8, 16)
_DEEPSEEK_V4_MTP_CAR_GRAPH_BUFFER_BYTES = 512 * 1024 * 1024
_DEEPSEEK_V4_MTP_CAR_EAGER_ALIGNMENT_BYTES = 4 * 1024 * 1024
_DEEPSEEK_V4_MTP_CAR_MAX_META_BYTES_PER_SLOT = 16 * 1024


def _normalize(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).lower().split(".")[-1]


def _text_config(model_config: Any) -> Any:
    text_config = getattr(model_config, "hf_text_config", None)
    if text_config is not None:
        return text_config
    hf_config = getattr(model_config, "hf_config", None)
    return getattr(hf_config, "text_config", hf_config)


def _architectures(model_config: Any, text_config: Any) -> tuple[str, ...]:
    values = getattr(model_config, "architectures", None)
    if not values:
        hf_config = getattr(model_config, "hf_config", None)
        values = getattr(hf_config, "architectures", None)
    if not values:
        values = getattr(text_config, "architectures", None)
    return tuple(str(value) for value in values or ())


def _model_type(model_config: Any, text_config: Any) -> str | None:
    value = getattr(model_config, "model_type", None)
    if value is None:
        hf_config = getattr(model_config, "hf_config", None)
        value = getattr(hf_config, "model_type", None)
    if value is None:
        value = getattr(text_config, "model_type", None)
    return _normalize(value)


def _quantization(model_config: Any, text_config: Any) -> str | None:
    value = getattr(model_config, "quantization", None)
    if value is not None:
        return _normalize(value)
    quantization_config = getattr(text_config, "quantization_config", None)
    if isinstance(quantization_config, dict):
        return _normalize(quantization_config.get("quant_method"))
    return None


def _quant_block_shape(vllm_config: Any, text_config: Any) -> tuple[int, ...] | None:
    quant_config = getattr(vllm_config, "quant_config", None)
    value = getattr(quant_config, "weight_block_size", None)
    if value is None:
        quantization_config = getattr(text_config, "quantization_config", None)
        if isinstance(quantization_config, dict):
            value = quantization_config.get("weight_block_size") or (
                quantization_config.get("weight_block_shape")
            )
    if not isinstance(value, (list, tuple)) or not value:
        return None
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        return None
    return tuple(value)


def _has_routed_experts(model_config: Any, text_config: Any) -> bool:
    is_model_moe = getattr(model_config, "is_model_moe", None)
    if callable(is_model_moe):
        try:
            return bool(is_model_moe())
        except (AttributeError, TypeError, ValueError):
            return False
    is_moe = getattr(model_config, "is_moe", None)
    if is_moe is not None:
        return bool(is_moe)
    return any(
        bool(getattr(text_config, name, 0))
        for name in (
            "num_experts",
            "moe_num_experts",
            "n_routed_experts",
            "num_local_experts",
        )
    )


def _uses_mla(model_config: Any, text_config: Any) -> bool:
    value = getattr(model_config, "use_mla", None)
    if not isinstance(value, bool):
        value = getattr(text_config, "use_mla", None)
    if not isinstance(value, bool):
        value = getattr(text_config, "kv_lora_rank", None) is not None
    return value


def _is_hybrid(model_config: Any) -> bool | None:
    value = getattr(model_config, "is_hybrid", None)
    if callable(value):
        try:
            value = value()
        except (AttributeError, TypeError, ValueError):
            return None
    return value if isinstance(value, bool) else None


def _matches_flash_model(vllm_config: Any, *, mtp_draft: bool) -> bool:
    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None:
        return False
    text_config = _text_config(model_config)
    if text_config is None:
        return False
    expected_architectures = (
        _DEEPSEEK_V4_MTP_ARCHITECTURES if mtp_draft else _DEEPSEEK_V4_ARCHITECTURES
    )
    expected_model_type = "deepseek_mtp" if mtp_draft else "deepseek_v4"
    return (
        _architectures(model_config, text_config) == expected_architectures
        and _model_type(model_config, text_config) == expected_model_type
        and _normalize(getattr(model_config, "dtype", None)) == "bfloat16"
        and _quantization(model_config, text_config) in _DEEPSEEK_V4_QUANTIZATION
        and getattr(text_config, "hidden_size", None) == 4096
        and getattr(text_config, "num_hidden_layers", None) == 43
        and getattr(text_config, "num_attention_heads", None) == 64
        and getattr(text_config, "num_key_value_heads", None) == 1
        and getattr(text_config, "head_dim", None) == 512
        and getattr(text_config, "vocab_size", None) == 129280
        and getattr(text_config, "n_routed_experts", None) == 256
        and getattr(text_config, "num_experts_per_tok", None) == 6
        and getattr(text_config, "n_shared_experts", None) == 1
        and getattr(text_config, "moe_intermediate_size", None) == 2048
        and _normalize(getattr(text_config, "expert_dtype", None)) == "fp8"
        and _normalize(getattr(text_config, "hidden_act", None)) == "silu"
        and getattr(text_config, "swiglu_limit", None) == 10.0
        and _quant_block_shape(vllm_config, text_config) == (128, 128)
        and _has_routed_experts(model_config, text_config)
        and _uses_mla(model_config, text_config)
        and getattr(text_config, "index_topk", None) == 512
        and _is_hybrid(model_config) is False
    )


def _matches_tp8_execution(
    vllm_config: Any,
    *,
    attention_backend_hint: str | None = None,
) -> bool:
    parallel_config = getattr(vllm_config, "parallel_config", None)
    cache_config = getattr(vllm_config, "cache_config", None)
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    attention_config = getattr(vllm_config, "attention_config", None)
    compilation_config = getattr(vllm_config, "compilation_config", None)
    backend = _normalize(getattr(attention_config, "backend", None))
    if backend is None:
        backend = _normalize(attention_backend_hint)
    return (
        parallel_config is not None
        and getattr(parallel_config, "tensor_parallel_size", 1) == 8
        and getattr(parallel_config, "pipeline_parallel_size", 1) == 1
        and getattr(parallel_config, "data_parallel_size", 1) == 1
        and getattr(parallel_config, "decode_context_parallel_size", 1) == 1
        and getattr(vllm_config, "quant_config", None) is not None
        and _normalize(getattr(cache_config, "cache_dtype", "auto"))
        in {"fp8", "fp8_ds_mla"}
        and isinstance(getattr(scheduler_config, "max_num_seqs", None), int)
        and not isinstance(getattr(scheduler_config, "max_num_seqs", None), bool)
        and getattr(scheduler_config, "max_num_seqs") > 0
        and backend == "flashmla"
        and _normalize(getattr(compilation_config, "mode", None)) == "none"
        and _normalize(getattr(compilation_config, "cudagraph_mode", None))
        == "full_decode_only"
    )


def _matches_target_mtp(
    vllm_config: Any,
    *,
    allow_bs1: bool,
    attention_backend_hint: str | None = None,
) -> bool:
    speculative_config = getattr(vllm_config, "speculative_config", None)
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    max_num_seqs = getattr(scheduler_config, "max_num_seqs", 0)
    return (
        _matches_flash_model(vllm_config, mtp_draft=False)
        and _matches_tp8_execution(
            vllm_config,
            attention_backend_hint=attention_backend_hint,
        )
        and speculative_config is not None
        and _normalize(getattr(speculative_config, "method", None))
        in {"mtp", "deepseek_mtp"}
        and (allow_bs1 or max_num_seqs > 1)
    )


def _matches_mtp_draft(
    vllm_config: Any,
    *,
    attention_backend_hint: str | None = None,
) -> bool:
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    return (
        _matches_flash_model(vllm_config, mtp_draft=True)
        and _matches_tp8_execution(
            vllm_config,
            attention_backend_hint=attention_backend_hint,
        )
        and getattr(vllm_config, "speculative_config", None) is None
        and getattr(scheduler_config, "max_num_seqs", 0) > 1
    )


def deepseek_v4_mtp_sparse_direct_out_enabled(
    vllm_config: Any,
    *,
    attention_backend_hint: str | None = None,
) -> bool:
    return _matches_target_mtp(
        vllm_config,
        allow_bs1=False,
        attention_backend_hint=attention_backend_hint,
    ) or _matches_mtp_draft(
        vllm_config,
        attention_backend_hint=attention_backend_hint,
    )


def deepseek_v4_mtp_car_graph_guard_enabled(vllm_config: Any) -> bool:
    return _matches_target_mtp(vllm_config, allow_bs1=True)


def deepseek_v4_mtp_graph_registered_inputs_enabled(vllm_config: Any) -> bool:
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    return deepseek_v4_mtp_car_graph_guard_enabled(vllm_config) and (
        getattr(scheduler_config, "max_num_seqs", None) == 1
    )


@dataclass(frozen=True, slots=True)
class DeepSeekV4MtpCarGraphStagingPlan:
    """Capture-time resource contract for DSV4 MTP4 JIT CAR graphs."""

    eager_reserve_bytes: int
    capture_descriptors: frozenset[tuple[int, int]]
    car_ops_per_descriptor: int
    bytes_per_token: int
    graph_data_capacity_bytes: int
    graph_meta_capacity_bytes: int
    max_meta_bytes_per_slot: int
    communicator_buffer_bytes: int

    def allows_descriptor(self, descriptor: Any) -> bool:
        if descriptor is None or not bool(getattr(descriptor, "uniform", False)):
            return False
        if bool(getattr(descriptor, "has_lora", False)) or int(
            getattr(descriptor, "num_active_loras", 0) or 0
        ):
            return False
        num_tokens = getattr(descriptor, "num_tokens", None)
        num_reqs = getattr(descriptor, "num_reqs", None)
        if (
            not isinstance(num_tokens, int)
            or isinstance(num_tokens, bool)
            or not isinstance(num_reqs, int)
            or isinstance(num_reqs, bool)
        ):
            return False
        return (num_tokens, num_reqs) in self.capture_descriptors

    def expected_descriptor_data_bytes(self, num_tokens: int) -> int:
        return self.car_ops_per_descriptor * num_tokens * self.bytes_per_token


def deepseek_v4_mtp_car_graph_staging_plan(
    vllm_config: Any,
) -> DeepSeekV4MtpCarGraphStagingPlan | None:
    if not deepseek_v4_mtp_car_graph_guard_enabled(vllm_config):
        return None
    speculative_config = getattr(vllm_config, "speculative_config", None)
    if (
        getattr(speculative_config, "num_speculative_tokens", None)
        != _DEEPSEEK_V4_MTP4_TOKENS_PER_REQUEST - 1
    ):
        return None
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    model_config = getattr(vllm_config, "model_config", None)
    text_config = _text_config(model_config)
    max_num_batched_tokens = int(
        getattr(scheduler_config, "max_num_batched_tokens", 0) or 0
    )
    hidden_size = int(getattr(text_config, "hidden_size", 0) or 0)
    num_hidden_layers = int(getattr(text_config, "num_hidden_layers", 0) or 0)
    if max_num_batched_tokens <= 0 or hidden_size <= 0 or num_hidden_layers <= 0:
        return None
    required_eager_bytes = max_num_batched_tokens * hidden_size * 2
    alignment = _DEEPSEEK_V4_MTP_CAR_EAGER_ALIGNMENT_BYTES
    eager_reserve_bytes = (
        (required_eager_bytes + alignment - 1) // alignment * alignment
    )
    capture_descriptors = frozenset(
        (_DEEPSEEK_V4_MTP4_TOKENS_PER_REQUEST * num_reqs, num_reqs)
        for num_reqs in _DEEPSEEK_V4_MTP_CAR_GRAPH_REQUEST_SIZES
    )
    car_ops_per_descriptor = 2 * num_hidden_layers + 1
    bytes_per_token = hidden_size * 2
    graph_slot_count = car_ops_per_descriptor * len(capture_descriptors)
    graph_data_capacity_bytes = (
        car_ops_per_descriptor
        * sum(num_tokens for num_tokens, _ in capture_descriptors)
        * bytes_per_token
    )
    graph_meta_capacity_bytes = (
        graph_data_capacity_bytes
        + graph_slot_count * _DEEPSEEK_V4_MTP_CAR_MAX_META_BYTES_PER_SLOT
    )
    required_communicator_buffer_bytes = max(
        eager_reserve_bytes + graph_data_capacity_bytes,
        eager_reserve_bytes
        + _DEEPSEEK_V4_MTP_CAR_MAX_META_BYTES_PER_SLOT
        + graph_meta_capacity_bytes,
    )
    communicator_buffer_bytes = (
        (
            required_communicator_buffer_bytes
            + _DEEPSEEK_V4_MTP_CAR_EAGER_ALIGNMENT_BYTES
            - 1
        )
        // _DEEPSEEK_V4_MTP_CAR_EAGER_ALIGNMENT_BYTES
        * _DEEPSEEK_V4_MTP_CAR_EAGER_ALIGNMENT_BYTES
    )
    if communicator_buffer_bytes > _DEEPSEEK_V4_MTP_CAR_GRAPH_BUFFER_BYTES:
        return None
    return DeepSeekV4MtpCarGraphStagingPlan(
        eager_reserve_bytes=eager_reserve_bytes,
        capture_descriptors=capture_descriptors,
        car_ops_per_descriptor=car_ops_per_descriptor,
        bytes_per_token=bytes_per_token,
        graph_data_capacity_bytes=graph_data_capacity_bytes,
        graph_meta_capacity_bytes=graph_meta_capacity_bytes,
        max_meta_bytes_per_slot=_DEEPSEEK_V4_MTP_CAR_MAX_META_BYTES_PER_SLOT,
        communicator_buffer_bytes=communicator_buffer_bytes,
    )


def deepseek_v4_mtp_async_prefill_queue_fence_enabled(vllm_config: Any) -> bool:
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    return _matches_target_mtp(vllm_config, allow_bs1=False) and bool(
        getattr(scheduler_config, "async_scheduling", False)
    )


def deepseek_v4_mtp_prefill_step_requires_sync(scheduler_output: Any) -> bool:
    if getattr(scheduler_output, "scheduled_new_reqs", ()):
        return True
    cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
    return 0 in getattr(cached_reqs, "num_output_tokens", ())


def deepseek_v4_mtp_sparse_prefill_headroom_bytes(vllm_config: Any) -> int:
    if not deepseek_v4_mtp_sparse_direct_out_enabled(vllm_config):
        return 0
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    max_num_batched_tokens = int(
        getattr(scheduler_config, "max_num_batched_tokens", 0) or 0
    )
    if max_num_batched_tokens <= 0:
        return 0
    workspace_bytes = (
        max_num_batched_tokens
        * _DEEPSEEK_V4_SPARSE_PADDED_HEADS
        * _DEEPSEEK_V4_SPARSE_HEAD_DIM
        * _DEEPSEEK_V4_SPARSE_DTYPE_BYTES
    )
    parallel_config = getattr(vllm_config, "parallel_config", None)
    if not bool(getattr(parallel_config, "disable_custom_all_reduce", False)):
        workspace_bytes += _DEEPSEEK_V4_CUSTOM_AR_ALLOCATOR_MARGIN_BYTES
    return workspace_bytes
