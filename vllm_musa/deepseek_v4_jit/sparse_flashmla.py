# SPDX-License-Identifier: Apache-2.0
"""Default-off sparse FlashMLA provider adapters for DeepSeek-V4.

This module keeps the current MUSA torch correctness fallback as the default.
It only activates when explicitly requested, and currently targets the SGLang
AMD DPSK-V4 TileLang reference as a source-fit probe.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

import torch

_IMPL_ENV = "VLLM_MUSA_DEEPSEEK_V4_SPARSE_FLASHMLA_IMPL"
_PROVIDER_ENV = "VLLM_MUSA_DEEPSEEK_V4_SPARSE_FLASHMLA_PROVIDER"
_DEFAULT_PROVIDER = "sglang.srt.layers.attention.nsa.tilelang_kernel"
_PROVIDER_FN = "dpsk_v4_fp8_attention_fwd"


def _impl_mode() -> str:
    return os.getenv(_IMPL_ENV, "torch").strip().lower()


def sparse_flashmla_provider_enabled() -> bool:
    return _impl_mode() in {"sglang_tilelang", "tilelang", "external_sglang"}


def _to_int32_contiguous(name: str, tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.dtype != torch.int32:
        raise RuntimeError(
            f"{_IMPL_ENV}=sglang_tilelang requires {name} to be int32 to avoid "
            f"hidden dtype-conversion allocations, got {tensor.dtype}."
        )
    if not tensor.is_contiguous():
        raise RuntimeError(
            f"{_IMPL_ENV}=sglang_tilelang requires {name} to be contiguous to "
            "avoid hidden graph-unsafe copies."
        )
    return tensor


def _load_sglang_tilelang_provider():
    module_name = os.getenv(_PROVIDER_ENV, _DEFAULT_PROVIDER).strip()
    module = importlib.import_module(module_name)
    provider = getattr(module, _PROVIDER_FN, None)
    if provider is None:
        raise RuntimeError(
            f"{_IMPL_ENV}=sglang_tilelang requested {module_name}.{_PROVIDER_FN}, "
            "but the function is not available."
        )
    return provider


def maybe_sglang_tilelang_flash_mla_with_kvcache(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor | None,
    cache_seqlens: torch.Tensor | None,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    extra_k_cache: torch.Tensor | None = None,
    extra_indices_in_kvcache: torch.Tensor | None = None,
    extra_topk_length: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not sparse_flashmla_provider_enabled():
        return None
    if kwargs:
        raise TypeError(
            f"{_IMPL_ENV}=sglang_tilelang does not support kwargs: "
            f"{', '.join(sorted(kwargs))}"
        )
    if q.dim() != 4:
        raise RuntimeError(
            f"{_IMPL_ENV}=sglang_tilelang expects q shape [B, S, H, D], "
            f"got {q.shape}."
        )
    if q.dtype != torch.bfloat16:
        raise RuntimeError(
            f"{_IMPL_ENV}=sglang_tilelang expects bfloat16 q, got {q.dtype}."
        )
    if not is_fp8_kvcache or k_cache.dtype != torch.uint8:
        raise RuntimeError(
            f"{_IMPL_ENV}=sglang_tilelang requires packed fp8_ds_mla uint8 "
            f"k_cache, got is_fp8_kvcache={is_fp8_kvcache}, dtype={k_cache.dtype}."
        )
    if k_cache.dim() != 4 or k_cache.shape[2] != 1:
        raise RuntimeError(
            f"{_IMPL_ENV}=sglang_tilelang expects k_cache shape "
            f"[blocks, block, 1, bytes], got {k_cache.shape}."
        )
    if block_table is not None or cache_seqlens is not None:
        raise RuntimeError(
            f"{_IMPL_ENV}=sglang_tilelang only supports the DeepSeek-V4 sparse "
            "decode contract with block_table=None and cache_seqlens=None."
        )
    if causal:
        raise RuntimeError(f"{_IMPL_ENV}=sglang_tilelang does not support causal=True.")
    if head_dim_v != 512:
        raise RuntimeError(
            f"{_IMPL_ENV}=sglang_tilelang expects head_dim_v=512, got {head_dim_v}."
        )
    if indices is None:
        raise RuntimeError(f"{_IMPL_ENV}=sglang_tilelang requires sparse indices.")
    if extra_k_cache is None and extra_indices_in_kvcache is not None:
        raise RuntimeError(
            f"{_IMPL_ENV}=sglang_tilelang requires extra_k_cache when "
            "extra_indices_in_kvcache is provided."
        )
    if extra_k_cache is not None:
        if extra_indices_in_kvcache is None:
            raise RuntimeError(
                f"{_IMPL_ENV}=sglang_tilelang requires extra_indices_in_kvcache "
                "when extra_k_cache is provided."
            )
        if (
            extra_k_cache.dtype != torch.uint8
            or extra_k_cache.dim() != 4
            or extra_k_cache.shape[2] != 1
        ):
            raise RuntimeError(
                f"{_IMPL_ENV}=sglang_tilelang expects packed uint8 extra_k_cache "
                f"with shape [blocks, block, 1, bytes], got "
                f"dtype={extra_k_cache.dtype}, shape={extra_k_cache.shape}."
            )

    provider = _load_sglang_tilelang_provider()
    provider_out, provider_lse = provider(
        q=q,
        k_cache=k_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        head_dim_v=head_dim_v,
        tile_scheduler_metadata=tile_scheduler_metadata,
        num_splits=num_splits,
        softmax_scale=softmax_scale,
        causal=causal,
        is_fp8_kvcache=is_fp8_kvcache,
        indices=_to_int32_contiguous("indices", indices),
        attn_sink=attn_sink,
        extra_k_cache=extra_k_cache,
        extra_indices_in_kvcache=_to_int32_contiguous(
            "extra_indices_in_kvcache", extra_indices_in_kvcache
        ),
        topk_length=_to_int32_contiguous("topk_length", topk_length),
        extra_topk_length=_to_int32_contiguous(
            "extra_topk_length", extra_topk_length
        ),
    )

    expected_out_shape = (q.shape[0], q.shape[1], q.shape[2], head_dim_v)
    if tuple(provider_out.shape) != expected_out_shape:
        raise RuntimeError(
            f"{_IMPL_ENV}=sglang_tilelang provider returned output shape "
            f"{tuple(provider_out.shape)}, expected {expected_out_shape}."
        )
    if out is not None:
        if tuple(out.shape) != expected_out_shape:
            raise RuntimeError(
                f"{_IMPL_ENV}=sglang_tilelang received out shape "
                f"{tuple(out.shape)}, expected {expected_out_shape}."
            )
        out.copy_(provider_out.to(out.dtype))
        provider_out = out

    # SGLang's combine kernel returns [B, S, H]; MATE/vLLM callers expect
    # [B, H, S]. DeepSeek-V4 decode ignores LSE today, but keep the contract.
    if (
        provider_lse.dim() == 3
        and provider_lse.shape[0] == q.shape[0]
        and provider_lse.shape[1] == q.shape[1]
        and provider_lse.shape[2] == q.shape[2]
    ):
        provider_lse = provider_lse.permute(0, 2, 1).contiguous()
    elif tuple(provider_lse.shape) != (q.shape[0], q.shape[2], q.shape[1]):
        raise RuntimeError(
            f"{_IMPL_ENV}=sglang_tilelang provider returned LSE shape "
            f"{tuple(provider_lse.shape)}, expected "
            f"{(q.shape[0], q.shape[1], q.shape[2])} or "
            f"{(q.shape[0], q.shape[2], q.shape[1])}."
        )
    return provider_out, provider_lse
