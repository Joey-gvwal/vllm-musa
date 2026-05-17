# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
import math
import os

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

try:
    import mate.flashmla as _mate_flashmla
except ImportError as e:
    raise ImportError(
        "MUSA platform requires MATE to be installed. Please install mate first."
    ) from e

_mate_flash_mla_with_kvcache = _mate_flashmla.flash_mla_with_kvcache
_mate_get_mla_metadata = _mate_flashmla.get_mla_metadata
_flash_mla_sparse_fwd = getattr(_mate_flashmla, "flash_mla_sparse_fwd", None)
_DEEPSEEK_V4_SPARSE_KVCACHE_KWARGS = {
    "topk_length",
    "attn_sink",
    "extra_k_cache",
    "extra_indices_in_kvcache",
    "extra_topk_length",
    "out",
}
_DSV4_FP8_NOPE_DIM = 448
_DSV4_BF16_ROPE_DIM = 64
_DSV4_TOKEN_DATA_BYTES = _DSV4_FP8_NOPE_DIM + _DSV4_BF16_ROPE_DIM * 2
_DSV4_TOKEN_SCALE_BYTES = 8
_DSV4_QUANT_BLOCK_SIZE = 64


def _fp8_ds_mla_sparse_gather_impl() -> str:
    return os.getenv(
        "VLLM_MUSA_FP8_DS_MLA_SPARSE_GATHER_IMPL", "native"
    ).strip().lower()


def _fp8_ds_mla_sparse_reduce_impl() -> str:
    return os.getenv(
        "VLLM_MUSA_FP8_DS_MLA_SPARSE_REDUCE_IMPL", "off"
    ).strip().lower()


def _raise_deepseek_v4_sparse_flashmla_unavailable() -> None:
    raise RuntimeError(
        "DeepSeek-V4 sparse FlashMLA on MUSA requires a provider that supports "
        "`flash_mla_sparse_fwd` and the sparse `flash_mla_with_kvcache` kwargs "
        "used by upstream vLLM (`topk_length`, `attn_sink`, extra cache/index "
        "arguments, and `out`). The installed MATE FlashMLA exposes only the "
        "dense/standard sparse kvcache interface."
    )


def _supports_deepseek_v4_sparse_kvcache_kwargs() -> bool:
    try:
        signature = inspect.signature(_mate_flash_mla_with_kvcache)
    except (TypeError, ValueError):
        return False
    parameters = signature.parameters.values()
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters):
        return True
    return _DEEPSEEK_V4_SPARSE_KVCACHE_KWARGS.issubset(signature.parameters)


def _torch_flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if kwargs:
        raise TypeError(
            "Torch sparse FlashMLA fallback does not support kwargs: "
            f"{', '.join(sorted(kwargs))}"
        )
    if kv.shape[1] != 1 or indices.shape[1] != 1:
        raise RuntimeError(
            "Torch sparse FlashMLA fallback only supports MQA sparse MLA "
            f"with kv.shape[1] == indices.shape[1] == 1, got kv={kv.shape}, "
            f"indices={indices.shape}."
        )

    num_tokens, num_heads, _ = q.shape
    num_kv_tokens = kv.shape[0]
    topk = indices.shape[-1]
    idx = indices[:, 0, :].to(torch.long)
    valid = (idx >= 0) & (idx < num_kv_tokens)
    if topk_length is not None:
        topk_range = torch.arange(topk, device=indices.device)
        valid &= topk_range.unsqueeze(0) < topk_length.to(torch.long).unsqueeze(1)

    gathered = kv[:, 0, :].to(torch.float32).index_select(
        0, idx.masked_fill(~valid, 0).reshape(-1)
    )
    gathered = gathered.view(num_tokens, topk, kv.shape[-1])

    # Match the upstream sparse-MLA reference math in natural-log space. This
    # provider prioritizes correctness on MUSA when MATE lacks sparse FlashMLA;
    # it is not a performance replacement for a fused sparse FlashMLA kernel.
    logits = torch.einsum("thd,tkd->thk", q.to(torch.float32), gathered) * sm_scale
    logits = logits.masked_fill(~valid.unsqueeze(1), -float("inf"))
    no_key_mask = ~valid.any(dim=-1)
    orig_lse = torch.logsumexp(logits, dim=-1)
    max_logits = torch.max(logits, dim=-1).values
    orig_lse = orig_lse.masked_fill(no_key_mask[:, None], -float("inf"))
    max_logits = max_logits.masked_fill(no_key_mask[:, None], -float("inf"))
    lse_for_o = orig_lse
    if attn_sink is not None:
        sink = attn_sink[:num_heads].to(torch.float32).view(1, num_heads)
        lse_for_o = torch.logsumexp(
            torch.stack((orig_lse, sink.expand(num_tokens, -1)), dim=0),
            dim=0,
        )

    lse_for_o = lse_for_o.masked_fill(torch.isneginf(lse_for_o), float("inf"))
    weights = torch.exp(logits - lse_for_o.unsqueeze(-1))
    weights = weights.masked_fill(~valid.unsqueeze(1), 0.0)
    result = torch.einsum("thk,tkd->thd", weights, gathered[:, :, :d_v])
    result = result.masked_fill(no_key_mask[:, None, None], 0.0).to(q.dtype)
    lse = orig_lse.masked_fill(no_key_mask[:, None], float("inf"))
    if out is not None:
        out.copy_(result)
        result = out
    return result, max_logits, lse


def _reshape_sparse_indices(
    name: str,
    indices: torch.Tensor,
    num_queries: int,
) -> torch.Tensor:
    if indices.dim() == 3:
        if indices.shape[0] * indices.shape[1] != num_queries:
            raise RuntimeError(
                f"Torch sparse FlashMLA fallback expected {name} leading "
                f"dims to contain {num_queries} queries, got {indices.shape}."
            )
        return indices.reshape(num_queries, indices.shape[-1])
    if indices.dim() == 2 and indices.shape[0] == num_queries:
        return indices
    raise RuntimeError(
        f"Torch sparse FlashMLA fallback expected {name} shape "
        f"[batch, seq, topk] or [queries, topk], got {indices.shape}."
    )


def _reshape_sparse_lengths(
    name: str,
    lengths: torch.Tensor | None,
    num_queries: int,
) -> torch.Tensor | None:
    if lengths is None:
        return None
    if lengths.numel() != num_queries:
        raise RuntimeError(
            f"Torch sparse FlashMLA fallback expected {name} to contain "
            f"{num_queries} entries, got {lengths.shape}."
        )
    return lengths.reshape(num_queries).to(torch.long)


def _flatten_mqa_k_cache(name: str, k_cache: torch.Tensor) -> torch.Tensor:
    if k_cache.dtype == torch.uint8:
        if k_cache.dim() == 4 and k_cache.shape[2] == 1:
            return k_cache
        if k_cache.dim() == 3:
            return k_cache
        raise RuntimeError(
            "Torch sparse FlashMLA fallback expected packed fp8_ds_mla "
            f"{name} shape [blocks, block, 1, bytes] or [blocks, block, bytes], "
            f"got {k_cache.shape}."
        )
    if not k_cache.is_floating_point():
        raise RuntimeError(
            "Torch sparse FlashMLA fallback requires a dequantized floating "
            f"KV cache for diagnostics; {name} has dtype={k_cache.dtype}. "
            "Packed fp8_ds_mla decode still requires a real MUSA sparse "
            "FlashMLA provider."
        )
    if k_cache.dim() == 4:
        if k_cache.shape[2] != 1:
            raise RuntimeError(
                "Torch sparse FlashMLA fallback only supports MQA KV cache "
                f"with one KV head; {name} has shape {k_cache.shape}."
            )
        return k_cache.reshape(-1, 1, k_cache.shape[-1])
    if k_cache.dim() == 3:
        if k_cache.shape[1] != 1:
            raise RuntimeError(
                "Torch sparse FlashMLA fallback only supports MQA KV cache "
                f"with one KV head; {name} has shape {k_cache.shape}."
            )
        return k_cache
    if k_cache.dim() == 2:
        return k_cache.unsqueeze(1)
    raise RuntimeError(
        "Torch sparse FlashMLA fallback expected KV cache shape "
        f"[blocks, block, 1, dim], [tokens, 1, dim], or [tokens, dim]; "
        f"{name} has shape {k_cache.shape}."
    )


def _gather_fp8_ds_mla_sparse_kv(
    cache: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cache.dim() == 4 and cache.shape[2] != 1:
        raise RuntimeError(
            "Torch sparse FlashMLA fallback only supports MQA packed fp8_ds_mla "
            f"cache with one KV head, got {cache.shape}."
        )
    num_queries, topk = indices.shape
    num_blocks = cache.shape[0]
    block_size = cache.shape[1]
    num_kv_tokens = num_blocks * block_size
    if (
        current_platform.is_musa()
        and _fp8_ds_mla_sparse_gather_impl() == "native"
        and cache.dtype == torch.uint8
        and cache.is_contiguous()
        and indices.dtype in (torch.int32, torch.int64)
    ):
        native_gather = getattr(
            getattr(torch.ops, "_C_musa_ops", None),
            "fp8_ds_mla_sparse_gather",
            None,
        )
        native_lengths = lengths
        lengths_supported = native_lengths is None or (
            native_lengths.dtype in (torch.int32, torch.int64)
            and native_lengths.is_contiguous()
        )
        if native_gather is not None and lengths_supported:
            native_indices = indices if indices.is_contiguous() else indices.contiguous()
            gathered = torch.empty(
                (num_queries, topk, _DSV4_FP8_NOPE_DIM + _DSV4_BF16_ROPE_DIM),
                device=cache.device,
                dtype=torch.float32,
            )
            valid = torch.empty(
                (num_queries, topk), device=cache.device, dtype=torch.bool
            )
            native_gather(cache, native_indices, native_lengths, gathered, valid)
            return gathered, valid

    idx = indices.to(torch.long)
    valid = (idx >= 0) & (idx < num_kv_tokens)
    if lengths is not None:
        topk_range = torch.arange(topk, device=indices.device)
        valid &= topk_range.unsqueeze(0) < lengths.unsqueeze(1)

    safe_idx = idx.masked_fill(~valid, 0).reshape(-1)
    block_idx = torch.div(safe_idx, block_size, rounding_mode="floor")
    pos_in_block = safe_idx.remainder(block_size)
    flat_cache = cache.reshape(num_blocks, -1)

    token_offsets = (
        pos_in_block.unsqueeze(1) * _DSV4_TOKEN_DATA_BYTES
        + torch.arange(_DSV4_TOKEN_DATA_BYTES, device=cache.device).unsqueeze(0)
    )
    token_bytes = torch.gather(flat_cache[block_idx], 1, token_offsets)

    scale_offsets = (
        block_size * _DSV4_TOKEN_DATA_BYTES
        + pos_in_block.unsqueeze(1) * _DSV4_TOKEN_SCALE_BYTES
        + torch.arange(_DSV4_TOKEN_SCALE_BYTES, device=cache.device).unsqueeze(0)
    )
    scale_bytes = torch.gather(flat_cache[block_idx], 1, scale_offsets)

    fp8_nope = (
        token_bytes[:, :_DSV4_FP8_NOPE_DIM]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .to(torch.float32)
    )
    num_scale_blocks = _DSV4_FP8_NOPE_DIM // _DSV4_QUANT_BLOCK_SIZE
    scale = torch.exp2(scale_bytes[:, :num_scale_blocks].to(torch.float32) - 127.0)
    fp8_nope = (
        fp8_nope.view(-1, num_scale_blocks, _DSV4_QUANT_BLOCK_SIZE)
        * scale.unsqueeze(-1)
    ).reshape(-1, _DSV4_FP8_NOPE_DIM)

    bf16_rope = (
        token_bytes[:, _DSV4_FP8_NOPE_DIM : _DSV4_TOKEN_DATA_BYTES]
        .contiguous()
        .reshape(-1)
        .view(torch.bfloat16)
        .view(-1, _DSV4_BF16_ROPE_DIM)
        .to(torch.float32)
    )
    gathered = torch.cat((fp8_nope, bf16_rope), dim=-1)
    return gathered.view(num_queries, topk, -1), valid


def _gather_sparse_kv(
    cache: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cache.dtype == torch.uint8:
        return _gather_fp8_ds_mla_sparse_kv(cache, indices, lengths)
    num_queries, topk = indices.shape
    num_kv_tokens = cache.shape[0]
    idx = indices.to(torch.long)
    valid = (idx >= 0) & (idx < num_kv_tokens)
    if lengths is not None:
        topk_range = torch.arange(topk, device=indices.device)
        valid &= topk_range.unsqueeze(0) < lengths.unsqueeze(1)

    gathered = cache[:, 0, :].to(torch.float32).index_select(
        0, idx.masked_fill(~valid, 0).reshape(-1)
    )
    gathered = gathered.view(num_queries, topk, cache.shape[-1])
    return gathered, valid


def _try_musa_fp8_ds_mla_sparse_reduce(
    q_flat: torch.Tensor,
    gathered: torch.Tensor,
    valid: torch.Tensor,
    batch: int,
    seq_len: int,
    head_dim_v: int,
    softmax_scale: float,
    attn_sink: torch.Tensor | None,
    out: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if (
        not current_platform.is_musa()
        or _fp8_ds_mla_sparse_reduce_impl() != "native"
        or gathered.dtype != torch.float32
        or valid.dtype != torch.bool
        or not q_flat.is_contiguous()
        or not gathered.is_contiguous()
        or not valid.is_contiguous()
        or q_flat.dtype not in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return None

    native_reduce = getattr(
        getattr(torch.ops, "_C_musa_ops", None),
        "fp8_ds_mla_sparse_reduce",
        None,
    )
    if native_reduce is None:
        return None
    if attn_sink is not None and (
        not attn_sink.is_contiguous()
        or attn_sink.dtype not in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return None

    num_queries, num_heads, _ = q_flat.shape
    if out is None:
        result_flat = torch.empty(
            (num_queries, num_heads, head_dim_v),
            device=q_flat.device,
            dtype=q_flat.dtype,
        )
    else:
        if not out.is_contiguous():
            return None
        result_flat = out.reshape(num_queries, num_heads, head_dim_v)
    lse_flat = torch.empty(
        (num_queries, num_heads), device=q_flat.device, dtype=torch.float32
    )
    native_reduce(
        q_flat,
        gathered,
        valid,
        attn_sink,
        softmax_scale,
        result_flat,
        lse_flat,
    )
    result = result_flat.reshape(batch, seq_len, num_heads, head_dim_v)
    lse = lse_flat.reshape(batch, seq_len, num_heads).permute(0, 2, 1).contiguous()
    if out is not None:
        result = out
    return result, lse


def _torch_flash_mla_with_kvcache_sparse_fallback(
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
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    if kwargs:
        raise TypeError(
            "Torch sparse FlashMLA fallback does not support kwargs: "
            f"{', '.join(sorted(kwargs))}"
        )
    if q.dim() != 4:
        raise RuntimeError(
            "Torch sparse FlashMLA kvcache fallback expects q shape "
            f"[batch, seq, heads, dim], got {q.shape}."
        )
    if causal:
        raise RuntimeError(
            "Torch sparse FlashMLA kvcache fallback only supports sparse "
            "non-causal decode; causal dense decode should use the MATE path."
        )
    if indices is None:
        raise RuntimeError(
            "Torch sparse FlashMLA kvcache fallback requires sparse indices."
        )
    if extra_k_cache is None and extra_indices_in_kvcache is not None:
        raise RuntimeError(
            "Torch sparse FlashMLA kvcache fallback requires extra_k_cache "
            "when extra_indices_in_kvcache is provided."
        )
    if extra_k_cache is not None and extra_indices_in_kvcache is None:
        raise RuntimeError(
            "Torch sparse FlashMLA kvcache fallback requires "
            "extra_indices_in_kvcache when extra_k_cache is provided."
        )

    del block_table, cache_seqlens, tile_scheduler_metadata, num_splits
    if is_fp8_kvcache:
        logger.warning_once(
            "Using vllm-musa torch sparse FlashMLA kvcache correctness provider "
            "with "
            "is_fp8_kvcache=True. Packed fp8_ds_mla cache bytes are "
            "dequantized with torch operations; this is not a fused sparse "
            "FlashMLA performance kernel."
        )
    else:
        logger.warning_once(
            "Using vllm-musa torch sparse FlashMLA kvcache correctness "
            "provider. This is not a fused sparse FlashMLA performance kernel."
        )

    batch, seq_len, num_heads, q_dim = q.shape
    num_queries = batch * seq_len
    if softmax_scale is None:
        softmax_scale = q_dim ** (-0.5)

    main_indices = _reshape_sparse_indices("indices", indices, num_queries)
    main_lengths = _reshape_sparse_lengths("topk_length", topk_length, num_queries)
    main_cache = _flatten_mqa_k_cache("k_cache", k_cache)
    gathered, valid = _gather_sparse_kv(main_cache, main_indices, main_lengths)

    if extra_k_cache is not None:
        assert extra_indices_in_kvcache is not None
        extra_indices = _reshape_sparse_indices(
            "extra_indices_in_kvcache", extra_indices_in_kvcache, num_queries
        )
        extra_lengths = _reshape_sparse_lengths(
            "extra_topk_length", extra_topk_length, num_queries
        )
        extra_cache = _flatten_mqa_k_cache("extra_k_cache", extra_k_cache)
        extra_gathered, extra_valid = _gather_sparse_kv(
            extra_cache, extra_indices, extra_lengths
        )
        gathered = torch.cat((gathered, extra_gathered), dim=1)
        valid = torch.cat((valid, extra_valid), dim=1)

    required_dim = max(q_dim, head_dim_v)
    if gathered.shape[-1] < required_dim:
        raise RuntimeError(
            "Torch sparse FlashMLA kvcache fallback requires KV dim >= "
            f"max(q_dim={q_dim}, head_dim_v={head_dim_v}), got "
            f"{gathered.shape[-1]}."
        )

    q_flat_view = q.reshape(num_queries, num_heads, q_dim)
    native_reduce = _try_musa_fp8_ds_mla_sparse_reduce(
        q_flat_view,
        gathered,
        valid,
        batch,
        seq_len,
        head_dim_v,
        softmax_scale,
        attn_sink,
        out,
    )
    if native_reduce is not None:
        return native_reduce

    q_flat = q_flat_view.to(torch.float32)
    key = gathered[:, :, :q_dim]
    value = gathered[:, :, :head_dim_v]
    logits = torch.einsum("qhd,qkd->qhk", q_flat, key) * softmax_scale
    logits = logits.masked_fill(~valid.unsqueeze(1), -float("inf"))
    no_key_mask = ~valid.any(dim=-1)
    key_lse = torch.logsumexp(logits, dim=-1)
    key_lse = key_lse.masked_fill(no_key_mask[:, None], -float("inf"))
    lse_for_o = key_lse
    if attn_sink is not None:
        sink = attn_sink[:num_heads].to(torch.float32).view(1, num_heads)
        lse_for_o = torch.logaddexp(key_lse, sink.expand(num_queries, -1))

    lse_for_o = lse_for_o.masked_fill(torch.isneginf(lse_for_o), float("inf"))
    weights = torch.exp(logits - lse_for_o.unsqueeze(-1))
    weights = weights.masked_fill(~valid.unsqueeze(1), 0.0)
    result = torch.einsum("qhk,qkd->qhd", weights, value)
    result = result.masked_fill(no_key_mask[:, None, None], 0.0)
    result = result.reshape(batch, seq_len, num_heads, head_dim_v).to(q.dtype)
    lse = (
        key_lse.masked_fill(no_key_mask[:, None], float("inf"))
        .reshape(batch, seq_len, num_heads)
        .permute(0, 2, 1)
        .contiguous()
    )

    if out is not None:
        out.copy_(result.to(out.dtype))
        result = out
    return result, lse


def flash_mla_with_kvcache(
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
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    has_deepseek_v4_sparse_kwargs = (
        topk_length is not None
        or attn_sink is not None
        or extra_k_cache is not None
        or extra_indices_in_kvcache is not None
        or extra_topk_length is not None
        or out is not None
        or kwargs
    )
    if has_deepseek_v4_sparse_kwargs:
        if _flash_mla_sparse_fwd is None:
            return _torch_flash_mla_with_kvcache_sparse_fallback(
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
                indices=indices,
                topk_length=topk_length,
                attn_sink=attn_sink,
                extra_k_cache=extra_k_cache,
                extra_indices_in_kvcache=extra_indices_in_kvcache,
                extra_topk_length=extra_topk_length,
                out=out,
                **kwargs,
            )
        if _supports_deepseek_v4_sparse_kvcache_kwargs():
            return _mate_flash_mla_with_kvcache(
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
                indices=indices,
                topk_length=topk_length,
                attn_sink=attn_sink,
                extra_k_cache=extra_k_cache,
                extra_indices_in_kvcache=extra_indices_in_kvcache,
                extra_topk_length=extra_topk_length,
                out=out,
                **kwargs,
            )
        return _torch_flash_mla_with_kvcache_sparse_fallback(
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
            indices=indices,
            topk_length=topk_length,
            attn_sink=attn_sink,
            extra_k_cache=extra_k_cache,
            extra_indices_in_kvcache=extra_indices_in_kvcache,
            extra_topk_length=extra_topk_length,
            out=out,
            **kwargs,
        )
    return _mate_flash_mla_with_kvcache(
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
        indices=indices,
    )


# vllm.v1.Attention.ops.flashmla will be registered and used earlier than this patch, but it will not affect
def _is_flashmla_available() -> tuple[bool, str | None]:
    return True, None


def _is_flashmla_sparse_available() -> tuple[bool, str | None]:
    if _flash_mla_sparse_fwd is None:
        return True, None
    if not _supports_deepseek_v4_sparse_kvcache_kwargs():
        return True, None
    return True, None


def is_flashmla_dense_supported() -> tuple[bool, str | None]:
    """
    Return: is_supported_flag, unsupported_reason (optional).
    """
    is_available, maybe_reason = _is_flashmla_available()
    if not is_available:
        return False, maybe_reason
    # Only MUSA devices support FlashMLA Dense
    if not current_platform.is_musa():
        return False, "FlashMLA Dense is only supported on MUSA devices."
    return True, None


def is_flashmla_sparse_supported() -> tuple[bool, str | None]:
    """
    Return: is_supported_flag, unsupported_reason (optional).
    """
    is_available, maybe_reason = _is_flashmla_available()
    if not is_available:
        return False, maybe_reason
    is_sparse_available, sparse_reason = _is_flashmla_sparse_available()
    if not is_sparse_available:
        return False, sparse_reason
    # MUSA devices use compute capability 3
    device_capability = current_platform.get_device_capability()
    if device_capability is None or device_capability[0] != 3:
        return (
            False,
            "FlashMLA Sparse is only supported on MUSA devices.",
        )
    return True, None


def _raise_flashmla_unavailable(*_args, **_kwargs):
    _, reason = _is_flashmla_available()
    raise RuntimeError(reason or "FlashMLA is not available")


def _raise_flashmla_sparse_unavailable(*_args, **_kwargs):
    _, reason = _is_flashmla_sparse_available()
    raise RuntimeError(reason or "FlashMLA sparse is not available")


class FlashMLASchedMeta:
    def __init__(self, tile_scheduler_metadata: torch.Tensor, num_splits: torch.Tensor):
        self.tile_scheduler_metadata = tile_scheduler_metadata
        self.num_splits = num_splits


def get_mla_metadata(*args, **kwargs):
    if not args and not kwargs:
        return (FlashMLASchedMeta(None, None),)
    return _mate_get_mla_metadata(*args, **kwargs)


if _flash_mla_sparse_fwd is not None:
    flash_mla_sparse_fwd = _flash_mla_sparse_fwd
else:
    flash_mla_sparse_fwd = _torch_flash_mla_sparse_fwd


def get_mla_metadata_dense_fp8(
    cache_seqlens: torch.Tensor,
    num_q_tokens_per_head_k: int,
    num_heads_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _is_flashmla_available()[0]:
        _raise_flashmla_unavailable()
    return torch.ops._flashmla_extension_C.get_mla_decoding_metadata_dense_fp8(
        cache_seqlens,
        num_q_tokens_per_head_k,
        num_heads_k,
    )


def flash_mla_with_kvcache_fp8(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
    descale_q: torch.Tensor | None = None,
    descale_k: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _is_flashmla_available()[0]:
        _raise_flashmla_unavailable()
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    out, softmax_lse = torch.ops._flashmla_extension_C.fwd_kvcache_mla_fp8(
        q,
        k_cache,
        head_dim_v,
        cache_seqlens,
        block_table,
        softmax_scale,
        causal,
        tile_scheduler_metadata,
        num_splits,
        descale_q,
        descale_k,
    )
    return out, softmax_lse
