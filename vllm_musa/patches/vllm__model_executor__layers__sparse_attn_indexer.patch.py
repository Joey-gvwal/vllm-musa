# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch sparse-attention indexer with an opt-in MUSA diagnostic fallback.
"""

PATCHES = [
    (
        """import torch

import vllm.envs as envs
""",
        """import os

import torch

import vllm.envs as envs
""",
    ),
    (
        """logger = init_logger(__name__)
""",
        """logger = init_logger(__name__)


def _musa_fill_recent_sparse_indexer_indices(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    topk_tokens: int,
    topk_indices_buffer: torch.Tensor,
) -> torch.Tensor:
    # Diagnostic fallback: preserve sparse-attention control flow by selecting a
    # small recent compressed-token window when the learned indexer kernels are
    # unavailable on MUSA.
    topk = min(topk_tokens, topk_indices_buffer.shape[-1])
    topk_indices_buffer[: hidden_states.shape[0], :topk] = -1
    if topk <= 0:
        return topk_indices_buffer

    cap = int(os.getenv("VLLM_MUSA_SPARSE_INDEXER_FALLBACK_TOPK", "16"))
    cap = max(0, min(cap, topk))
    if cap == 0:
        return topk_indices_buffer

    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        return topk_indices_buffer

    metadata = attn_metadata.get(_resolve_layer_name(k_cache_prefix))
    if not isinstance(metadata, DeepseekV32IndexerMetadata):
        return topk_indices_buffer

    def fill_rows(row_start: int, lengths: torch.Tensor) -> None:
        flat_lengths = lengths.reshape(-1).to(torch.long)
        rows = min(
            flat_lengths.numel(),
            hidden_states.shape[0] - row_start,
            topk_indices_buffer.shape[0] - row_start,
        )
        for row_offset in range(rows):
            length = int(flat_lengths[row_offset].item())
            count = min(cap, max(length, 0))
            if count <= 0:
                continue
            start = length - count
            row = row_start + row_offset
            topk_indices_buffer[row, :count] = torch.arange(
                start,
                length,
                device=topk_indices_buffer.device,
                dtype=topk_indices_buffer.dtype,
            )

    if metadata.num_decodes > 0 and metadata.decode is not None:
        fill_rows(0, metadata.decode.seq_lens)

    if metadata.num_prefills > 0 and metadata.prefill is not None:
        for chunk in metadata.prefill.chunks:
            fill_rows(
                int(chunk.token_start),
                chunk.cu_seqlen_ke - chunk.cu_seqlen_ks,
            )

    return topk_indices_buffer


def _musa_indexer_cache_block(kv_cache: torch.Tensor, block_id: int) -> torch.Tensor:
    return kv_cache[block_id].view(torch.uint8).flatten()


def _musa_dequant_indexer_fp8_cache_row(
    kv_cache: torch.Tensor,
    block_id: int,
    pos_in_block: int,
    head_dim: int,
) -> torch.Tensor:
    block_size = kv_cache.shape[1]
    scale_dim = 4
    cache_block = _musa_indexer_cache_block(kv_cache, block_id)
    token_base = pos_in_block * head_dim
    scale_base = block_size * head_dim + pos_in_block * scale_dim
    values = (
        cache_block[token_base : token_base + head_dim]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .to(torch.float32)
    )
    scale = (
        cache_block[scale_base : scale_base + scale_dim]
        .contiguous()
        .view(torch.float32)
    )
    return values * scale


def _musa_gather_indexer_fp8_cache(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    total_seq_lens = int(cu_seq_lens[-1].item())
    gathered = torch.empty(
        (total_seq_lens, head_dim),
        dtype=torch.float32,
        device=kv_cache.device,
    )
    block_size = kv_cache.shape[1]
    for req_idx in range(block_table.shape[0]):
        start = int(cu_seq_lens[req_idx].item())
        end = int(cu_seq_lens[req_idx + 1].item())
        for local_pos in range(end - start):
            physical_block = int(block_table[req_idx, local_pos // block_size].item())
            pos_in_block = local_pos % block_size
            gathered[start + local_pos] = _musa_dequant_indexer_fp8_cache_row(
                kv_cache,
                physical_block,
                pos_in_block,
                head_dim,
            )
    return gathered


def _musa_sparse_indexer_logits(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    # q is FP8-dequantized without an explicit q scale; the FP8 q scale is
    # already folded into weights by fused_indexer_q_rope_quant.
    per_head = torch.einsum("h d, n d -> h n", q.to(torch.float32), k)
    return (per_head * weights.to(torch.float32).unsqueeze(-1)).sum(dim=0)


def _musa_fill_topk_rows_from_indexer_logits(
    q_deq: torch.Tensor,
    k_deq: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
) -> None:
    rows = min(q_deq.shape[0], topk_indices.shape[0], cu_seqlen_ks.numel())
    for row in range(rows):
        start = int(cu_seqlen_ks[row].item())
        end = int(cu_seqlen_ke[row].item())
        row_len = max(0, end - start)
        if row_len == 0:
            continue
        k_i = min(int(topk_tokens), row_len, topk_indices.shape[1])
        logits = _musa_sparse_indexer_logits(
            q_deq[row],
            k_deq[start:end],
            weights[row],
        )
        topk_indices[row, :k_i] = torch.topk(logits, k_i, dim=-1).indices.to(
            topk_indices.dtype
        )


def _musa_fill_decode_topk_from_indexer_cache(
    q_deq: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    decode_metadata,
    topk_indices_buffer: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
) -> None:
    seq_lens = decode_metadata.seq_lens.reshape(-1)
    rows = min(q_deq.shape[0], seq_lens.numel(), topk_indices_buffer.shape[0])
    block_size = kv_cache.shape[1]
    for row in range(rows):
        seq_len = int(seq_lens[row].item())
        if seq_len <= 0:
            continue
        block_row = min(row, decode_metadata.block_table.shape[0] - 1)
        gathered = torch.empty(
            (seq_len, head_dim),
            dtype=torch.float32,
            device=kv_cache.device,
        )
        for local_pos in range(seq_len):
            physical_block = int(
                decode_metadata.block_table[
                    block_row, local_pos // block_size
                ].item()
            )
            gathered[local_pos] = _musa_dequant_indexer_fp8_cache_row(
                kv_cache,
                physical_block,
                local_pos % block_size,
                head_dim,
            )
        k_i = min(int(topk_tokens), seq_len, topk_indices_buffer.shape[1])
        logits = _musa_sparse_indexer_logits(q_deq[row], gathered, weights[row])
        topk_indices_buffer[row, :k_i] = torch.topk(logits, k_i, dim=-1).indices.to(
            topk_indices_buffer.dtype
        )


def _musa_fill_exact_sparse_indexer_indices(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
    topk_indices_buffer: torch.Tensor,
    use_fp4_cache: bool,
) -> torch.Tensor:
    if use_fp4_cache or isinstance(q_quant, tuple):
        raise NotImplementedError(
            "MUSA exact sparse-attention indexer fallback currently supports "
            "the FP8 indexer-cache path only."
        )

    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        return topk_indices_buffer

    metadata = attn_metadata.get(_resolve_layer_name(k_cache_prefix))
    if not isinstance(metadata, DeepseekV32IndexerMetadata):
        return topk_indices_buffer

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    q_deq = q_quant.to(torch.float32)
    weights = weights.to(torch.float32)

    if metadata.num_prefills > 0 and metadata.prefill is not None:
        for chunk in metadata.prefill.chunks:
            k_deq = _musa_gather_indexer_fp8_cache(
                kv_cache,
                chunk.block_table,
                chunk.cu_seq_lens,
                head_dim,
            )
            token_start = int(chunk.token_start)
            token_end = int(chunk.token_end)
            _musa_fill_topk_rows_from_indexer_logits(
                q_deq[token_start:token_end],
                k_deq,
                weights[token_start:token_end],
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                topk_indices_buffer[token_start:token_end, :topk_tokens],
                topk_tokens,
            )

    if metadata.num_decodes > 0 and metadata.decode is not None:
        _musa_fill_decode_topk_from_indexer_cache(
            q_deq[: metadata.num_decode_tokens],
            kv_cache,
            weights[: metadata.num_decode_tokens],
            metadata.decode,
            topk_indices_buffer[: metadata.num_decode_tokens, :topk_tokens],
            topk_tokens,
            head_dim,
        )

    return topk_indices_buffer
""",
    ),
    (
        """        else:
            raise NotImplementedError(
                "SparseAttnIndexer native forward is only implemented for "
                "CUDA, ROCm and XPU platforms."
            )
""",
        """        elif (
            current_platform.is_musa()
            and os.getenv(
                "VLLM_MUSA_ENABLE_TORCH_SPARSE_ATTN_INDEXER_FALLBACK",
                "0",
            )
            == "1"
        ):
            logger.warning_once(
                "Using opt-in MUSA DeepSeek-V4 sparse-attention indexer "
                "fallback. It computes the learned sparse top-k in torch for "
                "the FP8 indexer-cache path; this is diagnostic, not a "
                "production indexer backend."
            )
            try:
                return _musa_fill_exact_sparse_indexer_indices(
                    hidden_states,
                    self.k_cache.prefix,
                    self.k_cache.kv_cache,
                    q_quant,
                    weights,
                    self.topk_tokens,
                    self.head_dim,
                    self.topk_indices_buffer,
                    self.use_fp4_cache,
                )
            except NotImplementedError:
                logger.warning_once(
                    "Falling back to bounded recent compressed-token sparse "
                    "indices because exact MUSA torch sparse-attention indexer "
                    "fallback is unavailable for this cache format."
                )
                return _musa_fill_recent_sparse_indexer_indices(
                    hidden_states,
                    self.k_cache.prefix,
                    self.topk_tokens,
                    self.topk_indices_buffer,
                )
        else:
            raise NotImplementedError(
                "SparseAttnIndexer native forward is only implemented for "
                "CUDA, ROCm and XPU platforms."
        )
""",
    ),
    (
        """        elif (
            current_platform.is_musa()
            and os.getenv(
                "VLLM_MUSA_ENABLE_TORCH_SPARSE_ATTN_INDEXER_FALLBACK",
                "0",
            )
            == "1"
        ):
            logger.warning_once(
                "Using opt-in MUSA DeepSeek-V4 sparse-attention indexer "
                "fallback. It selects a bounded recent compressed-token "
                "window instead of learned global sparse picks; this is "
                "diagnostic, not a production indexer backend."
            )
            return _musa_fill_recent_sparse_indexer_indices(
                hidden_states,
                self.k_cache.prefix,
                self.topk_tokens,
                self.topk_indices_buffer,
            )
""",
        """        elif (
            current_platform.is_musa()
            and os.getenv(
                "VLLM_MUSA_ENABLE_TORCH_SPARSE_ATTN_INDEXER_FALLBACK",
                "0",
            )
            == "1"
        ):
            logger.warning_once(
                "Using opt-in MUSA DeepSeek-V4 sparse-attention indexer "
                "fallback. It computes the learned sparse top-k in torch for "
                "the FP8 indexer-cache path; this is diagnostic, not a "
                "production indexer backend."
            )
            try:
                return _musa_fill_exact_sparse_indexer_indices(
                    hidden_states,
                    self.k_cache.prefix,
                    self.k_cache.kv_cache,
                    q_quant,
                    weights,
                    self.topk_tokens,
                    self.head_dim,
                    self.topk_indices_buffer,
                    self.use_fp4_cache,
                )
            except NotImplementedError:
                logger.warning_once(
                    "Falling back to bounded recent compressed-token sparse "
                    "indices because exact MUSA torch sparse-attention indexer "
                    "fallback is unavailable for this cache format."
                )
                return _musa_fill_recent_sparse_indexer_indices(
                    hidden_states,
                    self.k_cache.prefix,
                    self.topk_tokens,
                    self.topk_indices_buffer,
                )
""",
    ),
]

RELOAD_AFTER_PATCH = True
