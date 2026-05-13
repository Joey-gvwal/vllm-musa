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
        else:
            raise NotImplementedError(
                "SparseAttnIndexer native forward is only implemented for "
                "CUDA, ROCm and XPU platforms."
            )
""",
    ),
]

RELOAD_AFTER_PATCH = True
