# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch DeepSeek-V4 cache utility kernels with MUSA-specific gates.
"""

PATCHES = [
    (
        """import torch

from vllm.triton_utils import tl, triton
""",
        """import os

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
import vllm_musa._custom_ops as musa_ops

logger = init_logger(__name__)


def _raise_musa_deepseek_v4_cache_unsupported(op_name: str) -> None:
    raise NotImplementedError(
        f"DeepSeek-V4 {op_name} is not implemented for MUSA yet. "
        "A MUSA-safe cache quantization, dequantization, top-k metadata, "
        "or sparse prefill index implementation is required before model "
        "execution can proceed."
    )


def _is_musa_tensor(tensor: torch.Tensor) -> bool:
    return (
        current_platform.is_musa()
        or getattr(torch.version, "musa", None) is not None
        or tensor.device.type == "musa"
    )


def _musa_deepseek_v4_cache_fallback_enabled() -> bool:
    return (
        os.getenv("VLLM_MUSA_ENABLE_TORCH_DEEPSEEK_V4_CACHE_FALLBACK", "0")
        == "1"
    )


def _musa_deepseek_v4_cache_dequant_triton_enabled() -> bool:
    return (
        os.getenv("VLLM_MUSA_DEEPSEEK_V4_CACHE_DEQUANT_TRITON", "0")
        == "1"
    )


def _musa_warn_cache_fallback_once(op_name: str) -> None:
    logger.warning_once(
        "Using opt-in MUSA torch DeepSeek-V4 %s fallback. This emulates "
        "the Triton cache/top-k utility in torch; it is a correctness "
        "fallback, not a production backend.",
        op_name,
    )


def _musa_dequantize_and_gather_k_cache_native(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    return musa_ops.deepseek_v4_dequantize_and_gather_k_cache(
        out, k_cache, seq_lens, gather_lens, block_table, block_size, offset
    )


def _musa_compute_global_topk_indices_and_lens_native(
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return musa_ops.deepseek_v4_compute_global_topk_indices_and_lens(
        topk_indices, token_to_req_indices, block_table, block_size, is_valid_token
    )


def _musa_combine_topk_swa_indices_native(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return musa_ops.deepseek_v4_combine_topk_swa_indices(
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        window_size,
        compress_ratio,
        topk,
        M,
        N,
    )


def _musa_deepseek_v4_block_cache_view(k_cache: torch.Tensor) -> torch.Tensor:
    if k_cache.dtype != torch.uint8:
        raise AssertionError(f"DeepSeek-V4 K cache must be uint8, got {k_cache.dtype}")
    block_stride = k_cache.stride(0)
    return k_cache.as_strided((k_cache.shape[0], block_stride), (block_stride, 1))


def _musa_dequantize_and_gather_k_cache_fallback(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    # Keep this helper structurally distinct from the upstream function body so
    # later string replacements do not patch the fallback into calling itself.
    TOKEN_FP8_DIM = 448
    TOKEN_BF16_DIM = 64
    TOKEN_SCALE_DIM = 8
    QUANT_BLOCK_SIZE = 64
    TOKEN_DATA_SIZE = TOKEN_FP8_DIM + TOKEN_BF16_DIM * 2
    N_QUANT_BLOCKS = 7

    if out.shape[-1] != TOKEN_FP8_DIM + TOKEN_BF16_DIM:
        raise AssertionError(f"DeepSeek-V4 gather output must end in 512, got {out.shape}")

    cache_blocks = _musa_deepseek_v4_block_cache_view(k_cache)
    for req_idx in range(seq_lens.shape[0]):
        seq_len = int(seq_lens[req_idx].item())
        gather_len = (
            int(gather_lens[req_idx].item()) if gather_lens is not None else seq_len
        )
        start_pos = seq_len - gather_len
        for i in range(gather_len):
            pos = start_pos + i
            block_in_seq = pos // block_size
            pos_in_block = pos % block_size
            physical_block_idx = int(block_table[req_idx, block_in_seq].item())
            block_bytes = cache_blocks[physical_block_idx]

            token_base = pos_in_block * TOKEN_DATA_SIZE
            scale_base = block_size * TOKEN_DATA_SIZE + pos_in_block * TOKEN_SCALE_DIM
            output_row = out[req_idx, offset + i]

            for qblock_idx in range(N_QUANT_BLOCKS):
                qblock_start = qblock_idx * QUANT_BLOCK_SIZE
                qbytes = block_bytes[
                    token_base + qblock_start : token_base + qblock_start + QUANT_BLOCK_SIZE
                ]
                encoded_scale = block_bytes[scale_base + qblock_idx].to(torch.float32)
                scale = torch.exp2(encoded_scale - 127.0)
                dequant = qbytes.view(torch.float8_e4m3fn).to(torch.float32) * scale
                output_row[qblock_start : qblock_start + QUANT_BLOCK_SIZE].copy_(
                    dequant.to(out.dtype)
                )

            bf16_start = token_base + TOKEN_FP8_DIM
            bf16_bytes = block_bytes[bf16_start : bf16_start + TOKEN_BF16_DIM * 2]
            output_row[TOKEN_FP8_DIM:].copy_(bf16_bytes.view(torch.bfloat16).to(out.dtype))


def _musa_compute_global_topk_indices_and_lens_fallback(
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = topk_indices.shape[0]
    local_indices = topk_indices[:num_tokens].to(torch.long)
    valid = local_indices >= 0
    safe_local_indices = local_indices.clamp(min=0)
    req_indices = token_to_req_indices[:num_tokens].to(torch.long).clamp(
        0, block_table.shape[0] - 1
    )
    block_indices = torch.div(
        safe_local_indices, block_size, rounding_mode="floor"
    ).clamp(0, block_table.shape[1] - 1)
    block_offsets = safe_local_indices.remainder(block_size)
    block_numbers = block_table[
        req_indices.unsqueeze(-1),
        block_indices,
    ].to(torch.long)
    slot_ids = block_numbers * block_size + block_offsets
    global_topk_indices = torch.where(
        valid,
        slot_ids.to(topk_indices.dtype),
        torch.full_like(topk_indices[:num_tokens], -1),
    )
    valid_counts = valid.sum(dim=-1).to(torch.int32)
    topk_lens = torch.where(
        is_valid_token[:num_tokens].to(torch.bool),
        valid_counts,
        torch.zeros_like(valid_counts),
    )
    return global_topk_indices, topk_lens


def _musa_combine_topk_swa_indices_fallback(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = topk_indices.shape[0]
    # Keep this helper structurally distinct from the upstream function body so
    # later string replacements do not patch the fallback into calling itself.
    num_reqs = seq_lens.shape[0]
    combined_topk = (
        (topk + window_size + _SPARSE_PREFILL_TOPK_ALIGNMENT - 1)
        // _SPARSE_PREFILL_TOPK_ALIGNMENT
        * _SPARSE_PREFILL_TOPK_ALIGNMENT
    )
    combined_indices = torch.full(
        (num_tokens, combined_topk),
        fill_value=-1,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    combined_lens = torch.zeros(
        num_tokens, dtype=torch.int32, device=topk_indices.device
    )
    base = int(query_start_loc[0].item())
    for batch_idx in range(num_reqs):
        query_start = int(query_start_loc[batch_idx].item()) - base
        query_end = int(query_start_loc[batch_idx + 1].item()) - base
        query_len = query_end - query_start
        seq_len = int(seq_lens[batch_idx].item())
        gather_len = int(gather_lens[batch_idx].item())
        start_pos = seq_len - query_len
        gather_start = seq_len - gather_len
        req_offset = int(M) * batch_idx
        for token_idx in range(query_start, query_end):
            pos = start_pos + token_idx - query_start
            topk_len = min((pos + 1) // int(compress_ratio), int(topk))
            swa_len = min(pos + 1, int(window_size))
            if topk_len > 0:
                topk_values = topk_indices[token_idx, :topk_len].to(torch.int32)
                combined_indices[token_idx, :topk_len] = topk_values + req_offset
            if swa_len > 0:
                swa_values = (
                    torch.arange(swa_len, device=topk_indices.device, dtype=torch.int32)
                    + req_offset
                    + int(N)
                    + pos
                    - swa_len
                    + 1
                    - gather_start
                )
                combined_indices[
                    token_idx,
                    topk_len : topk_len + swa_len,
                ] = swa_values
            combined_lens[token_idx] = topk_len + swa_len
    return combined_indices, combined_lens
""",
    ),
    (
        """    assert k.dim() == 2 and k.shape[1] == 512, (
        f"K must be [num_tokens, 512], got {k.shape}"
    )
""",
        """    if _is_musa_tensor(k):
        _raise_musa_deepseek_v4_cache_unsupported("quantize_and_insert_k_cache")
    assert k.dim() == 2 and k.shape[1] == 512, (
        f"K must be [num_tokens, 512], got {k.shape}"
    )
""",
    ),
    (
        """) -> None:
    TOKEN_FP8_DIM = 448
    TOKEN_BF16_DIM = 64
""",
        """) -> None:
    if _is_musa_tensor(out):
        if _musa_deepseek_v4_cache_fallback_enabled():
            _musa_warn_cache_fallback_once("dequantize_and_gather_k_cache")
            return _musa_dequantize_and_gather_k_cache_fallback(
                out, k_cache, seq_lens, gather_lens, block_table, block_size, offset
            )
        if _musa_deepseek_v4_cache_dequant_triton_enabled():
            logger.warning_once(
                "Using opt-in MUSA Triton DeepSeek-V4 "
                "dequantize_and_gather_k_cache path."
            )
        else:
            return _musa_dequantize_and_gather_k_cache_native(
                out, k_cache, seq_lens, gather_lens, block_table, block_size, offset
            )
    TOKEN_FP8_DIM = 448
    TOKEN_BF16_DIM = 64
""",
    ),
    (
        """    num_tokens = topk_indices.shape[0]
    global_topk_indices = torch.empty_like(topk_indices)
""",
        """    if _is_musa_tensor(topk_indices):
        if _musa_deepseek_v4_cache_fallback_enabled():
            _musa_warn_cache_fallback_once("compute_global_topk_indices_and_lens")
            return _musa_compute_global_topk_indices_and_lens_fallback(
                topk_indices,
                token_to_req_indices,
                block_table,
                block_size,
                is_valid_token,
            )
        return _musa_compute_global_topk_indices_and_lens_native(
            topk_indices,
            token_to_req_indices,
            block_table,
            block_size,
            is_valid_token,
        )
    num_tokens = topk_indices.shape[0]
    global_topk_indices = torch.empty_like(topk_indices)
""",
    ),
    (
        """    num_tokens = topk_indices.shape[0]
    num_reqs = seq_lens.shape[0]
""",
        """    if _is_musa_tensor(topk_indices):
        if _musa_deepseek_v4_cache_fallback_enabled():
            _musa_warn_cache_fallback_once("combine_topk_swa_indices")
            return _musa_combine_topk_swa_indices_fallback(
                topk_indices,
                query_start_loc,
                seq_lens,
                gather_lens,
                window_size,
                compress_ratio,
                topk,
                M,
                N,
            )
        return _musa_combine_topk_swa_indices_native(
            topk_indices,
            query_start_loc,
            seq_lens,
            gather_lens,
            window_size,
            compress_ratio,
            topk,
            M,
            N,
        )
    num_tokens = topk_indices.shape[0]
    num_reqs = seq_lens.shape[0]
""",
    ),
    (
        """def _musa_compute_global_topk_indices_and_lens_fallback(
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = topk_indices.shape[0]
    global_topk_indices = torch.full_like(topk_indices, -1)
    topk_lens = torch.empty(num_tokens, dtype=torch.int32, device=topk_indices.device)
    for token_idx in range(num_tokens):
        local_indices = topk_indices[token_idx].to(torch.long)
        valid = local_indices >= 0
        if bool(valid.any().item()):
            req_idx = int(token_to_req_indices[token_idx].item())
            selected_local = local_indices[valid]
            block_indices = selected_local // block_size
            block_offsets = selected_local % block_size
            block_numbers = block_table[req_idx].index_select(0, block_indices)
            slot_ids = block_numbers.to(torch.long) * block_size + block_offsets
            global_topk_indices[token_idx, valid] = slot_ids.to(global_topk_indices.dtype)
        valid_count = int(valid.sum().item())
        topk_lens[token_idx] = (
            valid_count if bool(is_valid_token[token_idx].item()) else 0
        )
    return global_topk_indices, topk_lens
""",
        """def _musa_compute_global_topk_indices_and_lens_fallback(
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = topk_indices.shape[0]
    local_indices = topk_indices[:num_tokens].to(torch.long)
    valid = local_indices >= 0
    safe_local_indices = local_indices.clamp(min=0)
    req_indices = token_to_req_indices[:num_tokens].to(torch.long).clamp(
        0, block_table.shape[0] - 1
    )
    block_indices = torch.div(
        safe_local_indices, block_size, rounding_mode="floor"
    ).clamp(0, block_table.shape[1] - 1)
    block_offsets = safe_local_indices.remainder(block_size)
    block_numbers = block_table[
        req_indices.unsqueeze(-1),
        block_indices,
    ].to(torch.long)
    slot_ids = block_numbers * block_size + block_offsets
    global_topk_indices = torch.where(
        valid,
        slot_ids.to(topk_indices.dtype),
        torch.full_like(topk_indices[:num_tokens], -1),
    )
    valid_counts = valid.sum(dim=-1).to(torch.int32)
    topk_lens = torch.where(
        is_valid_token[:num_tokens].to(torch.bool),
        valid_counts,
        torch.zeros_like(valid_counts),
    )
    return global_topk_indices, topk_lens
""",
    ),
]

RELOAD_AFTER_PATCH = [
    "__TARGET_MODULE__",
    "vllm.v1.attention.ops.deepseek_v4_ops",
]
