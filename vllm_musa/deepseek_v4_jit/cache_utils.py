# SPDX-License-Identifier: Apache-2.0
"""TileLang-backed DeepSeek-V4 sparse MLA cache utility helpers."""

from __future__ import annotations

import os

import torch

_HIDDEN_SIZE = 512
_TOKEN_FP8_DIM = 448
_TOKEN_BF16_DIM = 64
_TOKEN_SCALE_DIM = 8
_TOKEN_DATA_SIZE = _TOKEN_FP8_DIM + _TOKEN_BF16_DIM * 2
_AUTO_DISABLED_REASON: str | None = None


def _is_musa_tensor(tensor: torch.Tensor | None) -> bool:
    return (
        tensor is not None
        and getattr(tensor, "device", None) is not None
        and tensor.device.type == "musa"
    )


def _cache_block_view(k_cache: torch.Tensor) -> torch.Tensor:
    block_stride = k_cache.stride(0)
    return k_cache.as_strided((k_cache.shape[0], block_stride), (block_stride, 1))


def _guard_tilelang_dequantize_and_gather_k_cache(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> tuple[bool, str]:
    tensors: tuple[torch.Tensor | None, ...] = (
        out,
        k_cache,
        seq_lens,
        gather_lens,
        block_table,
    )
    if not all(tensor is None or _is_musa_tensor(tensor) for tensor in tensors):
        return False, "all tensors must be on MUSA"
    devices = {tensor.device for tensor in tensors if tensor is not None}
    if len(devices) != 1:
        return False, "all tensors must be on the same MUSA device"
    if out.dtype != torch.bfloat16:
        return False, f"expected bf16 output workspace, got {out.dtype}"
    if k_cache.dtype != torch.uint8:
        return False, f"expected uint8 K cache, got {k_cache.dtype}"
    if seq_lens.dtype != torch.int32:
        return False, f"expected int32 seq_lens, got {seq_lens.dtype}"
    if gather_lens is not None and gather_lens.dtype != torch.int32:
        return False, f"expected int32 gather_lens, got {gather_lens.dtype}"
    if block_table.dtype != torch.int32:
        return False, f"expected int32 block_table, got {block_table.dtype}"
    if out.dim() != 3 or out.shape[-1] != _HIDDEN_SIZE:
        return False, f"expected out shape [reqs, tokens, 512], got {tuple(out.shape)}"
    if seq_lens.dim() != 1 or seq_lens.shape[0] != out.shape[0]:
        return False, "seq_lens must be 1D and match output request count"
    if gather_lens is not None and (
        gather_lens.dim() != 1 or gather_lens.shape[0] != out.shape[0]
    ):
        return False, "gather_lens must be 1D and match output request count"
    if block_table.dim() != 2 or block_table.shape[0] != out.shape[0]:
        return False, "block_table must be 2D and match output request count"
    if out.stride(-1) != 1 or not out.is_contiguous():
        return False, "out must be contiguous with contiguous last dimension"
    if k_cache.dim() < 2 or k_cache.stride(-1) != 1:
        return False, "k_cache must expose a byte-contiguous last dimension"
    cache_2d = _cache_block_view(k_cache)
    if not cache_2d.is_contiguous() or cache_2d.shape[1] % 4 != 0:
        return False, "k_cache block view must be contiguous and 4-byte aligned"
    expected_min_row = int(block_size) * (_TOKEN_DATA_SIZE + _TOKEN_SCALE_DIM)
    if cache_2d.shape[1] < expected_min_row:
        return False, (
            f"cache row bytes {cache_2d.shape[1]} < expected {expected_min_row} "
            f"for block_size={block_size}"
        )
    if not seq_lens.is_contiguous():
        return False, "seq_lens must be contiguous"
    if gather_lens is not None and not gather_lens.is_contiguous():
        return False, "gather_lens must be contiguous"
    if not block_table.is_contiguous():
        return False, "block_table must be contiguous"
    if int(block_size) <= 0:
        return False, f"block_size must be positive, got {block_size}"
    if int(offset) < 0 or int(offset) >= out.shape[1]:
        return False, f"offset {offset} is outside output token dimension {out.shape[1]}"
    return True, ""


def _combined_topk_width(topk: int, window_size: int) -> int:
    alignment = 128
    return ((int(topk) + int(window_size) + alignment - 1) // alignment) * alignment


def _guard_tilelang_combine_topk_swa_indices(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
) -> tuple[bool, str]:
    tensors = (topk_indices, query_start_loc, seq_lens, gather_lens)
    if not all(_is_musa_tensor(tensor) for tensor in tensors):
        return False, "all tensors must be on MUSA"
    devices = {tensor.device for tensor in tensors}
    if len(devices) != 1:
        return False, "all tensors must be on the same MUSA device"
    for name, tensor in (
        ("topk_indices", topk_indices),
        ("query_start_loc", query_start_loc),
        ("seq_lens", seq_lens),
        ("gather_lens", gather_lens),
    ):
        if tensor.dtype != torch.int32:
            return False, f"expected int32 {name}, got {tensor.dtype}"
        if not tensor.is_contiguous():
            return False, f"{name} must be contiguous"
    if topk_indices.dim() != 2:
        return False, f"topk_indices must be 2D, got {tuple(topk_indices.shape)}"
    if query_start_loc.dim() != 1:
        return False, "query_start_loc must be 1D"
    if seq_lens.dim() != 1 or gather_lens.dim() != 1:
        return False, "seq_lens and gather_lens must be 1D"
    if query_start_loc.shape[0] != seq_lens.shape[0] + 1:
        return False, "query_start_loc length must equal num_reqs + 1"
    if gather_lens.shape[0] != seq_lens.shape[0]:
        return False, "gather_lens must match seq_lens length"
    if topk_indices.shape[1] < int(topk):
        return False, (
            f"topk_indices width {topk_indices.shape[1]} is smaller than topk={topk}"
        )
    if int(window_size) <= 0 or int(compress_ratio) <= 0:
        return False, (
            f"window_size and compress_ratio must be positive, got "
            f"{window_size}, {compress_ratio}"
        )
    if int(topk) < 0 or int(M) < 0 or int(N) < 0:
        return False, f"topk, M, and N must be non-negative, got {topk}, {M}, {N}"
    return True, ""


def try_tilelang_dequantize_and_gather_k_cache(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> tuple[bool, str]:
    """Try the TileLang gather/dequant path and report whether it handled the call."""

    global _AUTO_DISABLED_REASON
    mode = (
        os.environ.get("VLLM_MUSA_DEEPSEEK_V4_DEQUANT_GATHER_IMPL", "auto")
        .strip()
        .lower()
    )
    if mode in {"", "torch", "fallback", "0", "off"}:
        return False, "disabled by VLLM_MUSA_DEEPSEEK_V4_DEQUANT_GATHER_IMPL"
    if mode == "auto" and _AUTO_DISABLED_REASON is not None:
        return False, _AUTO_DISABLED_REASON

    supported, reason = _guard_tilelang_dequantize_and_gather_k_cache(
        out, k_cache, seq_lens, gather_lens, block_table, block_size, offset
    )
    if not supported:
        if mode in {"tilelang", "jit", "force"}:
            raise NotImplementedError(reason)
        return False, reason

    try:
        from vllm_musa.deepseek_v4_jit.tilelang_kernels import (
            dequantize_gather_k_cache_kernel,
        )

        cache_2d = _cache_block_view(k_cache)
        gather_lens_arg = seq_lens if gather_lens is None else gather_lens
        dequantize_gather_k_cache_kernel()(
            out,
            out.view(torch.uint32),
            cache_2d,
            cache_2d.view(torch.uint32),
            seq_lens,
            gather_lens_arg,
            block_table,
            int(block_size),
            int(offset),
            int(gather_lens is not None),
        )
    except Exception as exc:
        if mode in {"tilelang", "jit", "force"}:
            raise
        _AUTO_DISABLED_REASON = f"{type(exc).__name__}: {exc}"
        return False, _AUTO_DISABLED_REASON
    return True, "tilelang"


def try_tilelang_combine_topk_swa_indices(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
) -> tuple[bool, tuple[torch.Tensor, torch.Tensor] | None, str]:
    """Try the TileLang combine-topk/SWA path for DeepSeek-V4 sparse prefill."""

    mode = (
        os.environ.get("VLLM_MUSA_DEEPSEEK_V4_COMBINE_TOPK_SWA_IMPL", "auto")
        .strip()
        .lower()
    )
    if mode in {"", "torch", "fallback", "0", "off"}:
        return False, None, "disabled by VLLM_MUSA_DEEPSEEK_V4_COMBINE_TOPK_SWA_IMPL"

    supported, reason = _guard_tilelang_combine_topk_swa_indices(
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
    if not supported:
        if mode in {"tilelang", "jit", "force"}:
            raise NotImplementedError(reason)
        return False, None, reason

    combined_topk = _combined_topk_width(int(topk), int(window_size))
    combined_indices = torch.full(
        (topk_indices.shape[0], combined_topk),
        -1,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    combined_lens = torch.empty(
        topk_indices.shape[0],
        dtype=torch.int32,
        device=topk_indices.device,
    )
    if topk_indices.shape[0] == 0:
        return True, (combined_indices, combined_lens), "empty"

    try:
        from vllm_musa.deepseek_v4_jit.tilelang_kernels import (
            combine_topk_swa_indices_kernel,
        )

        combine_topk_swa_indices_kernel(
            int(topk_indices.shape[1]),
            int(topk),
            int(window_size),
            int(compress_ratio),
            int(combined_topk),
        )(
            combined_indices,
            combined_lens,
            topk_indices,
            query_start_loc,
            seq_lens,
            gather_lens,
            int(M),
            int(N),
        )
    except Exception:
        if mode in {"tilelang", "jit", "force"}:
            raise
        return False, None, "tilelang combine_topk_swa_indices failed"
    return True, (combined_indices, combined_lens), "tilelang"
