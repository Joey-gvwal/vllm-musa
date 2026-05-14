# SPDX-License-Identifier: Apache-2.0
"""TileLang-backed DeepSeek-V4 QNorm/RoPE/KV-cache insert helper.

This module is intentionally imported lazily by the DeepSeek-V4 source patch.
TileLang is optional in the current remote images, so import or compile failures
must leave the existing torch correctness fallback available.
"""

from __future__ import annotations

import os
from functools import lru_cache

import torch

from .kernel_common import _tilelang_musa_pass_configs


_HIDDEN_SIZE = 512
_NOPE_DIM = 448
_ROPE_DIM = 64
_HALF_ROPE_DIM = _ROPE_DIM // 2
_SCALE_DIM = _NOPE_DIM // 64
_TOKEN_VALUE_BYTES = _NOPE_DIM + _ROPE_DIM * 2
_TOKEN_SCALE_BYTES = _SCALE_DIM + 1
_FP8_MAX = 448.0
_AUTO_DISABLED_REASON: str | None = None


def _is_musa_tensor(tensor: torch.Tensor) -> bool:
    return getattr(tensor, "device", None) is not None and tensor.device.type == "musa"


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.int32:
        return "int32"
    if dtype == torch.int64:
        return "int64"
    return str(dtype).split(".")[-1]


def _guard_tilelang_qnorm_rope_kv_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    k_cache_2d: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    block_size: int,
) -> tuple[bool, str]:
    tensors = (q, kv, k_cache_2d, slot_mapping, positions, cos_sin_cache)
    if not all(_is_musa_tensor(tensor) for tensor in tensors):
        return False, "all tensors must be on MUSA"
    if len({tensor.device for tensor in tensors}) != 1:
        return False, "all tensors must be on the same MUSA device"
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        return False, f"expected bf16 q/kv, got q={q.dtype} kv={kv.dtype}"
    if k_cache_2d.dtype != torch.uint8:
        return False, f"expected uint8 cache, got {k_cache_2d.dtype}"
    if cos_sin_cache.dtype != torch.float32:
        return False, f"expected float32 cos_sin_cache, got {cos_sin_cache.dtype}"
    if positions.dtype not in (torch.int32, torch.int64):
        return False, f"unsupported positions dtype {positions.dtype}"
    if slot_mapping.dtype not in (torch.int32, torch.int64):
        return False, f"unsupported slot_mapping dtype {slot_mapping.dtype}"
    if positions.dtype != torch.int64 or slot_mapping.dtype != torch.int64:
        return False, "TileLang path currently requires int64 positions and slots"
    if q.dim() != 3 or q.shape[-1] != _HIDDEN_SIZE:
        return False, f"expected q shape [tokens, heads, 512], got {tuple(q.shape)}"
    if kv.dim() != 2 or kv.shape[-1] != _HIDDEN_SIZE:
        return False, f"expected kv shape [tokens, 512], got {tuple(kv.shape)}"
    if q.shape[0] != kv.shape[0]:
        return False, f"q/kv token mismatch: q={q.shape[0]} kv={kv.shape[0]}"
    if positions.dim() != 1 or positions.shape[0] != kv.shape[0]:
        return False, "positions must be 1D and match token count"
    if slot_mapping.dim() != 1 or slot_mapping.shape[0] < kv.shape[0]:
        return False, "slot_mapping must be 1D and cover every token"
    if cos_sin_cache.dim() != 2 or cos_sin_cache.shape[-1] != _ROPE_DIM:
        return False, (
            "cos_sin_cache must have shape [positions, 64], got "
            f"{tuple(cos_sin_cache.shape)}"
        )
    if not q.is_contiguous() or not kv.is_contiguous():
        return False, "q and kv must be contiguous"
    if cos_sin_cache.stride(-1) != 1:
        return False, "cos_sin_cache must have contiguous last dimension"
    if not k_cache_2d.is_contiguous() or k_cache_2d.dim() != 2:
        return False, "k_cache_2d must be a contiguous 2D uint8 tensor"
    expected_cache_row = int(block_size) * (_TOKEN_VALUE_BYTES + _TOKEN_SCALE_BYTES)
    if k_cache_2d.shape[1] != expected_cache_row:
        return False, (
            f"cache row bytes {k_cache_2d.shape[1]} != expected {expected_cache_row} "
            f"for block_size={block_size}"
        )
    if int(block_size) <= 0:
        return False, f"block_size must be positive, got {block_size}"
    if not positions.is_contiguous() or not slot_mapping.is_contiguous():
        return False, "positions and slot_mapping must be contiguous"
    return True, ""


@lru_cache(maxsize=None)
def _tilelang_qnorm_rope_kernel(positions_dtype: str):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    num_heads = T.dynamic("num_heads")
    num_positions = T.dynamic("num_positions")
    threads = 256
    warps_per_cta = threads // 32
    tl_positions_dtype = T.int32 if positions_dtype == "int32" else T.int64

    @tilelang.jit(pass_configs=_tilelang_musa_pass_configs(tilelang))
    def qnorm_rope_kernel(
        q,
        out,
        cos_sin_cache,
        positions,
        eps,
    ):
        num_tokens = T.dynamic("num_tokens")
        num_heads = T.dynamic("num_heads")
        num_positions = T.dynamic("num_positions")
        q: T.Tensor[[num_tokens, num_heads, _HIDDEN_SIZE], T.bfloat16]
        out: T.Tensor[[num_tokens, num_heads, _HIDDEN_SIZE], T.bfloat16]
        cos_sin_cache: T.Tensor[[num_positions, _ROPE_DIM], T.float32]
        positions: T.Tensor[[num_tokens], tl_positions_dtype]
        with T.Kernel(num_tokens, num_heads, threads=threads) as (token_id, head_id):
            tx = T.get_thread_binding()
            lane = tx % 32
            warp = tx // 32
            partial_sumsq = T.alloc_local((1,), T.float32)
            warp_sumsq = T.alloc_shared((warps_per_cta,), T.float32)

            partial_sumsq[0] = 0.0
            for col_base in T.serial(0, _HIDDEN_SIZE, threads):
                col = col_base + tx
                if col < _HIDDEN_SIZE:
                    value = T.cast(q[token_id, head_id, col], T.float32)
                    partial_sumsq[0] += value * value

            partial_sumsq[0] = T.warp_reduce_sum(partial_sumsq[0])
            if lane == 0:
                warp_sumsq[warp] = partial_sumsq[0]
            T.sync_threads()

            partial_sumsq[0] = T.if_then_else(tx < warps_per_cta, warp_sumsq[tx], 0.0)
            if warp == 0:
                partial_sumsq[0] = T.warp_reduce_sum(partial_sumsq[0])
                if lane == 0:
                    warp_sumsq[0] = T.rsqrt(partial_sumsq[0] / float(_HIDDEN_SIZE) + eps)
            T.sync_threads()

            for col_base in T.serial(0, _NOPE_DIM, threads):
                col = col_base + tx
                if col < _NOPE_DIM:
                    value = T.cast(q[token_id, head_id, col], T.float32)
                    out[token_id, head_id, col] = T.cast(
                        value * warp_sumsq[0], T.bfloat16
                    )

            if tx < _HALF_ROPE_DIM:
                pos = positions[token_id]
                even_col = _NOPE_DIM + tx * 2
                odd_col = even_col + 1
                even = (
                    T.cast(q[token_id, head_id, even_col], T.float32)
                    * warp_sumsq[0]
                )
                odd = (
                    T.cast(q[token_id, head_id, odd_col], T.float32)
                    * warp_sumsq[0]
                )
                c = cos_sin_cache[pos, tx]
                s = cos_sin_cache[pos, _HALF_ROPE_DIM + tx]
                out[token_id, head_id, even_col] = T.cast(even * c - odd * s, T.bfloat16)
                out[token_id, head_id, odd_col] = T.cast(even * s + odd * c, T.bfloat16)

    return qnorm_rope_kernel


@lru_cache(maxsize=None)
def _tilelang_kv_rope_pack_kernel(
    block_size: int,
    page_bytes: int,
    positions_dtype: str,
    slot_dtype: str,
):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    num_pages = T.dynamic("num_pages")
    num_positions = T.dynamic("num_positions")
    threads = 256
    tile_dim = 64
    rope_pack_elems = 2
    page_size_is_pow2 = block_size > 0 and (block_size & (block_size - 1)) == 0
    page_size_shift = block_size.bit_length() - 1 if page_size_is_pow2 else 0
    page_size_mask = block_size - 1
    tl_positions_dtype = T.int32 if positions_dtype == "int32" else T.int64
    tl_slot_dtype = T.int32 if slot_dtype == "int32" else T.int64

    def pow2_scale_byte_and_inv(value):
        clamped = T.max(value, 1.0e-4)
        bits = T.reinterpret("uint32", clamped)
        exp = (bits >> 23) & 0xFF
        man_bits = bits & ((1 << 23) - 1)
        exp_scale = T.Cast("int32", exp - 127 + T.if_then_else(man_bits != 0, 1, 0))
        scale_byte = T.Cast("uint8", exp_scale + 127)
        inv_scale = T.reinterpret("float32", (127 - exp_scale) << 23)
        return scale_byte, inv_scale

    def abs_f32(value):
        return T.if_then_else(value < 0.0, -value, value)

    def page_index_and_offset(loc):
        if page_size_is_pow2:
            return loc >> page_size_shift, loc & page_size_mask
        return loc // block_size, loc % block_size

    @tilelang.jit(pass_configs=_tilelang_musa_pass_configs(tilelang))
    def kv_rope_pack_kernel(
        kv,
        cache_u8,
        cache_fp8,
        cache_u32,
        slot_mapping,
        positions,
        cos_sin_cache,
    ):
        num_tokens = T.dynamic("num_tokens")
        num_pages = T.dynamic("num_pages")
        num_positions = T.dynamic("num_positions")
        kv: T.Tensor[[num_tokens, _HIDDEN_SIZE], T.bfloat16]
        cache_u8: T.Tensor[[num_pages, page_bytes], T.uint8]
        cache_fp8: T.Tensor[[num_pages, page_bytes], T.float8_e4m3fn]
        cache_u32: T.Tensor[[num_pages, page_bytes // 4], T.uint32]
        slot_mapping: T.Tensor[[num_tokens], tl_slot_dtype]
        positions: T.Tensor[[num_tokens], tl_positions_dtype]
        cos_sin_cache: T.Tensor[[num_positions, _ROPE_DIM], T.float32]
        with T.Kernel(num_tokens, threads=threads) as token_id:
            tx = T.get_thread_binding()
            lane = tx % 32
            warp = tx // 32
            loc = slot_mapping[token_id]

            if loc >= 0:
                page_idx, token_offset = page_index_and_offset(loc)
                page_idx_i64 = T.Cast("int64", page_idx)

                if warp < _SCALE_DIM:
                    vals = T.alloc_local((2,), dtype=T.bfloat16)
                    fvals = T.alloc_local((2,), dtype=T.float32)
                    local_amax = T.alloc_local((1,), dtype=T.float32)
                    tile_amax = T.alloc_local((1,), dtype=T.float32)
                    elem_base = warp * tile_dim + lane * 2

                    local_amax[0] = 0.0
                    for vec in T.vectorized(2):
                        vals[vec] = kv[token_id, elem_base + vec]
                        fvals[vec] = T.cast(vals[vec], T.float32)
                        local_amax[0] = T.max(local_amax[0], abs_f32(fvals[vec]))

                    tile_amax[0] = T.warp_reduce_max(local_amax[0])
                    scale_byte, inv_scale = pow2_scale_byte_and_inv(
                        tile_amax[0] / _FP8_MAX
                    )
                    if lane == 0:
                        cache_u8[
                            page_idx_i64,
                            T.Cast("int64", block_size * _TOKEN_VALUE_BYTES)
                            + T.Cast("int64", token_offset * _TOKEN_SCALE_BYTES)
                            + T.Cast("int64", warp),
                        ] = scale_byte
                    tile_offset = token_offset * _TOKEN_VALUE_BYTES + elem_base
                    for vec in T.vectorized(2):
                        cache_fp8[
                            page_idx_i64,
                            T.Cast("int64", tile_offset + vec),
                        ] = T.clamp(fvals[vec] * inv_scale, -_FP8_MAX, _FP8_MAX)
                else:
                    pos = positions[token_id]
                    elem = lane * rope_pack_elems
                    even_col = _NOPE_DIM + elem
                    odd_col = even_col + 1
                    even = T.cast(kv[token_id, even_col], T.float32)
                    odd = T.cast(kv[token_id, odd_col], T.float32)
                    pair_idx = elem // 2
                    c = cos_sin_cache[pos, pair_idx]
                    s = cos_sin_cache[pos, _HALF_ROPE_DIM + pair_idx]
                    rope_even = T.cast(even * c - odd * s, T.bfloat16)
                    rope_odd = T.cast(even * s + odd * c, T.bfloat16)
                    lo = T.reinterpret("uint16", rope_even)
                    hi = T.reinterpret("uint16", rope_odd)
                    rope_offset_u32 = (
                        token_offset * _TOKEN_VALUE_BYTES + _NOPE_DIM
                    ) // (2 * rope_pack_elems) + lane
                    cache_u32[page_idx_i64, T.Cast("int64", rope_offset_u32)] = (
                        T.Cast("uint32", lo) | (T.Cast("uint32", hi) << 16)
                    )
                    if lane == 0:
                        cache_u8[
                            page_idx_i64,
                            T.Cast("int64", block_size * _TOKEN_VALUE_BYTES)
                            + T.Cast("int64", token_offset * _TOKEN_SCALE_BYTES)
                            + T.Cast("int64", _SCALE_DIM),
                        ] = T.Cast("uint8", 0)

    return kv_rope_pack_kernel


def try_tilelang_qnorm_rope_kv_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    k_cache_2d: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    block_size: int,
) -> tuple[bool, str]:
    """Try the TileLang path and report whether it handled the call."""
    global _AUTO_DISABLED_REASON
    mode = (
        os.environ.get("VLLM_MUSA_DEEPSEEK_V4_QNORM_ROPE_KV_INSERT_IMPL", "torch")
        .strip()
        .lower()
    )
    if mode in {"torch", "fallback", "0", "off"}:
        return False, "disabled by VLLM_MUSA_DEEPSEEK_V4_QNORM_ROPE_KV_INSERT_IMPL"
    if mode == "auto" and _AUTO_DISABLED_REASON is not None:
        return False, _AUTO_DISABLED_REASON

    supported, reason = _guard_tilelang_qnorm_rope_kv_insert(
        q, kv, k_cache_2d, slot_mapping, positions, cos_sin_cache, block_size
    )
    if not supported:
        if mode in {"tilelang", "jit", "force"}:
            raise NotImplementedError(reason)
        return False, reason

    try:
        from vllm_musa.deepseek_v4_jit.tilelang_kernels import (
            kv_rope_pack_kernel,
            qnorm_rope_kernel,
        )

        q_out = torch.empty_like(q)
        qnorm_rope_kernel(
            q,
            q_out,
            cos_sin_cache,
            positions,
            float(eps),
        )
        kv_rope_pack_kernel(
            kv,
            k_cache_2d,
            k_cache_2d.view(torch.float8_e4m3fn),
            k_cache_2d.view(torch.uint32),
            slot_mapping[: kv.shape[0]],
            positions,
            cos_sin_cache,
            int(block_size),
        )
        q.copy_(q_out)
    except Exception as exc:
        if mode in {"tilelang", "jit", "force"}:
            raise
        _AUTO_DISABLED_REASON = f"{type(exc).__name__}: {exc}"
        return False, _AUTO_DISABLED_REASON
    return True, "tilelang"
