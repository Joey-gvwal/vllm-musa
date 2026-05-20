# SPDX-License-Identifier: Apache-2.0
"""TileLang kernels for DeepSeek-V4 MUSA JIT helpers."""

import tilelang
import tilelang.language as T

from .kernel_common import (
    _patch_tilelang_musa_wrapper,
    _tilelang_musa_pass_configs,
)


_patch_tilelang_musa_wrapper()

HIDDEN_SIZE = 512
NOPE_DIM = 448
ROPE_DIM = 64
HALF_ROPE_DIM = ROPE_DIM // 2
SCALE_DIM = NOPE_DIM // 64
TOKEN_VALUE_BYTES = NOPE_DIM + ROPE_DIM * 2
TOKEN_SCALE_BYTES = SCALE_DIM + 1
FP8_MAX = 448.0


def _warp_reduce_sum(value):
    mask = T.tvm_warp_activemask()
    value += T.tvm_warp_shuffle_down(mask, value, 16, 32, 32)
    value += T.tvm_warp_shuffle_down(mask, value, 8, 32, 32)
    value += T.tvm_warp_shuffle_down(mask, value, 4, 32, 32)
    value += T.tvm_warp_shuffle_down(mask, value, 2, 32, 32)
    value += T.tvm_warp_shuffle_down(mask, value, 1, 32, 32)
    return T.tvm_warp_shuffle(mask, value, 0, 32, 32)


def _warp_reduce_max(value):
    mask = T.tvm_warp_activemask()
    value = T.max(value, T.tvm_warp_shuffle_down(mask, value, 16, 32, 32))
    value = T.max(value, T.tvm_warp_shuffle_down(mask, value, 8, 32, 32))
    value = T.max(value, T.tvm_warp_shuffle_down(mask, value, 4, 32, 32))
    value = T.max(value, T.tvm_warp_shuffle_down(mask, value, 2, 32, 32))
    value = T.max(value, T.tvm_warp_shuffle_down(mask, value, 1, 32, 32))
    return T.tvm_warp_shuffle(mask, value, 0, 32, 32)


@tilelang.jit(target="musa", pass_configs=_tilelang_musa_pass_configs(tilelang))
def qnorm_rope_kernel():
    num_tokens = T.dynamic("num_tokens")
    num_heads = T.dynamic("num_heads")
    num_positions = T.dynamic("num_positions")
    threads = 256
    warps_per_cta = threads // 32

    @T.prim_func
    def _qnorm_rope_kernel(
        q: T.Tensor((num_tokens, num_heads, HIDDEN_SIZE), T.bfloat16),
        out: T.Tensor((num_tokens, num_heads, HIDDEN_SIZE), T.bfloat16),
        cos_sin_cache: T.Tensor((num_positions, ROPE_DIM), T.float32),
        positions: T.Tensor((num_tokens,), T.int64),
        eps: T.float32,
    ):
        with T.Kernel(num_tokens, num_heads, threads=threads) as (
            token_id,
            head_id,
        ):
            tx = T.get_thread_binding()
            lane = tx % 32
            warp = tx // 32
            partial_sumsq = T.alloc_local((1,), T.float32)
            warp_sumsq = T.alloc_shared((warps_per_cta,), T.float32)

            partial_sumsq[0] = 0.0
            for col_base in T.serial(0, HIDDEN_SIZE, threads):
                col = col_base + tx
                if col < HIDDEN_SIZE:
                    value = T.cast(q[token_id, head_id, col], T.float32)
                    partial_sumsq[0] += value * value

            partial_sumsq[0] = _warp_reduce_sum(partial_sumsq[0])
            if lane == 0:
                warp_sumsq[warp] = partial_sumsq[0]
            T.sync_threads()

            partial_sumsq[0] = T.if_then_else(
                tx < warps_per_cta, warp_sumsq[tx], 0.0
            )
            if warp == 0:
                partial_sumsq[0] = _warp_reduce_sum(partial_sumsq[0])
                if lane == 0:
                    warp_sumsq[0] = T.rsqrt(
                        partial_sumsq[0] / float(HIDDEN_SIZE) + eps
                    )
            T.sync_threads()

            for col_base in T.serial(0, NOPE_DIM, threads):
                col = col_base + tx
                if col < NOPE_DIM:
                    value = T.cast(q[token_id, head_id, col], T.float32)
                    out[token_id, head_id, col] = T.cast(
                        value * warp_sumsq[0], T.bfloat16
                    )

            if tx < HALF_ROPE_DIM:
                pos = positions[token_id]
                even_col = NOPE_DIM + tx * 2
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
                s = cos_sin_cache[pos, HALF_ROPE_DIM + tx]
                out[token_id, head_id, even_col] = T.cast(
                    even * c - odd * s, T.bfloat16
                )
                out[token_id, head_id, odd_col] = T.cast(
                    even * s + odd * c, T.bfloat16
                )

    return _qnorm_rope_kernel


@tilelang.jit(target="musa", pass_configs=_tilelang_musa_pass_configs(tilelang))
def kv_rope_pack_kernel():
    num_tokens = T.dynamic("num_tokens")
    num_pages = T.dynamic("num_pages")
    page_bytes = T.dynamic("page_bytes")
    page_u32 = T.dynamic("page_u32")
    num_positions = T.dynamic("num_positions")
    threads = 256
    tile_dim = 64
    rope_pack_elems = 2

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

    @T.prim_func
    def _kv_rope_pack_kernel(
        kv: T.Tensor((num_tokens, HIDDEN_SIZE), T.bfloat16),
        cache_u8: T.Tensor((num_pages, page_bytes), T.uint8),
        cache_u32: T.Tensor((num_pages, page_u32), T.uint32),
        slot_mapping: T.Tensor((num_tokens,), T.int64),
        positions: T.Tensor((num_tokens,), T.int64),
        cos_sin_cache: T.Tensor((num_positions, ROPE_DIM), T.float32),
        block_size: T.int32,
    ):
        with T.Kernel(num_tokens, threads=threads) as token_id:
            tx = T.get_thread_binding()
            lane = tx % 32
            warp = tx // 32
            loc = slot_mapping[token_id]

            if loc >= 0:
                loc_i32 = T.Cast("int32", loc)
                page_idx = loc_i32 // block_size
                token_offset = loc_i32 % block_size

                if warp < SCALE_DIM:
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

                    tile_amax[0] = _warp_reduce_max(local_amax[0])
                    scale_byte, inv_scale = pow2_scale_byte_and_inv(
                        tile_amax[0] / FP8_MAX
                    )
                    if lane == 0:
                        cache_u8[
                            page_idx,
                            block_size * TOKEN_VALUE_BYTES
                            + token_offset * TOKEN_SCALE_BYTES
                            + warp,
                        ] = scale_byte
                    tile_offset = token_offset * TOKEN_VALUE_BYTES + elem_base
                    for vec in T.vectorized(2):
                        cache_u8[
                            page_idx,
                            tile_offset + vec,
                        ] = T.reinterpret(
                            "uint8",
                            T.Cast(
                                "float8_e4m3fn",
                                T.clamp(fvals[vec] * inv_scale, -FP8_MAX, FP8_MAX),
                            ),
                        )
                else:
                    pos = positions[token_id]
                    elem = lane * rope_pack_elems
                    even_col = NOPE_DIM + elem
                    odd_col = even_col + 1
                    even = T.cast(kv[token_id, even_col], T.float32)
                    odd = T.cast(kv[token_id, odd_col], T.float32)
                    pair_idx = elem // 2
                    c = cos_sin_cache[pos, pair_idx]
                    s = cos_sin_cache[pos, HALF_ROPE_DIM + pair_idx]
                    rope_even = T.cast(even * c - odd * s, T.bfloat16)
                    rope_odd = T.cast(even * s + odd * c, T.bfloat16)
                    lo = T.reinterpret("uint16", rope_even)
                    hi = T.reinterpret("uint16", rope_odd)
                    rope_offset_u32 = (
                        token_offset * TOKEN_VALUE_BYTES + NOPE_DIM
                    ) // (2 * rope_pack_elems) + lane
                    cache_u32[page_idx, rope_offset_u32] = (
                        T.Cast("uint32", lo) | (T.Cast("uint32", hi) << 16)
                    )
                    if lane == 0:
                        cache_u8[
                            page_idx,
                            block_size * TOKEN_VALUE_BYTES
                            + token_offset * TOKEN_SCALE_BYTES
                            + SCALE_DIM,
                        ] = T.Cast("uint8", 0)

    return _kv_rope_pack_kernel
