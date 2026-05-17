# SPDX-License-Identifier: Apache-2.0
"""TileLang kernels for DeepSeek-V4 MUSA JIT helpers."""

from functools import lru_cache

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


def _abs_f32(value):
    return T.if_then_else(value < 0.0, -value, value)


def _index_dtype(dtype_name: str):
    normalized = dtype_name.strip().lower()
    if normalized == "int32":
        return T.int32
    if normalized == "int64":
        return T.int64
    raise ValueError(f"unsupported index dtype {dtype_name!r}")


@lru_cache(maxsize=None)
def inv_rope_fp8_quant_kernel(heads_per_group: int, threads: int = 128):
    """Return a TileLang kernel for DeepSeek-V4 inverse-RoPE + FP8 quant.

    The kernel writes into storage-shaped buffers [G, T, D] so callers can
    return the same transposed [T, G, D] contract as upstream vLLM.
    """

    d = heads_per_group * HIDDEN_SIZE
    num_scale_blocks = d // 128
    warps_per_cta = threads // 32

    @tilelang.jit(target="musa", pass_configs=_tilelang_musa_pass_configs(tilelang))
    def _inv_rope_fp8_quant_kernel():
        num_tokens = T.dynamic("num_tokens")
        num_heads = T.dynamic("num_heads")
        num_groups = T.dynamic("num_groups")
        num_positions = T.dynamic("num_positions")

        @T.prim_func
        def _kernel(
            o: T.Tensor((num_tokens, num_heads, HIDDEN_SIZE), T.bfloat16),
            positions: T.Tensor((num_tokens,), T.int64),
            cos_sin_cache: T.Tensor((num_positions, ROPE_DIM), T.float32),
            out_u8: T.Tensor((num_groups, num_tokens, d), T.uint8),
            scale_out: T.Tensor((num_groups, num_tokens, num_scale_blocks), T.float32),
        ):
            with T.Kernel(num_tokens, num_groups, num_scale_blocks, threads=threads) as (
                token_id,
                group_id,
                block_id,
            ):
                tx = T.get_thread_binding()
                lane = tx % 32
                warp = tx // 32
                warp_max = T.alloc_shared((warps_per_cta,), T.float32)
                scale_holder = T.alloc_shared((1,), T.float32)

                flat_dim = block_id * 128 + tx
                head_in_group = flat_dim // HIDDEN_SIZE
                dim = flat_dim - head_in_group * HIDDEN_SIZE
                head_id = group_id * heads_per_group + head_in_group
                pos = positions[token_id]

                value = T.alloc_var(T.float32)
                value = T.cast(o[token_id, head_id, dim], T.float32)
                if dim >= NOPE_DIM:
                    rope_idx = dim - NOPE_DIM
                    pair_idx = rope_idx // 2
                    even_col = NOPE_DIM + pair_idx * 2
                    odd_col = even_col + 1
                    even = T.cast(o[token_id, head_id, even_col], T.float32)
                    odd = T.cast(o[token_id, head_id, odd_col], T.float32)
                    c = cos_sin_cache[pos, pair_idx]
                    s = cos_sin_cache[pos, HALF_ROPE_DIM + pair_idx]
                    value = T.if_then_else(
                        (rope_idx % 2) == 0,
                        even * c + odd * s,
                        odd * c - even * s,
                    )

                local_max = _warp_reduce_max(_abs_f32(value))
                if lane == 0:
                    warp_max[warp] = local_max
                T.sync_threads()

                partial = T.if_then_else(tx < warps_per_cta, warp_max[tx], 0.0)
                if warp == 0:
                    block_max = _warp_reduce_max(partial)
                    if lane == 0:
                        raw_scale = T.max(block_max / FP8_MAX, 1.0e-10)
                        scale_holder[0] = T.exp2(T.ceil(T.log2(raw_scale)))
                T.sync_threads()

                inv_scale = 1.0 / scale_holder[0]
                quant = T.clamp(value * inv_scale, -FP8_MAX, FP8_MAX)
                out_u8[group_id, token_id, flat_dim] = T.reinterpret(
                    "uint8",
                    T.Cast("float8_e4m3fn", quant),
                )
                if tx == 0:
                    scale_out[group_id, token_id, block_id] = scale_holder[0]

        return _kernel

    return _inv_rope_fp8_quant_kernel()


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


@tilelang.jit(target="musa", pass_configs=_tilelang_musa_pass_configs(tilelang))
def dequantize_gather_k_cache_kernel():
    num_reqs = T.dynamic("num_reqs")
    out_tokens = T.dynamic("out_tokens")
    num_pages = T.dynamic("num_pages")
    page_bytes = T.dynamic("page_bytes")
    page_u32 = T.dynamic("page_u32")
    max_blocks_per_seq = T.dynamic("max_blocks_per_seq")
    threads = 256

    @T.prim_func
    def _dequantize_gather_k_cache_kernel(
        out: T.Tensor((num_reqs, out_tokens, HIDDEN_SIZE), T.bfloat16),
        out_u32: T.Tensor((num_reqs, out_tokens, HIDDEN_SIZE // 2), T.uint32),
        cache_u8: T.Tensor((num_pages, page_bytes), T.uint8),
        cache_u32: T.Tensor((num_pages, page_u32), T.uint32),
        seq_lens: T.Tensor((num_reqs,), T.int32),
        gather_lens: T.Tensor((num_reqs,), T.int32),
        block_table: T.Tensor((num_reqs, max_blocks_per_seq), T.int32),
        block_size: T.int32,
        offset: T.int32,
        has_gather_lens: T.int32,
    ):
        with T.Kernel(num_reqs, out_tokens, threads=threads) as (
            req_id,
            gather_id,
        ):
            tx = T.get_thread_binding()
            seq_len = seq_lens[req_id]
            gather_len = T.if_then_else(
                has_gather_lens != 0,
                gather_lens[req_id],
                seq_len,
            )
            out_token = offset + gather_id

            if gather_id < gather_len and out_token < out_tokens:
                pos = seq_len - gather_len + gather_id
                block_in_seq = pos // block_size
                pos_in_block = pos - block_in_seq * block_size
                physical_block = block_table[req_id, block_in_seq]

                if physical_block >= 0 and physical_block < num_pages:
                    token_base = pos_in_block * TOKEN_VALUE_BYTES
                    scale_base = (
                        block_size * TOKEN_VALUE_BYTES
                        + pos_in_block * TOKEN_SCALE_BYTES
                    )

                    for col_base in T.serial(0, NOPE_DIM, threads):
                        col = col_base + tx
                        if col < NOPE_DIM:
                            qblock_id = col // 64
                            q = T.Cast(
                                "float32",
                                T.reinterpret(
                                    "float8_e4m3fn",
                                    cache_u8[physical_block, token_base + col],
                                ),
                            )
                            encoded_scale = T.Cast(
                                "float32",
                                cache_u8[physical_block, scale_base + qblock_id],
                            )
                            scale = T.exp2(encoded_scale - 127.0)
                            out[req_id, out_token, col] = T.Cast(
                                "bfloat16",
                                q * scale,
                            )

                    if tx < ROPE_DIM // 2:
                        out_u32[
                            req_id,
                            out_token,
                            NOPE_DIM // 2 + tx,
                        ] = cache_u32[
                            physical_block,
                            (token_base + NOPE_DIM) // 4 + tx,
                        ]

    return _dequantize_gather_k_cache_kernel


@tilelang.jit(target="musa", pass_configs=_tilelang_musa_pass_configs(tilelang))
def combine_topk_swa_indices_kernel(
    topk_width: int,
    topk: int,
    window_size: int,
    compress_ratio: int,
    combined_topk: int,
):
    num_tokens = T.dynamic("num_tokens")
    num_reqs = T.dynamic("num_reqs")
    num_query_locs = T.dynamic("num_query_locs")
    threads = 128

    @T.prim_func
    def _combine_topk_swa_indices_kernel(
        combined_indices: T.Tensor((num_tokens, combined_topk), T.int32),
        combined_lens: T.Tensor((num_tokens,), T.int32),
        topk_indices: T.Tensor((num_tokens, topk_width), T.int32),
        query_start_loc: T.Tensor((num_query_locs,), T.int32),
        seq_lens: T.Tensor((num_reqs,), T.int32),
        gather_lens: T.Tensor((num_reqs,), T.int32),
        M: T.int32,
        N: T.int32,
    ):
        with T.Kernel(num_reqs, num_tokens, threads=threads) as (
            req_id,
            token_offset,
        ):
            tx = T.get_thread_binding()

            base = query_start_loc[0]
            query_start = query_start_loc[req_id] - base
            query_end = query_start_loc[req_id + 1] - base
            query_len = query_end - query_start

            if token_offset < query_len:
                token_idx = query_start + token_offset
                seq_len = seq_lens[req_id]
                gather_len = gather_lens[req_id]
                pos = seq_len - query_len + token_offset
                gather_start = seq_len - gather_len
                topk_len = T.min((pos + 1) // compress_ratio, topk)
                swa_len = T.min(pos + 1, window_size)
                req_offset = M * req_id

                for col_base in T.serial(0, topk, threads):
                    col = col_base + tx
                    if col < topk_len and col < topk_width:
                        combined_indices[token_idx, col] = (
                            topk_indices[token_idx, col] + req_offset
                        )

                for col_base in T.serial(0, window_size, threads):
                    col = col_base + tx
                    out_col = topk_len + col
                    if col < swa_len and out_col < combined_topk:
                        combined_indices[token_idx, out_col] = (
                            req_offset
                            + N
                            + col
                            + pos
                            - swa_len
                            + 1
                            - gather_start
                        )

                if tx == 0:
                    combined_lens[token_idx] = topk_len + swa_len

    return _combine_topk_swa_indices_kernel


@tilelang.jit(target="musa", pass_configs=_tilelang_musa_pass_configs(tilelang))
def sparse_indexer_topk_rows_kernel(max_width: int, topk: int, score_stride: int):
    rows = T.dynamic("rows")
    threads = 128

    @T.prim_func
    def _sparse_indexer_topk_rows_kernel(
        scores: T.StridedTensor((rows, max_width), (score_stride, 1), T.float32),
        starts: T.Tensor((rows,), T.int32),
        ends: T.Tensor((rows,), T.int32),
        out: T.Tensor((rows, topk), T.int32),
    ):
        with T.Kernel(rows, threads=threads) as row_id:
            tx = T.get_thread_binding()
            selected = T.alloc_shared((max_width,), dtype=T.int32)
            thread_scores = T.alloc_shared((threads,), dtype=T.float32)
            thread_indices = T.alloc_shared((threads,), dtype=T.int32)
            local_best_score = T.alloc_local((1,), dtype=T.float32)
            local_best_idx = T.alloc_local((1,), dtype=T.int32)

            row_start = starts[row_id]
            row_end = ends[row_id]
            row_len = row_end - row_start

            for init_base in T.serial(0, max_width, threads):
                pos = init_base + tx
                if pos < max_width:
                    selected[pos] = 0
            T.sync_threads()

            for kth in T.serial(0, topk):
                local_best_score[0] = -3.4028234663852886e38
                local_best_idx[0] = -1

                if kth < row_len:
                    for pos_base in T.serial(0, max_width, threads):
                        pos = pos_base + tx
                        if (
                            pos >= row_start
                            and pos < row_end
                            and selected[pos] == 0
                        ):
                            score = scores[row_id, pos]
                            if score > local_best_score[0] or (
                                score == local_best_score[0]
                                and (
                                    local_best_idx[0] < 0
                                    or pos < local_best_idx[0]
                                )
                            ):
                                local_best_score[0] = score
                                local_best_idx[0] = pos

                thread_scores[tx] = local_best_score[0]
                thread_indices[tx] = local_best_idx[0]
                T.sync_threads()

                if tx == 0:
                    for reduce_idx in T.serial(1, threads):
                        other_score = thread_scores[reduce_idx]
                        other_idx = thread_indices[reduce_idx]
                        if other_idx >= 0 and (
                            other_score > thread_scores[0]
                            or (
                                other_score == thread_scores[0]
                                and (
                                    thread_indices[0] < 0
                                    or other_idx < thread_indices[0]
                                )
                            )
                        ):
                            thread_scores[0] = other_score
                            thread_indices[0] = other_idx

                    if thread_indices[0] >= 0:
                        selected[thread_indices[0]] = 1
                        out[row_id, kth] = thread_indices[0] - row_start
                    else:
                        out[row_id, kth] = -1
                T.sync_threads()

    return _sparse_indexer_topk_rows_kernel


@tilelang.jit(target="musa", pass_configs=_tilelang_musa_pass_configs(tilelang))
def hash_topk_softplus_sqrt_kernel(
    topk: int,
    input_tokens_dtype: str,
    hash_indices_dtype: str,
    renormalize: bool,
    apply_routed_scaling_factor: bool,
):
    """Return a graph-friendly DeepSeek-V4 hash-router top-k kernel."""

    num_tokens = T.dynamic("num_tokens")
    num_experts = T.dynamic("num_experts")
    vocab_size = T.dynamic("vocab_size")
    threads = 32
    tl_input_tokens_dtype = _index_dtype(input_tokens_dtype)
    tl_hash_indices_dtype = _index_dtype(hash_indices_dtype)

    @T.prim_func
    def _hash_topk_softplus_sqrt_kernel(
        gating_output: T.Tensor((num_tokens, num_experts), T.float32),
        input_tokens: T.Tensor((num_tokens,), tl_input_tokens_dtype),
        hash_indices_table: T.Tensor((vocab_size, topk), tl_hash_indices_dtype),
        topk_weights: T.Tensor((num_tokens, topk), T.float32),
        topk_indices: T.Tensor((num_tokens, topk), T.int64),
        token_expert_indices: T.Tensor((num_tokens, topk), T.int32),
        routed_scaling_factor: T.float32,
    ):
        with T.Kernel(num_tokens, threads=threads) as token_id:
            tx = T.get_thread_binding()
            score = T.alloc_local((1,), dtype=T.float32)
            expert_id = T.alloc_local((1,), dtype=T.int64)
            out_score = T.alloc_local((1,), dtype=T.float32)
            token = input_tokens[token_id]

            score[0] = 0.0
            expert_id[0] = 0
            out_score[0] = 0.0
            if tx < topk:
                expert_id[0] = T.cast(hash_indices_table[token, tx], T.int64)
                logit = T.cast(gating_output[token_id, expert_id[0]], T.float32)
                # Stable sqrt(softplus(x)) to avoid graph replay NaNs on wide logits.
                softplus = T.max(logit, 0.0) + T.log(1.0 + T.exp(-_abs_f32(logit)))
                score[0] = T.sqrt(T.max(softplus, 1.0e-20))

            denominator = _warp_reduce_sum(score[0])
            if tx < topk:
                out_score[0] = score[0]
                if renormalize:
                    out_score[0] = out_score[0] / T.max(denominator, 1.0e-20)
                if apply_routed_scaling_factor:
                    out_score[0] = out_score[0] * routed_scaling_factor
                topk_weights[token_id, tx] = out_score[0]
                topk_indices[token_id, tx] = expert_id[0]
                token_expert_indices[token_id, tx] = T.cast(expert_id[0], T.int32)

    return _hash_topk_softplus_sqrt_kernel


@tilelang.jit(target="musa", pass_configs=_tilelang_musa_pass_configs(tilelang))
def biased_topk_softplus_sqrt_256_kernel(
    topk: int,
    renormalize: bool,
    apply_routed_scaling_factor: bool,
):
    num_tokens = T.dynamic("num_tokens")
    num_experts = 256
    threads = 256

    @T.prim_func
    def _biased_topk_softplus_sqrt_256_kernel(
        gating_output: T.Tensor((num_tokens, num_experts), T.float32),
        correction_bias: T.Tensor((num_experts,), T.float32),
        topk_weights: T.Tensor((num_tokens, topk), T.float32),
        topk_indices: T.Tensor((num_tokens, topk), T.int64),
        token_expert_indices: T.Tensor((num_tokens, topk), T.int32),
        routed_scaling_factor: T.float32,
    ):
        with T.Kernel(num_tokens, threads=threads) as token_id:
            tx = T.get_thread_binding()
            selected_ids = T.alloc_shared((topk,), dtype=T.int32)
            selected_sum = T.alloc_shared((1,), dtype=T.float32)
            reduction_scores = T.alloc_shared((threads,), dtype=T.float32)
            reduction_weights = T.alloc_shared((threads,), dtype=T.float32)
            reduction_ids = T.alloc_shared((threads,), dtype=T.int32)
            raw_score = T.alloc_local((1,), dtype=T.float32)
            choice_score = T.alloc_local((1,), dtype=T.float32)
            candidate_id = T.alloc_local((1,), dtype=T.int32)
            already_selected = T.alloc_local((1,), dtype=T.int32)

            if tx == 0:
                selected_sum[0] = 0.0
            T.sync_threads()

            for kth in T.serial(0, topk):
                raw_score[0] = T.sqrt(
                    T.log(1.0 + T.exp(gating_output[token_id, tx]))
                )
                choice_score[0] = raw_score[0] + correction_bias[tx]
                candidate_id[0] = tx
                already_selected[0] = 0

                for prev in T.serial(0, kth):
                    if tx == selected_ids[prev]:
                        already_selected[0] = 1
                if already_selected[0] != 0:
                    choice_score[0] = -3.4028234663852886e38

                reduction_scores[tx] = choice_score[0]
                reduction_weights[tx] = raw_score[0]
                reduction_ids[tx] = candidate_id[0]
                T.sync_threads()

                if tx < 128:
                    other_score = reduction_scores[tx + 128]
                    other_id = reduction_ids[tx + 128]
                    if (other_score > reduction_scores[tx]) or (
                        other_score == reduction_scores[tx]
                        and other_id < reduction_ids[tx]
                    ):
                        reduction_scores[tx] = other_score
                        reduction_weights[tx] = reduction_weights[tx + 128]
                        reduction_ids[tx] = other_id
                T.sync_threads()
                if tx < 64:
                    other_score = reduction_scores[tx + 64]
                    other_id = reduction_ids[tx + 64]
                    if (other_score > reduction_scores[tx]) or (
                        other_score == reduction_scores[tx]
                        and other_id < reduction_ids[tx]
                    ):
                        reduction_scores[tx] = other_score
                        reduction_weights[tx] = reduction_weights[tx + 64]
                        reduction_ids[tx] = other_id
                T.sync_threads()
                if tx < 32:
                    other_score = reduction_scores[tx + 32]
                    other_id = reduction_ids[tx + 32]
                    if (other_score > reduction_scores[tx]) or (
                        other_score == reduction_scores[tx]
                        and other_id < reduction_ids[tx]
                    ):
                        reduction_scores[tx] = other_score
                        reduction_weights[tx] = reduction_weights[tx + 32]
                        reduction_ids[tx] = other_id
                T.sync_threads()
                if tx < 16:
                    other_score = reduction_scores[tx + 16]
                    other_id = reduction_ids[tx + 16]
                    if (other_score > reduction_scores[tx]) or (
                        other_score == reduction_scores[tx]
                        and other_id < reduction_ids[tx]
                    ):
                        reduction_scores[tx] = other_score
                        reduction_weights[tx] = reduction_weights[tx + 16]
                        reduction_ids[tx] = other_id
                T.sync_threads()
                if tx < 8:
                    other_score = reduction_scores[tx + 8]
                    other_id = reduction_ids[tx + 8]
                    if (other_score > reduction_scores[tx]) or (
                        other_score == reduction_scores[tx]
                        and other_id < reduction_ids[tx]
                    ):
                        reduction_scores[tx] = other_score
                        reduction_weights[tx] = reduction_weights[tx + 8]
                        reduction_ids[tx] = other_id
                T.sync_threads()
                if tx < 4:
                    other_score = reduction_scores[tx + 4]
                    other_id = reduction_ids[tx + 4]
                    if (other_score > reduction_scores[tx]) or (
                        other_score == reduction_scores[tx]
                        and other_id < reduction_ids[tx]
                    ):
                        reduction_scores[tx] = other_score
                        reduction_weights[tx] = reduction_weights[tx + 4]
                        reduction_ids[tx] = other_id
                T.sync_threads()
                if tx < 2:
                    other_score = reduction_scores[tx + 2]
                    other_id = reduction_ids[tx + 2]
                    if (other_score > reduction_scores[tx]) or (
                        other_score == reduction_scores[tx]
                        and other_id < reduction_ids[tx]
                    ):
                        reduction_scores[tx] = other_score
                        reduction_weights[tx] = reduction_weights[tx + 2]
                        reduction_ids[tx] = other_id
                T.sync_threads()
                if tx == 0:
                    other_score = reduction_scores[1]
                    other_id = reduction_ids[1]
                    if (other_score > reduction_scores[0]) or (
                        other_score == reduction_scores[0]
                        and other_id < reduction_ids[0]
                    ):
                        reduction_scores[0] = other_score
                        reduction_weights[0] = reduction_weights[1]
                        reduction_ids[0] = other_id

                    selected_ids[kth] = reduction_ids[0]
                    selected_sum[0] += reduction_weights[0]
                    topk_weights[token_id, kth] = reduction_weights[0]
                    topk_indices[token_id, kth] = T.Cast("int64", reduction_ids[0])
                    token_expert_indices[token_id, kth] = reduction_ids[0]
                T.sync_threads()

            if tx == 0:
                if renormalize:
                    denom = T.max(selected_sum[0], 1.0e-20)
                    for kth in T.serial(0, topk):
                        topk_weights[token_id, kth] = (
                            topk_weights[token_id, kth] / denom
                        )
                if apply_routed_scaling_factor:
                    for kth in T.serial(0, topk):
                        topk_weights[token_id, kth] = (
                            topk_weights[token_id, kth] * routed_scaling_factor
                        )

    return _biased_topk_softplus_sqrt_256_kernel


@tilelang.jit(target="musa", pass_configs=_tilelang_musa_pass_configs(tilelang))
def biased_topk_softplus_sqrt_256_warp_kernel(
    topk: int,
    renormalize: bool,
    apply_routed_scaling_factor: bool,
    tokens_per_block: int = 16,
):
    num_tokens = T.dynamic("num_tokens")
    num_experts = 256
    warp_size = 32
    elems_per_thread = 8
    threads = tokens_per_block * warp_size

    @T.prim_func
    def _biased_topk_softplus_sqrt_256_warp_kernel(
        gating_output: T.Tensor((num_tokens, num_experts), T.float32),
        correction_bias: T.Tensor((num_experts,), T.float32),
        topk_weights: T.Tensor((num_tokens, topk), T.float32),
        topk_indices: T.Tensor((num_tokens, topk), T.int64),
        token_expert_indices: T.Tensor((num_tokens, topk), T.int32),
        routed_scaling_factor: T.float32,
    ):
        with T.Kernel(T.ceildiv(num_tokens, tokens_per_block), threads=threads) as block_id:
            tx = T.get_thread_binding()
            lane_id = tx % warp_size
            warp_id = tx // warp_size
            token_id = block_id * tokens_per_block + warp_id

            raw_scores = T.alloc_local((elems_per_thread,), dtype=T.float32)
            choice_scores = T.alloc_local((elems_per_thread,), dtype=T.float32)
            selected_sum = T.alloc_local((1,), dtype=T.float32)
            best_choice = T.alloc_local((1,), dtype=T.float32)
            best_raw = T.alloc_local((1,), dtype=T.float32)
            best_id = T.alloc_local((1,), dtype=T.int32)

            if token_id < num_tokens:
                selected_sum[0] = 0.0
                for j in T.serial(0, elems_per_thread):
                    expert_id = j * warp_size + lane_id
                    raw = T.sqrt(
                        T.log(1.0 + T.exp(gating_output[token_id, expert_id]))
                    )
                    raw_scores[j] = raw
                    choice_scores[j] = raw + correction_bias[expert_id]

                for kth in T.serial(0, topk):
                    best_choice[0] = -3.4028234663852886e38
                    best_raw[0] = 0.0
                    best_id[0] = -1

                    for j in T.serial(0, elems_per_thread):
                        expert_id = j * warp_size + lane_id
                        if (choice_scores[j] > best_choice[0]) or (
                            choice_scores[j] == best_choice[0]
                            and (
                                best_id[0] < 0
                                or expert_id < best_id[0]
                            )
                        ):
                            best_choice[0] = choice_scores[j]
                            best_raw[0] = raw_scores[j]
                            best_id[0] = expert_id

                    for i in T.serial(0, 5):
                        mask = T.int32(16) >> i
                        other_choice = T.shfl_xor(best_choice[0], mask)
                        other_raw = T.shfl_xor(best_raw[0], mask)
                        other_id = T.shfl_xor(best_id[0], mask)
                        take_other = (other_choice > best_choice[0]) or (
                            other_choice == best_choice[0]
                            and (
                                best_id[0] < 0
                                or other_id < best_id[0]
                            )
                        )
                        best_choice[0] = T.if_then_else(
                            take_other, other_choice, best_choice[0]
                        )
                        best_raw[0] = T.if_then_else(
                            take_other, other_raw, best_raw[0]
                        )
                        best_id[0] = T.if_then_else(take_other, other_id, best_id[0])

                    selected_sum[0] += best_raw[0]
                    if lane_id == 0:
                        topk_weights[token_id, kth] = best_raw[0]
                        topk_indices[token_id, kth] = T.Cast("int64", best_id[0])
                        token_expert_indices[token_id, kth] = best_id[0]

                    for j in T.serial(0, elems_per_thread):
                        if j * warp_size + lane_id == best_id[0]:
                            choice_scores[j] = -3.4028234663852886e38

                if lane_id == 0:
                    if renormalize:
                        denom = T.max(selected_sum[0], 1.0e-20)
                        for kth in T.serial(0, topk):
                            topk_weights[token_id, kth] = (
                                topk_weights[token_id, kth] / denom
                            )
                    if apply_routed_scaling_factor:
                        for kth in T.serial(0, topk):
                            topk_weights[token_id, kth] = (
                                topk_weights[token_id, kth] * routed_scaling_factor
                            )

    return _biased_topk_softplus_sqrt_256_warp_kernel
