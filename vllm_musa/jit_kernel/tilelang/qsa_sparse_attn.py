# SPDX-License-Identifier: Apache-2.0
"""TileLang sparse paged GQA attention for the QSA prefill path (MMA).

Same numerics as qsa_sparse_attn_mma.py; changes for speed:
- the K/V gather is a flat 2D Parallel loop (head-dim contiguous vector loads,
  no serial per-token steps);
- the per-tile row max / row sum use fragment T.reduce_max / T.reduce_sum
  instead of serial reduction loops.
"""

import functools

import tilelang
import tilelang.language as T
import torch

from vllm_musa.jit_kernel.tilelang.utils import (
    MUSA_COMMON_PASS_CONFIGS,
    MUSA_COMPILE_FLAGS,
    tilelang_dtype,
)

_PASS_CONFIGS = dict(MUSA_COMMON_PASS_CONFIGS)
if hasattr(tilelang.PassConfigKey, "TL_DISABLE_THREAD_STORAGE_SYNC"):
    _PASS_CONFIGS[tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC] = False
for _key, _value in (
    ("TL_DISABLE_SAFE_COPY_PREDICATION", True),
    ("TL_DISABLE_SAFE_ROBUST_COPY_PREDICATION", True),
    ("TL_CONFIG_INDEX_BITWIDTH", 32),
):
    if hasattr(tilelang.PassConfigKey, _key):
        _PASS_CONFIGS[getattr(tilelang.PassConfigKey, _key)] = _value

__all__ = ["qsa_prefill_attention"]

_NEG_INF = -1.0e30


@functools.lru_cache(maxsize=8)
@tilelang.jit(
    target="musa",
    pass_configs=_PASS_CONFIGS,
    compile_flags=MUSA_COMPILE_FLAGS,
)
def _qsa_prefill_kernel(
    group_size: int,
    head_dim: int,
    topk: int,
    block_n: int,
    page_size: int,
    page_table_width: int,
    dtype: str,
):
    num_tiles = (topk + block_n - 1) // block_n
    block_m = 8  # MMA-aligned grouped-query tile (3 real heads + 5 zeros)
    softmax_scale_log2 = (head_dim**-0.5) * 1.4426950408889634
    num_rows = T.dynamic("num_rows")
    num_blocks = T.dynamic("num_blocks")
    num_requests = T.dynamic("num_requests")

    @T.prim_func
    def qsa_prefill(
        q: T.Tensor((num_rows, group_size, head_dim), dtype),
        k_cache: T.Tensor((num_blocks, page_size, head_dim), dtype),
        v_cache: T.Tensor((num_blocks, page_size, head_dim), dtype),
        indices: T.Tensor((num_rows, topk + 1), "int32"),
        block_table: T.Tensor((num_requests, page_table_width), "int32"),
        token_to_req: T.Tensor((num_rows,), "int32"),
        out: T.Tensor((num_rows, group_size, head_dim), dtype),
    ):
        with T.Kernel(num_rows, threads=128) as row:
            req = token_to_req[row]
            req_safe = T.min(T.max(req, 0), num_requests - 1)
            valid_count = indices[row, topk]

            s_q = T.alloc_shared((block_m, head_dim), dtype)
            s_k = T.alloc_shared((block_n, head_dim), dtype)
            s_v = T.alloc_shared((block_n, head_dim), dtype)
            s_pb = T.alloc_shared((block_m, block_n), dtype)
            s_idx = T.alloc_shared((block_n,), "int32")
            s_off = T.alloc_shared((block_n,), "int32")
            s_phys = T.alloc_shared((block_n,), "int32")
            s_valid = T.alloc_shared((block_n,), "float32")
            s_scores = T.alloc_shared((block_m, block_n), "float32")
            s_p = T.alloc_shared((block_m, block_n), "float32")
            s_mrun = T.alloc_shared((block_m,), "float32")
            s_lrun = T.alloc_shared((block_m,), "float32")
            s_alpha = T.alloc_shared((block_m,), "float32")
            s_rowmax = T.alloc_shared((block_m,), "float32")
            s_rowsum = T.alloc_shared((block_m,), "float32")

            acc_frag = T.alloc_fragment((block_m, head_dim), "float32")
            scores_frag = T.alloc_fragment((block_m, block_n), "float32")
            rowmax_frag = T.alloc_fragment((block_m,), "float32")
            rowsum_frag = T.alloc_fragment((block_m,), "float32")

            for m, d in T.Parallel(block_m, head_dim):
                if m < group_size:
                    s_q[m, d] = q[row, m, d]
                else:
                    s_q[m, d] = 0.0
            for m in T.Parallel(block_m):
                s_mrun[m] = _NEG_INF
                s_lrun[m] = 0.0

            for tile in T.serial(num_tiles):
                for i in T.Parallel(block_n):
                    s_idx[i] = indices[row, tile * block_n + i]
                for i in T.Parallel(block_n):
                    tok = T.max(s_idx[i], 0)
                    raw_page = block_table[
                        req_safe, T.min(tok // page_size, page_table_width - 1)
                    ]
                    valid = 1.0
                    valid = T.if_then_else(req >= 0, valid, 0.0)
                    valid = T.if_then_else(req < num_requests, valid, 0.0)
                    valid = T.if_then_else(s_idx[i] >= 0, valid, 0.0)
                    valid = T.if_then_else(
                        tile * block_n + i < valid_count, valid, 0.0
                    )
                    valid = T.if_then_else(
                        tok // page_size < page_table_width, valid, 0.0
                    )
                    valid = T.if_then_else(raw_page >= 0, valid, 0.0)
                    valid = T.if_then_else(raw_page < num_blocks, valid, 0.0)
                    s_off[i] = tok % page_size
                    s_phys[i] = T.max(raw_page, 0)
                    s_valid[i] = valid
                for i, d in T.Parallel(block_n, head_dim):
                    s_k[i, d] = k_cache[s_phys[i], s_off[i], d]
                    s_v[i, d] = v_cache[s_phys[i], s_off[i], d]

                T.clear(scores_frag)
                T.gemm(s_q, s_k, scores_frag, transpose_B=True)
                T.copy(scores_frag, s_scores)
                for m, n in T.Parallel(block_m, block_n):
                    s_scores[m, n] = T.if_then_else(
                        s_valid[n] > 0.5,
                        s_scores[m, n] * softmax_scale_log2,
                        _NEG_INF,
                    )
                T.copy(s_scores, scores_frag)

                T.reduce_max(scores_frag, rowmax_frag, dim=1, clear=True)
                T.copy(rowmax_frag, s_rowmax)
                for m in T.Parallel(block_m):
                    new_max = T.max(s_mrun[m], s_rowmax[m])
                    s_alpha[m] = T.exp2(s_mrun[m] - new_max)
                    s_mrun[m] = new_max
                    s_lrun[m] = s_lrun[m] * s_alpha[m]

                for m, n in T.Parallel(block_m, block_n):
                    s_p[m, n] = T.if_then_else(
                        s_valid[n] > 0.5,
                        T.exp2(s_scores[m, n] - s_mrun[m]),
                        0.0,
                    )
                T.copy(s_p, scores_frag)
                T.reduce_sum(scores_frag, rowsum_frag, dim=1, clear=True)
                T.copy(rowsum_frag, s_rowsum)
                for m in T.Parallel(block_m):
                    s_lrun[m] += s_rowsum[m]
                for m, n in T.Parallel(block_m, block_n):
                    s_pb[m, n] = T.cast(s_p[m, n], dtype)

                for m, d in T.Parallel(block_m, head_dim):
                    acc_frag[m, d] = acc_frag[m, d] * s_alpha[m]
                T.gemm(s_pb, s_v, acc_frag)

            for m, d in T.Parallel(block_m, head_dim):
                if m < group_size:
                    out[row, m, d] = T.cast(
                        T.if_then_else(
                            s_lrun[m] > 0.0,
                            acc_frag[m, d] / T.max(s_lrun[m], 1.0e-20),
                            0.0,
                        ),
                        dtype,
                    )

    return qsa_prefill


_qsa_prefill_kernel.mode = "lazy"


def qsa_prefill_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    num_rows = q.shape[0]
    if num_rows == 0:
        return out
    group_size = q.shape[1] // k_cache.shape[2]
    head_dim = q.shape[2]
    topk = logical_indices.shape[1] - 1
    page_size = k_cache.shape[1]
    page_table_width = block_table.shape[1]
    k_view = k_cache.squeeze(2)
    v_view = v_cache.squeeze(2)
    kernel = _qsa_prefill_kernel(
        group_size,
        head_dim,
        topk,
        64,
        page_size,
        page_table_width,
        tilelang_dtype(q.dtype),
    )
    kernel(q, k_view, v_view, logical_indices, block_table, token_to_req, out)
    return out
