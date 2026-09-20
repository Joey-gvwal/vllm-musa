# SPDX-License-Identifier: Apache-2.0
"""TileLang sparse paged GQA for the QSA prefill and decode paths.

Both stages gather K/V tile by tile with per-token head-dim-contiguous vector
loads (the sparse page addresses are constant along the dim axis) and compute
QK^T / PV with T.gemm; the grouped-query tile is padded to 8 rows to satisfy the
MUSA MMA M%8 requirement. The online softmax stays in fragments.

Decode splits the tile loop over ``num_splits`` programs that each emit a
normalized partial output plus its log2-domain LSE, merged by a second kernel:
one program per row leaves the device idle at small batch sizes.

``block_n`` is tied to the thread count (``threads == block_n * _DUP``) so that
no loop in the tile body is thread-predicated. TileLang's storage-sync pass
refuses to keep a barrier inside a predicated region and hoists it out
("[ThreadSync] Hoisting sync from inside if to before if"), which drops the
write-then-read ordering on the shared gather metadata and makes the kernel
nondeterministic.
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
# The gather stages its page metadata in shared memory before reading through
# it, so the cross-loop thread storage syncs that the common config disables for
# pure elementwise kernels must stay on.
if hasattr(tilelang.PassConfigKey, "TL_DISABLE_THREAD_STORAGE_SYNC"):
    _PASS_CONFIGS[tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC] = False
for _key, _value in (
    ("TL_DISABLE_SAFE_COPY_PREDICATION", True),
    ("TL_DISABLE_SAFE_ROBUST_COPY_PREDICATION", True),
    ("TL_CONFIG_INDEX_BITWIDTH", 32),
):
    if hasattr(tilelang.PassConfigKey, _key):
        _PASS_CONFIGS[getattr(tilelang.PassConfigKey, _key)] = _value

__all__ = [
    "qsa_prefill_attention_v2",
    "qsa_decode_attention",
    "prewarm_qsa_kernels",
]

# Scores are masked to a large finite negative value instead of -inf so that an
# all-invalid tile keeps the running max arithmetic well defined.
_NEG_INF = -1.0e30

_BLOCK_M = 8
_BLOCK_N = 128
_DUP = 2
_THREADS = _BLOCK_N * _DUP
_DECODE_SPLITS = 16


@functools.lru_cache(maxsize=16)
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
    threads: int,
    dup: int,
):
    num_tiles = (topk + block_n - 1) // block_n
    block_m = _BLOCK_M
    # log2-scaled softmax so the loop uses exp2 and never requantizes Q.
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
        with T.Kernel(num_rows, threads=threads) as row:
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

            acc_frag = T.alloc_fragment((block_m, head_dim), "float32")
            scores_frag = T.alloc_fragment((block_m, block_n), "float32")
            mrun = T.alloc_fragment((block_m,), "float32")
            mprev = T.alloc_fragment((block_m,), "float32")
            alpha = T.alloc_fragment((block_m,), "float32")
            lrun = T.alloc_fragment((block_m,), "float32")
            lsum_i = T.alloc_fragment((block_m,), "float32")

            for m, d in T.Parallel(block_m, head_dim):
                if m < group_size:
                    s_q[m, d] = q[row, m, d]
                else:
                    s_q[m, d] = 0.0
            T.fill(acc_frag, 0.0)
            T.fill(lrun, 0.0)
            T.fill(mrun, -(2.0**30))

            for tile in T.serial(num_tiles):
                # Tiles that start beyond the row's valid count are entirely
                # masked: skip their gather and softmax work.
                if tile * block_n < valid_count:
                    # Keep the index-load address affine on the fast path; only
                    # the last tile of a non-multiple width needs the clamp.
                    # Clamped lanes read the trailing count column and stay
                    # invalid through the column < valid_count check below.
                    if tile < num_tiles - 1:
                        for i, _u in T.Parallel(block_n, dup):
                            s_idx[i] = indices[row, tile * block_n + i]
                    else:
                        for i, _u in T.Parallel(block_n, dup):
                            s_idx[i] = indices[row, T.min(tile * block_n + i, topk)]

                    T.sync_threads()
                    for i, _u in T.Parallel(block_n, dup):
                        tok = T.max(s_idx[i], 0)
                        raw_page = block_table[
                            req_safe,
                            T.min(tok // page_size, page_table_width - 1),
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
                        s_phys[i] = T.min(T.max(raw_page, 0), num_blocks - 1)
                        s_valid[i] = valid

                    T.sync_threads()
                    for i, d in T.Parallel(block_n, head_dim):
                        s_k[i, d] = k_cache[s_phys[i], s_off[i], d]
                        s_v[i, d] = v_cache[s_phys[i], s_off[i], d]
                    T.sync_threads()

                    for m, n in T.Parallel(block_m, block_n):
                        scores_frag[m, n] = T.if_then_else(
                            s_valid[n] > 0.5, 0.0, _NEG_INF
                        )
                    T.gemm(s_q, s_k, scores_frag, transpose_B=True)
                    T.copy(mrun, mprev)
                    T.reduce_max(scores_frag, mrun, dim=1, clear=False)
                    for m in T.Parallel(block_m):
                        alpha[m] = T.exp2((mprev[m] - mrun[m]) * softmax_scale_log2)
                    for m, n in T.Parallel(block_m, block_n):
                        scores_frag[m, n] = T.exp2(
                            scores_frag[m, n] * softmax_scale_log2
                            - mrun[m] * softmax_scale_log2
                        )
                    T.reduce_sum(scores_frag, lsum_i, dim=1)
                    for m in T.Parallel(block_m):
                        lrun[m] = lrun[m] * alpha[m] + lsum_i[m]
                    for m, d in T.Parallel(block_m, head_dim):
                        acc_frag[m, d] = acc_frag[m, d] * alpha[m]
                    T.copy(scores_frag, s_pb)
                    T.sync_threads()
                    T.gemm(s_pb, s_v, acc_frag)
                    T.sync_threads()

            for m, d in T.Parallel(block_m, head_dim):
                if m < group_size:
                    out[row, m, d] = T.cast(
                        T.if_then_else(
                            lrun[m] > 0.0,
                            acc_frag[m, d] / T.max(lrun[m], 1.0e-20),
                            0.0,
                        ),
                        dtype,
                    )

    return qsa_prefill


@functools.lru_cache(maxsize=16)
@tilelang.jit(
    target="musa",
    pass_configs=_PASS_CONFIGS,
    compile_flags=MUSA_COMPILE_FLAGS,
)
def _qsa_decode_kernel(
    group_size: int,
    head_dim: int,
    topk: int,
    block_n: int,
    page_size: int,
    page_table_width: int,
    dtype: str,
    threads: int,
    dup: int,
    num_splits: int,
):
    num_tiles = (topk + block_n - 1) // block_n
    block_m = _BLOCK_M
    softmax_scale_log2 = (head_dim**-0.5) * 1.4426950408889634
    num_rows = T.dynamic("num_rows")
    num_blocks = T.dynamic("num_blocks")
    num_requests = T.dynamic("num_requests")

    @T.prim_func
    def qsa_decode(
        q: T.Tensor((num_rows, group_size, head_dim), dtype),
        k_cache: T.Tensor((num_blocks, page_size, head_dim), dtype),
        v_cache: T.Tensor((num_blocks, page_size, head_dim), dtype),
        indices: T.Tensor((num_rows, topk + 1), "int32"),
        block_table: T.Tensor((num_requests, page_table_width), "int32"),
        token_to_req: T.Tensor((num_rows,), "int32"),
        partial_o: T.Tensor((num_splits, num_rows, block_m, head_dim), "float32"),
        partial_l: T.Tensor((num_splits, num_rows, block_m), "float32"),
    ):
        with T.Kernel(num_rows, num_splits, threads=threads) as (row, split):
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

            acc_frag = T.alloc_fragment((block_m, head_dim), "float32")
            scores_frag = T.alloc_fragment((block_m, block_n), "float32")
            mrun = T.alloc_fragment((block_m,), "float32")
            mprev = T.alloc_fragment((block_m,), "float32")
            alpha = T.alloc_fragment((block_m,), "float32")
            lrun = T.alloc_fragment((block_m,), "float32")
            lsum_i = T.alloc_fragment((block_m,), "float32")

            for m, d in T.Parallel(block_m, head_dim):
                if m < group_size:
                    s_q[m, d] = q[row, m, d]
                else:
                    s_q[m, d] = 0.0
            T.fill(acc_frag, 0.0)
            T.fill(lrun, 0.0)
            T.fill(mrun, -(2.0**30))

            for t_i in T.serial((num_tiles + num_splits - 1) // num_splits):
                tile = split + t_i * num_splits
                if (tile < num_tiles) and (tile * block_n < valid_count):
                    if tile < num_tiles - 1:
                        for i, _u in T.Parallel(block_n, dup):
                            s_idx[i] = indices[row, tile * block_n + i]
                    else:
                        for i, _u in T.Parallel(block_n, dup):
                            s_idx[i] = indices[row, T.min(tile * block_n + i, topk)]

                    T.sync_threads()
                    for i, _u in T.Parallel(block_n, dup):
                        tok = T.max(s_idx[i], 0)
                        raw_page = block_table[
                            req_safe,
                            T.min(tok // page_size, page_table_width - 1),
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
                        s_phys[i] = T.min(T.max(raw_page, 0), num_blocks - 1)
                        s_valid[i] = valid

                    T.sync_threads()
                    for i, d in T.Parallel(block_n, head_dim):
                        s_k[i, d] = k_cache[s_phys[i], s_off[i], d]
                        s_v[i, d] = v_cache[s_phys[i], s_off[i], d]
                    T.sync_threads()

                    for m, n in T.Parallel(block_m, block_n):
                        scores_frag[m, n] = T.if_then_else(
                            s_valid[n] > 0.5, 0.0, _NEG_INF
                        )
                    T.gemm(s_q, s_k, scores_frag, transpose_B=True)
                    T.copy(mrun, mprev)
                    T.reduce_max(scores_frag, mrun, dim=1, clear=False)
                    for m in T.Parallel(block_m):
                        alpha[m] = T.exp2((mprev[m] - mrun[m]) * softmax_scale_log2)
                    for m, n in T.Parallel(block_m, block_n):
                        scores_frag[m, n] = T.exp2(
                            scores_frag[m, n] * softmax_scale_log2
                            - mrun[m] * softmax_scale_log2
                        )
                    T.reduce_sum(scores_frag, lsum_i, dim=1)
                    for m in T.Parallel(block_m):
                        lrun[m] = lrun[m] * alpha[m] + lsum_i[m]
                    for m, d in T.Parallel(block_m, head_dim):
                        acc_frag[m, d] = acc_frag[m, d] * alpha[m]
                    T.copy(scores_frag, s_pb)
                    T.sync_threads()
                    T.gemm(s_pb, s_v, acc_frag)
                    T.sync_threads()

            for m, d in T.Parallel(block_m, head_dim):
                partial_o[split, row, m, d] = T.if_then_else(
                    lrun[m] > 0.0, acc_frag[m, d] / T.max(lrun[m], 1.0e-20), 0.0
                )
            for m in T.Parallel(block_m):
                partial_l[split, row, m] = T.if_then_else(
                    lrun[m] > 0.0,
                    mrun[m] * softmax_scale_log2 + T.log2(lrun[m]),
                    _NEG_INF,
                )

    return qsa_decode


@functools.lru_cache(maxsize=16)
@tilelang.jit(
    target="musa",
    pass_configs=_PASS_CONFIGS,
    compile_flags=MUSA_COMPILE_FLAGS,
)
def _qsa_merge_kernel(
    group_size: int,
    head_dim: int,
    num_splits: int,
    dtype: str,
    threads: int,
):
    block_m = _BLOCK_M
    num_rows = T.dynamic("num_rows")

    @T.prim_func
    def qsa_merge(
        partial_o: T.Tensor((num_splits, num_rows, block_m, head_dim), "float32"),
        partial_l: T.Tensor((num_splits, num_rows, block_m), "float32"),
        out: T.Tensor((num_rows, group_size, head_dim), dtype),
    ):
        with T.Kernel(num_rows, threads=threads) as row:
            acc = T.alloc_fragment((block_m, head_dim), "float32")
            lmax = T.alloc_fragment((block_m,), "float32")
            lsum = T.alloc_fragment((block_m,), "float32")
            a_old = T.alloc_fragment((block_m,), "float32")
            a_new = T.alloc_fragment((block_m,), "float32")
            T.fill(acc, 0.0)
            T.fill(lsum, 0.0)
            T.fill(lmax, _NEG_INF)
            for s_i in T.serial(num_splits):
                for m in T.Parallel(block_m):
                    lnew = T.max(lmax[m], partial_l[s_i, row, m])
                    a_old[m] = T.exp2(lmax[m] - lnew)
                    a_new[m] = T.exp2(partial_l[s_i, row, m] - lnew)
                    lsum[m] = lsum[m] * a_old[m] + a_new[m]
                    lmax[m] = lnew
                for m, d in T.Parallel(block_m, head_dim):
                    acc[m, d] = (
                        acc[m, d] * a_old[m] + partial_o[s_i, row, m, d] * a_new[m]
                    )
            for m, d in T.Parallel(block_m, head_dim):
                if m < group_size:
                    out[row, m, d] = T.cast(
                        T.if_then_else(lsum[m] > 0.0, acc[m, d] / lsum[m], 0.0),
                        dtype,
                    )

    return qsa_merge


_qsa_prefill_kernel.mode = "lazy"
_qsa_decode_kernel.mode = "lazy"
_qsa_merge_kernel.mode = "lazy"

# Split-k scratch, keyed so a captured decode graph always sees the same buffers.
_SCRATCH: dict = {}


def _shape_args(q, k_cache, logical_indices, block_table):
    return (
        q.shape[1] // k_cache.shape[2],
        q.shape[2],
        logical_indices.shape[1] - 1,
        _BLOCK_N,
        k_cache.shape[1],
        block_table.shape[1],
        tilelang_dtype(q.dtype),
        _THREADS,
        _DUP,
    )


def qsa_prefill_attention_v2(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    if q.shape[0] == 0:
        return out
    kernel = _qsa_prefill_kernel(
        *_shape_args(q, k_cache, logical_indices, block_table)
    )
    # The upstream forward canonicalizes the singleton KV-head stride to 0;
    # TileLang takes dense tensors, so drop the singleton dim (a view).
    kernel(q, k_cache.squeeze(2), v_cache.squeeze(2), logical_indices,
           block_table, token_to_req, out)
    return out


def _split_scratch(num_rows, head_dim, device, num_splits):
    key = (num_splits, num_rows, head_dim, str(device))
    buf = _SCRATCH.get(key)
    if buf is None:
        buf = (
            torch.empty(num_splits, num_rows, _BLOCK_M, head_dim,
                        device=device, dtype=torch.float32),
            torch.empty(num_splits, num_rows, _BLOCK_M,
                        device=device, dtype=torch.float32),
        )
        _SCRATCH[key] = buf
    return buf


def qsa_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    out: torch.Tensor,
    num_splits: int = _DECODE_SPLITS,
) -> torch.Tensor:
    num_rows = q.shape[0]
    if num_rows == 0:
        return out
    group_size = q.shape[1] // k_cache.shape[2]
    head_dim = q.shape[2]
    partial_o, partial_l = _split_scratch(num_rows, head_dim, q.device, num_splits)
    split = _qsa_decode_kernel(
        *_shape_args(q, k_cache, logical_indices, block_table), num_splits
    )
    split(q, k_cache.squeeze(2), v_cache.squeeze(2), logical_indices,
          block_table, token_to_req, partial_o, partial_l)
    merge = _qsa_merge_kernel(
        group_size, head_dim, num_splits, tilelang_dtype(q.dtype), _THREADS
    )
    merge(partial_o, partial_l, out)
    return out


def prewarm_qsa_kernels(
    group_size: int,
    head_dim: int,
    topk: int,
    page_size: int,
    page_table_width: int,
    dtype: torch.dtype,
    decode_rows: tuple = (),
    num_splits: int = _DECODE_SPLITS,
) -> None:
    """Compile the kernels and allocate the split-k scratch ahead of capture.

    TileLang compiles on the first call for a given signature; a first call
    inside a captured graph deadlocks the worker.
    """
    args = (
        group_size, head_dim, topk, _BLOCK_N, page_size, page_table_width,
        tilelang_dtype(dtype), _THREADS, _DUP,
    )
    _qsa_prefill_kernel(*args)
    _qsa_decode_kernel(*args, num_splits)
    _qsa_merge_kernel(group_size, head_dim, num_splits, tilelang_dtype(dtype),
                      _THREADS)
    device = torch.musa.current_device() if hasattr(torch, "musa") else None
    for rows in decode_rows:
        _split_scratch(rows, head_dim, device, num_splits)
