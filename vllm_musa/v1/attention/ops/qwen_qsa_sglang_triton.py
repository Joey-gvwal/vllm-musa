# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SGLang-MUSA packed sparse-GQA operators for Qwen3.8 QSA.

The packed GQA kernel follows sglang-musa commit ``0b267040b4``.  The adapter
adds vLLM block-table addressing while retaining the kernel's cu-seqlens and
online-softmax contract.
"""

import logging
import os
from typing import Optional

import torch
import triton
import triton.language as tl


_DECODE_WORKSPACES = {}
_logger = logging.getLogger(__name__)


@triton.jit
def _sparse_gqa_decode_paged_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    table_ptr,
    req_ptr,
    idx_ptr,
    seq_ptr,
    out_ptr,
    rows,
    topk,
    heads,
    kv_heads,
    dim,
    q_s0,
    q_s1,
    k_s0,
    k_s1,
    k_s2,
    v_s0,
    v_s1,
    v_s2,
    table_s0,
    idx_s0,
    page_size,
    scale,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, BLOCK_D)
    q = tl.load(q_ptr + row * q_s0 + head * q_s1 + d, mask=d < dim, other=0.0).to(tl.float32)
    kv_head = head * kv_heads // heads
    req = tl.load(req_ptr + row)
    seq_len = tl.load(seq_ptr + row)
    m = tl.full((), -float("inf"), tl.float32)
    l = tl.zeros((), tl.float32)
    acc = tl.zeros((BLOCK_D,), tl.float32)
    for start in tl.range(0, topk, BLOCK_TOPK):
        j = start + tl.arange(0, BLOCK_TOPK)
        idx = tl.load(idx_ptr + row * idx_s0 + j, mask=j < topk, other=-1)
        valid = (j < topk) & (idx >= 0) & (idx < seq_len) & (req >= 0)
        page = tl.load(table_ptr + req * table_s0 + idx // page_size, mask=valid, other=0)
        slot = page * page_size + idx % page_size
        k = tl.load(
            k_ptr + slot[:, None] * k_s0 + kv_head * k_s1 + d[None, :] * k_s2,
            mask=valid[:, None] & (d[None, :] < dim),
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            v_ptr + slot[:, None] * v_s0 + kv_head * v_s1 + d[None, :] * v_s2,
            mask=valid[:, None] & (d[None, :] < dim),
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(k * q[None, :], axis=1) * scale
        scores = tl.where(valid, scores, -float("inf"))
        block_m = tl.max(scores, axis=0)
        new_m = tl.maximum(m, block_m)
        alpha = tl.exp(m - new_m)
        beta = tl.exp(scores - new_m)
        l = l * alpha + tl.sum(beta, axis=0)
        acc = acc * alpha + tl.sum(beta[:, None] * v, axis=0)
        m = new_m
    out = acc / l
    tl.store(out_ptr + row * q_s0 + head * q_s1 + d, out, mask=d < dim)


def sparse_gqa_decode_paged_triton(
    q, k_cache, v_cache, block_table, token_to_req, logical_indices,
    sequence_lengths, scale,
):
    """Fuse paged selection and decode GQA (SGLang fallback shape)."""
    rows, heads, dim = q.shape
    page_size, kv_heads = int(k_cache.shape[1]), int(k_cache.shape[2])
    if dim != 256 or v_cache.shape != k_cache.shape:
        raise ValueError("fused QSA decode requires head_dim=256 and matching K/V")
    if heads % kv_heads:
        raise ValueError("QSA heads must be divisible by KV heads")
    table = block_table.to(device=q.device, dtype=torch.int32).contiguous()
    req = token_to_req.to(device=q.device, dtype=torch.int32).contiguous()
    idx = logical_indices.to(device=q.device, dtype=torch.int32).contiguous()
    seq = sequence_lengths.to(device=q.device, dtype=torch.int32).contiguous()
    out = torch.empty_like(q)
    k = k_cache.contiguous()
    v = v_cache.contiguous()
    k_flat = k.reshape(-1, kv_heads, dim)
    v_flat = v.reshape(-1, kv_heads, dim)
    _sparse_gqa_decode_paged_kernel[(rows, heads)](
        q, k_flat, v_flat,
        table, req, idx, seq, out, rows, idx.shape[1], heads, kv_heads, dim,
        q.stride(0), q.stride(1), k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
        v_flat.stride(0), v_flat.stride(1), v_flat.stride(2), table.stride(0), idx.stride(0),
        page_size, scale, BLOCK_TOPK=16, BLOCK_D=256, num_warps=4,
    )
    return out


def _decode_workspace(device, dtype, rows, topk, heads, dim):
    key = (device.type, device.index, dtype, int(rows), int(topk), int(heads), int(dim))
    workspace = _DECODE_WORKSPACES.get(key)
    if workspace is None:
        workspace = (
            torch.empty(rows, device=device, dtype=torch.int32),
            torch.empty(rows + 1, device=device, dtype=torch.int32),
            torch.empty(rows * topk, heads, dim, device=device, dtype=dtype),
            torch.empty(rows * topk, heads, dim, device=device, dtype=dtype),
            torch.empty(rows + 1, device=device, dtype=torch.int32),
            torch.empty(rows, topk, device=device, dtype=torch.int32),
            torch.empty(rows, topk, device=device, dtype=torch.int32),
            torch.empty(rows, device=device, dtype=torch.int32),
            torch.empty(rows, device=device, dtype=torch.int32),
        )
        _DECODE_WORKSPACES[key] = workspace
    return workspace
_H20_CONFIGS = [
    (32, (32, 8, 2)),
    (64, (64, 8, 2)),
    (1024, (32, 4, 2)),
    (float("inf"), (16, 1, 2)),
]
_L20_CONFIGS = [
    (32, (32, 8, 2)),
    (64, (64, 8, 2)),
    (128, (64, 4, 2)),
    (512, (32, 4, 2)),
    (float("inf"), (16, 1, 2)),
]
_MUSA_CONFIGS = _L20_CONFIGS


def _get_best_config(total_q: int, device: torch.device | None = None):
    device = torch.device("musa" if device is None else device)
    if device.type == "musa":
        table = _MUSA_CONFIGS
    else:
        table = _H20_CONFIGS if "H20" in torch.cuda.get_device_name(device) else _L20_CONFIGS
    return next(cfg for limit, cfg in table if total_q <= limit)


@triton.jit
def _sparse_gqa_prefill(
    q,
    k,
    v,
    out,
    indices,
    cu_seqlens,
    scale,
    topk,
    sq_m: tl.constexpr,
    sq_h: tl.constexpr,
    sq_d: tl.constexpr,
    sk_n: tl.constexpr,
    sk_h: tl.constexpr,
    sk_d: tl.constexpr,
    sv_n: tl.constexpr,
    sv_h: tl.constexpr,
    sv_d: tl.constexpr,
    so_m: tl.constexpr,
    so_h: tl.constexpr,
    so_d: tl.constexpr,
    si_m: tl.constexpr,
    si_g: tl.constexpr,
    si_n: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    batch_group = tl.program_id(1)
    group = batch_group % NUM_KV_HEADS
    batch = batch_group // NUM_KV_HEADS
    seq_start = tl.load(cu_seqlens + batch).to(tl.int64)
    seq_end = tl.load(cu_seqlens + batch + 1).to(tl.int64)
    query_relative = tl.program_id(0).to(tl.int64)
    query = seq_start + query_relative
    if query >= seq_end:
        return

    row_topk = tl.minimum(topk, query_relative + 1)
    row_limit = tl.minimum(topk, ((row_topk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N)
    offs_h = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    head_start = group * GROUP_SIZE
    q_values = tl.load(
        q
        + query * sq_m
        + (head_start + offs_h[:, None]) * sq_h
        + offs_d[None, :] * sq_d,
        mask=(offs_h < GROUP_SIZE)[:, None],
        other=0.0,
    )
    q_values = (q_values * scale * 1.4426950408).to(q_values.dtype)
    k_base = k + seq_start * sk_n + group * sk_h
    v_base = v + seq_start * sv_n + group * sv_h
    idx_row = indices + query * si_m + group * si_g
    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for start in range(0, row_limit, BLOCK_N):
        current = start + offs_n
        token = tl.load(idx_row + current * si_n, mask=current < topk, other=-1)
        valid = token >= 0
        keys = tl.load(
            k_base + token[None, :] * sk_n + offs_d[:, None] * sk_d,
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_base + token[:, None] * sv_n + offs_d[None, :] * sv_d,
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.where(valid[None, :], tl.dot(q_values, keys), -float("inf"))
        next_max = tl.maximum(max_value, tl.max(scores, 1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.math.exp2(scores - next_max[:, None])
        accumulator = tl.dot(
            probabilities.to(values.dtype), values, accumulator * alpha[:, None]
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, 1)
        max_value = next_max
    output = accumulator / tl.where(normalizer[:, None] > 0, normalizer[:, None], 1.0)
    tl.store(
        out
        + query * so_m
        + (head_start + offs_h[:, None]) * so_h
        + offs_d[None, :] * so_d,
        output,
        mask=(offs_h < GROUP_SIZE)[:, None],
    )


def sparse_gqa_fwd_interface_triton(q, k, v, max_seqlen_k, indices, cu_seqlens, scale):
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group_size = num_q_heads // num_kv_heads
    block_m = max(16, triton.next_power_of_2(group_size))
    block_n, warps, stages = _get_best_config(total_q)
    out = torch.empty_like(q)
    _sparse_gqa_prefill[(max_seqlen_k, (cu_seqlens.shape[0] - 1) * num_kv_heads)](
        q,
        k,
        v,
        out,
        indices,
        cu_seqlens,
        scale,
        indices.shape[-1],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        indices.stride(0),
        indices.stride(1) if indices.ndim == 3 else 0,
        indices.stride(2) if indices.ndim == 3 else indices.stride(1),
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        num_warps=warps,
        num_stages=stages,
    )
    return out


@triton.jit
def _sparse_gqa_chunk_prefill(
    q,
    k,
    v,
    out,
    indices,
    cu_q,
    cu_k,
    kv_lens,
    scale,
    topk,
    sq_m: tl.constexpr,
    sq_h: tl.constexpr,
    sq_d: tl.constexpr,
    sk_n: tl.constexpr,
    sk_h: tl.constexpr,
    sk_d: tl.constexpr,
    sv_n: tl.constexpr,
    sv_h: tl.constexpr,
    sv_d: tl.constexpr,
    so_m: tl.constexpr,
    so_h: tl.constexpr,
    so_d: tl.constexpr,
    si_m: tl.constexpr,
    si_g: tl.constexpr,
    si_n: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    query_relative = tl.program_id(0).to(tl.int64)
    batch_group = tl.program_id(1)
    group = batch_group % NUM_KV_HEADS
    batch = batch_group // NUM_KV_HEADS
    q_start = tl.load(cu_q + batch)
    q_end = tl.load(cu_q + batch + 1)
    query = (q_start + query_relative).to(tl.int64)
    if query >= q_end:
        return
    k_start = tl.load(cu_k + batch).to(tl.int64)
    kv_len = tl.load(kv_lens + batch).to(tl.int64)
    visible = query_relative + kv_len - (q_end - q_start) + 1
    row_topk = tl.minimum(topk, visible)
    row_limit = tl.minimum(topk, ((row_topk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N)
    offs_h = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_values = tl.load(
        q
        + query * sq_m
        + (group * GROUP_SIZE + offs_h[:, None]) * sq_h
        + offs_d[None, :] * sq_d,
        mask=(offs_h < GROUP_SIZE)[:, None],
        other=0.0,
    )
    q_values = (q_values * scale * 1.4426950408).to(q_values.dtype)
    k_base = k + k_start * sk_n + group * sk_h
    v_base = v + k_start * sv_n + group * sv_h
    idx_row = indices + query * si_m + group * si_g
    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for start in range(0, row_limit, BLOCK_N):
        current = start + offs_n
        token = tl.load(idx_row + current * si_n, mask=current < topk, other=-1)
        valid = token >= 0
        keys = tl.load(
            k_base + token[None, :] * sk_n + offs_d[:, None] * sk_d,
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_base + token[:, None] * sv_n + offs_d[None, :] * sv_d,
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.where(valid[None, :], tl.dot(q_values, keys), -float("inf"))
        next_max = tl.maximum(max_value, tl.max(scores, 1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.math.exp2(scores - next_max[:, None])
        accumulator = tl.dot(
            probabilities.to(values.dtype), values, accumulator * alpha[:, None]
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, 1)
        max_value = next_max
    output = accumulator / tl.where(normalizer[:, None] > 0, normalizer[:, None], 1.0)
    tl.store(
        out
        + query * so_m
        + (group * GROUP_SIZE + offs_h[:, None]) * so_h
        + offs_d[None, :] * so_d,
        output,
        mask=(offs_h < GROUP_SIZE)[:, None],
    )


def sparse_gqa_fwd_interface_triton_ck(
    q, k, v, indices, cu_q, cu_k, kv_lens, scale, out=None
):
    k, v = k.contiguous(), v.contiguous()
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group_size = num_q_heads // num_kv_heads
    # Decode has exactly one query token per request. Avoid a device-to-host
    # max reduction in that case so the path remains CUDA-graph capturable.
    if q.shape[0] == cu_q.shape[0] - 1:
        max_q = 1
    else:
        max_q = int((cu_q[1:] - cu_q[:-1]).max().item())
    block_m = max(16, triton.next_power_of_2(group_size))
    block_n, warps, stages = _get_best_config(total_q)
    if out is None:
        out = torch.empty_like(q)
    elif out.shape != q.shape or out.device != q.device or out.dtype != q.dtype:
        raise ValueError("packed QSA output must match the query")
    _sparse_gqa_chunk_prefill[(max_q, (cu_q.shape[0] - 1) * num_kv_heads)](
        q,
        k,
        v,
        out,
        indices,
        cu_q,
        cu_k,
        kv_lens,
        scale,
        indices.shape[-1],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        indices.stride(0),
        indices.stride(1) if indices.ndim == 3 else 0,
        indices.stride(2) if indices.ndim == 3 else indices.stride(1),
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        num_warps=warps,
        num_stages=stages,
    )
    return out


@triton.jit
def _fa2_valid_counts(
    seq_lens,
    indices,
    counts,
    topk: tl.constexpr,
    stride_i: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_TOPK)
    length = tl.load(seq_lens + row)
    positions = tl.load(
        indices + row * stride_i + cols,
        mask=cols < topk,
        other=-1,
    )
    valid = (positions >= 0) & (positions < length)
    tl.store(counts + row, tl.sum(valid.to(tl.int32), axis=0))


@triton.jit
def _fa2_prefix_sum(counts, cu_k, batch, BLOCK_B: tl.constexpr):
    rows = tl.arange(0, BLOCK_B)
    valid_rows = rows < batch
    row_counts = tl.load(counts + rows, mask=valid_rows, other=0)
    tl.store(cu_k, 0)
    tl.store(cu_k + rows + 1, tl.cumsum(row_counts, 0), mask=valid_rows)


def qwen_sparse_fa2_cu_seqlens_triton(
    seq_lens, indices, counts, cu_k, batch, topk, block_b: Optional[int] = None
):
    block_b = block_b or triton.next_power_of_2(batch)
    # Count one request per program. The previous implementation formed a
    # [next_power_of_2(topk), next_power_of_2(batch)] tensor in one program;
    # topk=2051 and batch=512 therefore exceeded Triton's 1M-element limit.
    _fa2_valid_counts[(batch,)](
        seq_lens,
        indices,
        counts,
        topk,
        indices.stride(0),
        BLOCK_TOPK=triton.next_power_of_2(topk),
        num_warps=8,
    )
    # Prefix sum is only over the batch dimension and remains a small 1-D
    # tensor, including during CUDA graph capture.
    _fa2_prefix_sum[(1,)](
        counts,
        cu_k,
        batch,
        BLOCK_B=block_b,
        num_warps=8,
    )


@triton.jit
def _compact_kv(
    k,
    v,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    topk: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    req_stride: tl.constexpr,
    idx_stride: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    cols = block * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
    dims = tl.arange(0, BLOCK_D)
    length = tl.load(seq_lens + batch)
    req = tl.load(req_indices + batch)
    pack_start = tl.load(cu_k + batch)
    valid_count = tl.load(cu_k + batch + 1) - pack_start
    positions = tl.load(indices + batch * idx_stride + cols, mask=cols < topk, other=-1)
    valid = (cols < valid_count) & (positions >= 0) & (positions < length)
    slots = tl.load(
        req_to_token + req * req_stride + tl.where(valid, positions, 0),
        mask=valid,
        other=0,
    )
    src = slots[:, None] * heads * dim + head * dim + dims[None, :]
    dst = (pack_start + cols)[:, None] * heads * dim + head * dim + dims[None, :]
    mask = valid[:, None] & (dims[None, :] < dim)
    tl.store(out_k + dst, tl.load(k + src, mask=mask, other=0.0), mask=mask)
    tl.store(out_v + dst, tl.load(v + src, mask=mask, other=0.0), mask=mask)


def qwen_sparse_valid_counts_triton(seq_lens, indices, counts, batch, topk):
    """Valid-count pass alone, for consumers that need per-row lengths but
    not the packed cu_seqlens prefix sum (trtllm paged decode packs rows at
    a fixed page-aligned stride instead)."""
    _fa2_valid_counts[(batch,)](
        seq_lens,
        indices,
        counts,
        topk,
        indices.stride(0),
        BLOCK_TOPK=triton.next_power_of_2(topk),
        num_warps=8,
    )


def qwen_sparse_kv_extraction_compact_triton(
    k, v, req_to_token, req_indices, indices, seq_lens, cu_k, out_k, out_v, batch, topk
):
    _, heads, dim = k.shape
    block_topk = 16
    _compact_kv[(batch, heads, triton.cdiv(topk, block_topk))](
        k,
        v,
        req_to_token,
        req_indices,
        indices,
        seq_lens,
        cu_k,
        out_k,
        out_v,
        topk,
        heads,
        dim,
        req_to_token.stride(0),
        indices.stride(0),
        BLOCK_TOPK=block_topk,
        BLOCK_D=triton.next_power_of_2(dim),
        num_warps=8,
    )


@triton.jit
def _paged_row_metadata(
    indices, token_to_req, seq_lens, table,
    ranks, row_requests, row_lengths, counts, packed_indices, cu_q,
    stride_idx_row: tl.constexpr, stride_idx_col: tl.constexpr,
    stride_req: tl.constexpr, stride_len: tl.constexpr,
    stride_table_row: tl.constexpr, stride_table_col: tl.constexpr,
    num_requests: tl.constexpr, table_width: tl.constexpr,
    num_pages: tl.constexpr, page_size: tl.constexpr,
    rows: tl.constexpr, topk: tl.constexpr, BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_K)
    req = tl.load(token_to_req + row * stride_req).to(tl.int64)
    valid_req = (req >= 0) & (req < num_requests)
    length = tl.load(seq_lens + req * stride_len, mask=valid_req, other=0)
    pos = tl.load(
        indices + row * stride_idx_row + cols * stride_idx_col,
        mask=cols < topk, other=-1,
    ).to(tl.int64)
    page = pos // page_size
    valid = (valid_req & (cols < topk) & (pos >= 0) & (pos < length)
             & (page < table_width))
    physical = tl.load(
        table + req * stride_table_row + page * stride_table_col,
        mask=valid, other=-1,
    )
    valid = valid & (physical >= 0) & (physical < num_pages)
    # Ranks span all tiles, including gaps in the original selection.  Gather
    # CTAs can therefore write disjoint positions in a valid packed prefix.
    rank = tl.cumsum(valid.to(tl.int32), axis=0) - 1
    count = tl.sum(valid.to(tl.int32), axis=0)
    tl.store(ranks + row * topk + cols,
             tl.where(valid, rank, -1), mask=cols < topk)
    tl.store(packed_indices + row * topk + cols,
             tl.where(cols < count, cols, -1), mask=cols < topk)
    tl.store(counts + row, count)
    tl.store(row_requests + row, tl.where(valid_req, req, 0))
    tl.store(row_lengths + row, length)
    tl.store(cu_q + row, row)
    if row == rows - 1:
        tl.store(cu_q + rows, rows)


@triton.jit
def _compact_paged_kv(
    k,
    v,
    block_table,
    req_indices,
    indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    stride_k_block: tl.constexpr,
    stride_k_token: tl.constexpr,
    stride_k_head: tl.constexpr,
    stride_k_dim: tl.constexpr,
    stride_v_block: tl.constexpr,
    stride_v_token: tl.constexpr,
    stride_v_head: tl.constexpr,
    stride_v_dim: tl.constexpr,
    num_blocks: tl.constexpr,
    topk: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    table_stride: tl.constexpr,
    idx_stride: tl.constexpr,
    table_col_stride: tl.constexpr,
    idx_col_stride: tl.constexpr,
    ranks,
    page_size: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compact selected KV directly from vLLM's block table.

    SGLang receives a per-token ``req_to_token`` table. vLLM stores the same
    mapping as block IDs, so deriving the physical slot inside this kernel
    avoids rebuilding a full request table on every decode step.
    """
    batch, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    cols = block * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
    dims = tl.arange(0, BLOCK_D)
    length = tl.load(seq_lens + batch)
    req = tl.load(req_indices + batch)
    pack_start = tl.load(cu_k + batch)
    positions = tl.load(
        indices + batch * idx_stride + cols * idx_col_stride,
        mask=cols < topk, other=-1,
    ).to(tl.int64)
    rank = tl.load(ranks + batch * topk + cols, mask=cols < topk, other=-1)
    valid = (cols < topk) & (rank >= 0) & (positions >= 0) & (positions < length)
    logical_page = positions // page_size
    block_id = tl.load(
        block_table + req * table_stride + logical_page * table_col_stride,
        mask=valid,
        other=0,
    )
    valid &= (block_id >= 0) & (block_id < num_blocks)
    # MUSA cache pages may have padding between physical blocks.  Address the
    # original cache strides rather than reshaping/copying the whole cache.
    block_id = block_id.to(tl.int64)
    page_offset = positions % page_size
    src_k = (
        block_id[:, None] * stride_k_block
        + page_offset[:, None] * stride_k_token
        + head * stride_k_head
        + dims[None, :] * stride_k_dim
    )
    src_v = (
        block_id[:, None] * stride_v_block
        + page_offset[:, None] * stride_v_token
        + head * stride_v_head
        + dims[None, :] * stride_v_dim
    )
    dst = (pack_start + rank).to(tl.int64)[:, None] * heads * dim + head * dim + dims[None, :]
    mask = valid[:, None] & (dims[None, :] < dim)
    tl.store(out_k + dst, tl.load(k + src_k, mask=mask, other=0.0), mask=mask)
    tl.store(out_v + dst, tl.load(v + src_v, mask=mask, other=0.0), mask=mask)


def qwen_sparse_kv_extraction_compact_paged_triton(
    k,
    v,
    block_table,
    req_indices,
    indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    batch,
    topk,
    page_size,
    ranks,
):
    _, _, heads, dim = k.shape
    block_topk = 16
    _compact_paged_kv[(batch, heads, triton.cdiv(topk, block_topk))](
        k,
        v,
        block_table,
        req_indices,
        indices,
        seq_lens,
        cu_k,
        out_k,
        out_v,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        k.shape[0],
        topk,
        heads,
        dim,
        # ``block_table`` is indexed as ``request * row_stride + page`` in
        # the kernel.  Passing stride(1) (normally 1) aliases every request
        # to the first row as soon as a request has more than one page.
        block_table.stride(0),
        indices.stride(0),
        block_table.stride(1),
        indices.stride(1),
        ranks,
        page_size,
        BLOCK_TOPK=block_topk,
        BLOCK_D=triton.next_power_of_2(dim),
        num_warps=8,
    )


__all__ = [
    "qwen_sparse_fa2_cu_seqlens_triton",
    "qwen_sparse_valid_counts_triton",
    "qwen_sparse_kv_extraction_compact_triton",
    "qwen_sparse_kv_extraction_compact_paged_triton",
    "sparse_gqa_fwd_interface_triton",
    "sparse_gqa_fwd_interface_triton_ck",
    "sparse_gqa_from_paged_triton",
    "sparse_gqa_decode_paged_triton",
]


def sparse_gqa_from_paged_triton(
    q,
    k_cache,
    v_cache,
    block_table,
    token_to_req,
    logical_indices,
    sequence_lengths,
    scale,
    out=None,
):
    """SGLang-MUSA sequence: compact paged KV -> packed sparse GQA."""
    profile_stages = os.getenv("VLLM_MUSA_QSA_STAGE_TIMING", "0") == "1"
    stage_events = {}

    def stage_begin(name):
        if profile_stages:
            start = torch.musa.Event(enable_timing=True)
            start.record()
            stage_events[name] = [start, None]

    def stage_end(name):
        if profile_stages:
            end = torch.musa.Event(enable_timing=True)
            end.record()
            stage_events[name][1] = end

    if q.ndim != 3 or logical_indices.ndim != 2:
        raise ValueError("QSA packed path expects q=[rows,heads,dim] and 2-D indices")
    rows, topk = logical_indices.shape
    if q.shape[0] != rows or rows == 0 or topk <= 0:
        raise ValueError("QSA query and selection rows must be non-empty and match")
    if k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError("QSA paged K/V caches must be 4-D NHD tensors")
    page_size = int(k_cache.shape[1])
    kv_heads = int(k_cache.shape[2])
    head_dim = int(k_cache.shape[3])
    if head_dim != 256 or v_cache.shape != k_cache.shape:
        raise ValueError(
            "QSA packed path requires matching paged K/V with head_dim=256"
        )
    if (
        page_size <= 0
        or kv_heads <= 0
        or q.shape[2] != head_dim
        or q.shape[1] % kv_heads
    ):
        raise ValueError("QSA query heads must be divisible by KV heads")
    if block_table.ndim != 2 or token_to_req.shape != (rows,):
        raise ValueError("QSA paged metadata has invalid shapes")
    if os.getenv("VLLM_MUSA_QSA_SGLANG_FUSED_DECODE", "0") == "1":
        fused = sparse_gqa_decode_paged_triton(
            q, k_cache, v_cache, block_table, token_to_req,
            logical_indices, sequence_lengths, scale,
        )
        if out is not None:
            if out.shape != fused.shape or out.device != fused.device:
                raise ValueError("QSA output must match the query shape and device")
            out.copy_(fused)
            return out
        return fused
    req = token_to_req
    if req.device != q.device or req.dtype != torch.int32:
        req = req.to(device=q.device, dtype=torch.int32)
    indices = logical_indices
    if indices.device != q.device or indices.dtype != torch.int32:
        indices = indices.to(device=q.device, dtype=torch.int32)
    if sequence_lengths is None:
        raise ValueError("QSA packed path requires per-request sequence lengths")
    if sequence_lengths.ndim != 1:
        raise ValueError("QSA sequence lengths must be a one-dimensional tensor")
    seq_lens = sequence_lengths.to(device=q.device, dtype=torch.int32)
    if block_table.device != q.device or block_table.dtype != torch.int32:
        block_table = block_table.to(device=q.device, dtype=torch.int32)
    (
        counts,
        cu_k,
        packed_k,
        packed_v,
        cu_q,
        packed_indices,
        ranks,
        row_requests,
        row_seq_lens,
    ) = _decode_workspace(q.device, k_cache.dtype, rows, topk, kv_heads, head_dim)
    stage_begin("row_metadata")
    _paged_row_metadata[(rows,)](
        indices,
        req,
        seq_lens,
        block_table,
        ranks,
        row_requests,
        row_seq_lens,
        counts,
        packed_indices,
        cu_q,
        indices.stride(0),
        indices.stride(1),
        req.stride(0),
        seq_lens.stride(0),
        block_table.stride(0),
        block_table.stride(1),
        min(seq_lens.numel(), block_table.shape[0]),
        block_table.shape[1],
        k_cache.shape[0],
        page_size,
        rows,
        topk,
        BLOCK_K=triton.next_power_of_2(topk),
        num_warps=4,
    )
    _fa2_prefix_sum[(1,)](
        counts,
        cu_k,
        rows,
        BLOCK_B=triton.next_power_of_2(rows),
    )
    stage_end("row_metadata")
    stage_begin("compact")
    qwen_sparse_kv_extraction_compact_paged_triton(
        k_cache,
        v_cache,
        block_table,
        row_requests,
        indices,
        row_seq_lens,
        cu_k,
        packed_k,
        packed_v,
        rows,
        topk,
        page_size,
        ranks,
    )
    stage_end("compact")
    stage_begin("gqa_attention")
    result = sparse_gqa_fwd_interface_triton_ck(
        q,
        packed_k,
        packed_v,
        packed_indices,
        cu_q,
        cu_k,
        counts,
        scale,
        out=out,
    )
    stage_end("gqa_attention")
    if profile_stages:
        torch.musa.synchronize()
        _logger.warning(
            "QSA stage timing ms=%s",
            {
                name: start.elapsed_time(end)
                for name, (start, end) in stage_events.items()
            },
        )
    # The vLLM owner supplies its destination tensor and deliberately ignores
    # the operator return value.  The packed kernel writes it directly; the
    # copy fallback only applies when this adapter is called without ``out``.
    if out is not None and result is not out:
        if out.shape != result.shape or out.device != result.device:
            raise ValueError("QSA output must match the query shape and device")
        out.copy_(result)
        return out
    return result
