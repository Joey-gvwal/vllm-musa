# SPDX-License-Identifier: Apache-2.0
"""FP8 MoE whose gate/up projection steps K by 128 where the expert scales allow.

Expert weights stored in 128x128 FP8 blocks are refined to 32x32 when the TP
shard's intermediate size is not a multiple of 128, and the upstream kernel then
steps K by 32 with a scale multiply per step. Along the gate/up projection's K,
the hidden size, the refined scale still changes only every 128 elements, so with
the activation quantized per 128-wide group one scale multiply covers a 128-wide
step. Experts whose scale does change within 128 (a folded shared expert with its
own 32x32 blocks) take 32-wide steps in the same launch. The activation, the down
projection and the sum are the upstream ones.
"""

from __future__ import annotations

import os

import torch
from vllm.triton_utils import tl, triton

_ENABLED = os.environ.get("VLLM_MUSA_FP8_MOE_W1_K128", "1") != "0"
# Gate/up tile measured on MTT S5000 for the DeepSeek-V4.1 TP8 experts. It stays
# the same for every batch, so a token's output does not depend on its batch.
_BLOCK_N = 32
_NUM_WARPS = 4
# Attribute of a w1 scale tensor holding its per-expert flags (1 where the scale
# changes inside a 128-wide K group) and how many experts are flagged.
_FINE_ATTR = "_musa_k128_fine_experts"


@triton.jit
def _moe_w1_k128_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    fine_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_be,
    stride_bn,
    stride_cm,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsn,
    TOP_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(EM, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    first_pid_m = (pid // num_pid_in_group) * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    num_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_M >= num_post_padded:
        return
    # Highest expert ids first: a folded shared expert sorts last and takes the most
    # K steps, so it starts in the first wave instead of finishing alone.
    pid_m = tl.cdiv(num_post_padded, BLOCK_M) - 1 - pid_m

    offs_token = tl.load(
        sorted_token_ids_ptr + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    ).to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_token[:, None] * stride_cm + offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if expert == -1:
        zeros = tl.zeros((BLOCK_M, BLOCK_N), dtype=c_ptr.dtype.element_ty)
        tl.store(c_ptrs, zeros, mask=c_mask)
        return

    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)) % N
    rows = offs_token // TOP_K
    a_rows = a_ptr + rows[:, None] * stride_am
    as_rows = a_scale_ptr + rows * stride_asm
    b_cols = b_ptr + expert * stride_be + offs_bn[None, :] * stride_bn
    bs_cols = b_scale_ptr + expert * stride_bse + (offs_bn // 32) * stride_bsn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if tl.load(fine_ptr + expert) != 0:
        offs_k32 = tl.arange(0, 32)
        for g in range(0, K // 32):
            a32 = tl.load(
                a_rows + g * 32 + offs_k32[None, :], mask=token_mask[:, None], other=0.0
            )
            b32 = tl.load(b_cols + g * 32 + offs_k32[:, None])
            as32 = tl.load(as_rows + (g // 4) * stride_ask, mask=token_mask, other=0.0)
            bs32 = tl.load(bs_cols + g)
            acc += tl.dot(a32, b32) * as32[:, None] * bs32[None, :]
    else:
        offs_k128 = tl.arange(0, 128)
        for g in range(0, K // 128):
            a128 = tl.load(
                a_rows + g * 128 + offs_k128[None, :],
                mask=token_mask[:, None],
                other=0.0,
            )
            b128 = tl.load(b_cols + g * 128 + offs_k128[:, None])
            as128 = tl.load(as_rows + g * stride_ask, mask=token_mask, other=0.0)
            # The first of the four equal 32-wide scales of this 128-wide group.
            bs128 = tl.load(bs_cols + g * 4)
            acc += tl.dot(a128, b128) * as128[:, None] * bs128[None, :]
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


def _fine_experts(w1_scale: torch.Tensor) -> tuple[torch.Tensor, int] | None:
    """Per-expert flags of ``w1_scale`` (E, N/32, K/32), 1 where an expert's scale
    changes inside a 128-wide K group, and their count. Computed on the first call
    outside graph capture and kept on the tensor."""
    hit = getattr(w1_scale, _FINE_ATTR, None)
    if hit is not None or torch.cuda.is_current_stream_capturing():
        return hit
    e, n, k = w1_scale.shape
    groups = w1_scale.view(e, n, k // 4, 4)
    fine = (groups != groups[..., :1]).flatten(1).any(dim=1).to(torch.int32)
    hit = (fine, int(fine.sum().item()))
    setattr(w1_scale, _FINE_ATTR, hit)
    return hit


def fp8_moe_k128_supported(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor | None,
    block_shape: list[int] | None,
) -> bool:
    """32x32-block FP8 experts whose w1 scales mostly repeat over 128-wide K groups."""
    if (
        not _ENABLED
        or block_shape is None
        or list(block_shape) != [32, 32]
        or w1_scale is None
        or hidden_states.dim() != 2
        or hidden_states.dtype != torch.bfloat16
        or not hidden_states.is_contiguous()
        or w1.dim() != 3
        or w1.dtype != torch.float8_e4m3fn
        or not w1.is_contiguous()
    ):
        return False
    e, n, k = w1.shape
    if (
        k % 128
        or n % _BLOCK_N
        or w1_scale.dtype != torch.float32
        or not w1_scale.is_contiguous()
        or tuple(w1_scale.shape) != (e, n // 32, k // 32)
    ):
        return False
    hit = _fine_experts(w1_scale)
    # Flagged experts run 32-wide steps; the path pays off while they are few.
    return hit is not None and hit[1] * 8 <= e


def moe_w1_k128(
    xq: torch.Tensor,
    xs: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    out: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_post_padded: torch.Tensor,
    config: dict[str, int],
) -> None:
    """Gate/up projection of the expert-sorted rows of ``xq`` (FP8 with one scale per
    128-wide K group in ``xs``) into ``out`` (tokens, top_k, N)."""
    hit = _fine_experts(w1_scale)
    assert hit is not None, "fine-expert flags are computed outside graph capture"
    num_tokens, top_k, n = out.shape
    block_m = config["BLOCK_SIZE_M"]
    em = sorted_ids.shape[0]
    if num_tokens < block_m:
        em = min(em, num_tokens * top_k * block_m)
    grid = (triton.cdiv(em, block_m) * triton.cdiv(n, _BLOCK_N),)
    _moe_w1_k128_kernel[grid](
        xq,
        w1,
        out,
        xs,
        w1_scale,
        hit[0],
        sorted_ids,
        expert_ids,
        num_post_padded,
        n,
        w1.shape[2],
        em,
        num_tokens * top_k,
        xq.stride(0),
        w1.stride(0),
        w1.stride(1),
        out.stride(1),
        xs.stride(0),
        xs.stride(1),
        w1_scale.stride(0),
        w1_scale.stride(1),
        TOP_K=top_k,
        BLOCK_M=block_m,
        BLOCK_N=_BLOCK_N,
        GROUP_M=config.get("GROUP_SIZE_M", 1),
        num_warps=_NUM_WARPS,
        num_stages=1,
    )


def fp8_moe_k128_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    block_shape: list[int],
    global_num_experts: int,
    gemm1_clamp_limit: float | None,
) -> torch.Tensor:
    """The upstream FP8 block MoE with the SiLU activation, its gate/up projection on
    128-wide K steps (``fp8_moe_k128_supported`` must hold)."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe import fused_moe as fm
    from vllm.model_executor.layers.fused_moe.activation import (
        ApplyMoEActivationConfig,
        MoEActivation,
        apply_moe_activation,
    )
    from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input

    from vllm_musa.jit_kernel.csrc.moe import maybe_fast_moe_sum

    num_tokens, hidden = hidden_states.shape
    e, n, _ = w1.shape
    top_k = topk_ids.shape[1]
    if global_num_experts == -1:
        global_num_experts = e
    config = fm.try_get_optimal_moe_config(
        w1.size(), w2.size(), top_k, "fp8_w8a8", num_tokens, block_shape=block_shape
    )
    cache13 = torch.empty(
        num_tokens * top_k * max(n, hidden),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    cache1 = cache13[: num_tokens * top_k * n].view(num_tokens, top_k, n)
    cache3 = cache13[: num_tokens * top_k * hidden].view(num_tokens, top_k, hidden)
    cache2 = torch.empty(
        (num_tokens * top_k, n // 2),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    out = torch.empty_like(hidden_states)

    xq, xs = moe_kernel_quantize_input(
        A=hidden_states,
        A_scale=None,
        quant_dtype=torch.float8_e4m3fn,
        per_act_token_quant=False,
        block_shape=[block_shape[0], 128],
    )
    sorted_ids, expert_ids, num_post_padded = fm._prepare_expert_assignment(
        topk_ids,
        config,
        num_tokens,
        top_k,
        global_num_experts,
        None,
        block_shape=block_shape,
        ignore_invalid_experts=True,
    )
    moe_w1_k128(
        xq, xs, w1, w1_scale, cache1, sorted_ids, expert_ids, num_post_padded, config
    )
    apply_moe_activation(
        MoEActivation.SILU,
        cache2,
        cache1.view(-1, n),
        activation_config=(
            ApplyMoEActivationConfig(clamp_limit=gemm1_clamp_limit)
            if gemm1_clamp_limit is not None
            else None
        ),
    )
    q2, s2 = moe_kernel_quantize_input(
        A=cache2,
        A_scale=None,
        quant_dtype=torch.float8_e4m3fn,
        per_act_token_quant=False,
        block_shape=block_shape,
    )
    fm.dispatch_fused_moe_kernel(
        q2,
        w2,
        cache3,
        s2,
        w2_scale,
        None,
        topk_weights,
        sorted_ids,
        expert_ids,
        num_post_padded,
        True,
        1,
        config,
        compute_type=tl.bfloat16,
        use_fp8_w8a8=True,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        block_shape=block_shape,
        B_bias=None,
    )
    if not maybe_fast_moe_sum(cache3, out):
        ops.moe_sum(cache3, out)
    return out
