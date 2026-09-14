"""Decode-sized Qwen4Exp PLE short-convolution state movement."""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _gate_value_kernel(
    gate_ptr,
    value_ptr,
    out_ptr,
    n_elements,
    H: tl.constexpr,
    HC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    stream = tl.program_id(1)
    col = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    mask = (token < n_elements) & (col < H)
    value = tl.load(value_ptr + token * H + col, mask=mask, other=0.0)
    gate = tl.load(gate_ptr + token * HC + stream, mask=token < n_elements, other=0.0).to(tl.float32)
    gate = gate * tl.rsqrt(tl.maximum(tl.abs(gate), 1e-6))
    tl.store(out_ptr + (token * HC + stream) * H + col, value.to(tl.float32) * tl.sigmoid(gate), mask=mask)


def _fused_gate_value(gate: torch.Tensor, value: torch.Tensor, hc_count: int) -> torch.Tensor:
    if (
        gate.ndim != 3
        or gate.shape[2] != 1
        or value.ndim != 2
        or gate.shape[0] != value.shape[0]
        or gate.dtype != value.dtype
        or gate.dtype not in (torch.bfloat16, torch.float16)
        or not gate.is_contiguous()
        or not value.is_contiguous()
        or gate.shape[1] != hc_count
    ):
        raise ValueError("unsupported Qwen4 PLE gate-value input")
    out = torch.empty(
        (value.shape[0], hc_count, value.shape[1]), dtype=value.dtype, device=value.device
    )
    _gate_value_kernel[(value.shape[0], hc_count, triton.cdiv(value.shape[1], 256))](
        gate, value, out, value.shape[0], value.shape[1], hc_count, BLOCK=256, num_warps=4
    )
    return out


def _fused_gate_value_fake(gate: torch.Tensor, value: torch.Tensor, hc_count: int) -> torch.Tensor:
    del gate
    return value.new_empty((value.shape[0], hc_count, value.shape[1]))


direct_register_custom_op(
    op_name="qwen4_exp_ple_gate_value",
    op_func=_fused_gate_value,
    fake_impl=_fused_gate_value_fake,
)


def fused_qwen4_gate_value(gate: torch.Tensor, value: torch.Tensor, hc_count: int) -> torch.Tensor:
    return torch.ops.vllm.qwen4_exp_ple_gate_value(gate, value, hc_count)


@triton.jit
def _short_conv_decode_kernel(
    state_ptr,
    state_indices_ptr,
    has_initial_ptr,
    x_ptr,
    weight_ptr,
    out_ptr,
    batch,
    CHANNELS: tl.constexpr,
    STATE_LEN: tl.constexpr,
    DILATION: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
    STATE_SLOT_STRIDE: tl.constexpr,
    STATE_CHANNEL_STRIDE: tl.constexpr,
    STATE_COL_STRIDE: tl.constexpr,
):
    row = tl.program_id(0)
    channel = tl.program_id(1) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
    mask = (row < batch) & (channel < CHANNELS)
    state_index = tl.load(state_indices_ptr + row, mask=row < batch, other=0)
    valid = mask & (state_index != 0)
    has_initial = tl.load(has_initial_ptr + row, mask=row < batch, other=0)
    state_base = (
        state_index * STATE_SLOT_STRIDE + channel * STATE_CHANNEL_STRIDE
    )
    x = tl.load(x_ptr + row * CHANNELS + channel, mask=mask, other=0.0).to(tl.float32)
    w0 = tl.load(weight_ptr + channel * 4, mask=mask, other=0.0).to(tl.float32)
    w1 = tl.load(weight_ptr + channel * 4 + 1, mask=mask, other=0.0).to(tl.float32)
    w2 = tl.load(weight_ptr + channel * 4 + 2, mask=mask, other=0.0).to(tl.float32)
    w3 = tl.load(weight_ptr + channel * 4 + 3, mask=mask, other=0.0).to(tl.float32)
    s0 = tl.load(state_ptr + state_base, mask=valid & has_initial, other=0.0).to(tl.float32)
    s1 = tl.load(state_ptr + state_base + DILATION * STATE_COL_STRIDE, mask=valid & has_initial, other=0.0).to(tl.float32)
    s2 = tl.load(state_ptr + state_base + 2 * DILATION * STATE_COL_STRIDE, mask=valid & has_initial, other=0.0).to(tl.float32)
    acc = s0 * w0 + s1 * w1 + s2 * w2 + x * w3
    acc = acc * tl.sigmoid(acc)
    tl.store(
        out_ptr + row * CHANNELS + channel,
        tl.where(valid, acc, 0.0),
        mask=mask,
    )
    # Advance the dilated history in place.  State values beyond the three
    # sampled positions still matter on the next decode step.
    for i in range(STATE_LEN - 1):
        next_value = tl.load(
            state_ptr + state_base + (i + 1) * STATE_COL_STRIDE,
            mask=valid & has_initial,
            other=0.0,
        )
        tl.store(state_ptr + state_base + i * STATE_COL_STRIDE, next_value, mask=valid)
    tl.store(
        state_ptr + state_base + (STATE_LEN - 1) * STATE_COL_STRIDE,
        x,
        mask=valid,
    )


@triton.jit
def _state_to_conv_input(
    state_ptr,
    state_indices_ptr,
    has_initial_ptr,
    x_ptr,
    conv_input_ptr,
    num_tokens,
    CHANNELS: tl.constexpr,
    STATE_LEN: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
    BLOCK_STATE_LEN: tl.constexpr,
):
    token = tl.program_id(0)
    channel = tl.program_id(1) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)[:, None]
    state_col = tl.arange(0, BLOCK_STATE_LEN)[None, :]
    channel_mask = (token < num_tokens) & (channel < CHANNELS)
    state_mask = channel_mask & (state_col < STATE_LEN)
    state_index = tl.load(state_indices_ptr + token, mask=token < num_tokens, other=0)
    has_initial = tl.load(has_initial_ptr + token, mask=token < num_tokens, other=0)
    state_base = state_index * CHANNELS * STATE_LEN
    state_offset = state_base + channel * STATE_LEN + state_col
    output_base = token * CHANNELS * (STATE_LEN + 1)
    output_offset = output_base + channel * (STATE_LEN + 1) + state_col

    old_state = tl.load(
        state_ptr + state_offset,
        mask=state_mask & has_initial,
        other=0.0,
    )
    tl.store(conv_input_ptr + output_offset, old_state, mask=state_mask)
    x = tl.load(x_ptr + token * CHANNELS + channel, mask=channel_mask, other=0.0)
    tl.store(
        conv_input_ptr + output_base + channel * (STATE_LEN + 1) + STATE_LEN,
        x,
        mask=channel_mask,
    )
    tl.debug_barrier()

    # Slot zero is the graph padding slot and may be shared by many rows.
    update_mask = state_mask & (state_col < STATE_LEN - 1) & (state_index != 0)
    next_value = tl.load(
        conv_input_ptr + output_offset + 1, mask=update_mask, other=0.0
    )
    tl.store(state_ptr + state_offset, next_value, mask=update_mask)
    tl.store(
        state_ptr + state_base + channel * STATE_LEN + STATE_LEN - 1,
        x,
        mask=channel_mask & (state_index != 0),
    )


def can_fuse_qwen4_short_conv_state(
    state: torch.Tensor, state_indices: torch.Tensor, x: torch.Tensor
) -> bool:
    return (
        state.is_cuda
        and state.dtype in (torch.bfloat16, torch.float16)
        and state.ndim == 3
        and all(s >= 0 for s in state.stride())
        and 0 < state.shape[2] <= 32
        and state_indices.is_cuda
        and state_indices.dtype == torch.long
        and state_indices.ndim == 1
        and state_indices.is_contiguous()
        and x.is_cuda
        and x.dtype == state.dtype
        and x.ndim == 2
        and x.is_contiguous()
        and x.shape == (state_indices.shape[0], state.shape[1])
    )


def fused_qwen4_short_conv_state(
    state: torch.Tensor,
    state_indices: torch.Tensor,
    x: torch.Tensor,
    has_initial: torch.Tensor,
) -> torch.Tensor:
    if not state.is_contiguous() or not can_fuse_qwen4_short_conv_state(
        state, state_indices, x
    ):
        raise ValueError("unsupported input for fused Qwen4 short-conv state")
    if (
        has_initial.device != state.device
        or has_initial.dtype != torch.bool
        or has_initial.ndim != 1
        or has_initial.numel() < state_indices.numel()
    ):
        raise ValueError("invalid initial-state mask")
    state_len = state.shape[2]
    conv_input = torch.empty(
        (x.shape[0], x.shape[1], state_len + 1), dtype=x.dtype, device=x.device
    )
    if x.shape[0]:
        _state_to_conv_input[(x.shape[0], triton.cdiv(x.shape[1], 128))](
            state,
            state_indices,
            has_initial,
            x,
            conv_input,
            x.shape[0],
            CHANNELS=state.shape[1],
            STATE_LEN=state_len,
            BLOCK_CHANNELS=128,
            BLOCK_STATE_LEN=triton.next_power_of_2(state_len),
            num_warps=8,
        )
    return conv_input


def fused_qwen4_short_conv_decode(
    state: torch.Tensor,
    state_indices: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    has_initial: torch.Tensor,
    dilation: int,
) -> torch.Tensor:
    """Compute dilated width-4 PLE decode and advance state in one launch."""
    if (
        not can_fuse_qwen4_short_conv_state(state, state_indices, x)
        or weight.ndim != 2
        or weight.shape != (state.shape[1], 4)
        or weight.dtype != x.dtype
        or not weight.is_contiguous()
        or dilation <= 0
        or state.shape[2] != 3 * dilation
        or has_initial.device != state.device
        or has_initial.dtype != torch.bool
        or has_initial.ndim != 1
        or has_initial.numel() < state_indices.numel()
    ):
        raise ValueError("unsupported input for fused Qwen4 short-conv decode")
    out = torch.empty_like(x)
    if x.shape[0]:
        _short_conv_decode_kernel[(x.shape[0], triton.cdiv(x.shape[1], 128))](
            state,
            state_indices,
            has_initial,
            x,
            weight,
            out,
            x.shape[0],
            CHANNELS=state.shape[1],
            STATE_LEN=state.shape[2],
            DILATION=dilation,
            BLOCK_CHANNELS=128,
            STATE_SLOT_STRIDE=state.stride(0),
            STATE_CHANNEL_STRIDE=state.stride(1),
            STATE_COL_STRIDE=state.stride(2),
            num_warps=8,
        )
    return out
