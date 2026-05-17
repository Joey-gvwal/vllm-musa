# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch
import torch.nn.functional as F

_FP8_E4M3_MAX = 448.0
_UE8M0_BIAS = 127


def _require_contiguous(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _scale_bytes_to_float(scale_bytes: torch.Tensor) -> torch.Tensor:
    return (scale_bytes.to(torch.int32) << 23).view(torch.float32)


def _write_ue8m0_scales(output_scale: torch.Tensor,
                        scale_bytes: torch.Tensor) -> None:
    scale_bytes = scale_bytes.contiguous()
    if output_scale.dtype == torch.uint8:
        if output_scale.numel() != scale_bytes.numel():
            raise ValueError(
                "uint8 output_scale must have one element per quant group")
        output_scale.view(-1).copy_(scale_bytes.view(-1))
        return

    if output_scale.dtype == torch.int32:
        if output_scale.numel() * 4 != scale_bytes.numel():
            raise ValueError(
                "int32 output_scale must pack exactly four UE8M0 scales")
        output_scale.zero_()
        output_scale.view(torch.uint8).view(-1).copy_(scale_bytes.view(-1))
        return

    if output_scale.dtype == torch.float32:
        if output_scale.numel() != scale_bytes.numel():
            raise ValueError(
                "float32 output_scale must have one element per quant group")
        output_scale.view(-1).copy_(
            _scale_bytes_to_float(scale_bytes).view(-1))
        return

    raise TypeError(f"unsupported output_scale dtype {output_scale.dtype}")


def quantize_fp8_e4m3_ue8m0(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
) -> None:
    """Quantize the last dimension to FP8 E4M3 with UE8M0 group scales."""
    _require_contiguous("input", input)
    _require_contiguous("output", output)
    _require_contiguous("output_scale", output_scale)
    if input.shape != output.shape:
        raise ValueError(
            f"input/output shape mismatch: {input.shape} vs {output.shape}")
    if group_size not in (32, 64, 128):
        raise ValueError(f"unsupported group_size={group_size}")
    if input.shape[-1] % group_size != 0:
        raise ValueError("last dimension must be divisible by group_size")

    rows = input.numel() // input.shape[-1]
    num_groups = input.shape[-1] // group_size
    grouped = input.float().reshape(rows, num_groups, group_size)
    raw_scale = grouped.abs().amax(dim=-1).clamp(min=eps) / _FP8_E4M3_MAX
    scale_exp = torch.ceil(torch.log2(raw_scale)).to(torch.int32)
    scale_bytes = torch.clamp(scale_exp + _UE8M0_BIAS, 0,
                              255).to(torch.uint8)
    scale = _scale_bytes_to_float(scale_bytes).unsqueeze(-1)

    quantized = torch.clamp(grouped / scale, -_FP8_E4M3_MAX,
                            _FP8_E4M3_MAX).reshape_as(input)
    output.copy_(quantized.to(output.dtype))
    _write_ue8m0_scales(output_scale, scale_bytes)


def deepseek_v4_mega_moe_pre_dispatch(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    buf_x: torch.Tensor,
    buf_x_sf: torch.Tensor,
    buf_topk_idx: torch.Tensor,
    buf_topk_weights: torch.Tensor,
    quant_group_size: int = 32,
) -> None:
    """Prepare routed activations and metadata for DeepSeek-V4 MegaMoE."""
    _require_contiguous("x", x)
    _require_contiguous("topk_idx", topk_idx)
    _require_contiguous("topk_weights", topk_weights)
    _require_contiguous("buf_x", buf_x)
    _require_contiguous("buf_x_sf", buf_x_sf)
    _require_contiguous("buf_topk_idx", buf_topk_idx)
    _require_contiguous("buf_topk_weights", buf_topk_weights)

    if x.dim() != 2:
        raise ValueError("x must have shape [num_tokens, hidden]")
    if topk_idx.shape != topk_weights.shape:
        raise ValueError("topk_idx and topk_weights must have the same shape")
    if topk_idx.dim() != 2 or topk_idx.shape[0] != x.shape[0]:
        raise ValueError("topk tensors must have shape [num_tokens, top_k]")
    if buf_x.shape[0] < x.shape[0] or buf_x.shape[1] != x.shape[1]:
        raise ValueError("buf_x must have shape [padded_max, hidden]")
    if buf_topk_idx.shape != buf_topk_weights.shape:
        raise ValueError("topk output buffers must have the same shape")
    if buf_topk_idx.shape[0] != buf_x.shape[0]:
        raise ValueError("topk output buffers must match padded_max")
    if buf_topk_idx.shape[1] != topk_idx.shape[1]:
        raise ValueError("topk output buffers must match top_k")

    num_tokens = x.shape[0]
    quantize_fp8_e4m3_ue8m0(
        x,
        buf_x[:num_tokens],
        _scale_prefix(buf_x_sf, num_tokens, x.shape[1], quant_group_size),
        quant_group_size,
    )
    buf_topk_idx[:num_tokens].copy_(topk_idx.to(buf_topk_idx.dtype))
    buf_topk_weights[:num_tokens].copy_(topk_weights.to(buf_topk_weights.dtype))
    if buf_x.shape[0] > num_tokens:
        buf_topk_idx[num_tokens:].fill_(-1)
        buf_topk_weights[num_tokens:].zero_()


def _scale_prefix(scale: torch.Tensor, rows: int, hidden: int,
                  group_size: int) -> torch.Tensor:
    groups = hidden // group_size
    if scale.dtype == torch.int32:
        if groups % 4 != 0:
            raise ValueError("int32 UE8M0 scale buffer requires groups % 4 == 0")
        return scale[:rows, :groups // 4]
    return scale[:rows, :groups]


def _silu_and_mul(input: torch.Tensor,
                  swiglu_limit: float | None = None) -> torch.Tensor:
    gate, up = input.chunk(2, dim=-1)
    gate = gate.float()
    up = up.float()
    if swiglu_limit is not None and swiglu_limit > 0:
        gate = torch.clamp(gate, max=swiglu_limit)
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
    return F.silu(gate) * up


def deepseek_v4_silu_and_mul_masked_post_quant(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    quant_group_size: int,
    masked_m: torch.Tensor,
    swiglu_limit: float | None = None,
) -> None:
    """Apply SiLU/SwiGLU to valid expert rows and quantize to FP8/UE8M0."""
    _require_contiguous("input", input)
    _require_contiguous("output", output)
    _require_contiguous("output_scale", output_scale)
    _require_contiguous("masked_m", masked_m)
    if input.shape[:-1] != output.shape[:-1]:
        raise ValueError("input/output leading dimensions must match")
    if input.shape[-1] != output.shape[-1] * 2:
        raise ValueError("input last dimension must be twice output last dim")
    if input.dim() not in (2, 3):
        raise ValueError("input must be 2D or 3D")

    if input.dim() == 2:
        if masked_m.numel() != 1:
            raise ValueError("2D input requires scalar masked_m")
        valid_rows = int(masked_m.item())
        if valid_rows <= 0:
            return
        value = _silu_and_mul(input[:valid_rows], swiglu_limit).contiguous()
        quantize_fp8_e4m3_ue8m0(
            value,
            output[:valid_rows],
            _scale_prefix(output_scale, valid_rows, output.shape[-1],
                          quant_group_size),
            quant_group_size,
        )
        return

    if masked_m.numel() != input.shape[0]:
        raise ValueError("3D input requires one masked_m entry per expert")
    for expert_id in range(input.shape[0]):
        valid_rows = int(masked_m[expert_id].item())
        if valid_rows <= 0:
            continue
        value = _silu_and_mul(input[expert_id, :valid_rows],
                              swiglu_limit).contiguous()
        quantize_fp8_e4m3_ue8m0(
            value,
            output[expert_id, :valid_rows],
            _scale_prefix(output_scale[expert_id], valid_rows,
                          output.shape[-1], quant_group_size),
            quant_group_size,
        )
