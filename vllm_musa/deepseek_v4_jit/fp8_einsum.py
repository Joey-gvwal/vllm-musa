# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 FP8 einsum helpers for MUSA provider experiments."""

from __future__ import annotations

import os

import torch

_GROUP_SIZE = 128
_IMPL_ENV = "VLLM_MUSA_DEEPSEEK_V4_FP8_EINSUM_IMPL"


def _impl_mode() -> str:
    return os.getenv(_IMPL_ENV, "auto").strip().lower()


def _is_musa_tensor(tensor: torch.Tensor) -> bool:
    return getattr(tensor, "device", None) is not None and tensor.device.type == "musa"


def _normalize_weight(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    groups: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    if weight.dim() == 2:
        flat_out_dim, in_dim = weight.shape
        if flat_out_dim % groups != 0:
            raise ValueError(
                "DeepSeek-V4 FP8 GEMV provider expected 2D weight rows to be "
                f"divisible by groups={groups}, got {tuple(weight.shape)}"
            )
        out_dim = flat_out_dim // groups
        weight = weight.reshape(groups, out_dim, in_dim)
    elif weight.dim() == 3:
        weight_groups, out_dim, in_dim = weight.shape
        if weight_groups != groups:
            raise ValueError(
                "DeepSeek-V4 FP8 GEMV provider group mismatch: "
                f"activation groups={groups}, weight groups={weight_groups}"
            )
    else:
        raise ValueError(
            "DeepSeek-V4 FP8 GEMV provider expects a 2D or 3D weight tensor, "
            f"got {tuple(weight.shape)}"
        )

    if out_dim % _GROUP_SIZE != 0 or in_dim % _GROUP_SIZE != 0:
        raise ValueError(
            "DeepSeek-V4 FP8 GEMV provider requires 128-aligned dimensions, "
            f"got out_dim={out_dim}, in_dim={in_dim}"
        )
    out_blocks = out_dim // _GROUP_SIZE
    in_blocks = in_dim // _GROUP_SIZE

    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is not None and weight_scale.dtype == e8m0_dtype:
        exp_bits = weight_scale.view(torch.uint8).to(torch.int32)
        scales = (exp_bits << 23).view(torch.float32)
    else:
        scales = weight_scale.to(torch.float32)

    if scales.dim() == 2 and groups == 1 and scales.shape == (out_blocks, in_blocks):
        scales = scales.unsqueeze(0)
    elif scales.dim() == 2 and scales.numel() == groups * out_blocks * in_blocks:
        scales = scales.reshape(groups, out_blocks, in_blocks)
    elif scales.shape == (groups, in_blocks, out_blocks):
        scales = scales.transpose(-1, -2)

    if scales.shape != (groups, out_blocks, in_blocks):
        raise ValueError(
            "DeepSeek-V4 FP8 GEMV provider scale shape mismatch: "
            f"scale_shape={tuple(weight_scale.shape)}, normalized={tuple(scales.shape)}, "
            f"expected={(groups, out_blocks, in_blocks)}"
        )

    return weight, scales, out_dim, in_dim


def try_musa_deepseek_v4_fp8_einsum_gemv(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
) -> tuple[bool, str]:
    """Try the native MUSA GEMV path for ``bhr,hdr->bhd`` FP8 einsum."""
    mode = _impl_mode()
    if mode in {"torch", "fallback", "off", "0"}:
        return False, f"disabled by {_IMPL_ENV}"
    if equation != "bhr,hdr->bhd":
        return False, f"unsupported equation {equation!r}"
    if activation.dim() != 3 or out.dim() != 3:
        return False, (
            "expected activation/out shapes [tokens, groups, hidden] and "
            f"[tokens, groups, out], got {tuple(activation.shape)} -> "
            f"{tuple(out.shape)}"
        )
    if activation_scale.dim() != 3:
        return False, f"expected activation_scale dim 3, got {activation_scale.dim()}"
    if activation.dtype != torch.float8_e4m3fn:
        return False, f"expected fp8 activation, got {activation.dtype}"
    if weight.dtype != torch.float8_e4m3fn:
        return False, f"expected fp8 weight, got {weight.dtype}"
    tensors = (activation, activation_scale, weight, weight_scale, out)
    if not all(_is_musa_tensor(tensor) for tensor in tensors):
        return False, "all tensors must be on MUSA"

    tokens, groups, in_dim = activation.shape
    if activation_scale.shape != (tokens, groups, in_dim // _GROUP_SIZE):
        return False, (
            "activation_scale shape mismatch: "
            f"{tuple(activation_scale.shape)} for activation={tuple(activation.shape)}"
        )

    try:
        weight, scales, out_dim, normalized_in_dim = _normalize_weight(
            weight,
            weight_scale,
            groups,
        )
    except Exception as exc:
        if mode in {"gemv", "native", "force"}:
            raise
        return False, f"{type(exc).__name__}: {exc}"

    if normalized_in_dim != in_dim or out.shape != (tokens, groups, out_dim):
        return False, (
            "shape mismatch after weight normalization: "
            f"activation={tuple(activation.shape)}, weight={tuple(weight.shape)}, "
            f"out={tuple(out.shape)}"
        )

    try:
        from vllm_musa import _custom_ops as musa_ops

        for group_idx in range(groups):
            group_out = musa_ops.musa_fused_gemv(
                activation[:, group_idx, :].contiguous(),
                weight[group_idx].contiguous(),
                activation_scale[:, group_idx, :].contiguous(),
                scales[group_idx].contiguous(),
            )
            out[:, group_idx, :].copy_(group_out)
    except Exception as exc:
        if mode in {"gemv", "native", "force"}:
            raise
        return False, f"{type(exc).__name__}: {exc}"

    return True, "musa_fused_gemv"


def _torch_fp8_einsum_fallback_enabled() -> bool:
    return os.getenv("VLLM_MUSA_ENABLE_TORCH_FP8_EINSUM_FALLBACK", "0") == "1"


def _validate_group_size_divisible(name: str, size: int) -> None:
    if size % _GROUP_SIZE != 0:
        raise ValueError(
            f"{name} dimension must be divisible by {_GROUP_SIZE}, got {size}"
        )


def _dequant_activation(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
) -> torch.Tensor:
    tokens, groups, hidden = activation.shape
    _validate_group_size_divisible("activation hidden", hidden)
    scale_blocks = hidden // _GROUP_SIZE
    activation_blocks = activation.to(torch.float32).reshape(
        tokens,
        groups,
        scale_blocks,
        _GROUP_SIZE,
    )
    return (
        activation_blocks * activation_scale.to(torch.float32).unsqueeze(-1)
    ).reshape(tokens, groups, hidden)


def _dequant_weight(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    groups: int,
) -> torch.Tensor:
    weight, scales, out_dim, in_dim = _normalize_weight(
        weight,
        weight_scale,
        groups,
    )
    _validate_group_size_divisible("weight output", out_dim)
    _validate_group_size_divisible("weight input", in_dim)
    out_blocks = out_dim // _GROUP_SIZE
    in_blocks = in_dim // _GROUP_SIZE
    weight_blocks = weight.to(torch.float32).reshape(
        groups,
        out_blocks,
        _GROUP_SIZE,
        in_blocks,
        _GROUP_SIZE,
    )
    return (
        weight_blocks * scales[:, :, None, :, None]
    ).reshape(groups, out_dim, in_dim)


def try_musa_deepseek_v4_fp8_einsum(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
) -> tuple[bool, str]:
    """Try supported MUSA replacements for DeepSeek-V4 FP8 einsum."""
    handled, reason = try_musa_deepseek_v4_fp8_einsum_gemv(
        activation,
        activation_scale,
        weight,
        weight_scale,
        out,
        equation,
    )
    if handled:
        return True, reason
    if not _torch_fp8_einsum_fallback_enabled():
        return False, reason
    if equation != "bhr,hdr->bhd":
        return False, f"unsupported equation {equation!r}"

    activation_deq = _dequant_activation(activation, activation_scale)
    weight_deq = _dequant_weight(weight, weight_scale, activation.shape[1])
    out.copy_(torch.einsum(equation, activation_deq, weight_deq).to(out.dtype))
    return True, "torch_dequant_einsum_fallback"
