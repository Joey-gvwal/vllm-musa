# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 FP8 einsum helpers for MUSA provider experiments."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import torch

_GROUP_SIZE = 128
_DEEPGEMM_MIN_TOKENS = 128


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


def _normalize_bf16_weight(
    weight: torch.Tensor,
    groups: int,
) -> torch.Tensor:
    if weight.dim() == 2:
        flat_out_dim, in_dim = weight.shape
        if flat_out_dim % groups != 0:
            raise ValueError(
                "DeepSeek-V4 BF16 O-projection expected 2D weight rows to be "
                f"divisible by groups={groups}, got {tuple(weight.shape)}"
            )
        return weight.reshape(groups, flat_out_dim // groups, in_dim)
    if weight.dim() == 3 and weight.shape[0] == groups:
        return weight
    raise ValueError(
        "DeepSeek-V4 BF16 O-projection expects a 2D or group-aligned 3D "
        f"weight, got {tuple(weight.shape)} for groups={groups}"
    )


def _dequant_checkpoint_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    if weight.dim() != 2 or scale.dim() != 2:
        raise ValueError(
            "DeepSeek-V4 wo_a FP8 checkpoint expects 2D weight and scale, "
            f"got {tuple(weight.shape)} and {tuple(scale.shape)}"
        )
    out_blocks, in_blocks = scale.shape
    if weight.shape != (out_blocks * _GROUP_SIZE, in_blocks * _GROUP_SIZE):
        raise ValueError(
            "DeepSeek-V4 wo_a FP8 checkpoint shape mismatch: "
            f"weight={tuple(weight.shape)}, scale={tuple(scale.shape)}"
        )
    blocks = weight.to(torch.float32).reshape(
        out_blocks,
        _GROUP_SIZE,
        in_blocks,
        _GROUP_SIZE,
    )
    return (blocks * scale.to(torch.float32)[:, None, :, None]).reshape(
        weight.shape
    ).to(torch.bfloat16)


def prepare_musa_deepseek_v4_wo_a_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Normalize DeepSeek-V4 ``wo_a`` checkpoints for the MUSA BF16 path.

    The upstream checkpoint stores FP8 ``wo_a.weight`` plus ``wo_a.scale``.
    SGLang's repacked FP8 checkpoint stores BF16 ``wo_a.weight`` and omits the
    scale tensor, while retaining stale scale entries in its index. Buffer only
    these small projection tensors, dequantize the upstream form, and pass the
    SGLang form through unchanged.
    """
    pending: dict[str, dict[str, tuple[str, torch.Tensor]]] = {}
    order: list[str] = []
    weight_suffix = ".attn.wo_a.weight"
    scale_suffix = ".attn.wo_a.scale"

    for name, tensor in weights:
        if name.endswith(weight_suffix):
            prefix = name[: -len(".weight")]
            if prefix not in pending:
                pending[prefix] = {}
                order.append(prefix)
            pending[prefix]["weight"] = (name, tensor)
        elif name.endswith(scale_suffix):
            prefix = name[: -len(".scale")]
            if prefix not in pending:
                pending[prefix] = {}
                order.append(prefix)
            pending[prefix]["scale"] = (name, tensor)
        else:
            yield name, tensor

    fp8_dtype = torch.float8_e4m3fn
    for prefix in order:
        values = pending[prefix]
        if "weight" not in values:
            raise ValueError(f"DeepSeek-V4 wo_a scale has no weight: {prefix}")
        weight_name, weight = values["weight"]
        scale_entry = values.get("scale")
        if weight.dtype == fp8_dtype:
            if scale_entry is None:
                raise ValueError(
                    f"DeepSeek-V4 FP8 wo_a weight has no scale: {weight_name}"
                )
            yield weight_name, _dequant_checkpoint_weight(weight, scale_entry[1])
        elif weight.dtype in (torch.bfloat16, torch.float16, torch.float32):
            yield weight_name, weight.to(torch.bfloat16)
        else:
            raise TypeError(
                f"Unsupported DeepSeek-V4 wo_a dtype for {weight_name}: "
                f"{weight.dtype}"
            )


def try_musa_deepseek_v4_fp8_einsum_gemv(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
) -> tuple[bool, str]:
    """Dispatch ``bhr,hdr->bhd`` to DeepGEMM or GEMV based on token count."""
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
        return False, f"{type(exc).__name__}: {exc}"

    if normalized_in_dim != in_dim or out.shape != (tokens, groups, out_dim):
        return False, (
            "shape mismatch after weight normalization: "
            f"activation={tuple(activation.shape)}, weight={tuple(weight.shape)}, "
            f"out={tuple(out.shape)}"
        )

    deepgemm_error: str | None = None
    if tokens >= _DEEPGEMM_MIN_TOKENS and out.dtype == torch.bfloat16:
        try:
            from vllm.utils.deep_gemm import fp8_gemm_nt

            for group_idx in range(groups):
                group_out_view = out[:, group_idx, :]
                copy_group_out = not group_out_view.is_contiguous()
                group_out = (
                    torch.empty_like(
                        group_out_view,
                        memory_format=torch.contiguous_format,
                    )
                    if copy_group_out
                    else group_out_view
                )
                fp8_gemm_nt(
                    (
                        activation[:, group_idx, :].contiguous(),
                        activation_scale[:, group_idx, :].contiguous(),
                    ),
                    (
                        weight[group_idx].contiguous(),
                        scales[group_idx].contiguous(),
                    ),
                    group_out,
                    is_deep_gemm_e8m0_used=False,
                )
                if copy_group_out:
                    group_out_view.copy_(group_out)
        except Exception as exc:
            deepgemm_error = f"{type(exc).__name__}: {exc}"
        else:
            return True, "musa_deepgemm_fp8_o_proj"

    try:
        from vllm_musa import _custom_ops as musa_ops

        for group_idx in range(groups):
            group_out_view = out[:, group_idx, :]
            direct_group_out = (
                group_out_view if group_out_view.is_contiguous() else None
            )
            group_out = musa_ops.musa_fused_gemv(
                activation[:, group_idx, :].contiguous(),
                weight[group_idx].contiguous(),
                activation_scale[:, group_idx, :].contiguous(),
                scales[group_idx].contiguous(),
                output=direct_group_out,
            )
            if direct_group_out is None:
                group_out_view.copy_(group_out)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"

    if deepgemm_error is not None:
        return True, f"musa_fused_gemv (DeepGEMM fallback: {deepgemm_error})"
    return True, "musa_fused_gemv"


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
    return (weight_blocks * scales[:, :, None, :, None]).reshape(
        groups, out_dim, in_dim
    )


def try_musa_deepseek_v4_fp8_einsum(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor | None,
    out: torch.Tensor,
    equation: str,
) -> tuple[bool, str]:
    """Try supported MUSA replacements for DeepSeek-V4 FP8 einsum."""
    if weight.dtype in (torch.bfloat16, torch.float16, torch.float32):
        if equation != "bhr,hdr->bhd":
            return False, f"unsupported equation {equation!r}"
        activation_deq = _dequant_activation(activation, activation_scale).to(
            torch.bfloat16
        )
        weight_bf16 = _normalize_bf16_weight(weight, activation.shape[1])
        out.copy_(
            torch.einsum(equation, activation_deq, weight_bf16).to(out.dtype)
        )
        return True, "torch_bf16_wo_a_einsum"

    if weight_scale is None:
        return False, "FP8 weight requires a weight scale"

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
    if equation != "bhr,hdr->bhd":
        return False, f"unsupported equation {equation!r}"

    activation_deq = _dequant_activation(activation, activation_scale)
    weight_deq = _dequant_weight(weight, weight_scale, activation.shape[1])
    out.copy_(torch.einsum(equation, activation_deq, weight_deq).to(out.dtype))
    return True, "torch_dequant_einsum_fallback"
