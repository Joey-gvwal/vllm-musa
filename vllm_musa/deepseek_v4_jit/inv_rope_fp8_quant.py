# SPDX-License-Identifier: Apache-2.0
"""TileLang-backed DeepSeek-V4 inverse-RoPE FP8 quant helper."""

from __future__ import annotations

import os

import torch

_HIDDEN_SIZE = 512
_NOPE_DIM = 448
_ROPE_DIM = 64
_QUANT_GROUP_SIZE = 128
_AUTO_DISABLED_REASON: str | None = None


def _is_musa_tensor(tensor: torch.Tensor) -> bool:
    return (
        getattr(tensor, "device", None) is not None
        and tensor.device.type == "musa"
    )


def _disabled_mode() -> bool:
    mode = os.environ.get(
        "VLLM_MUSA_DEEPSEEK_V4_INV_ROPE_FP8_QUANT_IMPL",
        "auto",
    ).strip().lower()
    return mode in {"torch", "fallback", "0", "off"}


def _force_mode() -> bool:
    mode = os.environ.get(
        "VLLM_MUSA_DEEPSEEK_V4_INV_ROPE_FP8_QUANT_IMPL",
        "auto",
    ).strip().lower()
    return mode in {"tilelang", "jit", "force"}


def _guard_tilelang_inv_rope_fp8_quant(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    quant_group_size: int,
    tma_aligned_scales: bool,
) -> tuple[bool, str]:
    if not all(_is_musa_tensor(t) for t in (o, positions, cos_sin_cache)):
        return False, "all tensors must be on MUSA"
    if o.dtype != torch.bfloat16:
        return False, f"expected bf16 attention output, got {o.dtype}"
    if positions.dtype != torch.int64:
        return False, f"expected int64 positions, got {positions.dtype}"
    if cos_sin_cache.dtype != torch.float32:
        return False, f"expected float32 cos_sin_cache, got {cos_sin_cache.dtype}"
    if tma_aligned_scales:
        return False, "TileLang path currently supports FP32 scale layout only"
    if nope_dim != _NOPE_DIM or rope_dim != _ROPE_DIM:
        return False, f"expected nope/rope dims 448/64, got {nope_dim}/{rope_dim}"
    if quant_group_size != _QUANT_GROUP_SIZE:
        return False, f"expected quant_group_size=128, got {quant_group_size}"
    if o.dim() != 3 or o.shape[-1] != _HIDDEN_SIZE:
        return False, f"expected o shape [tokens, heads, 512], got {tuple(o.shape)}"
    if o.shape[1] != n_groups * heads_per_group:
        return False, "n_groups * heads_per_group must match local head count"
    if positions.dim() != 1 or positions.shape[0] != o.shape[0]:
        return False, "positions must be 1D and match token count"
    if cos_sin_cache.dim() != 2 or cos_sin_cache.shape[-1] != _ROPE_DIM:
        return (
            False,
            "expected cos_sin_cache shape [positions, 64], got "
            f"{tuple(cos_sin_cache.shape)}",
        )
    if heads_per_group <= 0:
        return False, f"heads_per_group must be positive, got {heads_per_group}"
    return True, ""


def try_tilelang_inv_rope_fp8_quant(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    quant_group_size: int,
    tma_aligned_scales: bool,
) -> tuple[bool, tuple[torch.Tensor, torch.Tensor] | None, str]:
    """Try the TileLang fused inverse-RoPE + FP8 quant path."""
    global _AUTO_DISABLED_REASON

    if _disabled_mode():
        return (
            False,
            None,
            "disabled by VLLM_MUSA_DEEPSEEK_V4_INV_ROPE_FP8_QUANT_IMPL",
        )
    if _AUTO_DISABLED_REASON is not None and not _force_mode():
        return False, None, _AUTO_DISABLED_REASON

    supported, reason = _guard_tilelang_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups,
        heads_per_group,
        nope_dim,
        rope_dim,
        quant_group_size,
        tma_aligned_scales,
    )
    if not supported:
        if _force_mode():
            raise NotImplementedError(reason)
        return False, None, reason

    try:
        from vllm_musa.deepseek_v4_jit.tilelang_kernels import (
            inv_rope_fp8_quant_kernel,
        )

        src = o if o.is_contiguous() else o.contiguous()
        d = heads_per_group * _HIDDEN_SIZE
        num_scale_blocks = d // _QUANT_GROUP_SIZE
        fp8_storage = torch.empty(
            (n_groups, o.shape[0], d),
            dtype=torch.float8_e4m3fn,
            device=o.device,
        )
        scale_storage = torch.empty(
            (n_groups, o.shape[0], num_scale_blocks),
            dtype=torch.float32,
            device=o.device,
        )
        inv_rope_fp8_quant_kernel(int(heads_per_group))(
            src,
            positions,
            cos_sin_cache,
            fp8_storage.view(torch.uint8),
            scale_storage,
        )
    except Exception as exc:
        if _force_mode():
            raise
        _AUTO_DISABLED_REASON = f"{type(exc).__name__}: {exc}"
        return False, None, _AUTO_DISABLED_REASON

    return (
        True,
        (fp8_storage.transpose(0, 1), scale_storage.transpose(0, 1)),
        "tilelang",
    )
