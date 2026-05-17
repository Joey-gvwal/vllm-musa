# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 FP8 einsum helpers for MUSA fallback/provider experiments."""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

_GROUP_SIZE = 128


def _mode() -> str:
    return os.getenv("VLLM_MUSA_DEEPSEEK_V4_FP8_EINSUM_IMPL", "torch").strip().lower()


def _is_musa_tensor(tensor: torch.Tensor) -> bool:
    if tensor.device.type == "musa":
        return True
    try:
        is_musa = getattr(current_platform, "is_musa", None)
        return callable(is_musa) and is_musa()
    except Exception:
        return False


def _trace_enabled() -> bool:
    return os.getenv("VLLM_MUSA_DEEPSEEK_V4_FP8_EINSUM_TRACE", "0") == "1"


def _trace_limit() -> int:
    return int(os.getenv("VLLM_MUSA_DEEPSEEK_V4_FP8_EINSUM_TRACE_LIMIT", "96"))


def _trace(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
    decision: str,
    reason: str,
) -> None:
    if not _trace_enabled():
        return
    count = getattr(_trace, "_count", 0)
    if count < _trace_limit():
        logger.warning(
            "MUSA_FP8_EINSUM_FALLBACK_TRACE call=%s mode=%s decision=%s "
            "reason=%s equation=%s a_shape=%s a_stride=%s a_dtype=%s "
            "a_scale_shape=%s a_scale_stride=%s a_scale_dtype=%s "
            "b_shape=%s b_dtype=%s b_scale_shape=%s b_scale_dtype=%s "
            "out_shape=%s out_dtype=%s",
            count + 1,
            os.getenv("VLLM_MUSA_DEEPSEEK_V4_FP8_EINSUM_IMPL", "torch"),
            decision,
            reason,
            equation,
            tuple(a.shape),
            tuple(a.stride()),
            a.dtype,
            tuple(a_scale.shape),
            tuple(a_scale.stride()),
            a_scale.dtype,
            tuple(b.shape),
            b.dtype,
            tuple(b_scale.shape),
            b_scale.dtype,
            tuple(out.shape),
            out.dtype,
        )
    _trace._count = count + 1


def _dequant_activation(
    a: torch.Tensor,
    a_scale: torch.Tensor,
) -> torch.Tensor:
    bsz, groups, hidden = a.shape
    scale_blocks = hidden // _GROUP_SIZE
    a_blocks = a.to(torch.float32).reshape(
        bsz, groups, scale_blocks, _GROUP_SIZE
    )
    return (a_blocks * a_scale.to(torch.float32).unsqueeze(-1)).reshape(
        bsz, groups, hidden
    )


def _prepare_weight(
    b: torch.Tensor,
    b_scale: torch.Tensor,
    groups: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    if b.dim() == 2:
        flat_out_dim, in_dim = b.shape
        if flat_out_dim % groups != 0:
            raise ValueError(
                "MUSA DeepSeek-V4 FP8 einsum fallback expected 2D weight "
                f"rows to be divisible by groups={groups}, got {b.shape}"
            )
        out_dim = flat_out_dim // groups
        b = b.reshape(groups, out_dim, in_dim)
    elif b.dim() == 3:
        b_groups, out_dim, in_dim = b.shape
        if b_groups != groups:
            raise ValueError(
                "MUSA DeepSeek-V4 FP8 einsum fallback group mismatch: "
                f"a/groups={groups}, b/groups={b_groups}"
            )
    else:
        raise ValueError(
            "MUSA DeepSeek-V4 FP8 einsum fallback expects a 2D or 3D "
            f"weight tensor, got shape={tuple(b.shape)}"
        )

    out_blocks = out_dim // _GROUP_SIZE
    in_blocks = in_dim // _GROUP_SIZE
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is not None and b_scale.dtype == e8m0_dtype:
        exp_bits = b_scale.view(torch.uint8).to(torch.int32)
        scales = (exp_bits << 23).view(torch.float32)
    else:
        scales = b_scale.to(torch.float32)
    if scales.dim() == 2:
        expected = groups * out_blocks * in_blocks
        if scales.numel() != expected:
            raise ValueError(
                "MUSA DeepSeek-V4 FP8 einsum fallback scale element mismatch: "
                f"scale_shape={tuple(b_scale.shape)}, expected elements={expected}"
            )
        scales = scales.reshape(groups, out_blocks, in_blocks)
    elif scales.shape == (groups, in_blocks, out_blocks):
        scales = scales.transpose(-1, -2)
    assert scales.shape == (groups, out_blocks, in_blocks)
    return b, scales.contiguous(), out_dim, in_dim


def _dequant_weight(
    b: torch.Tensor,
    b_scale: torch.Tensor,
    groups: int,
) -> torch.Tensor:
    b, scales, out_dim, in_dim = _prepare_weight(b, b_scale, groups)
    b_blocks = b.to(torch.float32).reshape(
        groups,
        out_dim // _GROUP_SIZE,
        _GROUP_SIZE,
        in_dim // _GROUP_SIZE,
        _GROUP_SIZE,
    )
    return (b_blocks * scales[:, :, None, :, None]).reshape(
        groups, out_dim, in_dim
    )


def _try_native_gemv(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
) -> tuple[bool, str]:
    mode = _mode()
    if mode not in {"gemv", "musa_gemv", "native_gemv"}:
        return False, f"disabled mode={mode}"
    if equation != "bhr,hdr->bhd":
        return False, f"unsupported equation={equation}"
    if not all(_is_musa_tensor(tensor) for tensor in (a, b, out)):
        return False, "not on MUSA tensors"
    fp8_dtype = torch.float8_e4m3fn
    if a.dtype != fp8_dtype or b.dtype != fp8_dtype:
        return False, f"expected fp8 tensors, got a={a.dtype}, b={b.dtype}"
    if a.dim() != 3 or a_scale.dim() != 3 or out.dim() != 3:
        return False, "expected 3D activation, scale, and output tensors"

    tokens, groups, hidden = a.shape
    if groups == 1:
        if tokens > 2:
            return False, f"groups=1 tokens={tokens} exceeds GEMV gate"
    elif groups == 2:
        if tokens > 1:
            return False, f"groups=2 tokens={tokens} exceeds GEMV gate"
    else:
        return False, f"unsupported groups={groups}"

    if hidden % _GROUP_SIZE != 0:
        return False, f"hidden={hidden} is not divisible by {_GROUP_SIZE}"
    if tuple(a_scale.shape) != (tokens, groups, hidden // _GROUP_SIZE):
        return False, f"a_scale shape mismatch: {tuple(a_scale.shape)}"

    try:
        import vllm_musa._custom_ops  # noqa: F401
    except Exception as exc:
        return False, f"custom ops import failed: {type(exc).__name__}: {exc}"

    gemv = getattr(getattr(torch.ops, "_C_musa_ops", None), "musa_fused_gemv", None)
    if gemv is None:
        return False, "musa_fused_gemv op is unavailable"

    b, b_scales, out_dim, in_dim = _prepare_weight(b, b_scale, groups)
    if in_dim != hidden or tuple(out.shape) != (tokens, groups, out_dim):
        return (
            False,
            "shape mismatch after weight prep: "
            f"in_dim={in_dim}, hidden={hidden}, out={tuple(out.shape)}, "
            f"expected={(tokens, groups, out_dim)}",
        )

    try:
        for group in range(groups):
            tmp = torch.empty((tokens, out_dim), device=out.device, dtype=out.dtype)
            gemv(
                a[:, group, :].contiguous(),
                b[group].contiguous(),
                tmp,
                a_scale[:, group, :].contiguous().to(torch.float32),
                b_scales[group],
                False,
                False,
                False,
                None,
                1.0e-6,
            )
            out[:, group, :].copy_(tmp)
    except Exception as exc:
        return False, f"musa_fused_gemv failed: {type(exc).__name__}: {exc}"
    return True, "native_gemv"


def musa_deepseek_v4_fp8_einsum_fallback(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
) -> None:
    handled, reason = _try_native_gemv(a, a_scale, b, b_scale, out, equation)
    if handled:
        logger.warning_once(
            "Using opt-in MUSA native GEMV DeepSeek-V4 FP8 einsum path for "
            "decode-sized rows; larger rows stay on the torch fallback."
        )
        _trace(a, a_scale, b, b_scale, out, equation, "native_gemv", reason)
        return

    if equation != "bhr,hdr->bhd":
        _trace(a, a_scale, b, b_scale, out, equation, "error", reason)
        raise NotImplementedError(
            f"MUSA DeepSeek-V4 FP8 einsum fallback does not support {equation!r}"
        )

    _trace(a, a_scale, b, b_scale, out, equation, "torch_fallback", reason)
    a_deq = _dequant_activation(a, a_scale)
    b_deq = _dequant_weight(b, b_scale, a.shape[1])
    out.copy_(torch.einsum(equation, a_deq, b_deq).to(out.dtype))
