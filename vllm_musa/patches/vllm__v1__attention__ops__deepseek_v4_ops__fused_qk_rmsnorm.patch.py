# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch DeepSeek-V4 fused Q/K RMSNorm with a native MUSA implementation.
"""

_HELPERS = """import os

from vllm.platforms import current_platform
# MUSA fallback helpers below use current_platform.
from vllm.triton_utils import tl, triton


def _is_musa_tensor(tensor: torch.Tensor) -> bool:
    return (
        current_platform.is_musa()
        or getattr(torch.version, "musa", None) is not None
        or tensor.device.type == "musa"
    )


def _musa_rmsnorm_fallback(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    x_float = x.to(torch.float32)
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    out = x_float * torch.rsqrt(variance + eps)
    out = out * weight.to(torch.float32)
    return out.to(x.dtype)


def _musa_qkv_rmsnorm_fallback_enabled() -> bool:
    return os.getenv("VLLM_MUSA_DEEPSEEK_V4_QKV_RMSNORM_FALLBACK", "0") == "1"


def _musa_fused_q_kv_rmsnorm(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm_musa import _custom_ops as _musa_custom_ops

    return _musa_custom_ops.deepseek_v4_fused_q_kv_rmsnorm(
        qr,
        kv,
        q_weight,
        kv_weight,
        eps,
    )
"""

_FALLBACK_RETURN = """    if _is_musa_tensor(qr) or _is_musa_tensor(kv):
        if not _musa_qkv_rmsnorm_fallback_enabled():
            return _musa_fused_q_kv_rmsnorm(qr, kv, q_weight, kv_weight, eps)
        return (
            _musa_rmsnorm_fallback(qr, q_weight, eps),
            _musa_rmsnorm_fallback(kv, kv_weight, eps),
        )
"""

_OLD_FALLBACK_RETURN = """    if _is_musa_tensor(qr) or _is_musa_tensor(kv):
        return (
            _musa_rmsnorm_fallback(qr, q_weight, eps),
            _musa_rmsnorm_fallback(kv, kv_weight, eps),
        )
"""

_OLD_HELPERS = """from vllm.platforms import current_platform
# MUSA fallback helpers below use current_platform.
from vllm.triton_utils import tl, triton


def _is_musa_tensor(tensor: torch.Tensor) -> bool:
    return (
        current_platform.is_musa()
        or getattr(torch.version, "musa", None) is not None
        or tensor.device.type == "musa"
    )


def _musa_rmsnorm_fallback(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    x_float = x.to(torch.float32)
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    out = x_float * torch.rsqrt(variance + eps)
    out = out * weight.to(torch.float32)
    return out.to(x.dtype)
"""

PATCHES = [
    (
        _OLD_HELPERS,
        _HELPERS,
    ),
    (
        _OLD_FALLBACK_RETURN,
        _FALLBACK_RETURN,
    ),
    (
        """import torch

from vllm.triton_utils import tl, triton
""",
        f"""import torch

{_HELPERS}""",
    ),
    (
        """from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
""",
        _HELPERS,
    ),
    (
        """    if current_platform.is_musa() or qr.device.type == "musa":
        raise NotImplementedError(
            "DeepSeek-V4 fused_q_kv_rmsnorm is not implemented for MUSA yet. "
            "A MUSA-safe Q/K RMSNorm implementation is required before model "
            "execution can proceed."
        )
""",
        _FALLBACK_RETURN,
    ),
    (
        """    if (
        current_platform.is_musa()
        or getattr(torch.version, "musa", None) is not None
        or qr.device.type == "musa"
    ):
        raise NotImplementedError(
            "DeepSeek-V4 fused_q_kv_rmsnorm is not implemented for MUSA yet. "
            "A MUSA-safe Q/K RMSNorm implementation is required before model "
            "execution can proceed."
        )
""",
        _FALLBACK_RETURN,
    ),
    (
        """    assert qr.ndim == 2 and kv.ndim == 2
    assert qr.shape[0] == kv.shape[0], (
""",
        f"""{_FALLBACK_RETURN}    assert qr.ndim == 2 and kv.ndim == 2, "qr and kv must be 2D"
    assert qr.shape[0] == kv.shape[0], (
""",
    ),
]

RELOAD_AFTER_PATCH = [
    "__TARGET_MODULE__",
    "vllm.v1.attention.ops.deepseek_v4_ops",
]
