# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch DeepSeek-V4 fused Q/K RMSNorm with a MUSA-specific gate.
"""

PATCHES = [
    (
        """import torch

from vllm.triton_utils import tl, triton
""",
        """import torch

from vllm.platforms import current_platform
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
""",
    ),
    (
        """    assert qr.ndim == 2 and kv.ndim == 2
    assert qr.shape[0] == kv.shape[0], (
""",
        """    if _is_musa_tensor(qr) or _is_musa_tensor(kv):
        return (
            _musa_rmsnorm_fallback(qr, q_weight, eps),
            _musa_rmsnorm_fallback(kv, kv_weight, eps),
        )
    assert qr.ndim == 2 and kv.ndim == 2
    assert qr.shape[0] == kv.shape[0], (
""",
    ),
]

RELOAD_AFTER_PATCH = [
    "__TARGET_MODULE__",
    "vllm.v1.attention.ops.deepseek_v4_ops",
]
