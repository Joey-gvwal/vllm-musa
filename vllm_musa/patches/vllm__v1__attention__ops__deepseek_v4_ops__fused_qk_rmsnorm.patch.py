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
""",
    ),
    (
        """    assert qr.ndim == 2 and kv.ndim == 2
    assert qr.shape[0] == kv.shape[0], (
""",
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
    assert qr.ndim == 2 and kv.ndim == 2
    assert qr.shape[0] == kv.shape[0], (
""",
    ),
]

RELOAD_AFTER_PATCH = [
    "__TARGET_MODULE__",
    "vllm.v1.attention.ops.deepseek_v4_ops",
]
