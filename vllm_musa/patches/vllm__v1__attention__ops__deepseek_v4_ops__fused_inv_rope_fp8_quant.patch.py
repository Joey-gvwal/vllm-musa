# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch DeepSeek-V4 inverse-RoPE FP8 quantization with a MUSA-specific gate.
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
        """    from vllm.utils.deep_gemm import get_tma_aligned_size

    num_tokens, num_heads, head_dim = o.shape
""",
        """    if current_platform.is_musa() or o.device.type == "musa":
        raise NotImplementedError(
            "DeepSeek-V4 fused_inv_rope_fp8_quant is not implemented for "
            "MUSA yet. A MUSA-safe inverse-RoPE, FP8 quantization, and "
            "scale-layout path is required before model execution can proceed."
        )
    from vllm.utils.deep_gemm import get_tma_aligned_size

    num_tokens, num_heads, head_dim = o.shape
""",
    ),
]

RELOAD_AFTER_PATCH = [
    "__TARGET_MODULE__",
    "vllm.v1.attention.ops.deepseek_v4_ops",
]
