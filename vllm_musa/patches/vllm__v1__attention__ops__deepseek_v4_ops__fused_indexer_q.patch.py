# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch DeepSeek-V4 sparse-indexer Q quantization with a MUSA-specific gate.
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
        """    assert positions.ndim == 1
    assert index_q.ndim == 3
""",
        """    if (
        current_platform.is_musa()
        or getattr(torch.version, "musa", None) is not None
        or index_q.device.type == "musa"
    ):
        raise NotImplementedError(
            "DeepSeek-V4 fused_indexer_q_rope_quant is not implemented for "
            "MUSA yet. A MUSA-safe sparse indexer Q RoPE/FP8-or-MXFP4 "
            "quantization path is required before model execution can proceed."
        )
    assert positions.ndim == 1
    assert index_q.ndim == 3
""",
    ),
]

RELOAD_AFTER_PATCH = [
    "__TARGET_MODULE__",
    "vllm.v1.attention.ops.deepseek_v4_ops",
]
