# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch DeepSeek-V4 cache utility kernels with MUSA-specific gates.
"""

PATCHES = [
    (
        """import torch

from vllm.triton_utils import tl, triton
""",
        """import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


def _raise_musa_deepseek_v4_cache_unsupported(op_name: str) -> None:
    raise NotImplementedError(
        f"DeepSeek-V4 {op_name} is not implemented for MUSA yet. "
        "A MUSA-safe cache quantization, dequantization, top-k metadata, "
        "or sparse prefill index implementation is required before model "
        "execution can proceed."
    )
""",
    ),
    (
        """    assert k.dim() == 2 and k.shape[1] == 512, (
        f"K must be [num_tokens, 512], got {k.shape}"
    )
""",
        """    if current_platform.is_musa() or k.device.type == "musa":
        _raise_musa_deepseek_v4_cache_unsupported("quantize_and_insert_k_cache")
    assert k.dim() == 2 and k.shape[1] == 512, (
        f"K must be [num_tokens, 512], got {k.shape}"
    )
""",
    ),
    (
        """) -> None:
    TOKEN_FP8_DIM = 448
    TOKEN_BF16_DIM = 64
""",
        """) -> None:
    if current_platform.is_musa() or out.device.type == "musa":
        _raise_musa_deepseek_v4_cache_unsupported("dequantize_and_gather_k_cache")
    TOKEN_FP8_DIM = 448
    TOKEN_BF16_DIM = 64
""",
    ),
    (
        """    num_tokens = topk_indices.shape[0]
    global_topk_indices = torch.empty_like(topk_indices)
""",
        """    if current_platform.is_musa() or topk_indices.device.type == "musa":
        _raise_musa_deepseek_v4_cache_unsupported(
            "compute_global_topk_indices_and_lens"
        )
    num_tokens = topk_indices.shape[0]
    global_topk_indices = torch.empty_like(topk_indices)
""",
    ),
    (
        """    num_tokens = topk_indices.shape[0]
    num_reqs = seq_lens.shape[0]
""",
        """    if current_platform.is_musa() or topk_indices.device.type == "musa":
        _raise_musa_deepseek_v4_cache_unsupported("combine_topk_swa_indices")
    num_tokens = topk_indices.shape[0]
    num_reqs = seq_lens.shape[0]
""",
    ),
]

RELOAD_AFTER_PATCH = [
    "__TARGET_MODULE__",
    "vllm.v1.attention.ops.deepseek_v4_ops",
]
