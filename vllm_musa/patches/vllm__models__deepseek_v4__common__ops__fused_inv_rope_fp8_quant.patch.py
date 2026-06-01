# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Patch vLLM v0.22 DeepSeek-V4 inverse-RoPE FP8 quantization for MUSA."""

PATCHES = [
    (
        """import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op
""",
        """import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


def _musa_deepseek_v4_is_musa_tensor(tensor: torch.Tensor) -> bool:
    return (
        current_platform.is_musa()
        or getattr(torch.version, "musa", None) is not None
        or getattr(tensor.device, "type", None) == "musa"
    )
""",
    ),
    (
        """    from vllm.utils.deep_gemm import get_tma_aligned_size

    num_tokens, num_heads, head_dim = o.shape
""",
        """    if _musa_deepseek_v4_is_musa_tensor(o):
        from vllm_musa import _custom_ops as _musa_custom_ops

        return _musa_custom_ops.deepseek_v4_fused_inv_rope_fp8_quant(
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

    from vllm.utils.deep_gemm import get_tma_aligned_size

    num_tokens, num_heads, head_dim = o.shape
""",
    ),
]
