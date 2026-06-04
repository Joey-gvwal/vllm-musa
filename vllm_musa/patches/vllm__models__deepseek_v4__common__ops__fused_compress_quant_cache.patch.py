# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Patch vLLM v0.22 DeepSeek-V4 fused compress-cache launch for MUSA."""

PATCHES = [
    (
        """import torch

from vllm.triton_utils import tl, triton

from .fused_indexer_q import _fp32x2_to_fp4x2
""",
        """import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from .fused_indexer_q import _fp32x2_to_fp4x2


def _musa_deepseek_v4_compress_cache_pdl_kwargs(
    tensor: torch.Tensor,
    pdl_kwargs: dict | None,
) -> dict:
    active_pdl_kwargs = dict(pdl_kwargs or {})
    if (
        current_platform.is_musa()
        or getattr(torch.version, "musa", None) is not None
        or getattr(tensor.device, "type", None) == "musa"
    ):
        active_pdl_kwargs.pop("launch_pdl", None)
    return active_pdl_kwargs
""",
    ),
    (
        """        KV_BLOCK_STRIDE=kv_cache.stride(0),
        num_warps=num_warps,
        **pdl_kwargs,
    )
""",
        """        KV_BLOCK_STRIDE=kv_cache.stride(0),
        num_warps=num_warps,
        **_musa_deepseek_v4_compress_cache_pdl_kwargs(state_cache, pdl_kwargs),
    )
""",
    ),
    (
        """    score = tl.softmax(score, dim=0)
""",
        """    # MUSA Triton does not accept the upstream tl.softmax(dim=0) kwarg.
    score_max = tl.max(score, axis=0)
    score_max = tl.where(mask, score_max, 0.0)
    score_exp = tl.exp(score - score_max)
    score_exp = tl.where(mask[None, :], score_exp, 0.0)
    score_denom = tl.sum(score_exp, axis=0)
    score_denom = tl.where(score_denom > 0.0, score_denom, 1.0)
    score = score_exp / score_denom
""",
    ),
]
