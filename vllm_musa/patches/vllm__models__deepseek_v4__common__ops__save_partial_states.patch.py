# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Patch vLLM v0.22 DeepSeek-V4 partial-state save for MUSA Triton."""

PATCHES = [
    (
        """import torch

from vllm.triton_utils import tl, triton
""",
        """import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


def _musa_deepseek_v4_save_partial_pdl_kwargs(
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
        """        COMPRESS_RATIO=compress_ratio,
        **(pdl_kwargs or {}),
    )
""",
        """        COMPRESS_RATIO=compress_ratio,
        **_musa_deepseek_v4_save_partial_pdl_kwargs(kv, pdl_kwargs),
    )
""",
    ),
]
