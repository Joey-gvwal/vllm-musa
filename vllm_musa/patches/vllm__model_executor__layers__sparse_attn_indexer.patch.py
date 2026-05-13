# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch sparse-attention indexer with an opt-in MUSA diagnostic fallback.
"""

PATCHES = [
    (
        """import torch

import vllm.envs as envs
""",
        """import os

import torch

import vllm.envs as envs
""",
    ),
    (
        """        else:
            raise NotImplementedError(
                "SparseAttnIndexer native forward is only implemented for "
                "CUDA, ROCm and XPU platforms."
            )
""",
        """        elif (
            current_platform.is_musa()
            and os.getenv(
                "VLLM_MUSA_ENABLE_TORCH_SPARSE_ATTN_INDEXER_FALLBACK",
                "0",
            )
            == "1"
        ):
            logger.warning_once(
                "Using opt-in MUSA DeepSeek-V4 sparse-attention indexer "
                "fallback. It returns no learned global sparse picks and is "
                "diagnostic, not a production indexer backend."
            )
            self.topk_indices_buffer[: hidden_states.shape[0], : self.topk_tokens] = -1
            return self.topk_indices_buffer
        else:
            raise NotImplementedError(
                "SparseAttnIndexer native forward is only implemented for "
                "CUDA, ROCm and XPU platforms."
            )
""",
    ),
]

RELOAD_AFTER_PATCH = True
