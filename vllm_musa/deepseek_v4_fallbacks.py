# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 MUSA correctness fallback defaults."""

import logging
import os

DEEPSEEK_V4_CORRECTNESS_FALLBACK_DEFAULTS = {
    "VLLM_MUSA_ENABLE_TORCH_MHC_PRENORM_FALLBACK": "1",
    "VLLM_MUSA_ENABLE_TORCH_FP8_EINSUM_FALLBACK": "1",
    "VLLM_MUSA_ENABLE_TORCH_DEEPSEEK_V4_COMPRESSOR_FALLBACK": "1",
    "VLLM_MUSA_DEEPSEEK_V4_COMPRESSOR_TRITON": "1",
    "VLLM_MUSA_ENABLE_TORCH_SPARSE_ATTN_INDEXER_FALLBACK": "1",
    "VLLM_MUSA_SPARSE_INDEXER_FALLBACK_TOPK": "16",
}


def enable_deepseek_v4_sparse_correctness_fallbacks(
    logger: logging.Logger | None = None,
) -> None:
    changed = []
    for name, value in DEEPSEEK_V4_CORRECTNESS_FALLBACK_DEFAULTS.items():
        if name not in os.environ:
            os.environ[name] = value
            changed.append(name)

    if changed and logger is not None:
        logger.warning(
            "Enabling DeepSeek-V4 MUSA correctness fallback defaults: %s. "
            "These torch fallback paths let DeepSeek-V4 sparse FlashMLA run "
            "correctly on MUSA but are not fused production kernels. Set an "
            "individual variable to 0 before startup to opt out.",
            ", ".join(sorted(changed)),
        )
