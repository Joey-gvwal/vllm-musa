# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch DeepSeek-V4 fused compressor Triton kernels for MUSA Triton syntax.
"""

PATCHES = [
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

RELOAD_AFTER_PATCH = True
