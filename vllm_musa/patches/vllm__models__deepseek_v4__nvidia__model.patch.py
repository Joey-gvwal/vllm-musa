# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Patch vLLM v0.22 DeepSeek-V4 NVIDIA model gates for MUSA."""

PATCHES = [
    (
        """        if layer is not None and current_platform.is_cuda():
            hidden_states = layer.hc_post(hidden_states, residual, post_mix, res_mix)
""",
        """        if layer is not None and (
            current_platform.is_cuda() or current_platform.is_musa()
        ):
            hidden_states = layer.hc_post(hidden_states, residual, post_mix, res_mix)
""",
    ),
]
