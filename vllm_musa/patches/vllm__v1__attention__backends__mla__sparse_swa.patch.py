# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch DeepSeek-V4 sparse SWA metadata kernel for MUSA Triton.
"""

PATCHES = [
    (
        """    is_valid = tl.load(is_valid_token_ptr + token_idx)
    if not is_valid:
        tl.store(swa_lens_ptr + token_idx, 0)
        return
""",
        """    is_valid = tl.load(is_valid_token_ptr + token_idx)
    if is_valid == 0:
        tl.store(swa_lens_ptr + token_idx, 0)
        return
""",
    ),
]

RELOAD_AFTER_PATCH = True
