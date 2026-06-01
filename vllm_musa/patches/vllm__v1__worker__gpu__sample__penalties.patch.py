# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA Triton compatibility patch for v0.22 worker penalties."""

PATCHES = [
    (
        "    use_penalty = use_rep_penalty or use_freq_penalty or use_pres_penalty\n",
        "    use_penalty = (use_rep_penalty or use_freq_penalty) or use_pres_penalty\n",
    ),
]
