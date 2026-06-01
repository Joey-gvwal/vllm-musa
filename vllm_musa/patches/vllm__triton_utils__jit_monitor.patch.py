# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA Triton compatibility patch for v0.22 JIT monitor."""

PATCHES = [
    (
        "    from triton import knobs  # type: ignore[import-untyped]\n",
        """    try:
        from triton import knobs  # type: ignore[import-untyped]
    except ImportError:
        logger.debug("Triton knobs API is unavailable; skipping JIT monitor setup.")
        return
""",
    ),
]
