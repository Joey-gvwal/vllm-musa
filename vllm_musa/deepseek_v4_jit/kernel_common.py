# SPDX-License-Identifier: Apache-2.0
"""Small TileLang compatibility helpers used by DeepSeek-V4 MUSA JIT paths."""

from __future__ import annotations

import os


_TILELANG_MUSA_OPT_FLAGS = [
    "-fmusa-flush-denormals-to-zero",
    "-fno-signed-zeros",
    "-mllvm",
    "-mtgpu-opt-level=1",
]


def _tilelang_musa_pass_configs(tilelang):
    """Return optional MUSA pass configs without requiring a specific TileLang build."""
    if os.environ.get("VLLM_MUSA_DEEPSEEK_V4_TILELANG_PASS_CONFIG") != "1":
        return None

    pass_configs = {}
    for key_name, value in (
        ("TL_ENABLE_MUSA_BURST", True),
        ("TL_ENABLE_REDUCE_BURST", True),
        ("TL_DEVICE_COMPILE_FLAGS", _TILELANG_MUSA_OPT_FLAGS),
    ):
        key = getattr(tilelang.PassConfigKey, key_name, None)
        if key is not None:
            pass_configs[key] = value
    return pass_configs or None


def _tilelang_jit(tilelang, name: str, pass_configs=None):
    if pass_configs is None:
        pass_configs = _tilelang_musa_pass_configs(tilelang)
    try:
        if pass_configs is None:
            return tilelang.jit()
        return tilelang.jit(pass_configs=pass_configs)
    except TypeError as exc:
        if "pass_configs" not in str(exc):
            raise
        return tilelang.jit()
