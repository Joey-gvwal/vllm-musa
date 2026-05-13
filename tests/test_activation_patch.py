# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MUSA activation source patches."""

import importlib.util
from pathlib import Path


def _load_activation_patch_module():
    patch_path = (
        Path(__file__).parents[1]
        / "vllm_musa"
        / "patches"
        / "vllm__model_executor__layers__activation.patch.py"
    )
    spec = importlib.util.spec_from_file_location("activation_patch", patch_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_activation_patch_guards_silu_and_mul_on_musa():
    module = _load_activation_patch_module()
    replacements = dict(module.PATCHES)

    upstream_silu_init = """        if current_platform.is_cuda_alike() or current_platform.is_xpu():
            self.op = torch.ops._C.silu_and_mul
        elif current_platform.is_cpu():
            self._forward_method = self.forward_native
"""

    replacement = replacements[upstream_silu_init]
    assert "if current_platform.is_musa()" in replacement
    assert "self._forward_method = self.forward_oot" in replacement
    assert replacement.index("current_platform.is_musa()") < replacement.index(
        "torch.ops._C.silu_and_mul"
    )
