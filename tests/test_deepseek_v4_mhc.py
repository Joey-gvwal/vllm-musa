# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source-level checks for DeepSeek-V4 MHC MUSA dispatch."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text()


def test_mhc_pre_defaults_to_native_provider():
    source = _read("vllm_musa/deepseek_v4_mhc.py")

    assert 'VLLM_MUSA_DEEPSEEK_V4_MHC_PRE_IMPL", "native"' in source
    assert "def _mhc_pre_native_provider(" in source
    assert "deepseek_v4_mhc_pre(" in source
    assert "mhc_pre_torch_fallback(" in source
    assert source.index("def _mhc_pre_native_provider(") < source.index(
        "def _mhc_pre_tilelang_provider("
    )


def test_mhc_pre_custom_op_is_registered():
    source = _read("vllm_musa/_custom_ops.py")
    bindings = _read("csrc/musa/torch_bindings.cpp")
    headers = _read("csrc/musa/musa_ops.h")
    setup = _read("setup.py")

    assert "def deepseek_v4_mhc_pre(" in source
    assert "torch.ops._C_musa_ops.deepseek_v4_mhc_pre" in source
    assert "deepseek_v4_mhc_pre(Tensor residual" in bindings
    assert "&deepseek_v4_mhc_pre" in bindings
    assert "void deepseek_v4_mhc_pre(" in headers
    assert "csrc/musa/mhc/deepseek_v4_mhc_pre.mu" in setup


def test_mhc_patch_uses_musa_provider_by_default():
    source = _read("vllm_musa/patches/vllm__model_executor__layers__mhc.patch.py")

    assert "from vllm_musa.deepseek_v4_mhc import mhc_pre_musa" in source
    assert "from vllm_musa.deepseek_v4_mhc import mhc_post_musa" in source
    assert "VLLM_MUSA_ENABLE_DEEPSEEK_V4_MHC_MUSA_IMPL" in source
    assert 'VLLM_MUSA_ENABLE_TORCH_MHC_PRENORM_FALLBACK",\n                "0"' in source
