# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DeepSeek-V4 MUSA compressor fallback patch helpers."""

import importlib.util
from pathlib import Path

import torch


def _load_musa_deepseek_v4_compressor_gemm():
    patch_path = (
        Path(__file__).parents[1]
        / "vllm_musa"
        / "patches"
        / "vllm__model_executor__layers__deepseek_compressor.patch.py"
    )
    spec = importlib.util.spec_from_file_location(
        "deepseek_v4_compressor_patch", patch_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for _, replacement in module.PATCHES:
        if "def _musa_deepseek_v4_compressor_gemm" not in replacement:
            continue
        start = replacement.index("def _musa_deepseek_v4_compressor_gemm")
        end = replacement.index("\n\ndef _musa_deepseek_v4_store_sparse_kv", start)
        namespace = {"torch": torch}
        exec(replacement[start:end], namespace)
        return namespace["_musa_deepseek_v4_compressor_gemm"]
    raise AssertionError("compressor GEMM fallback source was not found")


def test_musa_deepseek_v4_compressor_gemm_rounds_operands_to_bf16():
    gemm = _load_musa_deepseek_v4_compressor_gemm()
    x = torch.tensor(
        [[1.001, -2.003, 3.007], [4.011, -5.019, 6.023]],
        dtype=torch.float32,
    )
    weight = torch.tensor(
        [
            [0.101, -0.203, 0.307],
            [-0.409, 0.503, -0.607],
            [0.709, -0.809, 0.907],
            [-1.001, 1.103, -1.207],
        ],
        dtype=torch.float32,
    )

    actual = gemm(x, weight)
    expected = torch.nn.functional.linear(
        x.to(torch.bfloat16).to(torch.float32),
        weight.to(torch.bfloat16).to(torch.float32),
    )
    full_fp32 = torch.nn.functional.linear(x, weight)

    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, full_fp32)
