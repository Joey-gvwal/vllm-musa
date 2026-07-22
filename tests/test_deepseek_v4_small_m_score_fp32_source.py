# SPDX-License-Identifier: Apache-2.0
"""Source contracts for DeepSeek-V4 small-M FP32 score GEMM dispatch."""

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH = (
    REPO_ROOT
    / "vllm_musa/patches/series/0087-MUSA-DeepSeek-V4-small-M-score-FP32-DeepGEMM.patch"
)
APPLIED_SOURCE = (
    REPO_ROOT / "third_party/vllm/vllm/models/deepseek_v4/attention.py"
)


def test_patch_uses_fixed_small_m_policy_without_ab_env_controls():
    source = PATCH.read_text()
    applied_source = APPLIED_SOURCE.read_text()

    assert "VLLM_MUSA_DEEPSEEK_V4_SCORE_FP32_GEMM_IMPL" not in source
    assert "VLLM_MUSA_DEEPSEEK_V4_SCORE_FP32_DEEPGEMM_MAX_TOKENS" not in source
    assert "VLLM_USE_DEEP_GEMM" not in source
    assert "REUSE_SCORE_FP32_CAST" not in source
    assert "_MUSA_DEEPSEEK_V4_SCORE_FP32_DEEPGEMM_MAX_TOKENS = 16" in source
    assert '== "torch"' not in source
    assert '== "deepgemm"' not in source
    assert "deep_gemm.bf16_gemm_nt(a, weight, output)" in source
    assert "F.linear(a.to(out_dtype), weight.to(out_dtype))" in applied_source


def test_applied_helper_requires_bf16_inputs_and_fp32_output():
    applied_source = APPLIED_SOURCE.read_text()
    tree = ast.parse(applied_source)
    policy = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_musa_deepseek_v4_use_score_fp32_deepgemm"
    )
    source = ast.get_source_segment(applied_source, policy)

    assert "a.dtype == torch.bfloat16" in source
    assert "weight.dtype == torch.bfloat16" in source
    assert "out_dtype == torch.float32" in source
    assert "a.shape[0]" in source
    assert "_MUSA_DEEPSEEK_V4_SCORE_FP32_DEEPGEMM_MAX_TOKENS = 16" in applied_source
    assert "_MUSA_DEEPSEEK_V4_SCORE_FP32_GEMM_IMPL" not in source
    assert "_MUSA_DEEPSEEK_V4_USE_DEEP_GEMM" not in source


def test_applied_dispatch_has_no_score_cast_reuse_dependency():
    applied_source = APPLIED_SOURCE.read_text()
    tree = ast.parse(applied_source)
    execute = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "attn_gemm_parallel_execute"
    )
    source = ast.get_source_segment(applied_source, execute)

    assert "REUSE_SCORE_FP32_CAST" not in applied_source
    assert "hidden_states_fp32" not in source
    assert source.count("_musa_deepseek_v4_linear_out_dtype(") == 2
