from __future__ import annotations

import ast
import os
from functools import lru_cache
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "vllm_musa"
    / "deepseek_v4_jit"
    / "fp8_einsum.py"
)


def _load_threshold_helpers() -> dict[str, object]:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if names & {
                "_DEEPGEMM_MIN_TOKENS_ENV",
                "_DEFAULT_DEEPGEMM_MIN_TOKENS",
            }:
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in {
            "_deepgemm_min_tokens",
            "_deepgemm_enabled_for_tokens",
        }:
            selected.append(node)
    namespace: dict[str, object] = {"os": os, "lru_cache": lru_cache}
    exec(compile(ast.Module(selected, []), str(MODULE_PATH), "exec"), namespace)
    return namespace


def test_deepgemm_large_m_dispatch_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    helpers = _load_threshold_helpers()
    enabled = helpers["_deepgemm_enabled_for_tokens"]

    monkeypatch.delenv("VLLM_USE_DEEP_GEMM", raising=False)
    assert not enabled(4090, "auto")

    monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "1")
    enabled.cache_clear()
    monkeypatch.delenv(
        "VLLM_MUSA_DEEPSEEK_V4_FP8_EINSUM_DEEPGEMM_MIN_TOKENS",
        raising=False,
    )
    assert not enabled(127, "auto")
    assert enabled(128, "auto")
    assert enabled(4090, "auto")

    assert not enabled(4090, "gemv")
    assert enabled(1, "deepgemm")


def test_deepgemm_threshold_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helpers = _load_threshold_helpers()
    min_tokens = helpers["_deepgemm_min_tokens"]
    env_name = "VLLM_MUSA_DEEPSEEK_V4_FP8_EINSUM_DEEPGEMM_MIN_TOKENS"

    monkeypatch.setenv(env_name, "not-an-integer")
    with pytest.raises(ValueError, match="must be an integer"):
        min_tokens()

    monkeypatch.setenv(env_name, "0")
    with pytest.raises(ValueError, match="must be positive"):
        min_tokens()


def test_o_proj_dispatch_keeps_small_m_gemv_fallback() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    dispatcher = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "try_musa_deepseek_v4_fp8_einsum_gemv"
    )
    call_names = {
        node.func.id
        for node in ast.walk(dispatcher)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    attribute_calls = {
        node.func.attr
        for node in ast.walk(dispatcher)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert "fp8_gemm_nt" in call_names
    assert "musa_fused_gemv" in attribute_calls
    assert source.index("fp8_gemm_nt(") < source.index("musa_ops.musa_fused_gemv(")
    assert "is_deep_gemm_e8m0_used=False" in source
