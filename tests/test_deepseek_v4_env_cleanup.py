# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source-level guardrails for DeepSeek-V4 env-gate cleanup."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).parents[1]
SOURCE_ROOTS = (
    ROOT / "vllm_musa",
    ROOT / "csrc",
    ROOT / "setup.py",
    ROOT / "pyproject.toml",
)


def _musa_env(*parts: str) -> str:
    return "VLLM_MUSA_" + "".join(parts)

RETAINED_VLLM_MUSA_NAMES = {
    # Build/install and JIT infrastructure.
    "VLLM_MUSA_ARCH",
    "VLLM_MUSA_ARCH_LIST",
    "VLLM_MUSA_ARCH_MP31",
    "VLLM_MUSA_AVAILABLE",
    "VLLM_MUSA_CSRC_SOURCES",
    "VLLM_MUSA_EXTRA_CFLAGS",
    "VLLM_MUSA_EXTRA_LDFLAGS",
    "VLLM_MUSA_EXTRA_MUSAFLAGS",
    "VLLM_MUSA_JIT_CACHE_DIR",
    "VLLM_MUSA_JIT_VERBOSE",
    "VLLM_MUSA_MCC",
    "VLLM_MUSA_NO_BUILD_PATCH",
    "VLLM_MUSA_ROPE_STORE_HINT",
    # Model-generic runtime controls, not DeepSeek-V4 profile selectors.
    "VLLM_MUSA_CUSTOM_OP_USE_NATIVE",
    "VLLM_MUSA_DRAFT_TP1",
    "VLLM_MUSA_ENABLE_INDUCTOR_HEURISTICS",
    "VLLM_MUSA_ENABLE_JIT_TOPK",
    "VLLM_MUSA_FUSED_ADD_RMSNORM",
    "VLLM_MUSA_PTGQ128_REGISTER_FASTPATH",
    "VLLM_MUSA_RESHAPE_CACHE_FLASH",
    "VLLM_MUSA_RESHAPE_CACHE_TOKENS_PER_BLOCK",
    "VLLM_MUSA_SAMPLER_FAST_PATH",
    "VLLM_MUSA_SILU_FP8_QUANT_FUSION",
    "VLLM_MUSA_SILU_FP8_QUANT_MAX_TOKENS",
    "VLLM_MUSA_SPHERE_SILU_FP8",
    "VLLM_MUSA_SPHERE_SILU_FP8_MAX_TOKENS",
    "VLLM_MUSA_WORKER_TERMINATION_TIMEOUT_S",
}


REMOVED_DEEPSEEK_ENV_PATTERNS = (
    re.compile(r"VLLM_MUSA_[A-Z0-9_]*DEEP(?:SEEK)?[A-Z0-9_]*"),
    re.compile("DEEP" + "SEEK_V4"),
    re.compile("DFLASH" + "_FULL_WRAP"),
    re.compile("DRAFT" + "_FULL_WRAP"),
    re.compile("SPEC_DECODE_RANDOM" + "_FALLBACK"),
    re.compile("SPARSE_INDEXER_GRAPH" + "_EXACT_DECODE"),
)

REMOVED_GENERIC_PROFILE_NAMES = {
    _musa_env("GEMV_MOE_BLOCK"),
    _musa_env("FUSED_ADD_RMSNORM", "_BLOCK_X"),
    _musa_env("DEEPGEMM_ROW_MAJOR", "_ACT_SCALES"),
    _musa_env("FP8_SMALL_M", "_GEMV"),
    _musa_env("FP8_SMALL_M", "_GEMV_MAX_M"),
}


def _source_files():
    for root in SOURCE_ROOTS:
        if root.is_file():
            yield root
            continue
        for path in root.rglob("*"):
            if (
                path.is_file()
                and path.suffix not in {".pyc", ".so", ".o"}
                and "__pycache__" not in path.parts
            ):
                yield path


def _read_all_sources() -> str:
    chunks = []
    for path in _source_files():
        chunks.append(path.read_text(errors="ignore"))
    return "\n".join(chunks)


def test_no_deepseek_v4_env_gate_names_remain_in_active_source():
    source = _read_all_sources()

    for pattern in REMOVED_DEEPSEEK_ENV_PATTERNS:
        assert pattern.search(source) is None, pattern.pattern


def test_deepseek_profile_generic_env_names_removed_from_active_source():
    source = _read_all_sources()

    for name in REMOVED_GENERIC_PROFILE_NAMES:
        assert name not in source


def test_remaining_vllm_musa_env_names_are_reviewed_allowlist():
    source = _read_all_sources()
    names = set(re.findall(r"VLLM_MUSA_[A-Z0-9_]+", source))

    assert names <= RETAINED_VLLM_MUSA_NAMES
