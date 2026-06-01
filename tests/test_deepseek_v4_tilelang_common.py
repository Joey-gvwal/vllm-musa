# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DeepSeek-V4 TileLang MUSA helper compatibility."""

from vllm_musa.deepseek_v4_jit import kernel_common


class _PassConfigKey:
    TL_ENABLE_MUSA_BURST = "burst"
    TL_ENABLE_REDUCE_BURST = "reduce_burst"
    TL_DEVICE_COMPILE_FLAGS = "compile_flags"
    TL_DISABLE_THREAD_STORAGE_SYNC = "disable_thread_storage_sync"
    TL_DISABLE_SAFE_MEMORY_ACCESS = "disable_safe_memory"
    TL_ENABLE_LOWER_LDGSTG = "lower_ldgstg"
    TL_ENABLE_LOWER_LDGSTG_PREDICATED = "lower_ldgstg_predicated"
    TL_DISABLE_INDEX_TYPE_PROMOTION = "disable_index_promotion"
    TL_DISABLE_HOST_ASSERTS = "disable_host_asserts"


class _TileLangStub:
    PassConfigKey = _PassConfigKey

    def __init__(self, unsupported=()):
        self.unsupported = set(unsupported)
        self.calls = []

    def jit(self, **kwargs):
        for key in self.unsupported:
            if key in kwargs:
                raise TypeError(f"unexpected keyword argument: {key}")
        self.calls.append(kwargs)
        return kwargs


def test_pass_config_default_is_disabled(monkeypatch):
    monkeypatch.delenv("VLLM_MUSA_DEEPSEEK_V4_TILELANG_PASS_CONFIG", raising=False)
    monkeypatch.delenv("VLLM_MUSA_DEEPSEEK_V4_TILELANG_COMPILE_PROFILE", raising=False)

    assert kernel_common._tilelang_musa_pass_configs(_TileLangStub()) is None


def test_old_pass_config_env_preserves_opt1_flags(monkeypatch):
    monkeypatch.setenv("VLLM_MUSA_DEEPSEEK_V4_TILELANG_PASS_CONFIG", "1")
    monkeypatch.delenv("VLLM_MUSA_DEEPSEEK_V4_TILELANG_COMPILE_PROFILE", raising=False)

    configs = kernel_common._tilelang_musa_pass_configs(_TileLangStub())

    assert configs["burst"] is True
    assert configs["reduce_burst"] is True
    assert "-mtgpu-opt-level=1" in configs["compile_flags"]


def test_compile_profile_and_aggressive_switches(monkeypatch):
    monkeypatch.delenv("VLLM_MUSA_DEEPSEEK_V4_TILELANG_PASS_CONFIG", raising=False)
    monkeypatch.setenv("VLLM_MUSA_DEEPSEEK_V4_TILELANG_COMPILE_PROFILE", "ls")
    monkeypatch.setenv("VLLM_MUSA_DEEPSEEK_V4_TILELANG_AGGRESSIVE_PASS_CONFIG", "1")
    monkeypatch.setenv("VLLM_MUSA_DEEPSEEK_V4_TILELANG_DISABLE_INDEX_PROMOTION", "1")

    configs = kernel_common._tilelang_musa_pass_configs(_TileLangStub())

    assert "-mtgpu-load-store-opt=1" in configs["compile_flags"]
    assert configs["disable_thread_storage_sync"] is True
    assert configs["disable_safe_memory"] is True
    assert configs["lower_ldgstg"] is True
    assert configs["lower_ldgstg_predicated"] is True
    assert configs["disable_index_promotion"] is True


def test_host_asserts_can_be_disabled_without_aggressive_pass(monkeypatch):
    monkeypatch.delenv("VLLM_MUSA_DEEPSEEK_V4_TILELANG_PASS_CONFIG", raising=False)
    monkeypatch.setenv("VLLM_MUSA_DEEPSEEK_V4_TILELANG_COMPILE_PROFILE", "dsa_full")
    monkeypatch.setenv("VLLM_MUSA_DEEPSEEK_V4_TILELANG_DISABLE_HOST_ASSERTS", "1")
    monkeypatch.delenv(
        "VLLM_MUSA_DEEPSEEK_V4_TILELANG_AGGRESSIVE_PASS_CONFIG", raising=False
    )

    configs = kernel_common._tilelang_musa_pass_configs(_TileLangStub())

    assert configs["disable_host_asserts"] is True
    assert "disable_thread_storage_sync" not in configs
    assert "disable_safe_memory" not in configs
    assert "-mtgpu-combine-instr-with-burst=1" in configs["compile_flags"]


def test_explicit_dsa_profile_helper(monkeypatch):
    monkeypatch.delenv("VLLM_MUSA_DEEPSEEK_V4_TILELANG_COMPILE_PROFILE", raising=False)

    configs = kernel_common._tilelang_musa_dsa_pass_configs(
        _TileLangStub(),
        full=True,
    )

    assert configs["burst"] is True
    assert configs["reduce_burst"] is True
    assert configs["disable_index_promotion"] is True
    assert "-mtgpu-combine-instr-with-burst=1" in configs["compile_flags"]


def test_tilelang_jit_uses_name_target_and_pass_configs(monkeypatch):
    monkeypatch.setenv("VLLM_MUSA_DEEPSEEK_V4_TILELANG_PASS_CONFIG", "1")
    tilelang = _TileLangStub()

    result = kernel_common._tilelang_jit(tilelang, "kernel_name")

    assert result["name"] == "kernel_name"
    assert result["target"] == "musa"
    assert result["pass_configs"]["burst"] is True


def test_tilelang_jit_falls_back_for_old_tilelang_apis(monkeypatch):
    monkeypatch.setenv("VLLM_MUSA_DEEPSEEK_V4_TILELANG_PASS_CONFIG", "1")
    tilelang = _TileLangStub(unsupported={"name", "pass_configs"})

    result = kernel_common._tilelang_jit(tilelang, "kernel_name")

    assert result == {"target": "musa"}
