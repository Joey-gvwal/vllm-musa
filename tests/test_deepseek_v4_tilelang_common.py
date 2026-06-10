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


def test_pass_config_defaults_to_dsa_full_profile():
    configs = kernel_common._tilelang_musa_pass_configs(_TileLangStub())

    assert configs["disable_host_asserts"] is True
    assert "-mtgpu-combine-instr-with-burst=1" in configs["compile_flags"]


def test_pass_config_can_be_disabled_explicitly():
    assert (
        kernel_common._tilelang_musa_pass_configs(
            _TileLangStub(),
            compile_profile="none",
        )
        is None
    )


def test_burst_reduce_pass_config_uses_default_profile():
    configs = kernel_common._tilelang_musa_burst_reduce_pass_configs(_TileLangStub())

    assert configs["burst"] is True
    assert configs["reduce_burst"] is True
    assert configs["disable_host_asserts"] is True
    assert "-mtgpu-combine-instr-with-burst=1" in configs["compile_flags"]


def test_explicit_compile_profile_override():
    configs = kernel_common._tilelang_musa_pass_configs(
        _TileLangStub(),
        compile_profile="ls",
    )

    assert "-mtgpu-load-store-opt=1" in configs["compile_flags"]
    assert "disable_thread_storage_sync" not in configs
    assert "disable_safe_memory" not in configs


def test_aggressive_pass_configs_are_code_selected():
    configs = kernel_common._tilelang_musa_aggressive_pass_configs(
        _TileLangStub(),
        disable_index_promotion=True,
    )

    assert configs["disable_host_asserts"] is True
    assert configs["disable_thread_storage_sync"] is True
    assert configs["disable_safe_memory"] is True
    assert configs["lower_ldgstg"] is True
    assert configs["lower_ldgstg_predicated"] is True
    assert configs["disable_index_promotion"] is True
    assert "-mtgpu-combine-instr-with-burst=1" in configs["compile_flags"]


def test_explicit_dsa_profile_helper():
    configs = kernel_common._tilelang_musa_dsa_pass_configs(
        _TileLangStub(),
        full=True,
    )

    assert configs["burst"] is True
    assert configs["reduce_burst"] is True
    assert configs["disable_index_promotion"] is True
    assert "-mtgpu-combine-instr-with-burst=1" in configs["compile_flags"]


def test_tilelang_jit_uses_name_target_and_pass_configs():
    tilelang = _TileLangStub()

    result = kernel_common._tilelang_jit(tilelang, "kernel_name")

    assert result["name"] == "kernel_name"
    assert result["target"] == "musa"
    assert result["pass_configs"]["disable_host_asserts"] is True
    assert "-mtgpu-combine-instr-with-burst=1" in result["pass_configs"]["compile_flags"]


def test_tilelang_jit_falls_back_for_old_tilelang_apis():
    tilelang = _TileLangStub(unsupported={"name", "pass_configs"})

    result = kernel_common._tilelang_jit(tilelang, "kernel_name")

    assert result == {"target": "musa"}
