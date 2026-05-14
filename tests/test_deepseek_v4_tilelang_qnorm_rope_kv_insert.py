# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path

import torch

from vllm_musa.deepseek_v4_jit import qnorm_rope_kv_insert as jit_insert


def _load_attention_patch_module():
    patch_path = (
        Path(__file__).resolve().parents[1]
        / "vllm_musa"
        / "patches"
        / "vllm__model_executor__layers__deepseek_v4_attention.patch.py"
    )
    spec = importlib.util.spec_from_file_location(
        "deepseek_v4_attention_patch_for_tilelang_test", patch_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_tilelang_qnorm_rope_guard_rejects_cpu_before_tilelang_import(monkeypatch):
    monkeypatch.setenv("VLLM_MUSA_DEEPSEEK_V4_QNORM_ROPE_KV_INSERT_IMPL", "auto")
    sys.modules.pop("tilelang", None)
    q = torch.empty((2, 3, 512), dtype=torch.bfloat16)
    kv = torch.empty((2, 512), dtype=torch.bfloat16)
    cache = torch.empty((4, 256 * 584), dtype=torch.uint8)
    slot_mapping = torch.tensor([0, -1], dtype=torch.int64)
    positions = torch.tensor([0, 1], dtype=torch.int64)
    cos_sin_cache = torch.empty((8, 64), dtype=torch.float32)

    handled, reason = jit_insert.try_tilelang_qnorm_rope_kv_insert(
        q,
        kv,
        cache,
        slot_mapping,
        positions,
        cos_sin_cache,
        eps=1e-6,
        block_size=256,
    )

    assert handled is False
    assert "MUSA" in reason
    assert "tilelang" not in sys.modules


def test_tilelang_qnorm_rope_cache_layout_constants_match_fp8_ds_mla():
    assert jit_insert._TOKEN_VALUE_BYTES == 576
    assert jit_insert._TOKEN_SCALE_BYTES == 8
    assert 256 * (
        jit_insert._TOKEN_VALUE_BYTES + jit_insert._TOKEN_SCALE_BYTES
    ) == 256 * 584


def test_attention_patch_prefers_tilelang_before_torch_qnorm_fallback():
    module = _load_attention_patch_module()
    replacement = next(
        new
        for old, new in module.PATCHES
        if "def _musa_fused_deepseek_v4_qnorm_rope_kv_insert_fallback" in new
    )

    tilelang_pos = replacement.index(
        "_musa_try_tilelang_deepseek_v4_qnorm_rope_kv_insert"
    )
    torch_pos = replacement.index("q_float = q.to(torch.float32)")

    assert "vllm_musa.deepseek_v4_jit.qnorm_rope_kv_insert" in replacement
    assert tilelang_pos < torch_pos
