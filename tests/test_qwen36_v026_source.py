# SPDX-License-Identifier: Apache-2.0
"""Source contracts for Qwen3.6 compatibility with the v0.26 vLLM pin."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GDN_SOURCE = (
    ROOT
    / "vllm_musa"
    / "model_executor"
    / "layers"
    / "mamba"
    / "gdn"
    / "qwen_gdn_linear_attn.py"
)
QWEN_PATCH = (
    ROOT
    / "vllm_musa"
    / "patches"
    / "series"
    / "0085-MUSA-model-fuse-QK-RMSNorm-and-MRoPE-for-interleaved.patch"
)


def test_qwen_gdn_forwards_v026_reduce_results() -> None:
    source = GDN_SOURCE.read_text(encoding="utf-8")

    assert "reduce_results: bool = True" in source
    assert "gqa_interleaved_layout,\n            reduce_results," in source


def test_qwen_gdn_uses_v026_fla_import_path() -> None:
    source = GDN_SOURCE.read_text(encoding="utf-8")

    assert source.count("from vllm.third_party.flash_linear_attention.ops import") == 2
    assert "from vllm.model_executor.layers.fla.ops import" not in source


def test_qwen_mrope_patch_imports_os_before_use() -> None:
    source = QWEN_PATCH.read_text(encoding="utf-8")

    assert "+import os" in source
    assert source.index("+import os") < source.index("os.environ.get")
