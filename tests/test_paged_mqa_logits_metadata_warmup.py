# SPDX-License-Identifier: Apache-2.0
"""Startup build of mate's paged MQA logits metadata modules."""

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
WARMUP_SOURCE = ROOT / "vllm_musa" / "model_executor" / "warmup" / "deep_gemm_warmup.py"


def test_deep_gemm_warmup_builds_the_metadata_buckets() -> None:
    source = WARMUP_SOURCE.read_text()

    assert "_MQA_LOGITS_METADATA_MAX_ROWS = 512" in source
    assert "get_paged_mqa_logits_metadata_module(rows)" in source
    assert "_warmup_paged_mqa_logits_metadata(model)" in source


@pytest.fixture
def warmup(monkeypatch):
    pytest.importorskip("vllm")
    from vllm_musa.model_executor.warmup import deep_gemm_warmup

    built: list[int] = []
    fake = types.ModuleType("mate.jit.gemm.deep_gemm")
    fake.get_paged_mqa_logits_metadata_module = built.append
    monkeypatch.setitem(sys.modules, "mate.jit.gemm.deep_gemm", fake)
    return deep_gemm_warmup, built


def test_metadata_warmup_covers_every_32_row_bucket_up_to_512(
    warmup, monkeypatch
) -> None:
    module, built = warmup
    monkeypatch.setattr(module, "_uses_sparse_indexer", lambda model: True)

    module._warmup_paged_mqa_logits_metadata(object())

    assert built == list(range(32, 513, 32))


def test_metadata_warmup_skips_models_without_the_sparse_indexer(
    warmup, monkeypatch
) -> None:
    module, built = warmup
    monkeypatch.setattr(module, "_uses_sparse_indexer", lambda model: False)

    module._warmup_paged_mqa_logits_metadata(object())

    assert built == []


def test_metadata_warmup_without_mate_is_a_no_op(warmup, monkeypatch) -> None:
    module, built = warmup
    monkeypatch.setattr(module, "_uses_sparse_indexer", lambda model: True)
    monkeypatch.setitem(sys.modules, "mate.jit.gemm.deep_gemm", None)

    module._warmup_paged_mqa_logits_metadata(object())

    assert built == []
