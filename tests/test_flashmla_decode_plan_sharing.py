# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_musa.v1.attention.ops import flashmla

ROOT = Path(__file__).resolve().parents[1]
SERIES = ROOT / "vllm_musa" / "patches" / "series"


def _use_context(monkeypatch: pytest.MonkeyPatch, context) -> None:
    from vllm import forward_context

    monkeypatch.setattr(
        forward_context, "is_forward_context_available", lambda: context is not None
    )
    monkeypatch.setattr(forward_context, "get_forward_context", lambda: context)


def test_same_key_reuses_the_first_plan_of_the_forward(monkeypatch) -> None:
    pytest.importorskip("vllm")
    _use_context(monkeypatch, SimpleNamespace())
    first, second, other = object(), object(), object()

    assert flashmla.shared_decode_plan((2, 64, 2), first) is first
    assert flashmla.shared_decode_plan((2, 64, 2), second) is first
    assert flashmla.shared_decode_plan((1, 64, 2), other) is other
    assert flashmla.shared_decode_plan((2, 64, 4), second) is second


def test_each_forward_starts_without_plans(monkeypatch) -> None:
    pytest.importorskip("vllm")
    first, second = object(), object()
    _use_context(monkeypatch, SimpleNamespace())
    assert flashmla.shared_decode_plan((0, 64, None), first) is first

    _use_context(monkeypatch, SimpleNamespace())
    assert flashmla.shared_decode_plan((0, 64, None), second) is second


def test_without_a_forward_context_or_plan_nothing_is_shared(monkeypatch) -> None:
    pytest.importorskip("vllm")
    plan = object()
    _use_context(monkeypatch, None)
    assert flashmla.shared_decode_plan((0, 64, None), plan) is plan

    context = SimpleNamespace()
    _use_context(monkeypatch, context)
    assert flashmla.shared_decode_plan((0, 64, None), None) is None
    assert flashmla.shared_decode_plan((0, 64, None), plan) is plan


def test_v41_decode_keys_the_plan_by_layer_type_and_cache_layout() -> None:
    (patch,) = SERIES.glob("0038-*.patch")
    text = patch.read_text(encoding="utf-8")
    assert "+    shared_decode_plan," in text
    hook = text[text.index("+        tile_metadata = shared_decode_plan(") :]
    hook = hook[: hook.index("+        )\n")]
    for part in (
        "self.compress_ratio",
        "swa_cache.shape[1]",
        "kv_cache.shape[1] if kv_cache is not None else None",
    ):
        assert part in hook
