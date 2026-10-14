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


def test_global_topk_is_mapped_once_per_key_and_forward(monkeypatch) -> None:
    pytest.importorskip("vllm")
    calls = []

    def compute(tag):
        def run():
            calls.append(tag)
            return (tag, tag)

        return run

    _use_context(monkeypatch, SimpleNamespace())
    assert flashmla.shared_global_topk((2, "layers.2"), compute("a")) == ("a", "a")
    assert flashmla.shared_global_topk((2, "layers.2"), compute("b")) == ("a", "a")
    assert flashmla.shared_global_topk((24, "layers.20"), compute("c")) == ("c", "c")
    assert calls == ["a", "c"]

    _use_context(monkeypatch, SimpleNamespace())
    assert flashmla.shared_global_topk((2, "layers.2"), compute("d")) == ("d", "d")
    _use_context(monkeypatch, None)
    assert flashmla.shared_global_topk((2, "layers.2"), compute("e")) == ("e", "e")
    assert calls == ["a", "c", "d", "e"]


def test_v41_decode_keys_the_global_topk_by_index_source_and_cache() -> None:
    (patch,) = SERIES.glob("0040-*.patch")
    text = patch.read_text(encoding="utf-8")
    assert "+    shared_global_topk," in text
    start = "+            global_indices, topk_lens = shared_global_topk("
    hook = text[text.index(start) :]
    hook = hook[: hook.index("topk_indices = global_indices.view")]
    assert "(self.index_source_layer_id, self.compressed_cache_prefix)" in hook
    assert "compute_global_topk_indices_and_lens(" in hook


class _FakeTensor:
    def __init__(self, shape, start=0, size=1, contiguous=True):
        self.shape, self.dtype, self.device = shape, "bf16", "musa:0"
        self._start, self._size, self._contiguous = start, size, contiguous

    def is_contiguous(self):
        return self._contiguous

    def data_ptr(self):
        return self._start

    def numel(self):
        return self._size

    def element_size(self):
        return 1


def _fake_decode_common(monkeypatch):
    def allocate(batch, seq_len, heads, dim, *_):
        return "glse", "partial", _FakeTensor((batch, seq_len, heads, dim), 1000), "lse"

    common = SimpleNamespace(allocate_scheduled_decode_outputs=allocate)
    monkeypatch.setattr(flashmla, "_mate_decode_common", common)
    return common, allocate


def test_decode_writes_into_the_caller_buffer_once(monkeypatch) -> None:
    common, allocate = _fake_decode_common(monkeypatch)
    out = _FakeTensor((5, 1, 64, 512), 0)
    q = _FakeTensor((5, 1, 64, 512), 5000)
    with flashmla._decode_writes_into(out, q, None):
        first = common.allocate_scheduled_decode_outputs(5, 1, 64, 512)
        second = common.allocate_scheduled_decode_outputs(5, 1, 64, 512)
    assert first[2] is out and first[3] == "lse"
    assert second[2] is not out
    assert common.allocate_scheduled_decode_outputs is allocate


def test_decode_keeps_its_own_buffer_when_out_does_not_fit(monkeypatch) -> None:
    common, allocate = _fake_decode_common(monkeypatch)
    q = _FakeTensor((5, 1, 64, 512), 5000, size=10)
    cases = (
        (_FakeTensor((5, 1, 16, 512), 0), ()),
        (_FakeTensor((5, 1, 64, 512), 0, contiguous=False), ()),
        (_FakeTensor((5, 1, 64, 512), 5004, size=10), (q,)),
    )
    for out, inputs in cases:
        with flashmla._decode_writes_into(out, *inputs):
            got = common.allocate_scheduled_decode_outputs(5, 1, 64, 512)
        assert got[2] is not out
    assert common.allocate_scheduled_decode_outputs is allocate


def _has_musa_device() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return hasattr(torch, "musa") and torch.musa.is_available()


@pytest.mark.skipif(not _has_musa_device(), reason="requires a MUSA device")
def test_mate_sparse_decode_writes_into_out_bitwise() -> None:
    import torch

    flash_mla = pytest.importorskip("flash_mla")
    sparse_mla = pytest.importorskip("mate.testing.sparse_mla")
    case = sparse_mla.SparseMlaCase(
        name="direct-out",
        variant="v4",
        q_len=1,
        kv_len=4096,
        heads=64,
        topk=128,
        batch=5,
        page_size=64,
        topk_length=True,
        attn_sink=True,
        extra_kv_len=4096,
        extra_topk=512,
    )
    data = sparse_mla.make_decode_data(case, "musa")

    def decode(out):
        sched, _ = flash_mla.get_mla_metadata()
        return flashmla.flash_mla_with_kvcache(
            q=data.q,
            k_cache=data.k_cache,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=512,
            tile_scheduler_metadata=sched,
            softmax_scale=512**-0.5,
            is_fp8_kvcache=True,
            indices=data.indices.view(5, 1, -1),
            attn_sink=data.attn_sink,
            extra_k_cache=data.extra_k_cache,
            extra_indices_in_kvcache=data.extra_indices.view(5, 1, -1),
            topk_length=data.topk_length,
            extra_topk_length=data.extra_topk_length,
            out=out,
        )

    ref, _ = decode(None)
    out = torch.full_like(ref, float("nan"))
    got, _ = decode(out)
    assert got.data_ptr() == out.data_ptr()
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))
