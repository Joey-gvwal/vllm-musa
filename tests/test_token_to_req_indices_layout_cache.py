"""Regression: MTP must not reuse a stale token_to_req_indices cache.

v0.26 CommonAttentionMetadata.token_to_req_indices caches the first built
request-index map. MTP verify runs with multi-token query layout, then the
draft loop rewrites the same metadata object to 1 token/request. Without a
layout fingerprint + invalidation, builders get the multi-token map truncated
to the new token count and map draft tokens to the wrong request.

These tests run with CPU torch and do not require MUSA hardware. They consume
an already-patched vLLM tree through ``PYTHONPATH``.
"""

from __future__ import annotations

import pytest
import torch

import vllm.utils.torch_utils as torch_utils
from vllm.v1.attention.backend import CommonAttentionMetadata


@pytest.fixture(autouse=True)
def _disable_pin_memory_for_cpu_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep this CPU mapping contract independent of MUSA driver availability."""
    monkeypatch.setattr(torch_utils, "PIN_MEMORY", False)


def _make_cm(
    batch_size: int,
    query_len: int,
    *,
    device: str = "cpu",
) -> CommonAttentionMetadata:
    num_tokens = batch_size * query_len
    starts = torch.arange(batch_size + 1, dtype=torch.int32) * query_len
    query_start_loc = starts.to(device=device)
    query_start_loc_cpu = starts.clone()
    seq_lens = torch.full(
        (batch_size,), 128 + query_len, dtype=torch.int32, device=device
    )
    block_table = torch.zeros((batch_size, 4), dtype=torch.int32, device=device)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        num_reqs=batch_size,
        num_actual_tokens=num_tokens,
        max_query_len=query_len,
        max_seq_len=int(seq_lens.max().item()),
        block_table_tensor=block_table,
        slot_mapping=slot_mapping,
    )


def _expected_map(batch_size: int, query_len: int) -> list[int]:
    out: list[int] = []
    for req in range(batch_size):
        out.extend([req] * query_len)
    return out


def test_token_to_req_indices_rebuilds_after_mtp_layout_mutation() -> None:
    """Verify → draft mutation must not keep the multi-token request map."""
    batch_size = 64
    verify_q = 5  # 1 + num_speculative_tokens (MTP4)
    draft_q = 1

    cm = _make_cm(batch_size, verify_q)
    buf_a = torch.full((batch_size * verify_q,), -1, dtype=torch.int32)
    verify_map = cm.token_to_req_indices(buf_a).tolist()
    assert verify_map == _expected_map(batch_size, verify_q)
    assert cm._token_to_req_indices_layout == (
        batch_size * verify_q,
        *range(0, (batch_size + 1) * verify_q, verify_q),
    )

    # Draft loop mutation (same object, 1 token/request).
    cm.num_actual_tokens = batch_size
    cm.max_query_len = 1
    cm.query_start_loc[: batch_size + 1] = torch.arange(
        batch_size + 1, dtype=torch.int32
    )
    cm.query_start_loc_cpu[: batch_size + 1] = torch.arange(
        batch_size + 1, dtype=torch.int32
    )
    # Explicit invalidation (what the proposer patch adds).
    cm.invalidate_query_layout_caches()

    buf_b = torch.full((batch_size * verify_q,), -1, dtype=torch.int32)
    draft_map = cm.token_to_req_indices(buf_b).tolist()
    assert draft_map == _expected_map(batch_size, draft_q)
    # Stale multi-token prefix would be [0,0,0,0,0,1,1,...][:64], not [0..63].
    stale_prefix = _expected_map(batch_size, verify_q)[:batch_size]
    assert draft_map != stale_prefix
    assert draft_map == list(range(batch_size))


def test_token_to_req_indices_layout_fingerprint_without_explicit_invalidate() -> None:
    """Even without explicit invalidation, a layout fingerprint must rebuild."""
    batch_size = 16
    cm = _make_cm(batch_size, query_len=5)
    buf = torch.zeros(batch_size * 5, dtype=torch.int32)
    first = cm.token_to_req_indices(buf).clone()
    assert first.tolist() == _expected_map(batch_size, 5)

    # Mutate layout but do NOT call invalidate — fingerprint must still rebuild.
    cm.num_actual_tokens = batch_size
    cm.query_start_loc[: batch_size + 1] = torch.arange(
        batch_size + 1, dtype=torch.int32
    )
    cm.query_start_loc_cpu[: batch_size + 1] = torch.arange(
        batch_size + 1, dtype=torch.int32
    )

    second_buf = torch.zeros(batch_size * 5, dtype=torch.int32)
    second = cm.token_to_req_indices(second_buf)
    assert second.tolist() == list(range(batch_size))


def test_token_to_req_indices_always_materializes_into_caller_buffer() -> None:
    """Different builders pass different CUDA-graph-stable buffers."""
    batch_size = 8
    cm = _make_cm(batch_size, query_len=5)
    buf_a = torch.full((40,), -7, dtype=torch.int32)
    buf_b = torch.full((40,), -9, dtype=torch.int32)

    out_a = cm.token_to_req_indices(buf_a)
    out_b = cm.token_to_req_indices(buf_b)

    assert out_a.data_ptr() == buf_a.data_ptr()
    assert out_b.data_ptr() == buf_b.data_ptr()
    assert out_a.tolist() == out_b.tolist() == _expected_map(batch_size, 5)
    # Cached content is full multi-token map; both buffers must hold a copy.
    assert buf_a[:40].tolist() == _expected_map(batch_size, 5)
    assert buf_b[:40].tolist() == _expected_map(batch_size, 5)


def test_stale_prefix_math_matches_observed_acceptance_collapse() -> None:
    """Document the truncated-map failure that previously sank BS64 accept rate.

    The stale map is ``stale[i] = i // query_len`` while the correct 1-token map
    is ``correct[i] = i``. For ``query_len > 1`` those agree only at ``i == 0``,
    so exactly one of ``batch_size`` draft tokens lands on its own request and
    the map spans only ``ceil(batch_size / query_len)`` distinct requests.
    """
    for batch_size, query_len in ((4, 5), (16, 5), (64, 5)):
        stale = _expected_map(batch_size, query_len)[:batch_size]
        assert stale == [i // query_len for i in range(batch_size)]

        matches = sum(a == b for a, b in zip(stale, range(batch_size)))
        assert matches == 1
        assert matches / batch_size == 1.0 / batch_size

        expected_unique = -(-batch_size // query_len)  # ceil
        assert len(set(stale)) == expected_unique

    # BS64/MTP4: 1.5625% of draft tokens map to the right request, which bounds
    # the 2.11% acceptance observed before the fix.
    assert 1.0 / 64 < 0.0211


def test_layout_fingerprint_includes_per_request_boundaries() -> None:
    """Equal totals do not imply equal token-to-request mappings."""
    # [1, 3] and [2, 2] have the same request count and token count. A
    # size-only fingerprint would incorrectly reuse the first mapping.
    cm = _make_cm(batch_size=2, query_len=2)
    buf = torch.zeros(4, dtype=torch.int32)
    assert cm.token_to_req_indices(buf).tolist() == [0, 0, 1, 1]

    cm.query_start_loc[:3] = torch.tensor([0, 1, 4], dtype=torch.int32)
    cm.query_start_loc_cpu[:3] = torch.tensor([0, 1, 4], dtype=torch.int32)
    cm.num_actual_tokens = 4
    second = cm.token_to_req_indices(torch.empty(4, dtype=torch.int32))
    assert second.tolist() == [0, 1, 1, 1]
