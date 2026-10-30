# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract tests for the DeepSelect MUSA backend (``torch.ops.deep_select``).

The backend is a reference implementation of DeepSelect's documented semantics, so
these tests are the guard for everything vLLM's indexer relies on: per-entry
padding, ``end`` bounds, ``output_idx_offset`` applied to real entries only,
``sorted_index`` ordering with values kept paired, the unconditional NaN marker,
unique indices under ties, and ``k`` smaller than the preallocated output width.

Runs on CPU and, when the torch build has MUSA, on ``musa``. The op is registered
for both dispatch keys by ``vllm_musa.deep_select.registry``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vllm_musa.deep_select import backend, ensure_registered, get_alignment_requirement
from vllm_musa.deep_select.registry import is_registered

FILL = -1
VAL_FILL = float("-inf")
NAN_MARKER = 0x3F3F3F3F
TAIL_SENTINEL = 7


def _devices() -> list[str]:
    devices = ["cpu"]
    if getattr(torch.version, "musa", None):
        devices.append("musa")
    return devices


DEVICES = _devices()


@pytest.fixture(scope="module", autouse=True)
def _registered():
    ensure_registered()
    assert is_registered(), "torch.ops.deep_select was not registered"


def _call(
    input: torch.Tensor,
    k: int,
    *,
    end: torch.Tensor | None = None,
    offset: torch.Tensor | None = None,
    capacity: int | None = None,
    sorted_index: bool = False,
    return_value: bool = False,
    abort: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Drive the registered op the way vLLM's indexer does."""
    rows = input.shape[0]
    width = capacity if capacity is not None else k
    value_width = width
    if return_value:
        elements_per_alignment = 32 // input.element_size()
        value_width = max(
            value_width,
            ((k + elements_per_alignment - 1) // elements_per_alignment)
            * elements_per_alignment,
        )
    out_idx = torch.full(
        (rows, width), TAIL_SENTINEL, dtype=torch.int32, device=input.device
    )
    out_val = (
        torch.full(
            (rows, value_width), VAL_FILL, dtype=input.dtype, device=input.device
        )
        if return_value
        else None
    )
    torch.ops.deep_select.topk(
        input,
        k,
        None,
        end,
        False,
        sorted_index,
        out_val,
        out_idx,
        offset,
        FILL,
        VAL_FILL,
        return_value,
        abort,
    )
    return out_val[:, :k] if out_val is not None else None, out_idx


def _row_len(row: int, vocab: int, end: torch.Tensor | None) -> int:
    if end is None:
        return vocab
    return max(0, min(int(end[row]), vocab))


def _assert_contract(
    input: torch.Tensor,
    out_idx: torch.Tensor,
    out_val: torch.Tensor | None,
    k: int,
    *,
    end: torch.Tensor | None = None,
    offset: torch.Tensor | None = None,
    sorted_index: bool = False,
) -> None:
    vocab = input.shape[1]
    for row in range(input.shape[0]):
        n = _row_len(row, vocab, end)
        m = min(k, n)
        off = int(offset[row]) if offset is not None else 0
        got_idx = out_idx[row, :m].to(torch.int64)
        if off:
            got_idx = got_idx - off
        if m == 0:
            assert torch.all(out_idx[row, :k] == FILL), (
                f"row {row}: empty row not filled"
            )
            continue
        expected_val, expected_idx = torch.topk(
            input[row, :n], m, largest=True, sorted=True
        )
        assert torch.all((got_idx >= 0) & (got_idx < n)), (
            f"row {row}: out-of-range indices {got_idx[:8].tolist()}"
        )
        got_val = input[row, got_idx]
        assert torch.equal(torch.sort(got_val).values, torch.sort(expected_val).values), (
            f"row {row}: selected values do not match top-k values"
        )
        if out_val is not None:
            # Values must stay paired with their (unshifted) indices.
            assert torch.equal(out_val[row, :m], got_val)
            assert torch.equal(
                torch.sort(out_val[row, :m]).values, torch.sort(expected_val).values
            )
        # Padding: the documented fill, never shifted by the offset.
        assert torch.all(out_idx[row, m:k] == FILL), f"row {row}: padding not {FILL}"
        if sorted_index:
            assert torch.all(out_idx[row, :m][:-1] <= out_idx[row, :m][1:])


@pytest.mark.parametrize("device", DEVICES)
def test_alignment_requirement(device: str) -> None:
    assert get_alignment_requirement() == [32, 32]
    assert list(torch.ops.deep_select.get_alignment_requirement()) == [32, 32]


@pytest.mark.parametrize("device", DEVICES)
def test_full_rows_match_torch_topk(device: str) -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 4096, device=device, dtype=torch.float32)
    out_val, out_idx = _call(x, 512, return_value=True)
    _assert_contract(x, out_idx, out_val, 512)
    assert out_idx.dtype == torch.int32


@pytest.mark.parametrize("device", DEVICES)
def test_end_bounds_pad_the_tail(device: str) -> None:
    torch.manual_seed(1)
    x = torch.randn(5, 2048, device=device, dtype=torch.float32)
    # 0 (empty), < k, == k, between k and vocab, == vocab.
    end = torch.tensor([0, 23, 512, 1000, 2048], dtype=torch.int32, device=device)
    out_val, out_idx = _call(x, 512, end=end, return_value=True)
    _assert_contract(x, out_idx, out_val, 512, end=end)
    assert torch.all(out_idx[0] == FILL)


@pytest.mark.parametrize("device", DEVICES)
def test_offset_applies_to_real_entries_only(device: str) -> None:
    torch.manual_seed(2)
    x = torch.randn(3, 1024, device=device, dtype=torch.bfloat16)
    end = torch.tensor([0, 8, 1024], dtype=torch.int32, device=device)
    offset = torch.tensor([99, 99, 99], dtype=torch.int32, device=device)
    out_val, out_idx = _call(x, 64, end=end, offset=offset, return_value=True)
    _assert_contract(x, out_idx, out_val, 64, end=end, offset=offset)
    # The regression this guards: with a negative fill value a padded slot used
    # to come back as offset-1, a valid-looking key index. Row 0 is empty, so
    # every slot must be the fill, never the offset.
    assert torch.all(out_idx[0] == FILL)


@pytest.mark.parametrize("device", DEVICES)
def test_k_smaller_than_capacity_keeps_the_tail(device: str) -> None:
    torch.manual_seed(3)
    x = torch.randn(2, 512, device=device, dtype=torch.float32)
    end = torch.tensor([23, 512], dtype=torch.int32, device=device)
    out_val, out_idx = _call(x, 23, end=end, capacity=2048, return_value=True)
    _assert_contract(x, out_idx, out_val, 23, end=end)
    assert torch.all(out_idx[:, 23:] == TAIL_SENTINEL)


@pytest.mark.parametrize("device", DEVICES)
def test_sorted_index_orders_the_real_block(device: str) -> None:
    torch.manual_seed(4)
    x = torch.randn(3, 4096, device=device, dtype=torch.float32)
    end = torch.tensor([4096, 300, 0], dtype=torch.int32, device=device)
    out_val, out_idx = _call(
        x, 512, end=end, offset=None, sorted_index=True, return_value=True
    )
    _assert_contract(x, out_idx, out_val, 512, end=end, sorted_index=True)


@pytest.mark.parametrize("device", DEVICES)
def test_nan_marker_is_written_without_abort(device: str) -> None:
    torch.manual_seed(5)
    x = torch.randn(4, 1024, device=device, dtype=torch.float32)
    x[1, 3] = float("nan")
    out_val, out_idx = _call(x, 64, return_value=True, abort=False)
    assert int(out_idx[1, 0]) == NAN_MARKER
    # Clean rows keep a usable answer.
    for row in (0, 2, 3):
        assert int(out_idx[row, 0]) != NAN_MARKER


def test_abort_raises_on_nan_cpu() -> None:
    torch.manual_seed(6)
    x = torch.randn(4, 1024, device="cpu", dtype=torch.float32)
    x[0, 7] = float("nan")
    with pytest.raises(RuntimeError, match="NaN"):
        _call(x, 64, return_value=True, abort=True)


@pytest.mark.parametrize("device", DEVICES)
def test_topk_larger_than_vocab(device: str) -> None:
    torch.manual_seed(7)
    x = torch.randn(2, 8, device=device, dtype=torch.float32)
    end = torch.full((2,), 5, dtype=torch.int32, device=device)
    out_val, out_idx = _call(x, 8, end=end, return_value=True)
    _assert_contract(x, out_idx, out_val, 8, end=end)


@pytest.mark.parametrize("device", DEVICES)
def test_tied_bf16_values_give_unique_indices(device: str) -> None:
    """torch_musa's bf16 topk duplicates indices on ties; the backend must not."""
    torch.manual_seed(8)
    x = torch.randint(0, 3, (4, 4096), device=device).to(torch.bfloat16)
    _, out_idx = _call(x, 256)
    for row in range(x.shape[0]):
        selected = out_idx[row].tolist()
        assert len(set(selected)) == len(selected), f"row {row} has duplicate indices"


@pytest.mark.parametrize("device", DEVICES)
def test_indexer_prefill_call_shape(device: str) -> None:
    """Mirror ``index_attn._topk_prefill``: topk=512, end=klen, offset=ks."""
    torch.manual_seed(9)
    rows, vocab, klen = 8, 16384, 16385 - 1
    x = torch.randn(rows, vocab, device=device, dtype=torch.bfloat16)
    end = torch.full((rows,), klen, dtype=torch.int32, device=device)
    offset = torch.full((rows,), 100, dtype=torch.int32, device=device)
    _, out_idx = _call(
        x, 512, end=end, offset=offset, sorted_index=True, return_value=False
    )
    _assert_contract(x, out_idx, None, 512, end=end, offset=offset, sorted_index=True)


@pytest.mark.parametrize("device", DEVICES)
def test_op_matches_the_backend_function(device: str) -> None:
    torch.manual_seed(10)
    x = torch.randn(3, 2048, device=device, dtype=torch.float32)
    end = torch.tensor([2048, 100, 0], dtype=torch.int32, device=device)
    expected = torch.full((3, 256), TAIL_SENTINEL, dtype=torch.int32, device=device)
    backend.topk(
        x,
        256,
        None,
        end,
        False,
        False,
        None,
        expected,
        None,
        FILL,
        VAL_FILL,
        False,
        False,
    )
    _, got = _call(x, 256, end=end)
    _assert_contract(x, got, None, 256, end=end)
    assert torch.all(got[2] == FILL)
