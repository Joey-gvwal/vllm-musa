# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA reference backend for DeepSelect's top-k contract.

Vendored (with adaptations) from ``yeahdongcn/DeepSelect``, branch ``xd/musa-port``,
commit ``ef535f4c60bd304c91955e6fc1ac7c7b85e4e518`` (MIT, Copyright (c) 2025
DeepSeek), file ``deep_select/deep_select_musa.py``.

Why this backend exists: DeepSelect's shipped kernels are TMA-based (``v3`` /
``v3_cluster`` / ``v3_fp32`` instantiations), and TMA has no direct MUSA
counterpart -- the equivalent there is TME. So neither the CUDA instantiations nor
``vllm._deepselect_C`` can be reused, yet the whole platform surface is only two
entry points (``get_alignment_requirement`` and ``topk``), exactly like the Ascend
backend in the same upstream tree.

**This is a correctness path, not a performance path.** It is implemented with
``torch.topk`` and reproduces the documented contract, including the per-entry
padding behaviour, the out-of-range fill values and the ``topk`` larger than the
row length case. The CUDA/Ascend backends are 2-20x faster; treat any timing here
as a floor a native MUSA (TME) kernel must beat, not as a result.

Two MUSA defects are worked around here deliberately (both measured on S5000,
torch 2.11.0.post2+musa5.2.0, see the upstream commit log):

* ``torch_musa``'s bf16 ``topk`` returns *duplicate* indices on tied values -- 967
  duplicate slots in a 2132-wide row at (18, 262144), against 0 for the same call on
  CPU. The duplicate rows are re-derived from an ``argsort`` of the whole row, which
  is a permutation by construction.
* the NaN marker must be written when ``abort_when_nan_found=False``, because the
  check itself is unconditional. With ``abort_when_nan_found=True`` the native
  MUSA backend aborts the operation with a device assertion; this reference
  backend raises a host-side error instead.
"""

from __future__ import annotations

import torch

# Stride requirements in bytes for (input, output). The compiled backends require
# this alignment so their vectorised loads are valid. ``torch.topk`` has no such
# constraint, but callers divide this by the element size to round the output
# stride, so it has to be a real byte count -- anything smaller yields a zero row
# stride for dtypes wider than one byte (and trips the caller's assertion).
_ALIGNMENT = (32, 32)


def get_alignment_requirement() -> list[int]:
    """Stride requirements in bytes for the (input, output) pair."""
    return list(_ALIGNMENT)


def topk(
    input: torch.Tensor,
    topk: int,
    begin: torch.Tensor | None,
    end: torch.Tensor | None,
    sorted_value: bool,
    sorted_index: bool,
    output_val: torch.Tensor | None,
    output_idx: torch.Tensor,
    output_idx_offset: torch.Tensor | None,
    idx_oob_fill_value: int,
    value_oob_fill_value: float,
    return_value: bool,
    abort_when_nan_found: bool,
) -> None:
    """Select the largest ``topk`` entries of each row, in place.

    Mirrors the documented backend contract: results are written into
    ``output_val`` and ``output_idx`` and nothing is returned. Row ranges are
    ``[begin[i], end[i])``; ``begin`` is rejected by the caller, so only ``end``
    can narrow a row. Rows shorter than ``topk`` are padded with
    ``idx_oob_fill_value`` / ``value_oob_fill_value``, and the padding never
    receives ``output_idx_offset``.

    ``sorted_value`` is accepted for signature compatibility; only fp32 supports
    sorted values upstream and the MUSA callers pass ``False``.
    """
    assert begin is None, (
        "`begin` is not supported by this backend, matching the others"
    )
    assert not sorted_value, "sorted values are not supported by this backend"

    device = input.device
    n_rows, vocab = input.shape

    if end is None:
        lengths = torch.full((n_rows,), vocab, device=device, dtype=torch.long)
    else:
        lengths = end.to(torch.long).clamp(min=0, max=vocab)

    # NaN policy, as documented in ``interface.topk``: "The NaN check itself is
    # always enabled"; ``abort_when_nan_found`` chooses between aborting the
    # operation and writing the ``0x3F3F3F3F`` marker.
    bad_rows = None
    if input.numel() > 0:
        in_range = torch.arange(vocab, device=device)[None, :] < lengths[:, None]
        bad_rows = torch.isnan(input).logical_and(in_range).any(dim=-1)
        if bool(bad_rows.any()):
            if abort_when_nan_found:
                raise RuntimeError("DeepSelect MUSA: NaN found in valid input range")
            output_idx[bad_rows, 0] = 0x3F3F3F3F

    k = int(topk)
    width = max(k, vocab)

    work = input
    if width != vocab:
        pad = torch.full(
            (n_rows, width - vocab), float("-inf"), device=device, dtype=input.dtype
        )
        work = torch.cat((work, pad), dim=-1)

    # Mask the tail of each row so a narrowed row cannot select beyond ``end``.
    if end is not None:
        valid = torch.arange(width, device=device)[None, :] < lengths[:, None]
        work = work.masked_fill(~valid, float("-inf"))

    values, indices = torch.topk(work, k, dim=-1, largest=True, sorted=True)

    # torch_musa's bf16 ``topk`` duplicates indices on ties (see module docstring).
    # Detect it with the sorted pass the ``sorted_index`` path needs anyway and
    # re-derive only the affected rows from an argsort of the whole row.
    if k > 1:
        ordered = torch.sort(indices, dim=-1).values
        dup_rows = (ordered[:, 1:] == ordered[:, :-1]).any(dim=-1)
        if bool(dup_rows.any()):
            bad = torch.nonzero(dup_rows, as_tuple=False).flatten()
            fixed = torch.argsort(
                work[bad], dim=-1, descending=True, stable=True
            )[:, :k]
            indices = indices.clone()
            indices[bad] = fixed
            if return_value and output_val is not None:
                values = values.clone()
                values[bad] = torch.gather(work[bad], 1, fixed)

    # Padding is identified per entry, not per row: a row of length 5 with topk 8
    # keeps its 5 real indices and fills the other 3, and a topk larger than the
    # vocabulary keeps the real head instead of blanking the row.
    padded = indices >= lengths[:, None]
    has_padding = bool(padded.any())
    if has_padding:
        fill_idx = torch.full_like(indices, int(idx_oob_fill_value))
        indices = torch.where(padded, fill_idx, indices)
        if return_value and output_val is not None:
            fill_val = torch.full_like(values, value_oob_fill_value)
            values = torch.where(padded, fill_val, values)

    # "Real entry" is the complement of ``padded``, not ``indices < vocab``: the
    # latter is the row *capacity*, and the fill value can be smaller than it
    # (vLLM passes ``-1``), which would turn every padded entry into a valid-looking
    # key index after the offset.
    in_place = ~padded
    if output_idx_offset is not None:
        offset = output_idx_offset.to(torch.long)[:, None]
        shifted = indices + offset
        indices = torch.where(in_place, shifted, indices)

    # Output order is part of the contract: valid entries first, padding last, and
    # ``sorted_index`` sorts that real block ascending. A plain ascending sort would
    # put padding first for a negative fill value, so the real block is pinned first
    # by a stable partition, and ``values`` follows the same permutation so every
    # value stays paired with its index.
    perm = None
    if sorted_index and k > 1:
        perm = torch.argsort(indices.to(torch.int64), dim=-1, stable=True)
    if has_padding:
        padded_sorted = (
            torch.gather(padded.to(torch.int64), -1, perm)
            if perm is not None
            else padded.to(torch.int64)
        )
        partition = torch.argsort(padded_sorted, dim=-1, stable=True)
        perm = partition if perm is None else torch.gather(perm, -1, partition)
    if perm is not None:
        indices = torch.gather(indices, -1, perm)
        if return_value and output_val is not None:
            values = torch.gather(values, -1, perm)

    # ``k`` is the number of entries to select and may be smaller than
    # ``output_idx.shape[1]``, which is preallocated at the candidate capacity; the
    # columns past the selection keep the caller's pre-filled value. Write the first
    # ``k`` columns and leave the tail alone.
    output_idx[:, :k] = indices.to(output_idx.dtype)
    if return_value and output_val is not None:
        output_val[:, :k] = values.to(output_val.dtype)
    if bad_rows is not None and bool(bad_rows.any()):
        # The copy above overwrites the marker; the check is unconditional, so
        # restore it for the rows that carried a NaN. Only column 0 carries it,
        # matching CUDA.
        output_idx[bad_rows, 0] = 0x3F3F3F3F
