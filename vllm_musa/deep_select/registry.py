# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Register DeepSelect's ``torch.ops.deep_select`` surface on MUSA.

vLLM resolves DeepSelect through two ops only::

    torch.ops.deep_select.get_alignment_requirement() -> (input_b, output_b)
    torch.ops.deep_select.topk(input, k, begin, end, sorted_value, sorted_index,
                               output_val, output_idx, output_idx_offset,
                               idx_oob_fill_value, value_oob_fill_value,
                               return_value, abort_when_nan_found)

Upstream those come from the CUDA extension ``vllm._deepselect_C``, which cannot be
built or run on MUSA (TMA kernels). Registering the same names here means
``vllm/model_executor/layers/indexer_topk.py`` needs only its availability probe
relaxed (see the series patch), not a MUSA-specific code path.

The registration is deliberately lazy and idempotent: ``ensure_registered()`` is
called from ``vllm_musa.register_custom_ops()``, i.e. after the platform is
initialised, and again by tests.

Output mutation is declared in the schema so compilation cannot discard the op.
"""

from __future__ import annotations

import torch
from torch.library import Library

from vllm_musa.deep_select.backend import get_alignment_requirement, topk
from vllm_musa.deep_select.native import topk as native_topk

_LIB_NAMESPACE = "deep_select"

# Argument names follow the documented DeepSelect interface. ``sorted_value`` is
# the upstream ``sorted`` flag, renamed because ``sorted`` is a Python builtin.
_TOPK_SCHEMA = (
    "topk(Tensor input, int topk, Tensor? begin, Tensor? end, "
    "bool sorted_value, bool sorted_index, Tensor(a!)? output_val, "
    "Tensor(b!) output_idx, Tensor? output_idx_offset, int idx_oob_fill_value, "
    "float value_oob_fill_value, bool return_value, bool abort_when_nan_found) -> ()"
)
_ALIGNMENT_SCHEMA = "get_alignment_requirement() -> int[]"

_registered = False
# Keeps the registrations alive: a collected ``Library`` unregisters its ops.
_library: Library | None = None


def _topk_meta(*args, **kwargs) -> None:
    """Meta kernel: the op only writes into preallocated outputs."""
    return None


def _alignment_meta() -> list[int]:
    return get_alignment_requirement()


def _platform_dispatch_key() -> str | None:
    try:
        from vllm.platforms import current_platform

        return current_platform.dispatch_key
    except Exception:  # pragma: no cover - only used before vLLM initializes
        return None


def is_registered() -> bool:
    """Whether the DeepSelect namespace exposes the ops vLLM calls."""
    try:
        return hasattr(torch.ops.deep_select, "topk") and hasattr(
            torch.ops.deep_select, "get_alignment_requirement"
        )
    except AttributeError:
        return False


def ensure_registered() -> None:
    """Register the DeepSelect ops once; safe to call repeatedly."""
    global _registered, _library
    if _registered and is_registered():
        return

    lib = Library(_LIB_NAMESPACE, "FRAGMENT")
    # The Library object owns the registrations: dropping the last reference
    # deregisters the ops, which made the previous version register and vanish
    # within one call.
    _library = lib
    try:
        lib.define("deep_select::" + _TOPK_SCHEMA)
        lib.define("deep_select::" + _ALIGNMENT_SCHEMA)
    except RuntimeError:
        # Already defined by an earlier import (or by a vendored build of the
        # extension); the impls below still need to exist.
        pass

    lib.impl("deep_select::topk", topk, "CPU")
    if (
        getattr(torch.version, "musa", None)
        and _platform_dispatch_key() == "MUSA"
    ):
        lib.impl("deep_select::topk", native_topk, "MUSA")
    lib.impl(
        "deep_select::get_alignment_requirement",
        get_alignment_requirement,
        "CompositeExplicitAutograd",
    )
    lib.impl("deep_select::topk", _topk_meta, "Meta")
    lib.impl("deep_select::get_alignment_requirement", _alignment_meta, "Meta")
    _registered = True
