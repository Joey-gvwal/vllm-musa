"""Adapter for the separately built DeepSelect MUSA extension."""

from __future__ import annotations

import functools
import importlib
import logging
from types import ModuleType

import torch


@functools.lru_cache(maxsize=1)
def native_backend() -> ModuleType:
    try:
        extension = importlib.import_module("deep_select.deep_select_musa_cpp")
        backend = importlib.import_module("deep_select.deep_select_musa")
    except ImportError as error:
        raise RuntimeError(
            "DeepSelect MUSA requires its AOT extension. Build the revision in "
            "third_party/PINS with DEEP_SELECT_BUILD_TARGET_PLATFORM=MUSA; "
            "see docs/deepselect.md. No torch.topk fallback is used on MUSA."
        ) from error
    logging.getLogger(__name__).info(
        "DeepSelect MUSA native backend: %s", extension.__file__
    )
    return backend


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
    if input.shape[0] == 0:
        return
    native_backend().topk(
        input,
        topk,
        begin,
        end,
        sorted_value,
        sorted_index,
        output_val[: input.shape[0], :topk] if output_val is not None else None,
        output_idx[: input.shape[0], :topk],
        output_idx_offset,
        idx_oob_fill_value,
        value_oob_fill_value,
        return_value,
        abort_when_nan_found,
    )
