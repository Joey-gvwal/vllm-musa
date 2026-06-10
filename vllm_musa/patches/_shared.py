# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the MUSA cat-6 object patches.

These were inline in ``vllm_musa/__init__.py``; the cat-6 migration moves the
torch/vLLM config compat shims into individual ``*.patch.py`` object patches
(applied by :func:`vllm_musa.patches.apply_object_patches`) and the helpers they
share here. Not a ``*.patch.py`` so it is NOT discovered as an object patch;
imported by the patches as ``vllm_musa.patches._shared`` at runtime.
"""

from contextlib import nullcontext
from functools import wraps
from typing import Any


def filter_existing_config(config: dict, config_module: Any) -> dict:
    """Drop config keys that are absent in the installed Torch."""
    return {key: value for key, value in config.items() if hasattr(config_module, key)}


def make_config_patch_filter(original_patch: Any, config_module: Any) -> Any:
    """Wrap a ``config.patch`` callable so it silently drops keys/positional
    config names that the installed Torch ``config_module`` does not define."""

    @wraps(original_patch)
    def patch_existing_config(*args: Any, **kwargs: Any) -> Any:
        if args and isinstance(args[0], dict):
            config = filter_existing_config(args[0], config_module)
            if not config and not kwargs:
                return nullcontext()
            args = (config, *args[1:])
        elif args and isinstance(args[0], str):
            if not hasattr(config_module, args[0]):
                return nullcontext()

        if kwargs:
            kwargs = filter_existing_config(kwargs, config_module)
            if not args and not kwargs:
                return nullcontext()

        return original_patch(*args, **kwargs)

    patch_existing_config._musa_filters_missing_config_keys = True
    return patch_existing_config


def has_musa_rms_norm_kernel() -> bool:
    try:
        import torch

        if not hasattr(torch.ops, "_C") or not hasattr(torch.ops._C, "rms_norm"):
            return False
        return torch._C._dispatch_has_kernel_for_dispatch_key("_C::rms_norm", "MUSA")
    except Exception:
        return False


def has_musa_rotary_embedding_kernel() -> bool:
    try:
        import torch

        if not hasattr(torch.ops, "_C") or not hasattr(
            torch.ops._C, "rotary_embedding"
        ):
            return False
        return torch._C._dispatch_has_kernel_for_dispatch_key(
            "_C::rotary_embedding", "MUSA"
        )
    except Exception:
        return False


def musa_safe_rms_norm(out: Any, input: Any, weight: Any, epsilon: float) -> None:
    import torch
    import torch.nn as nn

    if getattr(input.device, "type", None) == "musa" and not has_musa_rms_norm_kernel():
        normalized_shape = (weight.shape[-1],)
        out.copy_(nn.functional.rms_norm(input, normalized_shape, weight, epsilon))
        return
    torch.ops._C.rms_norm(out, input, weight, epsilon)


def musa_safe_rotary_embedding(
    positions: Any,
    query: Any,
    key: Any,
    head_size: int,
    cos_sin_cache: Any,
    is_neox: bool,
    rope_dim_offset: int = 0,
    inverse: bool = False,
) -> None:
    import torch

    if (
        getattr(query.device, "type", None) == "musa"
        and rope_dim_offset == 0
        and not inverse
        and not has_musa_rotary_embedding_kernel()
    ):
        from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

        rotary_dim = cos_sin_cache.shape[-1]
        query_rot, key_rot = RotaryEmbedding.forward_static(
            positions, query, key, head_size, rotary_dim, cos_sin_cache, is_neox
        )
        query.copy_(query_rot)
        if key is not None:
            key.copy_(key_rot)
        return
    if rope_dim_offset == 0 and not inverse:
        torch.ops._C.rotary_embedding(
            positions, query, key, head_size, cos_sin_cache, is_neox
        )
    else:
        torch.ops._C.rotary_embedding(
            positions,
            query,
            key,
            head_size,
            cos_sin_cache,
            is_neox,
            rope_dim_offset,
            inverse,
        )
