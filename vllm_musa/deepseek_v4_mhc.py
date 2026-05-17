# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 MHC MUSA fallback and provider helpers."""

from collections.abc import Iterator
from contextlib import contextmanager
import os

import torch

_MHC_TORCH_IMPLS = {"0", "false", "fallback", "no", "off", "torch"}
_MHC_TILELANG_IMPLS = {"", "auto", "default", "jit", "tilelang"}
_MHC_STRICT_TILELANG_IMPLS = {"force", "strict", "tilelang_force"}
_MHC_PROVIDER_FALLBACK_EXCEPTIONS = (
    AssertionError,
    ImportError,
    NotImplementedError,
    RuntimeError,
    ValueError,
)


def _mhc_impl(env_name: str) -> str:
    return os.getenv(env_name, "tilelang").strip().lower()


def _mhc_try_tilelang(impl: str, tensor: torch.Tensor) -> bool:
    if tensor.device.type != "musa":
        return False
    if impl in _MHC_TORCH_IMPLS:
        return False
    if impl in _MHC_TILELANG_IMPLS or impl in _MHC_STRICT_TILELANG_IMPLS:
        return True
    raise ValueError(
        f"Unsupported MHC provider impl {impl!r}; use tilelang, force, or torch"
    )


def _mhc_strict_tilelang(impl: str) -> bool:
    return impl in _MHC_STRICT_TILELANG_IMPLS


@contextmanager
def _timed_or_noop(scope_name: str) -> Iterator[None]:
    try:
        from vllm_musa.deepseek_v4_timers import timed
    except Exception:
        yield
        return
    with timed(scope_name):
        yield


def mhc_pre_torch_fallback(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    impl = _mhc_impl("VLLM_MUSA_DEEPSEEK_V4_MHC_PRE_IMPL")
    if _mhc_try_tilelang(impl, residual):
        try:
            with _timed_or_noop("mhc.pre_tilelang_full_provider"):
                return _mhc_pre_tilelang_full_provider(
                    residual,
                    fn,
                    hc_scale,
                    hc_base,
                    rms_eps,
                    hc_pre_eps,
                    hc_sinkhorn_eps,
                    hc_post_mult_value,
                    sinkhorn_repeat,
                )
        except _MHC_PROVIDER_FALLBACK_EXCEPTIONS:
            if _mhc_strict_tilelang(impl):
                raise
        try:
            with _timed_or_noop("mhc.pre_tilelang_mix_provider"):
                return _mhc_pre_tilelang_mix_provider(
                    residual,
                    fn,
                    hc_scale,
                    hc_base,
                    rms_eps,
                    hc_pre_eps,
                    hc_sinkhorn_eps,
                    hc_post_mult_value,
                    sinkhorn_repeat,
                )
        except _MHC_PROVIDER_FALLBACK_EXCEPTIONS:
            if _mhc_strict_tilelang(impl):
                raise

    with _timed_or_noop("mhc.pre_torch_fallback"):
        assert residual.dtype == torch.bfloat16
        assert fn.dtype == torch.float32
        assert hc_scale.dtype == torch.float32
        assert hc_base.dtype == torch.float32

        hc_mult = residual.shape[-2]
        hidden_size = residual.shape[-1]
        hc_mult2 = hc_mult * hc_mult
        hc_mult3 = hc_mult * 2 + hc_mult2
        hc_hidden_size = hc_mult * hidden_size
        assert fn.shape == (hc_mult3, hc_hidden_size)
        assert hc_scale.shape == (3,)
        assert hc_base.shape == (hc_mult3,)

        outer_shape = residual.shape[:-2]
        residual_flat = residual.reshape(-1, hc_mult, hidden_size)
        num_tokens = residual_flat.shape[0]
        x_float = residual_flat.reshape(num_tokens, hc_hidden_size).to(torch.float32)

        mixes = x_float @ fn.t()
        rms = torch.rsqrt(
            x_float.square().sum(dim=-1) / float(hc_hidden_size) + rms_eps
        )
        mixes = mixes * rms.unsqueeze(-1)

        pre_mix = (
            torch.sigmoid(mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult])
            + hc_pre_eps
        )
        post_mix = (
            torch.sigmoid(
                mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1]
                + hc_base[hc_mult : 2 * hc_mult]
            )
            * hc_post_mult_value
        )
        comb_mix = (
            mixes[:, 2 * hc_mult :].reshape(num_tokens, hc_mult, hc_mult)
            * hc_scale[2]
            + hc_base[2 * hc_mult :].reshape(hc_mult, hc_mult)
        )
        comb_mix = torch.softmax(comb_mix, dim=-1) + hc_sinkhorn_eps
        comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
        for _ in range(sinkhorn_repeat - 1):
            comb_mix = comb_mix / (
                comb_mix.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps
            )
            comb_mix = comb_mix / (
                comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps
            )

        layer_input = (
            pre_mix.unsqueeze(-1) * residual_flat.to(torch.float32)
        ).sum(dim=1).to(torch.bfloat16)

        return (
            post_mix.reshape(*outer_shape, hc_mult, 1),
            comb_mix.reshape(*outer_shape, hc_mult, hc_mult),
            layer_input.reshape(*outer_shape, hidden_size),
        )


def mhc_post_torch_fallback(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    impl = _mhc_impl("VLLM_MUSA_DEEPSEEK_V4_MHC_POST_IMPL")
    if _mhc_try_tilelang(impl, residual):
        try:
            with _timed_or_noop("mhc.post_tilelang_provider"):
                return _mhc_post_tilelang_provider(
                    x, residual, post_layer_mix, comb_res_mix
                )
        except _MHC_PROVIDER_FALLBACK_EXCEPTIONS:
            if _mhc_strict_tilelang(impl):
                raise

    with _timed_or_noop("mhc.post_torch_fallback"):
        outer_shape = residual.shape[:-2]
        hc = residual.shape[-2]
        hidden = residual.shape[-1]
        residual_flat = residual.reshape(-1, hc, hidden).to(torch.float32)
        x_flat = x.reshape(-1, hidden).to(torch.float32)
        post_flat = post_layer_mix.reshape(-1, hc, 1).to(torch.float32)
        comb_flat = comb_res_mix.reshape(-1, hc, hc).to(torch.float32)

        out = torch.einsum("tij,tih->tjh", comb_flat, residual_flat)
        out = out + post_flat * x_flat.unsqueeze(1)
        return out.to(residual.dtype).reshape(*outer_shape, hc, hidden)


def _require_contiguous(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_contiguous():
        raise NotImplementedError(
            f"MHC TileLang provider requires contiguous {name}"
        )


def _mhc_pre_tilelang_mix_provider(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    _require_contiguous("residual", residual)
    _require_contiguous("fn", fn)
    _require_contiguous("hc_scale", hc_scale)
    _require_contiguous("hc_base", hc_base)

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    if hc_mult != 4:
        raise NotImplementedError(
            f"MHC pre TileLang provider only supports hc_mult=4, got {hc_mult}"
        )
    mhc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    hc_hidden_size = hc_mult * hidden_size
    if fn.shape != (mhc_mult3, hc_hidden_size):
        raise ValueError(
            "MHC pre TileLang provider fn shape mismatch: "
            f"expected {(mhc_mult3, hc_hidden_size)}, got {tuple(fn.shape)}"
        )
    if hc_scale.shape != (3,):
        raise ValueError(
            f"MHC pre TileLang provider hc_scale shape mismatch: {hc_scale.shape}"
        )
    if hc_base.shape != (mhc_mult3,):
        raise ValueError(
            f"MHC pre TileLang provider hc_base shape mismatch: {hc_base.shape}"
        )

    outer_shape = residual.shape[:-2]
    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    x_float = residual_flat.reshape(num_tokens, hc_hidden_size).to(torch.float32)
    mixes_raw = x_float @ fn.t()
    sqsum = x_float.square().sum(dim=-1)

    post_mix = torch.empty(
        num_tokens, hc_mult, 1, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, hc_mult, hc_mult, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )

    from vllm_musa.deepseek_v4_jit.tilelang_kernels import mhc_pre_mix_kernel

    mhc_pre_mix_kernel(hidden_size, sinkhorn_repeat)(
        mixes_raw,
        sqsum,
        residual_flat,
        hc_scale,
        hc_base,
        post_mix,
        comb_mix,
        layer_input,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
    )

    return (
        post_mix.view(*outer_shape, hc_mult, 1),
        comb_mix.view(*outer_shape, hc_mult, hc_mult),
        layer_input.view(*outer_shape, hidden_size),
    )


def _mhc_pre_tilelang_full_provider(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    _require_contiguous("residual", residual)
    _require_contiguous("fn", fn)
    _require_contiguous("hc_scale", hc_scale)
    _require_contiguous("hc_base", hc_base)

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    if hc_mult != 4:
        raise NotImplementedError(
            f"MHC pre full TileLang provider only supports hc_mult=4, got {hc_mult}"
        )
    mhc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    hc_hidden_size = hc_mult * hidden_size
    if fn.shape != (mhc_mult3, hc_hidden_size):
        raise ValueError(
            "MHC pre full TileLang provider fn shape mismatch: "
            f"expected {(mhc_mult3, hc_hidden_size)}, got {tuple(fn.shape)}"
        )
    if hc_scale.shape != (3,):
        raise ValueError(
            f"MHC pre full TileLang provider hc_scale shape mismatch: "
            f"{hc_scale.shape}"
        )
    if hc_base.shape != (mhc_mult3,):
        raise ValueError(
            f"MHC pre full TileLang provider hc_base shape mismatch: "
            f"{hc_base.shape}"
        )

    outer_shape = residual.shape[:-2]
    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]

    post_mix = torch.empty(
        num_tokens, hc_mult, 1, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, hc_mult, hc_mult, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )

    from vllm_musa.deepseek_v4_jit.tilelang_kernels import mhc_pre_full_kernel

    mhc_pre_full_kernel(hidden_size, sinkhorn_repeat)(
        residual_flat,
        fn,
        hc_scale,
        hc_base,
        post_mix,
        comb_mix,
        layer_input,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
    )

    return (
        post_mix.view(*outer_shape, hc_mult, 1),
        comb_mix.view(*outer_shape, hc_mult, hc_mult),
        layer_input.view(*outer_shape, hidden_size),
    )


def _mhc_post_tilelang_provider(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    assert x.dtype == torch.bfloat16
    assert residual.dtype == torch.bfloat16
    assert post_layer_mix.dtype == torch.float32
    assert comb_res_mix.dtype == torch.float32

    _require_contiguous("x", x)
    _require_contiguous("residual", residual)
    _require_contiguous("post_layer_mix", post_layer_mix)
    _require_contiguous("comb_res_mix", comb_res_mix)

    outer_shape = residual.shape[:-2]
    hc = residual.shape[-2]
    hidden = residual.shape[-1]
    if hc != 4:
        raise NotImplementedError(
            f"MHC post TileLang provider only supports hc_mult=4, got {hc}"
        )

    x_flat = x.view(-1, hidden)
    residual_flat = residual.view(-1, hc, hidden)
    post_flat = post_layer_mix.view(-1, hc, 1).squeeze(-1)
    comb_flat = comb_res_mix.view(-1, hc, hc)
    if x_flat.shape[0] != residual_flat.shape[0]:
        raise ValueError(
            "MHC post TileLang provider token mismatch: "
            f"x={x_flat.shape}, residual={residual_flat.shape}"
        )

    from vllm_musa.deepseek_v4_jit.tilelang_kernels import mhc_post_kernel

    out = torch.empty_like(residual_flat)
    mhc_post_kernel(hidden)(x_flat, residual_flat, post_flat, comb_flat, out)
    return out.view(*outer_shape, hc, hidden)
