# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 MHC helpers for MUSA runtime and diagnostic paths."""

from __future__ import annotations

import os

import torch


def mhc_pre_musa(
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
    impl = os.getenv("VLLM_MUSA_DEEPSEEK_V4_MHC_PRE_IMPL", "auto").lower()
    if impl == "auto":
        impl = _select_mhc_pre_auto_impl(residual)
    if impl in {"native", "musa", "mu"}:
        return _mhc_pre_native_provider(
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
    if impl in {"torch", "fallback"}:
        return mhc_pre_torch_fallback(
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
    if impl in {"tilelang", "jit"}:
        return _mhc_pre_tilelang_provider(
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
    raise ValueError(f"unsupported DeepSeek-V4 MHC pre impl: {impl!r}")


def _select_mhc_pre_auto_impl(residual: torch.Tensor) -> str:
    max_tilelang_tokens = int(
        os.getenv("VLLM_MUSA_DEEPSEEK_V4_MHC_PRE_TILELANG_MAX_TOKENS", "16")
    )
    if max_tilelang_tokens <= 0:
        return "native"
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    num_tokens = residual.numel() // (hc_mult * hidden_size)
    if (
        hc_mult == 4
        and hidden_size in {4096, 7168}
        and num_tokens <= max_tilelang_tokens
    ):
        return "tilelang"
    return "native"


def mhc_pre_musa_fallback(
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
    return mhc_pre_musa(
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
    rms = torch.rsqrt(x_float.square().sum(dim=-1) / float(hc_hidden_size) + rms_eps)
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
        mixes[:, 2 * hc_mult :].reshape(num_tokens, hc_mult, hc_mult) * hc_scale[2]
        + hc_base[2 * hc_mult :].reshape(hc_mult, hc_mult)
    )
    comb_mix = torch.softmax(comb_mix, dim=-1) + hc_sinkhorn_eps
    comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)

    layer_input = (pre_mix.unsqueeze(-1) * residual_flat.to(torch.float32)).sum(
        dim=1
    ).to(torch.bfloat16)

    return (
        post_mix.reshape(*outer_shape, hc_mult, 1),
        comb_mix.reshape(*outer_shape, hc_mult, hc_mult),
        layer_input.reshape(*outer_shape, hidden_size),
    )


def _mhc_pre_native_provider(
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
            f"MHC pre native provider only supports hc_mult=4, got {hc_mult}"
        )
    hc_mult3 = hc_mult * (2 + hc_mult)
    hc_hidden_size = hc_mult * hidden_size
    if fn.shape != (hc_mult3, hc_hidden_size):
        raise ValueError(
            "MHC pre native provider fn mismatch: "
            f"fn={fn.shape}, expected={(hc_mult3, hc_hidden_size)}"
        )
    if hc_scale.shape != (3,) or hc_base.shape != (hc_mult3,):
        raise ValueError(
            "MHC pre native provider scale/base mismatch: "
            f"hc_scale={hc_scale.shape}, hc_base={hc_base.shape}"
        )

    outer_shape = residual.shape[:-2]
    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    post_mix = torch.empty(
        (num_tokens, hc_mult), dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        (num_tokens, hc_mult, hc_mult),
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input = torch.empty(
        (num_tokens, hidden_size), dtype=torch.bfloat16, device=residual.device
    )

    from vllm_musa import _custom_ops as _musa_custom_ops

    _musa_custom_ops.deepseek_v4_mhc_pre(
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
        sinkhorn_repeat,
    )
    return (
        post_mix.view(*outer_shape, hc_mult, 1),
        comb_mix.view(*outer_shape, hc_mult, hc_mult),
        layer_input.view(*outer_shape, hidden_size),
    )


def _mhc_pre_tilelang_provider(
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

    hc_mult3 = hc_mult * (2 + hc_mult)
    hc_hidden_size = hc_mult * hidden_size
    if fn.shape != (hc_mult3, hc_hidden_size):
        raise ValueError(
            "MHC pre TileLang provider fn mismatch: "
            f"fn={fn.shape}, expected={(hc_mult3, hc_hidden_size)}"
        )
    if hidden_size % 256 != 0:
        raise NotImplementedError(
            "MHC pre TileLang provider requires hidden_size divisible by 256, "
            f"got {hidden_size}"
        )

    from vllm_musa.deepseek_v4_jit.tilelang_kernels import (
        mhc_pre_split_sinkhorn_kernel,
    )

    outer_shape = residual.shape[:-2]
    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    x_float = residual_flat.view(num_tokens, hc_hidden_size).to(torch.float32)
    mixes = x_float @ fn.t()
    rms = torch.rsqrt(
        x_float.square().sum(dim=-1) / float(hc_hidden_size) + rms_eps
    )
    mixes = (mixes * rms.unsqueeze(-1)).contiguous()

    post_mix = torch.empty(
        (num_tokens, hc_mult), dtype=torch.float32, device=residual.device
    )
    pre_mix = torch.empty(
        (num_tokens, hc_mult), dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        (num_tokens, hc_mult * hc_mult),
        dtype=torch.float32,
        device=residual.device,
    )

    mhc_pre_split_sinkhorn_kernel(hc_mult, sinkhorn_repeat)(
        mixes,
        hc_scale,
        hc_base,
        pre_mix,
        post_mix,
        comb_mix,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
    )
    layer_input = (
        pre_mix.unsqueeze(-1) * residual_flat.to(torch.float32)
    ).sum(dim=1).to(torch.bfloat16)

    return (
        post_mix.view(*outer_shape, hc_mult, 1),
        comb_mix.view(*outer_shape, hc_mult, hc_mult),
        layer_input.view(*outer_shape, hidden_size),
    )


def mhc_post_musa(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    impl = os.getenv("VLLM_MUSA_DEEPSEEK_V4_MHC_POST_IMPL", "tilelang").lower()
    if impl in {"tilelang", "jit", "native", "musa", "mu"}:
        return _mhc_post_tilelang_provider(x, residual, post_layer_mix, comb_res_mix)
    if impl in {"torch", "fallback"}:
        return mhc_post_torch_fallback(x, residual, post_layer_mix, comb_res_mix)
    raise ValueError(f"unsupported DeepSeek-V4 MHC post impl: {impl!r}")


def mhc_post_musa_fallback(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    return mhc_post_musa(x, residual, post_layer_mix, comb_res_mix)


def mhc_post_torch_fallback(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
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
