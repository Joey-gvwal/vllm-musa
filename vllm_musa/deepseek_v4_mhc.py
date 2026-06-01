# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 MHC helpers for MUSA runtime and diagnostic paths."""

from __future__ import annotations

import os

import torch

_MHC_PRE_DEEPGEMM_SPLIT_K_ENV = "VLLM_MUSA_DEEPSEEK_V4_MHC_PRE_DEEPGEMM_SPLIT_K"
_MHC_PRE_BIG_FUSE_THREADS_ENV = "VLLM_MUSA_DEEPSEEK_V4_MHC_PRE_BIG_FUSE_THREADS"
_MHC_PRE_BIG_FUSE_HIDDEN_BLOCK_ENV = (
    "VLLM_MUSA_DEEPSEEK_V4_MHC_PRE_BIG_FUSE_HIDDEN_BLOCK"
)
_MHC_PRE_BIG_FUSE_PASS_CONFIG_ENV = "VLLM_MUSA_DEEPSEEK_V4_MHC_PRE_BIG_FUSE_PASS_CONFIG"
_MHC_PRE_DECODE_PRENORM_IMPL_ENV = "VLLM_MUSA_DEEPSEEK_V4_MHC_PRE_DECODE_PRENORM_IMPL"


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
    if impl in {"deepgemm_big_fuse", "deepgemm-big-fuse"}:
        return _mhc_pre_deepgemm_big_fuse_provider(
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
        torch.sigmoid(mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + hc_pre_eps
    )
    post_mix = (
        torch.sigmoid(
            mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1]
            + hc_base[hc_mult : 2 * hc_mult]
        )
        * hc_post_mult_value
    )
    comb_mix = mixes[:, 2 * hc_mult :].reshape(num_tokens, hc_mult, hc_mult) * hc_scale[
        2
    ] + hc_base[2 * hc_mult :].reshape(hc_mult, hc_mult)
    comb_mix = torch.softmax(comb_mix, dim=-1) + hc_sinkhorn_eps
    comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)

    layer_input = (
        (pre_mix.unsqueeze(-1) * residual_flat.to(torch.float32))
        .sum(dim=1)
        .to(torch.bfloat16)
    )

    return (
        post_mix.reshape(*outer_shape, hc_mult, 1),
        comb_mix.reshape(*outer_shape, hc_mult, hc_mult),
        layer_input.reshape(*outer_shape, hidden_size),
    )


def select_mhc_prenorm_split_k(num_tokens: int, hc_hidden_size: int) -> int:
    """Return the measured default split-K for DeepSeek-V4 MHC PreNorm."""
    if hc_hidden_size == 16384:
        if num_tokens <= 64:
            return 64
        if num_tokens <= 128:
            return 16
        if num_tokens <= 256:
            return 8
        if num_tokens <= 1024:
            return 32
        if num_tokens <= 2048:
            return 16
        return 4

    return 16 if num_tokens <= 1024 else 8


def _get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _get_mhc_pre_deepgemm_split_k(
    num_tokens: int,
    hc_hidden_size: int,
) -> int:
    split_k = _get_env_int(
        _MHC_PRE_DEEPGEMM_SPLIT_K_ENV,
        select_mhc_prenorm_split_k(num_tokens, hc_hidden_size),
    )
    if split_k <= 0:
        raise ValueError(f"{_MHC_PRE_DEEPGEMM_SPLIT_K_ENV} must be > 0, got {split_k}")
    if hc_hidden_size % split_k != 0:
        raise ValueError(
            "DeepGEMM MHC prenorm requires K divisible by split_k, "
            f"got K={hc_hidden_size}, split_k={split_k}"
        )
    return split_k


def _select_mhc_pre_big_fuse_prenorm_impl(
    num_tokens: int,
    hc_hidden_size: int,
) -> str:
    impl = os.getenv(_MHC_PRE_DECODE_PRENORM_IMPL_ENV, "deepgemm").strip().lower()
    if impl in {"", "0", "false", "off", "deepgemm"}:
        return "deepgemm"
    if impl in {"1", "true", "on", "auto", "tilelang"}:
        if hc_hidden_size == 16384 and num_tokens <= 64:
            return "tilelang"
        return "deepgemm"
    raise ValueError(
        f"{_MHC_PRE_DECODE_PRENORM_IMPL_ENV} must be one of "
        "'deepgemm', 'tilelang', 'auto', '0', or '1', got {impl!r}"
    )


def _mhc_prenorm_gemm_sqrsum_tilelang_decode_partials(
    residual_flat: torch.Tensor,
    fn: torch.Tensor,
    *,
    split_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_flat = residual_flat.view(residual_flat.shape[0], -1).bfloat16()
    num_tokens, hc_hidden_size = x_flat.shape
    mhc_mult3 = fn.shape[0]
    if split_k <= 0:
        raise ValueError(f"TileLang MHC prenorm split_k must be > 0, got {split_k}")
    if num_tokens > 64 or hc_hidden_size != 16384:
        raise NotImplementedError(
            "TileLang decode MHC prenorm partials only support "
            f"num_tokens <= 64 and K=16384, got num_tokens={num_tokens}, "
            f"K={hc_hidden_size}"
        )
    if hc_hidden_size % split_k != 0:
        raise ValueError(
            "TileLang MHC prenorm requires K divisible by split_k, "
            f"got K={hc_hidden_size}, split_k={split_k}"
        )
    split_size = hc_hidden_size // split_k
    if split_size % 128 != 0:
        raise ValueError(
            "TileLang decode MHC prenorm partials require split_size "
            f"divisible by 128, got split_size={split_size}"
        )

    d_part = torch.empty(
        split_k,
        num_tokens,
        mhc_mult3,
        dtype=torch.float32,
        device=residual_flat.device,
    )
    s_part = torch.empty(
        split_k,
        num_tokens,
        dtype=torch.float32,
        device=residual_flat.device,
    )

    from vllm_musa.deepseek_v4_jit.tilelang_kernels import (
        mhc_prenorm_splitk_x_tme_cast_kernel,
    )

    mhc_prenorm_splitk_x_tme_cast_kernel(
        mhc_mult3,
        hc_hidden_size,
        split_k=split_k,
        token_block=32,
        hidden_block=128,
        num_stages=2,
    )(
        x_flat,
        fn.float().contiguous(),
        d_part,
        s_part,
    )
    return d_part, s_part


def _resolve_mhc_pre_big_fuse_config(
    num_tokens: int,
    n_splits: int,
) -> tuple[int, int, str]:
    is_tiny_decode = num_tokens <= 32
    is_decode_like = num_tokens <= 64
    is_mid_prefill = 128 < num_tokens <= 512

    threads = _get_env_int(_MHC_PRE_BIG_FUSE_THREADS_ENV, 0)
    if threads <= 0:
        threads = 128 if is_tiny_decode else 256 if is_decode_like else 128
    if threads not in (128, 256):
        raise ValueError(
            f"{_MHC_PRE_BIG_FUSE_THREADS_ENV} must be 128 or 256, got {threads}"
        )

    hidden_block = _get_env_int(_MHC_PRE_BIG_FUSE_HIDDEN_BLOCK_ENV, 0)
    if hidden_block <= 0:
        hidden_block = 512 if is_tiny_decode or is_mid_prefill else 1024

    pass_config = os.getenv(_MHC_PRE_BIG_FUSE_PASS_CONFIG_ENV, "auto").strip().lower()
    if pass_config == "auto":
        pass_config = (
            "aggressive_index32"
            if (is_decode_like or is_mid_prefill) and n_splits != 1
            else "safe"
        )
    return threads, hidden_block, pass_config


def _mhc_prenorm_gemm_sqrsum_deepgemm(
    residual_flat: torch.Tensor,
    fn: torch.Tensor,
    *,
    split_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        from deep_gemm.interface import tf32_hc_prenorm_gemm
    except ImportError:
        from deep_gemm import tf32_hc_prenorm_gemm

    x_flat = residual_flat.view(residual_flat.shape[0], -1).bfloat16()
    num_tokens, hc_hidden_size = x_flat.shape
    mhc_mult3 = fn.shape[0]
    if split_k <= 0:
        raise ValueError(f"DeepGEMM MHC prenorm split_k must be > 0, got {split_k}")
    if hc_hidden_size % split_k != 0:
        raise ValueError(
            "DeepGEMM MHC prenorm requires K divisible by split_k, "
            f"got K={hc_hidden_size}, split_k={split_k}"
        )

    if split_k == 1:
        d_out = torch.empty(
            num_tokens,
            mhc_mult3,
            dtype=torch.float32,
            device=residual_flat.device,
        )
        s_out = torch.empty(
            num_tokens,
            dtype=torch.float32,
            device=residual_flat.device,
        )
        tf32_hc_prenorm_gemm(
            x_flat,
            fn.float().contiguous(),
            d_out,
            s_out,
            num_splits=1,
        )
        return d_out.unsqueeze(0), s_out.unsqueeze(0)

    d_part = torch.empty(
        split_k,
        num_tokens,
        mhc_mult3,
        dtype=torch.float32,
        device=residual_flat.device,
    )
    s_part = torch.empty(
        split_k,
        num_tokens,
        dtype=torch.float32,
        device=residual_flat.device,
    )
    tf32_hc_prenorm_gemm(
        x_flat,
        fn.float().contiguous(),
        d_part,
        s_part,
        num_splits=split_k,
    )
    return d_part, s_part


def _mhc_pre_deepgemm_big_fuse_provider(
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
            f"MHC pre DeepGEMM big-fuse provider only supports hc_mult=4, "
            f"got {hc_mult}"
        )

    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * (2 + hc_mult)
    hc_hidden_size = hc_mult * hidden_size
    if hc_mult3 > 32:
        raise NotImplementedError(
            "MHC pre DeepGEMM big-fuse provider requires hc_mult3 <= 32, "
            f"got {hc_mult3}"
        )
    if fn.shape != (hc_mult3, hc_hidden_size):
        raise ValueError(
            "MHC pre DeepGEMM big-fuse provider fn mismatch: "
            f"fn={fn.shape}, expected={(hc_mult3, hc_hidden_size)}"
        )
    if hc_scale.shape != (3,) or hc_base.shape != (hc_mult3,):
        raise ValueError(
            "MHC pre DeepGEMM big-fuse provider scale/base mismatch: "
            f"hc_scale={hc_scale.shape}, hc_base={hc_base.shape}"
        )

    outer_shape = residual.shape[:-2]
    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    split_k = _get_mhc_pre_deepgemm_split_k(num_tokens, hc_hidden_size)
    prenorm_impl = _select_mhc_pre_big_fuse_prenorm_impl(
        num_tokens,
        hc_hidden_size,
    )
    threads, hidden_block, pass_config = _resolve_mhc_pre_big_fuse_config(
        num_tokens,
        split_k,
    )
    if hidden_block <= 0 or hidden_size % hidden_block != 0:
        raise NotImplementedError(
            "MHC pre DeepGEMM big-fuse provider requires hidden_size divisible "
            f"by hidden_block, got hidden_size={hidden_size}, "
            f"hidden_block={hidden_block}"
        )

    post_mix = torch.empty(
        (num_tokens, hc_mult),
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix = torch.empty(
        (num_tokens, hc_mult2),
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input = torch.empty(
        (num_tokens, hidden_size),
        dtype=torch.bfloat16,
        device=residual.device,
    )
    if prenorm_impl == "tilelang":
        gemm_out_mul, gemm_out_sqrsum = (
            _mhc_prenorm_gemm_sqrsum_tilelang_decode_partials(
                residual_flat,
                fn,
                split_k=split_k,
            )
        )
    else:
        gemm_out_mul, gemm_out_sqrsum = _mhc_prenorm_gemm_sqrsum_deepgemm(
            residual_flat,
            fn,
            split_k=split_k,
        )

    _require_contiguous("gemm_out_mul", gemm_out_mul)
    _require_contiguous("gemm_out_sqrsum", gemm_out_sqrsum)

    from vllm_musa.deepseek_v4_jit.tilelang_kernels import (
        mhc_pre_big_fuse_decode_split_kernel,
        mhc_pre_big_fuse_kernel,
    )

    kernel_factory = (
        mhc_pre_big_fuse_decode_split_kernel
        if num_tokens <= 64
        else mhc_pre_big_fuse_kernel
    )
    kernel_factory(
        hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits=gemm_out_mul.shape[0],
        hc_mult=hc_mult,
        threads=threads,
        hidden_block=hidden_block,
        pass_config=pass_config,
    )(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual_flat,
        post_mix,
        comb_mix,
        layer_input,
    )

    return (
        post_mix.view(*outer_shape, hc_mult, 1),
        comb_mix.view(*outer_shape, hc_mult, hc_mult),
        layer_input.view(*outer_shape, hidden_size),
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
    rms = torch.rsqrt(x_float.square().sum(dim=-1) / float(hc_hidden_size) + rms_eps)
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
        (pre_mix.unsqueeze(-1) * residual_flat.to(torch.float32))
        .sum(dim=1)
        .to(torch.bfloat16)
    )

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
        raise NotImplementedError(f"MHC TileLang provider requires contiguous {name}")


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
