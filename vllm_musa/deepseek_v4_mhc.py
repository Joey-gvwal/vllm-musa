# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 MHC MUSA fallback helpers.

These helpers keep the current torch correctness fallback behavior centralized
and measurable. They are not production MHC kernels.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import torch


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
