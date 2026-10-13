# SPDX-License-Identifier: Apache-2.0
"""FP32 split-K mHC prenorm partials for decode token counts.

The mHC prenorm projects each token's flattened hyper-connection residual onto
the 24 mix logits and also needs the residual's squared sum. At decode token
counts the TF32 tensor-core GEMM is latency bound; one program per (K slice,
token) accumulates the products with FMAs and reduces them once, producing the
same split-K partial layout in full FP32.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

# hc_hidden_size -> ((max_tokens, (split_k, block_k, num_warps)), ...), measured
# on S5000. Past the last bucket the tensor-core GEMM is as fast.
_MEASURED_CONFIGS: dict[int, tuple[tuple[int, tuple[int, int, int]], ...]] = {
    16384: ((8, (32, 128, 8)), (12, (32, 64, 4))),
    20480: ((2, (40, 128, 8)), (5, (40, 64, 8)), (8, (40, 64, 4)), (12, (40, 128, 4))),
}
_MAX_OUTPUTS = 32


@triton.jit
def _mhc_prenorm_fma_kernel(
    x_ptr,
    fn_ptr,
    out_mul_ptr,
    out_sqrsum_ptr,
    num_tokens,
    stride_x,
    stride_fn,
    SPLIT_LEN: tl.constexpr,
    N: tl.constexpr,
    N_PAD: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    split = tl.program_id(0)
    token = tl.program_id(1)
    offs_n = tl.arange(0, N_PAD)
    n_mask = offs_n < N
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((N_PAD, BLOCK_K), tl.float32)
    sqr = tl.zeros((BLOCK_K,), tl.float32)
    for k0 in tl.static_range(0, SPLIT_LEN, BLOCK_K):
        k = split * SPLIT_LEN + k0 + offs_k
        x = tl.load(x_ptr + token * stride_x + k).to(tl.float32)
        f = tl.load(
            fn_ptr + offs_n[:, None] * stride_fn + k[None, :],
            mask=n_mask[:, None],
            other=0.0,
        )
        acc += f * x[None, :]
        sqr += x * x
    row = split * num_tokens + token
    tl.store(out_mul_ptr + row * N + offs_n, tl.sum(acc, axis=1), mask=n_mask)
    tl.store(out_sqrsum_ptr + row, tl.sum(sqr, axis=0))


def mhc_prenorm_fma_config(
    num_tokens: int, hc_hidden_size: int, num_outputs: int
) -> tuple[int, int, int] | None:
    """Measured (split_k, block_k, num_warps), or None when the GEMM should run."""
    if num_tokens <= 0 or num_outputs > _MAX_OUTPUTS:
        return None
    for max_tokens, config in _MEASURED_CONFIGS.get(hc_hidden_size, ()):
        if num_tokens <= max_tokens:
            return config
    return None


def mhc_prenorm_fma(
    x: torch.Tensor,
    fn: torch.Tensor,
    config: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split-K partials of ``x @ fn.T`` and of ``(x * x).sum(-1)``.

    ``x`` is the [tokens, K] bfloat16 residual and ``fn`` the [N, K] FP32
    projection; returns [split_k, tokens, N] and [split_k, tokens] FP32.
    """
    split_k, block_k, num_warps = config
    num_tokens, hc_hidden_size = x.shape
    num_outputs = fn.shape[0]
    split_len = hc_hidden_size // split_k
    assert split_len * split_k == hc_hidden_size and split_len % block_k == 0
    assert x.stride(1) == 1 and fn.stride(1) == 1 and fn.dtype == torch.float32
    out_mul = torch.empty(
        (split_k, num_tokens, num_outputs), dtype=torch.float32, device=x.device
    )
    out_sqrsum = torch.empty(
        (split_k, num_tokens), dtype=torch.float32, device=x.device
    )
    _mhc_prenorm_fma_kernel[(split_k, num_tokens)](
        x,
        fn,
        out_mul,
        out_sqrsum,
        num_tokens,
        x.stride(0),
        fn.stride(0),
        SPLIT_LEN=split_len,
        N=num_outputs,
        N_PAD=triton.next_power_of_2(num_outputs),
        BLOCK_K=block_k,
        num_warps=num_warps,
    )
    return out_mul, out_sqrsum
