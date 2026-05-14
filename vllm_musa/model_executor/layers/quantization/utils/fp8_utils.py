# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import get_tma_aligned_size, is_deep_gemm_e8m0_used

logger = init_logger(__name__)


def _upcast_e8m0_to_fp32(scale: torch.Tensor) -> torch.Tensor:
    exp_bits = scale.view(torch.uint8).to(torch.int32)
    return (exp_bits << 23).view(torch.float32)


def deepgemm_post_process_fp8_weight_block(
    wq: torch.Tensor,
    ws: torch.Tensor,
    quant_block_shape: tuple[int, ...],
    use_e8m0: bool,
    is_bmm: bool = False,
    bmm_batch_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    del quant_block_shape, is_bmm, bmm_batch_size
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is not None and ws.dtype == e8m0_dtype and not use_e8m0:
        return wq, _upcast_e8m0_to_fp32(ws)
    return wq, ws


def _torch_per_token_group_quant_fp8(
    x: torch.Tensor,
    x_q: torch.Tensor,
    x_s: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    use_ue8m0: bool,
) -> None:
    groups = x.reshape(-1, group_size).to(torch.float32)
    scale_raw = torch.clamp(groups.abs().amax(dim=-1) / fp8_max, min=eps)
    if use_ue8m0:
        scale = torch.exp2(torch.ceil(torch.log2(scale_raw)))
    else:
        scale = scale_raw

    q = (groups / scale.unsqueeze(-1)).clamp(fp8_min, fp8_max).to(x_q.dtype)
    x_q.copy_(q.reshape_as(x_q))
    x_s.copy_(scale.reshape_as(x_s))


def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype | None = None,
    column_major_scales: bool = False,
    tma_aligned_scales: bool = False,
    out_q: torch.Tensor | None = None,
    use_ue8m0: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Function to perform per-token-group quantization on an input tensor `x`.
    It converts the tensor values into signed float8 values and returns the
    quantized tensor along with the scaling factor used for quantization.
    Args:
        x: The input tensor with ndim >= 2.
        group_size: The group size used for quantization.
        eps: The minimum to avoid dividing zero.
        dtype: The dtype of output tensor. Note that only `torch.float8_e4m3fn`
        is supported for now.
        column_major_scales: Outputs scales in column major.
        tma_aligned_scales: Outputs scales in TMA-aligned layout.
        out_q: Optional output tensor. If not provided, function will create.
    Returns:
        tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the
        scaling factor.
    """
    if use_ue8m0 is None:
        use_ue8m0 = is_deep_gemm_e8m0_used()
    dtype = current_platform.fp8_dtype() if dtype is None else dtype
    assert x.shape[-1] % group_size == 0, (
        f"the last dimension of `x` {x.shape[-1]} must be divisible "
        f"by `group_size` {group_size}"
    )
    assert x.stride(-1) == 1, "`x` groups must be contiguous"
    if current_platform.is_musa() and x.dim() == 2 and not x.is_contiguous():
        x = x.contiguous()

    fp8_min, fp8_max = get_fp8_min_max()

    assert out_q is None or out_q.shape == x.shape
    x_q = out_q
    if x_q is None:
        x_q = torch.empty(x.shape, device=x.device, dtype=dtype)

    # Allocate the scale tensor in either row- or column-major format.
    if column_major_scales:
        if tma_aligned_scales:
            m = x.shape[-2]
            sf_k = x.shape[-1] // group_size
            tma_aligned_m = get_tma_aligned_size(m, 4)
            shape = x.shape[:-2] + (m, sf_k)
            stride = (
                (1, tma_aligned_m)
                if x.dim() == 2
                else (tma_aligned_m * sf_k, 1, tma_aligned_m)
            )
            x_s = torch.empty_strided(
                shape, stride, device=x.device, dtype=torch.float32
            )
        else:
            shape = x.shape[:-2] + (x.shape[-1] // group_size, x.shape[-2])
            x_s = torch.empty(shape, device=x.device, dtype=torch.float32).permute(
                -1, -2
            )
    else:
        shape = x.shape[:-1] + (x.shape[-1] // group_size,)
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32)

    # prefer CUDA kernel if available
    # TODO(bnell): this causes some fp8 moe test to fail.
    if current_platform.is_musa() and x.is_contiguous():
        if x.dim() != 2:
            raise NotImplementedError(
                f"MUSA backend currently only supports 2D tensors for per_token_group_fp8_quant. "
                f"Got tensor with {x.dim()} dimensions, shape={x.shape}"
            )

        quant_op = getattr(
            getattr(torch.ops, "_C_musa_ops", None),
            "per_token_group_fp8_quant",
            None,
        )
        if quant_op is None:
            if (
                os.getenv(
                    "VLLM_MUSA_ENABLE_TORCH_FP8_GROUP_QUANT_FALLBACK", "0"
                )
                != "1"
            ):
                raise AttributeError(
                    "'_OpNamespace' '_C_musa_ops' object has no attribute "
                    "'per_token_group_fp8_quant'"
                )
            logger.warning_once(
                "Using opt-in MUSA torch per-token-group FP8 quant fallback. "
                "This is diagnostic and not a production replacement for the "
                "native _C_musa_ops.per_token_group_fp8_quant kernel."
            )
            _torch_per_token_group_quant_fp8(
                x,
                x_q,
                x_s,
                group_size,
                eps,
                fp8_min,
                fp8_max,
                use_ue8m0,
            )
        else:
            quant_op(
                x,
                x_q,
                x_s,
                group_size,
                eps,
                fp8_min,
                fp8_max,
                use_ue8m0,
                column_major_scales,
                tma_aligned_scales,
            )
        return x_q, x_s.contiguous()

    # TRITON FALLBACK
    # musa currently does not support triton fallback.
    raise NotImplementedError(
        f"per_token_group_fp8_quant is not supported for platform: {current_platform} or input is not contiguous. "
        "MUSA Triton fallback is currently not supported."
    )


import vllm.model_executor.layers.quantization.utils.fp8_utils

vllm.model_executor.layers.quantization.utils.fp8_utils.deepgemm_post_process_fp8_weight_block = (
    deepgemm_post_process_fp8_weight_block
)
