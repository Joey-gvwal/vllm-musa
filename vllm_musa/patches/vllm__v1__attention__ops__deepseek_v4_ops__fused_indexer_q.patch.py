# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch DeepSeek-V4 sparse-indexer Q quantization with a MUSA-specific gate.
"""

PATCHES = [
    (
        """import torch

from vllm.triton_utils import tl, triton
""",
        """import os

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
""",
    ),
    (
        """MXFP4_BLOCK_SIZE = 32
""",
        """MXFP4_BLOCK_SIZE = 32

logger = init_logger(__name__)


def _musa_is_musa_tensor(tensor: torch.Tensor) -> bool:
    return (
        current_platform.is_musa()
        or getattr(torch.version, "musa", None) is not None
        or tensor.device.type == "musa"
    )


def _musa_fused_indexer_q_triton_enabled() -> bool:
    return os.getenv("VLLM_MUSA_DEEPSEEK_V4_INDEXER_Q_TRITON", "1") == "1"


def _musa_apply_indexer_gptj_rope(
    index_q: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> torch.Tensor:
    out = index_q.to(torch.float32).clone()
    head_dim = out.shape[-1]
    rope_dim = cos_sin_cache.shape[-1]
    half_rope_dim = rope_dim // 2
    nope_dim = head_dim - rope_dim
    cos_sin = cos_sin_cache.index_select(0, positions.to(torch.long)).to(
        torch.float32
    )
    cos, sin = cos_sin.split(half_rope_dim, dim=-1)
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    rope = out[..., nope_dim:]
    even = rope[..., 0::2]
    odd = rope[..., 1::2]
    rotated = torch.empty_like(rope)
    rotated[..., 0::2] = (even * cos - odd * sin).to(torch.bfloat16).to(
        torch.float32
    )
    rotated[..., 1::2] = (odd * cos + even * sin).to(torch.bfloat16).to(
        torch.float32
    )
    out[..., nope_dim:] = rotated
    return out


def _musa_e2m1_nibble(x: torch.Tensor) -> torch.Tensor:
    abs_x = torch.minimum(
        x.abs(), torch.full((), 6.0, dtype=torch.float32, device=x.device)
    )
    code = torch.where(
        abs_x <= 0.25,
        0,
        torch.where(
            abs_x <= 0.75,
            1,
            torch.where(
                abs_x <= 1.25,
                2,
                torch.where(
                    abs_x <= 1.75,
                    3,
                    torch.where(
                        abs_x <= 2.5,
                        4,
                        torch.where(abs_x <= 3.5, 5, torch.where(abs_x <= 5.0, 6, 7)),
                    ),
                ),
            ),
        ),
    ).to(torch.uint8)
    sign = ((x < 0) & (code != 0)).to(torch.uint8)
    return code | (sign << 3)


def _musa_quantize_mxfp4_pair(
    even: torch.Tensor,
    odd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    amax = torch.maximum(even.abs().amax(dim=-1), odd.abs().amax(dim=-1))
    amax = torch.maximum(
        amax, torch.full_like(amax, 1.0e-4, dtype=torch.float32)
    )
    exponent = torch.ceil(torch.log2(amax / 6.0)).clamp(-127.0, 127.0)
    inv_scale = torch.exp2(-exponent).unsqueeze(-1)
    lo = _musa_e2m1_nibble(even * inv_scale)
    hi = _musa_e2m1_nibble(odd * inv_scale)
    return lo | (hi << 4), (exponent + 127.0).to(torch.uint8)


def _musa_fused_indexer_q_rope_quant_fallback(
    positions: torch.Tensor,
    index_q: torch.Tensor,
    index_q_cos_sin_cache: torch.Tensor,
    index_weights: torch.Tensor,
    index_weights_softmax_scale: float,
    index_weights_head_scale: float,
    use_fp4: bool,
) -> tuple[
    torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    torch.Tensor,
]:
    if (
        positions.ndim != 1
        or index_q.ndim != 3
        or index_q_cos_sin_cache.ndim != 2
    ):
        raise AssertionError("unexpected DeepSeek-V4 indexer Q fallback shape")
    q_rope = _musa_apply_indexer_gptj_rope(
        index_q, positions, index_q_cos_sin_cache
    )
    weights_out = index_weights.to(torch.float32).clone()
    weights_out *= index_weights_softmax_scale
    weights_out *= index_weights_head_scale

    if use_fp4:
        head_dim = index_q.shape[-1]
        assert head_dim % MXFP4_BLOCK_SIZE == 0
        q_blocks = q_rope.reshape(
            *q_rope.shape[:-1], head_dim // MXFP4_BLOCK_SIZE, MXFP4_BLOCK_SIZE
        )
        even = q_blocks[..., 0::2]
        odd = q_blocks[..., 1::2]
        packed, ue8m0 = _musa_quantize_mxfp4_pair(even, odd)
        packed = packed.reshape(*q_rope.shape[:-1], head_dim // 2).contiguous()
        return (
            packed,
            ue8m0.contiguous().view(torch.int32).squeeze(-1),
        ), weights_out

    amax = q_rope.abs().amax(dim=-1)
    amax = torch.maximum(
        amax, torch.full_like(amax, 1.0e-4, dtype=torch.float32)
    )
    q_scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    q_fp8 = (q_rope / q_scale.unsqueeze(-1)).clamp(-448.0, 448.0).to(
        torch.float8_e4m3fn
    )
    weights_out *= q_scale
    return q_fp8, weights_out
""",
    ),
    (
        """    assert positions.ndim == 1
    assert index_q.ndim == 3
""",
        """    if _musa_is_musa_tensor(index_q):
        if _musa_fused_indexer_q_triton_enabled():
            logger.warning_once(
                "Using MUSA Triton DeepSeek-V4 "
                "fused_indexer_q_rope_quant path."
            )
        elif (
            os.getenv(
                "VLLM_MUSA_ENABLE_TORCH_FUSED_INDEXER_Q_ROPE_QUANT_FALLBACK",
                "0",
            )
            == "1"
        ):
            logger.warning_once(
                "Using opt-in MUSA torch DeepSeek-V4 indexer Q RoPE/quant "
                "fallback. This emulates fused_indexer_q_rope_quant in torch; "
                "it is a correctness fallback, not a production backend."
            )
            return _musa_fused_indexer_q_rope_quant_fallback(
                positions,
                index_q,
                index_q_cos_sin_cache,
                index_weights,
                index_weights_softmax_scale,
                index_weights_head_scale,
                use_fp4,
            )
        else:
            raise NotImplementedError(
                "DeepSeek-V4 fused_indexer_q_rope_quant is not implemented for "
                "MUSA yet. A MUSA-safe sparse indexer Q RoPE/FP8-or-MXFP4 "
                "quantization path is required before model execution can proceed."
            )
    assert positions.ndim == 1
    assert index_q.ndim == 3
""",
    ),
    (
        """    if _musa_is_musa_tensor(index_q):
        if (
            os.getenv(
                "VLLM_MUSA_ENABLE_TORCH_FUSED_INDEXER_Q_ROPE_QUANT_FALLBACK",
                "0",
            )
            == "1"
        ):
            logger.warning_once(
                "Using opt-in MUSA torch DeepSeek-V4 indexer Q RoPE/quant "
                "fallback. This emulates fused_indexer_q_rope_quant in torch; "
                "it is a correctness fallback, not a production backend."
            )
            return _musa_fused_indexer_q_rope_quant_fallback(
                positions,
                index_q,
                index_q_cos_sin_cache,
                index_weights,
                index_weights_softmax_scale,
                index_weights_head_scale,
                use_fp4,
            )
        raise NotImplementedError(
            "DeepSeek-V4 fused_indexer_q_rope_quant is not implemented for "
            "MUSA yet. A MUSA-safe sparse indexer Q RoPE/FP8-or-MXFP4 "
            "quantization path is required before model execution can proceed."
        )
    assert positions.ndim == 1
    assert index_q.ndim == 3
""",
        """    if _musa_is_musa_tensor(index_q):
        if _musa_fused_indexer_q_triton_enabled():
            logger.warning_once(
                "Using MUSA Triton DeepSeek-V4 "
                "fused_indexer_q_rope_quant path."
            )
        elif (
            os.getenv(
                "VLLM_MUSA_ENABLE_TORCH_FUSED_INDEXER_Q_ROPE_QUANT_FALLBACK",
                "0",
            )
            == "1"
        ):
            logger.warning_once(
                "Using opt-in MUSA torch DeepSeek-V4 indexer Q RoPE/quant "
                "fallback. This emulates fused_indexer_q_rope_quant in torch; "
                "it is a correctness fallback, not a production backend."
            )
            return _musa_fused_indexer_q_rope_quant_fallback(
                positions,
                index_q,
                index_q_cos_sin_cache,
                index_weights,
                index_weights_softmax_scale,
                index_weights_head_scale,
                use_fp4,
            )
        else:
            raise NotImplementedError(
                "DeepSeek-V4 fused_indexer_q_rope_quant is not implemented for "
                "MUSA yet. A MUSA-safe sparse indexer Q RoPE/FP8-or-MXFP4 "
                "quantization path is required before model execution can proceed."
            )
    assert positions.ndim == 1
    assert index_q.ndim == 3
""",
    ),
]

RELOAD_AFTER_PATCH = [
    "__TARGET_MODULE__",
    "vllm.v1.attention.ops.deepseek_v4_ops",
]
