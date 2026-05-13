# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch DeepSeek-V4 inverse-RoPE FP8 quantization with a MUSA diagnostic fallback.
"""

PATCHES = [
    (
        """    if current_platform.is_musa() or o.device.type == "musa":
        raise NotImplementedError(
            "DeepSeek-V4 fused_inv_rope_fp8_quant is not implemented for "
            "MUSA yet. A MUSA-safe inverse-RoPE, FP8 quantization, and "
            "scale-layout path is required before model execution can proceed."
        )
""",
        "",
    ),
    (
        """    if (
        current_platform.is_musa()
        or getattr(torch.version, "musa", None) is not None
        or o.device.type == "musa"
    ):
        raise NotImplementedError(
            "DeepSeek-V4 fused_inv_rope_fp8_quant is not implemented for "
            "MUSA yet. A MUSA-safe inverse-RoPE, FP8 quantization, and "
            "scale-layout path is required before model execution can proceed."
        )
""",
        "",
    ),
    (
        """import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
""",
        """import os

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)
""",
    ),
    (
        """import torch

from vllm.triton_utils import tl, triton
""",
        """import os

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)
""",
    ),
    (
        """    from vllm.utils.deep_gemm import get_tma_aligned_size

    num_tokens, num_heads, head_dim = o.shape
""",
        """    if (
        current_platform.is_musa()
        or getattr(torch.version, "musa", None) is not None
        or o.device.type == "musa"
    ):
        if (
            os.getenv(
                "VLLM_MUSA_ENABLE_TORCH_FUSED_INV_ROPE_FP8_QUANT_FALLBACK",
                "0",
            )
            != "1"
        ):
            raise NotImplementedError(
                "DeepSeek-V4 fused_inv_rope_fp8_quant is not implemented for "
                "MUSA yet. A MUSA-safe inverse-RoPE, FP8 quantization, and "
                "scale-layout path is required before model execution can proceed."
            )
        from vllm.utils.deep_gemm import get_tma_aligned_size

        num_tokens, num_heads, head_dim = o.shape
        assert num_heads == n_groups * heads_per_group
        assert head_dim == nope_dim + rope_dim
        assert head_dim % quant_group_size == 0
        assert nope_dim % quant_group_size == (quant_group_size - rope_dim)
        assert rope_dim % 2 == 0
        assert cos_sin_cache.shape[-1] == rope_dim
        assert cos_sin_cache.dtype == torch.float32

        d = heads_per_group * head_dim
        num_scale_blocks = d // quant_group_size
        chunks_per_head = head_dim // quant_group_size
        fp8_dtype = torch.float8_e4m3fn
        fp8_max = torch.finfo(fp8_dtype).max

        x = o.to(torch.float32)
        rope_abs_start = (chunks_per_head - 1) * quant_group_size + (
            nope_dim % quant_group_size
        )
        rope = x[..., rope_abs_start : rope_abs_start + rope_dim]
        cos_sin = cos_sin_cache.index_select(0, positions.to(torch.long)).to(
            torch.float32
        )
        cos, sin = cos_sin.split(rope_dim // 2, dim=-1)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        even = rope[..., 0::2]
        odd = rope[..., 1::2]
        rotated = torch.empty_like(rope)
        rotated[..., 0::2] = even * cos + odd * sin
        rotated[..., 1::2] = odd * cos - even * sin
        x = x.clone()
        x[..., rope_abs_start : rope_abs_start + rope_dim] = rotated

        x = (
            x.reshape(num_tokens, n_groups, heads_per_group, head_dim)
            .permute(1, 0, 2, 3)
            .reshape(n_groups, num_tokens, d)
        )
        blocks = x.reshape(n_groups, num_tokens, num_scale_blocks, quant_group_size)
        scales = torch.exp2(
            torch.ceil(
                torch.log2(
                    torch.clamp(
                        blocks.abs().amax(dim=-1) / fp8_max,
                        min=1e-10,
                    )
                )
            )
        )
        fp8_buf = (blocks / scales.unsqueeze(-1)).clamp(
            -fp8_max, fp8_max
        ).to(fp8_dtype).reshape(n_groups, num_tokens, d)

        tma_aligned_T = get_tma_aligned_size(num_tokens, 4)
        if tma_aligned_scales:
            packed_sf_k = (num_scale_blocks + 3) // 4
            scale_buf = torch.empty(
                n_groups * packed_sf_k * tma_aligned_T,
                dtype=torch.int32,
                device=o.device,
            ).as_strided(
                (n_groups, num_tokens, packed_sf_k),
                (packed_sf_k * tma_aligned_T, 1, tma_aligned_T),
            )
            scale_bytes = (
                torch.log2(scales)
                .round()
                .to(torch.int32)
                .add_(127)
                .clamp_(0, 255)
            )
            scale_bytes = scale_bytes.reshape(
                n_groups, num_tokens, heads_per_group, chunks_per_head
            )
            shifts = (
                torch.arange(chunks_per_head, device=o.device, dtype=torch.int32)
                * 8
            )
            packed = torch.sum(scale_bytes << shifts, dim=-1)
            scale_buf.copy_(packed)
        else:
            scale_buf = torch.empty(
                n_groups * num_scale_blocks * tma_aligned_T,
                dtype=torch.float32,
                device=o.device,
            ).as_strided(
                (n_groups, num_tokens, num_scale_blocks),
                (num_scale_blocks * tma_aligned_T, 1, tma_aligned_T),
            )
            scale_buf.copy_(scales)

        logger.warning_once(
            "Using opt-in MUSA torch fused_inv_rope_fp8_quant fallback. "
            "This is diagnostic and not a production replacement for the "
            "native fused inverse-RoPE FP8 quant kernel."
        )
        return fp8_buf.transpose(0, 1), scale_buf.transpose(0, 1)
    from vllm.utils.deep_gemm import get_tma_aligned_size

    # Shape setup for the native Triton path.
    num_tokens, num_heads, head_dim = o.shape
""",
    ),
]

RELOAD_AFTER_PATCH = [
    "__TARGET_MODULE__",
    "vllm.v1.attention.ops.deepseek_v4_ops",
]
