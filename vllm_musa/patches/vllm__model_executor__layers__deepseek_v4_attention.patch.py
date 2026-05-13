# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch DeepSeek-V4 attention to use MUSA sparse FlashMLA backend shims.
"""

PATCHES = [
    (
        """from vllm.v1.attention.backends.mla.flashmla_sparse import (
    DeepseekV4FlashMLASparseBackend,
    FlashMLASparseBackend,
    FlashMLASparseMetadata,
)
""",
        """from vllm.v1.attention.backends.mla.flashmla_sparse import (
    FlashMLASparseBackend,
    FlashMLASparseMetadata,
)
from vllm_musa.v1.attention.backends.mla.flashmla_sparse import (
    MUSADeepseekV4FlashMLASparseBackend as DeepseekV4FlashMLASparseBackend,
)
""",
    ),
    (
        'assert cap is not None, "DeepseekV4 attention requires a CUDA device"',
        'assert cap is not None, "DeepseekV4 attention requires a MUSA device"',
    ),
    (
        """logger = init_logger(__name__)
""",
        """logger = init_logger(__name__)


def _musa_deepseek_v4_apply_gptj_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    nope_dim: int = 448,
    rope_dim: int = 64,
) -> torch.Tensor:
    x_float = x.to(torch.float32)
    rope = x_float[..., nope_dim : nope_dim + rope_dim]
    cos_sin = cos_sin_cache.index_select(0, positions.to(torch.long)).to(torch.float32)
    cos, sin = cos_sin.split(rope_dim // 2, dim=-1)
    while cos.dim() < rope.dim():
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    even = rope[..., 0::2]
    odd = rope[..., 1::2]
    rotated = torch.empty_like(rope)
    rotated[..., 0::2] = even * cos - odd * sin
    rotated[..., 1::2] = even * sin + odd * cos
    x_float[..., nope_dim : nope_dim + rope_dim] = rotated
    return x_float


def _musa_deepseek_v4_quant_insert(
    kv: torch.Tensor,
    k_cache_2d: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    fp8_dim = 448
    rope_dim = 64
    token_data_bytes = fp8_dim + rope_dim * 2
    scale_bytes = 8
    quant_block = 64
    fp8_max = 448.0
    valid_slots = slot_mapping[: kv.shape[0]]
    valid_mask = valid_slots >= 0
    if not torch.any(valid_mask):
        return
    kv_valid = kv[: valid_slots.shape[0]][valid_mask].to(kv.dtype).to(torch.float32)
    slots = valid_slots[valid_mask].to(torch.long)
    block_idx = torch.div(slots, block_size, rounding_mode="floor")
    pos_in_block = slots.remainder(block_size)

    for block_id in range(fp8_dim // quant_block):
        start = block_id * quant_block
        chunk = kv_valid[:, start : start + quant_block]
        amax = torch.maximum(
            chunk.abs().amax(dim=-1),
            torch.full((chunk.shape[0],), 1.0e-4, device=chunk.device),
        )
        exponent = torch.ceil(torch.log2(amax / fp8_max))
        scale = torch.exp2(exponent).unsqueeze(-1)
        qbytes = (
            (chunk / scale)
            .clamp(-fp8_max, fp8_max)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
        offsets = (
            pos_in_block.unsqueeze(1) * token_data_bytes
            + start
            + torch.arange(quant_block, device=kv.device).unsqueeze(0)
        )
        k_cache_2d[block_idx.unsqueeze(1), offsets] = qbytes
        scale_offsets = block_size * token_data_bytes + pos_in_block * scale_bytes
        k_cache_2d[block_idx, scale_offsets + block_id] = (
            exponent + 127.0
        ).clamp(0, 255).to(torch.uint8)
    scale_offsets = block_size * token_data_bytes + pos_in_block * scale_bytes
    k_cache_2d[block_idx, scale_offsets + 7] = 0

    rope_bytes = (
        kv_valid[:, fp8_dim : fp8_dim + rope_dim]
        .to(torch.bfloat16)
        .contiguous()
        .view(torch.uint8)
    )
    rope_offsets = (
        pos_in_block.unsqueeze(1) * token_data_bytes
        + fp8_dim
        + torch.arange(rope_dim * 2, device=kv.device).unsqueeze(0)
    )
    k_cache_2d[block_idx.unsqueeze(1), rope_offsets] = rope_bytes


def _musa_fused_deepseek_v4_qnorm_rope_kv_insert_fallback(
    q: torch.Tensor,
    kv: torch.Tensor,
    k_cache_2d: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    block_size: int,
) -> None:
    q_float = q.to(torch.float32)
    variance = q_float.pow(2).mean(dim=-1, keepdim=True)
    q_float = q_float * torch.rsqrt(variance + eps)
    q_rope = _musa_deepseek_v4_apply_gptj_rope(q_float, positions, cos_sin_cache)
    q.copy_(q_rope.to(q.dtype))
    kv_rope = _musa_deepseek_v4_apply_gptj_rope(kv, positions, cos_sin_cache).to(
        kv.dtype
    )
    _musa_deepseek_v4_quant_insert(kv_rope, k_cache_2d, slot_mapping, block_size)
""",
    ),
    (
        """        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            q,
            kv,
            swa_kv_cache_2d,
            swa_metadata.slot_mapping,
            positions.to(torch.int64),
            self.rotary_emb.cos_sin_cache,
            self.eps,
            swa_metadata.block_size,
        )
""",
        """        fused_insert = getattr(
            getattr(torch.ops, "_C", None),
            "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
            None,
        )
        if fused_insert is None:
            _musa_fused_deepseek_v4_qnorm_rope_kv_insert_fallback(
                q,
                kv,
                swa_kv_cache_2d,
                swa_metadata.slot_mapping,
                positions.to(torch.int64),
                self.rotary_emb.cos_sin_cache,
                self.eps,
                swa_metadata.block_size,
            )
            return
        fused_insert(
            q,
            kv,
            swa_kv_cache_2d,
            swa_metadata.slot_mapping,
            positions.to(torch.int64),
            self.rotary_emb.cos_sin_cache,
            self.eps,
            swa_metadata.block_size,
        )
""",
    ),
]
