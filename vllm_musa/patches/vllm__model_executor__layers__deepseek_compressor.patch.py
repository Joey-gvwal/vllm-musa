# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch DeepSeek-V4 compressor/cache paths with MUSA-specific gates.
"""

PATCHES = [
    (
        """from dataclasses import dataclass
from typing import Any, ClassVar, cast
""",
        """import os
from dataclasses import dataclass
from typing import Any, ClassVar, cast
""",
    ),
    (
        """from vllm.forward_context import get_forward_context
""",
        """from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
""",
    ),
    (
        """class DeepseekCompressor(nn.Module):
""",
        """logger = init_logger(__name__)


def _musa_deepseek_v4_is_musa_tensor(tensor: torch.Tensor) -> bool:
    return (
        current_platform.is_musa()
        or getattr(torch.version, "musa", None) is not None
        or tensor.device.type == "musa"
    )


def _musa_deepseek_v4_apply_gptj_rope(
    x: torch.Tensor,
    position: torch.Tensor,
    compress_ratio: int,
    rope_head_dim: int,
    cos_sin_cache: torch.Tensor,
) -> torch.Tensor:
    out = x.to(torch.float32).clone()
    nope_head_dim = out.shape[-1] - rope_head_dim
    compressed_pos = (
        torch.div(position, compress_ratio, rounding_mode="floor")
        * compress_ratio
    ).to(torch.long)
    cos_sin = cos_sin_cache[compressed_pos].to(torch.float32)
    cos, sin = cos_sin.split(rope_head_dim // 2, dim=-1)
    rope = out[..., nope_head_dim : nope_head_dim + rope_head_dim]
    even = rope[..., 0::2]
    odd = rope[..., 1::2]
    rotated = torch.empty_like(rope)
    rotated[..., 0::2] = even * cos - odd * sin
    rotated[..., 1::2] = odd * cos + even * sin
    out[..., nope_head_dim : nope_head_dim + rope_head_dim] = rotated
    return out


def _musa_deepseek_v4_e2m1_nibble(x: torch.Tensor) -> torch.Tensor:
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


def _musa_deepseek_v4_compressor_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    x_bf16 = x.to(torch.bfloat16).to(torch.float32)
    weight_bf16 = weight.to(torch.bfloat16).to(torch.float32)
    return torch.nn.functional.linear(
        x_bf16, weight_bf16
    )


def _musa_deepseek_v4_store_sparse_kv(
    normed: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_slot_idx: torch.Tensor,
    kv_cache_block_size: int,
    rope_head_dim: int,
) -> None:
    fp8_dim = normed.shape[-1] - rope_head_dim
    quant_block = 64
    token_stride = fp8_dim + rope_head_dim * 2
    scale_dim = fp8_dim // quant_block + 1
    fp8_max = 448.0

    block_idx = int(
        torch.div(kv_slot_idx, kv_cache_block_size, rounding_mode="floor").item()
    )
    pos_in_block = int(kv_slot_idx.remainder(kv_cache_block_size).item())
    if (
        block_idx < 0
        or block_idx >= kv_cache.shape[0]
        or pos_in_block < 0
        or pos_in_block >= kv_cache_block_size
    ):
        return
    cache_block = kv_cache[block_idx].view(torch.uint8).flatten()
    token_base = pos_in_block * token_stride
    scale_base = kv_cache_block_size * token_stride + pos_in_block * scale_dim
    quant_input = normed.to(torch.bfloat16).to(torch.float32)

    for block_id in range(fp8_dim // quant_block):
        start = block_id * quant_block
        chunk = quant_input[start : start + quant_block]
        amax = torch.maximum(
            chunk.abs().amax(),
            torch.full((), 1.0e-4, dtype=torch.float32, device=chunk.device),
        )
        exponent = torch.ceil(torch.log2(amax / fp8_max))
        inv_scale = torch.exp2(-exponent)
        qbytes = (
            (chunk * inv_scale)
            .clamp(-fp8_max, fp8_max)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
        cache_block[token_base + start : token_base + start + quant_block] = qbytes
        cache_block[scale_base + block_id] = (
            exponent + 127.0
        ).clamp(0, 255).to(torch.uint8)
    cache_block[scale_base + scale_dim - 1] = 0

    rope_bytes = (
        normed[fp8_dim : fp8_dim + rope_head_dim]
        .to(torch.bfloat16)
        .contiguous()
        .view(torch.uint8)
    )
    cache_block[token_base + fp8_dim : token_base + token_stride] = rope_bytes


def _musa_deepseek_v4_store_indexer_fp8(
    normed: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_slot_idx: torch.Tensor,
    kv_cache_block_size: int,
) -> None:
    head_dim = normed.shape[-1]
    scale_dim = 4
    fp8_max = 448.0
    block_idx = int(
        torch.div(kv_slot_idx, kv_cache_block_size, rounding_mode="floor").item()
    )
    pos_in_block = int(kv_slot_idx.remainder(kv_cache_block_size).item())
    if (
        block_idx < 0
        or block_idx >= kv_cache.shape[0]
        or pos_in_block < 0
        or pos_in_block >= kv_cache_block_size
    ):
        return
    cache_block = kv_cache[block_idx].view(torch.uint8).flatten()
    token_base = pos_in_block * head_dim
    scale_base = kv_cache_block_size * head_dim + pos_in_block * scale_dim
    quant_input = normed.to(torch.bfloat16).to(torch.float32)
    amax = torch.maximum(
        quant_input.abs().amax(),
        torch.full((), 1.0e-4, dtype=torch.float32, device=quant_input.device),
    )
    exponent = torch.ceil(torch.log2(amax / fp8_max))
    scale = torch.exp2(exponent)
    qbytes = (
        (quant_input / scale)
        .clamp(-fp8_max, fp8_max)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    cache_block[token_base : token_base + head_dim] = qbytes
    cache_block[scale_base : scale_base + scale_dim] = (
        scale.reshape(1).to(torch.float32).view(torch.uint8)
    )


def _musa_deepseek_v4_store_indexer_mxfp4(
    normed: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_slot_idx: torch.Tensor,
    kv_cache_block_size: int,
) -> None:
    head_dim = normed.shape[-1]
    quant_block = MXFP4_BLOCK_SIZE
    token_stride = head_dim // 2
    scale_dim = head_dim // quant_block
    block_idx = int(
        torch.div(kv_slot_idx, kv_cache_block_size, rounding_mode="floor").item()
    )
    pos_in_block = int(kv_slot_idx.remainder(kv_cache_block_size).item())
    if (
        block_idx < 0
        or block_idx >= kv_cache.shape[0]
        or pos_in_block < 0
        or pos_in_block >= kv_cache_block_size
    ):
        return
    cache_block = kv_cache[block_idx].view(torch.uint8).flatten()
    token_base = pos_in_block * token_stride
    scale_base = kv_cache_block_size * token_stride + pos_in_block * scale_dim

    x = normed.to(torch.bfloat16).to(torch.float32).reshape(scale_dim, quant_block)
    even = x[:, 0::2]
    odd = x[:, 1::2]
    amax = torch.maximum(even.abs().amax(dim=1), odd.abs().amax(dim=1))
    amax = torch.maximum(
        amax, torch.full_like(amax, 1.0e-4, dtype=torch.float32)
    )
    exponent = torch.ceil(torch.log2(amax / 6.0)).clamp(-127.0, 127.0)
    inv_scale = torch.exp2(-exponent).unsqueeze(-1)
    lo = _musa_deepseek_v4_e2m1_nibble(even * inv_scale)
    hi = _musa_deepseek_v4_e2m1_nibble(odd * inv_scale)
    packed = (lo | (hi << 4)).reshape(-1)
    cache_block[token_base : token_base + token_stride] = packed
    cache_block[scale_base : scale_base + scale_dim] = (
        exponent + 127.0
    ).to(torch.uint8)


def _musa_deepseek_v4_cache_block_view(kv_cache: torch.Tensor) -> torch.Tensor:
    cache_u8 = kv_cache.view(torch.uint8)
    block_stride = cache_u8.stride(0)
    return cache_u8.as_strided(
        (cache_u8.shape[0], block_stride),
        (block_stride, 1),
    )


def _musa_deepseek_v4_store_sparse_kv_vectorized(
    normed: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_slot_idx: torch.Tensor,
    kv_cache_block_size: int,
    rope_head_dim: int,
) -> None:
    fp8_dim = normed.shape[-1] - rope_head_dim
    quant_block = 64
    token_stride = fp8_dim + rope_head_dim * 2
    scale_dim = fp8_dim // quant_block + 1
    fp8_max = 448.0

    block_idx = torch.div(
        kv_slot_idx, kv_cache_block_size, rounding_mode="floor"
    ).to(torch.long)
    pos_in_block = kv_slot_idx.remainder(kv_cache_block_size).to(torch.long)
    valid = (
        (block_idx >= 0)
        & (block_idx < kv_cache.shape[0])
        & (pos_in_block >= 0)
        & (pos_in_block < kv_cache_block_size)
    )
    if not bool(torch.any(valid).item()):
        return

    block_idx = block_idx[valid]
    pos_in_block = pos_in_block[valid]
    normed = normed[valid]
    cache_blocks = _musa_deepseek_v4_cache_block_view(kv_cache)
    base = pos_in_block * token_stride
    scale_base = kv_cache_block_size * token_stride + pos_in_block * scale_dim

    quant_input = normed[:, :fp8_dim].to(torch.bfloat16).to(torch.float32)
    qblocks = quant_input.reshape(normed.shape[0], fp8_dim // quant_block, quant_block)
    amax = torch.maximum(
        qblocks.abs().amax(dim=-1),
        torch.full((), 1.0e-4, dtype=torch.float32, device=normed.device),
    )
    exponent = torch.ceil(torch.log2(amax / fp8_max))
    inv_scale = torch.exp2(-exponent).unsqueeze(-1)
    qbytes = (
        (qblocks * inv_scale)
        .clamp(-fp8_max, fp8_max)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )

    q_offsets = (
        base[:, None, None]
        + torch.arange(fp8_dim // quant_block, device=normed.device)[:, None]
        * quant_block
        + torch.arange(quant_block, device=normed.device)
    )
    q_block_idx = block_idx[:, None, None].expand_as(q_offsets)
    cache_blocks[q_block_idx.reshape(-1), q_offsets.reshape(-1)] = qbytes.reshape(-1)

    scale_offsets = (
        scale_base[:, None]
        + torch.arange(fp8_dim // quant_block, device=normed.device)
    )
    scale_block_idx = block_idx[:, None].expand_as(scale_offsets)
    cache_blocks[scale_block_idx.reshape(-1), scale_offsets.reshape(-1)] = (
        exponent + 127.0
    ).clamp(0, 255).to(torch.uint8).reshape(-1)
    cache_blocks[block_idx, scale_base + scale_dim - 1] = 0

    rope_bytes = (
        normed[:, fp8_dim : fp8_dim + rope_head_dim]
        .to(torch.bfloat16)
        .contiguous()
        .view(torch.uint8)
        .reshape(normed.shape[0], rope_head_dim * 2)
    )
    rope_offsets = (
        base[:, None]
        + fp8_dim
        + torch.arange(rope_head_dim * 2, device=normed.device)
    )
    rope_block_idx = block_idx[:, None].expand_as(rope_offsets)
    cache_blocks[rope_block_idx.reshape(-1), rope_offsets.reshape(-1)] = (
        rope_bytes.reshape(-1)
    )


def _musa_deepseek_v4_store_indexer_fp8_vectorized(
    normed: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_slot_idx: torch.Tensor,
    kv_cache_block_size: int,
) -> None:
    head_dim = normed.shape[-1]
    scale_dim = 4
    fp8_max = 448.0
    block_idx = torch.div(
        kv_slot_idx, kv_cache_block_size, rounding_mode="floor"
    ).to(torch.long)
    pos_in_block = kv_slot_idx.remainder(kv_cache_block_size).to(torch.long)
    valid = (
        (block_idx >= 0)
        & (block_idx < kv_cache.shape[0])
        & (pos_in_block >= 0)
        & (pos_in_block < kv_cache_block_size)
    )
    if not bool(torch.any(valid).item()):
        return

    block_idx = block_idx[valid]
    pos_in_block = pos_in_block[valid]
    normed = normed[valid]
    cache_blocks = _musa_deepseek_v4_cache_block_view(kv_cache)
    base = pos_in_block * head_dim
    scale_base = kv_cache_block_size * head_dim + pos_in_block * scale_dim

    quant_input = normed.to(torch.bfloat16).to(torch.float32)
    amax = torch.maximum(
        quant_input.abs().amax(dim=-1),
        torch.full((quant_input.shape[0],), 1.0e-4, dtype=torch.float32, device=normed.device),
    )
    exponent = torch.ceil(torch.log2(amax / fp8_max))
    scale = torch.exp2(exponent)
    qbytes = (
        (quant_input / scale.unsqueeze(-1))
        .clamp(-fp8_max, fp8_max)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    offsets = base[:, None] + torch.arange(head_dim, device=normed.device)
    token_block_idx = block_idx[:, None].expand_as(offsets)
    cache_blocks[token_block_idx.reshape(-1), offsets.reshape(-1)] = qbytes.reshape(-1)
    scale_bytes = scale.reshape(-1).to(torch.float32).view(torch.uint8).reshape(-1, 4)
    scale_offsets = scale_base[:, None] + torch.arange(scale_dim, device=normed.device)
    scale_block_idx = block_idx[:, None].expand_as(scale_offsets)
    cache_blocks[scale_block_idx.reshape(-1), scale_offsets.reshape(-1)] = (
        scale_bytes.reshape(-1)
    )


def _musa_deepseek_v4_store_indexer_mxfp4_vectorized(
    normed: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_slot_idx: torch.Tensor,
    kv_cache_block_size: int,
) -> None:
    head_dim = normed.shape[-1]
    quant_block = MXFP4_BLOCK_SIZE
    token_stride = head_dim // 2
    scale_dim = head_dim // quant_block
    block_idx = torch.div(
        kv_slot_idx, kv_cache_block_size, rounding_mode="floor"
    ).to(torch.long)
    pos_in_block = kv_slot_idx.remainder(kv_cache_block_size).to(torch.long)
    valid = (
        (block_idx >= 0)
        & (block_idx < kv_cache.shape[0])
        & (pos_in_block >= 0)
        & (pos_in_block < kv_cache_block_size)
    )
    if not bool(torch.any(valid).item()):
        return

    block_idx = block_idx[valid]
    pos_in_block = pos_in_block[valid]
    normed = normed[valid]
    cache_blocks = _musa_deepseek_v4_cache_block_view(kv_cache)
    base = pos_in_block * token_stride
    scale_base = kv_cache_block_size * token_stride + pos_in_block * scale_dim

    x = normed.to(torch.bfloat16).to(torch.float32).reshape(
        normed.shape[0], scale_dim, quant_block
    )
    even = x[..., 0::2]
    odd = x[..., 1::2]
    amax = torch.maximum(even.abs().amax(dim=-1), odd.abs().amax(dim=-1))
    amax = torch.maximum(
        amax, torch.full_like(amax, 1.0e-4, dtype=torch.float32)
    )
    exponent = torch.ceil(torch.log2(amax / 6.0)).clamp(-127.0, 127.0)
    inv_scale = torch.exp2(-exponent).unsqueeze(-1)
    lo = _musa_deepseek_v4_e2m1_nibble(even * inv_scale)
    hi = _musa_deepseek_v4_e2m1_nibble(odd * inv_scale)
    packed = (lo | (hi << 4)).reshape(normed.shape[0], token_stride)
    offsets = base[:, None] + torch.arange(token_stride, device=normed.device)
    token_block_idx = block_idx[:, None].expand_as(offsets)
    cache_blocks[token_block_idx.reshape(-1), offsets.reshape(-1)] = packed.reshape(-1)
    scale_offsets = scale_base[:, None] + torch.arange(scale_dim, device=normed.device)
    scale_block_idx = block_idx[:, None].expand_as(scale_offsets)
    cache_blocks[scale_block_idx.reshape(-1), scale_offsets.reshape(-1)] = (
        (exponent + 127.0).to(torch.uint8).reshape(-1)
    )


def _musa_deepseek_v4_try_vectorized_compressor_store(
    module: "DeepseekCompressor",
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    state_cache: torch.Tensor,
    state_width: int,
    kv_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    num_actual: int,
) -> bool:
    mode = os.getenv(
        "VLLM_MUSA_DEEPSEEK_V4_COMPRESSOR_FALLBACK_IMPL", "vectorized"
    ).strip().lower()
    if mode not in {"vectorized", "vec"}:
        return False
    active_slots = slot_mapping[:num_actual]
    active_positions = positions[:num_actual]
    active_kv_slots = kv_slot_mapping[:num_actual]
    boundary_mask = (
        (active_slots >= 0)
        & (active_kv_slots >= 0)
        & (active_positions + 1).remainder(module.compress_ratio).eq(0)
    )
    if not bool(torch.any(boundary_mask).item()):
        return True

    token_indices = torch.nonzero(boundary_mask, as_tuple=False).flatten()
    comp_positions = active_positions[token_indices].to(torch.long)
    req_indices = token_to_req_indices[token_indices].to(torch.long)
    kv_slot_idx = active_kv_slots[token_indices].to(torch.long)

    window = module.coff * module.compress_ratio
    offsets = torch.arange(window, device=positions.device, dtype=torch.long)
    gather_pos = comp_positions[:, None] - window + 1 + offsets[None, :]
    valid = gather_pos >= 0
    block_indices = torch.div(gather_pos, block_size, rounding_mode="floor")
    valid = valid & (block_indices < block_table.shape[1])
    block_indices_safe = block_indices.clamp(0, max(block_table.shape[1] - 1, 0))
    block_numbers = block_table[req_indices[:, None], block_indices_safe].to(torch.long)
    valid = valid & (block_numbers >= 0) & (block_numbers < state_cache.shape[0])
    if not bool(torch.any(valid).item()):
        return True

    block_numbers_safe = block_numbers.clamp(0, max(state_cache.shape[0] - 1, 0))
    block_offsets = gather_pos.remainder(block_size).clamp(0, block_size - 1).to(torch.long)
    state_rows = state_cache[block_numbers_safe, block_offsets]

    head_offsets = (offsets >= module.compress_ratio).to(torch.long) * module.head_dim
    cols = head_offsets[:, None] + torch.arange(module.head_dim, device=positions.device)
    kv_rows = state_rows.gather(2, cols[None, :, :].expand(state_rows.shape[0], -1, -1))
    score_rows = state_rows.gather(
        2,
        (state_width + cols)[None, :, :].expand(state_rows.shape[0], -1, -1),
    )
    valid_3d = valid[:, :, None]
    kv_rows = torch.where(valid_3d, kv_rows.to(torch.float32), torch.zeros_like(kv_rows, dtype=torch.float32))
    score_rows = torch.where(
        valid_3d,
        score_rows.to(torch.float32),
        torch.full_like(score_rows, float("-inf"), dtype=torch.float32),
    )
    weights = torch.softmax(score_rows, dim=1)
    compressed = (kv_rows * weights).sum(dim=1)
    variance = compressed.pow(2).sum(dim=-1, keepdim=True) / module.head_dim
    normed = (
        compressed
        * torch.rsqrt(variance + module.rms_norm_eps)
        * module.norm.weight.to(torch.float32)
    )
    normed = _musa_deepseek_v4_apply_gptj_rope(
        normed,
        comp_positions,
        module.compress_ratio,
        module.rope_head_dim,
        cos_sin_cache,
    )

    if module.head_dim == 512:
        _musa_deepseek_v4_store_sparse_kv_vectorized(
            normed,
            kv_cache,
            kv_slot_idx,
            kv_cache.shape[1],
            module.rope_head_dim,
        )
    elif module.use_fp4_cache:
        _musa_deepseek_v4_store_indexer_mxfp4_vectorized(
            normed,
            kv_cache,
            kv_slot_idx,
            kv_cache.shape[1],
        )
    else:
        _musa_deepseek_v4_store_indexer_fp8_vectorized(
            normed,
            kv_cache,
            kv_slot_idx,
            kv_cache.shape[1],
        )
    return True


def _musa_deepseek_v4_compressor_forward(
    module: "DeepseekCompressor",
    x: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb,
) -> None:
    num_tokens, _ = x.shape
    kv_score = _musa_deepseek_v4_compressor_gemm(
        x, module.fused_wkv_wgate.weight
    )
    kv, score = kv_score.split(
        [module.coff * module.head_dim, module.coff * module.head_dim],
        dim=-1,
    )

    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        return

    state_metadata = cast(
        CompressorMetadata, attn_metadata[module.state_cache.prefix]
    )
    token_to_req_indices = state_metadata.token_to_req_indices
    slot_mapping = state_metadata.slot_mapping
    num_actual = min(slot_mapping.shape[0], num_tokens)
    block_table = state_metadata.block_table
    block_size = state_metadata.block_size

    state_cache = module.state_cache.kv_cache
    state_width = state_cache.shape[-1] // 2
    kv_width = kv.shape[-1]

    valid_slots = slot_mapping[:num_actual]
    state_cache_capacity = state_cache.shape[0] * block_size
    valid_mask = (valid_slots >= 0) & (valid_slots < state_cache_capacity)
    if bool(torch.any(valid_mask).item()):
        valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
        slots = valid_slots[valid_indices].to(torch.long)
        block_idx = torch.div(slots, block_size, rounding_mode="floor")
        pos_in_block = slots.remainder(block_size)
        state_cache[block_idx, pos_in_block, :kv_width] = kv[valid_indices]
        ape_rows = positions[:num_actual][valid_indices].remainder(
            module.compress_ratio
        ).to(torch.long)
        state_cache[block_idx, pos_in_block, state_width : state_width + kv_width] = (
            score[valid_indices] + module.ape[ape_rows]
        )

    cos_sin_cache = rotary_emb.cos_sin_cache
    k_cache_metadata = cast(Any, attn_metadata[module.k_cache_prefix])
    kv_cache = module._static_forward_context[module.k_cache_prefix].kv_cache
    kv_slot_mapping = k_cache_metadata.slot_mapping
    if _musa_deepseek_v4_try_vectorized_compressor_store(
        module,
        positions,
        cos_sin_cache,
        token_to_req_indices,
        slot_mapping,
        block_table,
        block_size,
        state_cache,
        state_width,
        kv_cache,
        kv_slot_mapping,
        num_actual,
    ):
        return

    for token_idx in range(num_actual):
        slot = slot_mapping[token_idx]
        if slot < 0:
            continue
        position = positions[token_idx]
        if int((position + 1).item()) % module.compress_ratio != 0:
            continue
        kv_slot_idx = kv_slot_mapping[token_idx]
        if kv_slot_idx < 0:
            continue

        req_idx = token_to_req_indices[token_idx].to(torch.long)
        start = int((position - module.coff * module.compress_ratio + 1).item())
        kv_rows = []
        score_rows = []
        for offset_idx in range(module.coff * module.compress_ratio):
            pos = start + offset_idx
            if pos < 0:
                continue
            block_index = pos // block_size
            if block_index >= block_table.shape[1]:
                continue
            block_number = block_table[req_idx, block_index].to(torch.long)
            if block_number < 0:
                continue
            if int(block_number.item()) >= state_cache.shape[0]:
                continue
            block_offset = pos % block_size
            head_offset = module.head_dim if offset_idx >= module.compress_ratio else 0
            kv_rows.append(
                state_cache[
                    block_number,
                    block_offset,
                    head_offset : head_offset + module.head_dim,
                ].to(torch.float32)
            )
            score_rows.append(
                state_cache[
                    block_number,
                    block_offset,
                    state_width + head_offset : state_width + head_offset + module.head_dim,
                ].to(torch.float32)
            )
        if not kv_rows:
            continue
        kv_stack = torch.stack(kv_rows, dim=0)
        score_stack = torch.stack(score_rows, dim=0)
        weights = torch.softmax(score_stack, dim=0)
        compressed = (kv_stack * weights).sum(dim=0)
        variance = compressed.pow(2).sum() / module.head_dim
        normed = (
            compressed
            * torch.rsqrt(variance + module.rms_norm_eps)
            * module.norm.weight.to(torch.float32)
        )
        normed = _musa_deepseek_v4_apply_gptj_rope(
            normed,
            position,
            module.compress_ratio,
            module.rope_head_dim,
            cos_sin_cache,
        )
        if module.head_dim == 512:
            _musa_deepseek_v4_store_sparse_kv(
                normed,
                kv_cache,
                kv_slot_idx.to(torch.long),
                kv_cache.shape[1],
                module.rope_head_dim,
            )
        elif module.use_fp4_cache:
            _musa_deepseek_v4_store_indexer_mxfp4(
                normed,
                kv_cache,
                kv_slot_idx.to(torch.long),
                kv_cache.shape[1],
            )
        else:
            _musa_deepseek_v4_store_indexer_fp8(
                normed,
                kv_cache,
                kv_slot_idx.to(torch.long),
                kv_cache.shape[1],
            )


class DeepseekCompressor(nn.Module):
""",
    ),
    (
        """    def forward(
        self,
        # [num_tokens, hidden_size]
        x: torch.Tensor,
        # [num_tokens]
        positions: torch.Tensor,
        rotary_emb,
    ) -> None:
        num_tokens, _ = x.shape
""",
        """    def forward(
        self,
        # [num_tokens, hidden_size]
        x: torch.Tensor,
        # [num_tokens]
        positions: torch.Tensor,
        rotary_emb,
    ) -> None:
        if (
            _musa_deepseek_v4_is_musa_tensor(x)
            and os.getenv(
                "VLLM_MUSA_ENABLE_TORCH_DEEPSEEK_V4_COMPRESSOR_FALLBACK",
                "0",
            )
            == "1"
        ):
            logger.warning_once(
                "Using opt-in MUSA torch DeepSeek-V4 compressor/cache "
                "fallback. This emulates fused compress, RMSNorm/RoPE, and "
                "FP8/MXFP4 cache insert in torch; it is diagnostic, not a "
                "production backend."
            )
            return _musa_deepseek_v4_compressor_forward(
                self, x, positions, rotary_emb
            )
        if _musa_deepseek_v4_is_musa_tensor(x):
            raise NotImplementedError(
                "DeepSeek-V4 compressor/cache updates are not implemented for "
                "MUSA yet. A MUSA implementation of the fused compress, "
                "RMSNorm/RoPE, FP8/MXFP4 quantization, and KV-cache insert "
                "path is required before model execution can proceed."
            )
        num_tokens, _ = x.shape
""",
    ),
    (
        """        if (
            current_platform.is_musa()
            or getattr(torch.version, "musa", None) is not None
            or x.device.type == "musa"
        ):
            raise NotImplementedError(
                "DeepSeek-V4 compressor/cache updates are not implemented for "
                "MUSA yet. A MUSA implementation of the fused compress, "
                "RMSNorm/RoPE, FP8/MXFP4 quantization, and KV-cache insert "
                "path is required before model execution can proceed."
            )
        num_tokens, _ = x.shape
""",
        """        if (
            _musa_deepseek_v4_is_musa_tensor(x)
            and os.getenv(
                "VLLM_MUSA_ENABLE_TORCH_DEEPSEEK_V4_COMPRESSOR_FALLBACK",
                "0",
            )
            == "1"
        ):
            logger.warning_once(
                "Using opt-in MUSA torch DeepSeek-V4 compressor/cache "
                "fallback. This emulates fused compress, RMSNorm/RoPE, and "
                "FP8/MXFP4 cache insert in torch; it is diagnostic, not a "
                "production backend."
            )
            return _musa_deepseek_v4_compressor_forward(
                self, x, positions, rotary_emb
            )
        if _musa_deepseek_v4_is_musa_tensor(x):
            raise NotImplementedError(
                "DeepSeek-V4 compressor/cache updates are not implemented for "
                "MUSA yet. A MUSA implementation of the fused compress, "
                "RMSNorm/RoPE, FP8/MXFP4 quantization, and KV-cache insert "
                "path is required before model execution can proceed."
            )
        num_tokens, _ = x.shape
""",
    ),
]

RELOAD_AFTER_PATCH = True
