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


def _musa_deepseek_v4_is_current_stream_capturing() -> bool:
    cuda_module = getattr(torch, "cuda", None)
    if cuda_module is None:
        return False
    is_capturing = getattr(cuda_module, "is_current_stream_capturing", None)
    if is_capturing is None:
        return False
    try:
        return bool(is_capturing())
    except Exception:
        return False


def _musa_deepseek_v4_vectorized_eager_compressor_enabled() -> bool:
    return (
        os.getenv(
            "VLLM_MUSA_DEEPSEEK_V4_COMPRESSOR_VECTORIZE_EAGER",
            "0",
        )
        == "1"
    )


def _musa_deepseek_v4_native_sparse_store_enabled() -> bool:
    return (
        os.getenv(
            "VLLM_MUSA_DEEPSEEK_V4_NATIVE_SPARSE_STORE",
            "0",
        )
        == "1"
    )


def _musa_deepseek_v4_triton_compressor_enabled() -> bool:
    return (
        os.getenv(
            "VLLM_MUSA_DEEPSEEK_V4_COMPRESSOR_TRITON",
            "1",
        )
        == "1"
    )


def _musa_deepseek_v4_native_store_sparse_kv(
    normed: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_slots: torch.Tensor,
    kv_cache_block_size: int,
    write_mask: torch.Tensor,
) -> None:
    if kv_cache_block_size != kv_cache.shape[1]:
        raise AssertionError(
            "native DeepSeek-V4 sparse store expects kv_cache_block_size "
            f"to match kv_cache.shape[1], got {kv_cache_block_size} and "
            f"{kv_cache.shape[1]}"
        )
    from vllm_musa import _custom_ops as _musa_custom_ops

    _musa_custom_ops.deepseek_v4_store_sparse_kv(
        normed.to(torch.bfloat16).contiguous(),
        kv_cache,
        kv_slots.contiguous(),
        write_mask.to(torch.bool).contiguous(),
    )


def _musa_deepseek_v4_save_partial_states_capture(
    kv: torch.Tensor,
    score: torch.Tensor,
    positions: torch.Tensor,
    ape: torch.Tensor,
    state_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    compress_ratio: int,
) -> torch.Tensor:
    num_actual = min(slot_mapping.shape[0], kv.shape[0])
    state_width = state_cache.shape[-1] // 2
    kv_width = kv.shape[-1]
    state_capacity = state_cache.shape[0] * block_size

    slots = slot_mapping[:num_actual].to(torch.long)
    valid = (slots >= 0) & (slots < state_capacity)
    safe_slots = slots.clamp(0, state_capacity - 1)
    block_idx = torch.div(safe_slots, block_size, rounding_mode="floor")
    pos_in_block = safe_slots.remainder(block_size)

    ape_rows = positions[:num_actual].remainder(compress_ratio).to(torch.long)
    score_with_ape = score[:num_actual] + ape[ape_rows]

    existing_kv = state_cache[block_idx, pos_in_block, :kv_width]
    state_cache[block_idx, pos_in_block, :kv_width] = torch.where(
        valid.unsqueeze(-1), kv[:num_actual], existing_kv
    )

    existing_score = state_cache[
        block_idx,
        pos_in_block,
        state_width : state_width + kv_width,
    ]
    state_cache[
        block_idx,
        pos_in_block,
        state_width : state_width + kv_width,
    ] = torch.where(valid.unsqueeze(-1), score_with_ape, existing_score)
    return valid


def _musa_deepseek_v4_gather_compressed_states_capture(
    module: "DeepseekCompressor",
    positions: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    state_cache: torch.Tensor,
    block_size: int,
    num_actual: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    window = module.coff * module.compress_ratio
    device = positions.device
    offsets = torch.arange(window, device=device, dtype=positions.dtype)
    rel_positions = positions[:num_actual].unsqueeze(-1) - window + 1 + offsets

    valid_pos = rel_positions >= 0
    block_indices = torch.div(rel_positions, block_size, rounding_mode="floor")
    valid_block_index = (
        valid_pos
        & (block_indices >= 0)
        & (block_indices < block_table.shape[1])
    )
    safe_block_indices = block_indices.clamp(0, block_table.shape[1] - 1).to(
        torch.long
    )

    req_indices = token_to_req_indices[:num_actual].to(torch.long).clamp(
        0, block_table.shape[0] - 1
    )
    block_numbers = block_table[req_indices.unsqueeze(-1), safe_block_indices].to(
        torch.long
    )
    valid_blocks = (
        valid_block_index
        & (block_numbers >= 0)
        & (block_numbers < state_cache.shape[0])
    )
    safe_block_numbers = block_numbers.clamp(0, state_cache.shape[0] - 1)
    block_offsets = rel_positions.remainder(block_size).to(torch.long)

    flat_indices = safe_block_numbers * block_size + block_offsets
    flat_state = state_cache.reshape(
        state_cache.shape[0] * state_cache.shape[1],
        state_cache.shape[2],
    )
    gathered = flat_state.index_select(0, flat_indices.reshape(-1)).reshape(
        num_actual,
        window,
        state_cache.shape[-1],
    )

    head_offsets = (
        (offsets >= module.compress_ratio).to(torch.long) * module.head_dim
    )
    head_cols = head_offsets.unsqueeze(-1) + torch.arange(
        module.head_dim,
        device=device,
        dtype=torch.long,
    )
    head_cols = head_cols.unsqueeze(0).expand(num_actual, -1, -1)
    state_width = state_cache.shape[-1] // 2

    kv_rows = torch.gather(gathered, 2, head_cols)
    score_rows = torch.gather(gathered, 2, head_cols + state_width)
    valid_expanded = valid_blocks.unsqueeze(-1)
    kv_rows = torch.where(valid_expanded, kv_rows, torch.zeros_like(kv_rows))
    score_rows = torch.where(
        valid_expanded,
        score_rows,
        torch.full_like(score_rows, -1.0e20),
    )
    weights = torch.softmax(score_rows, dim=1)
    compressed = (kv_rows * weights).sum(dim=1)
    has_rows = valid_blocks.any(dim=1)
    return compressed, has_rows


def _musa_deepseek_v4_cache_u8_2d(kv_cache: torch.Tensor) -> torch.Tensor:
    return kv_cache.view(torch.uint8).reshape(kv_cache.shape[0], -1)


def _musa_deepseek_v4_write_cache_rows_capture(
    cache_u8: torch.Tensor,
    block_idx: torch.Tensor,
    cols: torch.Tensor,
    values: torch.Tensor,
    write_mask: torch.Tensor,
) -> None:
    rows = block_idx.unsqueeze(-1).expand_as(cols)
    existing = cache_u8[rows, cols]
    cache_u8[rows, cols] = torch.where(write_mask.unsqueeze(-1), values, existing)


def _musa_deepseek_v4_store_sparse_kv_capture(
    normed: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_slots: torch.Tensor,
    kv_cache_block_size: int,
    rope_head_dim: int,
    write_mask: torch.Tensor,
) -> None:
    if rope_head_dim == 64 and _musa_deepseek_v4_native_sparse_store_enabled():
        return _musa_deepseek_v4_native_store_sparse_kv(
            normed, kv_cache, kv_slots, kv_cache_block_size, write_mask
        )
    fp8_dim = normed.shape[-1] - rope_head_dim
    quant_block = 64
    token_stride = fp8_dim + rope_head_dim * 2
    scale_dim = fp8_dim // quant_block + 1
    fp8_max = 448.0

    cache_u8 = _musa_deepseek_v4_cache_u8_2d(kv_cache)
    cache_capacity = kv_cache.shape[0] * kv_cache_block_size
    slots = kv_slots.to(torch.long)
    valid_slots = (slots >= 0) & (slots < cache_capacity)
    safe_slots = slots.clamp(0, cache_capacity - 1)
    block_idx = torch.div(safe_slots, kv_cache_block_size, rounding_mode="floor")
    pos_in_block = safe_slots.remainder(kv_cache_block_size)
    final_write_mask = write_mask & valid_slots

    quant_input = normed[:, :fp8_dim].to(torch.bfloat16).to(torch.float32)
    chunks = quant_input.reshape(normed.shape[0], fp8_dim // quant_block, quant_block)
    amax = torch.maximum(
        chunks.abs().amax(dim=-1),
        torch.full_like(chunks[..., 0], 1.0e-4, dtype=torch.float32),
    )
    exponent = torch.ceil(torch.log2(amax / fp8_max))
    inv_scale = torch.exp2(-exponent).unsqueeze(-1)
    qbytes = (
        (chunks * inv_scale)
        .clamp(-fp8_max, fp8_max)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .reshape(normed.shape[0], fp8_dim)
    )
    rope_bytes = (
        normed[:, fp8_dim : fp8_dim + rope_head_dim]
        .to(torch.bfloat16)
        .contiguous()
        .view(torch.uint8)
        .reshape(normed.shape[0], rope_head_dim * 2)
    )
    value_bytes = torch.cat([qbytes, rope_bytes], dim=-1)
    value_cols = pos_in_block.unsqueeze(-1) * token_stride + torch.arange(
        token_stride,
        device=normed.device,
        dtype=torch.long,
    )
    _musa_deepseek_v4_write_cache_rows_capture(
        cache_u8, block_idx, value_cols, value_bytes, final_write_mask
    )

    scale_values = torch.cat(
        [
            (exponent + 127.0).clamp(0, 255).to(torch.uint8),
            torch.zeros((normed.shape[0], 1), device=normed.device, dtype=torch.uint8),
        ],
        dim=-1,
    )
    scale_cols = (
        kv_cache_block_size * token_stride
        + pos_in_block.unsqueeze(-1) * scale_dim
        + torch.arange(scale_dim, device=normed.device, dtype=torch.long)
    )
    _musa_deepseek_v4_write_cache_rows_capture(
        cache_u8, block_idx, scale_cols, scale_values, final_write_mask
    )


def _musa_deepseek_v4_store_indexer_fp8_capture(
    normed: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_slots: torch.Tensor,
    kv_cache_block_size: int,
    write_mask: torch.Tensor,
) -> None:
    head_dim = normed.shape[-1]
    scale_dim = 4
    fp8_max = 448.0
    cache_u8 = _musa_deepseek_v4_cache_u8_2d(kv_cache)
    cache_capacity = kv_cache.shape[0] * kv_cache_block_size
    slots = kv_slots.to(torch.long)
    valid_slots = (slots >= 0) & (slots < cache_capacity)
    safe_slots = slots.clamp(0, cache_capacity - 1)
    block_idx = torch.div(safe_slots, kv_cache_block_size, rounding_mode="floor")
    pos_in_block = safe_slots.remainder(kv_cache_block_size)
    final_write_mask = write_mask & valid_slots

    quant_input = normed.to(torch.bfloat16).to(torch.float32)
    amax = torch.maximum(
        quant_input.abs().amax(dim=-1),
        torch.full((normed.shape[0],), 1.0e-4, dtype=torch.float32, device=normed.device),
    )
    exponent = torch.ceil(torch.log2(amax / fp8_max))
    scale = torch.exp2(exponent)
    qbytes = (
        (quant_input / scale.unsqueeze(-1))
        .clamp(-fp8_max, fp8_max)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .reshape(normed.shape[0], head_dim)
    )
    value_cols = pos_in_block.unsqueeze(-1) * head_dim + torch.arange(
        head_dim,
        device=normed.device,
        dtype=torch.long,
    )
    _musa_deepseek_v4_write_cache_rows_capture(
        cache_u8, block_idx, value_cols, qbytes, final_write_mask
    )

    scale_bytes = scale.to(torch.float32).contiguous().view(torch.uint8).reshape(
        normed.shape[0], scale_dim
    )
    scale_cols = (
        kv_cache_block_size * head_dim
        + pos_in_block.unsqueeze(-1) * scale_dim
        + torch.arange(scale_dim, device=normed.device, dtype=torch.long)
    )
    _musa_deepseek_v4_write_cache_rows_capture(
        cache_u8, block_idx, scale_cols, scale_bytes, final_write_mask
    )


def _musa_deepseek_v4_store_indexer_mxfp4_capture(
    normed: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_slots: torch.Tensor,
    kv_cache_block_size: int,
    write_mask: torch.Tensor,
) -> None:
    head_dim = normed.shape[-1]
    quant_block = MXFP4_BLOCK_SIZE
    token_stride = head_dim // 2
    scale_dim = head_dim // quant_block
    cache_u8 = _musa_deepseek_v4_cache_u8_2d(kv_cache)
    cache_capacity = kv_cache.shape[0] * kv_cache_block_size
    slots = kv_slots.to(torch.long)
    valid_slots = (slots >= 0) & (slots < cache_capacity)
    safe_slots = slots.clamp(0, cache_capacity - 1)
    block_idx = torch.div(safe_slots, kv_cache_block_size, rounding_mode="floor")
    pos_in_block = safe_slots.remainder(kv_cache_block_size)
    final_write_mask = write_mask & valid_slots

    x = normed.to(torch.bfloat16).to(torch.float32).reshape(
        normed.shape[0], scale_dim, quant_block
    )
    even = x[..., 0::2]
    odd = x[..., 1::2]
    amax = torch.maximum(even.abs().amax(dim=-1), odd.abs().amax(dim=-1))
    amax = torch.maximum(amax, torch.full_like(amax, 1.0e-4, dtype=torch.float32))
    exponent = torch.ceil(torch.log2(amax / 6.0)).clamp(-127.0, 127.0)
    inv_scale = torch.exp2(-exponent).unsqueeze(-1)
    lo = _musa_deepseek_v4_e2m1_nibble(even * inv_scale)
    hi = _musa_deepseek_v4_e2m1_nibble(odd * inv_scale)
    packed = (lo | (hi << 4)).reshape(normed.shape[0], token_stride)
    value_cols = pos_in_block.unsqueeze(-1) * token_stride + torch.arange(
        token_stride,
        device=normed.device,
        dtype=torch.long,
    )
    _musa_deepseek_v4_write_cache_rows_capture(
        cache_u8, block_idx, value_cols, packed, final_write_mask
    )

    scale_values = (exponent + 127.0).to(torch.uint8)
    scale_cols = (
        kv_cache_block_size * token_stride
        + pos_in_block.unsqueeze(-1) * scale_dim
        + torch.arange(scale_dim, device=normed.device, dtype=torch.long)
    )
    _musa_deepseek_v4_write_cache_rows_capture(
        cache_u8, block_idx, scale_cols, scale_values, final_write_mask
    )


def _musa_deepseek_v4_compressor_forward_capture(
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
    if token_to_req_indices is None:
        return
    slot_mapping = state_metadata.slot_mapping
    num_actual = min(slot_mapping.shape[0], num_tokens)
    block_table = state_metadata.block_table
    block_size = state_metadata.block_size
    state_cache = module.state_cache.kv_cache

    valid_state_slots = _musa_deepseek_v4_save_partial_states_capture(
        kv,
        score,
        positions,
        module.ape,
        state_cache,
        slot_mapping,
        block_size,
        module.compress_ratio,
    )
    compressed, has_rows = _musa_deepseek_v4_gather_compressed_states_capture(
        module,
        positions,
        token_to_req_indices,
        block_table,
        state_cache,
        block_size,
        num_actual,
    )
    variance = compressed.pow(2).mean(dim=-1, keepdim=True)
    normed = (
        compressed
        * torch.rsqrt(variance + module.rms_norm_eps)
        * module.norm.weight.to(torch.float32)
    )
    normed = _musa_deepseek_v4_apply_gptj_rope(
        normed,
        positions[:num_actual],
        module.compress_ratio,
        module.rope_head_dim,
        rotary_emb.cos_sin_cache,
    )

    k_cache_metadata = cast(Any, attn_metadata[module.k_cache_prefix])
    kv_cache = module._static_forward_context[module.k_cache_prefix].kv_cache
    kv_slot_mapping = k_cache_metadata.slot_mapping[:num_actual]
    compress_boundary = (
        (positions[:num_actual] + 1).remainder(module.compress_ratio) == 0
    )
    write_mask = valid_state_slots & compress_boundary & has_rows

    if module.head_dim == 512:
        _musa_deepseek_v4_store_sparse_kv_capture(
            normed,
            kv_cache,
            kv_slot_mapping,
            kv_cache.shape[1],
            module.rope_head_dim,
            write_mask,
        )
    elif module.use_fp4_cache:
        _musa_deepseek_v4_store_indexer_mxfp4_capture(
            normed,
            kv_cache,
            kv_slot_mapping,
            kv_cache.shape[1],
            write_mask,
        )
    else:
        _musa_deepseek_v4_store_indexer_fp8_capture(
            normed,
            kv_cache,
            kv_slot_mapping,
            kv_cache.shape[1],
            write_mask,
        )


def _musa_deepseek_v4_compressor_forward_triton(
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
    if token_to_req_indices is None:
        return
    slot_mapping = state_metadata.slot_mapping
    num_actual = min(slot_mapping.shape[0], num_tokens)
    block_table = state_metadata.block_table
    block_size = state_metadata.block_size

    state_cache = module.state_cache.kv_cache
    state_width = state_cache.shape[-1] // 2

    _save_partial_states_kernel[(num_actual,)](
        kv,
        kv.stride(0),
        score,
        score.stride(0),
        module.ape,
        module.ape.stride(0),
        positions,
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        slot_mapping,
        block_size,
        HEAD_SIZE=kv.shape[-1],
        TRITON_BLOCK_SIZE=triton.next_power_of_2(kv.shape[-1]),
        STATE_WIDTH=state_width,
        COMPRESS_RATIO=module.compress_ratio,
    )

    cos_sin_cache = rotary_emb.cos_sin_cache
    k_cache_metadata = cast(Any, attn_metadata[module.k_cache_prefix])
    kv_cache = module._static_forward_context[module.k_cache_prefix].kv_cache

    module._fused_kernel[(num_actual,)](
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        block_table.stride(0),
        block_size,
        module.norm.weight,
        module.rms_norm_eps,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        kv_cache,
        k_cache_metadata.slot_mapping,
        kv_cache.shape[1],
        HEAD_SIZE=module.head_dim,
        TRITON_BLOCK_SIZE=triton.next_power_of_2(module.head_dim),
        STATE_WIDTH=state_width,
        COMPRESS_RATIO=module.compress_ratio,
        OVERLAP=module.overlap,
        ROPE_HEAD_DIM=module.rope_head_dim,
        FP8_MAX=448.0,
        QUANT_BLOCK=module._quant_block,
        TOKEN_STRIDE=module._token_stride,
        SCALE_DIM=module._scale_dim,
        KV_BLOCK_STRIDE=kv_cache.stride(0),
        num_warps=module._num_warps,
    )


def _musa_deepseek_v4_compressor_forward(
    module: "DeepseekCompressor",
    x: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb,
) -> None:
    if _musa_deepseek_v4_triton_compressor_enabled():
        return _musa_deepseek_v4_compressor_forward_triton(
            module, x, positions, rotary_emb
        )
    if (
        _musa_deepseek_v4_is_current_stream_capturing()
        or _musa_deepseek_v4_vectorized_eager_compressor_enabled()
    ):
        return _musa_deepseek_v4_compressor_forward_capture(
            module, x, positions, rotary_emb
        )
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
    native_sparse_store = (
        module.head_dim == 512
        and module.rope_head_dim == 64
        and _musa_deepseek_v4_native_sparse_store_enabled()
    )
    native_sparse_normed_rows: list[torch.Tensor] = []
    native_sparse_slots: list[torch.Tensor] = []

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
            if native_sparse_store:
                native_sparse_normed_rows.append(normed)
                native_sparse_slots.append(kv_slot_idx.to(torch.long))
            else:
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

    if native_sparse_store and native_sparse_normed_rows:
        native_normed = torch.stack(native_sparse_normed_rows, dim=0)
        native_slots = torch.stack(native_sparse_slots, dim=0)
        native_mask = torch.ones(
            (native_normed.shape[0],),
            dtype=torch.bool,
            device=native_normed.device,
        )
        _musa_deepseek_v4_native_store_sparse_kv(
            native_normed,
            kv_cache,
            native_slots,
            kv_cache.shape[1],
            native_mask,
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
            # Do not log from forward: this path may run while TorchDynamo is
            # active, where logger calls cause graph breaks.
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
            # Do not log from forward: this path may run while TorchDynamo is
            # active, where logger calls cause graph breaks.
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
        """        STATE_WIDTH=state_width,
        COMPRESS_RATIO=module.compress_ratio,
        launch_pdl=False,
""",
        """        STATE_WIDTH=state_width,
        COMPRESS_RATIO=module.compress_ratio,
        # launch_pdl omitted: MUSA Triton rejects this kwarg.
""",
    ),
    (
        """        STATE_WIDTH=state_width,
        COMPRESS_RATIO=self.compress_ratio,
        launch_pdl=False,
""",
        """        STATE_WIDTH=state_width,
        COMPRESS_RATIO=self.compress_ratio,
        # launch_pdl omitted: MUSA Triton rejects this kwarg.
""",
    ),
    (
        """        KV_BLOCK_STRIDE=kv_cache.stride(0),
        num_warps=module._num_warps,
        launch_pdl=False,
""",
        """        KV_BLOCK_STRIDE=kv_cache.stride(0),
        num_warps=module._num_warps,
        # launch_pdl omitted: MUSA Triton rejects this kwarg.
""",
    ),
    (
        """        KV_BLOCK_STRIDE=kv_cache.stride(0),
        num_warps=self._num_warps,
        launch_pdl=False,
""",
        """        KV_BLOCK_STRIDE=kv_cache.stride(0),
        num_warps=self._num_warps,
        # launch_pdl omitted: MUSA Triton rejects this kwarg.
""",
    ),
    (
        """def _musa_deepseek_v4_cache_u8_2d(kv_cache: torch.Tensor) -> torch.Tensor:
    return kv_cache.view(torch.uint8).reshape(kv_cache.shape[0], -1)
""",
        """def _musa_deepseek_v4_cache_u8_2d(kv_cache: torch.Tensor) -> torch.Tensor:
    cache_u8 = kv_cache.view(torch.uint8)
    row_stride = int(kv_cache.stride(0)) * kv_cache.element_size()
    if cache_u8.stride(-1) != 1:
        raise RuntimeError(
            "DeepSeek-V4 MUSA cache writer expects byte-contiguous cache rows, "
            f"got shape={tuple(kv_cache.shape)} stride={tuple(kv_cache.stride())}"
        )
    return cache_u8.as_strided((kv_cache.shape[0], row_stride), (row_stride, 1))
""",
    ),
]

RELOAD_AFTER_PATCH = True
