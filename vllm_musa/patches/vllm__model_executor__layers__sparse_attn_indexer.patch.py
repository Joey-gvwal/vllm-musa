# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch sparse-attention indexer with an opt-in MUSA correctness fallback.
"""

import ast

_PREFILL_NATIVE_HELPER = """

def _musa_try_fill_prefill_topk_from_indexer_cache_native(
    q_quant: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    chunk,
    topk_indices: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
) -> bool:
    if (
        not _musa_sparse_indexer_native_decode_enabled()
        or head_dim != 128
        or topk_tokens > 512
        or q_quant.dtype != torch.float8_e4m3fn
        or kv_cache.dtype != torch.uint8
        or weights.dtype != torch.float32
        or int(chunk.total_seq_lens) > 4096
    ):
        return False

    rows = min(q_quant.shape[0], chunk.cu_seqlen_ks.numel(), topk_indices.shape[0])
    topk = min(int(topk_tokens), topk_indices.shape[1])
    if rows <= 0 or topk <= 0:
        return True

    _musa_custom_ops.deepseek_v4_indexer_topk_prefill(
        q_quant[:rows],
        kv_cache,
        weights[:rows],
        chunk.block_table,
        chunk.cu_seq_lens,
        chunk.token_to_seq,
        chunk.cu_seqlen_ks[:rows],
        chunk.cu_seqlen_ke[:rows],
        topk_indices[:rows, :topk],
        topk,
    )
    return True
"""


def _remove_shadowed_exact_fill_definitions(source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    lines = source.splitlines(keepends=True)
    exact_defs = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_musa_fill_exact_sparse_indexer_indices"
        and node.end_lineno is not None
    ]
    if len(exact_defs) <= 1:
        return source

    def function_source(node: ast.FunctionDef) -> str:
        return "".join(lines[node.lineno - 1 : node.end_lineno])

    native_defs = [
        node
        for node in exact_defs
        if "_musa_try_fill_prefill_topk_from_indexer_cache_native("
        in function_source(node)
    ]
    if not native_defs:
        return source

    keep = native_defs[-1]
    remove_ranges = [
        (node.lineno - 1, node.end_lineno) for node in exact_defs if node is not keep
    ]
    for start, end in sorted(remove_ranges, reverse=True):
        del lines[start:end]

    return "".join(lines)


def normalize_source(source: str) -> str:
    """Remove stale duplicate MUSA helper blocks from previously patched files."""
    helper_start = "\ndef _musa_sparse_indexer_is_current_stream_capturing() -> bool:\n"
    main_entry = "\ndef _musa_fill_exact_sparse_indexer_indices(\n"
    stale_forward = """        elif (
            current_platform.is_musa()
            and os.getenv(
                "VLLM_MUSA_ENABLE_TORCH_SPARSE_ATTN_INDEXER_FALLBACK",
                "0",
            )
            == "1"
        ):
            try:
                return _musa_fill_exact_sparse_indexer_indices(
                    hidden_states,
                    self.k_cache.prefix,
                    self.k_cache.kv_cache,
                    q_quant,
                    weights,
                    self.topk_tokens,
                    self.head_dim,
                    self.topk_indices_buffer,
                    self.use_fp4_cache,
                )
            except NotImplementedError:
                return _musa_fill_recent_sparse_indexer_indices(
                    hidden_states,
                    self.k_cache.prefix,
                    self.topk_tokens,
                    self.topk_indices_buffer,
                )
"""
    native_forward = """        elif (
            current_platform.is_musa()
            and (
                os.getenv(
                    "VLLM_MUSA_ENABLE_DEEPSEEK_V4_SPARSE_INDEXER_MUSA_IMPL",
                    "1",
                )
                == "1"
                or os.getenv(
                    "VLLM_MUSA_ENABLE_TORCH_SPARSE_ATTN_INDEXER_FALLBACK",
                    "0",
                )
                == "1"
            )
        ):
            try:
                return _musa_fill_exact_sparse_indexer_indices(
                    hidden_states,
                    self.k_cache.prefix,
                    self.k_cache.kv_cache,
                    q_quant,
                    weights,
                    self.topk_tokens,
                    self.head_dim,
                    self.topk_indices_buffer,
                    self.use_fp4_cache,
                )
            except NotImplementedError:
                return _musa_fill_recent_sparse_indexer_indices(
                    hidden_states,
                    self.k_cache.prefix,
                    self.topk_tokens,
                    self.topk_indices_buffer,
                )
"""
    source = source.replace(stale_forward, native_forward)

    first_start = source.find(helper_start)
    if first_start >= 0:
        search_from = first_start + len(helper_start)
        while True:
            duplicate_start = source.find(helper_start, search_from)
            if duplicate_start < 0:
                break

            duplicate_end = source.find(main_entry, duplicate_start)
            if duplicate_end < 0:
                break

            source = source[:duplicate_start] + source[duplicate_end:]
            search_from = first_start + len(helper_start)

    for default in ("0", "1"):
        source = source.replace(
            f"""def _musa_sparse_indexer_graph_exact_decode_enabled() -> bool:
    return os.getenv("VLLM_MUSA_SPARSE_INDEXER_GRAPH_EXACT_DECODE", "{default}") == "1"
""",
            """def _musa_sparse_indexer_graph_exact_decode_enabled() -> bool:
    return os.getenv("VLLM_MUSA_SPARSE_INDEXER_GRAPH_EXACT_DECODE", "0") == "1"
""",
        )

    if (
        "_musa_try_fill_prefill_topk_from_indexer_cache_native(" in source
        and "def _musa_try_fill_prefill_topk_from_indexer_cache_native" not in source
    ):
        source = source.replace(
            "\n\ndef _musa_indexer_cache_block(kv_cache: torch.Tensor, block_id: int)"
            " -> torch.Tensor:\n",
            _PREFILL_NATIVE_HELPER
            + "\n\ndef _musa_indexer_cache_block(kv_cache: torch.Tensor, block_id: int)"
            " -> torch.Tensor:\n",
        )

    source = _remove_shadowed_exact_fill_definitions(source)

    return source


PATCHES = [
    (
        """import torch

import vllm.envs as envs
""",
        """import os

import torch

import vllm.envs as envs
from vllm_musa import _custom_ops as _musa_custom_ops
""",
    ),
    (
        """def _musa_sparse_indexer_is_current_stream_capturing() -> bool:
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
""",
        """def _musa_sparse_indexer_is_current_stream_capturing() -> bool:
    for module_name in ("musa", "cuda"):
        module = getattr(torch, module_name, None)
        if module is None:
            continue
        is_capturing = getattr(module, "is_current_stream_capturing", None)
        if is_capturing is None:
            continue
        try:
            return bool(is_capturing())
        except Exception:
            continue
    return False
""",
    ),
    (
        """logger = init_logger(__name__)
""",
        """logger = init_logger(__name__)


def _musa_sparse_indexer_is_current_stream_capturing() -> bool:
    for module_name in ("musa", "cuda"):
        module = getattr(torch, module_name, None)
        if module is None:
            continue
        is_capturing = getattr(module, "is_current_stream_capturing", None)
        if is_capturing is None:
            continue
        try:
            return bool(is_capturing())
        except Exception:
            continue
    return False


def _musa_fill_recent_rows_capture(
    topk_indices_buffer: torch.Tensor,
    row_start: int,
    lengths: torch.Tensor,
    cap: int,
) -> None:
    flat_lengths = lengths.reshape(-1).to(torch.long)
    rows = min(
        flat_lengths.numel(),
        topk_indices_buffer.shape[0] - row_start,
    )
    if rows <= 0 or cap <= 0:
        return
    row_lengths = flat_lengths[:rows].clamp(min=0)
    counts = row_lengths.clamp(max=cap)
    starts = (row_lengths - counts).clamp(min=0)
    offsets = torch.arange(cap, device=topk_indices_buffer.device, dtype=torch.long)
    values = starts.unsqueeze(-1) + offsets.unsqueeze(0)
    valid = offsets.unsqueeze(0) < counts.unsqueeze(-1)
    rows_view = topk_indices_buffer[row_start : row_start + rows, :cap]
    rows_view.copy_(
        torch.where(
            valid,
            values.to(topk_indices_buffer.dtype),
            rows_view,
        )
    )


def _musa_fill_recent_sparse_indexer_indices_capture(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    topk_tokens: int,
    topk_indices_buffer: torch.Tensor,
) -> torch.Tensor:
    topk = min(topk_tokens, topk_indices_buffer.shape[-1])
    topk_indices_buffer[: hidden_states.shape[0], :topk] = -1
    if topk <= 0:
        return topk_indices_buffer

    cap = int(os.getenv("VLLM_MUSA_SPARSE_INDEXER_FALLBACK_TOPK", "16"))
    cap = max(0, min(cap, topk))
    if cap == 0:
        return topk_indices_buffer

    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        return topk_indices_buffer

    metadata = attn_metadata.get(_resolve_layer_name(k_cache_prefix))
    if not isinstance(metadata, DeepseekV32IndexerMetadata):
        return topk_indices_buffer

    if metadata.num_decodes > 0 and metadata.decode is not None:
        _musa_fill_recent_rows_capture(
            topk_indices_buffer,
            0,
            metadata.decode.seq_lens,
            cap,
        )

    if metadata.num_prefills > 0 and metadata.prefill is not None:
        for chunk in metadata.prefill.chunks:
            _musa_fill_recent_rows_capture(
                topk_indices_buffer,
                int(chunk.token_start),
                chunk.cu_seqlen_ke - chunk.cu_seqlen_ks,
                cap,
            )

    return topk_indices_buffer


def _musa_fill_recent_sparse_indexer_indices(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    topk_tokens: int,
    topk_indices_buffer: torch.Tensor,
) -> torch.Tensor:
    if _musa_sparse_indexer_is_current_stream_capturing():
        return _musa_fill_recent_sparse_indexer_indices_capture(
            hidden_states,
            k_cache_prefix,
            topk_tokens,
            topk_indices_buffer,
        )

    # Diagnostic fallback: preserve sparse-attention control flow by selecting a
    # small recent compressed-token window when the learned indexer kernels are
    # unavailable on MUSA.
    topk = min(topk_tokens, topk_indices_buffer.shape[-1])
    topk_indices_buffer[: hidden_states.shape[0], :topk] = -1
    if topk <= 0:
        return topk_indices_buffer

    cap = int(os.getenv("VLLM_MUSA_SPARSE_INDEXER_FALLBACK_TOPK", "16"))
    cap = max(0, min(cap, topk))
    if cap == 0:
        return topk_indices_buffer

    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        return topk_indices_buffer

    metadata = attn_metadata.get(_resolve_layer_name(k_cache_prefix))
    if not isinstance(metadata, DeepseekV32IndexerMetadata):
        return topk_indices_buffer

    def fill_rows(row_start: int, lengths: torch.Tensor) -> None:
        flat_lengths = lengths.reshape(-1).to(torch.long)
        rows = min(
            flat_lengths.numel(),
            hidden_states.shape[0] - row_start,
            topk_indices_buffer.shape[0] - row_start,
        )
        for row_offset in range(rows):
            length = int(flat_lengths[row_offset].item())
            count = min(cap, max(length, 0))
            if count <= 0:
                continue
            start = length - count
            row = row_start + row_offset
            topk_indices_buffer[row, :count] = torch.arange(
                start,
                length,
                device=topk_indices_buffer.device,
                dtype=topk_indices_buffer.dtype,
            )

    if metadata.num_decodes > 0 and metadata.decode is not None:
        fill_rows(0, metadata.decode.seq_lens)

    if metadata.num_prefills > 0 and metadata.prefill is not None:
        for chunk in metadata.prefill.chunks:
            fill_rows(
                int(chunk.token_start),
                chunk.cu_seqlen_ke - chunk.cu_seqlen_ks,
            )

    return topk_indices_buffer


def _musa_sparse_indexer_graph_exact_decode_enabled() -> bool:
    # The exact graph decode path is still diagnostic: it removes the recent
    # window approximation, but regresses DeepSeek-V4-Flash-Base graph+MTP TPS
    # at long context until the exact provider is fully optimized.
    return os.getenv("VLLM_MUSA_SPARSE_INDEXER_GRAPH_EXACT_DECODE", "0") == "1"


def _musa_sparse_indexer_native_decode_enabled() -> bool:
    return os.getenv("VLLM_MUSA_DEEPSEEK_V4_INDEXER_TOPK_NATIVE", "1") == "1"


def _musa_decode_block_table_for_token_rows(
    block_table: torch.Tensor,
    rows: int,
) -> torch.Tensor | None:
    block_rows = block_table.shape[0]
    if rows <= 0:
        return block_table[:0]
    if block_rows == rows:
        return block_table[:rows]
    if block_rows <= 0 or rows % block_rows != 0:
        return None
    return block_table.repeat_interleave(rows // block_rows, dim=0)


def _musa_try_fill_decode_topk_from_indexer_cache_native(
    q_quant: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    decode_metadata,
    topk_indices_buffer: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
) -> bool:
    if (
        not _musa_sparse_indexer_native_decode_enabled()
        or head_dim != 128
        or topk_tokens > 512
        or q_quant.dtype != torch.float8_e4m3fn
        or weights.dtype != torch.float32
        or decode_metadata.block_table.shape[1] * kv_cache.shape[1] > 4096
    ):
        return False

    seq_lens = decode_metadata.seq_lens.reshape(-1)
    rows = min(q_quant.shape[0], seq_lens.numel(), topk_indices_buffer.shape[0])
    topk = min(int(topk_tokens), topk_indices_buffer.shape[1])
    if rows <= 0 or topk <= 0:
        return True

    block_table = _musa_decode_block_table_for_token_rows(
        decode_metadata.block_table,
        rows,
    )
    if block_table is None:
        return False

    _musa_custom_ops.deepseek_v4_indexer_topk_decode(
        q_quant[:rows],
        kv_cache,
        weights[:rows],
        seq_lens[:rows],
        block_table,
        topk_indices_buffer[:rows, :topk],
        topk,
    )
    return True


def _musa_try_fill_prefill_topk_from_indexer_cache_native(
    q_quant: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    chunk,
    topk_indices: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
) -> bool:
    if (
        not _musa_sparse_indexer_native_decode_enabled()
        or head_dim != 128
        or topk_tokens > 512
        or q_quant.dtype != torch.float8_e4m3fn
        or kv_cache.dtype != torch.uint8
        or weights.dtype != torch.float32
        or int(chunk.total_seq_lens) > 4096
    ):
        return False

    rows = min(q_quant.shape[0], chunk.cu_seqlen_ks.numel(), topk_indices.shape[0])
    topk = min(int(topk_tokens), topk_indices.shape[1])
    if rows <= 0 or topk <= 0:
        return True

    _musa_custom_ops.deepseek_v4_indexer_topk_prefill(
        q_quant[:rows],
        kv_cache,
        weights[:rows],
        chunk.block_table,
        chunk.cu_seq_lens,
        chunk.token_to_seq,
        chunk.cu_seqlen_ks[:rows],
        chunk.cu_seqlen_ke[:rows],
        topk_indices[:rows, :topk],
        topk,
    )
    return True


def _musa_indexer_cache_block(kv_cache: torch.Tensor, block_id: int) -> torch.Tensor:
    return kv_cache[block_id].view(torch.uint8).flatten()


def _musa_indexer_cache_rows(kv_cache: torch.Tensor) -> torch.Tensor:
    return kv_cache.as_strided(
        (kv_cache.shape[0], kv_cache.stride(0)),
        (kv_cache.stride(0), 1),
    )


def _musa_dequant_indexer_fp8_cache_row(
    kv_cache: torch.Tensor,
    block_id: int,
    pos_in_block: int,
    head_dim: int,
) -> torch.Tensor:
    block_size = kv_cache.shape[1]
    scale_dim = 4
    cache_block = _musa_indexer_cache_block(kv_cache, block_id)
    token_base = pos_in_block * head_dim
    scale_base = block_size * head_dim + pos_in_block * scale_dim
    values = (
        cache_block[token_base : token_base + head_dim]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .to(torch.float32)
    )
    scale = (
        cache_block[scale_base : scale_base + scale_dim]
        .contiguous()
        .view(torch.float32)
    )
    return values * scale


def _musa_dequant_indexer_fp8_cache_rows(
    kv_cache: torch.Tensor,
    block_ids: torch.Tensor,
    pos_in_block: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    if block_ids.numel() == 0:
        return torch.empty((0, head_dim), dtype=torch.float32, device=kv_cache.device)
    block_size = kv_cache.shape[1]
    block_ids = block_ids.to(torch.long)
    pos_in_block = pos_in_block.to(torch.long)
    valid = (
        (block_ids >= 0)
        & (block_ids < kv_cache.shape[0])
        & (pos_in_block >= 0)
        & (pos_in_block < block_size)
    )
    safe_blocks = block_ids.clamp(0, kv_cache.shape[0] - 1)
    safe_pos = pos_in_block.clamp(0, block_size - 1)
    selected_blocks = _musa_indexer_cache_rows(kv_cache).index_select(0, safe_blocks)
    value_offsets = safe_pos.unsqueeze(-1) * head_dim + torch.arange(
        head_dim, device=kv_cache.device, dtype=torch.long
    )
    values = (
        torch.gather(selected_blocks, 1, value_offsets)
        .contiguous()
        .view(torch.float8_e4m3fn)
        .to(torch.float32)
    )
    scale_offsets = (
        block_size * head_dim
        + safe_pos.unsqueeze(-1) * 4
        + torch.arange(4, device=kv_cache.device, dtype=torch.long)
    )
    scales = (
        torch.gather(selected_blocks, 1, scale_offsets)
        .contiguous()
        .view(torch.float32)
        .reshape(-1, 1)
    )
    dequant = values * scales
    return torch.where(valid.unsqueeze(-1), dequant, torch.zeros_like(dequant))


def _musa_gather_indexer_fp8_cache(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    total_seq_lens = int(cu_seq_lens[-1].item())
    if total_seq_lens <= 0:
        return torch.empty((0, head_dim), dtype=torch.float32, device=kv_cache.device)
    block_size = kv_cache.shape[1]
    pieces = []
    for req_idx in range(block_table.shape[0]):
        start = int(cu_seq_lens[req_idx].item())
        end = int(cu_seq_lens[req_idx + 1].item())
        length = end - start
        if length <= 0:
            continue
        local_pos = torch.arange(length, device=kv_cache.device, dtype=torch.long)
        physical_blocks = block_table[
            req_idx,
            torch.div(local_pos, block_size, rounding_mode="floor").clamp(
                0, block_table.shape[1] - 1
            ),
        ]
        pieces.append(
            _musa_dequant_indexer_fp8_cache_rows(
                kv_cache,
                physical_blocks,
                local_pos.remainder(block_size),
                head_dim,
            )
        )
    if not pieces:
        return torch.empty((0, head_dim), dtype=torch.float32, device=kv_cache.device)
    return torch.cat(pieces, dim=0)


def _musa_sparse_indexer_logits(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    # q is FP8-dequantized without an explicit q scale; the FP8 q scale is
    # already folded into weights by fused_indexer_q_rope_quant.
    per_head = torch.einsum("h d, n d -> h n", q.to(torch.float32), k)
    per_head = per_head.clamp_min(0.0)
    return (per_head * weights.to(torch.float32).unsqueeze(-1)).sum(dim=0)


def _musa_fill_topk_rows_from_indexer_logits(
    q_deq: torch.Tensor,
    k_deq: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
) -> None:
    rows = min(q_deq.shape[0], topk_indices.shape[0], cu_seqlen_ks.numel())
    for row in range(rows):
        start = int(cu_seqlen_ks[row].item())
        end = int(cu_seqlen_ke[row].item())
        row_len = max(0, end - start)
        if row_len == 0:
            continue
        k_i = min(int(topk_tokens), row_len, topk_indices.shape[1])
        logits = _musa_sparse_indexer_logits(
            q_deq[row],
            k_deq[start:end],
            weights[row],
        )
        topk_indices[row, :k_i] = torch.topk(logits, k_i, dim=-1).indices.to(
            topk_indices.dtype
        )


def _musa_fill_decode_topk_from_indexer_cache(
    q_deq: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    decode_metadata,
    topk_indices_buffer: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
) -> None:
    seq_lens = decode_metadata.seq_lens.reshape(-1)
    rows = min(q_deq.shape[0], seq_lens.numel(), topk_indices_buffer.shape[0])
    block_size = kv_cache.shape[1]
    for row in range(rows):
        seq_len = int(seq_lens[row].item())
        if seq_len <= 0:
            continue
        block_row = min(row, decode_metadata.block_table.shape[0] - 1)
        local_pos = torch.arange(seq_len, device=kv_cache.device, dtype=torch.long)
        physical_blocks = decode_metadata.block_table[
            block_row,
            torch.div(local_pos, block_size, rounding_mode="floor").clamp(
                0, decode_metadata.block_table.shape[1] - 1
            ),
        ]
        gathered = _musa_dequant_indexer_fp8_cache_rows(
                kv_cache,
                physical_blocks,
                local_pos.remainder(block_size),
                head_dim,
        )
        k_i = min(int(topk_tokens), seq_len, topk_indices_buffer.shape[1])
        logits = _musa_sparse_indexer_logits(q_deq[row], gathered, weights[row])
        topk_indices_buffer[row, :k_i] = torch.topk(logits, k_i, dim=-1).indices.to(
            topk_indices_buffer.dtype
        )


def _musa_fill_decode_topk_from_indexer_cache_capture(
    q_deq: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    decode_metadata,
    topk_indices_buffer: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
) -> None:
    seq_lens = decode_metadata.seq_lens.reshape(-1).to(torch.long)
    rows = min(q_deq.shape[0], seq_lens.numel(), topk_indices_buffer.shape[0])
    topk = min(int(topk_tokens), topk_indices_buffer.shape[1])
    if rows <= 0 or topk <= 0:
        return

    block_table = _musa_decode_block_table_for_token_rows(
        decode_metadata.block_table,
        rows,
    )
    if block_table is None:
        return
    block_size = kv_cache.shape[1]
    max_positions = block_table.shape[1] * block_size
    topk = min(topk, max_positions)
    topk_indices_buffer[:rows, :topk] = -1
    if max_positions <= 0:
        return

    local_pos = torch.arange(max_positions, device=kv_cache.device, dtype=torch.long)
    block_cols = torch.div(local_pos, block_size, rounding_mode="floor").clamp(
        0, block_table.shape[1] - 1
    )
    physical_blocks = block_table[:, block_cols]
    pos_in_block = local_pos.remainder(block_size).expand(rows, -1)

    gathered = _musa_dequant_indexer_fp8_cache_rows(
        kv_cache,
        physical_blocks.reshape(-1),
        pos_in_block.reshape(-1),
        head_dim,
    ).view(rows, max_positions, head_dim)

    q_rows = q_deq[:rows].to(torch.float32)
    weight_rows = weights[:rows].to(torch.float32)
    per_head = torch.einsum("r h d, r n d -> r h n", q_rows, gathered)
    per_head = per_head.clamp_min(0.0)
    logits = (per_head * weight_rows.unsqueeze(-1)).sum(dim=1)

    valid = local_pos.unsqueeze(0) < seq_lens[:rows].unsqueeze(-1)
    logits = torch.where(
        valid,
        logits,
        torch.full((), float("-inf"), dtype=logits.dtype, device=logits.device),
    )
    indices = torch.topk(logits, topk, dim=-1).indices.to(topk_indices_buffer.dtype)
    valid_counts = seq_lens[:rows].clamp(min=0, max=topk).unsqueeze(-1)
    rank_offsets = torch.arange(topk, device=kv_cache.device, dtype=torch.long)
    indices = torch.where(
        rank_offsets.unsqueeze(0) < valid_counts,
        indices,
        torch.full_like(indices, -1),
    )
    topk_indices_buffer[:rows, :topk] = indices


def _musa_fill_exact_sparse_indexer_indices_capture(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
    topk_indices_buffer: torch.Tensor,
    use_fp4_cache: bool,
) -> torch.Tensor:
    # Avoid logger calls in fallback forward paths; these functions may be
    # reached while TorchDynamo/CUDA Graph capture is active.
    if use_fp4_cache or isinstance(q_quant, tuple):
        return _musa_fill_recent_sparse_indexer_indices(
            hidden_states,
            k_cache_prefix,
            topk_tokens,
            topk_indices_buffer,
        )

    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        return topk_indices_buffer

    metadata = attn_metadata.get(_resolve_layer_name(k_cache_prefix))
    if not isinstance(metadata, DeepseekV32IndexerMetadata):
        return topk_indices_buffer

    topk_indices_buffer[: hidden_states.shape[0]] = -1

    if metadata.num_decodes > 0 and metadata.decode is not None:
        if not _musa_try_fill_decode_topk_from_indexer_cache_native(
            q_quant[: metadata.num_decode_tokens],
            kv_cache,
            weights[: metadata.num_decode_tokens],
            metadata.decode,
            topk_indices_buffer[: metadata.num_decode_tokens, :topk_tokens],
            topk_tokens,
            head_dim,
        ):
            q_deq = q_quant.to(torch.float32)
            weights_fp32 = weights.to(torch.float32)
            _musa_fill_decode_topk_from_indexer_cache_capture(
                q_deq[: metadata.num_decode_tokens],
                kv_cache,
                weights_fp32[: metadata.num_decode_tokens],
                metadata.decode,
                topk_indices_buffer[: metadata.num_decode_tokens, :topk_tokens],
                topk_tokens,
                head_dim,
            )

    if metadata.num_prefills > 0:
        _musa_fill_recent_sparse_indexer_indices(
            hidden_states,
            k_cache_prefix,
            topk_tokens,
            topk_indices_buffer,
        )

    return topk_indices_buffer


def _musa_fill_exact_sparse_indexer_indices(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
    topk_indices_buffer: torch.Tensor,
    use_fp4_cache: bool,
) -> torch.Tensor:
    if _musa_sparse_indexer_is_current_stream_capturing():
        if _musa_sparse_indexer_graph_exact_decode_enabled():
            return _musa_fill_exact_sparse_indexer_indices_capture(
                hidden_states,
                k_cache_prefix,
                kv_cache,
                q_quant,
                weights,
                topk_tokens,
                head_dim,
                topk_indices_buffer,
                use_fp4_cache,
            )
        return _musa_fill_recent_sparse_indexer_indices(
            hidden_states,
            k_cache_prefix,
            topk_tokens,
            topk_indices_buffer,
        )

    if use_fp4_cache or isinstance(q_quant, tuple):
        raise NotImplementedError(
            "MUSA exact sparse-attention indexer fallback currently supports "
            "the FP8 indexer-cache path only."
        )

    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        return topk_indices_buffer

    metadata = attn_metadata.get(_resolve_layer_name(k_cache_prefix))
    if not isinstance(metadata, DeepseekV32IndexerMetadata):
        return topk_indices_buffer

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    q_deq = None
    weights_fp32 = None

    if metadata.num_prefills > 0 and metadata.prefill is not None:
        for chunk in metadata.prefill.chunks:
            token_start = int(chunk.token_start)
            token_end = int(chunk.token_end)
            if _musa_try_fill_prefill_topk_from_indexer_cache_native(
                q_quant[token_start:token_end],
                kv_cache,
                weights[token_start:token_end],
                chunk,
                topk_indices_buffer[token_start:token_end, :topk_tokens],
                topk_tokens,
                head_dim,
            ):
                continue
            if q_deq is None:
                q_deq = q_quant.to(torch.float32)
            if weights_fp32 is None:
                weights_fp32 = weights.to(torch.float32)
            k_deq = _musa_gather_indexer_fp8_cache(
                kv_cache,
                chunk.block_table,
                chunk.cu_seq_lens,
                head_dim,
            )
            _musa_fill_topk_rows_from_indexer_logits(
                q_deq[token_start:token_end],
                k_deq,
                weights_fp32[token_start:token_end],
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                topk_indices_buffer[token_start:token_end, :topk_tokens],
                topk_tokens,
            )

    if metadata.num_decodes > 0 and metadata.decode is not None:
        if not _musa_try_fill_decode_topk_from_indexer_cache_native(
            q_quant[: metadata.num_decode_tokens],
            kv_cache,
            weights[: metadata.num_decode_tokens],
            metadata.decode,
            topk_indices_buffer[: metadata.num_decode_tokens, :topk_tokens],
            topk_tokens,
            head_dim,
        ):
            if q_deq is None:
                q_deq = q_quant.to(torch.float32)
            if weights_fp32 is None:
                weights_fp32 = weights.to(torch.float32)
            _musa_fill_decode_topk_from_indexer_cache_capture(
                q_deq[: metadata.num_decode_tokens],
                kv_cache,
                weights_fp32[: metadata.num_decode_tokens],
                metadata.decode,
                topk_indices_buffer[: metadata.num_decode_tokens, :topk_tokens],
                topk_tokens,
                head_dim,
            )

    return topk_indices_buffer
""",
    ),
    (
        """        else:
            raise NotImplementedError(
                "SparseAttnIndexer native forward is only implemented for "
                "CUDA, ROCm and XPU platforms."
            )
""",
        """        elif (
            current_platform.is_musa()
            and (
                os.getenv(
                    "VLLM_MUSA_ENABLE_DEEPSEEK_V4_SPARSE_INDEXER_MUSA_IMPL",
                    "1",
                )
                == "1"
                or os.getenv(
                    "VLLM_MUSA_ENABLE_TORCH_SPARSE_ATTN_INDEXER_FALLBACK",
                    "0",
                )
                == "1"
            )
        ):
            try:
                return _musa_fill_exact_sparse_indexer_indices(
                    hidden_states,
                    self.k_cache.prefix,
                    self.k_cache.kv_cache,
                    q_quant,
                    weights,
                    self.topk_tokens,
                    self.head_dim,
                    self.topk_indices_buffer,
                    self.use_fp4_cache,
                )
            except NotImplementedError:
                return _musa_fill_recent_sparse_indexer_indices(
                    hidden_states,
                    self.k_cache.prefix,
                    self.topk_tokens,
                    self.topk_indices_buffer,
                )
        else:
            raise NotImplementedError(
                "SparseAttnIndexer native forward is only implemented for "
                "CUDA, ROCm and XPU platforms."
            )
""",
    ),
]

RELOAD_AFTER_PATCH = True
