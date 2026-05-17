# SPDX-License-Identifier: Apache-2.0
"""TileLang-backed DeepSeek-V4 sparse-indexer top-k helpers."""

from __future__ import annotations

import atexit
import json
import os
from collections import Counter

import torch

_MAX_TILELANG_TOPK_WIDTH = 2048
_ROUTER_TOPK_MODES = {
    "auto",
    "hash",
    "hash_tilelang",
    "tilelang",
    "jit",
    "warp",
    "warp_tilelang",
    "tilelang_warp",
    "fast",
}
_ROUTER_TOPK_DISABLE_MODES = {"", "0", "false", "off", "torch", "fallback"}
_ROUTER_TOPK_WARP_MODES = {"warp", "warp_tilelang", "tilelang_warp", "fast"}
_ROUTER_TOPK_HASH_ONLY_MODES = {"hash", "hash_tilelang"}
_ROUTER_TOPK_AUTO_DISABLED_REASON: str | None = None
_ROUTER_TOPK_TRACE_REGISTERED = False
_ROUTER_TOPK_TRACE_TOTAL = 0
_ROUTER_TOPK_TRACE_BUCKETS: Counter[str] = Counter()
_ROUTER_TOPK_TRACE_REASONS: Counter[str] = Counter()


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() not in {
        "",
        "0",
        "false",
        "off",
        "no",
    }


def _router_topk_trace_enabled() -> bool:
    return _env_enabled("VLLM_MUSA_DEEPSEEK_V4_ROUTER_TOPK_TRACE")


def _router_topk_trace_limit() -> int:
    raw_limit = os.environ.get(
        "VLLM_MUSA_DEEPSEEK_V4_ROUTER_TOPK_TRACE_LIMIT", "64"
    )
    try:
        return max(1, int(raw_limit))
    except ValueError:
        return 64


def _router_topk_trace_every() -> int:
    raw_every = os.environ.get(
        "VLLM_MUSA_DEEPSEEK_V4_ROUTER_TOPK_TRACE_EVERY", "0"
    )
    try:
        return max(0, int(raw_every))
    except ValueError:
        return 0


def _dtype_name(tensor: torch.Tensor | None) -> str:
    if tensor is None:
        return "none"
    return str(getattr(tensor, "dtype", "unknown")).replace("torch.", "")


def _shape_key(tensor: torch.Tensor | None) -> str:
    if tensor is None:
        return "none"
    try:
        return "x".join(str(int(dim)) for dim in tensor.shape)
    except Exception:
        return "unknown"


def _contiguous_key(tensor: torch.Tensor | None) -> str:
    if tensor is None:
        return "none"
    try:
        return "contig" if tensor.is_contiguous() else "noncontig"
    except Exception:
        return "unknown"


def _print_router_topk_trace_summary(event: str = "atexit") -> None:
    if _ROUTER_TOPK_TRACE_TOTAL == 0:
        return
    limit = _router_topk_trace_limit()
    payload = {
        "event": event,
        "pid": os.getpid(),
        "total_calls": _ROUTER_TOPK_TRACE_TOTAL,
        "top_buckets": _ROUTER_TOPK_TRACE_BUCKETS.most_common(limit),
        "top_reasons": _ROUTER_TOPK_TRACE_REASONS.most_common(limit),
    }
    print(
        "MUSA_ROUTER_TOPK_TRACE_SUMMARY "
        + json.dumps(payload, sort_keys=True),
        flush=True,
    )


def _record_router_topk_trace(
    *,
    mode: str,
    used_tilelang: bool,
    reason: str,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool,
    routed_scaling_factor: float,
    e_score_correction_bias: torch.Tensor | None,
    input_tokens: torch.Tensor | None,
    hash_indices_table: torch.Tensor | None,
) -> None:
    if not _router_topk_trace_enabled():
        return

    global _ROUTER_TOPK_TRACE_REGISTERED, _ROUTER_TOPK_TRACE_TOTAL
    if not _ROUTER_TOPK_TRACE_REGISTERED:
        atexit.register(_print_router_topk_trace_summary)
        _ROUTER_TOPK_TRACE_REGISTERED = True

    route_kind = (
        "hash"
        if input_tokens is not None or hash_indices_table is not None
        else "bias"
    )
    try:
        tokens = int(gating_output.shape[0])
        width = int(gating_output.shape[1])
    except Exception:
        tokens = -1
        width = -1
    try:
        topk = int(topk_indices.shape[1])
    except Exception:
        topk = -1
    decision = "tilelang" if used_tilelang else "fallback"
    reason = reason or decision
    bucket = {
        "route": route_kind,
        "decision": decision,
        "mode": mode,
        "tokens": tokens,
        "width": width,
        "topk": topk,
        "renorm": bool(renormalize),
        "scale": "scaled" if float(routed_scaling_factor) != 1.0 else "unit",
        "gating_dtype": _dtype_name(gating_output),
        "bias": "present" if e_score_correction_bias is not None else "none",
        "bias_dtype": _dtype_name(e_score_correction_bias),
        "input_tokens": _shape_key(input_tokens),
        "hash_table": _shape_key(hash_indices_table),
        "weights_dtype": _dtype_name(topk_weights),
        "indices_dtype": _dtype_name(topk_indices),
        "expert_indices_dtype": _dtype_name(token_expert_indices),
        "gating_contig": _contiguous_key(gating_output),
        "weights_contig": _contiguous_key(topk_weights),
        "indices_contig": _contiguous_key(topk_indices),
        "expert_indices_contig": _contiguous_key(token_expert_indices),
    }
    bucket_key = json.dumps(bucket, sort_keys=True, separators=(",", ":"))
    _ROUTER_TOPK_TRACE_TOTAL += 1
    _ROUTER_TOPK_TRACE_BUCKETS[bucket_key] += 1
    _ROUTER_TOPK_TRACE_REASONS[reason] += 1
    trace_every = _router_topk_trace_every()
    if trace_every and _ROUTER_TOPK_TRACE_TOTAL % trace_every == 0:
        _print_router_topk_trace_summary(event="periodic")


def record_router_grouped_topk_trace(
    *,
    decision: str,
    reason: str,
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    e_score_correction_bias: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    num_expert_group: int,
    topk_group: int,
    scoring_func: str,
    num_fused_shared_experts: int,
    topk_weights: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
) -> None:
    if not _router_topk_trace_enabled():
        return

    global _ROUTER_TOPK_TRACE_REGISTERED, _ROUTER_TOPK_TRACE_TOTAL
    if not _ROUTER_TOPK_TRACE_REGISTERED:
        atexit.register(_print_router_topk_trace_summary)
        _ROUTER_TOPK_TRACE_REGISTERED = True

    try:
        tokens = int(gating_output.shape[0])
        width = int(gating_output.shape[1])
    except Exception:
        tokens = -1
        width = -1
    bucket = {
        "source": "grouped_topk_router",
        "route": "input_ids" if input_ids is not None else "bias_or_group",
        "decision": decision,
        "reason": reason,
        "tokens": tokens,
        "width": width,
        "topk": int(topk),
        "num_expert_group": int(num_expert_group),
        "topk_group": int(topk_group),
        "scoring_func": str(scoring_func),
        "renorm": bool(renormalize),
        "scale": "scaled" if float(routed_scaling_factor) != 1.0 else "unit",
        "num_fused_shared_experts": int(num_fused_shared_experts),
        "hidden_dtype": _dtype_name(hidden_states),
        "gating_dtype": _dtype_name(gating_output),
        "bias": "present" if e_score_correction_bias is not None else "none",
        "bias_dtype": _dtype_name(e_score_correction_bias),
        "input_ids": _shape_key(input_ids),
        "weights_dtype": _dtype_name(topk_weights),
        "weights_shape": _shape_key(topk_weights),
        "ids_dtype": _dtype_name(topk_ids),
        "ids_shape": _shape_key(topk_ids),
        "hidden_contig": _contiguous_key(hidden_states),
        "gating_contig": _contiguous_key(gating_output),
    }
    bucket_key = json.dumps(bucket, sort_keys=True, separators=(",", ":"))
    _ROUTER_TOPK_TRACE_TOTAL += 1
    _ROUTER_TOPK_TRACE_BUCKETS[bucket_key] += 1
    _ROUTER_TOPK_TRACE_REASONS[f"grouped:{decision}:{reason}"] += 1
    trace_every = _router_topk_trace_every()
    if trace_every and _ROUTER_TOPK_TRACE_TOTAL % trace_every == 0:
        _print_router_topk_trace_summary(event="periodic")


def record_router_select_trace(
    *,
    router_name: str,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    input_ids: torch.Tensor | None,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    indices_type: torch.dtype | None,
    enable_eplb: bool,
    stage: str,
) -> None:
    if not _router_topk_trace_enabled():
        return

    global _ROUTER_TOPK_TRACE_REGISTERED, _ROUTER_TOPK_TRACE_TOTAL
    if not _ROUTER_TOPK_TRACE_REGISTERED:
        atexit.register(_print_router_topk_trace_summary)
        _ROUTER_TOPK_TRACE_REGISTERED = True

    bucket = {
        "source": "base_router_select",
        "stage": stage,
        "router": router_name,
        "decision": "observed",
        "reason": "select_experts",
        "tokens": int(router_logits.shape[0]) if router_logits.ndim >= 1 else -1,
        "width": int(router_logits.shape[1]) if router_logits.ndim >= 2 else -1,
        "hidden_shape": _shape_key(hidden_states),
        "hidden_dtype": _dtype_name(hidden_states),
        "router_logits_shape": _shape_key(router_logits),
        "router_logits_dtype": _dtype_name(router_logits),
        "input_ids": _shape_key(input_ids),
        "topk": int(topk_ids.shape[1]) if topk_ids.ndim >= 2 else -1,
        "weights_shape": _shape_key(topk_weights),
        "weights_dtype": _dtype_name(topk_weights),
        "ids_shape": _shape_key(topk_ids),
        "ids_dtype": _dtype_name(topk_ids),
        "indices_type": str(indices_type).replace("torch.", "")
        if indices_type is not None
        else "none",
        "enable_eplb": bool(enable_eplb),
        "hidden_contig": _contiguous_key(hidden_states),
        "router_logits_contig": _contiguous_key(router_logits),
        "weights_contig": _contiguous_key(topk_weights),
        "ids_contig": _contiguous_key(topk_ids),
    }
    bucket_key = json.dumps(bucket, sort_keys=True, separators=(",", ":"))
    _ROUTER_TOPK_TRACE_TOTAL += 1
    _ROUTER_TOPK_TRACE_BUCKETS[bucket_key] += 1
    _ROUTER_TOPK_TRACE_REASONS[
        f"base:{router_name}:{stage}:select_experts"
    ] += 1
    trace_every = _router_topk_trace_every()
    if trace_every and _ROUTER_TOPK_TRACE_TOTAL % trace_every == 0:
        _print_router_topk_trace_summary(event="periodic")


def record_moe_apply_trace(
    *,
    layer_name: str,
    quant_method_name: str,
    router_name: str,
    is_monolithic: bool,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    input_ids: torch.Tensor | None,
    shared_experts_input: torch.Tensor | None,
    stage: str,
) -> None:
    if not _router_topk_trace_enabled():
        return

    global _ROUTER_TOPK_TRACE_REGISTERED, _ROUTER_TOPK_TRACE_TOTAL
    if not _ROUTER_TOPK_TRACE_REGISTERED:
        atexit.register(_print_router_topk_trace_summary)
        _ROUTER_TOPK_TRACE_REGISTERED = True

    bucket = {
        "source": "moe_apply_quant_method",
        "stage": stage,
        "layer": layer_name,
        "quant_method": quant_method_name,
        "router": router_name,
        "decision": "monolithic" if is_monolithic else "router_select",
        "reason": "apply_quant_method",
        "tokens": int(router_logits.shape[0]) if router_logits.ndim >= 1 else -1,
        "width": int(router_logits.shape[1]) if router_logits.ndim >= 2 else -1,
        "hidden_shape": _shape_key(hidden_states),
        "hidden_dtype": _dtype_name(hidden_states),
        "router_logits_shape": _shape_key(router_logits),
        "router_logits_dtype": _dtype_name(router_logits),
        "input_ids": _shape_key(input_ids),
        "shared_input": _shape_key(shared_experts_input),
        "hidden_contig": _contiguous_key(hidden_states),
        "router_logits_contig": _contiguous_key(router_logits),
    }
    bucket_key = json.dumps(bucket, sort_keys=True, separators=(",", ":"))
    _ROUTER_TOPK_TRACE_TOTAL += 1
    _ROUTER_TOPK_TRACE_BUCKETS[bucket_key] += 1
    _ROUTER_TOPK_TRACE_REASONS[
        f"moe:{quant_method_name}:{router_name}:{bucket['decision']}"
    ] += 1
    trace_every = _router_topk_trace_every()
    if trace_every and _ROUTER_TOPK_TRACE_TOTAL % trace_every == 0:
        _print_router_topk_trace_summary(event="periodic")


def _is_musa_tensor(tensor: torch.Tensor | None) -> bool:
    return (
        tensor is not None
        and getattr(tensor, "device", None) is not None
        and tensor.device.type == "musa"
    )


def _index_dtype_name(dtype: torch.dtype) -> str | None:
    if dtype == torch.int32:
        return "int32"
    if dtype == torch.int64:
        return "int64"
    return None


def _guard_tilelang_sparse_indexer_topk_rows(
    scores: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    out: torch.Tensor,
    topk: int,
) -> tuple[bool, str]:
    tensors = (scores, starts, ends, out)
    if not all(_is_musa_tensor(tensor) for tensor in tensors):
        return False, "all tensors must be on MUSA"
    devices = {tensor.device for tensor in tensors}
    if len(devices) != 1:
        return False, "all tensors must be on the same MUSA device"
    if scores.dtype != torch.float32:
        return False, f"expected float32 scores, got {scores.dtype}"
    if starts.dtype != torch.int32 or ends.dtype != torch.int32:
        return False, f"expected int32 starts/ends, got {starts.dtype}/{ends.dtype}"
    if out.dtype != torch.int32:
        return False, f"expected int32 output, got {out.dtype}"
    if scores.dim() != 2:
        return False, f"expected scores shape [rows, width], got {tuple(scores.shape)}"
    if starts.dim() != 1 or ends.dim() != 1:
        return False, "starts and ends must be 1D"
    rows, width = scores.shape
    if starts.shape[0] != rows or ends.shape[0] != rows or out.shape[0] != rows:
        return False, "metadata and output rows must match scores rows"
    if out.dim() != 2 or int(topk) <= 0 or out.shape[1] < int(topk):
        return False, f"output must have at least topk columns, out={tuple(out.shape)} topk={topk}"
    if width <= 0:
        return False, "scores width must be positive"
    if width > _MAX_TILELANG_TOPK_WIDTH:
        return False, (
            f"scores width {width} exceeds TileLang top-k guard "
            f"{_MAX_TILELANG_TOPK_WIDTH}"
        )
    if int(topk) > width:
        return False, f"topk {topk} must be <= scores width {width}"
    if scores.stride(-1) != 1:
        return False, f"scores last dimension must be contiguous, stride={scores.stride()}"
    if not starts.is_contiguous() or not ends.is_contiguous() or not out.is_contiguous():
        return False, "starts, ends, and output must be contiguous"
    return True, ""


def try_tilelang_sparse_indexer_topk_rows(
    scores: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    out: torch.Tensor,
    topk: int,
) -> tuple[bool, str]:
    """Try the TileLang sparse-indexer raw-index top-k kernel."""

    mode = (
        os.environ.get("VLLM_MUSA_SPARSE_INDEXER_TOPK_IMPL", "auto")
        .strip()
        .lower()
    )
    if mode not in {"tilelang", "jit", "tilelang_small"}:
        return False, "disabled by VLLM_MUSA_SPARSE_INDEXER_TOPK_IMPL"

    supported, reason = _guard_tilelang_sparse_indexer_topk_rows(
        scores, starts, ends, out, topk
    )
    if not supported:
        return False, reason

    from vllm_musa.deepseek_v4_jit.tilelang_kernels import (
        sparse_indexer_topk_rows_kernel,
    )

    sparse_indexer_topk_rows_kernel(
        int(scores.shape[1]),
        int(topk),
        int(scores.stride(0)),
    )(scores, starts, ends, out[:, :topk])
    return True, "tilelang"


def _router_topk_mode() -> str:
    mode = (
        os.environ.get("VLLM_MUSA_DEEPSEEK_V4_ROUTER_TOPK_IMPL", "torch")
        .strip()
        .lower()
    )
    if mode in _ROUTER_TOPK_DISABLE_MODES or mode in _ROUTER_TOPK_MODES:
        return mode
    return "auto"


def _guard_tilelang_hash_topk_softplus_sqrt(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    e_score_correction_bias: torch.Tensor | None,
    input_tokens: torch.Tensor | None,
    hash_indices_table: torch.Tensor | None,
    topk: int,
) -> tuple[bool, str]:
    if input_tokens is None or hash_indices_table is None:
        return False, "hash routing requires both input_tokens and hash_indices_table"
    if e_score_correction_bias is not None:
        return False, "hash routing does not use correction bias"
    tensors = (
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        input_tokens,
        hash_indices_table,
    )
    if not all(_is_musa_tensor(tensor) for tensor in tensors):
        return False, "all tensors must be on MUSA"
    devices = {tensor.device for tensor in tensors}
    if len(devices) != 1:
        return False, "all tensors must be on the same MUSA device"
    if gating_output.dtype != torch.float32:
        return False, f"expected float32 gating_output, got {gating_output.dtype}"
    if topk_weights.dtype != torch.float32:
        return False, f"expected float32 topk_weights, got {topk_weights.dtype}"
    if topk_indices.dtype != torch.int64:
        return False, f"expected int64 topk_indices, got {topk_indices.dtype}"
    if token_expert_indices.dtype != torch.int32:
        return False, (
            "expected int32 token_expert_indices, got "
            f"{token_expert_indices.dtype}"
        )
    if _index_dtype_name(input_tokens.dtype) is None:
        return False, f"unsupported input_tokens dtype {input_tokens.dtype}"
    if _index_dtype_name(hash_indices_table.dtype) is None:
        return False, f"unsupported hash_indices_table dtype {hash_indices_table.dtype}"
    if gating_output.dim() != 2:
        return False, (
            "expected router logits shape [tokens, experts], got "
            f"{tuple(gating_output.shape)}"
        )
    if input_tokens.dim() != 1:
        return False, f"input_tokens must be 1D, got {tuple(input_tokens.shape)}"
    if hash_indices_table.dim() != 2:
        return False, (
            "hash_indices_table must be 2D, got "
            f"{tuple(hash_indices_table.shape)}"
        )
    if int(topk) != 6:
        return False, f"expected DeepSeek-V4-Flash hash topk=6, got {topk}"
    if input_tokens.shape[0] != gating_output.shape[0]:
        return False, "input_tokens and gating_output rows must match"
    if hash_indices_table.shape[1] != int(topk):
        return False, (
            "hash_indices_table second dimension must match topk, got "
            f"{tuple(hash_indices_table.shape)} topk={topk}"
        )
    expected_shape = (gating_output.shape[0], int(topk))
    if tuple(topk_weights.shape) != expected_shape:
        return False, f"unexpected topk_weights shape {tuple(topk_weights.shape)}"
    if tuple(topk_indices.shape) != expected_shape:
        return False, f"unexpected topk_indices shape {tuple(topk_indices.shape)}"
    if tuple(token_expert_indices.shape) != expected_shape:
        return False, (
            "unexpected token_expert_indices shape "
            f"{tuple(token_expert_indices.shape)}"
        )
    for name, tensor in (
        ("gating_output", gating_output),
        ("input_tokens", input_tokens),
        ("hash_indices_table", hash_indices_table),
        ("topk_weights", topk_weights),
        ("topk_indices", topk_indices),
        ("token_expert_indices", token_expert_indices),
    ):
        if not tensor.is_contiguous():
            return False, f"{name} must be contiguous"
    return True, ""


def _guard_tilelang_biased_topk_softplus_sqrt(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    e_score_correction_bias: torch.Tensor | None,
    input_tokens: torch.Tensor | None,
    hash_indices_table: torch.Tensor | None,
    topk: int,
) -> tuple[bool, str]:
    if input_tokens is not None or hash_indices_table is not None:
        return False, "hash-routed path stays on the existing fallback"
    tensors = (
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        e_score_correction_bias,
    )
    if not all(_is_musa_tensor(tensor) for tensor in tensors):
        return False, "all tensors must be on MUSA"
    devices = {tensor.device for tensor in tensors if tensor is not None}
    if len(devices) != 1:
        return False, "all tensors must be on the same MUSA device"
    if gating_output.dtype != torch.float32:
        return False, f"expected float32 gating_output, got {gating_output.dtype}"
    if e_score_correction_bias is None:
        return False, "correction bias is required"
    if e_score_correction_bias.dtype != torch.float32:
        return False, (
            "expected float32 correction bias, got "
            f"{e_score_correction_bias.dtype}"
        )
    if topk_weights.dtype != torch.float32:
        return False, f"expected float32 topk_weights, got {topk_weights.dtype}"
    if topk_indices.dtype != torch.int64:
        return False, f"expected int64 topk_indices, got {topk_indices.dtype}"
    if token_expert_indices.dtype != torch.int32:
        return False, (
            "expected int32 token_expert_indices, got "
            f"{token_expert_indices.dtype}"
        )
    if gating_output.dim() != 2 or gating_output.shape[1] != 256:
        return False, (
            "expected DeepSeek-V4 router logits shape [tokens, 256], got "
            f"{tuple(gating_output.shape)}"
        )
    if int(topk) != 6:
        return False, f"expected DeepSeek-V4-Flash topk=6, got {topk}"
    if e_score_correction_bias.dim() != 1 or e_score_correction_bias.numel() != 256:
        return False, "correction bias must be shape [256]"
    expected_shape = (gating_output.shape[0], int(topk))
    if tuple(topk_weights.shape) != expected_shape:
        return False, f"unexpected topk_weights shape {tuple(topk_weights.shape)}"
    if tuple(topk_indices.shape) != expected_shape:
        return False, f"unexpected topk_indices shape {tuple(topk_indices.shape)}"
    if tuple(token_expert_indices.shape) != expected_shape:
        return False, (
            "unexpected token_expert_indices shape "
            f"{tuple(token_expert_indices.shape)}"
        )
    for name, tensor in (
        ("gating_output", gating_output),
        ("correction_bias", e_score_correction_bias),
        ("topk_weights", topk_weights),
        ("topk_indices", topk_indices),
        ("token_expert_indices", token_expert_indices),
    ):
        if not tensor.is_contiguous():
            return False, f"{name} must be contiguous"
    return True, ""


def _try_tilelang_hash_topk_softplus_sqrt(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool,
    routed_scaling_factor: float,
    e_score_correction_bias: torch.Tensor | None,
    input_tokens: torch.Tensor | None,
    hash_indices_table: torch.Tensor | None,
    mode: str,
) -> tuple[bool, str]:
    global _ROUTER_TOPK_AUTO_DISABLED_REASON

    topk = int(topk_indices.shape[1])
    supported, reason = _guard_tilelang_hash_topk_softplus_sqrt(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        e_score_correction_bias,
        input_tokens,
        hash_indices_table,
        topk,
    )
    if not supported:
        return False, reason

    assert input_tokens is not None
    assert hash_indices_table is not None
    try:
        from vllm_musa.deepseek_v4_jit.tilelang_kernels import (
            hash_topk_softplus_sqrt_kernel,
        )

        input_dtype_name = _index_dtype_name(input_tokens.dtype)
        hash_dtype_name = _index_dtype_name(hash_indices_table.dtype)
        assert input_dtype_name is not None
        assert hash_dtype_name is not None
        kernel = hash_topk_softplus_sqrt_kernel(
            topk,
            input_dtype_name,
            hash_dtype_name,
            bool(renormalize),
            float(routed_scaling_factor) != 1.0,
        )
        kernel(
            gating_output,
            input_tokens,
            hash_indices_table,
            topk_weights,
            topk_indices,
            token_expert_indices,
            float(routed_scaling_factor),
        )
    except Exception as exc:
        reason = f"TileLang hash top-k failed: {type(exc).__name__}: {exc}"
        if mode == "auto":
            _ROUTER_TOPK_AUTO_DISABLED_REASON = reason
        return False, reason
    return True, "tilelang_hash"


def try_tilelang_biased_topk_softplus_sqrt(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool,
    routed_scaling_factor: float,
    e_score_correction_bias: torch.Tensor | None,
    input_tokens: torch.Tensor | None,
    hash_indices_table: torch.Tensor | None,
) -> tuple[bool, str]:
    """Try the DeepSeek-V4 non-hash biased top-k TileLang provider."""

    global _ROUTER_TOPK_AUTO_DISABLED_REASON
    mode = _router_topk_mode()

    def record(used_tilelang: bool, reason: str) -> None:
        _record_router_topk_trace(
            mode=mode,
            used_tilelang=used_tilelang,
            reason=reason,
            topk_weights=topk_weights,
            topk_indices=topk_indices,
            token_expert_indices=token_expert_indices,
            gating_output=gating_output,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            input_tokens=input_tokens,
            hash_indices_table=hash_indices_table,
        )

    if mode in _ROUTER_TOPK_DISABLE_MODES:
        reason = "disabled by VLLM_MUSA_DEEPSEEK_V4_ROUTER_TOPK_IMPL"
        record(False, reason)
        return False, reason
    if mode == "auto" and _ROUTER_TOPK_AUTO_DISABLED_REASON is not None:
        record(False, _ROUTER_TOPK_AUTO_DISABLED_REASON)
        return False, _ROUTER_TOPK_AUTO_DISABLED_REASON

    if input_tokens is not None or hash_indices_table is not None:
        used_tilelang, reason = _try_tilelang_hash_topk_softplus_sqrt(
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating_output,
            renormalize,
            routed_scaling_factor,
            e_score_correction_bias,
            input_tokens,
            hash_indices_table,
            mode,
        )
        record(used_tilelang, reason)
        return used_tilelang, reason

    if mode in _ROUTER_TOPK_HASH_ONLY_MODES:
        reason = "hash-only router top-k mode skips non-hash path"
        record(False, reason)
        return False, reason

    topk = int(topk_indices.shape[1])
    supported, reason = _guard_tilelang_biased_topk_softplus_sqrt(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        e_score_correction_bias,
        input_tokens,
        hash_indices_table,
        topk,
    )
    if not supported:
        record(False, reason)
        return False, reason

    try:
        if mode in _ROUTER_TOPK_WARP_MODES:
            from vllm_musa.deepseek_v4_jit.tilelang_kernels import (
                biased_topk_softplus_sqrt_256_warp_kernel,
            )

            kernel = biased_topk_softplus_sqrt_256_warp_kernel(
                topk,
                bool(renormalize),
                float(routed_scaling_factor) != 1.0,
            )
            provider = "tilelang_warp"
        else:
            from vllm_musa.deepseek_v4_jit.tilelang_kernels import (
                biased_topk_softplus_sqrt_256_kernel,
            )

            kernel = biased_topk_softplus_sqrt_256_kernel(
                topk,
                bool(renormalize),
                float(routed_scaling_factor) != 1.0,
            )
            provider = "tilelang"

        kernel(
            gating_output,
            e_score_correction_bias,
            topk_weights,
            topk_indices,
            token_expert_indices,
            float(routed_scaling_factor),
        )
    except Exception as exc:
        provider = (
            "TileLang warp biased top-k"
            if mode in _ROUTER_TOPK_WARP_MODES
            else "TileLang biased top-k"
        )
        reason = f"{provider} failed: {type(exc).__name__}: {exc}"
        if mode == "auto":
            _ROUTER_TOPK_AUTO_DISABLED_REASON = reason
        record(False, reason)
        return False, reason
    record(True, provider)
    return True, provider
