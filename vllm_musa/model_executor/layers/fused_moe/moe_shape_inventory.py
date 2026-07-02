# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional JSONL dump of fused-MoE call shapes, off unless its env is set.

Collects real routing and shape statistics that feed MoE dispatch-policy
tuning. Writing is gated by an env flag, a minimum token count, and a
per-process record cap so it never perturbs a normal serving run.
"""

from __future__ import annotations

import json
import os
import time

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_INVENTORY_ENV = "VLLM_MUSA_DEEPSEEK_V4_MOE_SHAPE_INVENTORY"
_INVENTORY_PATH_ENV = "VLLM_MUSA_DEEPSEEK_V4_MOE_SHAPE_INVENTORY_PATH"
_INVENTORY_MIN_TOKENS_ENV = "VLLM_MUSA_DEEPSEEK_V4_MOE_SHAPE_INVENTORY_MIN_TOKENS"
_INVENTORY_MAX_RECORDS_ENV = "VLLM_MUSA_DEEPSEEK_V4_MOE_SHAPE_INVENTORY_MAX_RECORDS"
_INVENTORY_DEFAULT_PATH = (
    "/tmp/vllm_omni_musa_outputs/deepseek_v4_moe_shape_inventory.jsonl"
)
_INVENTORY_EVENT = "deepseek_v4_moe_shape_inventory"

_RECORDS = 0
_WARNED = False


def _env_flag_enabled(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.lower() not in {"", "0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _tensor_meta(tensor: torch.Tensor | None) -> dict[str, object] | None:
    if tensor is None:
        return None
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "stride": list(tensor.stride()),
        "is_contiguous": tensor.is_contiguous(),
    }


def _routed_token_histogram(
    topk_ids: torch.Tensor,
    num_experts: int,
) -> dict[str, object]:
    ids_cpu = topk_ids.detach().to(device="cpu", dtype=torch.int64)
    flat_ids = ids_cpu.reshape(-1)
    valid_mask = flat_ids >= 0
    if num_experts > 0:
        valid_mask &= flat_ids < num_experts
    valid_ids = flat_ids[valid_mask]

    histogram_size = num_experts
    if histogram_size <= 0 and valid_ids.numel() > 0:
        histogram_size = int(valid_ids.max().item()) + 1

    if histogram_size > 0:
        histogram = torch.bincount(valid_ids, minlength=histogram_size).tolist()
    else:
        histogram = []

    slot_histograms = []
    for slot in range(ids_cpu.shape[1] if ids_cpu.dim() >= 2 else 0):
        slot_ids = ids_cpu[:, slot].reshape(-1)
        slot_valid = slot_ids >= 0
        if num_experts > 0:
            slot_valid &= slot_ids < num_experts
        if histogram_size > 0:
            slot_histograms.append(
                torch.bincount(slot_ids[slot_valid], minlength=histogram_size).tolist()
            )
        else:
            slot_histograms.append([])

    nonzero = [count for count in histogram if count]
    return {
        "histogram": histogram,
        "slot_histograms": slot_histograms,
        "invalid_count": int((~valid_mask).sum().item()),
        "nonzero_experts": len(nonzero),
        "max_routes_per_expert": max(nonzero) if nonzero else 0,
        "min_routes_per_nonzero_expert": min(nonzero) if nonzero else 0,
    }


def maybe_record_moe_shape_inventory(
    *,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    a1_scale: torch.Tensor | None,
    a2_scale: torch.Tensor | None,
    block_shape: list[int] | None,
    activation: str,
    apply_router_weight_on_input: bool,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    ocp_mx_scheme: str | None,
    per_channel_quant: bool,
    global_num_experts: int,
) -> None:
    global _RECORDS
    global _WARNED

    if not _env_flag_enabled(_INVENTORY_ENV):
        return

    num_tokens = hidden_states.size(0)
    min_tokens = _env_int(_INVENTORY_MIN_TOKENS_ENV, 4096)
    if num_tokens < min_tokens:
        return

    max_records = _env_int(_INVENTORY_MAX_RECORDS_ENV, 64)
    if max_records >= 0 and _RECORDS >= max_records:
        return

    try:
        E, N, _ = w1.size()
        K = w2.size(1)
        num_experts = global_num_experts if global_num_experts > 0 else E
        route_stats = _routed_token_histogram(topk_ids, num_experts)
        record = {
            "event": _INVENTORY_EVENT,
            "time": time.time(),
            "pid": os.getpid(),
            "record_index": _RECORDS,
            "num_tokens": num_tokens,
            "top_k": topk_ids.size(1),
            "num_local_experts": E,
            "global_num_experts": num_experts,
            "w1_intermediate_size": N,
            "w2_output_size": K,
            "hidden_states": _tensor_meta(hidden_states),
            "w1": _tensor_meta(w1),
            "w2": _tensor_meta(w2),
            "topk_weights": _tensor_meta(topk_weights),
            "topk_ids": _tensor_meta(topk_ids),
            "w1_scale": _tensor_meta(w1_scale),
            "w2_scale": _tensor_meta(w2_scale),
            "a1_scale": _tensor_meta(a1_scale),
            "a2_scale": _tensor_meta(a2_scale),
            "block_shape": block_shape,
            "activation": activation,
            "apply_router_weight_on_input": apply_router_weight_on_input,
            "use_fp8_w8a8": use_fp8_w8a8,
            "use_int8_w8a8": use_int8_w8a8,
            "use_int8_w8a16": use_int8_w8a16,
            "use_int4_w4a16": use_int4_w4a16,
            "ocp_mx_scheme": ocp_mx_scheme,
            "per_channel_quant": per_channel_quant,
            "routed_token_stats": route_stats,
        }

        output_path = os.environ.get(_INVENTORY_PATH_ENV, _INVENTORY_DEFAULT_PATH)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "a", encoding="utf-8") as inventory_file:
            inventory_file.write(json.dumps(record, sort_keys=True) + "\n")
        _RECORDS += 1
    except Exception as exc:
        if not _WARNED:
            logger.warning("Failed to write MoE shape inventory: %s", exc)
            _WARNED = True
