# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified MoE kernel dispatch policy for the MUSA fused-MoE path.

Two orthogonal decisions drive kernel choice, and neither reads a model name:

* Compute backend, driven by token count. Small batches run the native MUSA
  GEMV fused-MoE kernel; large prefill batches run the contiguous grouped
  DeepGEMM kernel once the token count clears a shape-specific threshold.
* Layout, driven by the parallel configuration. The contiguous grouped-GEMM
  path handles tensor-parallel-only MoE; the masked (expert-parallel) layout is
  not wired here yet, so expert-parallel inputs stay on the native path.

Per-shape thresholds live in a data table keyed on the intrinsic tensor
signature ``(hidden, intermediate, experts, top_k)`` rather than a model
identity, so the same shape on any model resolves to the same policy. An opt-in
autotuner may override an entry per ``(signature, expert_parallel, device)``
through the on-disk cache read here.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Unified dispatch knobs.
_FORCE_ENV = "VLLM_MUSA_MOE_DEEPGEMM"
_MIN_TOKENS_ENV = "VLLM_MUSA_MOE_DEEPGEMM_MIN_TOKENS"
_TUNED_CACHE_ENV = "VLLM_MUSA_MOE_DEEPGEMM_TUNED_CACHE"
_TUNED_CACHE_DEFAULT = "/tmp/vllm_omni_musa_outputs/musa_moe_deepgemm_tuned.json"

# Back-compat knobs honored by the calibration entries and top-level routing.
_FORCE_MUSA_IMPL_ENV = "VLLM_MUSA_DEEPSEEK_V4_FUSED_MOE_GEMV"
_LEGACY_ENABLE_ENV = "VLLM_MUSA_DEEPSEEK_V4_MOE_DEEPGEMM_PREFILL"
_LEGACY_MIN_TOKENS_ENV = "VLLM_MUSA_DEEPSEEK_V4_MOE_DEEPGEMM_PREFILL_MIN_TOKENS"

# Conservative default keeps small serving batches on the native GEMV kernel;
# the raw kernel crossover is far below the E2E-safe crossover on this device.
_DEFAULT_MIN_TOKENS = 65536


class MoEBackend(Enum):
    """MoE compute kernels selectable on MUSA.

    Only the first two are wired today; the rest name extension points so the
    policy table and selector need no change when a kernel is added.
    """

    NATIVE_GEMV = "native_gemv"
    GROUPED_DEEPGEMM_CONTIGUOUS = "grouped_deepgemm_contiguous"
    MASKED_DEEPGEMM = "masked_deepgemm"
    TRITON = "triton"


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _env_bool_optional(name: str) -> bool | None:
    if os.environ.get(name) is None:
        return None
    return _env_bool(name, default=False)


def _env_int_optional(name: str | None) -> int | None:
    if name is None:
        return None
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


@dataclass(frozen=True)
class MoEDispatchInput:
    """Model-agnostic descriptor derived from a fused-MoE call.

    Every field is a shape, dtype, quant flag, or parallel-config fact. There is
    no model-identity field, by design.
    """

    num_tokens: int
    hidden_size: int  # K
    intermediate_w1: int  # N (== 2 * per-expert intermediate for gated MoE)
    num_local_experts: int  # E
    global_num_experts: int
    top_k: int
    block_shape: tuple[int, int] | None
    hidden_dtype: torch.dtype
    weights_fp8_e4m3: bool
    scales_fp32: bool
    topk_dtype: torch.dtype
    use_fp8_w8a8: bool
    use_int8_w8a8: bool
    use_int8_w8a16: bool
    use_int4_w4a16: bool
    ocp_mx: bool
    per_channel_quant: bool
    activation: str
    apply_router_weight_on_input: bool
    has_expert_map: bool
    has_input_scales: bool
    has_biases: bool
    contiguous: bool
    w2_shape_matches: bool
    w1_scale_shape_ok: bool
    w2_scale_shape_ok: bool

    @property
    def expert_parallel(self) -> bool:
        return self.has_expert_map or self.global_num_experts > self.num_local_experts

    @property
    def signature(self) -> tuple[int, int, int, int, bool]:
        return (
            self.hidden_size,
            self.intermediate_w1,
            self.num_local_experts,
            self.top_k,
            self.expert_parallel,
        )


def _scale_shape_ok(scale: torch.Tensor | None, expected: tuple[int, int, int]) -> bool:
    return scale is not None and tuple(scale.shape) == expected


def build_dispatch_input(
    *,
    hidden_states: torch.Tensor | None,
    w1: torch.Tensor | None,
    w2: torch.Tensor | None,
    topk_ids: torch.Tensor | None,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    ocp_mx_scheme: str | None = None,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    block_shape: list[int] | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> MoEDispatchInput | None:
    """Build the descriptor, or ``None`` when the required tensors are absent."""
    if not all(isinstance(t, torch.Tensor) for t in (hidden_states, w1, w2, topk_ids)):
        return None
    if w1.dim() != 3 or hidden_states.dim() != 2 or topk_ids.dim() != 2:
        return None

    E, N, w1_contract = w1.shape
    K = hidden_states.shape[1]
    if global_num_experts is None or global_num_experts < 0:
        global_num_experts = E

    half_n = N // 2 if N % 2 == 0 else -1
    shapes_consistent = w1_contract == K and half_n > 0
    w2_shape_matches = shapes_consistent and tuple(w2.shape) == (E, K, half_n)
    if shapes_consistent and N % 128 == 0 and K % 128 == 0 and half_n % 128 == 0:
        w1_scale_ok = _scale_shape_ok(w1_scale, (E, N // 128, K // 128))
        w2_scale_ok = _scale_shape_ok(w2_scale, (E, K // 128, half_n // 128))
    else:
        w1_scale_ok = w2_scale_ok = False

    return MoEDispatchInput(
        num_tokens=hidden_states.shape[0],
        hidden_size=K,
        intermediate_w1=N,
        num_local_experts=E,
        global_num_experts=global_num_experts,
        top_k=topk_ids.shape[1],
        block_shape=tuple(block_shape) if block_shape is not None else None,
        hidden_dtype=hidden_states.dtype,
        weights_fp8_e4m3=w1.dtype == torch.float8_e4m3fn
        and w2.dtype == torch.float8_e4m3fn,
        scales_fp32=w1_scale is not None
        and w2_scale is not None
        and w1_scale.dtype == torch.float32
        and w2_scale.dtype == torch.float32,
        topk_dtype=topk_ids.dtype,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        ocp_mx=ocp_mx_scheme is not None,
        per_channel_quant=per_channel_quant,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        has_expert_map=expert_map is not None,
        has_input_scales=a1_scale is not None or a2_scale is not None,
        has_biases=w1_bias is not None or w2_bias is not None,
        contiguous=hidden_states.is_contiguous()
        and w1.is_contiguous()
        and w2.is_contiguous(),
        w2_shape_matches=w2_shape_matches,
        w1_scale_shape_ok=w1_scale_ok,
        w2_scale_shape_ok=w2_scale_ok,
    )


def grouped_deepgemm_contiguous_eligible(inp: MoEDispatchInput) -> bool:
    """General FP8 block-128 contiguous grouped-GEMM contract. No model gate."""
    if inp.expert_parallel:  # layout axis: the contiguous kernel is TP-only
        return False
    if not inp.use_fp8_w8a8:
        return False
    if (
        inp.use_int8_w8a8
        or inp.use_int8_w8a16
        or inp.use_int4_w4a16
        or inp.ocp_mx
        or inp.per_channel_quant
    ):
        return False
    if inp.has_input_scales or inp.has_biases:
        return False
    if inp.activation != "silu" or inp.apply_router_weight_on_input:
        return False
    if inp.block_shape != (128, 128):
        return False
    if inp.top_k <= 0:
        return False
    if inp.hidden_dtype != torch.bfloat16:
        return False
    if not inp.weights_fp8_e4m3 or not inp.scales_fp32:
        return False
    if inp.topk_dtype not in (torch.int32, torch.int64):
        return False
    if not inp.contiguous:
        return False
    E, N, K = inp.num_local_experts, inp.intermediate_w1, inp.hidden_size
    if E <= 0 or N <= 0 or K <= 0:
        return False
    if N % 2 or N % 128 or (N // 2) % 128 or K % 128:
        return False
    return inp.w2_shape_matches and inp.w1_scale_shape_ok and inp.w2_scale_shape_ok


@dataclass(frozen=True)
class _CalibratedPoint:
    hidden_size: int | None  # None matches any hidden size
    intermediate_w1: int | None  # None matches any w1 width (TP shards it)
    num_local_experts: int
    top_k: int
    enable_env: str | None  # None => enabled unconditionally
    min_tokens: int
    min_tokens_env: str | None

    def matches(self, signature: tuple[int, int, int, int, bool]) -> bool:
        hidden, intermediate, experts, top_k, expert_parallel = signature
        if expert_parallel:
            return False
        if self.hidden_size is not None and self.hidden_size != hidden:
            return False
        if self.intermediate_w1 is not None and self.intermediate_w1 != intermediate:
            return False
        return (experts, top_k) == (self.num_local_experts, self.top_k)

    def resolve(self) -> tuple[bool, int]:
        enabled = True if self.enable_env is None else _env_bool(self.enable_env, False)
        min_tokens = _env_int_optional(self.min_tokens_env)
        if min_tokens is None:
            min_tokens = self.min_tokens
        return enabled, min_tokens


# Empirically validated E2E operating points, keyed on the intrinsic tensor
# signature -- never model identity, so the same shape on any model resolves
# here. The opt-in autotuner regenerates equivalents per (signature, EP, device)
# into the on-disk cache, which takes precedence over this table.
_CALIBRATION: tuple[_CalibratedPoint, ...] = (
    # 256 experts, top-6, block-128 FP8: opt-in via the serving-profile env;
    # grouped DeepGEMM wins from ~4096 prefill tokens. Keyed independent of the
    # per-rank w1 width, which TP shards.
    _CalibratedPoint(None, None, 256, 6, _LEGACY_ENABLE_ENV, 4096, _LEGACY_MIN_TOKENS_ENV),
    # 2048 hidden, 256 experts, top-8, block-128 FP8: on by signature; grouped
    # DeepGEMM pays off only past very large prefill batches. Per-rank w1 width
    # varies with TP (1024 at TP1, 256 at TP4), so it is not part of the key.
    _CalibratedPoint(2048, None, 256, 8, None, 65536, None),
)


_TUNED_CACHE: dict[str, tuple[bool, int]] | None = None


def _device_name() -> str:
    try:
        if hasattr(torch, "musa") and torch.musa.is_available():
            return torch.musa.get_device_name()
    except Exception:  # noqa: BLE001 - device probing must never break dispatch
        pass
    return "unknown"


def _cache_key(signature: tuple[int, int, int, int, bool]) -> str:
    hidden, intermediate, experts, top_k, expert_parallel = signature
    return (
        f"{hidden}:{intermediate}:{experts}:{top_k}:"
        f"ep={int(expert_parallel)}:{_device_name()}"
    )


def _load_tuned_cache() -> dict[str, tuple[bool, int]]:
    global _TUNED_CACHE
    if _TUNED_CACHE is not None:
        return _TUNED_CACHE
    _TUNED_CACHE = {}
    path = os.environ.get(_TUNED_CACHE_ENV, _TUNED_CACHE_DEFAULT)
    try:
        with open(path, encoding="utf-8") as cache_file:
            raw = json.load(cache_file)
        for key, entry in raw.items():
            _TUNED_CACHE[key] = (bool(entry["enabled"]), int(entry["min_tokens"]))
    except FileNotFoundError:
        pass
    except Exception as exc:  # noqa: BLE001 - a bad cache must never break dispatch
        logger.warning("Ignoring unreadable MUSA MoE DeepGEMM tuned cache: %s", exc)
    return _TUNED_CACHE


def reset_tuned_cache() -> None:
    """Drop the memoized on-disk policy cache (used by tests)."""
    global _TUNED_CACHE
    _TUNED_CACHE = None


def _tuned_cache_lookup(
    signature: tuple[int, int, int, int, bool],
) -> tuple[bool, int] | None:
    return _load_tuned_cache().get(_cache_key(signature))


def resolve_deepgemm_policy(inp: MoEDispatchInput) -> tuple[bool, int]:
    """Resolve ``(enabled, min_tokens)`` for grouped DeepGEMM at this signature.

    Precedence, high to low: explicit env override, on-disk tuned cache,
    calibration table, conservative default.
    """
    enabled, min_tokens = False, _DEFAULT_MIN_TOKENS

    cached = _tuned_cache_lookup(inp.signature)
    if cached is not None:
        enabled, min_tokens = cached
    else:
        for point in _CALIBRATION:
            if point.matches(inp.signature):
                enabled, min_tokens = point.resolve()
                break

    forced = _env_bool_optional(_FORCE_ENV)
    if forced is not None:
        enabled = forced
    global_min = _env_int_optional(_MIN_TOKENS_ENV)
    if global_min is not None:
        min_tokens = global_min
    return enabled, min_tokens


def select_backend(inp: MoEDispatchInput) -> MoEBackend:
    """Pick the compute backend for one fused-MoE call."""
    if not grouped_deepgemm_contiguous_eligible(inp):
        return MoEBackend.NATIVE_GEMV
    enabled, min_tokens = resolve_deepgemm_policy(inp)
    if enabled and inp.num_tokens >= min_tokens:
        return MoEBackend.GROUPED_DEEPGEMM_CONTIGUOUS
    return MoEBackend.NATIVE_GEMV


def should_route_to_musa_impl(inp: MoEDispatchInput | None) -> bool:
    """Whether the MUSA fused-MoE impl should handle this call.

    The MUSA native GEMV kernel and grouped DeepGEMM path handle FP8 block-128
    grouped-eligible MoE, so those route here; the explicit force env also
    routes here. Everything else falls through to the upstream implementation.
    """
    if _env_bool(_FORCE_MUSA_IMPL_ENV, default=False):
        return True
    if inp is None:
        return False
    return grouped_deepgemm_contiguous_eligible(inp)
