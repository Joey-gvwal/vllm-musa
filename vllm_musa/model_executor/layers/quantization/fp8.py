import math
import os
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import vllm.model_executor.layers.quantization.fp8 as vllm_fp8
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE, fused_experts
from vllm.platforms import current_platform
from vllm.utils.torch_utils import is_torch_equal_or_newer

logger = init_logger(__name__)

_MUSA_FP8_MOE_MIXED_BACKEND_ENV = "VLLM_MUSA_FP8_MOE_MIXED_BACKEND"
_MUSA_FP8_MOE_AUTOTUNE_ENV = "VLLM_MUSA_FP8_MOE_AUTOTUNE"
_MUSA_FP8_MOE_DEEPGEMM_MIN_TOKENS_ENV = "VLLM_MUSA_FP8_MOE_DEEPGEMM_MIN_TOKENS"
_MUSA_FP8_MOE_DEEPGEMM_DEFAULT_MIN_TOKENS = 128
_MUSA_FP8_MOE_MODULAR_MIN_PER_EXPERT_M_ENV = (
    "VLLM_MUSA_FP8_MOE_MODULAR_MIN_PER_EXPERT_M"
)
_MUSA_FP8_MOE_MODULAR_DEFAULT_MIN_PER_EXPERT_M = 64
_MUSA_FP8_MOE_AUTOTUNE_MAX_TOKENS_ENV = "VLLM_MUSA_FP8_MOE_AUTOTUNE_MAX_TOKENS"
_MUSA_FP8_MOE_AUTOTUNE_WARMUP_ENV = "VLLM_MUSA_FP8_MOE_AUTOTUNE_WARMUP"
_MUSA_FP8_MOE_AUTOTUNE_ITERS_ENV = "VLLM_MUSA_FP8_MOE_AUTOTUNE_ITERS"
_MUSA_FP8_MOE_AUTOTUNE_WIN_MARGIN_ENV = "VLLM_MUSA_FP8_MOE_AUTOTUNE_WIN_MARGIN"
_MUSA_FP8_MOE_AUTOTUNE_DEFAULT_WARMUP = 3
_MUSA_FP8_MOE_AUTOTUNE_DEFAULT_ITERS = 7
_MUSA_FP8_MOE_AUTOTUNE_DEFAULT_WIN_MARGIN = 0.98
_MUSA_FP8_MOE_BACKEND_TRITON = "triton"
_MUSA_FP8_MOE_BACKEND_DEEPGEMM = "deepgemm"
_MUSA_FP8_MOE_BUCKET_POLICY: tuple["MusaFp8MoeBucket", ...] | None = None
_MUSA_FP8_MOE_FORCE_BACKEND: str | None = None
_MUSA_FP8_MOE_AUTOTUNE_DONE = False


@dataclass(frozen=True)
class MusaFp8MoeBucket:
    max_tokens: int
    backend: str


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _zero_fp8_weight(weight: torch.Tensor) -> None:
    # MUSA muDNN fill does not support FP8_E4M3 directly.
    weight.view(torch.uint8).zero_()


def _env_flag_enabled(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default)
    return value.lower() not in {"", "0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _fp8_backend_name(method: object) -> str:
    backend = getattr(method, "fp8_backend", None)
    return str(getattr(backend, "value", backend))


def set_musa_fp8_moe_bucket_policy(
    policy: tuple[MusaFp8MoeBucket, ...] | None,
) -> None:
    global _MUSA_FP8_MOE_BUCKET_POLICY
    _MUSA_FP8_MOE_BUCKET_POLICY = policy


def get_musa_fp8_moe_bucket_policy() -> tuple[MusaFp8MoeBucket, ...] | None:
    return _MUSA_FP8_MOE_BUCKET_POLICY


def _serialize_musa_fp8_moe_bucket_policy(
    policy: tuple[MusaFp8MoeBucket, ...] | None,
) -> list[tuple[int, str]] | None:
    if policy is None:
        return None
    return [(bucket.max_tokens, bucket.backend) for bucket in policy]


def _deserialize_musa_fp8_moe_bucket_policy(
    policy_data: list[tuple[int, str]] | None,
) -> tuple[MusaFp8MoeBucket, ...] | None:
    if policy_data is None:
        return None
    return tuple(
        MusaFp8MoeBucket(max_tokens=int(max_tokens), backend=str(backend))
        for max_tokens, backend in policy_data
    )


def _musa_mixed_deepgemm_static_supported(
    method: object,
    layer: FusedMoE,
) -> bool:
    if not current_platform.is_musa():
        return False

    if not _env_flag_enabled(_MUSA_FP8_MOE_MIXED_BACKEND_ENV):
        return False

    if (
        getattr(method, "is_monolithic", False)
        or getattr(method, "moe_kernel", None) is None
    ):
        return False

    if _fp8_backend_name(method) != "DEEPGEMM":
        return False

    ep_size = getattr(layer, "ep_size", None)
    return ep_size is not None and ep_size <= 1


def _get_musa_mixed_deepgemm_quant_method(layer: FusedMoE) -> object | None:
    candidates = (
        getattr(layer, "quant_method", None),
        getattr(layer, "base_quant_method", None),
        getattr(getattr(layer, "quant_method", None), "old_quant_method", None),
    )
    for method in candidates:
        if method is not None and _musa_mixed_deepgemm_static_supported(method, layer):
            return method
    return None


def _select_musa_fp8_moe_backend(num_tokens: int) -> str:
    if _MUSA_FP8_MOE_FORCE_BACKEND is not None:
        return _MUSA_FP8_MOE_FORCE_BACKEND

    if _env_flag_enabled(_MUSA_FP8_MOE_AUTOTUNE_ENV):
        policy = get_musa_fp8_moe_bucket_policy()
        if policy is None:
            return _MUSA_FP8_MOE_BACKEND_TRITON
        for bucket in policy:
            if num_tokens <= bucket.max_tokens:
                return bucket.backend
        return policy[-1].backend

    if num_tokens < _env_int(
        _MUSA_FP8_MOE_DEEPGEMM_MIN_TOKENS_ENV,
        _MUSA_FP8_MOE_DEEPGEMM_DEFAULT_MIN_TOKENS,
    ):
        return _MUSA_FP8_MOE_BACKEND_TRITON
    return _MUSA_FP8_MOE_BACKEND_DEEPGEMM


def _musa_fp8_moe_per_expert_m(layer: FusedMoE, num_tokens: int) -> float:
    # Average tokens routed to one expert. The modular grouped-GEMM path only
    # amortizes its prepare/finalize/permute overhead at large per-expert M
    # (prefill-sized work); small-M decode is cheaper on the lean legacy path.
    num_experts = max(int(getattr(layer, "global_num_experts", 0) or 0), 1)
    top_k = max(int(getattr(layer, "top_k", 1) or 1), 1)
    return num_tokens * top_k / num_experts


def _should_use_musa_mixed_deepgemm(
    method: object,
    layer: FusedMoE,
    x: torch.Tensor,
) -> bool:
    if not _musa_mixed_deepgemm_static_supported(method, layer):
        return False
    if _select_musa_fp8_moe_backend(x.shape[0]) != _MUSA_FP8_MOE_BACKEND_DEEPGEMM:
        return False
    min_per_expert_m = _env_int(
        _MUSA_FP8_MOE_MODULAR_MIN_PER_EXPERT_M_ENV,
        _MUSA_FP8_MOE_MODULAR_DEFAULT_MIN_PER_EXPERT_M,
    )
    return _musa_fp8_moe_per_expert_m(layer, x.shape[0]) >= min_per_expert_m


@contextmanager
def _force_musa_fp8_moe_backend(backend: str):
    global _MUSA_FP8_MOE_FORCE_BACKEND
    old_backend = _MUSA_FP8_MOE_FORCE_BACKEND
    _MUSA_FP8_MOE_FORCE_BACKEND = backend
    try:
        yield
    finally:
        _MUSA_FP8_MOE_FORCE_BACKEND = old_backend


def _device_synchronize() -> None:
    if hasattr(torch, "musa"):
        torch.musa.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def _find_musa_fp8_moe_autotune_target(
    model: torch.nn.Module,
) -> FusedMoE | None:
    for module in model.modules():
        if not isinstance(module, FusedMoE):
            continue
        if _get_musa_mixed_deepgemm_quant_method(module) is not None:
            return module
    return None


def _musa_fp8_moe_candidate_tokens(max_tokens: int) -> list[int]:
    max_tokens = max(1, max_tokens)
    tokens: list[int] = []
    value = 1
    while value <= max_tokens:
        tokens.append(value)
        value *= 2
    if tokens[-1] != max_tokens:
        tokens.append(max_tokens)
    return tokens


def _build_musa_fp8_moe_autotune_inputs(
    layer: FusedMoE,
    num_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = layer.w13_weight.device
    hidden_size = layer.w13_weight.shape[2]
    act_dtype = getattr(layer, "orig_dtype", torch.bfloat16)
    if act_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        act_dtype = torch.bfloat16

    x = torch.empty((num_tokens, hidden_size), device=device, dtype=act_dtype)
    x.normal_(mean=0.0, std=0.01)

    topk = int(layer.top_k)
    num_experts = int(layer.global_num_experts)
    base_method = getattr(layer, "base_quant_method", None)
    topk_ids_dtype = (
        getattr(base_method, "topk_indices_dtype", None)
        or getattr(layer.quant_method, "topk_indices_dtype", None)
        or torch.int64
    )
    topk_ids = (
        torch.arange(num_tokens * topk, device=device, dtype=torch.int64)
        .remainder(num_experts)
        .reshape(num_tokens, topk)
        .to(dtype=topk_ids_dtype)
    )
    topk_weights = torch.full(
        (num_tokens, topk),
        1.0 / max(topk, 1),
        device=device,
        dtype=torch.float32,
    )
    return x, topk_weights, topk_ids


def _measure_musa_fp8_moe_backend_us(
    layer: FusedMoE,
    backend: str,
    num_tokens: int,
    warmup: int,
    iters: int,
) -> float:
    x, topk_weights, topk_ids = _build_musa_fp8_moe_autotune_inputs(layer, num_tokens)
    method = _get_musa_mixed_deepgemm_quant_method(layer)
    assert method is not None
    times_us: list[float] = []

    with torch.inference_mode(), _force_musa_fp8_moe_backend(backend):
        for _ in range(warmup):
            method.apply(layer, x, topk_weights, topk_ids)
        _device_synchronize()

        for _ in range(iters):
            start = time.perf_counter()
            method.apply(layer, x, topk_weights, topk_ids)
            _device_synchronize()
            times_us.append((time.perf_counter() - start) * 1_000_000.0)

    return statistics.median(times_us)


def _smooth_musa_fp8_moe_points(
    points: list[tuple[int, str]],
) -> list[tuple[int, str]]:
    if len(points) < 3:
        return points

    smoothed = points.copy()
    for i in range(1, len(points) - 1):
        prev_backend = smoothed[i - 1][1]
        next_backend = points[i + 1][1]
        if prev_backend == next_backend and points[i][1] != prev_backend:
            smoothed[i] = (points[i][0], prev_backend)
    return smoothed


def _compress_musa_fp8_moe_policy(
    points: list[tuple[int, str]],
) -> tuple[MusaFp8MoeBucket, ...]:
    if not points:
        return ()

    buckets: list[MusaFp8MoeBucket] = []
    current_backend = points[0][1]
    for index in range(1, len(points)):
        tokens, backend = points[index]
        if backend == current_backend:
            continue
        buckets.append(
            MusaFp8MoeBucket(
                max_tokens=points[index - 1][0],
                backend=current_backend,
            )
        )
        current_backend = backend
    buckets.append(MusaFp8MoeBucket(max_tokens=points[-1][0], backend=current_backend))
    return tuple(buckets)


def _format_musa_fp8_moe_policy(
    policy: tuple[MusaFp8MoeBucket, ...] | None,
) -> str:
    if policy is None:
        return "<none>"
    return ", ".join(
        f"<= {bucket.max_tokens}: {bucket.backend}" for bucket in policy
    )


def maybe_autotune_musa_fp8_moe_policy(worker: object) -> None:
    global _MUSA_FP8_MOE_AUTOTUNE_DONE

    if _MUSA_FP8_MOE_AUTOTUNE_DONE:
        return
    if not _env_flag_enabled(_MUSA_FP8_MOE_AUTOTUNE_ENV):
        return
    if not _env_flag_enabled(_MUSA_FP8_MOE_MIXED_BACKEND_ENV):
        return
    if not current_platform.is_musa():
        return

    _MUSA_FP8_MOE_AUTOTUNE_DONE = True

    group = None
    is_leader = True
    try:
        from vllm.distributed.parallel_state import get_tp_group

        group = get_tp_group()
        is_leader = group.rank_in_group == 0
    except Exception:
        group = None

    policy_data: list[tuple[int, str]] | None = None
    if is_leader:
        try:
            model = worker.get_model()
            target = _find_musa_fp8_moe_autotune_target(model)
            if target is None:
                logger.warning(
                    "Skipping MUSA FP8 MoE autotune: no TP-only FP8 DeepGEMM "
                    "mixed MoE layer found."
                )
            else:
                max_tokens = int(worker.scheduler_config.max_num_batched_tokens)
                override_max_tokens = _env_int(
                    _MUSA_FP8_MOE_AUTOTUNE_MAX_TOKENS_ENV,
                    max_tokens,
                )
                max_tokens = max(1, min(max_tokens, override_max_tokens))
                warmup = max(
                    0,
                    _env_int(
                        _MUSA_FP8_MOE_AUTOTUNE_WARMUP_ENV,
                        _MUSA_FP8_MOE_AUTOTUNE_DEFAULT_WARMUP,
                    ),
                )
                iters = max(
                    1,
                    _env_int(
                        _MUSA_FP8_MOE_AUTOTUNE_ITERS_ENV,
                        _MUSA_FP8_MOE_AUTOTUNE_DEFAULT_ITERS,
                    ),
                )
                win_margin = _env_float(
                    _MUSA_FP8_MOE_AUTOTUNE_WIN_MARGIN_ENV,
                    _MUSA_FP8_MOE_AUTOTUNE_DEFAULT_WIN_MARGIN,
                )
                points: list[tuple[int, str]] = []
                for tokens in _musa_fp8_moe_candidate_tokens(max_tokens):
                    triton_us = _measure_musa_fp8_moe_backend_us(
                        target,
                        _MUSA_FP8_MOE_BACKEND_TRITON,
                        tokens,
                        warmup,
                        iters,
                    )
                    try:
                        deepgemm_us = _measure_musa_fp8_moe_backend_us(
                            target,
                            _MUSA_FP8_MOE_BACKEND_DEEPGEMM,
                            tokens,
                            warmup,
                            iters,
                        )
                    except Exception as exc:
                        logger.warning(
                            "MUSA FP8 MoE autotune DeepGEMM measurement failed "
                            "for %d tokens: %s",
                            tokens,
                            exc,
                        )
                        deepgemm_us = float("inf")
                    backend = (
                        _MUSA_FP8_MOE_BACKEND_DEEPGEMM
                        if deepgemm_us <= triton_us * win_margin
                        else _MUSA_FP8_MOE_BACKEND_TRITON
                    )
                    points.append((tokens, backend))
                    logger.info(
                        "MUSA FP8 MoE autotune tokens=%d triton=%.2fus "
                        "deepgemm=%.2fus winner=%s",
                        tokens,
                        triton_us,
                        deepgemm_us,
                        backend,
                    )

                policy = _compress_musa_fp8_moe_policy(
                    _smooth_musa_fp8_moe_points(points)
                )
                policy_data = _serialize_musa_fp8_moe_bucket_policy(policy)
        except Exception as exc:
            logger.warning("MUSA FP8 MoE autotune failed: %s", exc)

    if group is not None:
        policy_data = group.broadcast_object(policy_data, src=0)

    policy = _deserialize_musa_fp8_moe_bucket_policy(policy_data)
    set_musa_fp8_moe_bucket_policy(policy)
    if policy is None:
        logger.warning(
            "MUSA FP8 MoE autotune installed no policy; autotune mode will "
            "fall back to Triton."
        )
    else:
        logger.info(
            "MUSA FP8 MoE autotune installed bucket policy: %s",
            _format_musa_fp8_moe_policy(policy),
        )


_ORIGINAL_FP8_MOE_MAYBE_ROUNDUP_SIZES = vllm_fp8.Fp8MoEMethod.maybe_roundup_sizes
_ORIGINAL_FP8_MOE_CREATE_WEIGHTS = vllm_fp8.Fp8MoEMethod.create_weights


def maybe_roundup_sizes(
    self,
    hidden_size: int,
    intermediate_size_per_partition: int,
    act_dtype: torch.dtype,
    moe_parallel_config,
) -> tuple[int, int]:
    hidden_size, intermediate_size_per_partition = (
        _ORIGINAL_FP8_MOE_MAYBE_ROUNDUP_SIZES(
            self,
            hidden_size,
            intermediate_size_per_partition,
            act_dtype,
            moe_parallel_config,
        )
    )

    if not (
        current_platform.is_musa()
        and getattr(self, "block_quant", False)
        and getattr(moe_parallel_config, "tp_size", 1) > 1
    ):
        return hidden_size, intermediate_size_per_partition

    weight_block_size = getattr(self, "weight_block_size", None)
    if weight_block_size is None:
        return hidden_size, intermediate_size_per_partition

    block_n, block_k = int(weight_block_size[0]), int(weight_block_size[1])
    block_multiple = math.lcm(block_n, block_k)
    padded_intermediate = _round_up(
        intermediate_size_per_partition,
        block_multiple,
    )
    if padded_intermediate != intermediate_size_per_partition:
        logger.info_once(
            "Padding MUSA FP8 MoE intermediate partition from %d to %d "
            "for block_shape=[%d, %d].",
            intermediate_size_per_partition,
            padded_intermediate,
            block_n,
            block_k,
        )
    return hidden_size, padded_intermediate


def create_weights(
    self,
    layer: torch.nn.Module,
    num_experts: int,
    hidden_size: int,
    intermediate_size_per_partition: int,
    params_dtype: torch.dtype,
    **extra_weight_attrs,
):
    _ORIGINAL_FP8_MOE_CREATE_WEIGHTS(
        self,
        layer=layer,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size_per_partition=intermediate_size_per_partition,
        params_dtype=params_dtype,
        **extra_weight_attrs,
    )

    if not (current_platform.is_musa() and getattr(self, "block_quant", False)):
        return

    unpadded_intermediate = getattr(
        layer.moe_config,
        "intermediate_size_per_partition_unpadded",
        intermediate_size_per_partition,
    )
    if intermediate_size_per_partition == unpadded_intermediate:
        return

    _zero_fp8_weight(layer.w13_weight.data)
    _zero_fp8_weight(layer.w2_weight.data)
    logger.debug(
        "Zero initialized padded MUSA FP8 MoE weights for %s "
        "(intermediate=%d, unpadded=%d).",
        getattr(layer, "prefix", "<unknown>"),
        intermediate_size_per_partition,
        unpadded_intermediate,
    )


def apply(
    self,
    layer: FusedMoE,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_experts: object | None = None,
    shared_experts_input: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if layer.ep_size is not None and layer.ep_size <= 1:
        if _should_use_musa_mixed_deepgemm(self, layer, x):
            logger.info_once(
                "MUSA FP8 MoE mixed backend selected DeepGEMM for %d tokens "
                "(set %s=0 to disable).",
                x.shape[0],
                _MUSA_FP8_MOE_MIXED_BACKEND_ENV,
            )
            assert self.moe_kernel is not None
            return self.moe_kernel.apply(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                activation=layer.activation,
                global_num_experts=layer.global_num_experts,
                expert_map=layer.expert_map,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                shared_experts=shared_experts,
                shared_experts_input=shared_experts_input,
            )

        # the legacy fused_experts() path only computes routed
        # experts. For the no-overlap path used by DeepSeek-V2/V3 on MUSA, the
        # MoE runner computes shared experts separately and combines them with
        # this routed output. Only compute shared experts here when the runner
        # explicitly delegated them to the quant method via MK overlap.
        run_shared_in_quant_method = (
            shared_experts is not None and self.mk_can_overlap_shared_experts
        )
        if run_shared_in_quant_method:
            se_input = shared_experts_input if shared_experts_input is not None else x
        is_inplace = (not is_torch_equal_or_newer("2.9")) and shared_experts is None
        routed = fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=is_inplace,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            expert_map=layer.expert_map,
            quant_config=self.moe_quant_config,
        )
        if not run_shared_in_quant_method:
            return routed
        return routed + shared_experts._layer(se_input)
    else:
        assert not self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )


vllm_fp8.Fp8MoEMethod.maybe_roundup_sizes = maybe_roundup_sizes
vllm_fp8.Fp8MoEMethod.create_weights = create_weights
vllm_fp8.Fp8MoEMethod.apply = apply
