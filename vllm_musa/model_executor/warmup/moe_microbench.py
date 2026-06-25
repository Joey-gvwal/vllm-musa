# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Cold-cache MoE backend microbench, run in-worker during kernel warmup.

Opt-in via ``VLLM_MUSA_FP8_MOE_MICROBENCH=1``; default off. Measures three real
expert implementations on the loaded FP8 MoE layer so a backend decision rests
on the kernel that actually runs:

* ``triton``         — legacy ``fused_experts`` (the serving baseline path)
* ``modular``        — ``TritonOrDeepGemmExperts`` as-is (falls back to Triton
                       when ``_valid_deep_gemm`` rejects the shape, e.g. N<=512)
* ``deepgemm``       — ``TritonOrDeepGemmExperts`` with the validity gate forced
                       open, so ``DeepGemmExperts`` actually runs

Timing uses ``mate.bench_gpu_time_with_musa_event`` with ``l2_flush=True`` so
every iteration sees a cold L2, unlike the warm host-timer loop the autotune
uses. The op is multi-kernel (quant + permute + 2 grouped GEMMs + activation +
unpermute), so op-level event timing is the correct instrument here.

Runs only on the TP leader; results are logged as a table to grep from the
worker log. Decision axis printed is per-expert M (tokens*top_k/num_experts)
and N (intermediate_size_per_partition), not raw token count.
"""

import contextlib
import os
import statistics

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

_MICROBENCH_ENV = "VLLM_MUSA_FP8_MOE_MICROBENCH"
_MICROBENCH_TOKENS_ENV = "VLLM_MUSA_FP8_MOE_MICROBENCH_TOKENS"
_DEFAULT_TOKEN_SWEEP = (64, 256, 512, 1024, 2048, 4096, 8192)

# S5000 roofline (mtforge/docs/HARDWARE.md, .claude/rules/musa-kernel-bench.md).
_S5000_FP8_TFLOPS = 1000.0
_S5000_BW_GBPS = 1600.0


def _enabled() -> bool:
    value = os.environ.get(_MICROBENCH_ENV, "0")
    return value.lower() not in {"", "0", "false", "no", "off"}


def _token_sweep(max_tokens: int) -> list[int]:
    override = os.environ.get(_MICROBENCH_TOKENS_ENV, "")
    if override.strip():
        try:
            tokens = [int(t) for t in override.replace(",", " ").split()]
        except ValueError:
            tokens = list(_DEFAULT_TOKEN_SWEEP)
    else:
        tokens = list(_DEFAULT_TOKEN_SWEEP)
    return [t for t in tokens if 1 <= t <= max_tokens] or [min(max_tokens, 64)]


@contextlib.contextmanager
def _force_valid_deep_gemm(value: bool):
    import vllm.model_executor.layers.fused_moe.experts.triton_deep_gemm_moe as tdg

    original = tdg._valid_deep_gemm
    tdg._valid_deep_gemm = lambda *args, **kwargs: value
    try:
        yield
    finally:
        tdg._valid_deep_gemm = original


def _median_us(fn) -> float:
    from mate.testing.utils import bench_gpu_time_with_musa_event

    times_ms = bench_gpu_time_with_musa_event(fn, l2_flush=True, l2_flush_size_mb=1024)
    return statistics.median(times_ms) * 1000.0


def maybe_run_musa_fp8_moe_microbench(worker: object) -> None:
    if not _enabled() or not current_platform.is_musa():
        return

    is_leader = True
    try:
        from vllm.distributed.parallel_state import get_tp_group

        is_leader = get_tp_group().rank_in_group == 0
    except Exception:
        is_leader = True
    if not is_leader:
        return

    from vllm_musa.model_executor.layers.quantization.fp8 import (
        _build_musa_fp8_moe_autotune_inputs,
        _force_musa_fp8_moe_backend,
        _find_musa_fp8_moe_autotune_target,
        _get_musa_mixed_deepgemm_quant_method,
    )

    try:
        model = worker.get_model()
        layer = _find_musa_fp8_moe_autotune_target(model)
        if layer is None:
            logger.warning("MUSA FP8 MoE microbench: no eligible FP8 MoE layer found.")
            return
        method = _get_musa_mixed_deepgemm_quant_method(layer)
        assert method is not None

        num_experts = int(layer.w2_weight.shape[0])
        hidden = int(layer.w2_weight.shape[1])
        inter_pp = int(layer.w2_weight.shape[2])
        topk = int(layer.top_k)
        max_tokens = int(worker.scheduler_config.max_num_batched_tokens)
        sweep = _token_sweep(max_tokens)

        logger.info(
            "MUSA FP8 MoE microbench: E=%d hidden(K)=%d N=inter_pp=%d top_k=%d "
            "(N<=512 -> modular falls back to Triton). Cold-cache (l2_flush).",
            num_experts,
            hidden,
            inter_pp,
            topk,
        )
        logger.info(
            "MUSA FP8 MoE microbench header: tokens per_expert_M triton_us "
            "modular_us deepgemm_us winner deepgemm_TFLOPS deepgemm_GBs"
        )

        def _run(backend: str, num_tokens: int):
            x, tw, tids = _build_musa_fp8_moe_autotune_inputs(layer, num_tokens)

            def _fn():
                with _force_musa_fp8_moe_backend(backend):
                    method.apply(layer, x, tw, tids)

            return _fn

        for tokens in sweep:
            per_expert_m = tokens * topk / max(num_experts, 1)
            with torch.inference_mode():
                try:
                    triton_us = _median_us(_run("triton", tokens))
                except Exception as exc:
                    logger.warning("microbench triton failed @%d: %s", tokens, exc)
                    triton_us = float("inf")
                try:
                    modular_us = _median_us(_run("deepgemm", tokens))
                except Exception as exc:
                    logger.warning("microbench modular failed @%d: %s", tokens, exc)
                    modular_us = float("inf")
                try:
                    with _force_valid_deep_gemm(True):
                        deepgemm_us = _median_us(_run("deepgemm", tokens))
                except Exception as exc:
                    logger.warning("microbench deepgemm failed @%d: %s", tokens, exc)
                    deepgemm_us = float("inf")

            # Roofline for the real-DeepGEMM op: FP8 weights dominate cold bytes.
            flops = tokens * topk * 6.0 * hidden * inter_pp
            weight_bytes = num_experts * 3.0 * inter_pp * hidden  # FP8 = 1 byte
            tflops = (flops / (deepgemm_us * 1e-6)) / 1e12 if deepgemm_us not in (0, float("inf")) else 0.0
            gbs = (weight_bytes / (deepgemm_us * 1e-6)) / 1e9 if deepgemm_us not in (0, float("inf")) else 0.0

            best = min(
                ("triton", triton_us),
                ("modular", modular_us),
                ("deepgemm", deepgemm_us),
                key=lambda kv: kv[1],
            )[0]
            logger.info(
                "MUSA FP8 MoE microbench row: tokens=%d per_expert_M=%.1f "
                "triton_us=%.1f modular_us=%.1f deepgemm_us=%.1f winner=%s "
                "deepgemm_TFLOPS=%.1f deepgemm_GBs=%.1f",
                tokens,
                per_expert_m,
                triton_us,
                modular_us,
                deepgemm_us,
                best,
                tflops,
                gbs,
            )

        logger.info(
            "MUSA FP8 MoE microbench: roofline ref S5000 FP8=%.0f TFLOPS "
            "BW=%.0f GB/s; a memory-bound MoE near BW roofline is the ceiling.",
            _S5000_FP8_TFLOPS,
            _S5000_BW_GBPS,
        )
    except Exception as exc:
        logger.warning("MUSA FP8 MoE microbench failed: %s", exc)
