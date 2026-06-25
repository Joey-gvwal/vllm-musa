# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Log which inner expert impl the modular FP8 MoE path selects at runtime.

Opt-in via ``VLLM_MUSA_FP8_MOE_DISPATCH_DEBUG=1``; default off and does not
change dispatch behavior. ``TritonOrDeepGemmExperts`` decides DeepGEMM vs a
silent Triton fallback through ``_valid_deep_gemm``; this wrapper records that
decision (with the M/N/K it saw) so the actual path is observable instead of
inferred from the coarse outer "selected DeepGEMM" log line.
"""

import os

from vllm.logger import init_logger

logger = init_logger(__name__)

_DISPATCH_DEBUG_ENV = "VLLM_MUSA_FP8_MOE_DISPATCH_DEBUG"

_seen_decisions: set[tuple[str, int]] = set()
_decision_counts: dict[tuple[str, int], int] = {}


def _dispatch_debug_enabled() -> bool:
    value = os.environ.get(_DISPATCH_DEBUG_ENV, "0")
    return value.lower() not in {"", "0", "false", "no", "off"}


def install_musa_fp8_moe_dispatch_debug() -> None:
    if not _dispatch_debug_enabled():
        return
    try:
        import vllm.model_executor.layers.fused_moe.experts.triton_deep_gemm_moe as tdg
    except Exception as exc:
        logger.warning("MUSA FP8 MoE dispatch debug: cannot import dispatcher: %s", exc)
        return

    if getattr(tdg, "_musa_dispatch_debug_installed", False):
        return
    original = tdg._valid_deep_gemm

    def _valid_deep_gemm_logged(hidden_states, w1, w2):
        result = original(hidden_states, w1, w2)
        try:
            m = int(hidden_states.size(0))
            _, k, n = (int(dim) for dim in w2.size())
            branch = "deepgemm" if result else "triton_fallback"
            key = (branch, n)
            _decision_counts[key] = _decision_counts.get(key, 0) + 1
            if key not in _seen_decisions:
                _seen_decisions.add(key)
                logger.info(
                    "MUSA FP8 MoE inner dispatch=%s (M=%d N=%d K=%d); "
                    "N is intermediate_size_per_partition, N<=512 forces triton.",
                    branch,
                    m,
                    n,
                    k,
                )
            elif _decision_counts[key] % 2000 == 0:
                logger.info(
                    "MUSA FP8 MoE inner dispatch=%s N=%d count=%d",
                    branch,
                    n,
                    _decision_counts[key],
                )
        except Exception:
            pass
        return result

    tdg._valid_deep_gemm = _valid_deep_gemm_logged
    tdg._musa_dispatch_debug_installed = True
    logger.info(
        "MUSA FP8 MoE dispatch debug installed (set %s=0 to disable).",
        _DISPATCH_DEBUG_ENV,
    )
