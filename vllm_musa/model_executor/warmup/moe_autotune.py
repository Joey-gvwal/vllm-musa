# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger

logger = init_logger(__name__)


def _install_musa_fp8_moe_autotune_hook() -> None:
    import vllm.model_executor.warmup.kernel_warmup as kernel_warmup_module

    original = getattr(
        kernel_warmup_module,
        "_musa_original_kernel_warmup",
        kernel_warmup_module.kernel_warmup,
    )
    if not hasattr(kernel_warmup_module, "_musa_original_kernel_warmup"):
        kernel_warmup_module._musa_original_kernel_warmup = original

    def kernel_warmup(worker):
        result = original(worker)
        try:
            from vllm_musa.model_executor.warmup.moe_dispatch_debug import (
                install_musa_fp8_moe_dispatch_debug,
            )

            install_musa_fp8_moe_dispatch_debug()
        except Exception as exc:
            logger.warning("MUSA FP8 MoE dispatch debug hook failed: %s", exc)
        try:
            from vllm_musa.model_executor.layers.quantization.fp8 import (
                maybe_autotune_musa_fp8_moe_policy,
            )

            maybe_autotune_musa_fp8_moe_policy(worker)
        except Exception as exc:
            logger.warning("MUSA FP8 MoE autotune hook failed: %s", exc)
        try:
            from vllm_musa.model_executor.warmup.moe_microbench import (
                maybe_run_musa_fp8_moe_microbench,
            )

            maybe_run_musa_fp8_moe_microbench(worker)
        except Exception as exc:
            logger.warning("MUSA FP8 MoE microbench hook failed: %s", exc)
        return result

    if getattr(kernel_warmup_module.kernel_warmup, "__name__", "") != "kernel_warmup":
        return

    kernel_warmup_module.kernel_warmup = kernel_warmup

    try:
        import vllm.v1.worker.gpu_worker as gpu_worker_module

        if getattr(gpu_worker_module, "kernel_warmup", None) is original:
            gpu_worker_module.kernel_warmup = kernel_warmup
    except Exception:
        pass


_install_musa_fp8_moe_autotune_hook()
