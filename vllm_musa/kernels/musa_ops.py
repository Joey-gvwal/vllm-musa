# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA IR-op providers for vllm.ir.ops.*

Currently registers:
- rms_norm: delegates to torch.ops._C.rms_norm only when that op has a MUSA
  dispatch kernel. In vLLM v0.22 the upstream layernorm implementation moved
  to _C_stable_libtorch; vllm-musa builds that extension as a schema-only
  compatibility shim, so this provider self-disables until a real MUSA kernel
  is present.

Engine log from MUSA-0049 baseline confirms:
    ir_op_priority=IrOpPriorityConfig(rms_norm=['native'])

i.e. before this provider lands, every decoder layer per token uses
the pure-PyTorch native rms_norm. With the "musa" provider plus a
priority override in `vllm_musa.platform`, the dispatcher / Inductor
lowering picks the MUSA kernel when `_dispatch_has_kernel_for_dispatch_key`
confirms one is registered.
"""

import torch
from torch import Tensor
from vllm import ir
from vllm.platforms import current_platform

# vllm._C must be loaded before torch.ops._C.rms_norm is resolvable.
current_platform.import_kernels()


def _has_C_rms_norm() -> bool:
    try:
        if not hasattr(torch.ops._C, "rms_norm"):
            return False
        return torch._C._dispatch_has_kernel_for_dispatch_key(
            "_C::rms_norm", "MUSA"
        )
    except Exception:
        return False


MUSA_RMS_NORM_SUPPORTED = current_platform.is_musa() and _has_C_rms_norm()


def _rms_norm_supports_args(
    x: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> bool:
    return (
        variance_size is None
        and weight is not None
        and x.dim() >= 2
        and x.dtype in (torch.float16, torch.bfloat16)
        and weight.dtype == x.dtype
        and weight.is_contiguous()
    )


@ir.ops.rms_norm.register_impl(
    "musa",
    supports_args=_rms_norm_supports_args,
    supported=MUSA_RMS_NORM_SUPPORTED,
)
def rms_norm(
    x: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> Tensor:
    """MUSA provider for vllm.ir.ops.rms_norm.

    Delegates to torch.ops._C.rms_norm only when a MUSA dispatch kernel is
    registered for the upstream op namespace.
    """
    assert variance_size is None
    assert weight is not None
    output = torch.empty_like(x)
    torch.ops._C.rms_norm(output, x, weight, epsilon)
    return output
