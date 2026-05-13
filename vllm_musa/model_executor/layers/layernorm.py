# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn
from vllm.model_executor.layers.layernorm import RMSNorm, fused_add_rms_norm

from vllm_musa.utils.environ import envs


def _vllm_fused_add_rms_norm_available() -> bool:
    return (
        getattr(getattr(torch.ops, "_C", None), "fused_add_rms_norm", None)
        is not None
    )


@RMSNorm.register_oot
class MusaRMSNorm(RMSNorm):
    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if envs.VLLM_MUSA_CUSTOM_OP_USE_NATIVE.get():
            return self.forward_native(x, residual)

        # ==================== MUSA ADAPTATION ====================
        if self.variance_size_override is not None:
            return self.forward_native(x, residual)

        add_residual = residual is not None
        if add_residual:
            if not _vllm_fused_add_rms_norm_available():
                return self.forward_native(x, residual)
            return fused_add_rms_norm(
                x, residual, self.weight.data, self.variance_epsilon
            )
        else:
            out = nn.functional.rms_norm(
                x, (self.hidden_size,), self.weight.data, self.variance_epsilon
            )
            return out
        # ========================== END ==========================
