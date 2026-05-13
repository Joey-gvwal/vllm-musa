# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp

from vllm_musa.utils.environ import envs


@SiluAndMul.register_oot
class MusaSiluAndMul(SiluAndMul):
    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        if envs.VLLM_MUSA_CUSTOM_OP_USE_NATIVE.get():
            return self.forward_native(x)

        # ==================== MUSA ADAPTATION ====================
        return F.swish_glu(x)
        # ========================== END ==========================


@SiluAndMulWithClamp.register_oot
class MusaSiluAndMulWithClamp(SiluAndMulWithClamp):
    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)
