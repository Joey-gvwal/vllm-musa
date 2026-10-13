# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_musa.utils.environ import envs


@SiluAndMul.register_oot
class MusaSiluAndMul(SiluAndMul):
    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        if envs.VLLM_MUSA_CUSTOM_OP_USE_NATIVE.get():
            return self.forward_native(x)

        # ==================== MUSA ADAPTATION ====================
        return torch.ops.vllm.musa_swish_glu_op(x)
        # ========================== END ==========================


@SiluAndMulWithClamp.register_oot
class MusaSiluAndMulWithClamp(SiluAndMulWithClamp):
    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        if envs.VLLM_MUSA_CUSTOM_OP_USE_NATIVE.get():
            return self.forward_native(x)
        out = torch.empty(
            x.shape[:-1] + (x.shape[-1] // 2,), dtype=x.dtype, device=x.device
        )
        torch.ops._C.silu_and_mul_with_clamp(
            out, x, self.swiglu_limit, self.alpha, self.beta
        )
        return out


def _musa_swish_glu_op(x: torch.Tensor) -> torch.Tensor:
    return F.swish_glu(x)


def _musa_swish_glu_op_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty(
        x.shape[:-1] + (x.shape[-1] // 2,),
        dtype=x.dtype,
        device=x.device,
    )


direct_register_custom_op(
    "musa_swish_glu_op",
    _musa_swish_glu_op,
    fake_impl=_musa_swish_glu_op_fake,
)
