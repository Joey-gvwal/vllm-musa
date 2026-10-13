# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os

import torch
from vllm.model_executor.kernels.linear import FP8ScaledMMLinearLayerConfig
from vllm.model_executor.kernels.linear.scaled_mm.triton import (
    TritonFp8BlockScaledMMKernel,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _upcast_e8m0_to_fp32,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from vllm_musa.kernels.fp8_block_gemm import (
    fp8_block_scaled_mm,
    fp8_block_scaled_mm_supported,
)
from vllm_musa.kernels.fp8_block_gemv import (
    fp8_block_gemv,
    fp8_block_gemv_quant,
    fp8_block_gemv_quant_supported,
    fp8_block_gemv_supported,
)

# The GEMV re-reads the weight per row (or per few rows); past a few rows the
# block GEMM wins. Crossovers measured on MTT S5000 for the DeepSeek-V4.1 TP8
# projections, keyed by (N, K); other shapes use the default.
_GEMV_MAX_M_DEFAULT = 8
_GEMV_MAX_M_MEASURED = {
    (576, 5120): 32,
    (1792, 5120): 12,
    (4096, 1280): 6,
    (5120, 1024): 6,
    (5120, 288): 4,
    (25600, 6144): 2,
}


def _env_int(name: str) -> int | None:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return None


_GEMV_MAX_M_OVERRIDE = _env_int("VLLM_MUSA_FP8_BLOCK_GEMV_MAX_M")


def _gemv_max_m(weight: torch.Tensor) -> int:
    if _GEMV_MAX_M_OVERRIDE is not None:
        return _GEMV_MAX_M_OVERRIDE
    return _GEMV_MAX_M_MEASURED.get(tuple(weight.shape), _GEMV_MAX_M_DEFAULT)


class MUSAFp8BlockGemvScaledMMKernel(TritonFp8BlockScaledMMKernel):
    """Block-FP8 linear for scale blocks finer than DeepGEMM's 128.

    Small M runs the block GEMV, which also quantizes the activation; larger M
    keeps the separate activation quantization and runs a block GEMM whose K
    tile spans several scale groups.
    """

    @classmethod
    def is_supported(cls, compute_capability=None):
        if not current_platform.is_musa():
            return False, "only supported on MUSA"
        return True, None

    @classmethod
    def can_implement(cls, config: FP8ScaledMMLinearLayerConfig):
        can_implement_base, reason = super().can_implement(config)
        if not can_implement_base:
            return can_implement_base, reason
        if config.out_dtype != torch.bfloat16:
            return False, "Supports only output dtype of bfloat16"
        act_group = config.activation_quant_key.scale.group_shape
        weight_group = config.weight_quant_key.scale.group_shape
        if weight_group.col not in (16, 32, 64) or act_group.row != 1:
            return False, "Supports only K blocks of 16, 32 or 64."
        if act_group.col != weight_group.col:
            return False, "Activation groups must match the weight K block."
        return True, None

    def process_weights_after_loading(self, layer):
        super().process_weights_after_loading(layer)
        params = self._get_layer_params(layer)
        scale = params.block_scale
        fp32_scale = (
            _upcast_e8m0_to_fp32(scale)
            if scale.dtype == torch.float8_e8m0fnu
            else scale.to(torch.float32)
        ).contiguous()
        if fp32_scale is not scale:
            replace_parameter(layer, params.block_scale_attr, fp32_scale)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        group_size = self.config.activation_quant_key.scale.group_shape.col
        rows = x.numel() // x.shape[-1] if x.dim() > 0 else 0
        params = self._get_layer_params(layer)
        if (
            0 < rows <= _gemv_max_m(params.weight)
            and x.dtype in (torch.bfloat16, torch.float16)
            and x.stride(-1) == 1
            and params.input_scale is None
            and fp8_block_gemv_quant_supported(
                params.weight, params.block_scale, group_size
            )
        ):
            x_2d = x.reshape(rows, x.shape[-1])
            output = torch.empty(
                (rows, params.weight.shape[0]),
                dtype=self.config.out_dtype,
                device=x.device,
            )
            fp8_block_gemv_quant(
                x_2d, params.weight, params.block_scale, output, group_size
            )
            if bias is not None:
                output = output + bias
            return output.view(*x.shape[:-1], params.weight.shape[0])
        return super().apply_weights(layer, x, bias, **kwargs)

    def apply_swiglu_clamp_weights(
        self,
        layer: torch.nn.Module,
        gate_up: torch.Tensor,
        swiglu_limit: float,
    ) -> torch.Tensor | None:
        """``apply_weights`` of the clamped SwiGLU of ``gate_up`` as one GEMV.

        Returns ``None`` outside the small-M GEMV path.
        """
        group_size = self.config.activation_quant_key.scale.group_shape.col
        rows = gate_up.numel() // gate_up.shape[-1] if gate_up.dim() > 0 else 0
        params = self._get_layer_params(layer)
        if not (
            0 < rows <= _gemv_max_m(params.weight)
            and gate_up.dtype == torch.bfloat16
            and gate_up.is_contiguous()
            and gate_up.shape[-1] == 2 * params.weight.shape[1]
            and params.input_scale is None
            and fp8_block_gemv_quant_supported(
                params.weight, params.block_scale, group_size
            )
        ):
            return None
        output = torch.empty(
            (rows, params.weight.shape[0]),
            dtype=self.config.out_dtype,
            device=gate_up.device,
        )
        fp8_block_gemv_quant(
            gate_up.view(rows, gate_up.shape[-1]),
            params.weight,
            params.block_scale,
            output,
            group_size,
            swiglu_limit=swiglu_limit,
        )
        return output.view(*gate_up.shape[:-1], params.weight.shape[0])

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        if (
            A.dim() == 2
            and A.shape[0] <= _gemv_max_m(B)
            and A.stride(-1) == 1
            and As.dim() == 2
            and As.stride(-1) == 1
            and fp8_block_gemv_supported(As, B, Bs)
        ):
            output = torch.empty(
                (A.shape[0], B.shape[0]), dtype=self.config.out_dtype, device=A.device
            )
            return fp8_block_gemv(A, As, B, Bs, output)
        if fp8_block_scaled_mm_supported(A, As, B, Bs):
            return fp8_block_scaled_mm(A, As, B, Bs, self.config.out_dtype)
        return super().apply_block_scaled_mm(A, B, As, Bs)
