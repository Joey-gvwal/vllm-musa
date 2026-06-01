# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch DeepSeek-V4 model CUDA-only runtime gates for MUSA.
"""

PATCHES = [
    (
        """import typing
from collections.abc import Callable, Iterable
""",
        """import os
import typing
from collections.abc import Callable, Iterable
""",
    ),
    (
        """from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    fused_topk_bias,
)
""",
        """from vllm.model_executor.layers.fused_moe.router import (
    fused_topk_bias_router as _musa_fused_topk_bias_router,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    fused_topk_bias,
)
""",
    ),
    (
        """from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    extract_layer_index,
    make_layers,
    maybe_prefix,
)
""",
        """from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    extract_layer_index,
    make_layers,
    maybe_prefix,
)

from vllm_musa.deepseek_v4_fallbacks import (
    enable_deepseek_v4_sparse_correctness_fallbacks as _musa_enable_deepseek_v4_sparse_correctness_fallbacks,
)
from vllm_musa import _custom_ops as _musa_ops


def _musa_deepseek_v4_topk_softplus_sqrt_fallback(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    e_score_correction_bias: torch.Tensor | None = None,
    input_tokens: torch.Tensor | None = None,
    hash_indices_table: torch.Tensor | None = None,
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, ...]:
    scores = F.softplus(gating_output).sqrt()
    scores_for_choice = scores.view(-1, scores.shape[-1])
    if e_score_correction_bias is not None:
        scores_for_choice = scores_for_choice + e_score_correction_bias.unsqueeze(0)
    if hash_indices_table is not None:
        if input_tokens is None:
            raise ValueError(
                "input_tokens is required when hash_indices_table is provided"
            )
        topk_selected = hash_indices_table[input_tokens]
    else:
        topk_selected = torch.topk(
            scores_for_choice,
            k=topk_indices.shape[1],
            dim=-1,
            sorted=_musa_fused_topk_bias_router.envs.VLLM_BATCH_INVARIANT,
        )[1]
    selected_weights = scores.gather(1, topk_selected.to(torch.long))
    if renormalize:
        selected_weights = selected_weights / selected_weights.sum(
            dim=-1, keepdim=True
        )
    if routed_scaling_factor != 1.0:
        selected_weights = selected_weights * routed_scaling_factor
    topk_weights.copy_(selected_weights.to(topk_weights.dtype))
    topk_indices.copy_(topk_selected.to(topk_indices.dtype))
    token_expert_indices.copy_(topk_selected.to(token_expert_indices.dtype))
    return topk_weights, topk_indices


def _musa_deepseek_v4_topk_softplus_sqrt_native(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    e_score_correction_bias: torch.Tensor | None = None,
    input_tokens: torch.Tensor | None = None,
    hash_indices_table: torch.Tensor | None = None,
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, ...]:
    _musa_ops.deepseek_v4_topk_softplus_sqrt(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
        routed_scaling_factor,
        e_score_correction_bias,
        input_tokens,
        hash_indices_table,
    )
    return topk_weights, topk_indices


if current_platform.is_musa() or getattr(torch.version, "musa", None) is not None:
    _musa_enable_deepseek_v4_sparse_correctness_fallbacks()

if current_platform.is_musa() or getattr(torch.version, "musa", None) is not None:
    if os.getenv("VLLM_MUSA_ENABLE_TORCH_TOPK_SOFTPLUS_SQRT_FALLBACK", "0") == "1":
        _musa_fused_topk_bias_router.vllm_topk_softplus_sqrt = (
            _musa_deepseek_v4_topk_softplus_sqrt_fallback
        )
    else:
        _musa_fused_topk_bias_router.vllm_topk_softplus_sqrt = (
            _musa_deepseek_v4_topk_softplus_sqrt_native
        )


""",
    ),
    (
        """from vllm.model_executor.layers.quantization.fp8 import Fp8Config
""",
        """from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
""",
    ),
    (
        """    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "deepseek_v4_fp8"

    @classmethod
    def override_quantization_method(
""",
        """    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "deepseek_v4_fp8"

    @classmethod
    def get_min_capability(cls) -> int:
        if (
            current_platform.is_musa()
            or getattr(torch.version, "musa", None) is not None
        ):
            return 31
        return super().get_min_capability()

    @classmethod
    def override_quantization_method(
""",
    ),
    (
        """    def get_quant_method(self, layer, prefix):
        if isinstance(layer, FusedMoE):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            return Mxfp4MoEMethod(layer.moe_config)
        return super().get_quant_method(layer, prefix)
""",
        """    def get_quant_method(self, layer, prefix):
        if isinstance(layer, FusedMoE):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            if (
                current_platform.is_musa()
                or getattr(torch.version, "musa", None) is not None
            ):
                return Fp8MoEMethod(self, layer)
            return Mxfp4MoEMethod(layer.moe_config)
        return super().get_quant_method(layer, prefix)
""",
    ),
    (
        """    def is_mxfp4_quant(self, prefix, layer):
        return isinstance(layer, FusedMoE)
""",
        """    def is_mxfp4_quant(self, prefix, layer):
        if not isinstance(layer, FusedMoE):
            return False
        if (
            current_platform.is_musa()
            or getattr(torch.version, "musa", None) is not None
        ):
            return False
        return True
""",
    ),
    (
        """    def _check_runtime_supported(self) -> None:
        if not torch.cuda.is_available():
            raise NotImplementedError("DeepSeek V4 MegaMoE requires CUDA.")
        device = self.w13_weight.device
        if device.type != "cuda":
            raise NotImplementedError(
                "DeepSeek V4 MegaMoE expert weights must be loaded on CUDA."
            )
        if torch.cuda.get_device_capability(device)[0] != 10:
            raise NotImplementedError("DeepGEMM MegaMoE requires SM100 GPUs.")
        if self.hidden_size % 128 != 0 or self.intermediate_size % 128 != 0:
            raise ValueError(
                "DeepGEMM MegaMoE requires hidden and intermediate sizes "
                "to be multiples of 128."
            )
""",
        """    def _check_runtime_supported(self) -> None:
        device = self.w13_weight.device
        if (
            current_platform.is_musa()
            or getattr(torch.version, "musa", None) is not None
            or device.type == "musa"
        ):
            raise NotImplementedError(
                "DeepSeek V4 MegaMoE is not implemented for MUSA yet. "
                "A MUSA MegaMoE replacement path is "
                "required before enabling deep_gemm_mega_moe."
            )
        if not torch.cuda.is_available():
            raise NotImplementedError("DeepSeek V4 MegaMoE requires CUDA.")
        if device.type != "cuda":
            raise NotImplementedError(
                "DeepSeek V4 MegaMoE expert weights must be loaded on CUDA."
            )
        if torch.cuda.get_device_capability(device)[0] != 10:
            raise NotImplementedError("DeepGEMM MegaMoE requires SM100 GPUs.")
        if self.hidden_size % 128 != 0 or self.intermediate_size % 128 != 0:
            raise ValueError(
                "DeepGEMM MegaMoE requires hidden and intermediate sizes "
                "to be multiples of 128."
            )
""",
    ),
]

# DeepSeek-V4-Flash-Base FP8 checkpoints use per-expert names like
# `experts.0.w1.scale` / `experts.0.w1.weight`. Upstream v0.20.0 only maps
# the `weight.` form to `w13_weight` / `w2_weight`; add scale aliases and
# resolve them to either MegaMoE `*_weight_scale` or FP8 FusedMoE
# `*_weight_scale_inv` parameters at load time.
PATCHES.extend(
    [
        (
            '            "experts.w13_" if shard_id in ("w1", "w3") else "experts.w2_",\n            f"experts.{expert_id}.{weight_name}.",\n            expert_id,\n            shard_id,\n        )\n        for expert_id in range(num_experts)\n        for shard_id, weight_name in [\n            ("w1", "w1"),\n            ("w2", "w2"),\n            ("w3", "w3"),\n        ]\n    ]\n',
            '            "experts.w13_" if shard_id in ("w1", "w3") else "experts.w2_",\n            f"experts.{expert_id}.{weight_name}.",\n            expert_id,\n            shard_id,\n        )\n        for expert_id in range(num_experts)\n        for shard_id, weight_name in [\n            ("w1", "w1"),\n            ("w2", "w2"),\n            ("w3", "w3"),\n        ]\n    ] + [\n        (\n            "experts.w13_weight_scale"\n            if shard_id in ("w1", "w3")\n            else "experts.w2_weight_scale",\n            f"experts.{expert_id}.{weight_name}.scale",\n            expert_id,\n            shard_id,\n        )\n        for expert_id in range(num_experts)\n        for shard_id, weight_name in [\n            ("w1", "w1"),\n            ("w2", "w2"),\n            ("w3", "w3"),\n        ]\n    ]\n',
        ),
        (
            '                    if (\n                        "weight_scale" in name\n                        and loaded_weight.dtype == torch.float8_e8m0fnu\n                    ):\n                        loaded_weight = loaded_weight.view(torch.uint8)\n',
            "                    # MUSA: convert E8M0 expert scales after resolving the target param.\n",
        ),
        (
            "                    if loaded_weight.dtype == torch.float8_e8m0fnu:\n                        loaded_weight = loaded_weight.view(torch.uint8)\n",
            "                    # MUSA: convert E8M0 expert scales after resolving the target param.\n",
        ),
        (
            "                    for mapping in expert_mapping:\n                        param_name, weight_name, expert_id, shard_id = mapping\n                        if weight_name not in name:\n                            continue\n                        name_mapped = name.replace(weight_name, param_name)\n                        param = params_dict[name_mapped]\n                        # We should ask the weight loader to return success or not\n",
            '                    name_mapped = name\n                    for mapping in expert_mapping:\n                        param_name, weight_name, expert_id, shard_id = mapping\n                        if weight_name not in name:\n                            continue\n                        name_mapped = name.replace(weight_name, param_name)\n                        if name_mapped not in params_dict and name_mapped.endswith(\n                            "_weight_scale"\n                        ):\n                            inv_name_mapped = f"{name_mapped}_inv"\n                            if inv_name_mapped in params_dict:\n                                name_mapped = inv_name_mapped\n                        if name_mapped not in params_dict:\n                            continue\n                        param = params_dict[name_mapped]\n                        loaded_weight_for_param = loaded_weight\n                        if loaded_weight_for_param.dtype == torch.float8_e8m0fnu:\n                            if param.dtype == torch.uint8:\n                                loaded_weight_for_param = loaded_weight_for_param.view(\n                                    torch.uint8\n                                )\n                            else:\n                                loaded_weight_for_param = loaded_weight_for_param.to(\n                                    param.dtype\n                                )\n                        # We should ask the weight loader to return success or not\n',
        ),
        (
            "                        success = weight_loader(\n                            param,\n                            loaded_weight,\n                            name_mapped,\n",
            "                        success = weight_loader(\n                            param,\n                            loaded_weight_for_param,\n                            name_mapped,\n",
        ),
    ]
)

PATCHES.extend(
    [
        (
            """@torch.compile(backend=current_platform.simple_compile_backend)
def hc_head(
    hidden_states: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    x = hidden_states
    shape, dtype = x.size(), x.dtype
    x = x.flatten(1).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_norm_eps)
    mixes = F.linear(x, hc_fn) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
    return y.to(dtype)
""",
            """def hc_head(
    hidden_states: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    if (
        current_platform.is_musa()
        and os.getenv("VLLM_MUSA_DEEPSEEK_V4_HC_HEAD_COMPILE", "0") != "1"
    ):
        return _musa_deepseek_v4_hc_head_eager(
            hidden_states,
            hc_fn,
            hc_scale,
            hc_base,
            rms_norm_eps,
            hc_eps,
        )
    return _compiled_hc_head(
        hidden_states,
        hc_fn,
        hc_scale,
        hc_base,
        rms_norm_eps,
        hc_eps,
    )


@torch.compile(backend=current_platform.simple_compile_backend)
def _compiled_hc_head(
    hidden_states: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    x = hidden_states
    shape, dtype = x.size(), x.dtype
    x = x.flatten(1).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_norm_eps)
    mixes = F.linear(x, hc_fn) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
    return y.to(dtype)


def _musa_deepseek_v4_hc_head_eager(
    hidden_states: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    x = hidden_states
    shape, dtype = x.size(), x.dtype
    x = x.flatten(1).to(torch.float32)
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_norm_eps)
    mixes = F.linear(x, hc_fn.to(torch.float32)) * rsqrt
    pre = (
        torch.sigmoid(mixes * hc_scale.to(torch.float32) + hc_base.to(torch.float32))
        + hc_eps
    )
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
    return y.to(dtype)
""",
        ),
    ]
)
