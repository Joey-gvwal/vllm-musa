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
    from vllm_musa.deepseek_v4_jit.topk import (
        try_tilelang_biased_topk_softplus_sqrt,
    )

    used_tilelang, tilelang_reason = try_tilelang_biased_topk_softplus_sqrt(
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
    if used_tilelang:
        return topk_weights, topk_indices
    router_topk_mode = os.getenv(
        "VLLM_MUSA_DEEPSEEK_V4_ROUTER_TOPK_IMPL", "torch"
    ).strip().lower()
    if (
        router_topk_mode in {
        "tilelang",
        "jit",
        "warp",
        "warp_tilelang",
        "tilelang_warp",
        "fast",
        }
        and tilelang_reason != "hash-routed path stays on the existing fallback"
    ):
        raise RuntimeError(
            "Forced MUSA TileLang DeepSeek-V4 router top-k failed: "
            f"{tilelang_reason}"
        )

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


if current_platform.is_musa() or getattr(torch.version, "musa", None) is not None:
    _musa_enable_deepseek_v4_sparse_correctness_fallbacks()

if (
    current_platform.is_musa() or getattr(torch.version, "musa", None) is not None
) and os.getenv("VLLM_MUSA_ENABLE_TORCH_TOPK_SOFTPLUS_SQRT_FALLBACK", "0") == "1":
    _musa_fused_topk_bias_router.vllm_topk_softplus_sqrt = (
        _musa_deepseek_v4_topk_softplus_sqrt_fallback
    )
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
                "A MUSA fp8_fp4_mega_moe or supported replacement path is "
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
