# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch DeepSeek-V4 sqrtsoftplus MoE routing with a MUSA diagnostic fallback.
"""

PATCHES = [
    (
        """import functools
from collections.abc import Callable
""",
        """import functools
import os
from collections.abc import Callable
""",
    ),
    (
        """    ops.topk_hash_softplus_sqrt(
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
""",
        """    if (
        (
            gating_output.device.type == "musa"
            or getattr(torch.version, "musa", None) is not None
        )
        and os.getenv(
            "VLLM_MUSA_ENABLE_TORCH_TOPK_SOFTPLUS_SQRT_FALLBACK", "0"
        )
        == "1"
    ):
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
                sorted=envs.VLLM_BATCH_INVARIANT,
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

    ops.topk_hash_softplus_sqrt(
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
""",
    ),
]

RELOAD_AFTER_PATCH = "__TARGET_MODULE__"
