# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.model_executor.models.deepseek_v4_mtp.
"""

PATCHES = [
    # DeepSeek-V4-Flash-Base stores routed MTP expert scales as
    # `experts.N.wX.scale`. Upstream remaps those to `*_weight_scale`, which
    # exists for MegaMoE, but the active FP8 FusedMoE path owns
    # `*_weight_scale_inv`. Mirror the main DeepSeek-V4 loader's MUSA-safe
    # resolution and convert E8M0 bytes only after the destination parameter is
    # known.
    (
        """                    # Reinterpret E8M0 scales as uint8 to preserve raw
                    # exponent bytes; numeric copy_() would zero them.
                    # Mirrors the main DeepseekV4 loader.
                    if (
                        "weight_scale" in name
                        and loaded_weight.dtype == torch.float8_e8m0fnu
                    ):
                        loaded_weight = loaded_weight.view(torch.uint8)
                    for mapping in expert_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping
                        if weight_name not in name:
                            continue
                        name_mapped = name.replace(weight_name, param_name)
                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or not
""",
        """                    name_mapped = name
                    for mapping in expert_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping
                        if weight_name not in name:
                            continue
                        name_mapped = name.replace(weight_name, param_name)
                        if name_mapped not in params_dict and name_mapped.endswith(
                            "_weight_scale"
                        ):
                            inv_name_mapped = f"{name_mapped}_inv"
                            if inv_name_mapped in params_dict:
                                name_mapped = inv_name_mapped
                        if name_mapped not in params_dict:
                            continue
                        param = params_dict[name_mapped]
                        loaded_weight_for_param = loaded_weight
                        if loaded_weight_for_param.dtype == torch.float8_e8m0fnu:
                            if param.dtype == torch.uint8:
                                loaded_weight_for_param = loaded_weight_for_param.view(
                                    torch.uint8
                                )
                            else:
                                loaded_weight_for_param = loaded_weight_for_param.to(
                                    param.dtype
                                )
                        # We should ask the weight loader to return success or not
""",
    ),
    (
        """                        success = weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
""",
        """                        success = weight_loader(
                            param,
                            loaded_weight_for_param,
                            name_mapped,
""",
    ),
]
