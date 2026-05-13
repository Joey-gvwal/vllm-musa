# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import os

import torch
import torch.nn.functional as F
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.config import _get_config_dtype_str
from vllm.model_executor.layers.fused_moe.fused_moe import (
    _get_config_quant_dtype,
    try_get_optimal_moe_config,
)
from vllm.model_executor.layers.fused_moe.utils import (
    disable_inplace,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import dequant_mxfp4
from vllm.model_executor.layers.quantization.utils.mxfp6_utils import dequant_mxfp6
from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import OCP_MX_Scheme
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl

from vllm_musa import _custom_ops as musa_ops


def _musa_torch_fused_moe_fallback(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    apply_router_weight_on_input: bool,
    expert_map: torch.Tensor | None,
    w1_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
) -> torch.Tensor:
    if activation != "silu":
        raise NotImplementedError(
            "MUSA torch fused-MoE fallback only supports silu/SwiGLU activation"
        )

    num_tokens, hidden_size = hidden_states.shape
    top_k = topk_ids.shape[1]
    out_size = w2.shape[1]
    flat_hidden = (
        hidden_states[:, None, :]
        .expand(num_tokens, top_k, hidden_size)
        .reshape(num_tokens * top_k, hidden_size)
    )
    flat_ids = topk_ids.reshape(-1)
    flat_weights = topk_weights.reshape(-1).to(torch.float32)
    flat_out = torch.zeros(
        (num_tokens * top_k, out_size), device=hidden_states.device, dtype=torch.float32
    )

    for global_expert_id in torch.unique(flat_ids).to(torch.long).tolist():
        if global_expert_id < 0:
            continue
        local_expert_id = global_expert_id
        if expert_map is not None:
            if global_expert_id >= expert_map.numel():
                continue
            local_expert_id = int(expert_map[global_expert_id].item())
            if local_expert_id < 0:
                continue
        if local_expert_id >= w1.shape[0]:
            continue

        mask = flat_ids == global_expert_id
        x = flat_hidden[mask].to(torch.float32)
        router_weight = flat_weights[mask].unsqueeze(-1)
        if apply_router_weight_on_input:
            x = x * router_weight

        gate_up = x.matmul(w1[local_expert_id].to(torch.float32).transpose(0, 1))
        if w1_bias is not None:
            gate_up = gate_up + w1_bias[local_expert_id].to(torch.float32)
        gate, up = gate_up.chunk(2, dim=-1)
        intermediate = F.silu(gate) * up

        expert_out = intermediate.matmul(
            w2[local_expert_id].to(torch.float32).transpose(0, 1)
        )
        if w2_bias is not None:
            expert_out = expert_out + w2_bias[local_expert_id].to(torch.float32)
        if not apply_router_weight_on_input:
            expert_out = expert_out * router_weight
        flat_out[mask] = expert_out

    return (
        flat_out.view(num_tokens, top_k, out_size)
        .sum(dim=1)
        .to(hidden_states.dtype)
    )


def _dequant_mxfp4_musa(
    x: torch.Tensor, scale: torch.Tensor | None, float_dtype: torch.dtype
) -> torch.Tensor:
    if not current_platform.is_musa():
        return dequant_mxfp4(x, scale, float_dtype)
    if scale is None:
        raise ValueError("MXFP4 dequantization requires block scales")

    values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=x.device,
    )
    unpacked = torch.empty(
        (*x.shape[:-1], x.shape[-1] * 2), dtype=torch.uint8, device=x.device
    )
    unpacked[..., 0::2] = x & 0x0F
    unpacked[..., 1::2] = (x >> 4) & 0x0F

    sign = torch.where((unpacked & 0x08) != 0, -1.0, 1.0)
    magnitude = values[(unpacked & 0x07).long()]
    out = sign * magnitude

    block_size = 32
    out = out.reshape(*out.shape[:-1], -1, block_size)
    scale_factor = _musa_mxfp4_scale_to_float(scale).unsqueeze(-1)
    out = out * scale_factor
    out = out.reshape(*out.shape[:-2], -1)
    return out.to(float_dtype)


def _musa_mxfp4_scale_to_float(scale: torch.Tensor) -> torch.Tensor:
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is not None and scale.dtype == e8m0_dtype:
        scale_bytes = scale.view(torch.uint8)
        return (scale_bytes.to(torch.int32) << 23).view(torch.float32)
    if scale.dtype == torch.uint8:
        return (scale.to(torch.int32) << 23).view(torch.float32)
    return scale.to(torch.float32)


def _supports_quant_scheme(
    weight_key,
    activation_key,
) -> bool:
    p = current_platform
    if p.is_rocm():
        from vllm.platforms.rocm import on_gfx9

        is_rocm_on_gfx9 = on_gfx9()
    else:
        is_rocm_on_gfx9 = False
    # ==================== MUSA ADAPTATION ====================
    device_supports_fp8 = is_rocm_on_gfx9 or (
        p.is_musa() and p.has_device_capability((3, 1))
    )
    # ========================== END ==========================
    if not device_supports_fp8:
        return (weight_key, activation_key) == (None, None)

    SUPPORTED_W_A = [
        (None, None),
        (kFp8Static128BlockSym, kFp8Dynamic128Sym),
        (kFp8StaticChannelSym, kFp8DynamicTokenSym),
        (kFp8StaticTensorSym, kFp8DynamicTokenSym),
        (kFp8StaticTensorSym, kFp8StaticTensorSym),
    ]
    return (weight_key, activation_key) in SUPPORTED_W_A


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    ocp_mx_scheme: str | None = None,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    w1_zp: torch.Tensor | None = None,
    w2_zp: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    block_shape: list[int] | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    # Check constraints.
    if use_int4_w4a16:
        assert hidden_states.size(1) // 2 == w1.size(2), "Hidden size mismatch"
    elif ocp_mx_scheme is not None:
        if ocp_mx_scheme in {
            "w_mxfp4",
            "w_mxfp4_a_mxfp4",
            "w_mxfp4_a_mxfp6_e3m2",
            "w_mxfp4_a_mxfp6_e2m3",
        }:
            # 16bit activation and fp4x2 packed weight
            assert hidden_states.size(1) == w1.size(2) * 2, "hidden size mismatch"
        elif ocp_mx_scheme in {
            "w_mxfp6_e3m2_a_mxfp6_e3m2",
            "w_mxfp6_e2m3_a_mxfp6_e2m3",
        }:
            assert (
                hidden_states.size(1) == (w1.size(2) * 4) // 3
            ), "hidden size mismatch"
        else:
            raise NotImplementedError(f"Unsupported ocp_mx_scheme={ocp_mx_scheme}")
    else:
        assert hidden_states.size(1) == w1.size(
            2
        ), f"Hidden size mismatch {hidden_states.size(1)} != {w1.size(2)}"

    assert topk_weights.size() == topk_ids.size(), "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    num_tokens = hidden_states.size(0)
    E, N, _ = w1.size()
    K = w2.size(1)
    if global_num_experts == -1:
        global_num_experts = E
    top_k_num = topk_ids.size(1)

    M = num_tokens

    config_dtype = _get_config_dtype_str(
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        ocp_mx_scheme=ocp_mx_scheme,
        dtype=hidden_states.dtype,
    )

    # Note: for use_int8_w8a16 or use_int4_w4a16, the activations are
    # quantized prior to calling fused_experts.
    quant_dtype = _get_config_quant_dtype(
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        ocp_mx_scheme=ocp_mx_scheme,
    )

    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.size(),
        w2.size(),
        top_k_num,
        config_dtype,
        block_shape=block_shape,
    )

    config = get_config_func(M)

    # We can reuse the memory between these because by the time we need
    # cache3, we're done with cache1
    cache13 = torch.empty(
        M * top_k_num * max(N, K),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache1 = cache13[: M * top_k_num * N].view(M, top_k_num, N)
    intermediate_cache3 = cache13[: M * top_k_num * K].view(M, top_k_num, K)

    # This needs separate memory since it's used concurrently with cache1
    intermediate_cache2 = torch.empty(
        (M * top_k_num, N // 2), device=hidden_states.device, dtype=hidden_states.dtype
    )

    if hidden_states.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif hidden_states.dtype == torch.float16:
        compute_type = tl.float16
    elif hidden_states.dtype == torch.float32:
        compute_type = tl.float32
    else:
        raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

    if inplace and not disable_inplace():
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)

    if ocp_mx_scheme is not None:
        # TODO: On platforms for which `current_platform.supports_mx()` is True
        # and for which we have a native OCP mx fused MOE kernel,
        # this dequantization step should not be done.
        if ocp_mx_scheme in {
            OCP_MX_Scheme.w_mxfp4,
            OCP_MX_Scheme.w_mxfp4_a_mxfp4,
            OCP_MX_Scheme.w_mxfp4_a_mxfp6_e3m2,
            OCP_MX_Scheme.w_mxfp4_a_mxfp6_e2m3,
        }:
            # Weight has to be dequantized for mxfp4 emulation.
            w1 = _dequant_mxfp4_musa(w1, w1_scale, hidden_states.dtype)
            w1_scale = None
            w2 = _dequant_mxfp4_musa(w2, w2_scale, hidden_states.dtype)
            w2_scale = None
        elif ocp_mx_scheme == OCP_MX_Scheme.w_mxfp6_e3m2_a_mxfp6_e3m2:
            w1 = dequant_mxfp6(
                w1, w1_scale, quant_dtype="fp6_e3m2", float_dtype=hidden_states.dtype
            )
            w1_scale = None
            w2 = dequant_mxfp6(
                w2, w2_scale, quant_dtype="fp6_e3m2", float_dtype=hidden_states.dtype
            )
            w2_scale = None
        elif ocp_mx_scheme == OCP_MX_Scheme.w_mxfp6_e2m3_a_mxfp6_e2m3:
            w1 = dequant_mxfp6(
                w1, w1_scale, quant_dtype="fp6_e2m3", float_dtype=hidden_states.dtype
            )
            w1_scale = None
            w2 = dequant_mxfp6(
                w2, w2_scale, quant_dtype="fp6_e2m3", float_dtype=hidden_states.dtype
            )
            w2_scale = None
        else:
            raise NotImplementedError(f"Unsupported ocp_mx_scheme={ocp_mx_scheme}")

    musa_gemv = getattr(
        getattr(torch.ops, "_C_musa_ops", None), "musa_fused_gemv_moe", None
    )
    if (
        current_platform.is_musa()
        and os.getenv("VLLM_MUSA_ENABLE_MXFP4_MOE_FALLBACK", "0") == "1"
        and musa_gemv is None
    ):
        return _musa_torch_fused_moe_fallback(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            expert_map=expert_map,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
        )

    # ==================== MUSA ADAPTATION ====================
    # Due to the implementation of 0.20.0 relying on per_token_group_quant,
    # which is currently not supported by Musa, please refer to setup.py for details.
    # The version used here is 0.18.0
    CHUNK_SIZE = 16384
    M = min(num_tokens, CHUNK_SIZE)
    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (
            chunk * CHUNK_SIZE,
            min((chunk + 1) * CHUNK_SIZE, num_tokens),
        )
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.size()

        if tokens_in_chunk == 0:
            break

        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            # Adjust the intermediate cache size and config for the last
            # chunk. Note that in most cases we only have one chunk
            # so the cache size and config are already set correctly and
            # do not need to be adjusted.
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
            intermediate_cache2 = intermediate_cache2[
                : tokens_in_chunk * topk_ids.size(1)
            ]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]
            config = get_config_func(tokens_in_chunk)

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

        musa_ops.musa_fused_gemv_moe(
            curr_hidden_states,
            w1,
            intermediate_cache2,
            None,
            w1_scale,
            curr_topk_weights,
            curr_topk_ids,
            apply_router_weight_on_input,
            topk_ids.shape[1],
            use_int4_w4a16,
            use_swigelu=True,
        )
        musa_ops.musa_fused_gemv_moe(
            intermediate_cache2,
            w2,
            intermediate_cache3,
            None,
            w2_scale,
            curr_topk_weights,
            curr_topk_ids,
            not apply_router_weight_on_input,
            1,
            use_int4_w4a16,
            use_swigelu=False,
        )
        # ========================== END ====================
        ops.moe_sum(
            intermediate_cache3.view(*intermediate_cache3.size()),
            out_hidden_states,
        )

    return out_hidden_states


import vllm.model_executor.layers.fused_moe.fused_moe

vllm.model_executor.layers.fused_moe.fused_moe.fused_experts_impl = fused_experts_impl
vllm.model_executor.layers.fused_moe.fused_moe.TritonExperts._supports_quant_scheme = (
    _supports_quant_scheme
)
