import logging
import os

import torch

try:
    import vllm_musa._C  # noqa: F401
except ImportError as e:
    logging.error("Failed to import from vllm._C: %r", e)


def _is_musa_tensor(tensor: torch.Tensor) -> bool:
    return getattr(tensor, "device", None) is not None and tensor.device.type == "musa"


def _same_device(*tensors: torch.Tensor) -> bool:
    if not tensors:
        return True
    device = tensors[0].device
    return all(tensor.device == device for tensor in tensors)


def musa_fused_gemv_moe(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale,
    B_scale,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    mul_routed_weight: bool,
    topk: int,
    use_int4_w4a16: bool,
    use_swigelu: bool,
) -> None:
    return torch.ops._C_musa_ops.musa_fused_gemv_moe(
        A,
        B,
        C,
        A_scale,
        B_scale,
        topk_weights,
        topk_ids,
        mul_routed_weight,
        topk,
        use_int4_w4a16,
        use_swigelu,
    )


def musa_fused_gemv(
    x: torch.Tensor,
    qweight: torch.Tensor,
    x_scales: torch.Tensor = None,
    qweight_scales: torch.Tensor = None,
    use_swigelu: bool = False,
    use_rms_norm: bool = False,
    gamma: torch.Tensor = None,
    eps: float = 1e-6,
):
    use_int4_w4a16 = False
    out_shape = x.shape[:-1] + (
        qweight.shape[0] if not use_swigelu else qweight.shape[0] // 2,
    )
    assert not (
        use_swigelu and use_rms_norm
    ), "gemv only fused one activation (swigelu or rms_norm)!"

    if use_rms_norm:
        if gamma is None:
            assert False, "rms_norm gamm is None!"

    # fp8 grouped matmul
    if qweight.dtype == torch.float8_e4m3fn:
        # x: [m, k]
        # qweight: [n, k]
        # x_scales: [m, k / 128]
        # qweight: [n / 128, k / 128]
        # assert x_scales is not None, "FP8 grouped matmul x scales is None!"
        assert (
            qweight.dtype == torch.float8_e4m3fn
        ), "FP8 grouped matmul weight only support float8_e4m3fn!"
        assert qweight_scales is not None, "FP8 grouped matmul weight scales is None!"
        output = torch.empty(out_shape, device=x.device, dtype=torch.bfloat16)
        torch.ops._C_musa_ops.musa_fused_gemv(
            x,
            qweight,
            output,
            x_scales,
            qweight_scales,
            use_int4_w4a16,
            use_swigelu,
            use_rms_norm,
            gamma,
            eps,
        )
        return output
    # w4a16 gemv
    elif qweight_scales is not None:
        # qweight: [out, in/8]
        # scales: [out, in / group_size]
        assert (
            x.dtype == torch.bfloat16 or x.dtype == torch.float16
        ), "W4A16 gemv only support bfloat16 or float16!"
        use_int4_w4a16 = True
        out_shape = x.shape[:-1] + (
            qweight.shape[0] if not use_swigelu else qweight.shape[0] // 2,
        )
        output = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        torch.ops._C_musa_ops.musa_fused_gemv(
            x,
            qweight,
            output,
            None,
            qweight_scales,
            use_int4_w4a16,
            use_swigelu,
            use_rms_norm,
            gamma,
            eps,
        )
        return output
    # general gemv
    else:
        output = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        torch.ops._C_musa_ops.musa_fused_gemv(
            x,
            qweight,
            output,
            None,
            None,
            use_int4_w4a16,
            use_swigelu,
            use_rms_norm,
            gamma,
            eps,
        )
        return output


def mxfp4_dequant(
    x: torch.Tensor,
    scale: torch.Tensor,
    output: torch.Tensor,
) -> None:
    return torch.ops._C_musa_ops.mxfp4_dequant(x, scale, output)


def mxfp4_grouped_gemv(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    expert_ids: torch.Tensor,
    output: torch.Tensor,
    expert_map: torch.Tensor | None = None,
) -> None:
    return torch.ops._C_musa_ops.mxfp4_grouped_gemv(
        input,
        packed_weight,
        weight_scale,
        expert_ids,
        output,
        expert_map,
    )


def deepseek_v4_mega_moe_pre_dispatch(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    buf_x: torch.Tensor,
    buf_x_sf: torch.Tensor,
    buf_topk_idx: torch.Tensor,
    buf_topk_weights: torch.Tensor,
    quant_group_size: int = 32,
) -> None:
    native_supported = (
        _is_musa_tensor(x)
        and _is_musa_tensor(topk_idx)
        and _is_musa_tensor(topk_weights)
        and _is_musa_tensor(buf_x)
        and _is_musa_tensor(buf_x_sf)
        and _is_musa_tensor(buf_topk_idx)
        and _is_musa_tensor(buf_topk_weights)
        and _same_device(
            x,
            topk_idx,
            topk_weights,
            buf_x,
            buf_x_sf,
            buf_topk_idx,
            buf_topk_weights,
        )
        and x.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and topk_idx.dtype in (torch.int32, torch.int64)
        and topk_weights.dtype == torch.float32
        and buf_x.dtype == torch.float8_e4m3fn
        and buf_x_sf.dtype in (torch.int32, torch.uint8)
        and buf_topk_idx.dtype in (torch.int32, torch.int64)
        and buf_topk_weights.dtype == torch.float32
    )
    if (
        os.getenv("VLLM_MUSA_DEEPSEEK_V4_MEGA_MOE_PREDISPATCH_IMPL", "python")
        .strip()
        .lower()
        == "native"
        and native_supported
        and getattr(
            getattr(torch.ops, "_C_musa_ops", None),
            "deepseek_v4_mega_moe_pre_dispatch",
            None,
        )
        is not None
    ):
        return torch.ops._C_musa_ops.deepseek_v4_mega_moe_pre_dispatch(
            x,
            topk_idx,
            topk_weights,
            buf_x,
            buf_x_sf,
            buf_topk_idx,
            buf_topk_weights,
            quant_group_size,
        )

    from vllm_musa.deepseek_v4_moe_prereq import (
        deepseek_v4_mega_moe_pre_dispatch,
    )

    return deepseek_v4_mega_moe_pre_dispatch(
        x,
        topk_idx,
        topk_weights,
        buf_x,
        buf_x_sf,
        buf_topk_idx,
        buf_topk_weights,
        quant_group_size,
    )


def deepseek_v4_silu_and_mul_masked_post_quant(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    quant_group_size: int,
    masked_m: torch.Tensor,
    swiglu_limit: float | None = None,
) -> None:
    native_supported = (
        _is_musa_tensor(input)
        and _is_musa_tensor(output)
        and _is_musa_tensor(output_scale)
        and _is_musa_tensor(masked_m)
        and _same_device(input, output, output_scale, masked_m)
        and input.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and output.dtype == torch.float8_e4m3fn
        and output_scale.dtype in (torch.int32, torch.uint8)
        and masked_m.dtype == torch.int64
    )
    if (
        os.getenv("VLLM_MUSA_DEEPSEEK_V4_SWIGLU_POST_QUANT_IMPL", "python")
        .strip()
        .lower()
        == "native"
        and native_supported
        and getattr(
            getattr(torch.ops, "_C_musa_ops", None),
            "deepseek_v4_silu_and_mul_masked_post_quant",
            None,
        )
        is not None
    ):
        return torch.ops._C_musa_ops.deepseek_v4_silu_and_mul_masked_post_quant(
            input,
            output,
            output_scale,
            masked_m,
            quant_group_size,
            -1.0 if swiglu_limit is None else float(swiglu_limit),
        )

    from vllm_musa.deepseek_v4_moe_prereq import (
        deepseek_v4_silu_and_mul_masked_post_quant,
    )

    return deepseek_v4_silu_and_mul_masked_post_quant(
        input,
        output,
        output_scale,
        quant_group_size,
        masked_m,
        swiglu_limit,
    )


def fp8_ds_mla_sparse_gather(
    cache: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor | None,
    output: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    return torch.ops._C_musa_ops.fp8_ds_mla_sparse_gather(
        cache,
        indices,
        lengths,
        output,
        valid,
    )


def fp8_ds_mla_sparse_attention(
    q: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor | None,
    attn_sink: torch.Tensor | None,
    extra_cache: torch.Tensor | None,
    extra_indices: torch.Tensor | None,
    extra_lengths: torch.Tensor | None,
    output: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
) -> None:
    return torch.ops._C_musa_ops.fp8_ds_mla_sparse_attention(
        q,
        cache,
        indices,
        lengths,
        attn_sink,
        extra_cache,
        extra_indices,
        extra_lengths,
        output,
        lse,
        softmax_scale,
    )


def fp8_ds_mla_sparse_attention_grouped(
    q: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor | None,
    attn_sink: torch.Tensor | None,
    extra_cache: torch.Tensor | None,
    extra_indices: torch.Tensor | None,
    extra_lengths: torch.Tensor | None,
    output: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
) -> None:
    return torch.ops._C_musa_ops.fp8_ds_mla_sparse_attention_grouped(
        q,
        cache,
        indices,
        lengths,
        attn_sink,
        extra_cache,
        extra_indices,
        extra_lengths,
        output,
        lse,
        softmax_scale,
    )


def mxfp4_naive_grouped_moe(
    hidden: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    output: torch.Tensor,
    expert_map: torch.Tensor | None,
    apply_router_weight_on_input: bool,
) -> None:
    return torch.ops._C_musa_ops.mxfp4_naive_grouped_moe(
        hidden,
        w1,
        w2,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        output,
        expert_map,
        apply_router_weight_on_input,
    )


def fused_deepseek_v4_qnorm_rope_kv_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    block_size: int,
) -> None:
    return torch.ops._C_musa_ops.fused_deepseek_v4_qnorm_rope_kv_insert(
        q,
        kv,
        k_cache,
        slot_mapping,
        positions,
        cos_sin_cache,
        eps,
        block_size,
    )
