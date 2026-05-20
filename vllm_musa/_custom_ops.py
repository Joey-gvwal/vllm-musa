import logging

import torch

try:
    import vllm_musa._C  # noqa: F401
except ImportError as e:
    logging.error("Failed to import from vllm._C: %r", e)


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


def musa_fused_add_rms_norm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> None:
    return torch.ops._C_musa_ops.musa_fused_add_rms_norm(
        input,
        residual,
        weight,
        eps,
    )


def musa_reshape_and_cache_flash_nhd(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    return torch.ops._C_musa_ops.musa_reshape_and_cache_flash_nhd(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
    )


def deepseek_v4_store_sparse_kv(
    normed: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    write_mask: torch.Tensor,
) -> None:
    return torch.ops._C_musa_ops.deepseek_v4_store_sparse_kv(
        normed,
        kv_cache,
        slot_mapping,
        write_mask,
    )
