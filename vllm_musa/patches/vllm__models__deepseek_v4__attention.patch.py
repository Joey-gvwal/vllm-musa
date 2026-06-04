# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Patch vLLM v0.22 DeepSeek-V4 attention FP8 einsum for MUSA."""

PATCHES = [
    (
        """logger = init_logger(__name__)
""",
        """logger = init_logger(__name__)


def _musa_deepseek_v4_linear_out_dtype(
    a: torch.Tensor,
    weight: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    if (
        current_platform.is_musa()
        or getattr(torch.version, "musa", None) is not None
        or a.device.type == "musa"
        or weight.device.type == "musa"
    ):
        return F.linear(a.to(out_dtype), weight.to(out_dtype))
    return torch.mm(a, weight.T, out_dtype=out_dtype)


def _musa_deepseek_v4_is_musa_tensor(tensor: torch.Tensor) -> bool:
    return (
        current_platform.is_musa()
        or getattr(torch.version, "musa", None) is not None
        or getattr(tensor.device, "type", None) == "musa"
    )


def _musa_deepseek_v4_qnorm_rope_kv_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    padded_heads: int,
    eps: float,
    block_size: int,
) -> torch.Tensor:
    from vllm_musa import _custom_ops as _musa_custom_ops

    active_slot_mapping = slot_mapping
    if slot_mapping.shape[0] > q.shape[0]:
        active_slot_mapping = slot_mapping[: q.shape[0]].contiguous()

    _musa_custom_ops.deepseek_v4_qnorm_rope_kv_insert(
        q,
        kv,
        kv_cache,
        active_slot_mapping,
        positions.contiguous(),
        cos_sin_cache,
        eps,
        block_size,
    )
    if q.shape[1] < padded_heads:
        return F.pad(q, (0, 0, 0, padded_heads - q.shape[1]), value=0.0)
    return q
""",
    ),
    (
        """def deepseek_v4_fp8_einsum(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
    recipe: list[int],
) -> None:
    fp8_einsum(equation, (a, a_scale), (b, b_scale), out, recipe=tuple(recipe))
""",
        """def deepseek_v4_fp8_einsum(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
    recipe: list[int],
) -> None:
    if (
        current_platform.is_musa()
        or getattr(torch.version, "musa", None) is not None
        or a.device.type == "musa"
        or out.device.type == "musa"
    ):
        try:
            from vllm_musa.deepseek_v4_jit.fp8_einsum import (
                try_musa_deepseek_v4_fp8_einsum,
            )

            handled, reason = try_musa_deepseek_v4_fp8_einsum(
                a, a_scale, b, b_scale, out, equation
            )
        except Exception as exc:
            handled = False
            reason = f"{type(exc).__name__}: {exc}"
        if handled:
            logger.warning_once(
                "Using MUSA DeepSeek-V4 FP8 einsum provider: %s.",
                reason,
            )
            return
        raise NotImplementedError(
            "MUSA DeepSeek-V4 FP8 einsum could not use a supported local "
            f"provider for {equation!r}; upstream DeepGEMM fp8_einsum is "
            f"not available in the MUSA runtime. Reason: {reason}"
        )
    fp8_einsum(equation, (a, a_scale), (b, b_scale), out, recipe=tuple(recipe))
""",
    ),
    (
        """                return torch.mm(
                    hidden_states,
                    compressor.fused_wkv_wgate.weight.T,
                    out_dtype=torch.float32,
                )
""",
        """                return _musa_deepseek_v4_linear_out_dtype(
                    hidden_states,
                    compressor.fused_wkv_wgate.weight,
                    torch.float32,
                )
""",
    ),
    (
        """                return torch.mm(
                    hidden_states,
                    indexer.compressor.fused_wkv_wgate.weight.T,
                    out_dtype=torch.float32,
                )
""",
        """                return _musa_deepseek_v4_linear_out_dtype(
                    hidden_states,
                    indexer.compressor.fused_wkv_wgate.weight,
                    torch.float32,
                )
""",
    ),
    (
        """        return torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            q,
            kv,
            swa_kv_cache_2d,
            swa_metadata.slot_mapping,
            positions.to(torch.int64),
            self.rotary_emb.cos_sin_cache,
            self.padded_heads,
            self.eps,
            swa_metadata.block_size,
        )
""",
        """        if _musa_deepseek_v4_is_musa_tensor(q):
            return _musa_deepseek_v4_qnorm_rope_kv_insert(
                q,
                kv,
                swa_kv_cache,
                swa_metadata.slot_mapping,
                positions.to(torch.int64),
                self.rotary_emb.cos_sin_cache,
                self.padded_heads,
                self.eps,
                swa_metadata.block_size,
            )

        return torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            q,
            kv,
            swa_kv_cache_2d,
            swa_metadata.slot_mapping,
            positions.to(torch.int64),
            self.rotary_emb.cos_sin_cache,
            self.padded_heads,
            self.eps,
            swa_metadata.block_size,
        )
""",
    ),
]
