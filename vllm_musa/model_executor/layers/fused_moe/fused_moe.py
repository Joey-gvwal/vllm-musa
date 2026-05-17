# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch
import torch.nn.functional as F
from vllm import _custom_ops as ops
from vllm.logger import init_logger
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
    TopKWeightAndReduceNoOP,
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

from vllm_musa import _custom_ops as musa_ops

logger = init_logger(__name__)

_MXFP4_SCHEMES = {
    OCP_MX_Scheme.w_mxfp4,
    OCP_MX_Scheme.w_mxfp4_a_mxfp4,
    OCP_MX_Scheme.w_mxfp4_a_mxfp6_e3m2,
    OCP_MX_Scheme.w_mxfp4_a_mxfp6_e2m3,
}
_MXFP4_SIGNED_LUT_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)
_MXFP4_SIGNED_LUT_CACHE: dict[tuple[str, int | None], torch.Tensor] = {}


def _musa_mxfp4_dequant_impl() -> str:
    return os.getenv("VLLM_MUSA_MXFP4_DEQUANT_IMPL", "native").strip().lower()


def _musa_mxfp4_grouped_gemv_impl() -> str:
    return os.getenv("VLLM_MUSA_MXFP4_GROUPED_GEMV_IMPL", "off").strip().lower()


def _musa_mxfp4_naive_grouped_moe_impl() -> str:
    return (
        os.getenv("VLLM_MUSA_MXFP4_NAIVE_GROUPED_MOE_IMPL", "off")
        .strip()
        .lower()
    )


def _musa_mxfp4_fallback_impl() -> str:
    return os.getenv("VLLM_MUSA_MXFP4_FALLBACK_IMPL", "chunked_bmm").strip().lower()


def _musa_mxfp4_chunked_bmm_experts() -> int:
    try:
        return max(1, int(os.getenv("VLLM_MUSA_MXFP4_CHUNKED_BMM_EXPERTS", "16")))
    except ValueError:
        return 16


def _musa_timed(scope_name: str):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            try:
                from vllm_musa.deepseek_v4_timers import timed
            except Exception:
                return fn(*args, **kwargs)
            with timed(scope_name):
                return fn(*args, **kwargs)

        return wrapper

    return decorator


def _is_mxfp4_scheme(ocp_mx_scheme: str | None) -> bool:
    return ocp_mx_scheme in _MXFP4_SCHEMES


def _musa_mxfp4_signed_lut(device: torch.device) -> torch.Tensor:
    key = (device.type, device.index)
    lut = _MXFP4_SIGNED_LUT_CACHE.get(key)
    if lut is None:
        lut = torch.tensor(
            _MXFP4_SIGNED_LUT_VALUES,
            dtype=torch.float32,
            device=device,
        )
        _MXFP4_SIGNED_LUT_CACHE[key] = lut
    return lut


@_musa_timed("mxfp4_moe.torch_fused_moe_fallback")
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
    ocp_mx_scheme: str | None = None,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    swiglu_limit: float | None = None,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
) -> torch.Tensor:
    activation_value = str(getattr(activation, "value", activation)).lower()
    activation_value = activation_value.rsplit(".", 1)[-1]
    if activation_value not in ("silu", "swigluoai"):
        raise NotImplementedError(
            "MUSA torch fused-MoE fallback only supports silu/SWIGLUOAI "
            f"activation, got {activation!r}"
        )
    if (
        current_platform.is_musa()
        and _is_mxfp4_scheme(ocp_mx_scheme)
        and _musa_mxfp4_fallback_impl() == "chunked_bmm"
    ):
        if w1_scale is None or w2_scale is None:
            raise ValueError("MXFP4 chunked-bmm fallback requires block scales")
        return _musa_mxfp4_chunked_bmm_fallback(
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
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
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

    unique_expert_ids = sorted({int(expert_id) for expert_id in flat_ids.cpu().tolist()})
    for global_expert_id in unique_expert_ids:
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

        if _is_mxfp4_scheme(ocp_mx_scheme):
            w1_local = _dequant_mxfp4_musa(
                w1[local_expert_id], w1_scale[local_expert_id], torch.float32
            )
        else:
            w1_local = w1[local_expert_id].to(torch.float32)
        gate_up = x.matmul(w1_local.transpose(0, 1))
        del w1_local

        if w1_bias is not None:
            gate_up = gate_up + w1_bias[local_expert_id].to(torch.float32)
        if activation_value == "swigluoai":
            gate, up = gate_up[..., ::2], gate_up[..., 1::2]
            limit = 7.0 if swiglu_limit is None else swiglu_limit
            if limit > 0:
                gate = torch.clamp(gate, max=limit)
                up = torch.clamp(up, min=-limit, max=limit)
            alpha = 1.702 if swiglu_alpha is None else swiglu_alpha
            beta = 1.0 if swiglu_beta is None else swiglu_beta
            intermediate = (up + beta) * (gate * torch.sigmoid(gate * alpha))
        else:
            gate, up = gate_up.chunk(2, dim=-1)
            if swiglu_limit is not None and swiglu_limit > 0:
                gate = torch.clamp(gate, max=swiglu_limit)
                up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
            intermediate = F.silu(gate) * up

        if _is_mxfp4_scheme(ocp_mx_scheme):
            w2_local = _dequant_mxfp4_musa(
                w2[local_expert_id], w2_scale[local_expert_id], torch.float32
            )
        else:
            w2_local = w2[local_expert_id].to(torch.float32)
        expert_out = intermediate.matmul(w2_local.transpose(0, 1))
        del w2_local

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


@_musa_timed("mxfp4_moe.dequant_mxfp4")
def _dequant_mxfp4_musa(
    x: torch.Tensor, scale: torch.Tensor | None, float_dtype: torch.dtype
) -> torch.Tensor:
    if not current_platform.is_musa():
        return dequant_mxfp4(x, scale, float_dtype)
    if scale is None:
        raise ValueError("MXFP4 dequantization requires block scales")

    native_impl = _musa_mxfp4_dequant_impl() == "native"
    if (
        native_impl
        and x.dtype == torch.uint8
        and x.is_contiguous()
        and scale.device == x.device
    ):
        native_dequant = getattr(
            getattr(torch.ops, "_C_musa_ops", None), "mxfp4_dequant", None
        )
        if native_dequant is not None:
            try:
                scale_bytes = (
                    scale if scale.dtype == torch.uint8 else scale.view(torch.uint8)
                )
            except RuntimeError:
                scale_bytes = None
            if (
                scale_bytes is not None
                and scale_bytes.is_contiguous()
                and scale_bytes.numel() * 32 == x.numel() * 2
            ):
                out = torch.empty(
                    (*x.shape[:-1], x.shape[-1] * 2),
                    dtype=float_dtype,
                    device=x.device,
                )
                native_dequant(x, scale_bytes, out)
                return out

    signed_values = _musa_mxfp4_signed_lut(x.device)
    unpacked = torch.empty(
        (*x.shape[:-1], x.shape[-1] * 2), dtype=torch.uint8, device=x.device
    )
    unpacked[..., 0::2] = x & 0x0F
    unpacked[..., 1::2] = (x >> 4) & 0x0F

    out = signed_values[unpacked.long()]

    block_size = 32
    out = out.reshape(*out.shape[:-1], -1, block_size)
    scale_factor = _musa_mxfp4_scale_to_float(scale).unsqueeze(-1)
    out.mul_(scale_factor)
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


def _musa_mxfp4_scale_bytes(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype == torch.uint8:
        if not scale.is_contiguous():
            raise RuntimeError("MXFP4 scale byte tensor must be contiguous")
        return scale
    scale_bytes = scale.view(torch.uint8)
    if not scale_bytes.is_contiguous():
        raise RuntimeError("MXFP4 scale byte view must be contiguous")
    return scale_bytes


def _musa_mxfp4_scale_bytes_ready(scale: torch.Tensor) -> bool:
    try:
        _musa_mxfp4_scale_bytes(scale)
    except RuntimeError:
        return False
    return True


def _musa_moe_activation(
    gate_up: torch.Tensor,
    activation: str | MoEActivation,
    swiglu_limit: float | None = None,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
) -> torch.Tensor:
    activation_value = str(getattr(activation, "value", activation)).lower()
    activation_value = activation_value.rsplit(".", 1)[-1]
    if activation_value == "swigluoai":
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        limit = 7.0 if swiglu_limit is None else swiglu_limit
        if limit > 0:
            gate = torch.clamp(gate, max=limit)
            up = torch.clamp(up, min=-limit, max=limit)
        alpha = 1.702 if swiglu_alpha is None else swiglu_alpha
        beta = 1.0 if swiglu_beta is None else swiglu_beta
        return (up + beta) * (gate * torch.sigmoid(gate * alpha))
    if activation_value != "silu":
        raise NotImplementedError(
            "MUSA MXFP4 grouped GEMV path only supports silu/SWIGLUOAI "
            f"activation, got {activation!r}"
        )
    gate, up = gate_up.chunk(2, dim=-1)
    if swiglu_limit is not None and swiglu_limit > 0:
        gate = torch.clamp(gate, max=swiglu_limit)
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
    return F.silu(gate) * up


@_musa_timed("mxfp4_moe.chunked_bmm_fallback")
def _musa_mxfp4_chunked_bmm_fallback(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str | MoEActivation,
    apply_router_weight_on_input: bool,
    expert_map: torch.Tensor | None,
    w1_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    swiglu_limit: float | None = None,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
) -> torch.Tensor:
    num_tokens, hidden_size = hidden_states.shape
    top_k = topk_ids.shape[1]
    out_size = w2.shape[1]
    flat_rows = num_tokens * top_k
    if flat_rows == 0:
        return torch.empty(
            (num_tokens, out_size),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

    flat_hidden = (
        hidden_states[:, None, :]
        .expand(num_tokens, top_k, hidden_size)
        .reshape(flat_rows, hidden_size)
    )
    flat_ids = topk_ids.reshape(-1)
    flat_weights = topk_weights.reshape(-1, 1).to(torch.float32)
    flat_out = torch.zeros(
        (flat_rows, out_size),
        device=hidden_states.device,
        dtype=torch.float32,
    )

    selected_global: list[int] = []
    selected_local: list[int] = []
    for global_expert_id in sorted(
        {int(expert_id) for expert_id in flat_ids.cpu().tolist()}
    ):
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
        selected_global.append(global_expert_id)
        selected_local.append(local_expert_id)

    chunk_experts = _musa_mxfp4_chunked_bmm_experts()
    for start in range(0, len(selected_global), chunk_experts):
        global_chunk = selected_global[start : start + chunk_experts]
        local_chunk = selected_local[start : start + chunk_experts]
        masks = [flat_ids == global_expert_id for global_expert_id in global_chunk]
        row_indices = [
            torch.nonzero(mask, as_tuple=False).flatten() for mask in masks
        ]
        max_rows = max(
            (int(indices.numel()) for indices in row_indices),
            default=0,
        )
        if max_rows == 0:
            continue

        local_ids = torch.tensor(
            local_chunk,
            device=hidden_states.device,
            dtype=torch.long,
        )
        x_pack = torch.zeros(
            (len(local_chunk), max_rows, hidden_size),
            device=hidden_states.device,
            dtype=torch.float32,
        )
        for chunk_idx, indices in enumerate(row_indices):
            row_count = indices.numel()
            x_pack[chunk_idx, :row_count] = flat_hidden[indices].to(torch.float32)
            if apply_router_weight_on_input:
                x_pack[chunk_idx, :row_count] *= flat_weights[indices]

        w1_local = _dequant_mxfp4_musa(
            w1[local_ids].contiguous(),
            w1_scale[local_ids].contiguous(),
            torch.float32,
        )
        gate_up = torch.bmm(x_pack, w1_local.transpose(1, 2))
        del w1_local
        if w1_bias is not None:
            gate_up += w1_bias[local_ids].to(torch.float32).unsqueeze(1)

        intermediate = _musa_moe_activation(
            gate_up.reshape(len(local_chunk) * max_rows, -1),
            activation,
            swiglu_limit,
            swiglu_alpha,
            swiglu_beta,
        ).reshape(len(local_chunk), max_rows, -1)

        w2_local = _dequant_mxfp4_musa(
            w2[local_ids].contiguous(),
            w2_scale[local_ids].contiguous(),
            torch.float32,
        )
        out_pack = torch.bmm(intermediate, w2_local.transpose(1, 2))
        del w2_local
        if w2_bias is not None:
            out_pack += w2_bias[local_ids].to(torch.float32).unsqueeze(1)

        for chunk_idx, indices in enumerate(row_indices):
            row_count = indices.numel()
            expert_out = out_pack[chunk_idx, :row_count]
            if not apply_router_weight_on_input:
                expert_out = expert_out * flat_weights[indices]
            flat_out[indices] = expert_out

    return flat_out.view(num_tokens, top_k, out_size).sum(dim=1).to(
        hidden_states.dtype
    )


@_musa_timed("mxfp4_moe.grouped_gemv")
def _musa_mxfp4_grouped_gemv_moe(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str | MoEActivation,
    expert_map: torch.Tensor | None,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    apply_router_weight_on_input: bool,
) -> None:
    num_tokens, hidden_size = hidden_states.shape
    top_k = topk_ids.shape[1]
    n = w1.shape[1]
    k = w2.shape[1]
    flat_rows = num_tokens * top_k
    if flat_rows == 0:
        return

    flat_hidden = (
        hidden_states[:, None, :]
        .expand(num_tokens, top_k, hidden_size)
        .reshape(flat_rows, hidden_size)
        .contiguous()
    )
    flat_ids = topk_ids.reshape(-1).contiguous()
    flat_weights = topk_weights.reshape(-1, 1).to(hidden_states.dtype)
    if apply_router_weight_on_input:
        flat_hidden = flat_hidden * flat_weights

    gate_up = torch.empty(
        (flat_rows, n), device=hidden_states.device, dtype=hidden_states.dtype
    )
    musa_ops.mxfp4_grouped_gemv(
        flat_hidden,
        w1,
        _musa_mxfp4_scale_bytes(w1_scale),
        flat_ids,
        gate_up,
        expert_map,
    )
    intermediate = _musa_moe_activation(gate_up, activation).contiguous()

    flat_out = torch.empty(
        (flat_rows, k), device=hidden_states.device, dtype=hidden_states.dtype
    )
    musa_ops.mxfp4_grouped_gemv(
        intermediate,
        w2,
        _musa_mxfp4_scale_bytes(w2_scale),
        flat_ids,
        flat_out,
        expert_map,
    )
    if not apply_router_weight_on_input:
        flat_out = flat_out * flat_weights
    ops.moe_sum(flat_out.view(num_tokens, top_k, k), output)


@_musa_timed("mxfp4_moe.naive_grouped_moe")
def _musa_mxfp4_naive_grouped_moe(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    apply_router_weight_on_input: bool,
) -> None:
    musa_ops.mxfp4_naive_grouped_moe(
        hidden_states.contiguous(),
        w1,
        w2,
        _musa_mxfp4_scale_bytes(w1_scale),
        _musa_mxfp4_scale_bytes(w2_scale),
        topk_weights.contiguous(),
        topk_ids.contiguous(),
        output,
        expert_map,
        apply_router_weight_on_input,
    )


def _musa_mxfp4_make_w4a16_quant_config(
    quant_config: FusedMoEQuantConfig,
) -> FusedMoEQuantConfig:
    return FusedMoEQuantConfig.make(
        quant_dtype=None,
        weight_dtype="mxfp4",
        w1_scale=quant_config.w1_scale,
        w2_scale=quant_config.w2_scale,
        w1_bias=quant_config.w1_bias,
        w2_bias=quant_config.w2_bias,
        gemm1_alpha=quant_config.gemm1_alpha,
        gemm1_beta=quant_config.gemm1_beta,
        gemm1_clamp_limit=quant_config.gemm1_clamp_limit,
    )


class _MusaMxfp4ExpertsBase(mk.FusedMoEExpertsModular):

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_musa()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(weight_key, activation_key) -> bool:
        return weight_key == "mxfp4" and activation_key is None

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (MoEActivation.SILU, MoEActivation.SWIGLUOAI)

    @staticmethod
    def _supports_parallel_config(moe_parallel_config) -> bool:
        return True

    @staticmethod
    def _is_current_stream_capturing() -> bool:
        cuda_module = getattr(torch, "cuda", None)
        if cuda_module is None:
            return False
        is_capturing = getattr(cuda_module, "is_current_stream_capturing", None)
        if is_capturing is None:
            return False
        try:
            return bool(is_capturing())
        except Exception:
            return False

    def _apply_expert(
        self,
        x: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        activation: MoEActivation,
        expert_id: int,
    ) -> torch.Tensor:
        x = x.to(torch.float32)
        assert self.quant_config.w1_scale is not None
        assert self.quant_config.w2_scale is not None
        w1_local = _dequant_mxfp4_musa(
            w1[expert_id], self.quant_config.w1_scale[expert_id], torch.float32
        )
        gate_up = x.matmul(w1_local.transpose(0, 1))
        del w1_local

        if self.quant_config.w1_bias is not None:
            gate_up = gate_up + self.quant_config.w1_bias[expert_id].to(torch.float32)
        if activation == MoEActivation.SWIGLUOAI:
            gate, up = gate_up[..., ::2], gate_up[..., 1::2]
            limit = self.quant_config.gemm1_clamp_limit
            if limit is None:
                limit = 7.0
            if limit > 0:
                gate = torch.clamp(gate, max=limit)
                up = torch.clamp(up, min=-limit, max=limit)
            alpha = self.quant_config.gemm1_alpha
            beta = self.quant_config.gemm1_beta
            alpha = 1.702 if alpha is None else alpha
            beta = 1.0 if beta is None else beta
            intermediate = (up + beta) * (gate * torch.sigmoid(gate * alpha))
        else:
            gate, up = gate_up.chunk(2, dim=-1)
            limit = self.quant_config.gemm1_clamp_limit
            if limit is not None and limit > 0:
                gate = torch.clamp(gate, max=limit)
                up = torch.clamp(up, min=-limit, max=limit)
            intermediate = F.silu(gate) * up

        w2_local = _dequant_mxfp4_musa(
            w2[expert_id], self.quant_config.w2_scale[expert_id], torch.float32
        )
        out = intermediate.matmul(w2_local.transpose(0, 1))
        del w2_local

        if self.quant_config.w2_bias is not None:
            out = out + self.quant_config.w2_bias[expert_id].to(torch.float32)
        return out


class MusaMxfp4BatchedExperts(_MusaMxfp4ExpertsBase):

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int,
        num_dispatchers: int,
    ):
        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
        )

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceDelegate()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        assert self.max_num_tokens is not None
        assert self.num_dispatchers is not None
        max_tokens = self.max_num_tokens * self.num_dispatchers
        workspace13 = (local_num_experts, max_tokens, max(N, K))
        workspace2 = (local_num_experts, max_tokens, max(N // 2, K))
        output = (local_num_experts, max_tokens, K)
        return workspace13, workspace2, output

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        del topk_weights, topk_ids, global_num_experts, expert_map
        del a1q_scale, a2_scale, workspace13, workspace2
        del apply_router_weight_on_input

        assert hidden_states.dim() == 3
        assert expert_tokens_meta is not None
        output.zero_()
        expert_num_tokens = expert_tokens_meta.expert_num_tokens
        for expert_id in range(w1.shape[0]):
            if torch.compiler.is_compiling() or self._is_current_stream_capturing():
                num_tokens = hidden_states.shape[1]
            else:
                num_tokens = int(expert_num_tokens[expert_id].item())
            if num_tokens == 0:
                continue
            expert_out = self._apply_expert(
                hidden_states[expert_id, :num_tokens],
                w1,
                w2,
                activation,
                expert_id,
            )
            output[expert_id, :num_tokens] = expert_out.to(output.dtype)


class MusaMxfp4StandardExperts(_MusaMxfp4ExpertsBase):

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return (M, topk, max(N, K)), (M, topk, max(N // 2, K)), (M, K)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        del global_num_experts, a1q_scale, a2_scale, workspace13, workspace2
        del expert_tokens_meta

        expert_out = _musa_torch_fused_moe_fallback(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            expert_map=expert_map,
            w1_bias=self.quant_config.w1_bias,
            w2_bias=self.quant_config.w2_bias,
            ocp_mx_scheme=self.quant_config.ocp_mx_scheme,
            w1_scale=self.quant_config.w1_scale,
            w2_scale=self.quant_config.w2_scale,
            swiglu_limit=self.quant_config.gemm1_clamp_limit,
            swiglu_alpha=self.quant_config.gemm1_alpha,
            swiglu_beta=self.quant_config.gemm1_beta,
        )
        output.copy_(expert_out, non_blocking=True)


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

    intermediate_cache3 = torch.empty(
        (M, top_k_num, K), device=hidden_states.device, dtype=hidden_states.dtype
    )

    # The first GEMV writes activation input to cache2; the second GEMV writes
    # top-k outputs to cache3 for moe_sum.
    intermediate_cache2 = torch.empty(
        (M * top_k_num, N // 2), device=hidden_states.device, dtype=hidden_states.dtype
    )

    if inplace and not disable_inplace():
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)

    musa_gemv = getattr(
        getattr(torch.ops, "_C_musa_ops", None), "musa_fused_gemv_moe", None
    )
    use_musa_torch_moe_fallback = (
        current_platform.is_musa()
        and os.getenv("VLLM_MUSA_ENABLE_MXFP4_MOE_FALLBACK", "0") == "1"
        and musa_gemv is None
    )
    use_musa_mxfp4_grouped_gemv = (
        current_platform.is_musa()
        and _is_mxfp4_scheme(ocp_mx_scheme)
        and _musa_mxfp4_grouped_gemv_impl() == "native"
        and w1_scale is not None
        and w2_scale is not None
        and w1_scale.device == hidden_states.device
        and w2_scale.device == hidden_states.device
        and _musa_mxfp4_scale_bytes_ready(w1_scale)
        and _musa_mxfp4_scale_bytes_ready(w2_scale)
        and w1_bias is None
        and w2_bias is None
        and getattr(
            getattr(torch.ops, "_C_musa_ops", None), "mxfp4_grouped_gemv", None
        )
        is not None
    )
    activation_value = str(getattr(activation, "value", activation)).lower()
    activation_value = activation_value.rsplit(".", 1)[-1]
    use_musa_mxfp4_naive_grouped_moe = (
        current_platform.is_musa()
        and _is_mxfp4_scheme(ocp_mx_scheme)
        and _musa_mxfp4_naive_grouped_moe_impl() == "native"
        and activation_value == "silu"
        and w1_scale is not None
        and w2_scale is not None
        and w1_scale.device == hidden_states.device
        and w2_scale.device == hidden_states.device
        and _musa_mxfp4_scale_bytes_ready(w1_scale)
        and _musa_mxfp4_scale_bytes_ready(w2_scale)
        and w1_bias is None
        and w2_bias is None
        and swiglu_limit is None
        and swiglu_alpha is None
        and swiglu_beta is None
        and getattr(
            getattr(torch.ops, "_C_musa_ops", None),
            "mxfp4_naive_grouped_moe",
            None,
        )
        is not None
    )

    if ocp_mx_scheme is not None:
        # TODO: On platforms for which `current_platform.supports_mx()` is True
        # and for which we have a native OCP mx fused MOE kernel,
        # this dequantization step should not be done.
        if _is_mxfp4_scheme(ocp_mx_scheme):
            if (
                not use_musa_torch_moe_fallback
                and not use_musa_mxfp4_grouped_gemv
                and not use_musa_mxfp4_naive_grouped_moe
            ):
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

    if use_musa_torch_moe_fallback:
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
            ocp_mx_scheme=ocp_mx_scheme,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            swiglu_limit=None,
        )

    # ==================== MUSA ADAPTATION ====================
    # Due to the implementation of 0.20.0 relying on per_token_group_quant,
    # which is currently not supported by Musa, please refer to setup.py for details.
    # The version used here is 0.18.0
    logger.info_once(
        "MUSA fused MoE uses native GEMV block selection; skipping upstream "
        "Triton MoE JSON config lookup.",
        scope="global",
    )
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

        curr_intermediate_cache2 = intermediate_cache2[
            : tokens_in_chunk * topk_ids.size(1)
        ]
        curr_intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]
        curr_out_hidden_states = out_hidden_states[begin_chunk_idx:end_chunk_idx]

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

        if use_musa_mxfp4_grouped_gemv:
            _musa_mxfp4_grouped_gemv_moe(
                curr_out_hidden_states,
                curr_hidden_states,
                w1,
                w2,
                curr_topk_weights,
                curr_topk_ids,
                activation,
                expert_map,
                w1_scale,
                w2_scale,
                apply_router_weight_on_input,
            )
            continue

        if use_musa_mxfp4_naive_grouped_moe:
            _musa_mxfp4_naive_grouped_moe(
                curr_out_hidden_states,
                curr_hidden_states,
                w1,
                w2,
                curr_topk_weights,
                curr_topk_ids,
                expert_map,
                w1_scale,
                w2_scale,
                apply_router_weight_on_input,
            )
            continue

        musa_ops.musa_fused_gemv_moe(
            curr_hidden_states,
            w1,
            curr_intermediate_cache2,
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
            curr_intermediate_cache2,
            w2,
            curr_intermediate_cache3,
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
            curr_intermediate_cache3.view(*curr_intermediate_cache3.size()),
            curr_out_hidden_states,
        )

    return out_hidden_states


import vllm.model_executor.layers.fused_moe.fused_moe
import vllm.model_executor.layers.fused_moe.router.base_router
import vllm.model_executor.layers.fused_moe.runner.moe_runner
from vllm_musa.deepseek_v4_jit.topk import (
    record_moe_apply_trace,
    record_router_select_trace,
)


_ORIGINAL_BASE_ROUTER_SELECT_EXPERTS = (
    vllm.model_executor.layers.fused_moe.router.base_router.BaseRouter.select_experts
)


def _musa_router_select_experts_trace(
    self,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    *,
    input_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if os.environ.get("VLLM_MUSA_DEEPSEEK_V4_ROUTER_TOPK_TRACE", "0").lower() in {
        "",
        "0",
        "false",
        "off",
        "no",
    }:
        return _ORIGINAL_BASE_ROUTER_SELECT_EXPERTS(
            self,
            hidden_states,
            router_logits,
            input_ids=input_ids,
        )

    indices_type = self._get_indices_type()
    topk_weights, topk_ids = _ORIGINAL_BASE_ROUTER_SELECT_EXPERTS(
        self,
        hidden_states,
        router_logits,
        input_ids=input_ids,
    )
    record_router_select_trace(
        router_name=self.__class__.__name__,
        hidden_states=hidden_states,
        router_logits=router_logits,
        input_ids=input_ids,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        indices_type=indices_type,
        enable_eplb=getattr(self, "enable_eplb", False),
        stage="post_select",
    )
    return topk_weights, topk_ids


_ORIGINAL_MOE_RUNNER_APPLY_QUANT_METHOD = (
    vllm.model_executor.layers.fused_moe.runner.moe_runner.MoERunner._apply_quant_method
)


def _musa_moe_runner_apply_quant_method_trace(
    self,
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    if os.environ.get("VLLM_MUSA_DEEPSEEK_V4_ROUTER_TOPK_TRACE", "0").lower() not in {
        "",
        "0",
        "false",
        "off",
        "no",
    }:
        record_moe_apply_trace(
            layer_name=getattr(layer, "layer_name", "unknown"),
            quant_method_name=self.quant_method.__class__.__name__,
            router_name=self.router.__class__.__name__,
            is_monolithic=bool(self.quant_method.is_monolithic),
            hidden_states=hidden_states,
            router_logits=router_logits,
            input_ids=input_ids,
            shared_experts_input=shared_experts_input,
            stage="pre_apply",
        )
    return _ORIGINAL_MOE_RUNNER_APPLY_QUANT_METHOD(
        self,
        layer,
        hidden_states,
        router_logits,
        shared_experts_input,
        input_ids=input_ids,
    )

vllm.model_executor.layers.fused_moe.fused_moe.fused_experts_impl = fused_experts_impl
vllm.model_executor.layers.fused_moe.fused_moe.TritonExperts._supports_quant_scheme = (
    _supports_quant_scheme
)
vllm.model_executor.layers.fused_moe.router.base_router.BaseRouter.select_experts = (
    _musa_router_select_experts_trace
)
vllm.model_executor.layers.fused_moe.runner.moe_runner.MoERunner._apply_quant_method = (
    _musa_moe_runner_apply_quant_method_trace
)
