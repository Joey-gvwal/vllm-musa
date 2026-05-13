# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch MXFP4 MoE selection with an opt-in MUSA diagnostic fallback.
"""

PATCHES = [
    (
        """        self.weight_dtype = "mxfp4"
        self.mxfp4_backend, self.experts_cls = select_mxfp4_moe_backend(moe)
""",
        """        self.weight_dtype = "mxfp4"
        self.use_musa_mxfp4_fallback = self._musa_mxfp4_fallback_enabled()
        if self.use_musa_mxfp4_fallback:
            self.mxfp4_backend = Mxfp4MoeBackend.NONE
            self.experts_cls = None
            logger.warning_once(
                "Using opt-in MUSA MXFP4 MoE fallback. This dequantizes MXFP4 "
                "weights and runs the existing MUSA fused-MoE path; it is for "
                "DeepSeek-V4 diagnostic coverage, not a native MXFP4 production "
                "backend."
            )
        else:
            self.mxfp4_backend, self.experts_cls = select_mxfp4_moe_backend(moe)
""",
    ),
    (
        """    @property
    def skip_forward_padding(self) -> bool:
        # SM100_FI_MXFP4_MXFP8_TRTLLM supports padding with mxfp8 quant
        # so can skip the padding in the forward before applying the moe method
        return self.mxfp4_backend == Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8
""",
        """    @staticmethod
    def _musa_mxfp4_fallback_enabled() -> bool:
        import os
        from vllm.platforms import current_platform

        return (
            current_platform.is_musa()
            and os.getenv("VLLM_MUSA_ENABLE_MXFP4_MOE_FALLBACK", "0") == "1"
        )

    def _use_musa_mxfp4_fallback(self) -> bool:
        return getattr(self, "use_musa_mxfp4_fallback", False)

    @property
    def is_monolithic(self) -> bool:
        if self._use_musa_mxfp4_fallback():
            return False
        return super().is_monolithic

    @property
    def skip_forward_padding(self) -> bool:
        # SM100_FI_MXFP4_MXFP8_TRTLLM supports padding with mxfp8 quant
        # so can skip the padding in the forward before applying the moe method
        return self.mxfp4_backend == Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8
""",
    ),
    (
        """        if self.mxfp4_backend in TRITON_BACKENDS:
            assert self.w13_precision_config is not None
            assert self.w2_precision_config is not None
            w1_scale = self.w13_precision_config
            w2_scale = self.w2_precision_config

        return make_mxfp4_moe_quant_config(
            mxfp4_backend=self.mxfp4_backend,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            swiglu_limit=swiglu_limit,
        )
""",
        """        if self.mxfp4_backend in TRITON_BACKENDS:
            assert self.w13_precision_config is not None
            assert self.w2_precision_config is not None
            w1_scale = self.w13_precision_config
            w2_scale = self.w2_precision_config

        if self._use_musa_mxfp4_fallback():
            from vllm.model_executor.layers.fused_moe.config import (
                mxfp4_w4a16_moe_quant_config,
            )

            return mxfp4_w4a16_moe_quant_config(
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                w1_bias=w1_bias,
                w2_bias=w2_bias,
                gemm1_clamp_limit=swiglu_limit,
            )

        return make_mxfp4_moe_quant_config(
            mxfp4_backend=self.mxfp4_backend,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            swiglu_limit=swiglu_limit,
        )
""",
    ),
    (
        """    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert not self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
""",
        """    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        if self._use_musa_mxfp4_fallback():
            from vllm_musa.model_executor.layers.fused_moe.fused_moe import (
                _musa_torch_fused_moe_fallback,
            )

            quant_config = self.get_fused_moe_quant_config(layer)
            assert quant_config is not None
            return _musa_torch_fused_moe_fallback(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=layer.activation,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                expert_map=layer.expert_map,
                w1_scale=quant_config.w1_scale,
                w2_scale=quant_config.w2_scale,
                w1_bias=quant_config.w1_bias,
                w2_bias=quant_config.w2_bias,
                ocp_mx_scheme=quant_config.ocp_mx_scheme,
                swiglu_limit=quant_config.gemm1_clamp_limit,
                swiglu_alpha=quant_config.gemm1_alpha,
                swiglu_beta=quant_config.gemm1_beta,
            )

        assert not self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
""",
    ),
    (
        """        if self._use_musa_mxfp4_fallback():
            from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

            quant_config = self.get_fused_moe_quant_config(layer)
            assert quant_config is not None
            return fused_experts(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=False,
                activation=layer.activation,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                global_num_experts=layer.global_num_experts,
                expert_map=layer.expert_map,
                quant_config=quant_config,
            )
""",
        """        if self._use_musa_mxfp4_fallback():
            from vllm_musa.model_executor.layers.fused_moe.fused_moe import (
                _musa_torch_fused_moe_fallback,
            )

            quant_config = self.get_fused_moe_quant_config(layer)
            assert quant_config is not None
            return _musa_torch_fused_moe_fallback(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=layer.activation,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                expert_map=layer.expert_map,
                w1_scale=quant_config.w1_scale,
                w2_scale=quant_config.w2_scale,
                w1_bias=quant_config.w1_bias,
                w2_bias=quant_config.w2_bias,
                ocp_mx_scheme=quant_config.ocp_mx_scheme,
                swiglu_limit=quant_config.gemm1_clamp_limit,
                swiglu_alpha=quant_config.gemm1_alpha,
                swiglu_beta=quant_config.gemm1_beta,
            )
""",
    ),
    (
        """                ocp_mx_scheme=quant_config.ocp_mx_scheme,
                swiglu_limit=quant_config.gemm1_clamp_limit,
            )
""",
        """                ocp_mx_scheme=quant_config.ocp_mx_scheme,
                swiglu_limit=quant_config.gemm1_clamp_limit,
                swiglu_alpha=quant_config.gemm1_alpha,
                swiglu_beta=quant_config.gemm1_beta,
            )
""",
    ),
]
