# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch mHC blocks with MUSA native/JIT helpers.
"""

_MUSA_MHC_PRE_DISPATCH = """    if current_platform.is_musa():
        from vllm_musa.deepseek_v4_mhc import mhc_pre_musa

        return mhc_pre_musa(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
"""

_MUSA_MHC_PRE_TORCH_GATE = """    if (
        current_platform.is_musa()
        and os.getenv("VLLM_MUSA_ENABLE_TORCH_MHC_PRENORM_FALLBACK", "0") == "1"
    ):
        from vllm_musa.deepseek_v4_mhc import mhc_pre_torch_fallback

        return mhc_pre_torch_fallback(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
"""


def normalize_source(source: str) -> str:
    if _MUSA_MHC_PRE_TORCH_GATE not in source:
        return source
    if _MUSA_MHC_PRE_DISPATCH in source:
        return source.replace(_MUSA_MHC_PRE_TORCH_GATE + "\n", "")
    return source.replace(_MUSA_MHC_PRE_TORCH_GATE, _MUSA_MHC_PRE_DISPATCH)


PATCHES = [
    (
        """import math
from functools import cache
""",
        """import math
import os
from functools import cache
""",
    ),
    (
        """# tilelang is only available on CUDA platforms
if TYPE_CHECKING or current_platform.is_cuda_alike():
    if not has_tilelang():
        raise ImportError(
            "tilelang is required for mhc but is not installed. Install it with "
            "`pip install tilelang`."
        )
    import tilelang
    import tilelang.language as T
else:
    tilelang = None  # type: ignore[assignment]
    T = None  # type: ignore[assignment]
""",
        """# tilelang is only available on CUDA platforms
if TYPE_CHECKING or (current_platform.is_cuda_alike() and not current_platform.is_musa()):
    if not has_tilelang():
        raise ImportError(
            "tilelang is required for mhc but is not installed. Install it with "
            "`pip install tilelang`."
        )
    import tilelang
    import tilelang.language as T
else:
    class _MUSATilelangStub:
        class PassConfigKey:
            TL_DISABLE_WARP_SPECIALIZED = "TL_DISABLE_WARP_SPECIALIZED"
            TL_DISABLE_TMA_LOWER = "TL_DISABLE_TMA_LOWER"
            TL_PTXAS_REGISTER_USAGE_LEVEL = "TL_PTXAS_REGISTER_USAGE_LEVEL"

        @staticmethod
        def jit(*args, **kwargs):
            def decorator(func):
                return func

            return decorator

    tilelang = _MUSATilelangStub()  # type: ignore[assignment]
    T = None  # type: ignore[assignment]
""",
    ),
    (
        """    class _MUSATilelangStub:
        class PassConfigKey:
""",
        """    class _MUSATilelangStub:
        JITKernel = object

        class PassConfigKey:
""",
    ),
    (
        """    # Validate shapes
    assert residual.dtype == torch.bfloat16
""",
        _MUSA_MHC_PRE_DISPATCH + """
    # Validate shapes
    assert residual.dtype == torch.bfloat16
""",
    ),
    (
        _MUSA_MHC_PRE_TORCH_GATE,
        _MUSA_MHC_PRE_DISPATCH,
    ),
    (
        """        return mhc_kernels.mhc_pre_torch(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
""",
        """        if residual.device.type == "musa":
            from vllm_musa.deepseek_v4_mhc import mhc_pre_musa_with_norm

            return mhc_pre_musa_with_norm(
                residual,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                norm_weight,
                norm_eps,
            )
        return mhc_kernels.mhc_pre_torch(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
""",
    ),
    (
        """def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(residual)
""",
        """def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    if (
        current_platform.is_musa()
        and (
            os.getenv("VLLM_MUSA_ENABLE_DEEPSEEK_V4_MHC_MUSA_IMPL", "1") == "1"
            or os.getenv(
                "VLLM_MUSA_ENABLE_TORCH_MHC_PRENORM_FALLBACK",
                "0",
            )
            == "1"
        )
    ):
        from vllm_musa.deepseek_v4_mhc import mhc_post_musa

        return mhc_post_musa(x, residual, post_layer_mix, comb_res_mix)

    out = torch.empty_like(residual)
""",
    ),
    (
        """        return mhc_kernels.mhc_post_torch(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
        )
""",
        """        if x.device.type == "musa":
            from vllm_musa.deepseek_v4_mhc import mhc_post_musa

            return mhc_post_musa(x, residual, post_layer_mix, comb_res_mix)
        return mhc_kernels.mhc_post_torch(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
        )
""",
    ),
    (
        """    def forward_native(self, *args, **kwargs):
        raise NotImplementedError("Native implementation of hc_head is not available")
""",
        """    def forward_native(self, *args, **kwargs):
        hidden_states = args[0] if args else kwargs.get("hidden_states")
        if hidden_states is not None and hidden_states.device.type == "musa":
            from vllm_musa.deepseek_v4_mhc import hc_head_musa

            return hc_head_musa(*args, **kwargs)
        raise NotImplementedError("Native implementation of hc_head is not available")
""",
    ),
    (
        """    def forward_native(self, *args, **kwargs):
        raise NotImplementedError(
            "Native implementation of mhc_fused_post_pre is not available"
        )
""",
        """    def forward_native(self, *args, **kwargs):
        x = args[0] if args else kwargs.get("x")
        if x is not None and x.device.type == "musa":
            from vllm_musa.deepseek_v4_mhc import mhc_fused_post_pre_musa

            return mhc_fused_post_pre_musa(*args, **kwargs)
        raise NotImplementedError(
            "Native implementation of mhc_fused_post_pre is not available"
        )
""",
    ),
]
