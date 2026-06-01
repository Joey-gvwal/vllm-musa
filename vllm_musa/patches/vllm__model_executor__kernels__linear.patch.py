# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

_OLD_MUSA_FP8_FALLBACK = """    platform_kernels = possible_kernels.get(current_platform._enum)
    if platform_kernels is None and current_platform.is_musa():
        from vllm_musa.model_executor.kernels.linear.scaled_mm.deep_gemm import (
            MUSADeepGemmFp8BlockScaledMMKernel,
        )

        if possible_kernels is _POSSIBLE_FP8_BLOCK_KERNELS:
            platform_kernels = [MUSADeepGemmFp8BlockScaledMMKernel]

    if platform_kernels is None:
        raise ValueError(
            "Failed to find a kernel that can implement the "
            "ScaledMM linear layer. No kernels are registered for "
            f"platform {current_platform._enum.name}."
        )

    for kernel in platform_kernels:
"""

_MUSA_FP8_FALLBACK = """    platform_kernels = possible_kernels.get(current_platform._enum)
    if platform_kernels is None and current_platform.is_musa():
        if possible_kernels is _POSSIBLE_FP8_BLOCK_KERNELS:
            from vllm_musa.model_executor.kernels.linear.scaled_mm.deep_gemm import (
                MUSADeepGemmFp8BlockScaledMMKernel,
            )

            platform_kernels = [
                MUSADeepGemmFp8BlockScaledMMKernel,
                TritonFp8BlockScaledMMKernel,
            ]
        elif possible_kernels is _POSSIBLE_FP8_KERNELS:
            from vllm_musa.model_executor.kernels.linear.scaled_mm.torch_scaled_mm import (
                MUSAChannelWiseTorchFP8ScaledMMLinearKernel,
                MUSAPerTensorTorchFP8ScaledMMLinearKernel,
            )

            platform_kernels = [
                MUSAPerTensorTorchFP8ScaledMMLinearKernel,
                MUSAChannelWiseTorchFP8ScaledMMLinearKernel,
            ]

    if platform_kernels is None:
        raise ValueError(
            "Failed to find a kernel that can implement the "
            "ScaledMM linear layer. No kernels are registered for "
            f"platform {current_platform._enum.name}."
        )

    for kernel in platform_kernels:
"""

_OLD_MUSA_FP8_PLATFORM_KERNELS = """    platform_kernels = possible_kernels.get(current_platform._enum)
    if platform_kernels is None and current_platform.is_musa():
        if possible_kernels is _POSSIBLE_FP8_BLOCK_KERNELS:
            from vllm_musa.model_executor.kernels.linear.scaled_mm.deep_gemm import (
                MUSADeepGemmFp8BlockScaledMMKernel,
            )

            platform_kernels = [MUSADeepGemmFp8BlockScaledMMKernel]
        elif possible_kernels is _POSSIBLE_FP8_KERNELS:
            from vllm_musa.model_executor.kernels.linear.scaled_mm.torch_scaled_mm import (
                MUSAChannelWiseTorchFP8ScaledMMLinearKernel,
                MUSAPerTensorTorchFP8ScaledMMLinearKernel,
            )

            platform_kernels = [
                MUSAPerTensorTorchFP8ScaledMMLinearKernel,
                MUSAChannelWiseTorchFP8ScaledMMLinearKernel,
            ]

    if platform_kernels is None:
        raise ValueError(
            "Failed to find a kernel that can implement the "
            "ScaledMM linear layer. No kernels are registered for "
            f"platform {current_platform._enum.name}."
        )
"""

_MUSA_FP8_PLATFORM_KERNELS = """    platform_kernels = possible_kernels.get(current_platform._enum)
    if platform_kernels is None and current_platform.is_musa():
        if possible_kernels is _POSSIBLE_FP8_BLOCK_KERNELS:
            from vllm_musa.model_executor.kernels.linear.scaled_mm.deep_gemm import (
                MUSADeepGemmFp8BlockScaledMMKernel,
            )

            platform_kernels = [
                MUSADeepGemmFp8BlockScaledMMKernel,
                TritonFp8BlockScaledMMKernel,
            ]
        elif possible_kernels is _POSSIBLE_FP8_KERNELS:
            from vllm_musa.model_executor.kernels.linear.scaled_mm.torch_scaled_mm import (
                MUSAChannelWiseTorchFP8ScaledMMLinearKernel,
                MUSAPerTensorTorchFP8ScaledMMLinearKernel,
            )

            platform_kernels = [
                MUSAPerTensorTorchFP8ScaledMMLinearKernel,
                MUSAChannelWiseTorchFP8ScaledMMLinearKernel,
            ]

    if platform_kernels is None:
        raise ValueError(
            "Failed to find a kernel that can implement the "
            "ScaledMM linear layer. No kernels are registered for "
            f"platform {current_platform._enum.name}."
        )
"""

PATCHES = [
    (
        """    platform_kernels = possible_kernels[current_platform._enum]
""",
        _MUSA_FP8_PLATFORM_KERNELS,
    ),
    (
        _OLD_MUSA_FP8_PLATFORM_KERNELS,
        _MUSA_FP8_PLATFORM_KERNELS,
    ),
    (
        """    for kernel in possible_kernels[current_platform._enum]:
""",
        _MUSA_FP8_FALLBACK,
    ),
    (
        _OLD_MUSA_FP8_FALLBACK,
        _MUSA_FP8_FALLBACK,
    ),
]
