# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch clamped SwiGLU activation to avoid CUDA-only vLLM C ops on MUSA.
"""

PATCHES = [
    (
        """        if current_platform.is_cuda_alike() or current_platform.is_xpu():
            self.op = torch.ops._C.silu_and_mul
        elif current_platform.is_cpu():
            self._forward_method = self.forward_native
""",
        """        if current_platform.is_musa():
            self._forward_method = self.forward_oot
        elif current_platform.is_cuda_alike() or current_platform.is_xpu():
            self.op = torch.ops._C.silu_and_mul
        elif current_platform.is_cpu():
            self._forward_method = self.forward_native
""",
    ),
    (
        """        if current_platform.is_cuda_alike() or current_platform.is_xpu():
            self.op = torch.ops._C.silu_and_mul_with_clamp
        elif current_platform.is_cpu():
            self._forward_method = self.forward_native
""",
        """        if current_platform.is_musa():
            self._forward_method = self.forward_native
        elif current_platform.is_cuda_alike() or current_platform.is_xpu():
            self.op = torch.ops._C.silu_and_mul_with_clamp
        elif current_platform.is_cpu():
            self._forward_method = self.forward_native
""",
    ),
]

RELOAD_AFTER_PATCH = ["__TARGET_MODULE__"]
