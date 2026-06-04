# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Patch vLLM v0.22 DeepSeek-V4 NVIDIA model gates for MUSA."""

PATCHES = [
    (
        """import typing
from collections.abc import Callable, Iterable
""",
        """import os
import typing
from collections.abc import Callable, Iterable
""",
    ),
    (
        """from vllm.utils.torch_utils import direct_register_custom_op


class DeepseekV4MLP(nn.Module):
""",
        """from vllm.utils.torch_utils import direct_register_custom_op


def _musa_deepseek_v4_disable_aux_overlap() -> bool:
    return (
        current_platform.is_musa()
        and os.getenv("VLLM_MUSA_DEEPSEEK_V4_DISABLE_AUX_OVERLAP", "0") == "1"
    )


class DeepseekV4MLP(nn.Module):
""",
    ),
    (
        """        aux_stream_list = (
            None
            if current_platform.is_rocm() or current_platform.is_xpu()
            else [torch.cuda.Stream() for _ in range(3)]
        )
""",
        """        aux_stream_list = (
            None
            if (
                current_platform.is_rocm()
                or current_platform.is_xpu()
                or _musa_deepseek_v4_disable_aux_overlap()
            )
            else [torch.cuda.Stream() for _ in range(3)]
        )
""",
    ),
    (
        """        if layer is not None and current_platform.is_cuda():
            hidden_states = layer.hc_post(hidden_states, residual, post_mix, res_mix)
""",
        """        if layer is not None and (
            current_platform.is_cuda() or current_platform.is_musa()
        ):
            hidden_states = layer.hc_post(hidden_states, residual, post_mix, res_mix)
""",
    ),
]
